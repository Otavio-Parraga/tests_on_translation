"""Single-pass, multi-layer, multi-position activation extraction.

WHY THIS EXISTS NEXT TO extract_activations_with_resume (dataset.py): that
function reads ONE layer at ONE position per model load and per pass over the
sentences. Adding a second read position on top of the existing 7-layer sweep
would mean 14 full passes over 500k sentences per model — roughly 70 GPU-hours
for the two LLaDA models. But the hooked layer output is a [B, T, D] tensor from
which EVERY read position is a cheap reduction, and every layer up to the deepest
one requested is computed anyway on the way there. So one pass, with one hook per
requested layer, yields the entire layers x positions grid (~7.5 GPU-hours).

OUTPUT IS FORMAT-COMPATIBLE with the single-layer extractor: one cache per
(layer, position) at activation_path(data_dir, model_cfg), carrying the same
{"activations", "sentences", "n_sentences", "chat_template"} payload. So
fit_procrustes.py, verify_activations.py and every cache reader work unchanged,
and a cache already completed by EITHER extractor is recognised and skipped.

READ POSITIONS: "mean" (mask-weighted mean over real tokens) and -1 (last real
token). These are the two whose cache filenames are already distinct under
paths.py's slug convention ("_mean" vs the bare slug). Any OTHER int is refused:
model_slug() gives every int the bare slug, so e.g. -2 and -1 would write the
SAME filename and the second would be silently mistaken for the first by every
reader. Registering a suffix for arbitrary ints is a paths.py change (and its two
regex consumers), not something to do implicitly from here.

RESUME: chunk files per (layer, position) under one staging dir per model run,
flushed together every `flush_every` batches, so an interrupted run continues
from the last complete flush. A disk guard runs at every flush and stops cleanly
(resumable) rather than writing a file it cannot fit — on a shared disk, other
users' activity can eat the headroom mid-run.
"""
import gc
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from tqdm import tqdm

from ..utils.hf import get_layer, load_model_and_tokenizer
from ..utils.paths import activation_path
from .dataset import _StopForward, _apply_chat_template

Position = Union[str, int]


def parse_position(s: str) -> Position:
    """CLI helper: 'mean' -> 'mean', '-1' -> -1."""
    return "mean" if s == "mean" else int(s)


def validate_positions(positions: Sequence[Position]) -> None:
    for p in positions:
        if str(p) == "mean" or p == -1:
            continue
        raise ValueError(
            f"Unsupported read position {p!r}. This extractor supports 'mean' and -1 only: "
            "any other int shares the bare cache filename with -1 under paths.model_slug() and "
            "would silently be mistaken for it by every reader. Register a suffix in paths.py "
            "first if another position is needed."
        )


def position_key(pos: Position) -> str:
    """Stable staging-dir name for a read position: 'mean' | 'tok-1'."""
    return "mean" if str(pos) == "mean" else f"tok{pos}"


def _readout(hidden: torch.Tensor, mask: torch.Tensor, pos: Position) -> torch.Tensor:
    """[B, T, D] -> [B, D] at one read position. The expressions match
    dataset.extract_activations_with_resume EXACTLY so a cache written here is
    numerically the same as one written there (same masked_fill, same argmax)."""
    if str(pos) == "mean":
        m = mask.unsqueeze(-1).to(hidden.dtype)
        summed = hidden.masked_fill(m == 0, 0.0).sum(dim=1)
        counts = m.sum(dim=1).clamp(min=1.0)
        return summed / counts
    # -1: last REAL token, robust to either padding side.
    idx = (mask * torch.arange(mask.shape[1], device=mask.device)).argmax(dim=1)
    rows = torch.arange(hidden.shape[0], device=hidden.device)
    return hidden[rows, idx, :]


def _chunk_rows(key_dir: Path) -> Tuple[int, List[Path]]:
    files = sorted(key_dir.glob("chunk_*.pt")) if key_dir.is_dir() else []
    return sum(torch.load(f, weights_only=True, mmap=True).shape[0] for f in files), files


def _truncate_to(key_dir: Path, rows: int) -> None:
    """Drop trailing chunks until this key holds exactly `rows` rows (chunks are
    flushed together across keys, so a crash mid-flush leaves at most one extra)."""
    total, files = _chunk_rows(key_dir)
    while total > rows and files:
        f = files.pop()
        total -= torch.load(f, weights_only=True, mmap=True).shape[0]
        f.unlink()
    if total != rows:
        raise RuntimeError(f"{key_dir}: could not truncate to {rows} rows (at {total}); "
                           "chunk boundaries disagree across keys — remove the staging dir and restart.")


def extract_multi_with_resume(
    model_cfg: Dict,
    sentences: List[str],
    device: str,
    data_dir: Path,
    layers: Sequence[int],
    positions: Sequence[Position],
    batch_size: int,
    hf_cache_dir: Optional[str] = None,
    flush_every: int = 64,
    min_free_gb: float = 10.0,
    label: str = "",
) -> Dict[Tuple[int, Position], Path]:
    """Extract every (layer, position) in one pass. Returns {(layer, pos): cache_path}
    for everything that is now complete on disk (including caches skipped as
    already complete)."""
    validate_positions(positions)
    data_dir = Path(data_dir)
    prefix = f"[{label}] " if label else ""
    chat_template = bool(model_cfg.get("chat_template", False))
    n = len(sentences)

    # --- work list: skip caches already complete (from EITHER extractor) -------
    done: Dict[Tuple[int, Position], Path] = {}
    todo: Dict[Tuple[int, Position], Path] = {}
    for L in layers:
        for p in positions:
            mcfg = dict(model_cfg, layer=L, token_position=p)
            path = activation_path(data_dir, mcfg)
            if path.exists():
                ck = torch.load(path, weights_only=False, mmap=True)
                if ck.get("n_sentences") == n:
                    if bool(ck.get("chat_template", False)) != chat_template:
                        raise RuntimeError(
                            f"{prefix}chat_template mismatch: {path} was built with "
                            f"chat_template={bool(ck.get('chat_template', False))}, this run asks for "
                            f"{chat_template}. Refusing — use a separate training.data_dir per variant."
                        )
                    print(f"{prefix}Already complete: {path.name}")
                    done[(L, p)] = path
                    continue
            todo[(L, p)] = path
    if not todo:
        print(f"{prefix}Nothing to extract.")
        return done

    short = str(model_cfg["name"]).split("/")[-1]
    stage = data_dir / "activations" / f".multi_{short}_stage"
    stage.mkdir(parents=True, exist_ok=True)
    keys = {lp: f"l{lp[0]}_{position_key(lp[1])}" for lp in todo}

    # Same guard as the single-layer extractor: never splice templated and raw
    # rows (or two different sentence sets) into one staging area.
    meta_path = stage / "meta.json"
    meta = {"chat_template": chat_template, "n_sentences": n}
    if meta_path.exists():
        old = json.loads(meta_path.read_text())
        if {k: old.get(k) for k in meta} != meta:
            raise RuntimeError(f"{prefix}staging dir {stage} was started with {old}, this run is {meta}. "
                               "Refusing to mix; remove the staging dir to start over.")
    else:
        meta_path.write_text(json.dumps(meta))

    # --- resume point: all keys must agree; truncate any that ran ahead -------
    rows = {k: _chunk_rows(stage / k)[0] for k in keys.values()}
    start = min(rows.values())
    for k, r in rows.items():
        if r > start:
            print(f"{prefix}{k} has {r} rows but another key has {start}; truncating to resync.")
            _truncate_to(stage / k, start)
    if start:
        print(f"{prefix}Resuming from sentence {start}/{n}")

    if start < n:
        model, tok = load_model_and_tokenizer(model_cfg["name"], device)
        model.eval()
        texts = _apply_chat_template(tok, sentences, 512, prefix) if chat_template else sentences

        module_name = model_cfg.get("module", "residual")
        layer_mods = {L: get_layer(model, L, module_name) for L in sorted({L for L, _ in todo})}
        deepest = max(layer_mods)
        captured: Dict[int, torch.Tensor] = {}

        def mk_hook(L):
            def hook(module, inp, out):
                captured[L] = out[0] if isinstance(out, tuple) else out
                if L == deepest:
                    # Everything above the deepest requested layer is waste.
                    raise _StopForward
            return hook

        handles = [m.register_forward_hook(mk_hook(L)) for L, m in layer_mods.items()]
        for k in keys.values():
            (stage / k).mkdir(exist_ok=True)
        buffers: Dict[str, List[torch.Tensor]] = {k: [] for k in keys.values()}
        chunk_idx = len(_chunk_rows(stage / next(iter(keys.values())))[1])

        def flush():
            nonlocal chunk_idx
            if not any(buffers.values()):
                return
            for k, buf in buffers.items():
                torch.save(torch.cat(buf), stage / k / f"chunk_{chunk_idx:06d}.pt")
                buf.clear()
            chunk_idx += 1

        remaining = texts[start:]
        n_batches = (len(remaining) + batch_size - 1) // batch_size
        print(f"{prefix}{len(todo)} caches in one pass: layers {sorted(layer_mods)} x positions "
              f"{[str(p) for p in positions]}  |  {n_batches} batches of {batch_size}  |  "
              f"chat_template={chat_template}")
        try:
            bar = tqdm(range(0, len(remaining), batch_size), total=n_batches,
                       desc=f"{prefix}Extracting", unit="batch", mininterval=30)
            for step, i in enumerate(bar):
                batch = remaining[i:i + batch_size]
                encoded = tok(batch, padding=True, truncation=True, max_length=512,
                              add_special_tokens=not chat_template, return_tensors="pt")
                encoded = {k: v.to(device) for k, v in encoded.items()}
                mask = encoded["attention_mask"]
                with torch.no_grad():
                    try:
                        model(**encoded)
                    except _StopForward:
                        pass
                for (L, p), k in keys.items():
                    act = _readout(captured[L], mask, p).detach().cpu()
                    if not torch.isfinite(act).all():
                        n_bad = int((~torch.isfinite(act)).any(dim=1).sum())
                        raise RuntimeError(
                            f"{prefix}Non-finite activations ({n_bad}/{act.shape[0]} rows) at layer {L} "
                            f"pos {p!r}, sentences {start + i}..{start + i + len(batch) - 1}. "
                            "Refusing to write a corrupt cache.")
                    buffers[k].append(act)
                captured.clear()

                if (step + 1) % flush_every == 0:
                    free_gb = shutil.disk_usage(data_dir).free / 1e9
                    if free_gb < min_free_gb:
                        flush()
                        raise RuntimeError(
                            f"{prefix}only {free_gb:.1f}GB free under {data_dir} (< {min_free_gb}GB). "
                            f"Progress saved at sentence {start + i + len(batch)}; re-run to resume.")
                    flush()
            flush()
        finally:
            for h in handles:
                h.remove()
            captured.clear()
            del model, layer_mods
            gc.collect()
            torch.cuda.empty_cache()

    # --- merge each key into its final cache, then clear its staging ----------
    for (L, p), k in keys.items():
        _, files = _chunk_rows(stage / k)
        merged = torch.cat([torch.load(f, weights_only=True) for f in files], dim=0)
        if merged.shape[0] != n:
            raise RuntimeError(f"{prefix}{k}: merged {merged.shape[0]} rows, expected {n}.")
        out = todo[(L, p)]
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"activations": merged, "sentences": sentences, "n_sentences": n,
                    "chat_template": chat_template}, out)
        shutil.rmtree(stage / k)
        print(f"{prefix}Saved {tuple(merged.shape)} -> {out}")
        done[(L, p)] = out

    if not any(d.is_dir() for d in stage.iterdir()):
        shutil.rmtree(stage)
    return done
