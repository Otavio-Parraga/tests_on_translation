import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import torch
from tqdm import tqdm

from ..utils.hf import get_layer, load_model_and_tokenizer
from ..utils.paths import activation_path, data_dir_of, legacy_activation_path, sentences_path


def _model_hidden_size(model_cfg: Dict, hf_cache_dir: Optional[str]) -> Optional[int]:
    """Hidden width of a model, read from its config without loading weights."""
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_cfg["name"], cache_dir=hf_cache_dir)
        return getattr(cfg, "hidden_size", None) or getattr(cfg, "d_model", None)
    except Exception:
        return None


def _migrate_legacy_activations(
    new_path: Path, legacy_path: Path, n_sentences: int, expected_dim: Optional[int]
) -> None:
    """Adopt a pre-refactor flat activation file (source_activations.pt /
    target_activations.pt) into the per-model cache location — but only when it
    genuinely belongs to this model. The flat files carry no model identity, so
    we require both the sentence count AND the feature width to match; otherwise
    we leave it alone and let extraction run (e.g. an old gemma target file must
    not be mistaken for a Llama-3B target on the same sentences)."""
    if new_path.exists() or not legacy_path.exists():
        return
    try:
        ckpt = torch.load(legacy_path, weights_only=False, mmap=True)
    except Exception:
        try:
            ckpt = torch.load(legacy_path, weights_only=False)
        except Exception:
            return
    if ckpt.get("n_sentences") != n_sentences:
        return
    acts = ckpt.get("activations")
    if expected_dim is not None and acts is not None and acts.shape[1] != expected_dim:
        return
    new_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_path.rename(new_path)
    print(f"  Migrated legacy {legacy_path.name} -> {new_path.name}")


class _StopForward(Exception):
    """Raised inside the capture hook to abort the rest of the model's forward
    pass once the target layer's activations are in hand. Everything above the
    hooked layer — the remaining decoder blocks and especially the vocab-sized
    lm_head — is pure waste for activation extraction and is what OOMs on large
    targets (e.g. gemma's 256k-row lm_head needs ~8 GiB just for the logits)."""


def extract_activations_with_resume(
    model_cfg: Dict,
    sentences: List[str],
    device: str,
    save_path: Path,
    hf_cache_dir: Optional[str],
    batch_size: int,
    label: str = "",
    tqdm_position: Optional[int] = None,
) -> torch.Tensor:
    """Extract activations incrementally, resuming from save_path if interrupted.

    Saves one .pt file per batch to a temporary sibling directory. On completion,
    merges them into save_path and removes the temp directory.
    tqdm_position: pin the bar to this row (for multiprocessing — caller must also
    call tqdm.set_lock() with a shared RLock before spawning workers).
    """
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    save_path = Path(save_path)
    batch_dir = save_path.parent / f".{save_path.stem}_batches"
    prefix = f"[{label}] " if label else ""

    # Already fully complete?
    if save_path.exists():
        ckpt = torch.load(save_path, weights_only=False)
        if ckpt.get("n_sentences") == len(sentences):
            print(f"{prefix}Already complete ({len(sentences)} sentences), skipping extraction.")
            return ckpt["activations"]

    batch_dir.mkdir(parents=True, exist_ok=True)

    # Determine resume position from previously saved batch files
    existing_batches = sorted(batch_dir.glob("batch_*.pt"))
    n_existing = len(existing_batches)
    start_idx = 0
    if n_existing > 0:
        for bf in existing_batches:
            start_idx += torch.load(bf, weights_only=True).shape[0]
        print(f"{prefix}Resuming from sentence {start_idx}/{len(sentences)} ({n_existing} batches already saved)")

    if start_idx < len(sentences):
        model, tokenizer = load_model_and_tokenizer(model_cfg["name"], device)
        model.eval()

        layer = get_layer(model, model_cfg["layer"])
        # token_position: an int index (e.g. -1 = last token) or the string "mean"
        # to mean-pool over every real (non-pad) token in the sentence. Both modes
        # use the attention mask so they are correct under left- OR right-padding.
        token_position = model_cfg.get("token_position", -1)
        pooling_mean = str(token_position) == "mean"
        captured = {}

        def hook_fn(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output  # [B, T, D]
            mask = captured["attention_mask"]  # [B, T]
            if pooling_mean:
                m = mask.unsqueeze(-1).to(hidden.dtype)  # [B, T, 1]
                summed = (hidden * m).sum(dim=1)
                counts = m.sum(dim=1).clamp(min=1.0)
                captured["act"] = (summed / counts).detach().cpu()
            elif token_position == -1:
                # Last *real* token: the largest position whose mask is 1. Robust to
                # whichever side the tokenizer pads on (plain hidden[:, -1] would grab
                # a pad token under right-padding).
                pos = (mask * torch.arange(mask.shape[1], device=mask.device)).argmax(dim=1)
                rows = torch.arange(hidden.shape[0], device=hidden.device)
                captured["act"] = hidden[rows, pos, :].detach().cpu()
            else:
                captured["act"] = hidden[:, token_position, :].detach().cpu()
            # Got what we need from this layer — skip the rest of the forward
            # (upper decoder blocks + lm_head) to save memory and time.
            raise _StopForward

        handle = layer.register_forward_hook(hook_fn)
        remaining = sentences[start_idx:]
        n_batches = (len(remaining) + batch_size - 1) // batch_size

        tqdm_kwargs = dict(total=n_batches, desc=f"{prefix}Extracting", unit="batch")
        if tqdm_position is not None:
            tqdm_kwargs.update(position=tqdm_position, leave=True, dynamic_ncols=False)

        try:
            for step, i in enumerate(tqdm(range(0, len(remaining), batch_size), **tqdm_kwargs)):
                batch = remaining[i : i + batch_size]
                encoded = tokenizer(
                    batch, padding=True, truncation=True, max_length=512, return_tensors="pt"
                )
                encoded = {k: v.to(device) for k, v in encoded.items()}
                captured["attention_mask"] = encoded["attention_mask"]
                with torch.no_grad():
                    try:
                        model(**encoded)
                    except _StopForward:
                        pass
                torch.save(captured["act"], batch_dir / f"batch_{n_existing + step:06d}.pt")
        finally:
            handle.remove()

        del model
        torch.cuda.empty_cache()

    # Merge all batch files into the final output file
    all_batches = sorted(batch_dir.glob("batch_*.pt"))
    merged = torch.cat([torch.load(b, weights_only=True) for b in all_batches], dim=0)
    torch.save({"activations": merged, "sentences": sentences, "n_sentences": len(sentences)}, save_path)
    shutil.rmtree(batch_dir)
    print(f"{prefix}Saved {merged.shape[0]} activations to {save_path}")

    return merged


def _extraction_worker(worker_args: Dict) -> None:
    """Spawned process worker — runs extract_activations_with_resume on a dedicated GPU.

    Must be at module level to be picklable; the acttrans package is installed,
    so spawned interpreters can import this module directly.
    """
    tqdm_lock = worker_args.get("tqdm_lock")
    if tqdm_lock is not None:
        tqdm.set_lock(tqdm_lock)
    extract_activations_with_resume(
        model_cfg=worker_args["model_cfg"],
        sentences=worker_args["sentences"],
        device=worker_args["device"],
        save_path=Path(worker_args["save_path"]),
        hf_cache_dir=worker_args.get("hf_cache_dir"),
        batch_size=worker_args["batch_size"],
        label=worker_args["label"],
        tqdm_position=worker_args.get("tqdm_position"),
    )


def _load_caa(behaviors, split, data_root, limit, hf_cache_dir):
    sentences = []
    data_root_path = Path(data_root)
    filename = "generate_dataset.json" if split == "generate" else "test_dataset_ab.json"
    for behavior in behaviors or []:
        json_path = data_root_path / "CAA_datasets" / split / behavior / filename
        if not json_path.exists():
            raise FileNotFoundError(f"CAA dataset not found at {json_path}")
        with open(json_path, "r") as f:
            items = json.load(f)
        for item in items:
            question = item.get("question", "").strip()
            pos_answer = item.get("answer_matching_behavior", "").strip()
            neg_answer = item.get("answer_not_matching_behavior", "").strip()
            if question:
                sentences.append(question)
            if pos_answer:
                sentences.append(f"{question}\n\nAnswer: {pos_answer}")
            if neg_answer:
                sentences.append(f"{question}\n\nAnswer: {neg_answer}")
    return sentences


def _load_mwe(behaviors, split, data_root, limit, hf_cache_dir):
    import random
    import pandas as pd
    sentences = []
    mwe_root = Path(data_root) / "evals"
    if not mwe_root.exists():
        raise FileNotFoundError(f"MWE data directory not found at {mwe_root}")
    jsonl_files = sorted(f for f in mwe_root.glob("**/*.jsonl") if "wino" not in f.stem)
    if behaviors:
        jsonl_files = [f for f in jsonl_files if f.stem in behaviors]
    if not jsonl_files:
        raise FileNotFoundError(
            f"No MWE JSONL files found under {mwe_root} for behaviors={behaviors}"
        )
    for jsonl_file in jsonl_files:
        rows = pd.read_json(jsonl_file, lines=True).to_dict(orient="records")
        rng = random.Random(42)
        rng.shuffle(rows)
        n_train = int(len(rows) * 0.8)
        rows = rows[:n_train] if split in ("train", "generate") else rows[n_train:]
        for row in rows:
            question = str(row.get("question", "")).strip()
            pos = str(row.get("answer_matching_behavior", ""))
            neg = str(row.get("answer_not_matching_behavior", ""))
            bare_q = question.removesuffix("Answer:").rstrip()
            if bare_q:
                sentences.append(bare_q)
            if pos.strip():
                sentences.append(f"{question}{pos}")
            if neg.strip():
                sentences.append(f"{question}{neg}")
    return sentences


def _load_tqa(behaviors, split, data_root, limit, hf_cache_dir):
    tqa_path = os.environ.get("TQA_PATH")
    if not tqa_path:
        raise ValueError("TQA_PATH environment variable is not set")
    import pandas as pd
    sentences = []
    df = pd.read_csv(tqa_path)
    for _, row in df.iterrows():
        question = str(row.get("Question", "")).strip()
        pos = str(row.get("Best Answer", "")).strip()
        neg_raw = str(row.get("Incorrect Answers", "")).strip()
        neg = neg_raw.split(";")[0].strip()
        if question:
            sentences.append(question)
        if pos:
            sentences.append(f"{question}\n\nAnswer: {pos}")
        if neg:
            sentences.append(f"{question}\n\nAnswer: {neg}")
    return sentences


def _load_generated(behaviors, split, data_root, limit, hf_cache_dir):
    with open(data_root) as f:
        return json.load(f)


def _load_fineweb(behaviors, split, data_root, limit, hf_cache_dir):
    import random
    import re
    from datasets import load_dataset

    sentences = []
    # FineWeb is terabytes — stream it (never materialized) and pull from the
    # `text` column. The HF datasets cache is pinned to HF_CACHE_DIR, the same
    # cache the models use, so weights and corpus share one location.
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
        cache_dir=hf_cache_dir,
    )

    # Split each document into sentences; keep meaningful fragments only.
    sentence_re = re.compile(r"(?<=[.!?])\s+")
    min_chars = 20

    def doc_sentences(text: str) -> List[str]:
        out = []
        for frag in sentence_re.split(text or ""):
            frag = frag.strip()
            if len(frag) >= min_chars:
                out.append(frag)
        return out

    # Train/test split mirrors MWE: deterministic shuffle (seed 42), first 80%
    # train, last 20% test. We partition at the DOCUMENT level — never the
    # sentence level — so sentences from one doc can't leak across the split.
    want_train = split in ("train", "generate")

    # Stream enough documents to fill `limit` sentences on the requested side
    # without draining the corpus. After the shuffle, our side holds ~`ratio`
    # of all collected sentences, so we keep pulling docs until the projected
    # side total clears the target (plus a small buffer for the estimate).
    target = limit if limit is not None else 200_000
    ratio = 0.8 if want_train else 0.2

    docs: List[List[str]] = []
    total_sentences = 0
    for record in ds:
        sents = doc_sentences(record.get("text", ""))
        if not sents:
            continue
        docs.append(sents)
        total_sentences += len(sents)
        if total_sentences * ratio >= target + 5_000:
            break

    rng = random.Random(42)
    rng.shuffle(docs)
    n_train_docs = int(len(docs) * 0.8)
    chosen = docs[:n_train_docs] if want_train else docs[n_train_docs:]
    for doc in chosen:
        sentences.extend(doc)
    return sentences


# Dispatch: dataset name -> loader. Adding a dataset is adding a function here;
# the shared dedup + limit tail below stays in load_sentences_from_dataset.
_DATASET_LOADERS = {
    "CAA": _load_caa,
    "MWE": _load_mwe,
    "TQA": _load_tqa,
    "GENERATED": _load_generated,
    "FINEWEB": _load_fineweb,
}


def load_sentences_from_dataset(
    dataset_name: str,
    behaviors: Optional[List[str]],
    split: str,
    data_root: str,
    limit: Optional[int],
    hf_cache_dir: Optional[str] = None,
) -> List[str]:
    try:
        loader = _DATASET_LOADERS[dataset_name]
    except KeyError:
        raise ValueError(
            f"Unknown dataset_name: {dataset_name!r}. "
            "Expected 'CAA', 'MWE', 'TQA', 'GENERATED', or 'FINEWEB'."
        )
    sentences = loader(behaviors, split, data_root, limit, hf_cache_dir)

    # Deduplicate while preserving order
    seen = set()
    unique_sentences = []
    for s in sentences:
        if s not in seen:
            seen.add(s)
            unique_sentences.append(s)

    if limit is not None:
        unique_sentences = unique_sentences[:limit]

    return unique_sentences


def prepare_paired_activations(config: Dict) -> Dict:
    batch_size = config.get("batch_size", 8)
    hf_cache_dir = os.getenv("HF_CACHE_DIR")

    print(f"Loading sentences from {config['dataset']['name']} ({config['dataset'].get('split', 'train')} split)...")
    sentences = load_sentences_from_dataset(
        dataset_name=config["dataset"]["name"],
        behaviors=config["dataset"].get("behaviors"),
        split=config["dataset"].get("split", "train"),
        data_root=config["dataset"]["data_root"],
        limit=config["dataset"].get("limit"),
        hf_cache_dir=hf_cache_dir,
    )
    print(f"  {len(sentences)} sentences loaded")

    # Sentences and activations live under data_dir; only translators and
    # TensorBoard logs go to output_dir (written later by the trainer).
    data_dir = data_dir_of(config)
    data_dir.mkdir(parents=True, exist_ok=True)
    with open(sentences_path(data_dir), "w") as f:
        json.dump(sentences, f, indent=2)

    source_cfg = config["source_model"]
    target_cfg = config["target_model"]
    # Per-(model, sentence-set) caches so switching one model never clobbers the other.
    source_save_path = activation_path(data_dir, source_cfg)
    target_save_path = activation_path(data_dir, target_cfg)
    source_save_path.parent.mkdir(parents=True, exist_ok=True)
    # Reuse pre-refactor flat caches only if they match this sentence set AND model width.
    _migrate_legacy_activations(
        source_save_path, legacy_activation_path(data_dir, "source"),
        len(sentences), _model_hidden_size(source_cfg, hf_cache_dir),
    )
    _migrate_legacy_activations(
        target_save_path, legacy_activation_path(data_dir, "target"),
        len(sentences), _model_hidden_size(target_cfg, hf_cache_dir),
    )

    n_gpus = torch.cuda.device_count()

    if n_gpus >= 2:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        print(f"  {n_gpus} GPUs available — running source (GPU:0) and target (GPU:1) extraction in parallel")
        ctx = mp.get_context("spawn")
        manager = ctx.Manager()
        tqdm_lock = manager.Lock()
        tqdm.set_lock(tqdm_lock)
        print()  # reserve a blank line so position=1 bar has room above it
        with ProcessPoolExecutor(max_workers=2, mp_context=ctx) as executor:
            src_future = executor.submit(
                _extraction_worker,
                {
                    "model_cfg": source_cfg,
                    "sentences": sentences,
                    "device": "cuda:0",
                    "save_path": str(source_save_path),
                    "hf_cache_dir": hf_cache_dir,
                    "batch_size": batch_size,
                    "label": "source/GPU:0",
                    "tqdm_lock": tqdm_lock,
                    "tqdm_position": 0,
                },
            )
            tgt_future = executor.submit(
                _extraction_worker,
                {
                    "model_cfg": target_cfg,
                    "sentences": sentences,
                    "device": "cuda:1",
                    "save_path": str(target_save_path),
                    "hf_cache_dir": hf_cache_dir,
                    "batch_size": batch_size,
                    "label": "target/GPU:1",
                    "tqdm_lock": tqdm_lock,
                    "tqdm_position": 1,
                },
            )
            src_future.result()
            tgt_future.result()
        print()  # move cursor below both bars after they finish

        source_tensor = torch.load(source_save_path, weights_only=False)["activations"]
        target_tensor = torch.load(target_save_path, weights_only=False)["activations"]
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\nLoading source model: {source_cfg['name']}")
        source_tensor = extract_activations_with_resume(
            model_cfg=source_cfg,
            sentences=sentences,
            device=device,
            save_path=source_save_path,
            hf_cache_dir=hf_cache_dir,
            batch_size=batch_size,
            label="source",
        )

        print(f"\nLoading target model: {target_cfg['name']}")
        target_tensor = extract_activations_with_resume(
            model_cfg=target_cfg,
            sentences=sentences,
            device=device,
            save_path=target_save_path,
            hf_cache_dir=hf_cache_dir,
            batch_size=batch_size,
            label="target",
        )

    return {
        "source": source_tensor,
        "target": target_tensor,
        "sentences": sentences,
    }
