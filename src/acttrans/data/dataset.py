import gc
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import torch
from tqdm import tqdm

from ..utils.hf import append_mask_token, get_layer, load_model_and_tokenizer, resolve_mask_token_id
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


def _apply_chat_template(tokenizer, sentences: List[str], max_len: int, prefix: str = "") -> List[str]:
    """Wrap each sentence as a user turn in the model's chat template, with the
    assistant turn opened (`add_generation_prompt=True`) — the same construction
    extract_caa_diff_source.py uses for the steering vectors this data fits a map
    for, minus the trailing "Answer: {choice}" (there is no choice for plain text).

    Long sentences are shortened BEFORE templating so the template's own trailing
    tokens always survive. Right-truncating the templated string instead would cut
    the assistant-turn boilerplate off the end, which is exactly where a
    token_position=-1 readout lands — leaving a handful of rows read at a content
    token while every other row is read at boilerplate.
    """
    if getattr(tokenizer, "chat_template", None) is None:
        raise ValueError(
            f"chat_template=True but {tokenizer.__class__.__name__} has no chat_template. "
            "Only instruct/chat checkpoints can be templated."
        )

    def wrap(text: str) -> str:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True
        )

    overhead = len(tokenizer(wrap(""), add_special_tokens=False)["input_ids"])
    budget = max_len - overhead
    if budget <= 0:
        raise ValueError(f"Chat template alone needs {overhead} tokens, over the {max_len} budget.")

    out, n_truncated = [], 0
    for s in sentences:
        ids = tokenizer(s, add_special_tokens=False)["input_ids"]
        if len(ids) > budget:
            s = tokenizer.decode(ids[:budget])
            n_truncated += 1
        out.append(wrap(s))
    print(f"{prefix}chat_template ON: {overhead} template tokens/row, "
          f"{n_truncated}/{len(sentences)} sentences shortened to fit {budget}.")
    return out


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

    # chat_template (opt-in, default OFF so every existing cache and config keeps
    # its exact meaning): wrap each sentence in the model's user-turn chat template
    # before tokenizing, instead of feeding the raw sentence. The point is to fit
    # the translator in the SAME region of activation space the steering vectors it
    # transports were extracted from — those come from templated prompts (see
    # extract_caa_diff_source.py) while raw-sentence activations do not.
    #
    # This flag is NOT part of model_slug(), so a templated and a raw cache would
    # collide on filename. Rather than widen the slug convention (which every slug
    # PARSER in checkpoints.py / results_io.py / comparison.common would have to
    # learn), the flag is recorded in the cache and cross-checked on reuse below:
    # extraction skips any cache whose n_sentences already matches, so without this
    # check a templated run would silently REUSE raw activations and report success
    # — the exact failure mode paths.py's slug docstring warns about. Keep
    # templated runs in their own data_dir; the check is the backstop, not the plan.
    chat_template = bool(model_cfg.get("chat_template", False))
    meta_path = batch_dir / "meta.json"

    def _check_flag(found: bool, where: str):
        if found != chat_template:
            raise RuntimeError(
                f"{prefix}chat_template mismatch: {where} was built with "
                f"chat_template={found}, but this run asks for {chat_template}. "
                f"Refusing to mix templated and raw activations in one cache "
                f"({save_path}). Use a separate training.data_dir per variant."
            )

    # Already fully complete?
    if save_path.exists():
        ckpt = torch.load(save_path, weights_only=False)
        if ckpt.get("n_sentences") == len(sentences):
            _check_flag(bool(ckpt.get("chat_template", False)), "the existing cache")
            print(f"{prefix}Already complete ({len(sentences)} sentences), skipping extraction.")
            return ckpt["activations"]

    batch_dir.mkdir(parents=True, exist_ok=True)
    # Same check for a partially-finished run: resuming into batch files written
    # under the other setting would splice two different input formats together.
    if meta_path.exists():
        _check_flag(bool(json.loads(meta_path.read_text()).get("chat_template", False)),
                    "the partial batch directory")
    else:
        meta_path.write_text(json.dumps({"chat_template": chat_template}))

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
        # token_position selects what one vector per sentence is read from:
        #   -1       last *real* token (default; the causal-LM convention)
        #   "mean"   mask-weighted mean over every real (non-pad) token
        #   "mask"   the hidden state at an appended [MASK] position
        #   int      that fixed column, verbatim
        # Every mask-aware mode uses the attention mask, so all are correct under
        # left- OR right-padding.
        #
        # Why "mask" exists: a masked-diffusion LM (LLaDA) has bidirectional
        # attention, so no position is privileged the way a causal model's last
        # token is — but the *mask* position is the one slot the model was trained
        # to answer "what token belongs here", making it the structural analogue of
        # last-token readout. It is also where the A/B eval reads its logits, so a
        # vector fit here is fit in the space it will be applied in.
        token_position = model_cfg.get("token_position", -1)
        pooling_mean = str(token_position) == "mean"
        pooling_mask = str(token_position) == "mask"
        if not (pooling_mean or pooling_mask or isinstance(token_position, int)):
            # Fail here rather than let an unrecognized string reach the tensor
            # index below, where it would surface as an opaque IndexError from
            # inside a forward hook after the model was already loaded.
            raise ValueError(
                f"Unsupported token_position {token_position!r}. Expected an int "
                '(e.g. -1), "mean", or "mask".'
            )

        mask_token_id = resolve_mask_token_id(model, tokenizer) if pooling_mask else None
        captured = {}

        def hook_fn(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output  # [B, T, D]
            mask = captured["attention_mask"]  # [B, T]
            if pooling_mean:
                m = mask.unsqueeze(-1).to(hidden.dtype)  # [B, T, 1]
                # masked_fill, not `hidden * m`: multiplying by a 0 mask turns a
                # non-finite pad-position value into NaN (0 * NaN = NaN) and
                # poisons the whole pooled vector, so garbage at a position that
                # is meant to be EXCLUDED still decides the result. Substituting
                # zeros makes the exclusion real.
                summed = hidden.masked_fill(m == 0, 0.0).sum(dim=1)
                counts = m.sum(dim=1).clamp(min=1.0)
                captured["act"] = (summed / counts).detach().cpu()
            elif pooling_mask:
                # The appended [MASK] column, whose index the tokenize step
                # recorded per row (it varies with each sentence's length).
                rows = torch.arange(hidden.shape[0], device=hidden.device)
                captured["act"] = hidden[rows, captured["mask_pos"], :].detach().cpu()
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
        # max_length budget: one slot is reserved when a [MASK] will be appended.
        max_len = 511 if pooling_mask else 512
        texts = _apply_chat_template(tokenizer, sentences, max_len, prefix) if chat_template else sentences
        remaining = texts[start_idx:]
        n_batches = (len(remaining) + batch_size - 1) // batch_size

        tqdm_kwargs = dict(total=n_batches, desc=f"{prefix}Extracting", unit="batch")
        if tqdm_position is not None:
            tqdm_kwargs.update(position=tqdm_position, leave=True, dynamic_ncols=False)

        try:
            for step, i in enumerate(tqdm(range(0, len(remaining), batch_size), **tqdm_kwargs)):
                batch = remaining[i : i + batch_size]
                encoded = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    # Reserve a slot for the appended [MASK] so the widened
                    # sequence still respects the 512 budget.
                    max_length=max_len,
                    # A chat template already emits its own BOS/role headers, so
                    # letting the tokenizer add specials on top would duplicate BOS
                    # and shift every template-relative position by one.
                    add_special_tokens=not chat_template,
                    return_tensors="pt",
                )
                encoded = {k: v.to(device) for k, v in encoded.items()}
                if pooling_mask:
                    ids, attn, mask_pos = append_mask_token(
                        encoded,
                        mask_token_id,
                        tokenizer.pad_token_id,
                        getattr(tokenizer, "padding_side", "right"),
                    )
                    encoded["input_ids"], encoded["attention_mask"] = ids, attn
                    captured["mask_pos"] = mask_pos
                    # token_type_ids (if the tokenizer emitted any) would now be one
                    # column short of input_ids. None of these models use them, and
                    # dropping them is what the widened batch requires.
                    encoded.pop("token_type_ids", None)
                captured["attention_mask"] = encoded["attention_mask"]
                with torch.no_grad():
                    try:
                        model(**encoded)
                    except _StopForward:
                        pass
                act = captured["act"]
                # Fail on the FIRST bad batch rather than writing 30000 rows of
                # NaN and exiting 0. A corrupt model (e.g. a rotary table left
                # uninitialized) produces activations that are structurally
                # perfect — right shape, right count, right dtype — so nothing
                # downstream notices until a translator trains to nothing.
                if not torch.isfinite(act).all():
                    n_bad = int((~torch.isfinite(act)).any(dim=1).sum())
                    raise RuntimeError(
                        f"Non-finite activations in batch {n_existing + step} "
                        f"({n_bad}/{act.shape[0]} rows) for {model_cfg['name']} "
                        f"layer {model_cfg['layer']}. Refusing to write a corrupt "
                        f"cache. Sentences {start_idx + i}..{start_idx + i + len(batch) - 1}."
                    )
                torch.save(act, batch_dir / f"batch_{n_existing + step:06d}.pt")
        finally:
            handle.remove()

        # Release the weights before the caller loads the next layer's copy.
        # `layer` and `captured` are dropped too: `layer` holds a decoder block (so
        # its parameters), and `captured` holds the last batch's GPU attention mask.
        #
        # gc.collect() is not paranoia here. empty_cache() can only return memory
        # that no live tensor holds, so anything keeping the model reachable keeps
        # its weights resident — and a reference CYCLE keeps an object reachable
        # even after `del`, since refcounting alone cannot break one. That is
        # exactly what stranded ~16 GB per reload until the sweep OOM'd (the two
        # compat shims in utils/hf.py used to build such a cycle; they now patch at
        # class level). One collection per layer costs milliseconds against a
        # multi-minute extraction, and makes this loop robust to the next shim that
        # accidentally captures a model.
        captured.clear()
        del model, layer
        gc.collect()
        torch.cuda.empty_cache()

    # Merge all batch files into the final output file
    all_batches = sorted(batch_dir.glob("batch_*.pt"))
    merged = torch.cat([torch.load(b, weights_only=True) for b in all_batches], dim=0)
    # `sentences` stays the RAW sentence set even under chat_template — it is the
    # row-alignment key shared with every other cache on this sentence set, and
    # the templated form is fully recoverable from the flag + tokenizer.
    torch.save({"activations": merged, "sentences": sentences, "n_sentences": len(sentences),
                "chat_template": chat_template}, save_path)
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
