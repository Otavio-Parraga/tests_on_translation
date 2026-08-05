"""Extract FineWeb activations for ONE model across a sweep of layers.

Unlike prepare_activations.py (which extracts a source+target *pair* and re-streams
FineWeb from the network on every call), this driver:

  * extracts only the config's TARGET model, at each layer in --layers;
  * reuses the persisted data/<set>/sentences.json when present, so it never
    touches the network and produces activations ROW-ALIGNED with the existing
    Llama caches (same 30k sentences, same order) — which is exactly what the
    later Procrustes fit needs to pair source and target.

Each (model, layer) cache is keyed independently (see acttrans.utils.paths), so
already-complete layers are skipped and interrupted layers resume.

Run on a single GPU by exporting CUDA_VISIBLE_DEVICES first, e.g.:

    CUDA_VISIBLE_DEVICES=1 conda run -n acteng python extract_activations_sweep.py \
        --config config/new_models/fineweb_qwen0.5b.toml --layers 8 10 12 14 16
"""

import argparse
import json
import os

from dotenv import load_dotenv

load_dotenv()

import torch

from acttrans.data.dataset import extract_activations_with_resume, load_sentences_from_dataset
from acttrans.utils.config import load_config
from acttrans.utils.paths import activation_path, data_dir_of, sentences_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--layers", type=int, nargs="+", required=True,
                    help="Target-model layers to extract (each cached separately).")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="Override the config's batch_size.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_dir = data_dir_of(cfg)
    data_dir.mkdir(parents=True, exist_ok=True)
    hf_cache = os.getenv("HF_CACHE_DIR")
    batch_size = args.batch_size or cfg.get("batch_size", 32)

    # Reuse the EXACT sentence set the existing caches were built on. sentences.json
    # is the ground truth (FINEWEB loading is deterministic — seed 42 — but reading
    # the persisted list avoids the network entirely and removes any risk of a
    # different order breaking source/target row alignment).
    sp = sentences_path(data_dir)
    if sp.exists():
        with open(sp) as f:
            sentences = json.load(f)
        print(f"Loaded {len(sentences)} cached sentences from {sp}")
    else:
        print(f"No cached sentences at {sp}; reconstructing from the dataset config…")
        sentences = load_sentences_from_dataset(
            dataset_name=cfg["dataset"]["name"],
            behaviors=cfg["dataset"].get("behaviors"),
            split=cfg["dataset"].get("split", "train"),
            data_root=cfg["dataset"]["data_root"],
            limit=cfg["dataset"].get("limit"),
            hf_cache_dir=hf_cache,
        )
        with open(sp, "w") as f:
            json.dump(sentences, f, indent=2)
        print(f"Streamed and saved {len(sentences)} sentences to {sp}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tgt = dict(cfg["target_model"])
    name = tgt["name"]
    print(f"Model: {name}  |  device: {device}  |  layers: {args.layers}  |  batch: {batch_size}")

    for layer in args.layers:
        tgt["layer"] = layer
        save_path = activation_path(data_dir, tgt)
        print(f"\n=== {name} layer {layer} -> {save_path} ===", flush=True)
        acts = extract_activations_with_resume(
            model_cfg=tgt,
            sentences=sentences,
            device=device,
            save_path=save_path,
            hf_cache_dir=hf_cache,
            batch_size=batch_size,
            label=f"{name.split('/')[-1]}_l{layer}",
        )
        print(f"    done: {tuple(acts.shape)}", flush=True)

    print("\nALL LAYERS DONE")


if __name__ == "__main__":
    main()
