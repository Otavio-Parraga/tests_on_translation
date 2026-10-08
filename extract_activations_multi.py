"""Extract FineWeb activations for ONE model at SEVERAL layers and SEVERAL read
positions in a SINGLE pass over the sentences.

The sibling extract_activations_sweep.py reads one layer at one position per
pass; this driver hooks every requested layer at once and reduces each layer's
[B, T, D] output at every requested position, so the whole layers x positions
grid costs one pass (see acttrans.data.multi_extract for why that is ~10x
cheaper at 500k). Output caches are format-compatible with the sweep script's,
land at the same activation_path() locations, and already-complete caches from
either script are skipped.

The persisted data/<set>/sentences.json is REQUIRED (never re-streamed): it is
the row-alignment key shared with every other cache on the sentence set.

Usage:
    CUDA_VISIBLE_DEVICES=2 conda run -p /mnt/C-SSD/parraga/conda_envs/act_eng --no-capture-output python \\
        extract_activations_multi.py --config config/new_models/fineweb_llada8b_500k_chat.toml \\
        --layers 10 12 14 16 18 20 22 --positions mean -1
    ... --role source --layers 8 --positions -1      # the [source_model] block instead
"""
import argparse
import json
import os

from dotenv import load_dotenv

load_dotenv()

import torch

from acttrans.data.multi_extract import extract_multi_with_resume, parse_position
from acttrans.utils.config import load_config
from acttrans.utils.paths import data_dir_of, sentences_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--role", default="target", choices=["target", "source"],
                    help="Which model block to extract (default: target).")
    ap.add_argument("--layers", type=int, nargs="+", required=True)
    ap.add_argument("--positions", type=parse_position, nargs="+", default=["mean", -1],
                    metavar="POS", help="Read positions: 'mean' and/or -1 (default: both).")
    ap.add_argument("--batch-size", type=int, default=None, help="Override the config's batch_size.")
    ap.add_argument("--flush-every", type=int, default=64,
                    help="Write a resumable chunk every N batches (default 64).")
    ap.add_argument("--min-free-gb", type=float, default=10.0,
                    help="Stop cleanly (resumable) if free disk under data_dir drops below this.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_dir = data_dir_of(cfg)
    sp = sentences_path(data_dir)
    if not sp.exists():
        raise SystemExit(
            f"{sp} is missing. Refusing to re-stream FineWeb: the persisted sentence set is the "
            "row-alignment key for every cache in this data_dir. Copy it from the sentence set "
            "the paired caches were built on.")
    with open(sp) as f:
        sentences = json.load(f)
    print(f"Loaded {len(sentences)} cached sentences from {sp}")

    mcfg = dict(cfg[f"{args.role}_model"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = args.batch_size or cfg.get("batch_size", 32)
    print(f"Model: {mcfg['name']} ({args.role})  |  device: {device}  |  layers: {args.layers}  "
          f"|  positions: {[str(p) for p in args.positions]}  |  batch: {batch_size}  "
          f"|  chat_template: {bool(mcfg.get('chat_template', False))}")

    extract_multi_with_resume(
        model_cfg=mcfg, sentences=sentences, device=device, data_dir=data_dir,
        layers=args.layers, positions=args.positions, batch_size=batch_size,
        hf_cache_dir=os.getenv("HF_CACHE_DIR"), flush_every=args.flush_every,
        min_free_gb=args.min_free_gb, label=mcfg["name"].split("/")[-1],
    )
    print("\nALL DONE")


if __name__ == "__main__":
    main()
