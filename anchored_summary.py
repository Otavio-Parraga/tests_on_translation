"""Summarize the anchored-translator study into one table.

Answers the study's headline question in a single CSV: at each pair's best target
layer, does a translator ANCHORED on the frozen closed-form Procrustes map beat
that floor, and does the from-scratch control still lose to it?

Rows are (target model, target layer); columns are the three variants that live
side by side in outputs/anchored/:

    procrustes     the closed-form floor          (..__linear__procrustes.pt)
    mlp_scratch    trained from scratch           (..__mlp__cosine+info_nce.pt)
    mlp_anchored   trained on the frozen anchor   (..__mlp+procrustes__cosine+info_nce.pt)

Two families of number per cell:

  mean_cos / rank1   from the geometric comparison against NATIVE steering vectors
                     (compare_translated_and_original.py geometric). This is the
                     metric the study is actually about — but it needs native SVs
                     for the target model, which one pair does not have.

  val_cos            held-out ACTIVATION reconstruction cosine, recomputed here
                     from the checkpoint's own config (same preprocessing, same
                     seeded split, val half). It needs no steering vectors, so it
                     is the only measurable column for a target model with no
                     native SVs in the tree — and it doubles as a sanity check that
                     an anchored run never starts below its floor.

Usage:
    conda run -n acteng python anchored_summary.py
    conda run -n acteng python anchored_summary.py --cmp outputs/anchored/cmp \
        --translators 'outputs/anchored/best_translator__*.pt' --out outputs/anchored/summary.csv
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import torch
import torch.nn.functional as F

from acttrans.data.split import preprocess_activations, split_paired_activations
from acttrans.models.translator import build_translator
from acttrans.utils.checkpoints import discover_translators
from acttrans.utils.config import load_activations, resolve_activation_paths

# Column order in the output table: floor first, then the two trained variants, so
# a row reads left-to-right as "floor -> control -> the thing being tested".
VARIANTS = ["procrustes", "mlp_scratch", "mlp_anchored"]


def variant_of(info) -> str:
    """Map a parsed checkpoint to one of the three study variants (or None)."""
    if info.ttype == "linear":
        return "procrustes" if info.loss.startswith("procrustes") else None
    if info.ttype == "mlp":
        return "mlp_anchored" if info.anchor == "procrustes" else "mlp_scratch"
    return None


@torch.no_grad()
def val_cosine(path: Path, device: str) -> float:
    """Held-out activation reconstruction cosine for one checkpoint.

    Rebuilt from the checkpoint's own config so the preprocessing and the seeded
    split match exactly what that run trained under — comparing a raw-activation
    run against a normalized one on this metric would be meaningless."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    model = build_translator(
        config, input_dim=ckpt["input_dim"], output_dim=ckpt["output_dim"]
    )
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device).eval()

    src_path, tgt_path = resolve_activation_paths(config, None, None)
    source = load_activations(src_path).float()
    target = load_activations(tgt_path).float()
    source, target = preprocess_activations(source, target, config)
    _, _, val_src, val_tgt = split_paired_activations(source, target, config)

    # Chunked: the val half of a 30k FineWeb split at D=4096 is small, but the MLP's
    # intermediate activations are not, and this shares an 11 GB GPU.
    cos_sum, n = 0.0, 0
    for i in range(0, val_src.shape[0], 2048):
        xb = val_src[i:i + 2048].to(device)
        yb = val_tgt[i:i + 2048].to(device)
        c = F.cosine_similarity(model(xb), yb, dim=-1)
        cos_sum += c.sum().item()
        n += c.numel()
    return cos_sum / n


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cmp", default="outputs/anchored/cmp",
                        help="Root passed to compare_translated_and_original.py --out")
    parser.add_argument("--translators", nargs="+",
                        default=["outputs/anchored/best_translator__*.pt"],
                        help="Checkpoint glob(s) to summarize")
    parser.add_argument("--out", default="outputs/anchored/summary.csv")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-val-cos", action="store_true",
                        help="Skip the held-out reconstruction pass (geometry only)")
    args = parser.parse_args()

    # ── geometric numbers, keyed by checkpoint name ──────────────────────────
    geo = {}
    geo_csv = Path(args.cmp) / "geometric" / "per_translator.csv"
    if geo_csv.exists():
        for row in csv.DictReader(open(geo_csv)):
            geo[row["translator"]] = row
    else:
        print(f"!! no {geo_csv}; mean_cos/rank1 columns will be empty")

    # ── one row per (target model, target layer) ─────────────────────────────
    cells = defaultdict(dict)
    for info in discover_translators(args.translators):
        variant = variant_of(info)
        if variant is None:
            continue
        g = geo.get(info.name, {})
        # tgt_model is only in the geometric table; fall back to the checkpoint's
        # own config so a pair with no native SVs still gets a labelled row.
        tgt_model = g.get("tgt_model")
        if tgt_model is None:
            tgt_model = torch.load(info.path, map_location="cpu",
                                   weights_only=False)["config"]["target_model"]["name"]
        cell = {
            "mean_cos": g.get("mean_cos", ""),
            "rank1": g.get("identity_rank1_frac", ""),
            "val_cos": "",
        }
        if not args.no_val_cos:
            cell["val_cos"] = f"{val_cosine(info.path, args.device):.4f}"
        cells[(tgt_model, info.tgt_layer)][variant] = cell
        print(f"  {info.name}  -> {variant}")

    fields = ["target_model", "target_layer"]
    for v in VARIANTS:
        fields += [f"{v}__mean_cos", f"{v}__rank1", f"{v}__val_cos"]

    rows = []
    for (tgt_model, layer) in sorted(cells, key=lambda k: (k[0], k[1])):
        row = {"target_model": tgt_model, "target_layer": layer}
        for v in VARIANTS:
            cell = cells[(tgt_model, layer)].get(v, {})
            for key in ("mean_cos", "rank1", "val_cos"):
                val = cell.get(key, "")
                if key != "val_cos" and val not in ("", None):
                    val = f"{float(val):.4f}"
                row[f"{v}__{key}"] = val
        rows.append(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {out} ({len(rows)} rows)")

    # Console rendering of the same table.
    hdr = f"{'target model':<26s} {'lyr':>3s} " + " ".join(
        f"{v:>28s}" for v in VARIANTS
    )
    print("\n" + hdr)
    print("-" * len(hdr))
    for row in rows:
        cols = []
        for v in VARIANTS:
            mc, r1, vc = row[f"{v}__mean_cos"], row[f"{v}__rank1"], row[f"{v}__val_cos"]
            cols.append(f"{mc or '   -  ':>8s}/{r1 or ' -  ':>5s}/{vc or ' -  ':>7s}")
        print(f"{row['target_model'].split('/')[-1]:<26s} {row['target_layer']:>3d} "
              + " ".join(f"{c:>28s}" for c in cols))
    print("\n(cells are mean_cos / identity_rank1_frac / held-out val cosine)")


if __name__ == "__main__":
    main()
