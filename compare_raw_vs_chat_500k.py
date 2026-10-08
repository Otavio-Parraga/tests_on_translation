"""Side-by-side comparison of the RAW vs CHAT-TEMPLATED 500k Procrustes arms on
the LLaDA-8B-Instruct A/B steering sweep.

Reads two ab_report.py summary.csv files — one per arm — and joins them by
(target layer, Procrustes variant, norm_mode). The two arms MUST stay in separate
results directories: their translator checkpoints carry identical filenames
(e.g. ..._l22_mean__linear__procrustes) and results_io's dedup key is built from
that name, so merging the two JSONLs would silently collapse one arm into the
other. This script is the join instead.

Arms (source is always Llama-3.2-1B layer 8, read at the last token):
    raw-mean    raw sentences,        target mean-pooled          (outputs/ab_eval/llada_500k)
    chat-mean   chat-templated,       target mean-pooled          (outputs/ab_eval/llada_500k_chat)
    chat-tok-1  chat-templated,       target last token (-1)      (outputs/ab_eval/llada_500k_chat)

Metric: dP_coherent from ab_report (steering effect within the coherent
coefficient range; higher = more behaviour change), averaged over behaviours
per (layer, variant, norm_mode). Layers 24/26 exist only in the chat arm.

Usage:
    python compare_raw_vs_chat_500k.py
    python compare_raw_vs_chat_500k.py --out outputs/ab_eval/compare_raw_vs_chat_500k.csv
"""
import argparse
import csv
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
RAW = _HERE / "outputs" / "ab_eval" / "llada_500k" / "summary.csv"
CHAT = _HERE / "outputs" / "ab_eval" / "llada_500k_chat" / "summary.csv"
_TGT_RE = re.compile(r"_l(\d+)(_mean|_mask)?$")


def target_pooling(translator: str) -> str:
    """'best_translator__<src>__<tgt>__linear__<loss>' -> 'mean' | 'mask' | 'tok-1'."""
    parts = translator.split("__")
    if len(parts) < 3:
        return "?"
    m = _TGT_RE.search(parts[2])
    if not m:
        return "?"
    return m.group(2).lstrip("_") if m.group(2) else "tok-1"


def load_arm(path: Path, tag: str):
    """-> {(layer, variant, norm_mode, pooling): [dP_coherent per behaviour]}"""
    out = defaultdict(list)
    if not path.exists():
        print(f"!! {tag}: {path} not found — run ab_report.py on that results dir first.")
        return out
    with open(path) as f:
        for r in csv.DictReader(f):
            if r.get("translator", "").startswith("best_translator__") is False:
                continue  # native / source rows carry no translator
            try:
                layer = int(r["target_layer"]); v = float(r["dP_coherent"])
            except (KeyError, ValueError, TypeError):
                continue
            key = (layer, r.get("loss", "?"), r.get("norm_mode", "?"), target_pooling(r["translator"]))
            out[key].append(v)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-summary", default=str(RAW))
    ap.add_argument("--chat-summary", default=str(CHAT))
    ap.add_argument("--out", default=str(_HERE / "outputs" / "ab_eval" / "compare_raw_vs_chat_500k.csv"))
    args = ap.parse_args()

    raw = load_arm(Path(args.raw_summary), "raw")
    chat = load_arm(Path(args.chat_summary), "chat")

    # arm name -> {(layer, variant, norm_mode): values}
    arms = {"raw-mean": {}, "chat-mean": {}, "chat-tok-1": {}}
    for (layer, var, nm, pool), vals in raw.items():
        if pool == "mean":
            arms["raw-mean"][(layer, var, nm)] = vals
    for (layer, var, nm, pool), vals in chat.items():
        name = "chat-mean" if pool == "mean" else "chat-tok-1" if pool == "tok-1" else None
        if name:
            arms[name][(layer, var, nm)] = vals

    keys = sorted({k for a in arms.values() for k in a})
    if not keys:
        raise SystemExit("No rows found in either summary — nothing to compare.")

    rows = []
    for var in sorted({k[1] for k in keys}):
        for nm in sorted({k[2] for k in keys if k[1] == var}):
            print(f"\n=== variant={var}  norm_mode={nm}  |  mean dP_coherent over behaviours (n) ===")
            print(f"{'layer':>6} {'depth':>6} | {'raw-mean':>14} | {'chat-mean':>14} | {'chat-tok-1':>14}")
            for layer in sorted({k[0] for k in keys if k[1] == var and k[2] == nm}):
                cells = []
                rec = {"target_layer": layer, "variant": var, "norm_mode": nm}
                for arm in ("raw-mean", "chat-mean", "chat-tok-1"):
                    vals = arms[arm].get((layer, var, nm))
                    if vals:
                        m = st.mean(vals); cells.append(f"{m:+.4f} ({len(vals)})")
                        rec[f"{arm}_dP_coherent"] = f"{m:.6f}"; rec[f"{arm}_n"] = len(vals)
                    else:
                        cells.append(f"{'—':>14}"); rec[f"{arm}_dP_coherent"] = ""; rec[f"{arm}_n"] = 0
                print(f"{layer:>6} {100*layer/32:>5.0f}% | " + " | ".join(f"{c:>14}" for c in cells))
                rows.append(rec)
            for arm in ("raw-mean", "chat-mean", "chat-tok-1"):
                cand = {k: st.mean(v) for k, v in arms[arm].items() if k[1] == var and k[2] == nm}
                if cand:
                    best = max(cand, key=cand.get)
                    print(f"   best {arm:>10}: layer {best[0]}  dP_coherent={cand[best]:+.4f}")

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["target_layer", "variant", "norm_mode"] + [f"{a}_{s}" for a in ("raw-mean", "chat-mean", "chat-tok-1")
                                                       for s in ("dP_coherent", "n")]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
