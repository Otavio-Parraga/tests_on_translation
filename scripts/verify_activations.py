"""Post-extraction integrity check for activation caches.

Exists because an extraction sweep can finish with exit code 0 and still leave
useless output. A corrupt model produces activations that are structurally
perfect — right shape, right row count, right dtype, correct filename — so every
signal the launcher looks at says success. That is exactly how 10 caches of NaN
were written, reported OK, and only found later by hand.

Checks each (model, layer) cache for:
  * existence and the expected row count
  * row alignment with the sentence set the source caches were built on
    (a mismatch silently invalidates every downstream Procrustes fit)
  * finiteness — no NaN/Inf anywhere
  * non-degeneracy — not all-zero, and not the same vector repeated, which is
    what an uninitialized model can produce WITHOUT tripping the NaN check

Exits non-zero if any cache fails, so a launcher can gate on it.

Usage:
    python scripts/verify_activations.py --config <cfg.toml> --layers 6 8 10
"""

import argparse
import sys
import tomllib
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from acttrans.utils.paths import activation_path, data_dir_of, sentences_path  # noqa: E402


def check_one(path: Path, sentences: list) -> list:
    """Return a list of problem strings ([] means the cache is good)."""
    if not path.exists():
        return ["missing"]
    try:
        ckpt = torch.load(path, weights_only=False)
    except Exception as e:  # truncated/corrupt file
        return [f"unreadable ({type(e).__name__})"]

    problems = []
    acts = ckpt.get("activations")
    if acts is None:
        return ["no 'activations' key"]

    if acts.shape[0] != len(sentences):
        problems.append(f"{acts.shape[0]} rows, expected {len(sentences)}")
    if ckpt.get("sentences") != sentences:
        problems.append("sentence set differs -> NOT row-aligned with source caches")

    a = acts.float()
    n_bad = int((~torch.isfinite(a)).any(dim=1).sum())
    if n_bad:
        problems.append(f"{n_bad}/{a.shape[0]} rows non-finite (NaN/Inf)")

    # Degenerate output: an uninitialized or dead model can emit perfectly finite
    # but meaningless activations, which no NaN check would catch.
    norms = a.norm(dim=1)
    if float(norms.max()) == 0.0:
        problems.append("all rows are zero vectors")
    elif float(a.std()) == 0.0:
        problems.append("zero variance — every element identical")
    elif a.shape[0] > 1 and float(norms.std()) == 0.0:
        problems.append("every row has an identical norm — suspect constant output")

    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--layers", type=int, nargs="+", required=True)
    ap.add_argument("--role", default="target_model", choices=["target_model", "source_model"])
    ap.add_argument("--token-position", default=None,
                    help="Override the config block's token_position ('mean', 'mask', or an int "
                         "such as -1) so one config can verify caches at several read positions, "
                         "e.g. the mean and -1 arms extract_activations_multi.py writes together. "
                         "Default: whatever the config says.")
    args = ap.parse_args()
    if args.token_position is not None:
        tp = args.token_position
        args.token_position = tp if tp in ("mean", "mask") else int(tp)

    with open(args.config, "rb") as f:
        cfg = tomllib.load(f)
    data_dir = data_dir_of(cfg)
    import json

    sentences = json.loads(sentences_path(data_dir).read_text())

    print(f"Verifying {len(args.layers)} caches from {args.config}")
    failed = []
    for layer in args.layers:
        mcfg = dict(cfg[args.role])
        mcfg["layer"] = layer
        if args.token_position is not None:
            mcfg["token_position"] = args.token_position
        p = activation_path(data_dir, mcfg)
        problems = check_one(p, sentences)
        if problems:
            failed.append(p.name)
            print(f"  FAIL {p.name}: {'; '.join(problems)}")
        else:
            print(f"  ok   {p.name}")

    if failed:
        print(f"\n{len(failed)}/{len(args.layers)} caches are INVALID — do not use them.")
        return 1
    print(f"\nAll {len(args.layers)} caches valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
