"""Extract per-pair (positive - negative) activation differences on the SOURCE
model (Llama-3.2-1B-Instruct), over every item in build_caa_diff_manifest.py's
manifest, at the exact recipe the real CAA steering vectors were built with.

FAITHFULNESS TO THE REAL sv.pt FILES IS THE WHOLE POINT. The manifest's rows are
row-aligned across source and target extraction, but the SOURCE side additionally
must reproduce, item-for-item, what activation_engineering/methods/CAA.py actually
did when it built steering_vectors/meta-llama_Llama-3.2-1B-Instruct/CAA/*/sv.pt —
otherwise a translator fit on "some other vector that happens to differ per
pos/neg pair" would not be fitting on the quantity we transport at all. Read from
the checked-in sv.json sidecar (build metadata, `vars(self)` at save time) and
activation_engineering/data/base.py::_apply_chat_template:

    prompt   = tok.apply_chat_template([{"role": "user", "content": question}],
                                        tokenize=False, add_generation_prompt=True)
    pos_text = prompt + f"Answer: {pos_choice}"        # pos_choice e.g. "(A)"
    neg_text = prompt + f"Answer: {neg_choice}"
    token_position = -2   # sv.json: {"token_position": -2, "module_name": "residual", ...}
    module         = "residual"     (get_layer(model, layer, "residual"): the LAYER's
                                      own output — matches every other readout in
                                      this repo, see utils/hf.py::get_layer)
    tokenize with add_special_tokens=False, ONE SEQUENCE AT A TIME (no padding) —
    the original code never batches (methods/CAA.py's calculate_steering_vector
    loops `for pos_prompt, neg_prompt in zip(...)`), which is also why -2 is safe to
    read directly: with no padding, -2 always lands on the real second-to-last
    token, not a pad slot.

diff[i] = act(pos_text_i)[-2] - act(neg_text_i)[-2], at layer 8 (SOURCE_LAYER).

VERIFICATION, printed automatically at the end: per-behavior mean(diff) compared by
cosine + norm ratio against the real, already-on-disk sv.pt for that behavior. They
are not expected to match exactly — the real vectors were built with whatever
random/library-version state activation_engineering had at build time, and floating
point at 1B-param scale is not bit-reproducible across runs/machines/torch
versions — but a cosine anywhere below roughly 0.9 would mean the recipe above has
drifted from the real one and the extraction should not be trusted for fitting.

Resumable: partial progress is flushed to a temp file every --flush-every items and
picked back up on restart. Touches the GPU (loads the ~1B-param source model) —
this is why it's a separate script from build_caa_diff_manifest.py rather than a
step folded into it.

Usage:
    CUDA_VISIBLE_DEVICES=2 conda run -p /mnt/C-SSD/parraga/conda_envs/act_eng python \\
        extract_caa_diff_source.py
"""
import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import torch
from tqdm import tqdm

from acttrans.constants import LAYER, SOURCE_MODEL
from acttrans.utils.hf import get_layer, load_model_and_tokenizer

_HERE = Path(__file__).resolve().parent
_ACTENG = _HERE.parent / "activation_engineering"
MANIFEST_PATH = _HERE / "data" / "caa_diff" / "pairs_manifest.json"
TOKEN_POSITION = -2   # activation_engineering/steering_vectors/.../sv.json


def build_pair_text(tok, question: str, choice: str) -> str:
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True
    )
    return prompt + f"Answer: {choice}"


@torch.no_grad()
def extract_one(model, tok, layer_module, question: str, choice: str, device: str) -> torch.Tensor:
    text = build_pair_text(tok, question, choice)
    ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    captured = {}

    def hook(module, inp, out):
        act = out[0] if isinstance(out, tuple) else out
        captured["h"] = act[0, TOKEN_POSITION, :].detach().float().cpu()

    handle = layer_module.register_forward_hook(hook)
    try:
        model(input_ids=ids)
    finally:
        handle.remove()
    return captured["h"]


def load_partial(path: Path, n_total: int, dim: int):
    if path.exists():
        d = torch.load(path, weights_only=False)
        if d["diffs"].shape[1] == dim:
            return d["diffs"], d["pos_acts"], d["neg_acts"], d["n_done"]
    return (torch.zeros(n_total, dim), torch.zeros(n_total, dim), torch.zeros(n_total, dim), 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=str(MANIFEST_PATH))
    ap.add_argument("--model", default=SOURCE_MODEL)
    ap.add_argument("--layer", type=int, default=LAYER)
    ap.add_argument("--output", default=None,
                    help="Default: data/caa_diff/activations/<model-short>_l<layer>_diff.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--flush-every", type=int, default=200)
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    rows = manifest["rows"]
    n = len(rows)

    short = args.model.split("/")[-1]
    out_path = Path(args.output) if args.output else (
        _HERE / "data" / "caa_diff" / "activations" / f"{short}_l{args.layer}_diff.pt"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".partial.pt")

    print(f"Model: {args.model}  |  layer {args.layer}  |  device {args.device}  |  {n} pairs")
    model, tok = load_model_and_tokenizer(args.model, args.device)
    model.eval()
    layer_module = get_layer(model, args.layer, "residual")

    dim = model.config.hidden_size
    diffs, pos_acts, neg_acts, done = load_partial(tmp_path, n, dim)
    if done:
        print(f"  resuming from {done}/{n} (partial file found)")

    for i in tqdm(range(done, n), desc="  pairs", initial=done, total=n):
        r = rows[i]
        pos = extract_one(model, tok, layer_module, r["question"], r["pos_choice"], args.device)
        neg = extract_one(model, tok, layer_module, r["question"], r["neg_choice"], args.device)
        pos_acts[i], neg_acts[i] = pos, neg
        diffs[i] = pos - neg

        if (i + 1) % args.flush_every == 0 or i + 1 == n:
            torch.save({"diffs": diffs, "pos_acts": pos_acts, "neg_acts": neg_acts, "n_done": i + 1}, tmp_path)

    behaviors = [r["behavior"] for r in rows]
    torch.save({
        "diffs": diffs, "pos_acts": pos_acts, "neg_acts": neg_acts,
        "behaviors": behaviors, "n_pairs": n,
        "manifest_hash": manifest["hash"], "token_position": TOKEN_POSITION,
        "model": args.model, "layer": args.layer, "module": "residual",
    }, out_path)
    tmp_path.unlink(missing_ok=True)
    print(f"  wrote {out_path}")

    # ── Verification against the real, already-built sv.pt files ────────────
    print("\nVerification vs. the real activation_engineering sv.pt (cosine / norm ratio):")
    import collections
    by_beh = collections.defaultdict(list)
    for i, b in enumerate(behaviors):
        by_beh[b].append(i)
    for behavior, idx in by_beh.items():
        mean_diff = diffs[idx].mean(dim=0)
        sv_path = (_ACTENG / "steering_vectors" / args.model.replace("/", "_") / "CAA"
                   / behavior / "residual" / f"layer_{args.layer}" / "sv.pt")
        if not sv_path.exists():
            print(f"  {behavior:24s}  (no reference sv.pt at {sv_path})")
            continue
        # squeeze(): the real sv.pt keeps a retained batch dim ([1, D], from
        # methods/CAA.py's `target_act[:, pos, :]` never squeezing dim 0) —
        # comparing it against our clean [D] mean_diff without matching shapes
        # first broadcasts wrong and cosine_similarity(dim=0) silently reduces
        # the WRONG axis, returning a [D]-shaped result instead of a scalar.
        real = torch.load(sv_path, map_location="cpu", weights_only=True).float().squeeze()
        cos = torch.nn.functional.cosine_similarity(mean_diff, real, dim=0).item()
        ratio = (mean_diff.norm() / real.norm()).item()
        flag = "" if cos > 0.9 else "  <-- LOW, check the recipe above"
        print(f"  {behavior:24s}  cosine={cos:.4f}  norm_ratio={ratio:.4f}{flag}")


if __name__ == "__main__":
    main()
