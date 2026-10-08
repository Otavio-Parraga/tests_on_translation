"""Extract per-pair (positive - negative) activation differences on a TARGET
diffusion model, at TEMPLATE-RELATIVE candidate token positions (arXiv
2512.24143v2, "Activation Steering for Masked Diffusion Language Models"),
instead of the ad hoc "mean"/"mask" pooling extract_caa_diff_target.py writes.

WHY A SEPARATE ARM, NOT A REPLACEMENT: extract_caa_diff_target.py's "mean" and
"mask" arms are the two pooling choices the diffusion-target study has used so
far to build the per-behavior ground-truth diff vector fit_procrustes_lobo.py
validates a transported vector against. The paper's contribution is a more
principled way to pick WHERE to read that vector from: grid-search a set of
positions defined relative to the chat template — a few tokens before the
user's question even starts (pre-instruction — only meaningful for a
bidirectional model, since a causal LM's pre-instruction tokens can never see
the question) and a few tokens right where the user's turn ends and the
assistant turn opens (post-instruction; the existing "mask" arm is itself one
member of this family). This script produces the CANDIDATE bank; a separate
selection step (select_papersel_position.py) picks the winner per behavior via
the existing A/B metric and writes it as a new `papersel` arm cache in the
exact format fit_procrustes_lobo.py already knows how to consume.

TEXT CONSTRUCTION is unchanged from extract_caa_diff_target.py/
extract_caa_diff_source.py: the contrastive signal is still the two answer
choices to the same question (this repo's existing CAA recipe), not two
different prompts as in the paper — see chat_positions.py's module docstring
for why bidirectional attention still lets pre-instruction positions pick up a
signal correlated with the (later) answer choice even though the choice text
comes after them in the sequence.

    template_text = tok.apply_chat_template([{"role": "user", "content": question}],
                                             tokenize=False, add_generation_prompt=True)
    pos_text = template_text + f"Answer: {pos_choice}"
    neg_text = template_text + f"Answer: {neg_choice}"

ONE forward pass per (item, choice) — not two like extract_caa_diff_target.py,
since none of the candidate positions require an appended [MASK]; they are all
real tokens already in template_text. All candidate layers are hooked
simultaneously on a single model load, exactly as extract_caa_diff_target.py
already does.

SELF-CHECK BEFORE TOUCHING THE GPU: chat_positions.assert_positions_stable
verifies every question in the manifest resolves to the SAME token identity at
every pre/post-instruction offset — a drifting "structural marker" would
silently mix unrelated positions together across items. Run once, up front,
using only the tokenizer.

Usage:
    CUDA_VISIBLE_DEVICES=2 conda run -p /mnt/C-SSD/parraga/conda_envs/act_eng python \\
        extract_target_papersel.py --layers 10 12 14 16 18 20 22
"""
import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import torch
from tqdm import tqdm

from acttrans.utils.chat_positions import assert_positions_stable, candidate_positions
from acttrans.utils.hf import get_layer, load_model_and_tokenizer

_HERE = Path(__file__).resolve().parent
MANIFEST_PATH = _HERE / "data" / "caa_diff" / "pairs_manifest.json"
LLADA_8B = "GSAI-ML/LLaDA-8B-Instruct"
DEFAULT_LAYERS = [10, 12, 14, 16, 18, 20, 22]  # same middle-third sweep as the other diffusion-target scripts


def position_tags(n_pre: int, n_post: int):
    """Stable arm names for the n_pre + n_post candidates: 'pre0' is the very
    first template token, 'post1' is the token immediately before "Answer: "
    is appended (nearest the generation boundary), 'post2' the one before
    that, etc. — post-instruction tags count OUTWARD from the boundary, i.e.
    the paper's negative-indices-from-the-end convention."""
    pre_tags = [f"pre{i}" for i in range(n_pre)]
    post_tags = [f"post{k + 1}" for k in range(n_post)]
    return pre_tags, post_tags


@torch.no_grad()
def extract_one(model, layer_modules: dict, ids: torch.Tensor):
    """Returns {layer_idx: [seq, dim] hidden states (cpu, float32)} for one
    already-tokenized, unpadded sequence."""
    captured = {}

    def mk_hook(layer_idx):
        def hook(module, inp, out):
            act = out[0] if isinstance(out, tuple) else out
            captured[layer_idx] = act[0].detach().float().cpu()
        return hook

    handles = [m.register_forward_hook(mk_hook(l)) for l, m in layer_modules.items()]
    try:
        model(input_ids=ids)
    finally:
        for h in handles:
            h.remove()
    return captured


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=str(MANIFEST_PATH))
    ap.add_argument("--model", default=LLADA_8B)
    ap.add_argument("--layers", type=int, nargs="+", default=DEFAULT_LAYERS)
    ap.add_argument("--n-pre", type=int, default=4, help="Pre-instruction candidate count")
    ap.add_argument("--n-post", type=int, default=4, help="Post-instruction candidate count")
    ap.add_argument("--out-dir", default=str(_HERE / "data" / "caa_diff" / "activations"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--flush-every", type=int, default=200)
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    rows = manifest["rows"]
    n = len(rows)
    layers = sorted(args.layers)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    short = args.model.split("/")[-1]
    tmp_path = out_dir / f".{short}_papersel.partial.pt"

    print(f"Model: {args.model}  |  layers {layers}  |  device {args.device}  |  {n} pairs")
    model, tok = load_model_and_tokenizer(args.model, args.device)
    model.eval()

    print("  checking template-relative positions are stable across the manifest ...")
    assert_positions_stable(tok, [r["question"] for r in rows], args.n_pre, args.n_post)
    print("  ok — pre/post-instruction offsets resolve to the same token identity everywhere")

    layer_modules = {l: get_layer(model, l, "residual") for l in layers}
    pre_tags, post_tags = position_tags(args.n_pre, args.n_post)
    tags = pre_tags + post_tags

    dim = model.config.hidden_size
    keys = [(l, tag) for l in layers for tag in tags]
    if tmp_path.exists():
        state = torch.load(tmp_path, weights_only=False)
        done = state["n_done"]
        diffs = state["diffs"]
        print(f"  resuming from {done}/{n} (partial file found)")
    else:
        done = 0
        diffs = {k: torch.zeros(n, dim) for k in keys}

    for i in tqdm(range(done, n), desc="  pairs", initial=done, total=n):
        r = rows[i]
        template_text, template_len, pre_pos, post_pos = candidate_positions(
            tok, r["question"], args.n_pre, args.n_post
        )
        post_pos_desc = list(reversed(post_pos))  # post1 = nearest the boundary
        positions = list(zip(pre_tags, pre_pos)) + list(zip(post_tags, post_pos_desc))

        pos_text = template_text + f"Answer: {r['pos_choice']}"
        neg_text = template_text + f"Answer: {r['neg_choice']}"
        pos_ids = tok(pos_text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(args.device)
        neg_ids = tok(neg_text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(args.device)

        pos_h = extract_one(model, layer_modules, pos_ids)
        neg_h = extract_one(model, layer_modules, neg_ids)

        for l in layers:
            for tag, pos in positions:
                diffs[(l, tag)][i] = pos_h[l][pos] - neg_h[l][pos]

        if (i + 1) % args.flush_every == 0 or i + 1 == n:
            torch.save({"diffs": diffs, "n_done": i + 1}, tmp_path)

    behaviors = [r["behavior"] for r in rows]
    for (l, tag), t in diffs.items():
        out_path = out_dir / f"{short}_l{l}_{tag}_diff.pt"
        torch.save({
            "diffs": t, "behaviors": behaviors, "n_pairs": n,
            "manifest_hash": manifest["hash"], "pooling": tag,
            "model": args.model, "layer": l, "module": "residual",
        }, out_path)
        print(f"  wrote {out_path}")
    tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
