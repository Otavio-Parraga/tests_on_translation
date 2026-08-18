"""Extract per-pair (positive - negative) activation differences on the TARGET
model (GSAI-ML/LLaDA-8B-Instruct), over every item in build_caa_diff_manifest.py's
manifest, at every swept layer, for BOTH extraction arms — mean-pooled and
appended-[MASK] — in one model load.

TEXT CONSTRUCTION mirrors extract_caa_diff_source.py's recipe exactly, but through
LLaDA's OWN tokenizer/chat template rather than reusing Llama's literal formatted
string: the underlying (question, answer choice) pair is a model-agnostic
quantity, and every other per-model readout in this repo (ab_eval.py's run_ab_eval
in particular) already builds `tok.apply_chat_template(...) + "Answer: ("` against
whichever model is active rather than hard-coding one model's template text.

    prompt   = tok.apply_chat_template([{"role": "user", "content": question}],
                                        tokenize=False, add_generation_prompt=True)
    pos_text = prompt + f"Answer: {pos_choice}"
    neg_text = prompt + f"Answer: {neg_choice}"

TWO FORWARD PASSES PER ITEM, NOT ONE PER (LAYER x ARM). All 7 target layers
(--layers) are hooked simultaneously on a SINGLE model load, so one plain forward
captures the "mean" reading at every layer at once, and one appended-[MASK] forward
captures the "mask" reading at every layer at once — 2 forwards/text instead of
2 x 7 = 14. This is deliberately NOT extract_activations_sweep.py's per-layer
model-reload design: that design exists because the FineWeb sweep extracts ONE
pooling mode across ALL layers as separate --layers invocations sharing one config,
whereas this script always wants every layer AND both arms together, so hooking
once is strictly cheaper and the reload machinery (and the OOM shim it needed) buys
nothing here.

SINGLE SEQUENCE, NO PADDING, PER ITEM — matches extract_caa_diff_source.py's
choice for the same reason: mean-pooling a single unpadded sequence is a plain
mean(dim=1) over every position (all of them are real tokens), and the appended
[MASK] position is read via append_mask_token (the exact helper the completed A/B
sweep already validated), with attention_mask = all-ones so there is no padding
convention to get subtly wrong. It costs some throughput relative to batching, but
at ~4.9k items x 2 forwards this is minutes, not hours (see the launcher script's
docstring for the measured estimate) — not worth the padding-alignment risk for a
one-off preparation pass.

DIFFS ONLY ARE PERSISTED (not raw pos/neg activations): 7 layers x 2 arms x 4096
dims x ~4.9k items already totals ~1 GB just for the differences; keeping the raw
per-item pos/neg activations too would roughly triple that for no use the LOBO fit
script (or anything downstream) actually has. extract_caa_diff_source.py keeps
pos/neg on the source side because it's ~40x smaller (one layer, 2048-dim) and
directly useful for the sv.pt-fidelity check there.

Resumable: partial progress (all 7x2 running tensors + a row counter) flushes to
one temp file every --flush-every items and is picked back up on restart. Touches
the GPU (loads the ~16 GB LLaDA-8B-Instruct model) — this is why it's a separate
script from build_caa_diff_manifest.py, and why it must never run while another
LLaDA process holds the GPU (see the launcher script).

Usage:
    CUDA_VISIBLE_DEVICES=2 conda run -p /mnt/C-SSD/parraga/conda_envs/act_eng python \\
        extract_caa_diff_target.py --layers 10 12 14 16 18 20 22
"""
import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import torch
from tqdm import tqdm

from acttrans.utils.hf import append_mask_token, get_layer, load_model_and_tokenizer, resolve_mask_token_id

_HERE = Path(__file__).resolve().parent
MANIFEST_PATH = _HERE / "data" / "caa_diff" / "pairs_manifest.json"
LLADA_8B = "GSAI-ML/LLaDA-8B-Instruct"
DEFAULT_LAYERS = [10, 12, 14, 16, 18, 20, 22]  # same middle-third sweep as the FineWeb study


def build_pair_text(tok, question: str, choice: str) -> str:
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True
    )
    return prompt + f"Answer: {choice}"


@torch.no_grad()
def extract_one(model, tok, layer_modules: dict, mask_id: int, question: str, choice: str, device: str):
    """Returns (mean_by_layer, mask_by_layer), each {layer_idx: [dim] tensor}."""
    text = build_pair_text(tok, question, choice)
    ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)

    captured = {}

    def mk_hook(layer_idx):
        def hook(module, inp, out):
            act = out[0] if isinstance(out, tuple) else out
            captured[layer_idx] = act[0].detach().float()  # [seq, dim], cheap to slice below
        return hook

    handles = [m.register_forward_hook(mk_hook(l)) for l, m in layer_modules.items()]
    try:
        model(input_ids=ids)
        mean_by_layer = {l: h.mean(dim=0).cpu() for l, h in captured.items()}

        mask_ids, mask_attn, mask_pos = append_mask_token(
            {"input_ids": ids, "attention_mask": torch.ones_like(ids)},
            mask_id, tok.pad_token_id or 0, padding_side="right",
        )
        model(input_ids=mask_ids, attention_mask=mask_attn)
        pos = int(mask_pos[0])
        mask_by_layer = {l: h[pos].cpu() for l, h in captured.items()}
    finally:
        for h in handles:
            h.remove()
    return mean_by_layer, mask_by_layer


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=str(MANIFEST_PATH))
    ap.add_argument("--model", default=LLADA_8B)
    ap.add_argument("--layers", type=int, nargs="+", default=DEFAULT_LAYERS)
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
    tmp_path = out_dir / f".{short}_caa_diff.partial.pt"

    print(f"Model: {args.model}  |  layers {layers}  |  device {args.device}  |  {n} pairs")
    model, tok = load_model_and_tokenizer(args.model, args.device)
    model.eval()
    mask_id = resolve_mask_token_id(model, tok)
    layer_modules = {l: get_layer(model, l, "residual") for l in layers}

    dim = model.config.hidden_size
    keys = [(l, arm) for l in layers for arm in ("mean", "mask")]
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
        pos_mean, pos_mask = extract_one(model, tok, layer_modules, mask_id, r["question"], r["pos_choice"], args.device)
        neg_mean, neg_mask = extract_one(model, tok, layer_modules, mask_id, r["question"], r["neg_choice"], args.device)
        for l in layers:
            diffs[(l, "mean")][i] = pos_mean[l] - neg_mean[l]
            diffs[(l, "mask")][i] = pos_mask[l] - neg_mask[l]

        if (i + 1) % args.flush_every == 0 or i + 1 == n:
            torch.save({"diffs": diffs, "n_done": i + 1}, tmp_path)

    behaviors = [r["behavior"] for r in rows]
    for (l, arm), t in diffs.items():
        out_path = out_dir / f"{short}_l{l}_{arm}_diff.pt"
        torch.save({
            "diffs": t, "behaviors": behaviors, "n_pairs": n,
            "manifest_hash": manifest["hash"], "pooling": arm,
            "model": args.model, "layer": l, "module": "residual",
        }, out_path)
        print(f"  wrote {out_path}")
    tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
