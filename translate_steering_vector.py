"""
Translate a steering vector from source to target model space.

Expects the input path to follow activation_engineering folder structure:
  steering_vectors/{model}/{method}/{behavior}/{module}/layer_{idx}/sv.pt

The translated vector is saved under:
  {output_dir}/steering_vectors/{target_model}/{method}/{behavior}/{module}/layer_{target_layer}/sv.pt
  {output_dir}/steering_vectors/{target_model}/{method}/{behavior}/{module}/layer_{target_layer}/sv.json
"""
import argparse
import json
from pathlib import Path

import torch

from acttrans.models.transport import NORM_MODES, TranslatorRunner
from acttrans.utils.paths import parse_sv_path, sv_path

_PROJECT_ROOT = Path(__file__).resolve().parent


def translate_steering_vector(sv_path_in, checkpoint_path, output_dir=None, norm_mode="restore",
                              apply_bias=False):
    """Transport a CAA steering vector from source to target model space.

    The transport semantics (training-matched preprocessing, bias-free direction
    transport for linear translators, and the norm modes "restore" / "none" /
    "procrustes") live in ``acttrans.models.transport.TranslatorRunner``; this
    function adds the file plumbing: locating the source sv.pt, mirroring the
    activation_engineering folder structure on the output side, and carrying
    the sv.json metadata over.
    """
    sv_path_in = Path(sv_path_in)
    checkpoint_path = Path(checkpoint_path)

    if not sv_path_in.exists():
        raise FileNotFoundError(f"Steering vector not found: {sv_path_in}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Translator checkpoint not found: {checkpoint_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    runner = TranslatorRunner(checkpoint_path, device=device)
    config = runner.config
    target_model_name = config["target_model"]["name"]
    target_layer = config["target_model"]["layer"]
    target_module = config["target_model"]["module"]

    # Parse source path to recover folder structure components
    _, method, behavior, _, _ = parse_sv_path(sv_path_in)

    # Load vector and record original norm before any preprocessing
    vec = torch.load(sv_path_in, map_location=device, weights_only=True).float()
    src_norm = vec.norm(dim=-1).max().item()

    translated = runner.transport(vec, norm_mode=norm_mode, apply_bias=apply_bias)

    # Build output path mirroring activation_engineering structure, saved within this repo
    if output_dir is None:
        output_dir = _PROJECT_ROOT
    out_pt = sv_path(output_dir, target_model_name, method, behavior, target_module, target_layer)
    out_layer_dir = out_pt.parent
    out_layer_dir.mkdir(parents=True, exist_ok=True)

    # Save translated vector
    torch.save(translated, out_pt)

    # Save metadata: start from source sv.json if present, then update target fields
    src_json = sv_path_in.parent / "sv.json"
    meta = {}
    if src_json.exists():
        with open(src_json) as f:
            meta = json.load(f)

    meta.update(
        {
            "model": target_model_name,
            "tokenizer": target_model_name,
            "layer_idx": target_layer,
            "module_name": target_module,
            "translated_from": str(sv_path_in),
            "translator_checkpoint": str(checkpoint_path),
        }
    )

    out_json = out_layer_dir / "sv.json"
    with open(out_json, "w") as f:
        json.dump(meta, f, indent=2)

    norm_info = f"norm={translated.norm().item():.4f} (mode={norm_mode}"
    if norm_mode == "restore":
        norm_info += f", restored from source norm={src_norm:.4f}"
    elif norm_mode == "procrustes":
        norm_info += f", scale s={runner.procrustes_scale:.4f}"
    norm_info += ")"
    print(f"Translated vector : {out_pt}  {list(translated.shape)}  {norm_info}")
    print(f"Metadata          : {out_json}")
    return translated, out_pt


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "sv_path",
        help="Path to source sv.pt (inside an activation_engineering steering_vectors/ tree)",
    )
    parser.add_argument(
        "--checkpoint",
        default="outputs/best_translator__Llama-3.2-1B-Instruct_l8__gemma-3-1B-it_l8__mlp__cosine.pt",
        help="Translator checkpoint (default: the Llama-3.2-1B -> gemma-3-1B mlp/cosine translator)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Root output directory (default: this repo's root). steering_vectors/ is created inside.",
    )
    parser.add_argument(
        "--norm-mode",
        dest="norm_mode",
        choices=list(NORM_MODES),
        default="restore",
        help="How to set the translated vector's magnitude. "
             "'restore' (default): rescale to the source vector's original norm. "
             "'none': keep the raw translator output. "
             "'procrustes': faithful floor transport s*(W.sv), i.e. the linear output "
             "times the stored optimal scale (linear-only; requires a fit_procrustes.py checkpoint).",
    )
    parser.add_argument(
        "--apply-bias",
        dest="apply_bias",
        action="store_true",
        default=False,
        help="Add the affine bias when transporting with a linear/Procrustes translator. "
             "OFF by default because a steering vector is a difference direction, for which "
             "the bias cancels; leave off unless transporting a raw activation.",
    )
    args = parser.parse_args()
    translate_steering_vector(args.sv_path, args.checkpoint, args.output_dir, args.norm_mode,
                              args.apply_bias)


if __name__ == "__main__":
    main()
