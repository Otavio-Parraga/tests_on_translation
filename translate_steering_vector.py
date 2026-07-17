"""
Translate a steering vector from source to target model space.

Expects the input path to follow activation_engineering folder structure:
  steering_vectors/{model}/{method}/{behavior}/{module}/layer_{idx}/sv.pt

The translated vector is saved under:
  {output_dir}/steering_vectors/{target_model}/{method}/{behavior}/{module}/layer_{target_layer}/sv.pt
  {output_dir}/steering_vectors/{target_model}/{method}/{behavior}/{module}/layer_{target_layer}/sv.json
"""
import sys, argparse, json
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT))

import torch
import torch.nn.functional as F
from src.models.translator import build_translator, LinearTranslator


def _parse_sv_path(sv_path: Path):
    """Extract (model_slug, method, behavior, module, layer_idx) from an sv.pt path."""
    parts = sv_path.parts
    try:
        root_idx = parts.index("steering_vectors")
    except ValueError:
        raise ValueError(f"'steering_vectors' not found in path: {sv_path}")

    rel = parts[root_idx + 1:]
    # Expected: {model}/{method}/{behavior}/{module}/layer_{idx}/sv.pt
    if len(rel) < 6:
        raise ValueError(
            "Unexpected path structure. Expected:\n"
            "  steering_vectors/{model}/{method}/{behavior}/{module}/layer_{idx}/sv.pt\n"
            f"Got: {sv_path}"
        )

    model_slug, method, behavior, module, layer_part = rel[0], rel[1], rel[2], rel[3], rel[4]

    if not layer_part.startswith("layer_"):
        raise ValueError(f"Expected 'layer_{{idx}}', got: {layer_part!r}")
    layer_idx = int(layer_part.split("_", 1)[1])

    return model_slug, method, behavior, module, layer_idx


def _model_slug(model_name: str) -> str:
    return model_name.replace("/", "_")


def translate_steering_vector(sv_path, checkpoint_path, output_dir=None, norm_mode="restore",
                              apply_bias=False):
    """Transport a CAA steering vector from source to target model space.

    A CAA steering vector is a DIFFERENCE direction (mean_pos - mean_neg), not an
    activation. For a linear/Procrustes translator the map is affine (y = W.x + b),
    and the bias cancels for a difference: T(a) - T(b) = W(a - b). So by default
    (apply_bias=False) a LinearTranslator transports using the linear part only
    (vec @ W.T, no bias) -- the correct behavior for a direction. Setting
    apply_bias=True includes the affine bias and is only appropriate when
    transporting a raw activation. apply_bias is LINEAR-ONLY: for non-linear
    translators (mlp/encoder/flow/sae) there is no separable bias and the flag
    has no effect.

    norm_mode controls how the translated vector's magnitude is set, one of:
      "restore":    rescale the output to the source SV's original norm
                    (F.normalize(translated) * src_norm).
      "none":       leave the translator output as-is.
      "procrustes": multiply the (bias-free by default) linear output by the
                    stored optimal scale s, giving the faithful floor transport
                    s*(W.sv). LINEAR-ONLY: requires a checkpoint produced by
                    fit_procrustes.py that carries config.translator.procrustes_scale.
    """
    valid_norm_modes = {"restore", "none", "procrustes"}
    if norm_mode not in valid_norm_modes:
        raise ValueError(
            f"Unknown norm_mode {norm_mode!r}; expected one of {sorted(valid_norm_modes)}."
        )
    sv_path = Path(sv_path)
    checkpoint_path = Path(checkpoint_path)

    if not sv_path.exists():
        raise FileNotFoundError(f"Steering vector not found: {sv_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Translator checkpoint not found: {checkpoint_path}")

    # Load checkpoint once to get config + build model
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    target_model_name = config["target_model"]["name"]
    target_layer = config["target_model"]["layer"]
    target_module = config["target_model"]["module"]
    normalize_activations = config.get("training", {}).get("normalize_activations", False)

    translator = build_translator(config, input_dim=ckpt["input_dim"], output_dim=ckpt["output_dim"])
    translator.load_state_dict(ckpt["state_dict"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    translator = translator.to(device).eval()

    # Parse source path to recover folder structure components
    _, method, behavior, _, _ = _parse_sv_path(sv_path)

    # Load vector and record original norm before any preprocessing
    vec = torch.load(sv_path, map_location=device, weights_only=True).float()
    if vec.dim() == 1:
        vec = vec.unsqueeze(0)  # [1, D_src]

    src_norm = vec.norm(dim=-1, keepdim=True)  # [1, 1]

    # Match the preprocessing used during training
    if normalize_activations:
        vec = F.normalize(vec, dim=-1)

    with torch.no_grad():
        if isinstance(translator, LinearTranslator) and not apply_bias:
            # Direction transport: use the linear part only. The affine bias cancels
            # for a difference direction, so adding it would inject a spurious shift.
            out = vec @ translator.W.weight.T
        else:
            # Full map. For non-linear translators apply_bias has no effect (no
            # separable bias); for a LinearTranslator with apply_bias=True this
            # includes the affine bias (only correct for a raw activation).
            out = translator(vec)
        translated = out.squeeze(0).cpu()  # [D_tgt], unit norm if normalize_activations

    # Set the translated vector's magnitude according to norm_mode.
    s = None
    if norm_mode == "restore":
        # Rescale to the source steering vector's original norm.
        translated = F.normalize(translated, dim=-1) * src_norm.squeeze().cpu()
    elif norm_mode == "procrustes":
        # Faithful floor transport s*(W.sv). Requires the stored Procrustes scale.
        try:
            s = float(config["translator"]["procrustes_scale"])
        except KeyError:
            raise ValueError(
                "norm_mode='procrustes' requires a checkpoint produced by "
                "fit_procrustes.py, which stores config.translator.procrustes_scale; "
                "this checkpoint has no such key. This mode is linear-only."
            )
        translated = translated * s
    # norm_mode == "none": leave the translator output as-is.

    # Build output path mirroring activation_engineering structure, saved within this repo
    if output_dir is None:
        output_dir = _PROJECT_ROOT
    out_layer_dir = (
        Path(output_dir)
        / "steering_vectors"
        / _model_slug(target_model_name)
        / method
        / behavior
        / target_module
        / f"layer_{target_layer}"
    )
    out_layer_dir.mkdir(parents=True, exist_ok=True)

    # Save translated vector
    out_pt = out_layer_dir / "sv.pt"
    torch.save(translated, out_pt)

    # Save metadata: start from source sv.json if present, then update target fields
    src_json = sv_path.parent / "sv.json"
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
            "translated_from": str(sv_path),
            "translator_checkpoint": str(checkpoint_path),
        }
    )

    out_json = out_layer_dir / "sv.json"
    with open(out_json, "w") as f:
        json.dump(meta, f, indent=2)

    norm_info = f"norm={translated.norm().item():.4f} (mode={norm_mode}"
    if norm_mode == "restore":
        norm_info += f", restored from source norm={src_norm.item():.4f}"
    elif norm_mode == "procrustes":
        norm_info += f", scale s={s:.4f}"
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
        choices=["restore", "none", "procrustes"],
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
