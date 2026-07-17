"""Translate a .pt activation vector from source to target embedding space."""
import sys, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
from src.models.translator import load_translator

def translate(checkpoint_path, input_path, output_path=None, reverse=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_translator(checkpoint_path)
    model = model.to(device).eval()

    if reverse and not hasattr(model, "inverse"):
        raise ValueError(
            "--reverse requires a reversible translator (type = 'flow'); "
            "this checkpoint has no inverse mapping."
        )

    vec = torch.load(input_path, weights_only=True)
    if vec.dim() == 1:
        vec = vec.unsqueeze(0)  # [1, D]

    vec = vec.to(device).float()
    with torch.no_grad():
        translated = model.inverse(vec) if reverse else model(vec)

    translated = translated.cpu()
    if output_path:
        torch.save(translated, output_path)
        print(f"Saved translated vector to {output_path}")
    return translated

def main():
    parser = argparse.ArgumentParser(description="Translate a .pt activation vector to target model space")
    parser.add_argument("--checkpoint", required=True, help="Path to translator checkpoint")
    parser.add_argument("--input", required=True, help="Path to input .pt vector")
    parser.add_argument("--output", default=None, help="Path to save translated vector (optional)")
    parser.add_argument("--reverse", action="store_true", help="Map target->source using the translator's inverse (flow translators only)")
    args = parser.parse_args()

    result = translate(args.checkpoint, args.input, args.output, reverse=args.reverse)
    print(f"Input shape: loaded from {args.input}")
    print(f"Output shape: {result.shape}")
    if args.output is None:
        print(f"(Pass --output to save the result)")
    return result

if __name__ == "__main__":
    main()
