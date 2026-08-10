"""In-memory transport of direction vectors through a translator checkpoint.

Single implementation of the transport semantics shared by
translate_steering_vector.py (file-based CLI) and the comparison package
(in-memory analysis):

  - matches training preprocessing (normalize_activations)
  - linear translators transport bias-free by default: a CAA steering vector is
    a DIFFERENCE direction (mean_pos - mean_neg), and for an affine map
    y = W.x + b the bias cancels on a difference, T(a) - T(b) = W(a - b).
    ``apply_bias=True`` includes the affine bias and is only appropriate for a
    raw activation. apply_bias is LINEAR-ONLY: non-linear translators have no
    separable bias, so the flag has no effect on them.
  - norm modes control the translated vector's magnitude:
      "restore":    rescale the output to the source SV's original norm.
      "none":       leave the translator output as-is.
      "procrustes": multiply the (bias-free by default) linear output by the
                    fitted optimal scale s, giving the faithful floor transport
                    s*(W.sv). LINEAR-ONLY: requires a checkpoint produced by
                    fit_procrustes.py carrying config.translator.procrustes_scale.
"""

from typing import Optional

import torch
import torch.nn.functional as F

from .translator import build_translator, LinearTranslator

NORM_MODES = ("restore", "none", "procrustes")


class TranslatorRunner:
    """Loads a translator checkpoint once and transports direction vectors."""

    def __init__(self, checkpoint_path, device: str = "cpu"):
        self.checkpoint_path = checkpoint_path
        self.device = device
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.config = ckpt["config"]
        self.normalize_activations = (
            self.config.get("training", {}).get("normalize_activations", False)
        )
        self.module = build_translator(
            self.config, input_dim=ckpt["input_dim"], output_dim=ckpt["output_dim"]
        )
        self.module.load_state_dict(ckpt["state_dict"])
        self.module = self.module.to(device).eval()

    @property
    def procrustes_scale(self) -> Optional[float]:
        try:
            return float(self.config["translator"]["procrustes_scale"])
        except (KeyError, TypeError):
            return None

    def default_norm_mode(self) -> str:
        """Translator-aware default, matching ab_sweep's transport modes:
        linear/Procrustes checkpoints -> faithful floor transport, others ->
        restore the source SV's norm.

        Deliberately still an ``isinstance(LinearTranslator)`` check: an ANCHORED
        checkpoint also carries ``procrustes_scale``, but that scale is the LS scale
        of the anchor ``W`` alone, not of the composite ``W + gate*base``. Applying it
        to the composite output would be a wrong magnitude, so anchored checkpoints
        fall through to "restore" like every other trained translator."""
        if isinstance(self.module, LinearTranslator) and self.procrustes_scale is not None:
            return "procrustes"
        return "restore"

    @torch.no_grad()
    def transport(self, vec: torch.Tensor, norm_mode: str = "restore",
                  apply_bias: bool = False) -> torch.Tensor:
        """Transport a [D_src] direction vector; returns a float32 [D_tgt] cpu tensor."""
        if norm_mode not in NORM_MODES:
            raise ValueError(
                f"Unknown norm_mode {norm_mode!r}; expected one of {sorted(NORM_MODES)}."
            )

        x = vec.float().to(self.device)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        src_norm = x.norm(dim=-1, keepdim=True)

        # Match the preprocessing used during training.
        if self.normalize_activations:
            x = F.normalize(x, dim=-1)

        if not apply_bias and hasattr(self.module, "forward_direction"):
            # Bias-free direction transport (the bias cancels for a difference).
            # Any translator with a separable affine part advertises it through this
            # hook — LinearTranslator and AnchoredTranslator both do — so there is one
            # code path instead of a growing isinstance chain here.
            out = self.module.forward_direction(x)
        else:
            # Full map. For purely non-linear translators apply_bias has no effect (no
            # separable bias, and no forward_direction hook); with apply_bias=True this
            # includes the affine bias (only correct for a raw activation).
            out = self.module(x)
        out = out.squeeze(0).float().cpu()

        if norm_mode == "restore":
            out = F.normalize(out, dim=-1) * src_norm.squeeze().cpu()
        elif norm_mode == "procrustes":
            s = self.procrustes_scale
            if s is None:
                raise ValueError(
                    "norm_mode='procrustes' requires a checkpoint produced by "
                    "fit_procrustes.py, which stores config.translator.procrustes_scale; "
                    "this checkpoint has no such key. This mode is linear-only."
                )
            out = out * s
        # norm_mode == "none": leave the translator output as-is.
        return out
