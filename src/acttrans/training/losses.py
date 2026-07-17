"""Coordinate/relational losses used to train the activation translators.

``make_loss_fn`` maps a loss name to a ``(pred, tgt) -> scalar`` callable. The
trainer builds one per configured loss and sums them (weighted); keeping them in
one module makes each loss independently testable on toy tensors.
"""

import torch
import torch.nn.functional as F


def make_loss_fn(loss_type, temperature):
    """Return a ``(pred, tgt) -> scalar`` loss for ``loss_type``.

    ``temperature`` is only used by ``info_nce``. Supported: 'mse', 'cosine',
    'info_nce', 'vsp'.
    """
    if loss_type == "mse":
        return lambda pred, tgt: F.mse_loss(pred, tgt)
    elif loss_type == "cosine":
        return lambda pred, tgt: F.cosine_embedding_loss(
            pred, tgt, torch.ones(pred.size(0), device=pred.device)
        )
    elif loss_type == "info_nce":

        def fn(pred, tgt):
            p = F.normalize(pred, dim=-1)
            t = F.normalize(tgt, dim=-1)
            logits = (p @ t.T) / temperature
            labels = torch.arange(len(p), device=p.device)
            return (
                F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
            ) / 2

        return fn
    elif loss_type == "vsp":
        # VSP / SP-KD (vec2vec "vector-space preservation", Similarity-Preserving
        # KD): preserve the batchwise pairwise-similarity (Gram) structure instead
        # of matching coordinates. This is dimension-agnostic (pred and tgt can live
        # in different-sized spaces) and directly fights the collapse-onto-the-mean
        # failure mode, since a collapsed map cannot reproduce the target Gram.
        #
        # G = X X^T is [B, B]; row-normalize (SP-KD form) then match in Frobenius:
        #   loss = || rownorm(G_pred) - rownorm(G_tgt) ||_F^2 / B^2
        def fn(pred, tgt):
            B = pred.size(0)
            if B < 2:
                # A single vector has no pairwise structure to preserve.
                return pred.new_zeros(())
            g_pred = pred @ pred.T
            g_tgt = tgt @ tgt.T
            g_pred = F.normalize(g_pred, p=2, dim=1)
            g_tgt = F.normalize(g_tgt, p=2, dim=1)
            return ((g_pred - g_tgt) ** 2).sum() / (B * B)

        return fn
    else:
        raise ValueError(
            f"Unknown loss type: {loss_type!r}. "
            "Expected 'mse', 'cosine', 'info_nce', or 'vsp'."
        )
