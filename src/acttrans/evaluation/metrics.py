import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def retrieval_accuracy(sim: torch.Tensor, ks: list[int]) -> dict:
    """
    Compute top-k retrieval accuracy in both directions from a [N, N] cosine similarity matrix.

    Forward  (src→tgt): for each row i, is the true match (col i) in the top-k columns?
    Reverse  (tgt→src): for each col j, is the true match (row j) in the top-k rows?

    Returns {k: {"src2tgt": float, "tgt2src": float}, ...}
    """
    N = sim.shape[0]
    diag = torch.arange(N)
    results = {}
    for k in ks:
        fwd_hits = (
            (sim.topk(k, dim=1).indices == diag.unsqueeze(1)).any(dim=1).sum().item()
        )
        rev_hits = (
            (sim.topk(k, dim=0).indices == diag.unsqueeze(0)).any(dim=0).sum().item()
        )
        results[k] = {"src2tgt": fwd_hits / N, "tgt2src": rev_hits / N}
    return results


def compute_retrieval_metrics(
    model,
    val_source: torch.Tensor,
    val_target: torch.Tensor,
    ks: list[int],
    device: str,
    max_samples: int = None,
) -> dict:
    """
    Run inference on val_source, build the cosine-similarity matrix against val_target,
    and return retrieval_accuracy in both directions.

    max_samples: if set, randomly subsample this many pairs before computing the
    similarity matrix. Building an N×N matrix for large N is slow and memory-heavy
    (89k×89k ≈ 32 GB); use ~2048 for fast per-epoch estimates.
    """
    if max_samples is not None and val_source.shape[0] > max_samples:
        idx = torch.randperm(val_source.shape[0])[:max_samples]
        val_source = val_source[idx]
        val_target = val_target[idx]

    # Inputs are moved to `device` below, so the model must live there too. The
    # trainer already puts it on-device (no-op here); evaluate_from_checkpoint loads
    # it on CPU, so this is what makes standalone evaluation work on a GPU host.
    model = model.to(device)
    model.eval()
    loader = DataLoader(TensorDataset(val_source), batch_size=256, shuffle=False)
    preds = []
    with torch.no_grad():
        for (batch,) in loader:
            preds.append(model(batch.to(device, dtype=torch.float32)).cpu())
    predicted = torch.cat(preds, dim=0)

    pred_norm = F.normalize(predicted.float(), dim=-1)
    tgt_norm = F.normalize(val_target.float(), dim=-1)
    sim = pred_norm @ tgt_norm.t()

    return retrieval_accuracy(sim, ks)
