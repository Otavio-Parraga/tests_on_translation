"""Stage-0 alignment diagnostics.

Measures how alignable two activation spaces are BEFORE training any translator,
so you can tell whether a linear/orthogonal map will suffice, whether a nonlinear
translator is needed, or whether one of the spaces is degenerate (mean-collapsed)
and translation is doomed regardless.

It computes, on index-paired source/target activations (same rows = same
sentences, dims may differ):

  1. Linear CKA  — Centered Kernel Alignment. Invariant to orthogonal transforms
     and isotropic scaling. ~1 means the two spaces are linearly the same up to
     rotation/scale; near 0 means no shared linear structure.
  2. Mutual k-NN — the Platonic-Representation-Hypothesis metric (Huh et al.):
     average overlap between each sample's k nearest neighbours in X-space and in
     Y-space. Captures shared *local* (possibly nonlinear) neighbourhood structure.
  3. Per-space activation norms + "mean-collapse" ratio ||E[a]|| / mean(||a||):
     ~1 means activations are almost all a single shared mean direction (a
     documented Gemma failure mode), which mean-free steering vectors sit
     off-distribution from.

Usage (either resolve paths from a config like train.py does, or pass explicit
activation caches):

    conda run -n acteng python diagnose_alignment.py --config config/fineweb.toml
    conda run -n acteng python diagnose_alignment.py \
        --source-activations data/fineweb/activations/Llama-3.2-1B-Instruct_l8.pt \
        --target-activations data/fineweb/activations/Llama-3.2-3B-Instruct_l8.pt
"""

import argparse
import sys
import tomllib
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from src.utils.paths import data_dir_of, resolve_activation_path


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Linear CKA between paired samples X [n, p1] and Y [n, p2].

    Columns are centered (mean over the n samples subtracted), then

        CKA = ||Y_c^T X_c||_F^2 / ( ||X_c^T X_c||_F * ||Y_c^T Y_c||_F )

    This is the feature-space (Gram-free) form: it never materialises the n x n
    kernel, so it stays cheap even for large n. It equals the HSIC form
    HSIC(K,L)/sqrt(HSIC(K,K) HSIC(L,L)) with K = X_c X_c^T, L = Y_c Y_c^T.
    Result in [0, 1]; invariant to orthogonal transforms and isotropic scaling.
    """
    X = X.to(torch.float64)
    Y = Y.to(torch.float64)
    Xc = X - X.mean(dim=0, keepdim=True)
    Yc = Y - Y.mean(dim=0, keepdim=True)
    # ||Y_c^T X_c||_F^2 = <Y_c^T Y_c, X_c^T X_c>_F  (a p2 x p1 matrix, small)
    cross = Yc.t() @ Xc  # [p2, p1]
    hsic_xy = (cross**2).sum()
    hsic_xx = (Xc.t() @ Xc).pow(2).sum().sqrt()
    hsic_yy = (Yc.t() @ Yc).pow(2).sum().sqrt()
    denom = hsic_xx * hsic_yy
    if denom <= 0:
        return float("nan")
    return float(hsic_xy / denom)


def rbf_cka(X: torch.Tensor, Y: torch.Tensor, sigma_frac: float = 0.5) -> float:
    """RBF (Gaussian-kernel) CKA — a nonlinear variant. Uses n x n Gram matrices,
    so only call it on a subsampled set. Bandwidth = sigma_frac * median pairwise
    distance, computed per space."""
    X = X.to(torch.float64)
    Y = Y.to(torch.float64)
    n = X.shape[0]

    def gram(A):
        d2 = torch.cdist(A, A) ** 2
        med = d2[d2 > 0].median()
        sigma2 = sigma_frac * med
        K = torch.exp(-d2 / (2 * sigma2 + 1e-12))
        # center
        H = torch.eye(n, dtype=A.dtype) - 1.0 / n
        return H @ K @ H

    Kc = gram(X)
    Lc = gram(Y)
    hsic_xy = (Kc * Lc).sum()
    hsic_xx = (Kc * Kc).sum().sqrt()
    hsic_yy = (Lc * Lc).sum().sqrt()
    denom = hsic_xx * hsic_yy
    if denom <= 0:
        return float("nan")
    return float(hsic_xy / denom)


def _knn_indices(A: torch.Tensor, k: int, metric: str) -> torch.Tensor:
    """Indices of the k nearest neighbours of each row (self excluded).

    Returns a [n, k] LongTensor. cosine -> largest cosine similarity;
    euclidean -> smallest L2 distance."""
    n = A.shape[0]
    A = A.to(torch.float32)
    if metric == "cosine":
        An = torch.nn.functional.normalize(A, dim=-1)
        sim = An @ An.t()  # [n, n], higher = closer
        sim.fill_diagonal_(float("-inf"))
        return sim.topk(k, dim=1).indices
    elif metric == "euclidean":
        dist = torch.cdist(A, A)  # [n, n], lower = closer
        dist.fill_diagonal_(float("inf"))
        return dist.topk(k, dim=1, largest=False).indices
    else:
        raise ValueError(f"unknown metric {metric!r}")


def mutual_knn(X: torch.Tensor, Y: torch.Tensor, k: int, metric: str) -> float:
    """Mutual k-NN overlap (Platonic Representation Hypothesis, Huh et al.).

    For each sample i, take its k nearest neighbours in X-space and in Y-space
    (self excluded); score_i = |NN_X(i) ∩ NN_Y(i)| / k. Return the mean over i.
    Result in [0, 1]; chance level for independent spaces is ≈ k / (n - 1)."""
    nn_x = _knn_indices(X, k, metric)  # [n, k]
    nn_y = _knn_indices(Y, k, metric)
    n = X.shape[0]
    # Build boolean membership masks and count intersection per row.
    mask_x = torch.zeros(n, n, dtype=torch.bool)
    mask_y = torch.zeros(n, n, dtype=torch.bool)
    rows = torch.arange(n).unsqueeze(1)
    mask_x[rows, nn_x] = True
    mask_y[rows, nn_y] = True
    inter = (mask_x & mask_y).sum(dim=1).to(torch.float64)  # [n]
    return float((inter / k).mean())


def space_stats(A: torch.Tensor) -> dict:
    """Per-space norm stats and the mean-collapse ratio ||E[a]|| / mean(||a||)."""
    A = A.to(torch.float64)
    norms = A.norm(dim=1)
    mean_vec = A.mean(dim=0)
    mean_norm = float(mean_vec.norm())
    mean_of_norms = float(norms.mean())
    ratio = mean_norm / mean_of_norms if mean_of_norms > 0 else float("nan")
    return {
        "dim": int(A.shape[1]),
        "mean_norm": mean_of_norms,
        "norm_of_mean": mean_norm,
        "mean_collapse_ratio": ratio,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def load_activations(path: Path) -> torch.Tensor:
    obj = torch.load(path, weights_only=False)
    return obj["activations"] if isinstance(obj, dict) else obj


def resolve_paths(args) -> tuple[Path, Path]:
    if args.source_activations and args.target_activations:
        return Path(args.source_activations), Path(args.target_activations)
    if not args.config:
        raise SystemExit(
            "Provide either --config or both --source-activations and --target-activations"
        )
    with open(args.config, "rb") as f:
        config = tomllib.load(f)
    data_dir = data_dir_of(config)
    src = (
        Path(args.source_activations)
        if args.source_activations
        else resolve_activation_path(data_dir, config["source_model"], "source")
    )
    tgt = (
        Path(args.target_activations)
        if args.target_activations
        else resolve_activation_path(data_dir, config["target_model"], "target")
    )
    return src, tgt


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None, help="TOML config (resolves paths like train.py)")
    p.add_argument("--source-activations", default=None)
    p.add_argument("--target-activations", default=None)
    p.add_argument("--k", type=int, default=10, help="neighbours for mutual k-NN (default 10)")
    p.add_argument("--max-samples", type=int, default=4096,
                   help="subsample rows for O(n^2) mutual-kNN / RBF-CKA (default 4096)")
    p.add_argument("--metric", choices=["cosine", "euclidean"], default="cosine",
                   help="distance for mutual k-NN (default cosine)")
    p.add_argument("--normalize", action="store_true", help="L2-normalize rows before all metrics")
    p.add_argument("--rbf-cka", action="store_true", help="also compute (nonlinear) RBF CKA")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    src_path, tgt_path = resolve_paths(args)
    for path in (src_path, tgt_path):
        if not Path(path).exists():
            raise SystemExit(f"Activation cache not found: {path}")

    print(f"Source activations: {src_path}")
    print(f"Target activations: {tgt_path}")
    X_full = load_activations(src_path).float()
    Y_full = load_activations(tgt_path).float()
    if X_full.shape[0] != Y_full.shape[0]:
        raise SystemExit(
            f"Row mismatch: source has {X_full.shape[0]}, target {Y_full.shape[0]} "
            "(activations must be paired by index)"
        )
    n_total = X_full.shape[0]
    print(f"Paired samples: n={n_total}, source dim={X_full.shape[1]}, target dim={Y_full.shape[1]}")

    if args.normalize:
        X_full = torch.nn.functional.normalize(X_full, dim=-1)
        Y_full = torch.nn.functional.normalize(Y_full, dim=-1)

    # Per-space stats on the full set (cheap, O(n * dim)).
    sx = space_stats(X_full)
    sy = space_stats(Y_full)

    # Linear CKA on the full set — feature-space form is cheap regardless of n.
    device = torch.device(args.device)
    cka_lin = linear_cka(X_full.to(device), Y_full.to(device))

    # Subsample a common random subset for the O(n^2) metrics.
    g = torch.Generator().manual_seed(args.seed)
    if args.max_samples and n_total > args.max_samples:
        idx = torch.randperm(n_total, generator=g)[: args.max_samples]
        X_sub, Y_sub = X_full[idx], Y_full[idx]
    else:
        X_sub, Y_sub = X_full, Y_full
    n_sub = X_sub.shape[0]
    k = min(args.k, n_sub - 1)

    mknn = mutual_knn(X_sub.to(device), Y_sub.to(device), k, args.metric)
    chance = k / (n_sub - 1)

    cka_rbf = None
    if args.rbf_cka:
        cka_rbf = rbf_cka(X_sub.to(device), Y_sub.to(device))

    # ---- report ----
    print("\n" + "=" * 62)
    print("STAGE-0 ALIGNMENT DIAGNOSTICS")
    print("=" * 62)
    print(f"device={device.type}  metric={args.metric}  normalize={args.normalize}")
    print(f"\n-- Global alignment (n={n_total}) --")
    print(f"  Linear CKA            : {cka_lin:.4f}")
    if cka_rbf is not None:
        print(f"  RBF CKA  (n_sub={n_sub}) : {cka_rbf:.4f}")
    print(f"\n-- Local structure (mutual k-NN, k={k}, n_sub={n_sub}) --")
    print(f"  Mutual k-NN overlap   : {mknn:.4f}   (chance ≈ {chance:.4f})")
    print("\n-- Per-space geometry --")
    for name, s in (("source", sx), ("target", sy)):
        print(f"  {name:6s} dim={s['dim']:5d}  mean||a||={s['mean_norm']:.4f}  "
              f"||E[a]||={s['norm_of_mean']:.4f}  mean-collapse={s['mean_collapse_ratio']:.4f}")

    # ---- interpretation heuristic ----
    print("\n-- Interpretation --")
    lines = []
    collapsed = [n for n, s in (("source", sx), ("target", sy)) if s["mean_collapse_ratio"] > 0.9]
    if collapsed:
        lines.append(
            f"  ! {', '.join(collapsed)} space is mean-collapsed (ratio > 0.9): activations "
            "are ~all a shared mean direction. Mean-center before training; mean-free "
            "steering vectors will otherwise sit off-distribution."
        )
    if cka_lin > 0.6:
        lines.append("  linear CKA > 0.6 -> a linear/orthogonal map likely suffices.")
    elif cka_lin < 0.3 and mknn > 4 * chance:
        lines.append(
            "  low linear CKA but high mutual k-NN -> shared *nonlinear* structure; "
            "a nonlinear translator (MLP) is likely needed."
        )
    elif cka_lin < 0.3 and mknn <= 4 * chance:
        lines.append(
            "  low linear CKA AND near-chance mutual k-NN -> little shared structure at all; "
            "translation is unlikely to work well (check extraction / pairing / collapse)."
        )
    else:
        lines.append(
            "  moderate linear CKA -> a linear map captures some structure; an MLP may "
            "recover the rest. Compare mutual k-NN to chance for nonlinear signal."
        )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
