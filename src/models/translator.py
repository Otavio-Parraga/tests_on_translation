import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

_ACTIVATIONS = {
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "silu": nn.SiLU,
}


class _ResidualBlock(nn.Module):
    def __init__(self, dim, dropout, act_cls):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            act_cls(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.block(x)


class MLPTranslator(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, dropout, activation, use_residual=False):
        super().__init__()
        act_cls = _ACTIVATIONS[activation]
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            if use_residual and in_dim == h:
                layers.append(_ResidualBlock(h, dropout, act_cls))
            else:
                layers.append(nn.Linear(in_dim, h))
                layers.append(nn.LayerNorm(h))
                layers.append(act_cls())
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class EncoderTranslator(nn.Module):
    def __init__(
        self, input_dim, output_dim, d_model=512, nhead=8, num_layers=4, dropout=0.1
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=False
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, output_dim)

    def forward(self, x):
        x = self.input_proj(x)
        x = x.unsqueeze(0)
        x = self.encoder(x)
        x = x.squeeze(0)
        return self.output_proj(x)


class SparseAutoencoderTranslator(nn.Module):
    """Translate src activations -> trg activations through an overcomplete sparse bottleneck.

        src (input_dim) --encode--> sparse latent (latent_dim) --decode--> trg (output_dim)

    Because both latent spaces pack many concepts into few dimensions (superposition),
    a wide (latent_dim > input_dim) bottleneck with a sparsity constraint is used to
    pull those entangled concepts apart into (ideally) mono-semantic features before
    re-projecting them into the target space. This is a cross-space SAE / "transcoder":
    unlike a vanilla SAE the input and output spaces differ, so there is a learnable
    input bias (b_pre) for centering the source and a separate output bias (b_dec).

    Two sparsity regimes:
      - "topk":  keep the k largest (post-ReLU) latent units per sample, zero the rest.
                 Sparsity is enforced architecturally — no auxiliary loss term needed.
      - "l1":    ReLU latent with an L1 penalty exposed via ``sparsity_loss()``, which
                 the trainer adds to the reconstruction loss when present.

    The decoder columns (dictionary atoms) are optionally unit-normalized so the L1
    penalty pushes on feature *activations* rather than being trivially gamed by
    shrinking latents and inflating decoder norms.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        latent_dim=4096,
        activation="topk",
        k=64,
        l1_coeff=1e-3,
        normalize_decoder=True,
    ):
        super().__init__()
        if activation not in ("topk", "l1"):
            raise ValueError(
                f"Unknown sae activation: {activation!r}. Expected 'topk' or 'l1'."
            )
        self.latent_dim = latent_dim
        self.activation = activation
        self.k = min(k, latent_dim)
        self.l1_coeff = l1_coeff
        self.normalize_decoder = normalize_decoder

        self.b_pre = nn.Parameter(torch.zeros(input_dim))
        self.encoder = nn.Linear(input_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, output_dim)

        # cache of the most recent latent code, used by sparsity_loss()
        self._latent = None

    def encode(self, x):
        z = self.encoder(x - self.b_pre)
        z = torch.relu(z)
        if self.activation == "topk":
            topv, topi = z.topk(self.k, dim=-1)
            z = torch.zeros_like(z).scatter_(-1, topi, topv)
        return z

    def decode(self, z):
        if self.normalize_decoder:
            w = nn.functional.normalize(self.decoder.weight, dim=0)
            return nn.functional.linear(z, w, self.decoder.bias)
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        self._latent = z
        return self.decode(z)

    def sparsity_loss(self):
        """L1 sparsity penalty on the latest latent code (0 for top-k, which is
        sparse by construction). Called by the trainer if the attribute exists."""
        if self.activation != "l1" or self._latent is None:
            return self.encoder.weight.new_zeros(())
        return self.l1_coeff * self._latent.abs().sum(dim=-1).mean()


class _ActNorm(nn.Module):
    """Per-feature affine transform (Glow ActNorm): y = scale * x + bias.

    Initialized data-dependently on the first forward pass so the output of each
    channel has ~zero mean / unit variance. Channels that are constant in the
    first batch (e.g. the zero-padding dimensions added to bridge a source/target
    dimension mismatch) are left as identity so they pass through untouched until
    later coupling layers route real signal into them.
    """

    def __init__(self, dim):
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.register_buffer("initialized", torch.tensor(False))

    def _data_init(self, x):
        with torch.no_grad():
            mean = x.mean(dim=0)
            std = x.std(dim=0)
            live = std > 1e-3
            self.bias.copy_(torch.where(live, -mean, torch.zeros_like(mean)))
            self.log_scale.copy_(
                torch.where(live, -torch.log(std + 1e-6), torch.zeros_like(std))
            )
            self.initialized.fill_(True)

    def forward(self, x):
        if self.training and not self.initialized:
            self._data_init(x)
        # clamp keeps the per-channel gain bounded so stacked blocks can't
        # compound into a runaway output scale
        return torch.exp(self.log_scale.clamp(-5.0, 5.0)) * x + self.bias

    def inverse(self, y):
        return (y - self.bias) * torch.exp(-self.log_scale.clamp(-5.0, 5.0))


class _Inv1x1(nn.Module):
    """Invertible linear mixing layer with LU parameterization (Glow).

    W = P (L + I) (U + diag(s)) is invertible by construction: P is a fixed
    permutation, L is strictly-lower / U strictly-upper triangular, and the
    diagonal s is kept away from zero. This mixes information across all
    channels between coupling layers without ever needing a matrix inverse on
    the forward pass.
    """

    def __init__(self, dim):
        super().__init__()
        w = torch.linalg.qr(torch.randn(dim, dim))[0]  # random orthogonal init
        p, l, u = torch.linalg.lu(w)
        s = torch.diagonal(u).clone()
        self.register_buffer("P", p)
        self.register_buffer("lower_mask", torch.tril(torch.ones(dim, dim), -1))
        self.register_buffer("upper_mask", torch.triu(torch.ones(dim, dim), 1))
        self.register_buffer("eye", torch.eye(dim))
        self.L = nn.Parameter(l)
        self.U = nn.Parameter(torch.triu(u, 1))
        self.sign_s = nn.Parameter(torch.sign(s), requires_grad=False)
        self.log_s = nn.Parameter(torch.log(torch.abs(s) + 1e-6))

    def _weight(self):
        L = self.L * self.lower_mask + self.eye
        U = self.U * self.upper_mask + torch.diag(self.sign_s * torch.exp(self.log_s))
        return self.P @ L @ U

    def forward(self, x):
        return F.linear(x, self._weight())

    def inverse(self, y):
        W = self._weight()
        return torch.linalg.solve(W, y.t()).t()


class _Permute(nn.Module):
    """Fixed random permutation of channels (RealNVP-style mixing).

    A cheaper, far more stable alternative to the learned ``_Inv1x1``: it mixes
    information across the coupling split without any learned matrix, so it can't
    become ill-conditioned, and its inverse is an exact O(D) gather (no linear
    solve). Preferred over ``_Inv1x1`` at the large widths used here.
    """

    def __init__(self, dim):
        super().__init__()
        perm = torch.randperm(dim)
        self.register_buffer("perm", perm)
        self.register_buffer("inv_perm", torch.argsort(perm))

    def forward(self, x):
        return x[:, self.perm]

    def inverse(self, y):
        return y[:, self.inv_perm]


class _AffineCoupling(nn.Module):
    """Affine coupling layer (RealNVP). The first half conditions an affine
    transform of the second half, so the map is invertible regardless of what
    the conditioner net computes. The conditioner's last layer is zero-init, so
    the layer starts as the identity (a stable starting point for training)."""

    def __init__(self, dim, hidden, act_cls):
        super().__init__()
        self.d1 = dim // 2
        self.d2 = dim - self.d1
        self.net = nn.Sequential(
            nn.Linear(self.d1, hidden),
            act_cls(),
            nn.Linear(hidden, hidden),
            act_cls(),
            nn.Linear(hidden, 2 * self.d2),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def _params(self, x1):
        h = self.net(x1)
        log_s, t = h.chunk(2, dim=-1)
        log_s = torch.tanh(log_s)  # bound the scale for stability
        return log_s, t

    def forward(self, x):
        x1, x2 = x[:, : self.d1], x[:, self.d1 :]
        log_s, t = self._params(x1)
        y2 = x2 * torch.exp(log_s) + t
        return torch.cat([x1, y2], dim=-1)

    def inverse(self, y):
        y1, y2 = y[:, : self.d1], y[:, self.d1 :]
        log_s, t = self._params(y1)
        x2 = (y2 - t) * torch.exp(-log_s)
        return torch.cat([y1, x2], dim=-1)


class FlowTranslator(nn.Module):
    """Reversible (normalizing-flow) translator between two activation spaces.

    The flow is a stack of [ActNorm -> mixing -> AffineCoupling] blocks, each of
    which has a closed-form inverse, so the whole network is exactly invertible:
    ``inverse(forward(x)) == x``. A single trained model therefore gives both the
    source->target translation (``forward``) and the target->source translation
    (``inverse``) for free.

    The cross-channel ``mixing`` layer is either a fixed random permutation
    (default — stable and exactly invertible in O(D)) or a learned LU-factored
    1x1 linear (Glow-style). At the large widths used here the learned variant is
    prone to becoming ill-conditioned and blowing up training, so permutation is
    the recommended default.

    Dimension bridging: a strict bijection only exists between spaces of equal
    dimension, so the flow operates in ``D = max(input_dim, output_dim)``. The
    shorter side is zero-padded up to ``D`` on the way in, and the extra
    "auxiliary" dimensions are dropped on the way out. Invertibility holds in the
    padded D-dimensional space; the truncated auxiliary dims are the (inherent)
    price of mapping between differently-sized spaces.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        num_blocks=8,
        coupling_hidden=1024,
        activation="gelu",
        mixing="permutation",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dim = max(input_dim, output_dim)
        act_cls = _ACTIVATIONS[activation]
        if mixing not in ("permutation", "inv1x1"):
            raise ValueError(
                f"Unknown mixing: {mixing!r}. Expected 'permutation' or 'inv1x1'."
            )
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            mix = _Permute(self.dim) if mixing == "permutation" else _Inv1x1(self.dim)
            self.blocks.append(
                nn.ModuleList(
                    [
                        _ActNorm(self.dim),
                        mix,
                        _AffineCoupling(self.dim, coupling_hidden, act_cls),
                    ]
                )
            )

    @staticmethod
    def _pad(x, dim):
        if x.shape[-1] == dim:
            return x
        return F.pad(x, (0, dim - x.shape[-1]))

    def forward(self, x):
        z = self._pad(x, self.dim)
        for actnorm, mix, coupling in self.blocks:
            z = actnorm(z)
            z = mix(z)
            z = coupling(z)
        return z[:, : self.output_dim]

    def inverse(self, y):
        z = self._pad(y, self.dim)
        for actnorm, mix, coupling in reversed(self.blocks):
            z = coupling.inverse(z)
            z = mix.inverse(z)
            z = actnorm.inverse(z)
        return z[:, : self.input_dim]


class LinearTranslator(nn.Module):
    """A plain affine map ``y = x @ W.T + b`` between the two activation spaces.

    This is the closed-form baseline: fitted (not gradient-trained) by orthogonal
    Procrustes (see ``fit_orthogonal_procrustes``). An orthogonal / semi-orthogonal
    ``W`` preserves norms and inner products exactly, so it structurally avoids the
    scale blow-up and mean-collapse pathologies the learned translators suffer from,
    and is the strong "floor" any nonlinear translator has to beat.

    The weights are held in an ``nn.Linear`` so ``state_dict``/``load_state_dict``
    round-trip cleanly through ``save_translator``/``load_translator``.
    """

    def __init__(self, input_dim, output_dim, bias=True):
        super().__init__()
        self.W = nn.Linear(input_dim, output_dim, bias=bias)

    def set_weights(self, W, b=None):
        """Load a fitted weight matrix ``W`` [out, in] (and optional bias ``b`` [out])."""
        with torch.no_grad():
            self.W.weight.copy_(W.to(self.W.weight.dtype))
            if b is not None and self.W.bias is not None:
                self.W.bias.copy_(b.to(self.W.bias.dtype))
        return self

    def forward(self, x):
        return self.W(x)


def fit_orthogonal_procrustes(X, Y, center=True, bias=True, whiten=False):
    """Closed-form orthogonal Procrustes fit of an affine map ``Y ≈ X @ W.T + b``.

    Solves ``min_W || W X^T - Y^T ||_F`` subject to ``W`` having orthonormal rows
    or columns (whichever is possible given the shapes):

        M = Y_c^T @ X_c          # [out, in]
        U, S, Vt = SVD(M)        # thin SVD
        W = U @ Vt               # [out, in]

    Orthogonality note (dimension mismatch): when ``out == in`` this ``W`` is a
    true orthogonal matrix (``W W^T = W^T W = I``). Here ``out=3072 > in=2048``, so
    a matrix cannot have both orthonormal rows AND columns — with the thin SVD
    (``full_matrices=False``) ``U`` is [out, in] with orthonormal columns and ``Vt``
    is [in, in] orthogonal, so ``W = U @ Vt`` is [out, in] with **orthonormal
    columns** (``W^T W = I_in``). It is a semi-orthogonal / Stiefel-manifold map:
    injective and norm-preserving on the input subspace (``||W v|| = ||v||`` for all
    ``v``), which is exactly the property we want to preserve steering-vector scale.

    Args:
        X: source activations, [N, in].
        Y: target activations, [N, out].
        center: mean-center X and Y before fitting (recommended — the fit then
            learns the off-mean subspace and the bias captures the mean shift,
            sidestepping the project's "collapse onto the mean" failure mode).
        bias: if True, return a bias ``b``; with centering ``b = mean_Y - W @ mean_X``,
            otherwise ``b`` is zeros. If False, ``b`` is None.
        whiten: if True, fit a WHITENED (anisotropy-aware) Procrustes map instead of
            the plain orthogonal one. Plain orthogonal Procrustes implicitly assumes
            both spaces are isotropic, but transformer residual streams are strongly
            anisotropic; whitening each space, fitting the orthogonal map on the
            whitened coordinates, and folding the whitening back into ``W`` typically
            aligns anisotropic spaces better. TRADEOFF: the returned ``W`` is then NO
            LONGER orthonormal — it does not preserve norms exactly — trading exact
            norm-preservation for a better anisotropy-aware fit. Centering is implied
            and forced on when ``whiten=True`` (a ``center=False`` caller is
            overridden, since the covariances are only meaningful about the mean).

    Returns:
        (W, b): W is float32 [out, in]; b is float32 [out] or None.
    """
    # float64 for numerical stability of the SVD/eigendecomposition, cast back at end.
    X = X.to(torch.float64)
    Y = Y.to(torch.float64)

    # Whitening is only meaningful about the mean, so centering is forced on when
    # whiten=True even if the caller passed center=False.
    do_center = center or whiten

    if do_center:
        mean_X = X.mean(dim=0)
        mean_Y = Y.mean(dim=0)
        Xc = X - mean_X
        Yc = Y - mean_Y
    else:
        mean_X = torch.zeros(X.shape[1], dtype=torch.float64)
        mean_Y = torch.zeros(Y.shape[1], dtype=torch.float64)
        Xc, Yc = X, Y

    if whiten:
        N = Xc.shape[0]
        Cx = (Xc.t() @ Xc) / N  # [in, in]
        Cy = (Yc.t() @ Yc) / N  # [out, out]

        def _sym_powers(C):
            """Ridge-regularized symmetric C^{1/2} and C^{-1/2} via eigh."""
            evals, evecs = torch.linalg.eigh(C)
            eps = 1e-5 * evals.mean()  # ridge for stability of the inverse-sqrt
            evals = torch.clamp(evals, min=0.0) + eps
            sqrt = evecs @ torch.diag(evals.sqrt()) @ evecs.t()
            inv_sqrt = evecs @ torch.diag(evals.rsqrt()) @ evecs.t()
            return sqrt, inv_sqrt

        _, Wx = _sym_powers(Cx)  # Wx = Cx^{-1/2}  [in, in]
        Sy, Wy = _sym_powers(Cy)  # Sy = Cy^{1/2}, Wy = Cy^{-1/2}  [out, out]

        Xt = Xc @ Wx  # whitened source  [N, in]
        Yt = Yc @ Wy  # whitened target  [N, out]
        M = Yt.t() @ Xt  # [out, in]
        U, S, Vt = torch.linalg.svd(M, full_matrices=False)
        Q = U @ Vt  # orthogonal map in whitened coords  [out, in]
        # Fold whitening back so W acts on raw (centered) coords:
        # Yc ≈ Xc Wx Q^T Wy^{-1} = Xc Wx Q^T Sy  =>  W = Sy Q Wx.
        W = Sy @ Q @ Wx  # [out, in], no longer orthonormal
    else:
        M = Yc.t() @ Xc  # [out, in]
        U, S, Vt = torch.linalg.svd(M, full_matrices=False)
        W = U @ Vt  # [out, in], orthonormal columns when out > in

    if bias:
        b = mean_Y - W @ mean_X  # captures the mean shift the centered W ignores
        b = b.to(torch.float32)
    else:
        b = None

    return W.to(torch.float32), b


def procrustes_scale(X, Y, W, center=True):
    """Optimal least-squares scale ``s`` for the scaled variant ``Y ≈ s (W X^T) + b``.

    ``s = <W X^T, Y^T> / ||W X^T||^2`` (over the centered data if ``center``). The
    orthogonal map alone preserves scale exactly; ``s`` is offered only for the
    variant that additionally rescales to best match target magnitude. This is the
    LS scalar for whatever ``W`` is passed, so it stays valid for a whitened ``W``.
    """
    X = X.to(torch.float64)
    Y = Y.to(torch.float64)
    W = W.to(torch.float64)
    if center:
        X = X - X.mean(dim=0)
        Y = Y - Y.mean(dim=0)
    WX = X @ W.t()  # [N, out]
    num = (WX * Y).sum()
    den = (WX * WX).sum()
    return float(num / den)


def build_translator(config, input_dim=None, output_dim=None):
    if input_dim is None:
        input_dim = config["source_model"]["hidden_dim"]
    if output_dim is None:
        output_dim = config["target_model"]["hidden_dim"]
    tcfg = config["translator"]
    translator_type = tcfg["type"]
    if translator_type == "mlp":
        return MLPTranslator(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=tcfg.get("hidden_dims", [2048, 2048]),
            dropout=tcfg.get("dropout", 0.1),
            activation=tcfg.get("activation", "gelu"),
            use_residual=tcfg.get("use_residual", False),
        )
    elif translator_type == "encoder":
        return EncoderTranslator(
            input_dim=input_dim,
            output_dim=output_dim,
            d_model=tcfg.get("d_model", 512),
            nhead=tcfg.get("nhead", 8),
            num_layers=tcfg.get("num_layers", 4),
            dropout=tcfg.get("dropout", 0.1),
        )
    elif translator_type == "sae":
        return SparseAutoencoderTranslator(
            input_dim=input_dim,
            output_dim=output_dim,
            latent_dim=tcfg.get("latent_dim", 4096),
            activation=tcfg.get("sae_activation", "topk"),
            k=tcfg.get("k", 64),
            l1_coeff=tcfg.get("l1_coeff", 1e-3),
            normalize_decoder=tcfg.get("normalize_decoder", True),
        )
    elif translator_type == "linear":
        return LinearTranslator(
            input_dim=input_dim,
            output_dim=output_dim,
            bias=tcfg.get("bias", True),
        )
    elif translator_type == "flow":
        return FlowTranslator(
            input_dim=input_dim,
            output_dim=output_dim,
            num_blocks=tcfg.get("num_blocks", 8),
            coupling_hidden=tcfg.get("coupling_hidden", 1024),
            activation=tcfg.get("activation", "gelu"),
            mixing=tcfg.get("mixing", "permutation"),
        )
    else:
        raise ValueError(f"Unknown translator type: {translator_type}")


def save_translator(model, path, config, input_dim, output_dim):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": config,
            "input_dim": input_dim,
            "output_dim": output_dim,
        },
        path,
    )


def load_translator(path, config=None, input_dim=None, output_dim=None):
    checkpoint = torch.load(path, map_location="cpu")
    cfg = config if config is not None else checkpoint["config"]
    in_dim = input_dim if input_dim is not None else checkpoint["input_dim"]
    out_dim = output_dim if output_dim is not None else checkpoint["output_dim"]
    model = build_translator(cfg, input_dim=in_dim, output_dim=out_dim)
    model.load_state_dict(checkpoint["state_dict"])
    return model
