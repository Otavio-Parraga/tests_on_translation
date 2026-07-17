import datetime
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..data.split import preprocess_activations, split_paired_activations
from ..evaluation.metrics import compute_retrieval_metrics
from ..models.translator import save_translator
from ..utils.paths import best_translator_path, loss_tag, model_slug, resolve_losses
from .losses import make_loss_fn


class ActivationDataset(Dataset):
    def __init__(self, source, target):
        self.source = source
        self.target = target

    def __len__(self):
        return self.source.shape[0]

    def __getitem__(self, i):
        return self.source[i], self.target[i]


def train_translator(activations_dict, config, translator_model):
    tcfg = config["training"]
    train_ratio = tcfg.get("train_ratio", 0.8)
    seed = tcfg.get("seed", 42)
    batch_size = tcfg["batch_size"]
    lr = tcfg["lr"]
    weight_decay = tcfg.get("weight_decay", 1e-4)
    epochs = tcfg["epochs"]
    output_dir = Path(tcfg["output_dir"])
    normalize_activations = tcfg.get("normalize_activations", False)
    # Supports both `losses` (list) and legacy `loss` (single string).
    loss_names, loss_weights = resolve_losses(tcfg)
    temperature = tcfg.get("temperature", 0.07)
    grad_clip = tcfg.get("grad_clip", 0.0)
    lr_warmup_epochs = tcfg.get("lr_warmup_epochs", 0)
    # Reversible (flow) translators expose an `inverse`; when this weight is > 0
    # also supervise the target->source direction so one model learns both ways.
    inverse_loss_weight = tcfg.get("inverse_loss_weight", 0.0)
    # Isometry / norm-preservation penalty: E[(||F(x)|| - ||ref||)^2]. Keeps the
    # map near norm-preserving, countering the collapse/hubness pathology where
    # translated steering vectors lose their scale. 0 disables it.
    #   isometry_reference = "source": compare ||F(x)|| to the source-vector norm
    #     (the analysis default; dimension-agnostic, so 2048-d ||x|| vs 3072-d
    #     ||F(x)|| is intentional -- it preserves *norm*, not per-coordinate).
    #   isometry_reference = "target": compare ||F(x)|| to the target-vector norm.
    # Note: with normalize_activations=true the source norms are ~1, so the
    # "source" penalty just pushes outputs toward unit norm; it is most meaningful
    # without normalization (or with isometry_reference="target").
    isometry_loss_weight = tcfg.get("isometry_loss_weight", 0.0)
    isometry_reference = tcfg.get("isometry_reference", "source")
    # Early stopping on validation loss. patience = epochs to wait for an
    # improvement of at least `min_delta` before stopping; 0 disables it.
    early_stopping_patience = tcfg.get("early_stopping_patience", 0)
    early_stopping_min_delta = tcfg.get("early_stopping_min_delta", 0.0)

    source = activations_dict["source"].float()
    target = activations_dict["target"].float()
    source, target = preprocess_activations(source, target, config)

    train_src, train_tgt, val_src, val_tgt = split_paired_activations(source, target, config)
    n_train = train_src.shape[0]

    train_loader = DataLoader(
        ActivationDataset(train_src, train_tgt),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        ActivationDataset(val_src, val_tgt),
        batch_size=batch_size,
        shuffle=False,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = translator_model.to(device)

    _loss_fns = [make_loss_fn(name, temperature) for name in loss_names]

    def criterion(pred, tgt):
        return sum(w * fn(pred, tgt) for fn, w in zip(_loss_fns, loss_weights))

    train_inverse = inverse_loss_weight > 0 and hasattr(translator_model, "inverse")
    if inverse_loss_weight > 0 and not train_inverse:
        print(
            "inverse_loss_weight > 0 but the translator has no `inverse`; "
            "ignoring the reverse-direction loss."
        )

    def total_loss(model, src_batch, tgt_batch):
        # Compute pred once and share it with criterion and the isometry term.
        pred = model(src_batch)
        loss = criterion(pred, tgt_batch)
        # SAE-style translators expose an auxiliary sparsity penalty on their
        # bottleneck; add it when present (no-op for other model types).
        if hasattr(model, "sparsity_loss"):
            loss = loss + model.sparsity_loss()
        # Reversible translators also supervise target->source via model.inverse.
        if train_inverse:
            loss = loss + inverse_loss_weight * criterion(
                model.inverse(tgt_batch), src_batch
            )
        # Isometry / norm-preservation: push ||F(x)|| toward the reference norm.
        # Comparing a 3072-d output norm to a 2048-d source norm is intentional --
        # this preserves vector norm (scale), not per-coordinate values.
        if isometry_loss_weight > 0:
            ref = tgt_batch if isometry_reference == "target" else src_batch
            loss = loss + isometry_loss_weight * (
                (pred.norm(dim=-1) - ref.norm(dim=-1)) ** 2
            ).mean()
        return loss

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if lr_warmup_epochs > 0:
        warmup_sched = LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=lr_warmup_epochs
        )
        cosine_sched = CosineAnnealingLR(
            optimizer, T_max=max(epochs - lr_warmup_epochs, 1)
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_sched, cosine_sched],
            milestones=[lr_warmup_epochs],
        )
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    input_dim = source.shape[1]
    output_dim = target.shape[1]

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    epochs_since_improvement = 0
    global_step = 0

    src_cfg = config.get("source_model", {})
    tgt_cfg = config.get("target_model", {})
    tr_cfg = config.get("translator", {})

    # Experiment-named (pair + type + loss) so multiple translators coexist in one dir.
    checkpoint_path = best_translator_path(output_dir, config)

    # model_slug carries layer + pooling (_mean) so last-token and mean-pooled runs
    # land in distinct TensorBoard run dirs.
    src_tag = model_slug(src_cfg)
    tgt_tag = model_slug(tgt_cfg)
    tr_type = tr_cfg.get("type", "mlp")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{timestamp}__{src_tag}__{tgt_tag}__{tr_type}__{loss_tag(config)}"
    run_dir = output_dir / "tensorboard" / run_name
    writer = SummaryWriter(log_dir=str(run_dir))
    print(f"TensorBoard run: {run_dir}")

    tcfg_text = "\n".join(f"  {k} = {v}" for k, v in tcfg.items())
    config_text = (
        f"**source_model**\n  name = {src_cfg.get('name')}\n  layer = {src_cfg.get('layer')}\n\n"
        f"**target_model**\n  name = {tgt_cfg.get('name')}\n  layer = {tgt_cfg.get('layer')}\n\n"
        f"**translator**\n  type = {tr_cfg.get('type')}\n  hidden_dims = {tr_cfg.get('hidden_dims')}\n  dropout = {tr_cfg.get('dropout')}\n  activation = {tr_cfg.get('activation', 'gelu')}\n\n"
        f"**training**\n{tcfg_text}"
    )
    writer.add_text("config", config_text, global_step=0)

    epoch_bar = tqdm(range(1, epochs + 1), desc="Training", unit="epoch")
    for epoch in epoch_bar:
        model.train()
        total_train = 0.0
        batch_bar = tqdm(
            train_loader, desc=f"  Epoch {epoch}", leave=False, unit="batch"
        )
        for src_batch, tgt_batch in batch_bar:
            src_batch = src_batch.to(device, dtype=torch.float32)
            tgt_batch = tgt_batch.to(device, dtype=torch.float32)
            optimizer.zero_grad()
            loss = total_loss(model, src_batch, tgt_batch)
            # Skip any batch that produced a non-finite loss so one bad step
            # can't poison the weights (and propagate NaNs everywhere after).
            if not torch.isfinite(loss):
                writer.add_scalar("loss/nonfinite_skips", 1, global_step)
                global_step += 1
                continue
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            total_train += loss.item() * src_batch.size(0)
            batch_bar.set_postfix(loss=f"{loss.item():.4f}")
            writer.add_scalar("loss/train_step", loss.item(), global_step)
            global_step += 1
        train_loss = total_train / n_train
        scheduler.step()

        model.eval()
        total_val = 0.0
        n_val = val_src.shape[0]
        with torch.no_grad():
            for src_batch, tgt_batch in val_loader:
                src_batch = src_batch.to(device, dtype=torch.float32)
                tgt_batch = tgt_batch.to(device, dtype=torch.float32)
                loss = total_loss(model, src_batch, tgt_batch)
                total_val += loss.item() * src_batch.size(0)
        val_loss = total_val / n_val

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("lr", scheduler.get_last_lr()[0], epoch)

        metrics = compute_retrieval_metrics(
            model, val_src, val_tgt, ks=[1, 5], device=device, max_samples=2048
        )
        for k, dirs in metrics.items():
            writer.add_scalar(f"retrieval/acc@{k}/src2tgt", dirs["src2tgt"], epoch)
            writer.add_scalar(f"retrieval/acc@{k}/tgt2src", dirs["tgt2src"], epoch)

        epoch_bar.set_postfix(
            train=f"{train_loss:.4f}",
            val=f"{val_loss:.4f}",
            acc1=f"{metrics[1]['src2tgt']:.3f}",
            acc5=f"{metrics[5]['src2tgt']:.3f}",
        )

        if val_loss < best_val_loss - early_stopping_min_delta:
            best_val_loss = val_loss
            epochs_since_improvement = 0
            save_translator(model, checkpoint_path, config, input_dim, output_dim)
        else:
            # Still keep the best checkpoint up to date if val_loss improved by
            # less than min_delta (a real, if small, improvement).
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_translator(model, checkpoint_path, config, input_dim, output_dim)
            epochs_since_improvement += 1

        if (
            early_stopping_patience > 0
            and epochs_since_improvement >= early_stopping_patience
        ):
            print(
                f"\nEarly stopping at epoch {epoch}: no improvement > "
                f"{early_stopping_min_delta} in val loss for "
                f"{early_stopping_patience} epochs (best val loss = {best_val_loss:.4f})."
            )
            break

    final_metrics = compute_retrieval_metrics(
        model, val_src, val_tgt, ks=[1, 5], device=device
    )
    hparams = {
        "source_model": src_cfg.get("name", ""),
        "source_layer": src_cfg.get("layer", -1),
        "target_model": tgt_cfg.get("name", ""),
        "target_layer": tgt_cfg.get("layer", -1),
        "translator_type": tr_cfg.get("type", ""),
        "hidden_dims": str(tr_cfg.get("hidden_dims", [])),
        "dropout": tr_cfg.get("dropout", 0.0),
        "activation": tr_cfg.get("activation", "gelu"),
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "train_ratio": train_ratio,
        "seed": seed,
        "normalize_activations": normalize_activations,
        "losses": str(loss_names),
        "loss_weights": str(loss_weights),
        "temperature": temperature,
        "grad_clip": grad_clip,
        "lr_warmup_epochs": lr_warmup_epochs,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_delta": early_stopping_min_delta,
        "inverse_loss_weight": inverse_loss_weight,
        "isometry_loss_weight": isometry_loss_weight,
        "isometry_reference": isometry_reference,
    }
    hparam_metrics = {
        "hparam/val_loss": best_val_loss,
        "hparam/acc@1_src2tgt": final_metrics[1]["src2tgt"],
        "hparam/acc@1_tgt2src": final_metrics[1]["tgt2src"],
        "hparam/acc@5_src2tgt": final_metrics[5]["src2tgt"],
        "hparam/acc@5_tgt2src": final_metrics[5]["tgt2src"],
    }
    writer.add_hparams(hparams, hparam_metrics, run_name=".")
    writer.close()

    best_ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(best_ckpt["state_dict"])

    return model, train_losses, val_losses
