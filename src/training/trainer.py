"""Common training utilities with proper optimization and regularization."""
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from src.data.utils import nmse


CHECKPOINTS_DIR = Path("checkpoints")


def save_checkpoint(model, name: str, meta: dict = None):
    """Save model checkpoint with metadata.

    Args:
        model: PyTorch model
        name: descriptive name, e.g. "phase0/reesnet_bs1_snr20"
        meta: optional dict with training info (best_val, best_epoch, etc.)
    """
    path = CHECKPOINTS_DIR / f"{name}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model_state_dict": model.state_dict()}
    if meta:
        payload["meta"] = meta
    torch.save(payload, path)
    print(f"  Saved: {path}")
    return path


def load_checkpoint(model, name: str, device="cuda"):
    """Load model checkpoint. Returns meta dict (or None).

    Args:
        model: PyTorch model (same architecture)
        name: checkpoint name used in save_checkpoint
    """
    path = CHECKPOINTS_DIR / f"{name}.pt"
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    print(f"  Loaded: {path}")
    return payload.get("meta")


def train_epoch(model, loader, optimizer, device="cuda", grad_clip=1.0):
    """Train one epoch, returns average NMSE loss."""
    model.train()
    total_loss = 0.0
    n = 0
    for batch in loader:
        x = batch["input"].to(device)
        y = batch["target"].to(device)

        optimizer.zero_grad()
        y_hat = model(x)
        loss = nmse(y_hat, y)
        loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device="cuda"):
    """Evaluate model, returns NMSE and NMSE_dB."""
    model.eval()
    total_nmse = 0.0
    n = 0
    for batch in loader:
        x = batch["input"].to(device)
        y = batch["target"].to(device)
        y_hat = model(x)
        total_nmse += nmse(y_hat, y).item() * x.size(0)
        n += x.size(0)
    avg_nmse = total_nmse / max(n, 1)
    return avg_nmse, 10 * torch.log10(torch.tensor(avg_nmse)).item()


@torch.no_grad()
def evaluate_per_snr(model, dataset_cls, data_dir, bs_ids, snr_list,
                     batch_size=64, device="cuda"):
    """Evaluate at specific SNR values. Returns dict {snr: nmse_db}."""
    results = {}
    for snr in snr_list:
        ds = dataset_cls(data_dir=data_dir, bs_ids=bs_ids, fixed_snr_db=snr)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
        _, nmse_val = evaluate(model, loader, device)
        results[snr] = nmse_val
    return results


def train_local(model, train_loader, val_loader=None, epochs=200,
                lr=1e-3, weight_decay=1e-4, patience=25, device="cuda",
                verbose=True, grad_clip=1.0, warmup_epochs=5,
                use_cosine=True, save_as: str = None):
    """Full local training loop with proper optimization.

    Features:
    - AdamW optimizer (decoupled weight decay)
    - Warmup + cosine annealing LR schedule
    - Gradient clipping
    - Early stopping with best model restore
    - Optional checkpoint saving via save_as

    Args:
        save_as: checkpoint name, e.g. "phase0/reesnet_bs1_snr20".
                 If given, saves best model to checkpoints/{save_as}.pt

    Returns: dict with 'train_losses', 'val_losses', 'best_epoch', 'best_val'
    """
    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

    # LR schedule: linear warmup + cosine annealing
    if use_cosine:
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
            return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=patience // 3,
        )

    best_val = float("inf")
    best_state = None
    best_epoch = 0
    wait = 0

    train_losses = []
    val_losses = []

    pbar = tqdm(range(epochs), desc="Training", disable=not verbose)
    for epoch in pbar:
        t_loss = train_epoch(model, train_loader, optimizer, device, grad_clip)
        train_losses.append(t_loss)

        if use_cosine:
            scheduler.step()

        if val_loader is not None:
            v_nmse, v_db = evaluate(model, val_loader, device)
            val_losses.append(v_nmse)

            if not use_cosine:
                scheduler.step(v_nmse)

            if v_nmse < best_val:
                best_val = v_nmse
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch
                wait = 0
            else:
                wait += 1

            best_db = 10 * torch.log10(torch.tensor(best_val)).item()
            cur_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix(train=f"{t_loss:.4f}", val_db=f"{v_db:.1f}",
                             best=f"{best_db:.1f}", lr=f"{cur_lr:.1e}", wait=wait)

            if wait >= patience:
                pbar.set_description(f"Early stop @ {epoch+1}")
                break
        else:
            val_losses.append(t_loss)
            t_db = 10 * torch.log10(torch.tensor(t_loss)).item()
            pbar.set_postfix(train_db=f"{t_db:.1f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    result = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_epoch": best_epoch,
        "best_val": best_val,
    }

    if save_as:
        save_checkpoint(model, save_as, meta={
            "best_epoch": best_epoch,
            "best_val": best_val,
            "best_val_db": 10 * torch.log10(torch.tensor(best_val)).item(),
            "epochs_run": len(train_losses),
        })

    return result
