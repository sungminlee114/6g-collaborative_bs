"""Common training utilities with proper optimization and regularization."""
import torch
from torch.utils.data import DataLoader
from src.data.utils import nmse


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
                use_cosine=True):
    """Full local training loop with proper optimization.

    Features:
    - AdamW optimizer (decoupled weight decay)
    - Warmup + cosine annealing LR schedule
    - Gradient clipping
    - Early stopping with best model restore

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

    for epoch in range(epochs):
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

            if verbose and (epoch + 1) % 10 == 0:
                cur_lr = optimizer.param_groups[0]['lr']
                print(f"  Epoch {epoch+1}: train_nmse={t_loss:.6f}, "
                      f"val_nmse_db={v_db:.2f}, lr={cur_lr:.2e}")

            if wait >= patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch+1}")
                break
        else:
            val_losses.append(t_loss)
            if verbose and (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}: train_nmse={t_loss:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_epoch": best_epoch,
        "best_val": best_val,
    }
