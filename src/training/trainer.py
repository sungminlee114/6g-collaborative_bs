"""Common training utilities."""
import torch
from src.data.utils import nmse


def train_epoch(model, loader, optimizer, device="cuda"):
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


def train_local(model, train_loader, val_loader=None, epochs=100,
                lr=1e-3, weight_decay=1e-5, patience=15, device="cuda",
                verbose=True):
    """Full local training loop with early stopping.

    Returns: dict with 'train_losses', 'val_losses', 'best_epoch', 'best_state'
    """
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay,
    )
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
        t_loss = train_epoch(model, train_loader, optimizer, device)
        train_losses.append(t_loss)

        if val_loader is not None:
            v_nmse, v_db = evaluate(model, val_loader, device)
            val_losses.append(v_nmse)
            scheduler.step(v_nmse)

            if v_nmse < best_val:
                best_val = v_nmse
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch
                wait = 0
            else:
                wait += 1

            if verbose and (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}: train_nmse={t_loss:.6f}, val_nmse_db={v_db:.2f}")

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
