"""Standalone DL-CE training script.

Trains DL-CE (ResNet denoiser) on independent drops data and saves checkpoints.
These checkpoints are then used by S1 profiling and S2 Pareto experiments.

Usage:
    python -m src.experiments.S1_ce_profiling.train_dl_ce --preset munich_elaa_s_1k_15g --gpu 0
    python -m src.experiments.S1_ce_profiling.train_dl_ce --all --gpus 0 1
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import get_results_dir
from src.tracker import Tracker
from src.ce_skip import ExperimentConfig
from src.ce_skip.ce_algorithms import DLCEEstimator
from src.dataset_operation.dataset import ChannelEstimationDataset
from src.dataset_operation.utils import nmse

PRIMARY_PRESETS = [
    "munich_elaa_s_1k_15g",
    "munich_mimo_15g",
]

CKPT_DIR = Path("assets/checkpoints/ce_skip")


def ckpt_path_for(cfg: ExperimentConfig) -> Path:
    """Standard checkpoint path for a DL-CE model."""
    return CKPT_DIR / f"dl_ce_{cfg.deployment}_{cfg.num_subcarriers}_{int(cfg.frequency/1e9)}g.pt"


def train_one(preset: str, gpu: int = 0, epochs: int = 100, batch_size: int = 128):
    """Train DL-CE for a single preset."""
    device = f"cuda:{gpu}"
    torch.cuda.set_device(device)
    cfg = ExperimentConfig.from_preset(preset)

    if not Path(cfg.data_dir).exists():
        print(f"  ⚠ Data not found: {cfg.data_dir}")
        return None

    print(f"  Data: {cfg.data_dir}")
    print(f"  Train BS: {cfg.train_bs_ids}, Val BS: {cfg.val_bs_ids}")

    train_ds = ChannelEstimationDataset(cfg.data_dir, bs_ids=cfg.train_bs_ids, snr_range_db=(0.0, 30.0))
    val_ds = ChannelEstimationDataset(cfg.data_dir, bs_ids=cfg.val_bs_ids, snr_range_db=(0.0, 30.0))

    print(f"  Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = DLCEEstimator(n_blocks=8, channels=64).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            h_noisy = batch["input"].to(device)
            h_true = batch["target"].to(device)
            h_hat = model(h_noisy)
            loss = nmse(h_hat, h_true)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                h_noisy = batch["input"].to(device)
                h_true = batch["target"].to(device)
                h_hat = model(h_noisy)
                val_loss += nmse(h_hat, h_true).item()
        val_loss /= len(val_loader)

        scheduler.step()

        train_db = 10 * np.log10(max(train_loss, 1e-30))
        val_db = 10 * np.log10(max(val_loss, 1e-30))

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            marker = " *"
        else:
            patience_counter += 1
            marker = ""

        if (epoch + 1) % 10 == 0 or patience_counter == 0:
            print(f"    [{epoch+1:3d}/{epochs}] train={train_db:.1f}dB val={val_db:.1f}dB{marker}")

        if patience_counter >= 25:
            print(f"    Early stop at epoch {epoch+1}")
            break

    # Save checkpoint
    best_val_db = 10 * np.log10(max(best_val, 1e-30))
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    path = ckpt_path_for(cfg)
    torch.save(best_state, path)
    print(f"  Saved: {path} (best val NMSE: {best_val_db:.1f} dB)")

    return {"best_val_nmse_db": best_val_db, "n_params": n_params, "path": str(path)}


def main():
    parser = argparse.ArgumentParser(description="Train DL-CE models")
    parser.add_argument("--preset", type=str, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    presets = PRIMARY_PRESETS if args.all else [args.preset or PRIMARY_PRESETS[0]]

    with Tracker(
        "S1/dl_ce_training",
        config={"presets": presets, "epochs": args.epochs},
        capture_output=True,
        purpose="Train DL-CE ResNet denoiser on independent drops data",
        variables={
            "independent": ["preset"],
            "dependent": ["val_nmse_db"],
            "controlled": [f"epochs={args.epochs}", "n_blocks=8", "channels=64"],
        },
        hypothesis="DL-CE achieves < -15dB NMSE on validation set",
        eval_criteria="Best validation NMSE in dB",
    ) as run:
        for preset in presets:
            print(f"\n{'═'*60}")
            print(f"  Training DL-CE: {preset}")
            print(f"{'═'*60}")
            result = train_one(preset, gpu=args.gpu, epochs=args.epochs, batch_size=args.batch_size)
            if result:
                run.log(**{f"{preset}_val_db": result["best_val_nmse_db"]})

        run.set_result(n_presets=len(presets))


if __name__ == "__main__":
    main()
