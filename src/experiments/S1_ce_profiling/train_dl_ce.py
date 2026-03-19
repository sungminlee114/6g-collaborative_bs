"""Standalone DL-CE training script.

Uses the existing SiteAwareEstimator (site_integration='none' = plain estimator)
and train_local() from the training module — the same pipeline that achieved
-17.5dB in E0.

Key difference from E0: new presets (elaa_s, mimo_15g) have channel values ~1e-6,
which causes float32 instability in NMSE loss. We wrap the dataset with per-sample
normalization to train at unit scale, then denormalize at inference.

Usage:
    python -m src.experiments.S1_ce_profiling.train_dl_ce --preset munich_elaa_s_1k_15g --gpu 0
    python -m src.experiments.S1_ce_profiling.train_dl_ce --all
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.config import get_results_dir
from src.tracker import Tracker
from src.ce_skip import ExperimentConfig
from src.models.estimator import create_model
from src.training.trainer import train_local, save_checkpoint, load_checkpoint
from src.dataset_operation.dataset import ChannelEstimationDataset
from src.dataset_operation.utils import nmse

PRIMARY_PRESETS = [
    "munich_elaa_s_1k_15g",
    "munich_mimo_15g",
]

CKPT_DIR = Path("assets/checkpoints/ce_skip")


def ckpt_path_for(cfg: ExperimentConfig) -> Path:
    return CKPT_DIR / f"dl_ce_{cfg.deployment}_{cfg.num_subcarriers}_{int(cfg.frequency/1e9)}g.pt"


class NormalizedCEDataset(Dataset):
    """Wraps ChannelEstimationDataset with per-sample RMS normalization.

    Filters out near-zero samples (complete blockage) that would cause
    division-by-zero in NMSE loss.
    """

    def __init__(self, base_dataset: ChannelEstimationDataset, min_rms: float = 1e-10):
        self.base = base_dataset
        # Pre-filter: find valid indices (non-zero target)
        self.valid_indices = []
        for i in range(len(base_dataset)):
            target = base_dataset[i]["target"]
            rms = target.pow(2).mean().sqrt()
            if rms > min_rms:
                self.valid_indices.append(i)
        n_filtered = len(base_dataset) - len(self.valid_indices)
        if n_filtered > 0:
            print(f"  NormalizedCEDataset: filtered {n_filtered}/{len(base_dataset)} near-zero samples")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        sample = self.base[self.valid_indices[idx]]
        target = sample["target"]
        rms = target.pow(2).mean().sqrt()
        return {
            "input": sample["input"] / rms,
            "target": target / rms,
            "bs_id": sample["bs_id"],
            "snr_db": sample["snr_db"],
        }


def train_one(preset: str, gpu: int = 0, epochs: int = 100, batch_size: int = 32):
    """Train DL-CE for a single preset using existing infrastructure."""
    device = f"cuda:{gpu}"
    torch.cuda.set_device(device)
    cfg = ExperimentConfig.from_preset(preset)

    if not Path(cfg.data_dir).exists():
        print(f"  ⚠ Data not found: {cfg.data_dir}")
        return None

    n_ant = cfg.num_rx_ant * cfg.num_tx_ant
    print(f"  Data: {cfg.data_dir} ({n_ant} ant × {cfg.num_subcarriers} sc)")

    # Limit snapshots for memory
    max_snap = 5 if n_ant > 256 else 30
    train_ds = NormalizedCEDataset(ChannelEstimationDataset(
        cfg.data_dir, bs_ids=cfg.train_bs_ids,
        snapshot_ids=list(range(max_snap)), snr_range_db=(0.0, 30.0),
    ))
    val_ds = NormalizedCEDataset(ChannelEstimationDataset(
        cfg.data_dir, bs_ids=cfg.val_bs_ids,
        snapshot_ids=list(range(max_snap, min(max_snap + 3, 100))), snr_range_db=(0.0, 30.0),
    ))
    print(f"  Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")

    # Check normalized scale
    s = train_ds[0]
    print(f"  Normalized input std: {s['input'].std():.3f}, target std: {s['target'].std():.3f}")

    # Adjust batch size for large arrays
    if n_ant > 256:
        batch_size = min(batch_size, 4)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Use the SAME model architecture as E0
    model = create_model(site_integration="none", encoder_channels=64, encoder_blocks=3, task_head_blocks=2)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params:,} params (SiteAwareEstimator, site=none)")

    # Use train_local — the same training loop that worked in E0
    result = train_local(
        model, train_loader, val_loader,
        epochs=epochs, lr=1e-3, weight_decay=1e-4,
        patience=25, device=device, grad_clip=1.0,
        warmup_epochs=5, use_cosine=True, verbose=True,
    )

    best_val_db = result["best_val"]
    print(f"  Best val NMSE: {best_val_db:.1f} dB (epoch {result['best_epoch']})")

    # Save checkpoint
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    path = ckpt_path_for(cfg)
    save_checkpoint(model, f"ce_skip/dl_ce_{cfg.deployment}_{cfg.num_subcarriers}_{int(cfg.frequency/1e9)}g",
                    meta={"best_val_db": best_val_db, "normalization": "per_sample_rms"})
    print(f"  Saved: {path}")

    return {"best_val_nmse_db": best_val_db, "n_params": n_params, "path": str(path)}


def main():
    parser = argparse.ArgumentParser(description="Train DL-CE models")
    parser.add_argument("--preset", type=str, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    presets = PRIMARY_PRESETS if args.all else [args.preset or PRIMARY_PRESETS[0]]

    with Tracker(
        "S1/dl_ce_training",
        config={"presets": presets, "epochs": args.epochs},
        capture_output=True,
        purpose="Train DL-CE (SiteAwareEstimator, site=none) with per-sample normalization",
        variables={
            "independent": ["preset"],
            "dependent": ["val_nmse_db"],
            "controlled": [f"epochs={args.epochs}", "encoder_blocks=3", "task_head_blocks=2"],
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
