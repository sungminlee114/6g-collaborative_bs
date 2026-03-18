"""S1: CE Implementation + A100 GPU Profiling.

Implements 3 CE methods (LS, Genie-LMMSE, DL-CE) and measures:
- Wall-clock time (ms) on GPU
- NMSE accuracy
- GPU memory usage

DL-CE is trained on independent drops data before profiling.

Usage:
    python -m src.experiments.S1_ce_profiling.run --preset munich_elaa_l_1k_15g --gpu 0
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import SceneConfig, get_results_dir
from src.tracker import Tracker
from src.dataset_operation.dataset import ChannelEstimationDataset
from src.dataset_operation.utils import nmse, nmse_db
from src.ce_skip.ce_algorithms import (
    LSEstimator, GenieLMMSE, DLCEEstimator, create_ce_estimator,
)

PRIMARY_PRESETS = [
    "munich_elaa_s_1k_15g",   # 15 GHz, 16×16, 1024 SC
    "munich_mimo_15g",        # 15 GHz, 8×8, 1024 SC (FF baseline)
]

# ─── DL-CE Training ────────────────────────────────────────────────
def train_dl_ce(
    model: DLCEEstimator,
    data_dir: str,
    train_bs_ids: list,
    val_bs_ids: list,
    epochs: int = 100,
    batch_size: int = 128,
    lr: float = 1e-3,
    device: str = "cuda",
    snr_range: tuple = (0.0, 30.0),
) -> dict:
    """Train DL-CE on independent drops data."""
    train_ds = ChannelEstimationDataset(data_dir, bs_ids=train_bs_ids, snr_range_db=snr_range)
    val_ds = ChannelEstimationDataset(data_dir, bs_ids=val_bs_ids, snr_range_db=snr_range)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        # Train
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

        # Val
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

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs}: train={10*np.log10(train_loss):.1f}dB "
                  f"val={10*np.log10(val_loss):.1f}dB")

        if patience_counter >= 25:
            print(f"    Early stop at epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"best_val_nmse_db": 10 * np.log10(best_val)}


# ─── GPU Profiling ──────────────────────────────────────────────────
def profile_ce_method(
    ce_method,
    h_input: torch.Tensor,
    n_warmup: int = 10,
    n_repeat: int = 100,
    snr_db: float = 20.0,
) -> dict:
    """Profile a CE method on GPU.

    Args:
        ce_method: callable — CE estimator
        h_input: (batch, 2, n_ant, n_sc) — input tensor
        n_warmup: warm-up iterations
        n_repeat: timing iterations

    Returns dict with time_ms, memory_mb
    """
    device = h_input.device

    # Handle LMMSE snr parameter
    is_lmmse = isinstance(ce_method, GenieLMMSE)

    # Warm up
    with torch.no_grad():
        for _ in range(n_warmup):
            if is_lmmse:
                ce_method(h_input, snr_db=snr_db)
            else:
                ce_method(h_input)
    torch.cuda.synchronize()

    # Profile
    torch.cuda.reset_peak_memory_stats()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    with torch.no_grad():
        for _ in range(n_repeat):
            if is_lmmse:
                h_hat = ce_method(h_input, snr_db=snr_db)
            else:
                h_hat = ce_method(h_input)
    end_event.record()
    torch.cuda.synchronize()

    time_ms = start_event.elapsed_time(end_event) / n_repeat
    peak_mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    return {
        "time_ms": time_ms,
        "peak_memory_mb": peak_mem_mb,
    }


def run_profiling(preset: str, gpu: int = 0, train_dl: bool = True):
    """Run CE profiling for a single preset."""
    device = f"cuda:{gpu}"
    torch.cuda.set_device(device)

    cfg = SceneConfig.from_preset(preset)
    data_dir = cfg.data_dir

    if not Path(data_dir).exists():
        print(f"  ⚠ Data not found: {data_dir}")
        return None

    print(f"  Loading data from {data_dir}")

    # Dimensions from config
    n_rx = cfg.num_rx_ant
    n_tx = cfg.num_tx_ant
    n_ant = n_rx * n_tx
    n_sc = cfg.num_subcarriers

    # Create CE methods
    ls = LSEstimator()
    lmmse = GenieLMMSE(device=device)
    dl_ce = DLCEEstimator(n_blocks=8, channels=64).to(device)

    # Fit LMMSE on training data
    print("  Fitting Genie-LMMSE covariance...")
    train_ds = ChannelEstimationDataset(
        data_dir, bs_ids=cfg.train_bs_ids,
        snr_range_db=(20.0, 20.0),
    )
    # Collect clean channels for covariance
    h_samples = []
    loader = DataLoader(train_ds, batch_size=256, shuffle=False)
    for batch in loader:
        h_samples.append(batch["target"])
        if len(h_samples) * 256 >= 2000:
            break
    h_samples = torch.cat(h_samples, dim=0)[:2000].to(device)
    lmmse.fit(h_samples)
    print(f"  LMMSE fitted on {len(h_samples)} samples")

    # Train DL-CE
    if train_dl:
        print("  Training DL-CE...")
        dl_result = train_dl_ce(
            dl_ce, data_dir,
            train_bs_ids=cfg.train_bs_ids,
            val_bs_ids=cfg.val_bs_ids,
            epochs=100, batch_size=128, device=device,
        )
        print(f"  DL-CE best val NMSE: {dl_result['best_val_nmse_db']:.1f} dB")
    dl_ce.eval()

    # Profile each method
    # Use a representative batch
    val_ds = ChannelEstimationDataset(
        data_dir, bs_ids=cfg.val_bs_ids, fixed_snr_db=20.0,
    )
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    batch = next(iter(val_loader))
    h_noisy = batch["input"].to(device)
    h_true = batch["target"].to(device)

    results = {}
    for name, method in [("ls", ls), ("lmmse", lmmse), ("dl_ce", dl_ce)]:
        print(f"  Profiling {name}...")

        # Timing
        profile = profile_ce_method(method, h_noisy, snr_db=20.0)

        # Accuracy
        with torch.no_grad():
            if isinstance(method, GenieLMMSE):
                h_hat = method(h_noisy, snr_db=20.0)
            else:
                if isinstance(method, LSEstimator):
                    h_hat = method(h_noisy)
                else:
                    h_hat = method(h_noisy)

        nmse_val = nmse(h_hat, h_true).item()
        nmse_db_val = 10 * np.log10(nmse_val)

        results[name] = {
            "time_ms": profile["time_ms"],
            "peak_memory_mb": profile["peak_memory_mb"],
            "nmse": nmse_val,
            "nmse_db": nmse_db_val,
        }
        print(f"    {name}: {profile['time_ms']:.3f} ms, "
              f"NMSE={nmse_db_val:.1f} dB, "
              f"mem={profile['peak_memory_mb']:.0f} MB")

    return results


def main():
    parser = argparse.ArgumentParser(description="S1: CE Profiling")
    parser.add_argument("--preset", type=str, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--no-train-dl", action="store_true",
                        help="Skip DL-CE training (use random init)")
    args = parser.parse_args()

    presets = PRIMARY_PRESETS if args.all else [args.preset or PRIMARY_PRESETS[0]]
    results_dir = get_results_dir("S1_ce_profiling")

    with Tracker(
        "S1/ce_profiling",
        config={"presets": presets, "gpu": args.gpu},
        capture_output=True,
        purpose="CE algorithm implementation and A100 GPU profiling",
        variables={
            "independent": ["ce_method", "preset"],
            "dependent": ["time_ms", "nmse_db", "peak_memory_mb"],
            "controlled": ["batch_size=64", "snr=20dB", "n_repeat=100"],
        },
        hypothesis="DL-CE ~100x slower than LS, LMMSE ~10x slower than LS",
        eval_criteria="Measure wall-clock time, NMSE, memory per CE method",
    ) as run:
        all_results = {}
        for preset in presets:
            print(f"\n{'─'*60}")
            print(f"  Preset: {preset}")
            print(f"{'─'*60}")
            r = run_profiling(preset, gpu=args.gpu, train_dl=not args.no_train_dl)
            if r is not None:
                all_results[preset] = r
                run.log(**{f"{preset}_done": True})

        # Save results
        with open(results_dir / "ce_profiling.json", "w") as f:
            json.dump(all_results, f, indent=2)
        run.set_result(**{
            f"{p}_{m}_ms": r[m]["time_ms"]
            for p, r in all_results.items()
            for m in r
        })

        # Print summary table
        print(f"\n{'═'*60}")
        print("  CE Profiling Summary")
        print(f"{'═'*60}")
        print(f"{'Preset':<30} {'Method':<8} {'Time(ms)':<10} {'NMSE(dB)':<10} {'Mem(MB)':<10}")
        for preset, methods in all_results.items():
            short = preset.replace("munich_", "")
            for method, stats in methods.items():
                print(f"{short:<30} {method:<8} {stats['time_ms']:<10.3f} "
                      f"{stats['nmse_db']:<10.1f} {stats['peak_memory_mb']:<10.0f}")


if __name__ == "__main__":
    main()
