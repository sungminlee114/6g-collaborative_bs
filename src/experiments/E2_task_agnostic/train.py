"""Phase 2: Task-agnostic theta_BS verification (Killer Experiment).

Step 2-1: Task A (channel estimation) -> Task B (power profile prediction) transfer
Step 2-2: Pre-trained E as downstream backbone

Usage:
    python -m src.experiments.E2_task_agnostic.train --step 2-1 --gpu 0
    python -m src.experiments.E2_task_agnostic.train --step 2-1 --test_bs 6 --gpu 0
    python -m src.experiments.E2_task_agnostic.train --step 2-2 --gpu 0
    python -m src.experiments.E2_task_agnostic.train --step all --gpus 0 1 2 3
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.dataset_operation.dataset import ChannelEstimationDataset
from src.dataset_operation.utils import nmse
from src.models.estimator import (
    SiteAwareEstimator, SharedEncoder, TaskHead, SiteEmbedding,
    FiLMSiteInjection, create_model,
)
from src.models.baselines import PlainEstimator
from src.training.trainer import train_local, evaluate, save_checkpoint, load_checkpoint

# ── Defaults (override with --dataset uma/umi) ──
from src.config import SceneConfig, DatasetConfig
PRESETS = {"uma": "munich_uma8", "umi": "munich_umi16"}
DATASET = "uma"
_scene = SceneConfig.from_preset(PRESETS[DATASET])
_ds = DatasetConfig.from_scene(_scene)
DATA_DIR = _ds.data_dir
PRETRAIN_BS = _ds.pretrain_bs_ids
TEST_BS = _ds.test_bs_ids
SAVE_PREFIX = f"phase2/{DATASET}"
SAVE_DIR = Path("assets/checkpoints") / SAVE_PREFIX
PHASE1_PREFIX = f"phase1/{DATASET}"
BATCH_SIZE = 512
LR = 1e-3
EPOCHS = 100


class PowerProfileDataset(Dataset):
    """Task B: Predict power delay profile from noisy channel.

    Input: noisy LS estimate (2, 8, 1024) real
    Target: power profile per antenna pair (1, 8, 1024) — |H|^2 normalized
    """

    def __init__(self, data_dir, bs_ids=None, snr_range_db=(0, 30), fixed_snr_db=None):
        # Reuse channel estimation dataset loading
        self.ce_ds = ChannelEstimationDataset(
            data_dir=data_dir, bs_ids=bs_ids,
            snr_range_db=snr_range_db, fixed_snr_db=fixed_snr_db,
        )

    def __len__(self):
        return len(self.ce_ds)

    def __getitem__(self, idx):
        sample = self.ce_ds[idx]
        # Target: power profile from clean channel
        h_real = sample["target"]  # (2, 8, 1024)
        power = h_real[0] ** 2 + h_real[1] ** 2  # |H|^2, (8, 1024)
        # Normalize per sample
        power = power / (power.mean() + 1e-10)
        return {
            "input": sample["input"],        # (2, 8, 1024) noisy
            "target": power.unsqueeze(0),     # (1, 8, 1024) power profile
            "bs_id": sample["bs_id"],
            "snr_db": sample["snr_db"],
        }


class PowerProfileHead(nn.Module):
    """Task head for power profile prediction. Output: (B, 1, 8, 1024)."""

    def __init__(self, in_channels=64, kernel_size=3):
        super().__init__()
        from src.models.estimator import ResBlock2D
        self.net = nn.Sequential(
            ResBlock2D(in_channels, kernel_size),
            ResBlock2D(in_channels, kernel_size),
            nn.Conv2d(in_channels, 1, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),  # power is non-negative
        )

    def forward(self, x):
        return self.net(x)


class TaskBModel(nn.Module):
    """Model for Task B using pre-trained encoder + site embedding + new task head."""

    def __init__(self, encoder, site_embedding=None, site_injection=None,
                 task_head=None):
        super().__init__()
        self.encoder = encoder
        self.site_embedding = site_embedding
        self.site_injection = site_injection
        self.task_head = task_head or PowerProfileHead(in_channels=64)

    def forward(self, x):
        feat = self.encoder(x)
        if self.site_injection is not None and self.site_embedding is not None:
            feat = self.site_injection(feat, self.site_embedding.embedding)
        return self.task_head(feat)


def power_profile_loss(pred, target):
    """MSE loss for power profile prediction, normalized."""
    err = (pred - target).flatten(1)
    ref = target.flatten(1)
    return (err.pow(2).sum(1) / (ref.pow(2).sum(1) + 1e-10)).mean()


def train_task_b(model, train_loader, val_loader, epochs, lr, device, save_as=None):
    """Train Task B model with power profile loss."""
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val = float("inf")
    best_state = None
    best_epoch = 0
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        total_loss, n = 0.0, 0
        for batch in train_loader:
            x = batch["input"].to(device)
            y = batch["target"].to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = power_profile_loss(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            n += x.size(0)
        scheduler.step()
        train_losses.append(total_loss / max(n, 1))

        # Validate
        model.eval()
        v_total, v_n = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["input"].to(device)
                y = batch["target"].to(device)
                pred = model(x)
                v_total += power_profile_loss(pred, y).item() * x.size(0)
                v_n += x.size(0)
        v_loss = v_total / max(v_n, 1)
        val_losses.append(v_loss)

        if v_loss < best_val:
            best_val = v_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}: train={train_losses[-1]:.6f}, val={v_loss:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    result = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_epoch": best_epoch,
        "best_val": best_val,
    }

    if save_as:
        save_checkpoint(model, save_as, meta=result)

    return result


def step_2_1(gpu: int, epochs: int = EPOCHS, lr: float = LR, test_bs_ids: list = None):
    """Step 2-1: Train Task A theta_BS -> freeze -> apply to Task B."""
    device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
    if test_bs_ids is None:
        test_bs_ids = TEST_BS

    print("=== Step 2-1: Task-Agnostic theta_BS Transfer ===")

    # Load pre-trained 3-way model from Phase 1
    ref_model = create_model(site_integration="film", site_embed_dim=64).to(device)
    try:
        load_checkpoint(ref_model, f"{PHASE1_PREFIX}/1-1_3way_bs{PRETRAIN_BS[0]}", device=device)
    except FileNotFoundError:
        print("ERROR: Run phase1 step 1-1 first")
        return

    for test_bs in test_bs_ids:
        print(f"\n--- Test BS{test_bs} ---")

        # Load Task A theta_BS for this test BS (if available from phase1)
        task_a_model = create_model(site_integration="film", site_embed_dim=64).to(device)
        try:
            load_checkpoint(task_a_model, f"{PHASE1_PREFIX}/1-1_3way_bs{test_bs}", device=device)
        except FileNotFoundError:
            print(f"  No Task A model for BS{test_bs}, training theta_BS first...")
            task_a_model.load_shared_state_dict(ref_model.shared_state_dict())
            ds_a = ChannelEstimationDataset(data_dir=DATA_DIR, bs_ids=[test_bs], snr_range_db=(0, 30))
            n_v = max(int(len(ds_a) * 0.2), 1)
            t_ds, v_ds = torch.utils.data.random_split(ds_a, [len(ds_a) - n_v, n_v])
            task_a_model.freeze_encoder()
            task_a_model.freeze_task_head()
            train_local(task_a_model, DataLoader(t_ds, batch_size=BATCH_SIZE, shuffle=True),
                        DataLoader(v_ds, batch_size=BATCH_SIZE), epochs=50, lr=lr, device=device)

        # Task B dataset
        ds_b = PowerProfileDataset(data_dir=DATA_DIR, bs_ids=[test_bs], snr_range_db=(0, 30))
        n_val = max(int(len(ds_b) * 0.2), 1)
        t_ds_b, v_ds_b = torch.utils.data.random_split(ds_b, [len(ds_b) - n_val, n_val])
        train_b = DataLoader(t_ds_b, batch_size=BATCH_SIZE, shuffle=True)
        val_b = DataLoader(v_ds_b, batch_size=BATCH_SIZE)

        results = {}

        # Method 1: With frozen theta_BS from Task A
        print("  Training Task B with frozen theta_BS...")
        m_with = TaskBModel(
            encoder=ref_model.encoder,
            site_embedding=task_a_model.site_embedding,
            site_injection=ref_model.site_injection,
            task_head=PowerProfileHead(),
        ).to(device)
        # Freeze encoder + site embedding, only train task head
        for p in m_with.encoder.parameters():
            p.requires_grad = False
        for p in m_with.site_embedding.parameters():
            p.requires_grad = False
        for p in m_with.site_injection.parameters():
            p.requires_grad = False
        res_with = train_task_b(m_with, train_b, val_b, epochs, lr, device,
                                save_as=f"{SAVE_PREFIX}/2-1_with_theta_bs{test_bs}")
        results["with_theta_bs"] = res_with

        # Method 2: Without theta_BS (encoder only)
        print("  Training Task B without theta_BS...")
        m_without = TaskBModel(
            encoder=ref_model.encoder,
            site_embedding=None,
            site_injection=None,
            task_head=PowerProfileHead(),
        ).to(device)
        for p in m_without.encoder.parameters():
            p.requires_grad = False
        res_without = train_task_b(m_without, train_b, val_b, epochs, lr, device,
                                   save_as=f"{SAVE_PREFIX}/2-1_without_theta_bs{test_bs}")
        results["without_theta_bs"] = res_without

        # Method 3: From scratch
        print("  Training Task B from scratch...")
        from src.models.estimator import SharedEncoder
        m_scratch = TaskBModel(
            encoder=SharedEncoder(2, 64, 3, 3),
            task_head=PowerProfileHead(),
        ).to(device)
        res_scratch = train_task_b(m_scratch, train_b, val_b, epochs, lr, device,
                                   save_as=f"{SAVE_PREFIX}/2-1_scratch_bs{test_bs}")
        results["from_scratch"] = res_scratch

        for name, res in results.items():
            print(f"  {name}: best_val={res['best_val']:.6f} (epoch {res['best_epoch']})")


def step_2_2(gpu: int, epochs: int = EPOCHS, lr: float = LR, test_bs_ids: list = None):
    """Step 2-2: Pre-trained E as downstream backbone."""
    device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
    if test_bs_ids is None:
        test_bs_ids = TEST_BS

    print("=== Step 2-2: E as Downstream Backbone ===")

    # Load pre-trained encoder
    ref_model = create_model(site_integration="film", site_embed_dim=64).to(device)
    try:
        load_checkpoint(ref_model, f"{PHASE1_PREFIX}/1-1_3way_bs{PRETRAIN_BS[0]}", device=device)
    except FileNotFoundError:
        print("ERROR: Run phase1 step 1-1 first")
        return

    for test_bs in test_bs_ids:
        print(f"\n--- Test BS{test_bs} ---")

        ds = ChannelEstimationDataset(data_dir=DATA_DIR, bs_ids=[test_bs], snr_range_db=(0, 30))
        n_val = max(int(len(ds) * 0.2), 1)
        t_ds, v_ds = torch.utils.data.random_split(ds, [len(ds) - n_val, n_val])
        train_loader = DataLoader(t_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(v_ds, batch_size=BATCH_SIZE)

        results = {}

        # Frozen pre-trained E + trainable task head
        print("  Frozen pre-trained E...")
        m_frozen = PlainEstimator(encoder_channels=64, encoder_blocks=3).to(device)
        m_frozen.encoder.load_state_dict(ref_model.encoder.state_dict())
        for p in m_frozen.encoder.parameters():
            p.requires_grad = False
        res_frozen = train_local(m_frozen, train_loader, val_loader, epochs=epochs, lr=lr,
                                 device=device, save_as=f"{SAVE_PREFIX}/2-2_frozen_E_bs{test_bs}")
        results["frozen_pretrained_E"] = res_frozen

        # Random E (frozen) + trainable task head
        print("  Frozen random E...")
        m_random = PlainEstimator(encoder_channels=64, encoder_blocks=3).to(device)
        for p in m_random.encoder.parameters():
            p.requires_grad = False
        res_random = train_local(m_random, train_loader, val_loader, epochs=epochs, lr=lr,
                                 device=device, save_as=f"{SAVE_PREFIX}/2-2_random_E_bs{test_bs}")
        results["frozen_random_E"] = res_random

        # Full training from scratch (reference)
        print("  Full from scratch...")
        m_full = PlainEstimator(encoder_channels=64, encoder_blocks=3).to(device)
        res_full = train_local(m_full, train_loader, val_loader, epochs=epochs, lr=lr,
                               device=device, save_as=f"{SAVE_PREFIX}/2-2_full_bs{test_bs}")
        results["full_scratch"] = res_full

        for name, res in results.items():
            best_db = 10 * np.log10(res["best_val"])
            print(f"  {name}: {best_db:.2f} dB (epoch {res['best_epoch']})")


def launch_all(args):
    """Launch steps x test_bs in parallel via subprocess."""
    gpus = args.gpus or list(range(torch.cuda.device_count()))

    # Create (step, test_bs) combinations for maximum parallelism
    jobs = []
    for step in ["2-1", "2-2"]:
        for bs_id in TEST_BS:
            jobs.append((step, bs_id))

    print(f"Launching {len(jobs)} jobs on {len(gpus)} GPUs")
    processes = []
    for i, (step, bs_id) in enumerate(jobs):
        gpu = gpus[i % len(gpus)]
        cmd = [
            sys.executable, "-m", "src.experiments.E2_task_agnostic.train",
            "--step", step, "--gpu", str(gpu),
            "--test_bs", str(bs_id),
            "--epochs", str(args.epochs),
            "--lr", str(args.lr),
            "--dataset", args.dataset,
        ]
        label = f"{step}/BS{bs_id}"
        print(f"  {label} -> GPU{gpu}")
        p = subprocess.Popen(cmd)
        processes.append((label, gpu, p))

    for label, gpu, p in processes:
        ret = p.wait()
        status = "OK" if ret == 0 else f"FAILED (code={ret})"
        print(f"  {label} (GPU{gpu}): {status}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=str, required=True,
                        choices=["2-1", "2-2", "all"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--gpus", type=int, nargs="*", default=None, help="GPU indices (default: all)")
    parser.add_argument("--test_bs", type=int, nargs="*", default=None,
                        help="Test BS IDs to process (default: all test BSs)")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--dataset", type=str, default=DATASET, choices=list(PRESETS.keys()),
                        help="Dataset to use: uma (8BS) or umi (16BS)")
    args = parser.parse_args()

    # Apply dataset selection (re-derive from chosen preset)
    _sel_scene = SceneConfig.from_preset(PRESETS[args.dataset])
    _sel_ds = DatasetConfig.from_scene(_sel_scene)
    DATA_DIR = _sel_ds.data_dir
    PRETRAIN_BS = _sel_ds.pretrain_bs_ids
    TEST_BS = _sel_ds.test_bs_ids
    SAVE_PREFIX = f"phase2/{args.dataset}"
    SAVE_DIR = Path("assets/checkpoints") / SAVE_PREFIX
    PHASE1_PREFIX = f"phase1/{args.dataset}"

    test_bs_ids = args.test_bs if args.test_bs else None

    if args.step == "all":
        launch_all(args)
    elif args.step == "2-1":
        step_2_1(args.gpu, epochs=args.epochs, lr=args.lr, test_bs_ids=test_bs_ids)
    elif args.step == "2-2":
        step_2_2(args.gpu, epochs=args.epochs, lr=args.lr, test_bs_ids=test_bs_ids)
