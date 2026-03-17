"""Phase 3: Ablation Studies.

Step 3-1: theta_BS dimension {8, 16, 32, 64, 128, 256}
Step 3-2: Site injection type (FiLM vs concat vs add vs none)
Step 3-3: Pre-training BS count {2, 4, 6}
Step 3-5: Cold start analysis (NMSE vs training sample count)
Step 3-6: Multi-task theta_BS — joint Task A+B training vs single-task

Usage:
    python -m src.experiments.E3_ablation.train --step 3-1 --gpu 0
    python -m src.experiments.E3_ablation.train --step 3-1 --dims 8 16 --gpu 0
    python -m src.experiments.E3_ablation.train --step 3-2 --inject_types film concat --gpu 1
    python -m src.experiments.E3_ablation.train --step 3-3 --gpu 0
    python -m src.experiments.E3_ablation.train --step 3-5 --gpu 0
    python -m src.experiments.E3_ablation.train --step 3-6 --gpu 0
    python -m src.experiments.E3_ablation.train --step all --gpus 0 1 2 3
"""
import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.dataset_operation.dataset import ChannelEstimationDataset
from src.dataset_operation.utils import nmse
from src.models.estimator import create_model
from src.models.baselines import PlainEstimator
from src.training.trainer import train_local, evaluate, save_checkpoint


# ── Defaults (override with --dataset uma/umi) ──
from src.config import SceneConfig, DatasetConfig
PRESETS = {"uma": "munich_uma8", "umi": "munich_umi16"}
DATASET = "uma"
_scene = SceneConfig.from_preset(PRESETS[DATASET])
_ds = DatasetConfig.from_scene(_scene)
DATA_DIR = _ds.data_dir
PRETRAIN_BS = _ds.pretrain_bs_ids
TEST_BS_ID = _ds.test_bs_ids[0]
SAVE_PREFIX = f"phase3/{DATASET}"
SAVE_DIR = Path("assets/checkpoints") / SAVE_PREFIX
BATCH_SIZE = 128
LR = 1e-3
EPOCHS = 80
ADAPT_EPOCHS = 50


def _get_loaders(bs_ids, val_frac=0.2):
    """Build train/val loaders for given BS IDs."""
    ds = ChannelEstimationDataset(data_dir=DATA_DIR, bs_ids=bs_ids, snr_range_db=(0, 30))
    n_val = max(int(len(ds) * val_frac), 1)
    t_ds, v_ds = torch.utils.data.random_split(ds, [len(ds) - n_val, n_val])
    return (DataLoader(t_ds, batch_size=BATCH_SIZE, shuffle=True),
            DataLoader(v_ds, batch_size=BATCH_SIZE))


def step_3_1(gpu: int, epochs: int = EPOCHS, adapt_epochs: int = ADAPT_EPOCHS, lr: float = LR,
             dims: list = None):
    """Step 3-1: theta_BS dimension ablation."""
    device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
    print("=== Step 3-1: theta_BS Dimension Ablation ===")

    train_loader, val_loader = _get_loaders(PRETRAIN_BS)
    test_train_loader, test_val_loader = _get_loaders([TEST_BS_ID], val_frac=0.3)

    DIMS = dims or [8, 16, 32, 64, 128, 256]
    results = {}

    for dim in DIMS:
        print(f"\n--- theta_BS dim = {dim} ---")
        # Pre-train with this dim
        model = create_model(site_integration="film", site_embed_dim=dim).to(device)
        train_local(model, train_loader, val_loader, epochs=epochs, lr=lr, device=device,
                    save_as=f"{SAVE_PREFIX}/3-1_pretrain_dim{dim}")

        # Adapt theta_BS only on test BS
        adapted = copy.deepcopy(model).to(device)
        adapted.site_embedding.reset()
        adapted.freeze_encoder()
        adapted.freeze_task_head()
        res = train_local(adapted, test_train_loader, test_val_loader,
                          epochs=adapt_epochs, lr=1e-2, device=device,
                          save_as=f"{SAVE_PREFIX}/3-1_adapt_dim{dim}")

        nmse_db = 10 * np.log10(res["best_val"])
        results[dim] = nmse_db
        print(f"  dim={dim}: NMSE = {nmse_db:.2f} dB")

    # Save summary (merge with existing results from other GPU processes)
    summary_path = SAVE_DIR / "3-1_dim_results.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if summary_path.exists():
        with open(summary_path) as f:
            existing = json.load(f)
    existing.update({str(k): v for k, v in results.items()})
    with open(summary_path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"\nSaved: {summary_path}")


def step_3_2(gpu: int, epochs: int = EPOCHS, adapt_epochs: int = ADAPT_EPOCHS, lr: float = LR,
             inject_types: list = None):
    """Step 3-2: Site injection type ablation."""
    device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
    print("=== Step 3-2: Site Injection Type Ablation ===")

    train_loader, val_loader = _get_loaders(PRETRAIN_BS)
    test_train_loader, test_val_loader = _get_loaders([TEST_BS_ID], val_frac=0.3)

    INJECTION_TYPES = inject_types or ["film", "concat", "add", "none"]
    results = {}

    for inject_type in INJECTION_TYPES:
        print(f"\n--- Injection: {inject_type} ---")
        model = create_model(site_integration=inject_type, site_embed_dim=64).to(device)
        train_local(model, train_loader, val_loader, epochs=epochs, lr=lr, device=device,
                    save_as=f"{SAVE_PREFIX}/3-2_pretrain_{inject_type}")

        if inject_type != "none":
            adapted = copy.deepcopy(model).to(device)
            adapted.site_embedding.reset()
            adapted.freeze_encoder()
            adapted.freeze_task_head()
            res = train_local(adapted, test_train_loader, test_val_loader,
                              epochs=adapt_epochs, lr=1e-2, device=device,
                              save_as=f"{SAVE_PREFIX}/3-2_adapt_{inject_type}")
        else:
            # For 'none': fine-tune entire model (no site embedding to adapt)
            adapted = copy.deepcopy(model).to(device)
            res = train_local(adapted, test_train_loader, test_val_loader,
                              epochs=adapt_epochs, lr=1e-4, device=device,
                              save_as=f"{SAVE_PREFIX}/3-2_adapt_{inject_type}")

        nmse_db = 10 * np.log10(res["best_val"])
        results[inject_type] = nmse_db
        print(f"  {inject_type}: NMSE = {nmse_db:.2f} dB")

    # Save summary (merge with existing results from other GPU processes)
    summary_path = SAVE_DIR / "3-2_injection_results.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if summary_path.exists():
        with open(summary_path) as f:
            existing = json.load(f)
    existing.update(results)
    with open(summary_path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"\nSaved: {summary_path}")


def step_3_3(gpu: int, epochs: int = EPOCHS, adapt_epochs: int = ADAPT_EPOCHS, lr: float = LR):
    """Step 3-3: Pre-training BS count ablation."""
    device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
    print("=== Step 3-3: Pre-training BS Count Ablation ===")

    test_train_loader, test_val_loader = _get_loaders([TEST_BS_ID], val_frac=0.3)

    BS_COUNTS = [2, 4, 6]
    results = {}

    for n_bs in BS_COUNTS:
        bs_subset = PRETRAIN_BS[:n_bs]
        print(f"\n--- Pre-train BSs: {bs_subset} ---")

        train_loader, val_loader = _get_loaders(bs_subset)
        model = create_model(site_integration="film", site_embed_dim=64).to(device)
        train_local(model, train_loader, val_loader, epochs=epochs, lr=lr, device=device,
                    save_as=f"{SAVE_PREFIX}/3-3_pretrain_{n_bs}bs")

        adapted = copy.deepcopy(model).to(device)
        adapted.site_embedding.reset()
        adapted.freeze_encoder()
        adapted.freeze_task_head()
        res = train_local(adapted, test_train_loader, test_val_loader,
                          epochs=adapt_epochs, lr=1e-2, device=device,
                          save_as=f"{SAVE_PREFIX}/3-3_adapt_{n_bs}bs")

        nmse_db = 10 * np.log10(res["best_val"])
        results[n_bs] = nmse_db
        print(f"  {n_bs} BSs: NMSE = {nmse_db:.2f} dB")

    summary_path = SAVE_DIR / "3-3_bs_count_results.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {summary_path}")


def step_3_5(gpu: int, epochs_pretrain: int = EPOCHS, lr: float = LR):
    """Step 3-5: Cold start analysis — NMSE vs training sample count."""
    device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
    print("=== Step 3-5: Cold Start Analysis ===")

    SAMPLE_COUNTS = [5, 10, 20, 50, 100, 200, 500]
    N_REPEATS = 3

    # Pre-train model
    train_loader, val_loader = _get_loaders(PRETRAIN_BS)
    pretrained = create_model(site_integration="film", site_embed_dim=64).to(device)
    train_local(pretrained, train_loader, val_loader, epochs=epochs_pretrain, lr=lr,
                device=device, save_as=f"{SAVE_PREFIX}/3-5_pretrained")

    # Test dataset
    test_ds = ChannelEstimationDataset(data_dir=DATA_DIR, bs_ids=[TEST_BS_ID], snr_range_db=(0, 30))
    test_indices = list(range(len(test_ds)))
    np.random.seed(42)
    np.random.shuffle(test_indices)
    val_idx = test_indices[:int(len(test_ds) * 0.3)]
    available_train_idx = test_indices[int(len(test_ds) * 0.3):]
    fixed_val_loader = DataLoader(Subset(test_ds, val_idx), batch_size=BATCH_SIZE)

    results = {"theta_bs_adapt": {}, "from_scratch": {}}

    for n_samples in SAMPLE_COUNTS:
        if n_samples > len(available_train_idx):
            break

        adapt_scores, scratch_scores = [], []

        for rep in range(N_REPEATS):
            np.random.seed(rep * 1000 + n_samples)
            sub_idx = np.random.choice(available_train_idx, n_samples, replace=False).tolist()
            sub_loader = DataLoader(Subset(test_ds, sub_idx),
                                    batch_size=min(n_samples, BATCH_SIZE), shuffle=True)

            # theta_BS adapt
            m1 = copy.deepcopy(pretrained).to(device)
            m1.site_embedding.reset()
            m1.freeze_encoder()
            m1.freeze_task_head()
            train_local(m1, sub_loader, fixed_val_loader, epochs=50, lr=1e-2,
                        device=device, verbose=False)
            _, db = evaluate(m1, fixed_val_loader, device)
            adapt_scores.append(db)

            # From scratch
            m2 = PlainEstimator(encoder_channels=64, encoder_blocks=3).to(device)
            train_local(m2, sub_loader, fixed_val_loader, epochs=100, lr=lr,
                        device=device, verbose=False)
            _, db = evaluate(m2, fixed_val_loader, device)
            scratch_scores.append(db)

        results["theta_bs_adapt"][n_samples] = {
            "mean": float(np.mean(adapt_scores)), "std": float(np.std(adapt_scores)),
            "values": adapt_scores,
        }
        results["from_scratch"][n_samples] = {
            "mean": float(np.mean(scratch_scores)), "std": float(np.std(scratch_scores)),
            "values": scratch_scores,
        }
        print(f"n={n_samples}: adapt={np.mean(adapt_scores):.2f}+/-{np.std(adapt_scores):.2f}, "
              f"scratch={np.mean(scratch_scores):.2f}+/-{np.std(scratch_scores):.2f}")

    summary_path = SAVE_DIR / "3-5_cold_start_results.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {summary_path}")


def step_3_6(gpu: int, epochs: int = EPOCHS, adapt_epochs: int = ADAPT_EPOCHS, lr: float = LR):
    """Step 3-6: Multi-task theta_BS — joint Task A+B vs single-task.

    Compare theta_BS quality when trained on:
    1. Task A only (channel estimation)
    2. Task A + Task B jointly (channel estimation + power profile)
    Evaluate both on held-out Task B data.
    """
    from src.experiments.E2_task_agnostic.train import (
        PowerProfileDataset, PowerProfileHead, TaskBModel,
        power_profile_loss, train_task_b,
    )
    device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
    print("=== Step 3-6: Multi-task theta_BS Ablation ===")

    test_bs = TEST_BS_ID
    results = {}

    # --- Pre-train shared E + theta_task on pretrain BSs ---
    train_loader_a, val_loader_a = _get_loaders(PRETRAIN_BS)
    base_model = create_model(site_integration="film", site_embed_dim=64).to(device)
    train_local(base_model, train_loader_a, val_loader_a, epochs=epochs, lr=lr,
                device=device, save_as=f"{SAVE_PREFIX}/3-6_base_pretrain")

    # --- Task B data for test BS ---
    ds_b = PowerProfileDataset(data_dir=DATA_DIR, bs_ids=[test_bs], snr_range_db=(0, 30))
    n_val_b = max(int(len(ds_b) * 0.3), 1)
    t_b, v_b = torch.utils.data.random_split(ds_b, [len(ds_b) - n_val_b, n_val_b])
    train_b = DataLoader(t_b, batch_size=BATCH_SIZE, shuffle=True)
    val_b = DataLoader(v_b, batch_size=BATCH_SIZE)

    # --- Task A data for test BS (for theta_BS training) ---
    ds_a_test = ChannelEstimationDataset(data_dir=DATA_DIR, bs_ids=[test_bs], snr_range_db=(0, 30))
    n_val_a = max(int(len(ds_a_test) * 0.3), 1)
    t_a, v_a = torch.utils.data.random_split(ds_a_test, [len(ds_a_test) - n_val_a, n_val_a])
    train_a_test = DataLoader(t_a, batch_size=BATCH_SIZE, shuffle=True)
    val_a_test = DataLoader(v_a, batch_size=BATCH_SIZE)

    # --- Method 1: theta_BS from Task A only ---
    print("\n--- theta_BS from Task A only ---")
    m_a = copy.deepcopy(base_model).to(device)
    m_a.site_embedding.reset()
    m_a.freeze_encoder()
    m_a.freeze_task_head()
    train_local(m_a, train_a_test, val_a_test, epochs=adapt_epochs, lr=1e-2,
                device=device, verbose=False)
    theta_bs_a = copy.deepcopy(m_a.site_embedding)

    # Evaluate on Task B with this theta_BS
    m_eval_a = TaskBModel(
        encoder=base_model.encoder,
        site_embedding=theta_bs_a,
        site_injection=base_model.site_injection,
        task_head=PowerProfileHead(),
    ).to(device)
    for p in m_eval_a.encoder.parameters():
        p.requires_grad = False
    for p in m_eval_a.site_embedding.parameters():
        p.requires_grad = False
    for p in m_eval_a.site_injection.parameters():
        p.requires_grad = False
    res_a = train_task_b(m_eval_a, train_b, val_b, epochs, lr, device,
                          save_as=f"{SAVE_PREFIX}/3-6_taskA_only_bs{test_bs}")
    results["task_a_only"] = res_a["best_val"]
    print(f"  Task A only: best_val={res_a['best_val']:.6f}")

    # --- Method 2: theta_BS from Task A + Task B jointly ---
    print("\n--- theta_BS from Task A + B jointly ---")
    m_joint = copy.deepcopy(base_model).to(device)
    m_joint.site_embedding.reset()
    m_joint.freeze_encoder()
    m_joint.freeze_task_head()

    # Alternating training: Task A and Task B batches update the same theta_BS
    joint_optimizer = torch.optim.AdamW(m_joint.site_embedding.parameters(), lr=1e-2)
    # Task B head (separate, trainable)
    task_b_head = PowerProfileHead().to(device)
    task_b_optimizer = torch.optim.AdamW(task_b_head.parameters(), lr=lr)

    for _ in range(adapt_epochs):
        m_joint.train()
        task_b_head.train()

        # Task A update (channel estimation)
        for batch in train_a_test:
            x = batch["input"].to(device)
            y = batch["target"].to(device)
            joint_optimizer.zero_grad()
            pred = m_joint(x)
            loss_a = nmse(pred, y)
            loss_a.backward()
            joint_optimizer.step()
            break  # one batch per epoch

        # Task B update (power profile, theta_BS + head)
        for batch in train_b:
            x = batch["input"].to(device)
            y = batch["target"].to(device)
            joint_optimizer.zero_grad()
            task_b_optimizer.zero_grad()
            feat = m_joint.encoder(x)
            feat = m_joint.site_injection(feat, m_joint.site_embedding.embedding)
            pred_b = task_b_head(feat)
            loss_b = power_profile_loss(pred_b, y)
            loss_b.backward()
            joint_optimizer.step()
            task_b_optimizer.step()
            break  # one batch per epoch

    theta_bs_joint = copy.deepcopy(m_joint.site_embedding)

    # Evaluate on Task B with joint theta_BS (fresh task head)
    m_eval_joint = TaskBModel(
        encoder=base_model.encoder,
        site_embedding=theta_bs_joint,
        site_injection=base_model.site_injection,
        task_head=PowerProfileHead(),
    ).to(device)
    for p in m_eval_joint.encoder.parameters():
        p.requires_grad = False
    for p in m_eval_joint.site_embedding.parameters():
        p.requires_grad = False
    for p in m_eval_joint.site_injection.parameters():
        p.requires_grad = False
    res_joint = train_task_b(m_eval_joint, train_b, val_b, epochs, lr, device,
                              save_as=f"{SAVE_PREFIX}/3-6_joint_bs{test_bs}")
    results["task_ab_joint"] = res_joint["best_val"]
    print(f"  Task A+B joint: best_val={res_joint['best_val']:.6f}")

    # --- Method 3: No theta_BS (baseline) ---
    print("\n--- No theta_BS ---")
    m_none = TaskBModel(
        encoder=base_model.encoder,
        task_head=PowerProfileHead(),
    ).to(device)
    for p in m_none.encoder.parameters():
        p.requires_grad = False
    res_none = train_task_b(m_none, train_b, val_b, epochs, lr, device,
                             save_as=f"{SAVE_PREFIX}/3-6_no_theta_bs{test_bs}")
    results["no_theta_bs"] = res_none["best_val"]
    print(f"  No theta_BS: best_val={res_none['best_val']:.6f}")

    # Save
    summary_path = SAVE_DIR / "3-6_multitask_results.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {summary_path}")
    for name, val in results.items():
        print(f"  {name}: {val:.6f}")


def launch_all(args):
    """Launch steps via subprocess, distributing variants across GPUs."""
    gpus = args.gpus or list(range(torch.cuda.device_count()))

    # Build jobs: (step, extra_args, label)
    jobs = []
    # Step 3-1: split dimensions across GPUs
    all_dims = [8, 16, 32, 64, 128, 256]
    for dim in all_dims:
        jobs.append(("3-1", ["--dims", str(dim)], f"3-1/dim{dim}"))
    # Step 3-2: split injection types across GPUs
    for inj in ["film", "concat", "add", "none"]:
        jobs.append(("3-2", ["--inject_types", inj], f"3-2/{inj}"))
    # Steps 3-3, 3-5, 3-6: single jobs (internal loops are small)
    jobs.append(("3-3", [], "3-3"))
    jobs.append(("3-5", [], "3-5"))
    jobs.append(("3-6", [], "3-6"))

    print(f"Launching {len(jobs)} jobs on {len(gpus)} GPUs")
    processes = []
    for i, (step, extra, label) in enumerate(jobs):
        gpu = gpus[i % len(gpus)]
        cmd = [
            sys.executable, "-m", "src.experiments.E3_ablation.train",
            "--step", step, "--gpu", str(gpu),
            "--epochs", str(args.epochs),
            "--lr", str(args.lr),
            "--dataset", args.dataset,
        ] + extra
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
                        choices=["3-1", "3-2", "3-3", "3-5", "3-6", "all"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--gpus", type=int, nargs="*", default=None)
    parser.add_argument("--dims", type=int, nargs="*", default=None,
                        help="For step 3-1: theta_BS dims to test (default: all)")
    parser.add_argument("--inject_types", type=str, nargs="*", default=None,
                        choices=["film", "concat", "add", "none"],
                        help="For step 3-2: injection types to test (default: all)")
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
    TEST_BS_ID = _sel_ds.test_bs_ids[0]
    SAVE_PREFIX = f"phase3/{args.dataset}"
    SAVE_DIR = Path("assets/checkpoints") / SAVE_PREFIX

    if args.step == "all":
        launch_all(args)
    elif args.step == "3-1":
        step_3_1(args.gpu, epochs=args.epochs, lr=args.lr, dims=args.dims)
    elif args.step == "3-2":
        step_3_2(args.gpu, epochs=args.epochs, lr=args.lr, inject_types=args.inject_types)
    elif args.step == "3-3":
        step_3_3(args.gpu, epochs=args.epochs, lr=args.lr)
    elif args.step == "3-5":
        step_3_5(args.gpu, epochs_pretrain=args.epochs, lr=args.lr)
    elif args.step == "3-6":
        step_3_6(args.gpu, epochs=args.epochs, lr=args.lr)
