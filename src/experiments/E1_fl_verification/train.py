"""Phase 1: 3-way structure verification training scripts.

Step 1-1: FL comparison (3-way vs FedPer vs FedAvg vs Independent)
Step 1-2: Few-shot adaptation (k-shot on unseen BS)
Step 1-3: Pre-trained E vs from-scratch convergence

Usage:
    # Step 1-1: FL comparison (all methods on one GPU)
    python -m src.experiments.E1_fl_verification.train --step 1-1 --gpu 0

    # Step 1-1: single method on specific GPU
    python -m src.experiments.E1_fl_verification.train --step 1-1 --method 3way --gpu 2

    # Step 1-2: Few-shot adaptation
    python -m src.experiments.E1_fl_verification.train --step 1-2 --gpu 0

    # Step 1-3: Pre-trained vs from-scratch
    python -m src.experiments.E1_fl_verification.train --step 1-3 --gpu 0

    # Run all steps in parallel (1-1 methods parallel, then 1-2/1-3 parallel)
    python -m src.experiments.E1_fl_verification.train --step all --gpus 0 1 2
"""
import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset_operation.dataset import ChannelEstimationDataset
from src.dataset_operation.utils import nmse
from src.models.baselines import PlainEstimator, FedPerEstimator
from src.models.estimator import SiteAwareEstimator, create_model
from src.training.trainer import (
    train_local, evaluate, evaluate_per_snr, save_checkpoint, load_checkpoint,
)
from src.training.federated import federated_train
from src.training.meta_learning import maml_train, adapt_to_new_site

# ── Defaults (override with --dataset uma/umi) ──
from src.config import SceneConfig, DatasetConfig
PRESETS = {"uma": "munich_uma8", "umi": "munich_umi16"}
DATASET = "uma"
_scene = SceneConfig.from_preset(PRESETS[DATASET])
_ds = DatasetConfig.from_scene(_scene)
DATA_DIR = _ds.data_dir
PRETRAIN_BS = _ds.pretrain_bs_ids
TEST_BS = _ds.test_bs_ids
SAVE_PREFIX = f"phase1/{DATASET}"
SAVE_DIR = Path("assets/checkpoints") / SAVE_PREFIX
BATCH_SIZE = 512
LR = 1e-3
EVAL_SNRS = [0, 5, 10, 15, 20, 25, 30]
FL_ROUNDS = 50
LOCAL_EPOCHS = 5
FEWSHOT_EPOCHS = 50
FEWSHOT_K_SHOTS = [5, 10, 20, 50, 100, 200]
FEWSHOT_N_REPEATS = 5
MAML_INNER_LR = 0.001
MAML_INNER_STEPS = 5
MAML_META_EPOCHS = 100
STEP_1_3_EPOCHS = 100


def step_1_1(gpu: int, fl_rounds: int = FL_ROUNDS, local_epochs: int = LOCAL_EPOCHS,
             lr: float = LR, method: str = None):
    """Step 1-1: FL comparison — 3-way vs FedPer vs FedAvg vs Independent.

    Args:
        method: If specified, run only this method. If None, run all sequentially.
    """
    device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
    all_bs = PRETRAIN_BS + TEST_BS

    print("=== Step 1-1: FL Comparison ===")

    # Build per-BS data loaders
    train_loaders = {}
    val_loaders = {}
    for bs_id in all_bs:
        ds = ChannelEstimationDataset(data_dir=DATA_DIR, bs_ids=[bs_id], snr_range_db=(0, 30))
        n_val = max(int(len(ds) * 0.2), 1)
        n_train = len(ds) - n_val
        t_ds, v_ds = torch.utils.data.random_split(ds, [n_train, n_val])
        train_loaders[bs_id] = DataLoader(t_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loaders[bs_id] = DataLoader(v_ds, batch_size=BATCH_SIZE)

    all_methods = {
        "fedavg": lambda: PlainEstimator(encoder_channels=64, encoder_blocks=3),
        "fedper": lambda: FedPerEstimator(encoder_channels=64, encoder_blocks=3),
        "3way": lambda: create_model(site_integration="film", site_embed_dim=64),
    }

    if method is not None:
        methods = {method: all_methods[method]}
    else:
        methods = all_methods

    for name, model_fn in methods.items():
        print(f"\n--- {name} ---")
        result = federated_train(
            model_fn=model_fn,
            train_loaders=train_loaders,
            val_loaders=val_loaders,
            fl_rounds=fl_rounds,
            local_epochs=local_epochs,
            lr=lr,
            device=device,
        )

        # Save each BS model
        for bs_id, model in result["models"].items():
            save_checkpoint(model, f"{SAVE_PREFIX}/1-1_{name}_bs{bs_id}", meta={
                "method": name,
                "bs_id": bs_id,
                "fl_rounds": fl_rounds,
                "local_epochs": local_epochs,
            })

        # Save FL history
        history_path = SAVE_DIR / f"1-1_{name}_history.json"
        history_path.parent.mkdir(parents=True, exist_ok=True)
        serializable = {
            "round": result["history"]["round"],
            "train_nmse": {str(k): v for k, v in result["history"]["train_nmse"].items()},
            "val_nmse_db": {str(k): v for k, v in result["history"]["val_nmse_db"].items()},
        }
        with open(history_path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"  Saved history: {history_path}")

        # Per-BS SNR=20 evaluation
        for bs_id, model in result["models"].items():
            snr20 = evaluate_per_snr(
                model, ChannelEstimationDataset, DATA_DIR, [bs_id],
                [20.0], batch_size=BATCH_SIZE, device=device,
            )
            print(f"  {name} BS{bs_id} @ SNR=20: {snr20[20.0]:.2f} dB")


def step_1_2(gpu: int, k_shots: list = None, n_repeats: int = FEWSHOT_N_REPEATS, lr: float = LR):
    """Step 1-2: Few-shot adaptation on unseen BS."""
    device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
    if k_shots is None:
        k_shots = FEWSHOT_K_SHOTS

    print("=== Step 1-2: Few-Shot Adaptation ===")

    # Load pre-trained 3-way model from step 1-1 (use any pretrain BS as reference)
    ref_model = create_model(site_integration="film", site_embed_dim=64)
    try:
        load_checkpoint(ref_model, f"{SAVE_PREFIX}/1-1_3way_bs{PRETRAIN_BS[0]}", device=device)
    except FileNotFoundError:
        print("ERROR: Run step 1-1 first to get pre-trained models")
        return

    # Meta-train MAML on pretrain BSs
    print("\nMeta-training MAML on pretrain BSs...")
    maml_loaders = {}
    for bs_id in PRETRAIN_BS:
        ds = ChannelEstimationDataset(data_dir=DATA_DIR, bs_ids=[bs_id], snr_range_db=(0, 30))
        maml_loaders[bs_id] = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    maml_result = maml_train(
        model_fn=lambda: PlainEstimator(encoder_channels=64, encoder_blocks=3),
        task_loaders=maml_loaders,
        outer_lr=lr, inner_lr=MAML_INNER_LR, inner_steps=MAML_INNER_STEPS,
        tasks_per_batch=min(4, len(PRETRAIN_BS)),
        meta_epochs=MAML_META_EPOCHS, device=device,
    )
    meta_model = maml_result["meta_model"]
    print("MAML meta-training done.")

    results = {}  # {method: {k: [nmse_db_per_repeat]}}

    for test_bs in TEST_BS:
        print(f"\n--- Test BS{test_bs} ---")
        # Full dataset for this BS
        full_ds = ChannelEstimationDataset(
            data_dir=DATA_DIR, bs_ids=[test_bs], snr_range_db=(0, 30),
        )

        for k in k_shots:
            for method in ["theta_bs_only", "finetune_all", "from_scratch", "maml"]:
                key = f"bs{test_bs}_{method}"
                if key not in results:
                    results[key] = {}
                if k not in results[key]:
                    results[key][k] = []

                for repeat in range(n_repeats):
                    torch.manual_seed(repeat * 100 + k)
                    # Sample k training examples
                    indices = torch.randperm(len(full_ds))[:k].tolist()
                    k_ds = torch.utils.data.Subset(full_ds, indices)
                    rest_indices = [i for i in range(len(full_ds)) if i not in indices]
                    eval_ds = torch.utils.data.Subset(full_ds, rest_indices[:200])
                    k_loader = DataLoader(k_ds, batch_size=min(k, BATCH_SIZE), shuffle=True)
                    eval_loader = DataLoader(eval_ds, batch_size=BATCH_SIZE)

                    if method == "theta_bs_only":
                        model = create_model(site_integration="film", site_embed_dim=64).to(device)
                        model.load_shared_state_dict(ref_model.shared_state_dict())
                        model.freeze_encoder()
                        model.freeze_task_head()
                        # Only theta_BS trainable (64 params) — low overfitting risk
                        train_local(model, k_loader, epochs=FEWSHOT_EPOCHS, lr=lr, device=device, verbose=False)

                    elif method == "finetune_all":
                        model = create_model(site_integration="film", site_embed_dim=64).to(device)
                        model.load_shared_state_dict(ref_model.shared_state_dict())
                        model.unfreeze_all()
                        train_local(model, k_loader, epochs=FEWSHOT_EPOCHS, lr=lr * 0.1, device=device, verbose=False)

                    elif method == "from_scratch":
                        model = PlainEstimator(encoder_channels=64, encoder_blocks=3).to(device)
                        train_local(model, k_loader, epochs=FEWSHOT_EPOCHS, lr=lr, device=device, verbose=False)

                    elif method == "maml":
                        model = adapt_to_new_site(
                            meta_model, k_loader,
                            inner_lr=MAML_INNER_LR, inner_steps=MAML_INNER_STEPS * 2, device=device,
                        )

                    # Evaluate on held-out set
                    nmse_val, nmse_db = evaluate(model, eval_loader, device)
                    results[key][k].append(nmse_db)

                avg = np.mean(results[key][k])
                std = np.std(results[key][k])
                print(f"  k={k:3d}, {method:15s}: {avg:.2f} +/- {std:.2f} dB")

    # Save results
    results_path = SAVE_DIR / "1-2_fewshot_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {results_path}")


def step_1_3(gpu: int, epochs: int = STEP_1_3_EPOCHS, lr: float = LR):
    """Step 1-3: Pre-trained E vs from-scratch convergence comparison."""
    device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"

    print("=== Step 1-3: Pre-trained E vs From-Scratch ===")

    ref_model = create_model(site_integration="film", site_embed_dim=64)
    try:
        load_checkpoint(ref_model, f"{SAVE_PREFIX}/1-1_3way_bs{PRETRAIN_BS[0]}", device=device)
    except FileNotFoundError:
        print("ERROR: Run step 1-1 first")
        return

    for test_bs in TEST_BS:
        print(f"\n--- Test BS{test_bs} ---")
        ds = ChannelEstimationDataset(data_dir=DATA_DIR, bs_ids=[test_bs], snr_range_db=(0, 30))
        n_val = max(int(len(ds) * 0.2), 1)
        t_ds, v_ds = torch.utils.data.random_split(ds, [len(ds) - n_val, n_val])
        train_loader = DataLoader(t_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(v_ds, batch_size=BATCH_SIZE)

        methods = {}

        # Pre-trained: load shared weights, adapt theta_BS only
        m_pre = create_model(site_integration="film", site_embed_dim=64).to(device)
        m_pre.load_shared_state_dict(ref_model.shared_state_dict())
        m_pre.freeze_encoder()
        m_pre.freeze_task_head()
        res_pre = train_local(m_pre, train_loader, val_loader, epochs=epochs, lr=lr,
                              device=device, save_as=f"{SAVE_PREFIX}/1-3_pretrained_bs{test_bs}")
        methods["pretrained"] = res_pre

        # From scratch
        m_scratch = PlainEstimator(encoder_channels=64, encoder_blocks=3).to(device)
        res_scratch = train_local(m_scratch, train_loader, val_loader, epochs=epochs, lr=lr,
                                  device=device, save_as=f"{SAVE_PREFIX}/1-3_scratch_bs{test_bs}")
        methods["from_scratch"] = res_scratch

        for name, res in methods.items():
            best_db = 10 * np.log10(res["best_val"])
            print(f"  {name}: {best_db:.2f} dB (epoch {res['best_epoch']})")


def launch_all(args):
    """Launch steps via subprocess. Step 1-1 methods in parallel, then 1-2 and 1-3."""
    gpus = args.gpus or list(range(torch.cuda.device_count()))

    def make_cmd(step, gpu, extra=None):
        cmd = [
            sys.executable, "-m", "src.experiments.E1_fl_verification.train",
            "--step", step, "--gpu", str(gpu),
            "--fl_rounds", str(args.fl_rounds),
            "--local_epochs", str(args.local_epochs),
            "--lr", str(args.lr),
            "--dataset", args.dataset,
        ]
        if extra:
            cmd.extend(extra)
        return cmd

    # Step 1-1: run 3 FL methods in parallel on different GPUs
    fl_methods = ["fedavg", "fedper", "3way"]
    print(f"=== Step 1-1: {len(fl_methods)} methods on {len(gpus)} GPUs ===")
    processes = []
    for i, method in enumerate(fl_methods):
        gpu = gpus[i % len(gpus)]
        print(f"  {method} -> GPU{gpu}")
        p = subprocess.Popen(make_cmd("1-1", gpu, ["--method", method]))
        processes.append((f"1-1/{method}", gpu, p))

    failed = False
    for label, gpu, p in processes:
        ret = p.wait()
        status = "OK" if ret == 0 else f"FAILED (code={ret})"
        print(f"  {label} (GPU{gpu}): {status}")
        if ret != 0:
            failed = True

    if failed:
        print("Step 1-1 had failures, aborting.")
        return

    print("Step 1-1: ALL OK")

    # Steps 1-2 and 1-3 in parallel
    dependent_steps = ["1-2", "1-3"]
    processes = []
    for i, step in enumerate(dependent_steps):
        gpu = gpus[i % len(gpus)]
        print(f"  Step {step} -> GPU{gpu}")
        p = subprocess.Popen(make_cmd(step, gpu))
        processes.append((step, gpu, p))

    for step, gpu, p in processes:
        ret = p.wait()
        status = "OK" if ret == 0 else f"FAILED (code={ret})"
        print(f"Step {step} (GPU{gpu}): {status}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=str, required=True,
                        choices=["1-1", "1-2", "1-3", "all"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--gpus", type=int, nargs="*", default=None, help="GPU indices (default: all)")
    parser.add_argument("--method", type=str, default=None,
                        choices=["fedavg", "fedper", "3way"],
                        help="For step 1-1: run only this method (used by launch_all)")
    parser.add_argument("--fl_rounds", type=int, default=FL_ROUNDS)
    parser.add_argument("--local_epochs", type=int, default=LOCAL_EPOCHS)
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
    SAVE_PREFIX = f"phase1/{args.dataset}"
    SAVE_DIR = Path("assets/checkpoints") / SAVE_PREFIX

    if args.step == "all":
        launch_all(args)
    elif args.step == "1-1":
        step_1_1(args.gpu, fl_rounds=args.fl_rounds, local_epochs=args.local_epochs,
                 lr=args.lr, method=args.method)
    elif args.step == "1-2":
        step_1_2(args.gpu, lr=args.lr)
    elif args.step == "1-3":
        step_1_3(args.gpu, lr=args.lr)
