"""Phase 0.5: Architecture search — structure, adapter, placement, sharing.

Axes:
  1. Structure: A (split: E[3 blocks] + θ_task[2 blocks]) vs B (mono: 5 blocks, no E)
  2. Adapter type: SSF vs LoRA vs none
  3. Adapter placement (A only): E / θ_task / both.  B always = all blocks.
  4. Sharing: all_except_adapter / encoder_only (A only) / none

Usage:
    python -m src.experiments.E0_baseline.train_arch_search --config all --gpus 0 1 2 3
    python -m src.experiments.E0_baseline.train_arch_search --config A_ssf_E B_lora --gpu 0
"""
import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import SceneConfig, DatasetConfig
from src.dataset_operation.dataset import ChannelEstimationDataset
from src.models.estimator import BLOCK_REGISTRY
from src.models.adapters import make_adapter
from src.training.trainer import train_epoch, evaluate
from src.tracker import Tracker

# ── Defaults ──
PRESETS = {"uma": "munich_uma8", "umi": "munich_umi16"}
DATASET = "uma"  # default; override with --dataset umi
_scene = SceneConfig.from_preset(PRESETS[DATASET])
_ds = DatasetConfig.from_scene(_scene)
DATA_DIR = _ds.data_dir
ALL_BS = list(range(_scene.num_bs))
PRETRAIN_BS = _ds.pretrain_bs_ids
TEST_BS = _ds.test_bs_ids
SAVE_DIR = Path("assets/checkpoints/phase0_5") / DATASET
BATCH_SIZE = 512
LR = 1e-3
FL_ROUNDS = 50
LOCAL_EPOCHS = 5
LORA_RANK = 64
C = 64          # channels
N_BLOCKS = 5    # total ResBlocks (same depth for both structures)
KS = 3


# ═══════════════════════════════════════════════════════════════════════
# Unified Model
# ═══════════════════════════════════════════════════════════════════════

class Phase05Estimator(nn.Module):
    """Configurable estimator: stem → N ResBlocks (with adapters) → out.

    `split_at` divides blocks into encoder[:split_at] and task_head[split_at:].
      - Structure A (split): split_at=3 → 3 encoder + 2 task_head blocks
      - Structure B (mono):  split_at=0 → 0 encoder + 5 task_head blocks
        (i.e., NO encoder concept — everything is one network)

    Adapter placement:
      - "encoder": blocks 0..split_at-1
      - "task_head": blocks split_at..N-1
      - "both" / "all": all blocks

    Sharing strategy:
      - "all_except_adapter": share stem + all blocks (not adapters)
      - "encoder_only": share stem + blocks 0..split_at-1 only
      - "none": share nothing
    """

    def __init__(self, split_at=3, n_blocks=5, adapter_type="none",
                 adapter_target="none", channels=64, lora_rank=4,
                 kernel_size=3, block_type="resnet"):
        super().__init__()
        self._adapter_type = adapter_type
        self._split_at = split_at

        # Stem (always present)
        self.stem = nn.Sequential(
            nn.Conv2d(2, channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        # N blocks (backbone-agnostic)
        BlockClass = BLOCK_REGISTRY[block_type]
        self.blocks = nn.ModuleList(
            [BlockClass(channels, kernel_size) for _ in range(n_blocks)]
        )

        # Output conv
        self.out_conv = nn.Conv2d(
            channels, 2, kernel_size, padding=kernel_size // 2
        )

        # Determine which blocks get adapters
        self.site_adapters = nn.ModuleDict()
        if adapter_type != "none" and adapter_target != "none":
            indices = self._adapter_indices(adapter_target, split_at, n_blocks)
            for i in indices:
                self.site_adapters[str(i)] = make_adapter(
                    adapter_type, channels, kernel_size, lora_rank
                )

    @staticmethod
    def _adapter_indices(target, split_at, n_blocks):
        enc_idx = set(range(split_at))
        task_idx = set(range(split_at, n_blocks))
        if target == "encoder":
            return enc_idx
        elif target == "task_head":
            return task_idx
        elif target in ("both", "all"):
            return enc_idx | task_idx
        return set()

    def forward(self, h_ls: torch.Tensor) -> torch.Tensor:
        x = self.stem(h_ls)
        for i, block in enumerate(self.blocks):
            x = block(x)
            key = str(i)
            if key in self.site_adapters:
                a = self.site_adapters[key](x)
                x = a if self._adapter_type == "ssf" else x + a
        return h_ls + self.out_conv(x)

    # ── Parameter grouping ──

    def get_shared_keys(self, strategy: str) -> set:
        """Keys to aggregate in FL."""
        all_keys = set(self.state_dict().keys())
        adapter_keys = {k for k in all_keys if k.startswith("site_adapters.")}
        backbone = all_keys - adapter_keys

        # Encoder = stem + blocks[:split_at]
        encoder_keys = set()
        for k in backbone:
            if k.startswith("stem."):
                encoder_keys.add(k)
            elif k.startswith("blocks."):
                idx = int(k.split(".")[1])
                if idx < self._split_at:
                    encoder_keys.add(k)

        if strategy == "all_except_adapter":
            return backbone
        elif strategy == "encoder_only":
            return encoder_keys
        elif strategy == "none":
            return set()
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def n_adapter_params(self) -> int:
        return sum(p.numel() for k, p in self.named_parameters()
                   if k.startswith("site_adapters."))

    def n_local_params(self, strategy: str) -> int:
        shared = self.get_shared_keys(strategy)
        return sum(v.numel() for k, v in self.state_dict().items()
                   if k not in shared)


# ═══════════════════════════════════════════════════════════════════════
# Experiment Configurations
# ═══════════════════════════════════════════════════════════════════════
# Naming: {structure}_{adapter}_{placement}_{share}
#   structure: A (split_at=3) or B (split_at=0, no encoder)
#   adapter: ssf / lora / (none for baselines)
#   placement: E(encoder) / T(task_head) / ET(both) / (omitted for B = all)
#   share: omitted=all_except_adapter / fpE=encoder_only / ind=none

def _cfg(split_at, adapter_type, adapter_target, share, lora_rank=None,
         block_type="resnet"):
    return {
        "split_at": split_at,
        "adapter_type": adapter_type,
        "adapter_target": adapter_target,
        "share": share,
        "lora_rank": lora_rank,  # None = use global LORA_RANK default
        "block_type": block_type,
    }

CONFIGS = {}

# ══ Structure A (split_at=3): encoder[3] + task_head[2] ══

# Baselines
CONFIGS["A_fedavg"] = _cfg(3, "none", "none", "all_except_adapter")
CONFIGS["A_fedper"] = _cfg(3, "none", "none", "encoder_only")
CONFIGS["A_indep"]  = _cfg(3, "none", "none", "none")

# SSF — share E+θ_task
CONFIGS["A_ssf_E"]  = _cfg(3, "ssf", "encoder",   "all_except_adapter")
CONFIGS["A_ssf_T"]  = _cfg(3, "ssf", "task_head",  "all_except_adapter")
CONFIGS["A_ssf_ET"] = _cfg(3, "ssf", "both",       "all_except_adapter")

# LoRA — share E+θ_task
CONFIGS["A_lora_E"]  = _cfg(3, "lora", "encoder",   "all_except_adapter")
CONFIGS["A_lora_T"]  = _cfg(3, "lora", "task_head",  "all_except_adapter")
CONFIGS["A_lora_ET"] = _cfg(3, "lora", "both",       "all_except_adapter")

# SSF — share E only (FedPer-style)
CONFIGS["A_ssf_E_fpE"]  = _cfg(3, "ssf", "encoder",   "encoder_only")
CONFIGS["A_ssf_T_fpE"]  = _cfg(3, "ssf", "task_head",  "encoder_only")
CONFIGS["A_ssf_ET_fpE"] = _cfg(3, "ssf", "both",       "encoder_only")

# LoRA — share E only (FedPer-style)
CONFIGS["A_lora_E_fpE"]  = _cfg(3, "lora", "encoder",   "encoder_only")
CONFIGS["A_lora_T_fpE"]  = _cfg(3, "lora", "task_head",  "encoder_only")
CONFIGS["A_lora_ET_fpE"] = _cfg(3, "lora", "both",       "encoder_only")

# SSF/LoRA — share nothing (independent + adapter)
CONFIGS["A_ssf_ET_ind"]  = _cfg(3, "ssf",  "both", "none")
CONFIGS["A_lora_ET_ind"] = _cfg(3, "lora", "both", "none")

# ══ Structure B (split_at=0): mono[5], no encoder concept ══

# Baselines (architecturally = A baselines, listed for completeness)
CONFIGS["B_fedavg"] = _cfg(0, "none", "none", "all_except_adapter")
CONFIGS["B_indep"]  = _cfg(0, "none", "none", "none")

# SSF on all blocks
CONFIGS["B_ssf"]     = _cfg(0, "ssf", "all", "all_except_adapter")
CONFIGS["B_ssf_ind"] = _cfg(0, "ssf", "all", "none")

# LoRA on all blocks
CONFIGS["B_lora"]     = _cfg(0, "lora", "all", "all_except_adapter")
CONFIGS["B_lora_ind"] = _cfg(0, "lora", "all", "none")

# Note: B has no "encoder_only" sharing — no encoder concept.
# B_ssf (share backbone) ≈ A_ssf_ET (share E+θ_task) architecturally,
# but split_at differs → adapter indices differ (see _adapter_indices).
# With split_at=0, "encoder" = empty, "task_head" = all 5 blocks.
# With split_at=3, "encoder" = first 3, "task_head" = last 2.
# So B_ssf places adapters on blocks 0-4, A_ssf_ET also places on 0-4.
# They ARE equivalent — kept for conceptual clarity.

# ══ LoRA rank ablation (r=64) ══
# Q: LoRA r=8이 site-specific capacity 부족? r=64로 올리면?
# all_except_adapter: E/T/ET placement 비교
CONFIGS["A_lora_E_r64"]  = _cfg(3, "lora", "encoder",   "all_except_adapter", lora_rank=64)
CONFIGS["A_lora_T_r64"]  = _cfg(3, "lora", "task_head",  "all_except_adapter", lora_rank=64)
CONFIGS["A_lora_ET_r64"] = _cfg(3, "lora", "both",       "all_except_adapter", lora_rank=64)
# encoder_only: LoRA on shared encoder only (clean design)
CONFIGS["A_lora_E_fpE_r64"] = _cfg(3, "lora", "encoder", "encoder_only", lora_rank=64)

# Structure B with r=64 (fair comparison: no encoder concept)
CONFIGS["B_lora_r64"]     = _cfg(0, "lora", "all", "all_except_adapter", lora_rank=64)
CONFIGS["B_lora_ind_r64"] = _cfg(0, "lora", "all", "none", lora_rank=64)

# ══ Larger base model (C=128) ══
# Q: base model이 너무 작아서 FL이 불리한가? 키우면?
def _cfg_big(split_at, adapter_type, adapter_target, share, lora_rank=None):
    c = _cfg(split_at, adapter_type, adapter_target, share, lora_rank)
    c["channels"] = 128
    c["batch_size"] = 256
    return c

CONFIGS["A_fedavg_c128"]       = _cfg_big(3, "none", "none", "all_except_adapter")
CONFIGS["A_indep_c128"]        = _cfg_big(3, "none", "none", "none")
CONFIGS["A_lora_E_fpE_c128"]   = _cfg_big(3, "lora", "encoder", "encoder_only", lora_rank=64)
CONFIGS["B_lora_c128"]         = _cfg_big(0, "lora", "all", "all_except_adapter", lora_rank=64)

# ══ Backbone comparison (backbone-agnostic claim) ══
# For each backbone: FedAvg, Independent, Ours (LoRA+FedPer)
# se_resnet = SE attention (Hu et al.), pacenet = PSA (Yang et al., Entropy 2025), dws = depthwise-sep
for _bb, _tag in [("se_resnet", "se"), ("pacenet", "psa"), ("dws_resnet", "dws")]:
    CONFIGS[f"A_fedavg_{_tag}"]     = _cfg(3, "none", "none", "all_except_adapter", block_type=_bb)
    CONFIGS[f"A_indep_{_tag}"]      = _cfg(3, "none", "none", "none", block_type=_bb)
    CONFIGS[f"A_lora_E_fpE_{_tag}"] = _cfg(3, "lora", "encoder", "encoder_only", block_type=_bb)


# ═══════════════════════════════════════════════════════════════════════
# FL Training
# ═══════════════════════════════════════════════════════════════════════

def fl_aggregate(models: dict, shared_keys: set) -> dict:
    """FedAvg on shared keys."""
    if not shared_keys:
        return {}
    bs_ids = list(models.keys())
    avg = {}
    for key in shared_keys:
        tensors = [models[bs].state_dict()[key].float() for bs in bs_ids]
        avg[key] = torch.stack(tensors).mean(0)
    return avg


def train_config(config_name: str, gpu: int,
                 fl_rounds: int = FL_ROUNDS, local_epochs: int = LOCAL_EPOCHS,
                 lr: float = LR, dataset: str = DATASET):
    """Train one configuration with FL."""
    cfg = CONFIGS[config_name]
    device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"

    # ── Config overrides ──
    rank = cfg.get("lora_rank") or LORA_RANK
    channels = cfg.get("channels") or C
    batch_size = cfg.get("batch_size") or BATCH_SIZE
    block_type = cfg.get("block_type") or "resnet"

    tracker_config = {
        **cfg,
        "dataset": dataset,
        "fl_rounds": fl_rounds,
        "local_epochs": local_epochs,
        "lr": lr,
        "batch_size": batch_size,
        "channels": channels,
        "lora_rank": rank,
        "block_type": block_type,
        "num_bs": len(ALL_BS),
    }

    with Tracker(
        f"E0/{dataset}/{config_name}",
        config=tracker_config,
        capture_output=True,
        purpose="Architecture search: structure × adapter × placement × sharing strategy",
        variables={
            "independent": ["structure (A/B)", "adapter (ssf/lora/none)",
                            "placement (E/T/ET)", "sharing (all/encoder_only/none)",
                            "backbone (resnet/se_resnet/pacenet_psa/dws)", "dataset (uma/umi)"],
            "dependent": ["avg_nmse_db", "per-BS nmse_db (pretrain vs test)"],
            "controlled": ["fl_rounds=50", "local_epochs=5", "lr=1e-3", "C=64"],
        },
        hypothesis=f"LoRA on encoder + encoder_only sharing (A_lora_E_fpE) achieves best test BS NMSE on {dataset}",
        eval_criteria="Compare avg_test_bs NMSE across all configs; best = lowest dB",
    ) as run:
        structure = "A(split)" if cfg["split_at"] > 0 else "B(mono)"
        print(f"\n{'='*60}")
        print(f"Config: {config_name} [{structure}] dataset={dataset}")
        print(f"  split_at={cfg['split_at']}, adapter={cfg['adapter_type']}, "
              f"target={cfg['adapter_target']}, share={cfg['share']}")
        print(f"{'='*60}")

        # ── Data ──
        train_loaders, val_loaders = {}, {}
        for bs_id in ALL_BS:
            ds = ChannelEstimationDataset(
                data_dir=DATA_DIR, bs_ids=[bs_id], snr_range_db=(0, 30)
            )
            n_val = max(int(len(ds) * 0.2), 1)
            t_ds, v_ds = torch.utils.data.random_split(ds, [len(ds) - n_val, n_val])
            train_loaders[bs_id] = DataLoader(t_ds, batch_size=batch_size, shuffle=True)
            val_loaders[bs_id] = DataLoader(v_ds, batch_size=batch_size)

        # ── Model ──

        def model_fn():
            return Phase05Estimator(
                split_at=cfg["split_at"],
                n_blocks=N_BLOCKS,
                adapter_type=cfg["adapter_type"],
                adapter_target=cfg["adapter_target"],
                channels=channels,
                lora_rank=rank,
                kernel_size=KS,
                block_type=block_type,
            )

        init_model = model_fn().to(device)
        shared_keys = init_model.get_shared_keys(cfg["share"])
        n_total = sum(p.numel() for p in init_model.parameters())
        n_adapter = init_model.n_adapter_params()
        n_local = init_model.n_local_params(cfg["share"])

        print(f"  Params: total={n_total}, adapter={n_adapter}, "
              f"local={n_local}, shared={n_total - n_local}")
        print(f"  Shared keys: {len(shared_keys)} / {len(init_model.state_dict())}")
        adapter_blocks = sorted(int(k) for k in init_model.site_adapters.keys())
        print(f"  Adapter on blocks: {adapter_blocks if adapter_blocks else 'none'}")
        print(f"  Channels: {channels}, Batch size: {batch_size}")

        init_state = copy.deepcopy(init_model.state_dict())
        del init_model

        models, optimizers = {}, {}
        for bs in ALL_BS:
            m = model_fn().to(device)
            m.load_state_dict(init_state)
            models[bs] = m
            optimizers[bs] = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-5)

        # ── FL Loop ──
        history = {
            "round": [],
            "train_nmse": {bs: [] for bs in ALL_BS},
            "val_nmse_db": {bs: [] for bs in ALL_BS},
        }

        # Early stopping (conservative)
        ES_PATIENCE = 15
        ES_MIN_ROUNDS = 20
        best_avg_db = float("inf")
        best_round = 0
        best_states = None

        for rnd in range(fl_rounds):
            for bs in ALL_BS:
                models[bs].train()
                for _ in range(local_epochs):
                    train_epoch(models[bs], train_loaders[bs], optimizers[bs], device)

            if shared_keys:
                avg_state = fl_aggregate(models, shared_keys)
                for bs in ALL_BS:
                    state = models[bs].state_dict()
                    state.update(avg_state)
                    models[bs].load_state_dict(state)

            history["round"].append(rnd)
            avg_val_db_list = []
            for bs in ALL_BS:
                t_nmse, _ = evaluate(models[bs], train_loaders[bs], device)
                history["train_nmse"][bs].append(t_nmse)
                _, v_db = evaluate(models[bs], val_loaders[bs], device)
                history["val_nmse_db"][bs].append(v_db)
                avg_val_db_list.append(v_db)

            avg_db = np.mean(avg_val_db_list)
            avg_train = np.mean([history["train_nmse"][bs][-1] for bs in ALL_BS])

            # Track best (lower dB = better)
            if avg_db < best_avg_db:
                best_avg_db = avg_db
                best_round = rnd
                best_states = {bs: {k: v.clone() for k, v in models[bs].state_dict().items()} for bs in ALL_BS}

            log_entry = {
                "round": rnd,
                "avg_val_nmse_db": avg_db,
                "avg_train_nmse": avg_train,
                "best_avg_db": best_avg_db,
            }
            # Add per-BS val (for pretrain/test split tracking)
            for bs in PRETRAIN_BS:
                log_entry[f"val_db_bs{bs}"] = history["val_nmse_db"][bs][-1]
            for bs in TEST_BS:
                log_entry[f"val_db_test_bs{bs}"] = history["val_nmse_db"][bs][-1]
            run.log(**log_entry)

            if (rnd + 1) % 5 == 0:
                print(f"  Round {rnd+1}/{fl_rounds}: avg_val={avg_db:.2f} dB (best={best_avg_db:.2f} @r{best_round})")

            # Early stop check
            if rnd >= ES_MIN_ROUNDS and (rnd - best_round) >= ES_PATIENCE:
                print(f"  Early stop at round {rnd+1} (best was round {best_round+1})")
                run.log_message(f"Early stopped at round {rnd+1}, best round {best_round+1}")
                break

        # Restore best model states
        if best_states is not None:
            for bs in ALL_BS:
                models[bs].load_state_dict(best_states[bs])

        # ── Evaluate best model ──
        final_db = {}
        for bs in ALL_BS:
            _, v_db = evaluate(models[bs], val_loaders[bs], device)
            final_db[str(bs)] = v_db

        # ── Save ──
        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        for bs in ALL_BS:
            torch.save(models[bs].state_dict(), SAVE_DIR / f"{config_name}_bs{bs}.pt")

        total_rounds = len(history["round"])
        result = {
            "config_name": config_name,
            "config": cfg,
            "n_total_params": n_total,
            "n_adapter_params": n_adapter,
            "n_local_params": n_local,
            "n_shared_params": n_total - n_local,
            "adapter_blocks": adapter_blocks,
            "final_nmse_db": final_db,
            "avg_nmse_db": float(np.mean(list(final_db.values()))),
            "avg_pretrain_bs": float(np.mean([final_db[str(bs)] for bs in PRETRAIN_BS])),
            "avg_test_bs": float(np.mean([final_db[str(bs)] for bs in TEST_BS])),
            "best_round": best_round,
            "total_rounds": total_rounds,
            "early_stopped": total_rounds < fl_rounds,
            "history": {
                "round": history["round"],
                "val_nmse_db": {str(k): v for k, v in history["val_nmse_db"].items()},
            },
        }
        with open(SAVE_DIR / f"{config_name}_result.json", "w") as f:
            json.dump(result, f, indent=2)

        run.set_result(
            avg_nmse_db=result["avg_nmse_db"],
            avg_pretrain_bs=result["avg_pretrain_bs"],
            avg_test_bs=result["avg_test_bs"],
        )

        print(f"\n  Final: avg={result['avg_nmse_db']:.2f} dB, "
              f"pretrain={result['avg_pretrain_bs']:.2f}, "
              f"test={result['avg_test_bs']:.2f}")
        per_bs = [f"{final_db[str(bs)]:.1f}" for bs in ALL_BS]
        print(f"  Per-BS: {per_bs}")

    return result


# ═══════════════════════════════════════════════════════════════════════
# Multi-GPU
# ═══════════════════════════════════════════════════════════════════════

def launch_parallel(config_names: list, gpus: list, fl_rounds: int,
                    local_epochs: int, lr: float, dataset: str = DATASET):
    """Launch in batches — max 1 job per GPU at a time."""
    n_gpus = len(gpus)
    failed = []

    for batch_start in range(0, len(config_names), n_gpus):
        batch = config_names[batch_start:batch_start + n_gpus]
        batch_num = batch_start // n_gpus + 1
        print(f"\n--- Batch {batch_num}/{-(-len(config_names)//n_gpus)}: "
              f"{batch} ---")

        processes = []
        for i, name in enumerate(batch):
            gpu = gpus[i]
            cmd = [
                sys.executable, "-m", "src.experiments.E0_baseline.train_arch_search",
                "--config", name, "--gpu", str(gpu),
                "--fl_rounds", str(fl_rounds),
                "--local_epochs", str(local_epochs),
                "--lr", str(lr),
                "--dataset", dataset,
            ]
            print(f"  {name} -> GPU {gpu}")
            p = subprocess.Popen(cmd)
            processes.append((name, gpu, p))

        for name, gpu, p in processes:
            ret = p.wait()
            status = "OK" if ret == 0 else f"FAILED (code={ret})"
            print(f"  {name} (GPU{gpu}): {status}")
            if ret != 0:
                failed.append(name)

    if failed:
        print(f"\nFailed: {failed}")
    else:
        print(f"\nAll {len(config_names)} configs completed.")


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 0.5: Architecture search")
    parser.add_argument("--config", type=str, nargs="+", required=True,
                        help="Config name(s) or 'all'")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--gpus", type=int, nargs="*", default=None)
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
    ALL_BS = list(range(_sel_scene.num_bs))
    PRETRAIN_BS = _sel_ds.pretrain_bs_ids
    TEST_BS = _sel_ds.test_bs_ids
    SAVE_DIR = Path("assets/checkpoints/phase0_5") / args.dataset

    if args.config == ["all"]:
        config_names = list(CONFIGS.keys())
    else:
        for name in args.config:
            if name not in CONFIGS:
                parser.error(f"Unknown: {name}. Available: {list(CONFIGS.keys())}")
        config_names = args.config

    print(f"Configs to run ({len(config_names)}): {config_names}")

    if args.gpus and len(config_names) > 1:
        launch_parallel(config_names, args.gpus, args.fl_rounds,
                        args.local_epochs, args.lr, args.dataset)
    else:
        for name in config_names:
            train_config(name, args.gpu, args.fl_rounds,
                         args.local_epochs, args.lr, args.dataset)
