"""Manual Federated Learning implementation.

Only 8 clients (BSs), so no framework needed — just a simple loop.
Supports:
- FedAvg: aggregate all parameters
- FedPer: aggregate only encoder, keep task_head local
- Ours (3-way): aggregate encoder + task_head + site_injection, keep site_embedding local
"""
import copy
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from src.data.utils import nmse
from .trainer import train_epoch, evaluate


def fed_avg_aggregate(models: Dict[int, torch.nn.Module], shared_key_fn=None):
    """Aggregate shared parameters via FedAvg (simple mean).

    Args:
        models: {bs_id: model}
        shared_key_fn: function(model) -> set of keys to aggregate.
                       If None, aggregate all parameters.
    Returns:
        averaged state dict (shared keys only)
    """
    bs_ids = list(models.keys())
    ref_model = models[bs_ids[0]]

    if shared_key_fn is not None:
        keys = shared_key_fn(ref_model)
    else:
        keys = set(ref_model.state_dict().keys())

    avg_state = {}
    for key in keys:
        tensors = [models[bs].state_dict()[key].float() for bs in bs_ids]
        avg_state[key] = torch.stack(tensors).mean(dim=0)

    return avg_state


def _get_shared_keys(model):
    """Determine which keys are shared based on model type."""
    if hasattr(model, "shared_state_dict"):
        return set(model.shared_state_dict().keys())
    return set(model.state_dict().keys())


def federated_train(
    model_fn,
    train_loaders: Dict[int, DataLoader],
    val_loaders: Optional[Dict[int, DataLoader]] = None,
    fl_rounds: int = 50,
    local_epochs: int = 5,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    device: str = "cuda",
    verbose: bool = True,
):
    """Full federated training loop.

    Args:
        model_fn: callable() -> model, creates a fresh model
        train_loaders: {bs_id: DataLoader}
        val_loaders: {bs_id: DataLoader} or None
        fl_rounds: number of FL communication rounds
        local_epochs: local training epochs per round
        lr: learning rate
        device: device

    Returns:
        dict with models, history
    """
    bs_ids = sorted(train_loaders.keys())

    # Initialize models per BS (same initial weights)
    init_model = model_fn().to(device)
    init_state = copy.deepcopy(init_model.state_dict())

    models = {}
    optimizers = {}
    for bs in bs_ids:
        m = model_fn().to(device)
        m.load_state_dict(init_state)
        models[bs] = m
        trainable = filter(lambda p: p.requires_grad, m.parameters())
        optimizers[bs] = torch.optim.Adam(trainable, lr=lr, weight_decay=weight_decay)

    history = {"round": [], "train_nmse": {bs: [] for bs in bs_ids},
               "val_nmse_db": {bs: [] for bs in bs_ids}}

    for rnd in range(fl_rounds):
        # Local training
        for bs in bs_ids:
            models[bs].train()
            for _ in range(local_epochs):
                train_epoch(models[bs], train_loaders[bs], optimizers[bs], device)

        # Aggregate shared parameters
        shared_keys = _get_shared_keys(models[bs_ids[0]])
        avg_state = fed_avg_aggregate(models, lambda m: shared_keys)

        # Distribute aggregated weights
        for bs in bs_ids:
            current = models[bs].state_dict()
            current.update(avg_state)
            models[bs].load_state_dict(current)

        # Evaluate
        history["round"].append(rnd)
        for bs in bs_ids:
            t_nmse, _ = evaluate(models[bs], train_loaders[bs], device)
            history["train_nmse"][bs].append(t_nmse)

            if val_loaders and bs in val_loaders:
                _, v_db = evaluate(models[bs], val_loaders[bs], device)
                history["val_nmse_db"][bs].append(v_db)
            else:
                history["val_nmse_db"][bs].append(None)

        if verbose and (rnd + 1) % 5 == 0:
            avg_db = sum(
                v for v in [history["val_nmse_db"][bs][-1] for bs in bs_ids]
                if v is not None
            ) / max(sum(1 for bs in bs_ids if history["val_nmse_db"][bs][-1] is not None), 1)
            print(f"  Round {rnd+1}/{fl_rounds}: avg_val_nmse_db={avg_db:.2f}")

    return {"models": models, "history": history}
