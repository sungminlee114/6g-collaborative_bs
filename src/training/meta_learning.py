"""MAML-based meta-learning for site adaptation.

Each "task" = one BS's channel estimation data.
Goal: learn initialization that adapts quickly to new BS with few samples.

Uses torch.func.functional_call for correct gradient flow in MAML.
"""
import copy
from typing import Dict, Optional, Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.utils import nmse


def _functional_forward(model, params, x):
    """Forward pass with external parameters using functional_call."""
    return torch.func.functional_call(model, params, (x,))


def _inner_loop(model, params, support_batch, inner_lr, inner_steps, device):
    """First-order MAML inner loop: gradient descent on support set.

    Returns adapted parameter dict (detached from meta-graph for FO-MAML).
    """
    x = support_batch["input"].to(device)
    y = support_batch["target"].to(device)

    for _ in range(inner_steps):
        y_hat = _functional_forward(model, params, x)
        loss = nmse(y_hat, y)
        grads = torch.autograd.grad(loss, params.values())

        # First-order MAML: detach gradients (no second-order)
        params = {
            name: p - inner_lr * g.detach()
            for (name, p), g in zip(params.items(), grads)
        }

    return params


def maml_train(
    model_fn: Callable,
    task_loaders: Dict[int, DataLoader],
    val_loaders: Optional[Dict[int, DataLoader]] = None,
    outer_lr: float = 1e-3,
    inner_lr: float = 0.01,
    inner_steps: int = 5,
    tasks_per_batch: int = 4,
    meta_epochs: int = 100,
    device: str = "cuda",
    verbose: bool = True,
):
    """First-order MAML training loop.

    Each task = one BS's data. Support set for inner loop, query set for outer loss.
    """
    meta_model = model_fn().to(device)
    meta_optimizer = torch.optim.Adam(meta_model.parameters(), lr=outer_lr)

    bs_ids = list(task_loaders.keys())
    history = {"epoch": [], "meta_loss": [], "val_nmse_db": []}

    for epoch in range(meta_epochs):
        meta_model.train()
        meta_optimizer.zero_grad()

        # Sample tasks
        task_sample = torch.randperm(len(bs_ids))[:tasks_per_batch].tolist()
        sampled_bs = [bs_ids[i] for i in task_sample]

        total_outer_loss = 0.0
        n_tasks = 0

        for bs_id in sampled_bs:
            loader = task_loaders[bs_id]
            batches = list(loader)
            if len(batches) < 2:
                continue

            support_batch = batches[0]
            query_batch = batches[1]

            # Start from meta-model parameters
            params = {name: p.clone() for name, p in meta_model.named_parameters()}

            # Inner loop: adapt to this task
            adapted_params = _inner_loop(
                meta_model, params, support_batch, inner_lr, inner_steps, device
            )

            # Outer loss on query set with adapted params
            # For FO-MAML: we compute loss with adapted params and backprop
            # through the last adaptation step only
            qx = query_batch["input"].to(device)
            qy = query_batch["target"].to(device)
            q_pred = _functional_forward(meta_model, adapted_params, qx)
            query_loss = nmse(q_pred, qy)

            # Accumulate gradients on meta-model parameters
            # FO-MAML: gradient of outer loss w.r.t. adapted_params,
            # then transfer to meta_params (since adapted = meta - lr*grad, they share grad)
            meta_grads = torch.autograd.grad(query_loss, meta_model.parameters(),
                                              allow_unused=True)
            for p, g in zip(meta_model.parameters(), meta_grads):
                if g is not None:
                    if p.grad is None:
                        p.grad = g / tasks_per_batch
                    else:
                        p.grad += g / tasks_per_batch

            total_outer_loss += query_loss.item()
            n_tasks += 1

        if n_tasks > 0:
            meta_optimizer.step()

        avg_loss = total_outer_loss / max(n_tasks, 1)
        history["epoch"].append(epoch)
        history["meta_loss"].append(avg_loss)

        # Validation
        if val_loaders and (epoch + 1) % 10 == 0:
            val_results = evaluate_maml(
                meta_model, val_loaders, inner_lr, inner_steps, device
            )
            avg_val_db = sum(val_results.values()) / len(val_results) if val_results else 0
            history["val_nmse_db"].append(avg_val_db)
            if verbose:
                print(f"  Epoch {epoch+1}: meta_loss={avg_loss:.6f}, val_nmse_db={avg_val_db:.2f}")
        else:
            history["val_nmse_db"].append(None)

    return {"meta_model": meta_model, "history": history}


def evaluate_maml(model, val_loaders, inner_lr, inner_steps, device):
    """Evaluate MAML: adapt to each val task, then measure performance."""
    results = {}

    for bs_id, loader in val_loaders.items():
        batches = list(loader)
        if len(batches) < 2:
            continue

        # Adapt on first batch (needs gradients)
        model.train()
        params = {name: p.clone().detach().requires_grad_(True)
                  for name, p in model.named_parameters()}
        with torch.enable_grad():
            adapted_params = _inner_loop(
                model, params, batches[0], inner_lr, inner_steps, device
            )

        # Evaluate on remaining batches (no gradients needed)
        model.eval()
        total_nmse = 0.0
        n = 0
        with torch.no_grad():
            for batch in batches[1:]:
                x = batch["input"].to(device)
                y = batch["target"].to(device)
                # Detach adapted params for inference
                detached_params = {k: v.detach() for k, v in adapted_params.items()}
                y_hat = _functional_forward(model, detached_params, x)
                total_nmse += nmse(y_hat, y).item() * x.size(0)
                n += x.size(0)

        if n > 0:
            avg = total_nmse / n
            results[bs_id] = 10 * torch.log10(torch.tensor(avg)).item()

    return results


def adapt_to_new_site(meta_model, support_loader, inner_lr=0.01,
                      inner_steps=10, device="cuda"):
    """Adapt meta-learned model to a new site using support data.

    Returns adapted model (copy).
    """
    adapted = copy.deepcopy(meta_model).to(device)
    optimizer = torch.optim.SGD(adapted.parameters(), lr=inner_lr)

    adapted.train()
    for step in range(inner_steps):
        for batch in support_loader:
            x = batch["input"].to(device)
            y = batch["target"].to(device)
            optimizer.zero_grad()
            y_hat = adapted(x)
            loss = nmse(y_hat, y)
            loss.backward()
            optimizer.step()

    return adapted
