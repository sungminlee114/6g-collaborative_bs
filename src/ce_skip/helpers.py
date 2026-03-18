"""Shared helpers for CE skip experiments — memory-safe UE iteration."""
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch

from src.ce_skip.temporal_dataset import TemporalChannelData


def iter_ue_tensors(
    data: TemporalChannelData,
    bs_id: int,
    device: str = "cuda",
    max_snapshots: int = 200,
    max_ue: int = 20,
):
    """Iterate over UEs, yielding one (h_real, ue_id, distance, speed) at a time on GPU.

    Memory-safe: only one UE's data on GPU at a time.

    Yields:
        h_real: (T, 2, n_ant, n_sc) float32 tensor on device
        ue_id: int
        distance: float (UE-BS distance)
        speed: float (UE speed m/s)
    """
    T_max = min(data.num_snapshots, max_snapshots)

    # Get UE list for this BS
    if data.ue_bs_ids is not None:
        ue_ids = [i for i, b in enumerate(data.ue_bs_ids) if b == bs_id]
    else:
        cfr0 = data._load_snapshot(0)
        ue_ids = list(range(cfr0.shape[0]))

    for idx, uid in enumerate(ue_ids[:max_ue]):
        # Load one UE's time-series on CPU
        h_complex = data.get_ue_series(uid, bs_id, snap_range=(0, T_max))
        # (T, n_ant, n_sc) complex → (T, 2, n_ant, n_sc) float32
        h_real = np.stack([h_complex.real, h_complex.imag], axis=1).astype(np.float32)
        h_tensor = torch.from_numpy(h_real).to(device)

        dist = data.get_ue_distance(uid, bs_id, snap_idx=0)
        speed = data.get_ue_speed(uid)

        yield h_tensor, uid, dist, speed

        # Free GPU memory
        del h_tensor
