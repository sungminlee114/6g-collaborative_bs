"""Temporal channel dataset for CE skip scheduling experiments.

Loads consecutive snapshots from temporal trajectory data, providing
time-series channel data per (UE, BS) pair.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import json
import numpy as np
import torch

from src.dataset_operation.utils import load_cfr_from_npz


def _load_cir_from_snapshot(args):
    """Load CIR from a single snapshot (module-level for ProcessPoolExecutor pickling)."""
    data_dir, t, ue_id = args
    import os
    path = os.path.join(data_dir, f"snapshot_{t:04d}", "channels.npz")
    data = np.load(path)
    if "cfr" in data.files:
        return None
    a = data["cir_a"]
    tau = data["cir_tau"]
    if ue_id is not None:
        a = a[ue_id:ue_id+1]
        tau = tau[ue_id:ue_id+1]
    return a, tau


class TemporalChannelData:
    """Loads temporal channel data as contiguous time-series per (UE, BS).

    Unlike ChannelEstimationDataset (which shuffles i.i.d. samples),
    this preserves temporal ordering for skip scheduling evaluation.

    Data format:
        snapshot_XXXX/channels.npz → cfr: (N_UE, n_rx, n_tx, n_sc) complex64

    Usage:
        data = TemporalChannelData("assets/data/channels_elaa_l_1k_15g_temporal")
        h_series = data.get_ue_series(ue_id=5, bs_id=0)
        # → (T, n_ant, n_sc) complex64
    """

    def __init__(
        self,
        data_dir: str | Path,
        trajectory_dir: str | Path = None,
        max_snapshots: Optional[int] = None,
        bs_ids: Optional[List[int]] = None,
    ):
        self.data_dir = Path(data_dir)
        self.trajectory_dir = Path(trajectory_dir) if trajectory_dir else self.data_dir
        self._bs_ids = bs_ids

        # Load metadata
        self._load_metadata()

        # Discover snapshots
        snap_dirs = sorted(self.data_dir.glob("snapshot_*"))
        if max_snapshots is not None:
            snap_dirs = snap_dirs[:max_snapshots]
        self.num_snapshots = len(snap_dirs)
        assert self.num_snapshots > 0, f"No snapshots found in {data_dir}"

        # Load all channel data: (T, N_UE, n_rx, n_tx, n_sc)
        self._cfr_cache: Dict[int, np.ndarray] = {}

    def _load_metadata(self):
        """Load trajectory (from trajectory_dir) and BS metadata (from data_dir)."""
        # BS info lives with channels
        bs_info_path = self.data_dir / "bs_info.json"
        if bs_info_path.exists():
            with open(bs_info_path) as f:
                self.bs_info = json.load(f)
        else:
            self.bs_info = {}

        # Trajectory info, UE meta, and positions live in trajectory_dir (shared)
        traj_info_path = self.trajectory_dir / "trajectory_info.json"
        if traj_info_path.exists():
            with open(traj_info_path) as f:
                self.traj_info = json.load(f)
        else:
            self.traj_info = {}

        ue_meta_path = self.trajectory_dir / "ue_meta.json"
        if ue_meta_path.exists():
            with open(ue_meta_path) as f:
                self.ue_meta = json.load(f)
        else:
            self.ue_meta = []

        traj_path = self.trajectory_dir / "trajectories.npz"
        if traj_path.exists():
            traj = np.load(traj_path)
            self.positions = traj["positions"]   # (T, N_UE, 3)
            self.speeds = traj["speeds"]         # (N_UE,)
            self.ue_bs_ids = traj["bs_ids"]      # (N_UE,) serving BS per UE
        else:
            self.positions = None
            self.speeds = None
            self.ue_bs_ids = None

    def _get_subcarrier_params(self):
        """Lazily detect num_subcarriers and subcarrier_spacing from preset."""
        if not hasattr(self, "_sc_params"):
            progress = self.data_dir / "progress.json"
            if progress.exists():
                with open(progress) as f:
                    preset = json.load(f).get("preset")
                if preset:
                    from src.config import SceneConfig
                    cfg = SceneConfig.from_preset(preset)
                    self._sc_params = (cfg.num_subcarriers, cfg.subcarrier_spacing)
                    return self._sc_params
            self._sc_params = (1024, 0.0)
        return self._sc_params

    def _load_snapshot(self, snap_idx: int) -> np.ndarray:
        """Load CFR for a snapshot. Returns (N_UE, n_rx, n_tx, n_sc) complex."""
        if snap_idx not in self._cfr_cache:
            path = self.data_dir / f"snapshot_{snap_idx:04d}" / "channels.npz"
            n_sc, sc_spacing = self._get_subcarrier_params()
            self._cfr_cache[snap_idx] = load_cfr_from_npz(path, n_sc, sc_spacing)
        return self._cfr_cache[snap_idx]

    def _batch_cir_to_cfr(self, snap_indices: List[int], ue_id: int = None):
        """Load CIR from multiple snapshots and batch-convert to CFR on GPU.

        Much faster than per-snapshot conversion: one GPU kernel for all snapshots.
        If ue_id is given, only extracts that UE (saves memory).
        """
        import torch
        n_sc, sc_spacing = self._get_subcarrier_params()
        if sc_spacing <= 0:
            # Fallback to per-snapshot loading (CFR already stored)
            return None

        # Read all CIR data from disk — multiprocess (npz decompression is CPU-bound)
        from concurrent.futures import ProcessPoolExecutor

        data_dir_str = str(self.data_dir)
        n_workers = min(32, len(snap_indices))
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            results = list(pool.map(
                _load_cir_from_snapshot,
                [(data_dir_str, t, ue_id) for t in snap_indices],
            ))

        if results[0] is None:
            return None

        all_a = [r[0] for r in results]
        all_tau = [r[1] for r in results]

        # Pad to max n_paths (varies slightly across snapshots)
        max_paths = max(a.shape[-1] for a in all_a)
        for i in range(len(all_a)):
            if all_a[i].shape[-1] < max_paths:
                pad_width = [(0, 0)] * (all_a[i].ndim - 1) + [(0, max_paths - all_a[i].shape[-1])]
                all_a[i] = np.pad(all_a[i], pad_width)
                all_tau[i] = np.pad(all_tau[i], [(0, 0), (0, max_paths - all_tau[i].shape[-1])])

        stacked_a = np.stack(all_a, axis=0)
        stacked_tau = np.stack(all_tau, axis=0)
        T, N, n_rx, n_tx, n_paths = stacked_a.shape

        # Flatten T*N for GPU processing
        flat_a = stacked_a.reshape(T * N, n_rx, n_tx, n_paths)
        flat_tau = stacked_tau.reshape(T * N, n_paths)

        # Subcarrier frequencies
        if n_sc % 2 == 0:
            start = -n_sc // 2
        else:
            start = -(n_sc - 1) // 2
        freqs = np.arange(start, start + n_sc, dtype=np.float32) * sc_spacing

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        freqs_t = torch.tensor(freqs, dtype=torch.float32, device=device)

        # GPU einsum loop: avoids huge (T*N, rx, tx, paths, sc) intermediate tensor.
        # Per-sample: exp_phase (paths, sc) + einsum 'rtp,ps->rts' → (rx, tx, sc)
        # Memory: O(n_paths * n_sc) per sample, not O(rx * tx * paths * sc)
        a_gpu = torch.tensor(flat_a, dtype=torch.complex64, device=device)
        tau_gpu = torch.tensor(flat_tau, dtype=torch.float32, device=device)

        cfr_list = []
        for i in range(T * N):
            exp_phase = torch.exp(-2j * torch.pi * tau_gpu[i, :, None] * freqs_t[None, :])
            cfr_list.append(torch.einsum('rtp,ps->rts', a_gpu[i], exp_phase))

        cfr_flat = torch.stack(cfr_list).cpu().numpy()
        del a_gpu, tau_gpu
        return cfr_flat.reshape(T, N, n_rx, n_tx, n_sc)

    def get_ue_series(
        self,
        ue_id: int,
        bs_id: int,
        snap_range: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """Get channel time-series for a single (UE, BS) pair.

        Returns: (T, n_ant, n_sc) complex64 where n_ant = n_rx * n_tx
        """
        t_start, t_end = snap_range or (0, self.num_snapshots)
        snap_indices = list(range(t_start, t_end))

        # Try batch CIR→CFR (fast GPU path)
        batch_cfr = self._batch_cir_to_cfr(snap_indices, ue_id=ue_id)
        if batch_cfr is not None:
            # batch_cfr: (T, 1, n_rx, n_tx, n_sc) — squeeze UE dim
            h = batch_cfr[:, 0]  # (T, n_rx, n_tx, n_sc)
            n_rx, n_tx, n_sc = h.shape[1], h.shape[2], h.shape[3]
            return h.reshape(len(snap_indices), n_rx * n_tx, n_sc)

        # Fallback: per-snapshot (CFR already stored)
        series = []
        for t in snap_indices:
            cfr = self._load_snapshot(t)
            h = cfr[ue_id]
            n_rx, n_tx, n_sc = h.shape
            series.append(h.reshape(n_rx * n_tx, n_sc))
        return np.stack(series, axis=0)

    def get_all_series(
        self,
        bs_id: int,
        snap_range: Optional[Tuple[int, int]] = None,
    ) -> Tuple[np.ndarray, List[int]]:
        """Get channel series for all UEs served by a given BS.

        Returns:
            h_all: (N_UE_bs, T, n_ant, n_sc) complex64
            ue_ids: list of UE IDs served by this BS
        """
        if self.ue_bs_ids is not None:
            ue_ids = [i for i, b in enumerate(self.ue_bs_ids) if b == bs_id]
        else:
            # Fallback: load first snapshot and use all UEs
            cfr0 = self._load_snapshot(0)
            ue_ids = list(range(cfr0.shape[0]))

        if not ue_ids:
            return np.array([]), ue_ids

        t_start, t_end = snap_range or (0, self.num_snapshots)
        snap_indices = list(range(t_start, t_end))

        # Try batch CIR→CFR for ALL UEs at once (one GPU kernel, one disk pass)
        batch_cfr = self._batch_cir_to_cfr(snap_indices, ue_id=None)
        if batch_cfr is not None:
            # batch_cfr: (T, N_UE_all, n_rx, n_tx, n_sc)
            h_selected = batch_cfr[:, ue_ids]  # (T, N_UE_bs, n_rx, n_tx, n_sc)
            n_rx, n_tx, n_sc = h_selected.shape[2], h_selected.shape[3], h_selected.shape[4]
            T = len(snap_indices)
            N = len(ue_ids)
            # Reshape to (N_UE_bs, T, n_ant, n_sc)
            h_all = h_selected.transpose(1, 0, 2, 3, 4).reshape(N, T, n_rx * n_tx, n_sc)
            return h_all, ue_ids

        # Fallback: per-UE loading (CFR already stored on disk)
        series_list = []
        for uid in ue_ids:
            s = self.get_ue_series(uid, bs_id, snap_range)
            series_list.append(s)

        return np.stack(series_list, axis=0), ue_ids

    def get_ue_distance(self, ue_id: int, bs_id: int, snap_idx: int = 0) -> float:
        """Get UE-BS distance at a given snapshot."""
        if self.positions is None:
            return float("nan")
        positions = self.bs_info.get("positions", self.bs_info.get("bs_positions", []))
        if bs_id >= len(positions):
            return float("nan")
        bs_pos = np.array(positions[bs_id])
        ue_pos = self.positions[snap_idx, ue_id]
        return float(np.linalg.norm(ue_pos - bs_pos))

    def get_ue_speed(self, ue_id: int) -> float:
        """Get UE speed in m/s."""
        if self.speeds is not None:
            return float(self.speeds[ue_id])
        if self.ue_meta:
            return float(self.ue_meta[ue_id].get("speed", 0.0))
        return 0.0

    @property
    def dt_s(self) -> float:
        """Time step in seconds."""
        return self.traj_info.get("dt_s", self.traj_info.get("dt_ms", 10) / 1000)

    @property
    def num_ue(self) -> int:
        if self.ue_bs_ids is not None:
            return len(self.ue_bs_ids)
        cfr0 = self._load_snapshot(0)
        return cfr0.shape[0]

    def clear_cache(self):
        """Free cached snapshot data."""
        self._cfr_cache.clear()
