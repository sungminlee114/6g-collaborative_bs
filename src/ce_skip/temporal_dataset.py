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
        max_snapshots: Optional[int] = None,
        bs_ids: Optional[List[int]] = None,
    ):
        self.data_dir = Path(data_dir)
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
        """Load trajectory and BS metadata."""
        bs_info_path = self.data_dir / "bs_info.json"
        if bs_info_path.exists():
            with open(bs_info_path) as f:
                self.bs_info = json.load(f)
        else:
            self.bs_info = {}

        traj_info_path = self.data_dir / "trajectory_info.json"
        if traj_info_path.exists():
            with open(traj_info_path) as f:
                self.traj_info = json.load(f)
        else:
            self.traj_info = {}

        # Load UE metadata (speeds, positions)
        ue_meta_path = self.data_dir / "ue_meta.json"
        if ue_meta_path.exists():
            with open(ue_meta_path) as f:
                self.ue_meta = json.load(f)
        else:
            self.ue_meta = []

        # Load trajectories if available
        traj_path = self.data_dir / "trajectories.npz"
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
        series = []
        for t in range(t_start, t_end):
            cfr = self._load_snapshot(t)
            h = cfr[ue_id]  # (n_rx, n_tx, n_sc)
            n_rx, n_tx, n_sc = h.shape
            series.append(h.reshape(n_rx * n_tx, n_sc))
        return np.stack(series, axis=0)  # (T, n_ant, n_sc)

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

        series_list = []
        for uid in ue_ids:
            s = self.get_ue_series(uid, bs_id, snap_range)
            series_list.append(s)

        if not series_list:
            return np.array([]), ue_ids
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
