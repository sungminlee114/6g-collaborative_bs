"""PyTorch Dataset for channel estimation with per-BS grouping."""
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from .utils import complex_to_real, add_awgn, load_cfr_from_npz


def _detect_subcarrier_params(data_dir: Path):
    """Detect (num_subcarriers, subcarrier_spacing) from dataset metadata."""
    progress = data_dir / "progress.json"
    if progress.exists():
        import json
        with open(progress) as f:
            preset = json.load(f).get("preset")
        if preset:
            from src.config import SceneConfig
            cfg = SceneConfig.from_preset(preset)
            return cfg.num_subcarriers, cfg.subcarrier_spacing
    # Fallback: peek at first snapshot to infer from CFR shape
    return 1024, 0.0


class ChannelEstimationDataset(Dataset):
    """Dataset for channel estimation task.

    Loads CFR data, adds AWGN noise to create (H_LS, H_true) pairs.
    Groups data by BS for federated/site-specific training.
    """

    def __init__(
        self,
        data_dir: str,
        bs_ids: Optional[List[int]] = None,
        snapshot_ids: Optional[List[int]] = None,
        ue_ids: Optional[List[int]] = None,
        snr_range_db: Tuple[float, float] = (0.0, 30.0),
        fixed_snr_db: Optional[float] = None,
        transform=None,
    ):
        self.data_dir = Path(data_dir)
        self.snr_range_db = snr_range_db
        self.fixed_snr_db = fixed_snr_db
        self.transform = transform

        # Load metadata
        meta = pd.read_parquet(self.data_dir / "metadata.parquet")

        # Filter by BS
        if bs_ids is not None:
            meta = meta[meta["bs_id"].isin(bs_ids)]

        # Filter by snapshot
        if snapshot_ids is not None:
            meta = meta[meta["snapshot_id"].isin(snapshot_ids)]

        # Filter by UE
        if ue_ids is not None:
            meta = meta[meta["ue_id"].isin(ue_ids)]

        # Pre-load all channels into memory for speed
        self._cache = {}

        # Filter out NaN samples.
        # Sionna RT produces NaN in CFR when a UE-BS pair has no valid propagation
        # paths (complete NLOS blockage). cfr() with normalize=True divides by
        # total path energy, which is 0 → NaN. Mainly affects high-altitude BSs
        # (BS0-3) where some UEs are fully shadowed. ~3% of total samples.
        valid_mask = []
        for _, row in meta.iterrows():
            cfr_all = self._load_snapshot(int(row["snapshot_id"]))
            cfr_ue = cfr_all[int(row["ue_id"])]
            valid_mask.append(not np.any(np.isnan(cfr_ue)))
        n_before = len(meta)
        meta = meta[valid_mask]
        n_after = len(meta)
        if n_before != n_after:
            print(f"  Filtered {n_before - n_after}/{n_before} NaN samples")

        self.meta = meta.reset_index(drop=True)

    def _get_subcarrier_params(self):
        """Lazily detect num_subcarriers and subcarrier_spacing from preset."""
        if not hasattr(self, "_sc_params"):
            self._sc_params = _detect_subcarrier_params(self.data_dir)
        return self._sc_params

    def _load_snapshot(self, snapshot_id: int) -> np.ndarray:
        """Load and cache CFR for a snapshot."""
        if snapshot_id not in self._cache:
            path = self.data_dir / f"snapshot_{snapshot_id:04d}" / "channels.npz"
            n_sc, sc_spacing = self._get_subcarrier_params()
            self._cache[snapshot_id] = load_cfr_from_npz(path, n_sc, sc_spacing)
        return self._cache[snapshot_id]

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        snap_id = int(row["snapshot_id"])
        ue_id = int(row["ue_id"])
        bs_id = int(row["bs_id"])

        cfr_all = self._load_snapshot(snap_id)
        cfr = cfr_all[ue_id]  # (n_rx_ant, n_tx_ant, n_sc) complex

        # Flatten antenna dims
        n_rx, n_tx, n_sc = cfr.shape
        h_flat = cfr.reshape(n_rx * n_tx, n_sc)  # (8, 1024) complex

        # Complex → real: (2, 8, 1024)
        h_real = complex_to_real(h_flat)

        # Random or fixed SNR
        if self.fixed_snr_db is not None:
            snr_db = self.fixed_snr_db
        else:
            snr_db = np.random.uniform(*self.snr_range_db)

        h_noisy = add_awgn(h_real, snr_db)

        # Per-sample RMS normalization (prevents loss explosion for weak channels)
        target_t = torch.from_numpy(h_real)
        rms = target_t.pow(2).mean().sqrt().clamp(min=1e-10)
        target_t = target_t / rms
        input_t = torch.from_numpy(h_noisy) / rms

        sample = {
            "input": input_t,
            "target": target_t,
            "bs_id": bs_id,
            "snr_db": snr_db,
            "rms": rms,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


class TemporalCEDataset(Dataset):
    """DL-CE dataset from temporal channel data with UE-level split.

    Each sample = one (t, ue_id) snapshot → (H_LS, H_true) pair.

    Supports two modes:
    - From pre-loaded h_real tensors (ue_tensors): zero-copy, no h5 reload
    - From TemporalChannelData (temporal_data): loads from h5

    Args:
        ue_tensors: list of (T, 2, n_ant, n_sc) tensors (from load_all_ues)
        temporal_data: TemporalChannelData instance (fallback if ue_tensors=None)
        ue_ids: list of UE IDs (used with temporal_data)
        snap_range: (start, end) slot range. None = all snapshots
        snr_range_db: random SNR range for AWGN noise
        fixed_snr_db: fixed SNR (overrides range)
    """

    def __init__(
        self,
        temporal_data=None,
        ue_ids: List[int] = None,
        bs_id: int = 1,
        snap_range: Optional[Tuple[int, int]] = None,
        snr_range_db: Tuple[float, float] = (0.0, 30.0),
        fixed_snr_db: Optional[float] = None,
        ue_tensors: list = None,
    ):
        self.snr_range_db = snr_range_db
        self.fixed_snr_db = fixed_snr_db

        self._ue_data = []  # list of (T_range, 2, n_ant, n_sc) numpy arrays
        self._index = []    # list of (ue_local_idx, t_local)

        if ue_tensors is not None:
            # Zero-copy from pre-loaded tensors
            for ue_idx, h_real in enumerate(ue_tensors):
                # h_real: torch.Tensor (T, 2, n_ant, n_sc) or numpy
                if hasattr(h_real, 'numpy'):
                    h = h_real.numpy()
                else:
                    h = h_real
                t_start = snap_range[0] if snap_range else 0
                t_end = snap_range[1] if snap_range else h.shape[0]
                h_slice = h[t_start:t_end]
                self._ue_data.append(h_slice)
                for t in range(h_slice.shape[0]):
                    self._index.append((ue_idx, t))
        elif temporal_data is not None and ue_ids is not None:
            # Load from h5
            T = temporal_data.num_snapshots
            t_start = snap_range[0] if snap_range else 0
            t_end = snap_range[1] if snap_range else T
            for ue_idx, uid in enumerate(ue_ids):
                h_complex = temporal_data.get_ue_series(
                    uid, bs_id, snap_range=(t_start, t_end))
                h_real = np.stack([h_complex.real, h_complex.imag], axis=1).astype(np.float32)
                self._ue_data.append(h_real)
                for t in range(h_real.shape[0]):
                    self._index.append((ue_idx, t))
                del h_complex
        else:
            raise ValueError("Provide ue_tensors or (temporal_data + ue_ids)")

        ram_gb = sum(h.nbytes for h in self._ue_data) / 1e9
        print(f"  TemporalCEDataset: {len(self._ue_data)} UEs, "
              f"{len(self._index)} samples, {ram_gb:.1f} GB")

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        ue_idx, t = self._index[idx]
        h_real = self._ue_data[ue_idx][t]  # (2, n_ant, n_sc)

        if self.fixed_snr_db is not None:
            snr_db = self.fixed_snr_db
        else:
            snr_db = np.random.uniform(*self.snr_range_db)

        h_noisy = add_awgn(h_real, snr_db)

        # Per-sample RMS normalization (same as S1/train_dl_ce.py)
        target_t = torch.from_numpy(h_real)
        rms = target_t.pow(2).mean().sqrt().clamp(min=1e-10)

        return {
            "input": torch.from_numpy(h_noisy) / rms,
            "target": target_t / rms,
            "snr_db": snr_db,
        }


class PerBSDataLoader:
    """Creates separate DataLoaders per BS for federated training."""

    def __init__(
        self,
        data_dir: str,
        bs_ids: List[int],
        batch_size: int = 32,
        snr_range_db: Tuple[float, float] = (0.0, 30.0),
        snapshot_ids: Optional[List[int]] = None,
        num_workers: int = 0,
    ):
        self.loaders = {}
        for bs_id in bs_ids:
            ds = ChannelEstimationDataset(
                data_dir=data_dir,
                bs_ids=[bs_id],
                snapshot_ids=snapshot_ids,
                snr_range_db=snr_range_db,
            )
            self.loaders[bs_id] = DataLoader(
                ds, batch_size=batch_size, shuffle=True,
                num_workers=num_workers, drop_last=False,
            )

    def __getitem__(self, bs_id: int) -> DataLoader:
        return self.loaders[bs_id]

    def items(self):
        return self.loaders.items()

    def __len__(self):
        return len(self.loaders)
