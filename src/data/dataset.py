"""PyTorch Dataset for channel estimation with per-BS grouping."""
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from .utils import complex_to_real, add_awgn


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

        self.meta = meta.reset_index(drop=True)

        # Pre-load all channels into memory for speed
        self._cache = {}

    def _load_snapshot(self, snapshot_id: int) -> np.ndarray:
        """Load and cache CFR for a snapshot."""
        if snapshot_id not in self._cache:
            path = self.data_dir / f"snapshot_{snapshot_id:04d}" / "channels.npz"
            data = np.load(path)
            self._cache[snapshot_id] = data["cfr"]  # (N_UE, n_rx_ant, n_tx_ant, n_sc)
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

        sample = {
            "input": torch.from_numpy(h_noisy),     # (2, 8, 1024)
            "target": torch.from_numpy(h_real),      # (2, 8, 1024)
            "bs_id": bs_id,
            "snr_db": snr_db,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


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
