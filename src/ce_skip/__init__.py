"""CE skip scheduling — event-triggered CE inference scheduling framework."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml

from src.config import SceneConfig, CONFIGS_DIR

# ─── Shared experiment config ──────────────────────────────────────
# All S0-S7 experiments use this list. Change here to re-run on different presets.
PRIMARY_PRESETS = [
    "munich_elaa_m_1k_15g",   # Config A: 24×24, 15 GHz, R_ray=10.2m (NF primary)
    "munich_elaa_m_1k_28g",   # Config B: 24×24, 28 GHz, R_ray=5.5m  (NF secondary)
    "munich_5g_mimo_3g5",     # Config C: 8×8,  3.5 GHz, R_ray=2.7m  (5G FF baseline)
]

# Shared trajectory directory — all presets use the same UE positions/trajectories
# for fair cross-config comparison. Generated once with the most conservative preset
# (28 GHz, smallest coverage) so all presets have valid channels.
SHARED_TRAJECTORY_DIR = Path("assets/data/shared_trajectories")


@dataclass
class ExperimentConfig:
    """Combined scene + dataset config for CE skip experiments.

    Loads both SceneConfig fields and dataset fields (data_dir, BS splits)
    from the same YAML preset.
    """
    scene: SceneConfig
    data_dir: str
    train_bs_ids: List[int]
    val_bs_ids: List[int]
    test_bs_ids: List[int]

    @classmethod
    def from_preset(cls, name: str) -> "ExperimentConfig":
        path = CONFIGS_DIR / f"{name}.yaml"
        with open(path) as f:
            raw = yaml.safe_load(f)

        scene = SceneConfig.from_preset(name)
        return cls(
            scene=scene,
            data_dir=raw.get("data_dir", ""),
            train_bs_ids=raw.get("train_bs_ids", [0, 5]),
            val_bs_ids=raw.get("val_bs_ids", [2, 4]),
            test_bs_ids=raw.get("test_bs_ids", [1, 3, 6, 7]),
        )

    @property
    def _project_root(self) -> Path:
        """Project root (where assets/ lives)."""
        return Path(__file__).resolve().parent.parent.parent

    @property
    def temporal_dir(self) -> Path:
        """Temporal channel data directory (per-preset, channels only)."""
        p = Path(self.data_dir).parent / f"{Path(self.data_dir).name}_temporal"
        return self._project_root / p

    @property
    def trajectory_dir(self) -> Path:
        """Shared trajectory directory (same UE paths across all presets)."""
        return self._project_root / SHARED_TRAJECTORY_DIR

    # Delegate common SceneConfig properties
    @property
    def frequency(self) -> float:
        return self.scene.frequency

    @property
    def num_tx_ant(self) -> int:
        return self.scene.num_tx_ant

    @property
    def num_rx_ant(self) -> int:
        return self.scene.num_rx_ant

    @property
    def num_subcarriers(self) -> int:
        return self.scene.num_subcarriers

    @property
    def num_bs(self) -> int:
        return self.scene.num_bs

    @property
    def tx_rows(self) -> int:
        return self.scene.tx_rows

    @property
    def tx_cols(self) -> int:
        return self.scene.tx_cols

    @property
    def deployment(self) -> str:
        return self.scene.deployment

    @property
    def numerology_mu(self) -> int:
        return self.scene.numerology_mu

    @property
    def scs(self) -> float:
        return self.scene.scs

    @property
    def slot_duration_s(self) -> float:
        return self.scene.slot_duration_s

    def ue_split(self, train: int = 3, val: int = 1, test: int = 6,
                 seed: int = 42) -> Dict[str, List[int]]:
        """Stratified UE split by speed group.

        Per speed group: train/val/test = 3/1/6 (default, 10 UEs per speed).
        Deterministic with seed for reproducibility.

        Returns:
            {"train": [ue_ids], "val": [ue_ids], "test": [ue_ids]}
        """
        traj_path = self.trajectory_dir / "trajectories.npz"
        traj = np.load(traj_path)
        speeds = traj["speeds"]  # (N_UE,)
        n_ue = len(speeds)

        total = train + val + test
        rng = np.random.default_rng(seed)

        # Group UE IDs by speed
        speed_groups = {}
        for uid in range(n_ue):
            spd = round(float(speeds[uid]), 1)
            speed_groups.setdefault(spd, []).append(uid)

        train_ids, val_ids, test_ids = [], [], []
        for spd in sorted(speed_groups.keys()):
            uids = sorted(speed_groups[spd])
            rng.shuffle(uids)
            n = len(uids)
            n_train = round(n * train / total)
            n_val = round(n * val / total)
            # Remainder goes to test
            train_ids.extend(uids[:n_train])
            val_ids.extend(uids[n_train:n_train + n_val])
            test_ids.extend(uids[n_train + n_val:])

        return {"train": sorted(train_ids),
                "val": sorted(val_ids),
                "test": sorted(test_ids)}
