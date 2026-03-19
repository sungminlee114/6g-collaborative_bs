"""CE skip scheduling — event-triggered CE inference scheduling framework."""
from dataclasses import dataclass
from pathlib import Path
from typing import List

import yaml

from src.config import SceneConfig, CONFIGS_DIR

# ─── Shared experiment config ──────────────────────────────────────
# All S0-S7 experiments use this list. Change here to re-run on different presets.
PRIMARY_PRESETS = [
    "munich_elaa_m_1k_15g",   # Config A: 24×24, 15 GHz, R_ray=10.2m (NF primary)
    "munich_elaa_m_1k_28g",   # Config B: 24×24, 28 GHz, R_ray=5.5m  (NF secondary)
    "munich_5g_mimo_3g5",     # Config C: 8×8,  3.5 GHz, R_ray=2.7m  (5G FF baseline)
]


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
    def temporal_dir(self) -> Path:
        """Temporal trajectory data directory (convention: {data_dir}_temporal)."""
        return Path(self.data_dir).parent / f"{Path(self.data_dir).name}_temporal"

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
