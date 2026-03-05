from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from pathlib import Path


@dataclass
class SceneConfig:
    """Sionna RT scene configuration."""
    frequency: float = 15e9          # 15 GHz (FR3)
    bandwidth: float = 400e6         # 400 MHz
    num_subcarriers: int = 1024
    guard_band_ratio: float = 0.1    # 10% guard band
    temperature: float = 293.0       # 20°C

    # Antenna config
    tx_rows: int = 2
    tx_cols: int = 2
    rx_rows: int = 1
    rx_cols: int = 1
    tx_polarization: str = "V"
    rx_polarization: str = "cross"

    # BS config
    num_bs: int = 8
    power_dbm: float = 40.0
    bs_positions: List[Tuple[float, float, float]] = field(default_factory=lambda: [
        [-14.111, -164.816, 89.889],
        [-22.006, -57.554, 79.995],
        [144.688, -216.653, 53.074],
        [216.867, 352.080, 40.000],
        [-356.179, 177.271, 27.432],
        [-538.142, -184.078, 18.713],
        [-407.187, -144.209, 18.778],
        [-324.530, 288.101, 16.572],
    ])
    bs_orientations: List[List[float]] = field(default_factory=lambda: [
        [],
        [],
        [],
        [],
        [-336.518, 172.528, 1.500],
        [-546.988, -193.206, 1.500],
        [-404.938, -133.550, 1.500],
        [-313.634, 269.198, 1.500],
        # [404.941, -114.083, 1.500],  # BS9 not used (num_bs=8)
    ])

    # PathSolver config
    max_depth: int = 5
    max_num_paths_per_src: int = 100_000
    samples_per_src: int = 100_000

    # UE sampling config
    num_ue: int = 100
    sinr_min_db: float = 2.0
    sinr_max_db: float = 40.0
    dist_min: float = 10.0
    dist_max: float = 100.0

    # UE device diversity: list of (rx_rows, rx_cols, polarization) configs
    # Each UE is randomly assigned one of these device types
    ue_device_types: List[tuple] = field(default_factory=lambda: [
        (1, 1, "cross"),       # 1×1 cross-pol = 2 ant elements (basic phone)
        (1, 2, "V"),           # 1×2 V-pol = 2 ant elements (mid-range)
        (2, 2, "V"),           # 2×2 V-pol = 4 ant elements (flagship)
    ])

    @property
    def num_tx_ant(self) -> int:
        return self.tx_rows * self.tx_cols

    @property
    def num_rx_ant(self) -> int:
        """Cross-pol doubles the antenna count."""
        if self.rx_polarization == "cross":
            return self.rx_rows * self.rx_cols * 2
        return self.rx_rows * self.rx_cols

    @property
    def effective_bandwidth(self) -> float:
        return self.bandwidth * (1 - self.guard_band_ratio)

    @property
    def subcarrier_spacing(self) -> float:
        return self.effective_bandwidth / self.num_subcarriers


@dataclass
class DatasetConfig:
    """Dataset generation & loading config."""
    data_dir: str = "data/channels"
    num_snapshots: int = 100
    seed_offset: int = 0

    # Channel estimation
    pilot_ratio: float = 1.0        # 1.0 = full pilot (denoising only)
    snr_range_db: Tuple[float, float] = (0.0, 30.0)

    # Train/val/test split (by BS)
    pretrain_bs_ids: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5])
    test_bs_ids: List[int] = field(default_factory=lambda: [6, 7])


@dataclass
class ModelConfig:
    """Model architecture config."""
    # Encoder
    encoder_channels: List[int] = field(default_factory=lambda: [64, 64, 64])
    kernel_size: int = 3

    # Site embedding
    site_embed_dim: int = 64
    site_integration: str = "film"  # "film", "concat", "add", "none"

    # Task head
    task_head_channels: List[int] = field(default_factory=lambda: [64, 64])

    # Input/output dims (derived from SceneConfig)
    n_rx_ant: int = 2
    n_tx_ant: int = 4
    n_subcarriers: int = 1024


@dataclass
class TrainConfig:
    """Training config."""
    # General
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 100
    patience: int = 15

    # FL
    fl_rounds: int = 50
    local_epochs: int = 5
    fl_lr: float = 1e-3

    # MAML
    maml_inner_lr: float = 0.01
    maml_outer_lr: float = 1e-3
    maml_inner_steps: int = 5
    maml_tasks_per_batch: int = 4

    # Device
    device: str = "cuda"

    # Logging
    log_dir: str = "logs"
    save_dir: str = "checkpoints"
