from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from pathlib import Path

import yaml

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "assets" / "configs"
DEFAULT_PRESET = "munich_umi16"


@dataclass
class SceneConfig:
    """Sionna RT scene configuration — all values come from YAML.

    Usage:
        cfg = SceneConfig.from_preset("munich_umi16")
        cfg = SceneConfig.from_yaml("assets/configs/custom.yaml")
        cfg = SceneConfig.from_preset("munich_umi16", frequency=28e9)  # override
    """
    # Scene metadata
    scene_name: str = ""
    deployment: str = ""

    # RF
    frequency: float = 0.0
    bandwidth: float = 0.0
    num_subcarriers: int = 0
    guard_band_ratio: float = 0.0
    temperature: float = 0.0

    # Antenna config
    tx_rows: int = 0
    tx_cols: int = 0
    rx_rows: int = 0
    rx_cols: int = 0
    tx_polarization: str = ""
    rx_polarization: str = ""

    # BS config
    num_bs: int = 0
    power_dbm: float = 0.0
    bs_positions: List[Tuple[float, float, float]] = field(default_factory=list)
    bs_orientations: List[List[float]] = field(default_factory=list)

    # PathSolver config
    max_depth: int = 0
    max_num_paths_per_src: int = 0
    samples_per_src: int = 0

    # UE sampling config
    num_ue: int = 0
    sinr_min_db: float = 0.0
    sinr_max_db: float = 0.0
    dist_min: float = 0.0
    dist_max: float = 0.0

    # UE device diversity
    ue_device_types: List[tuple] = field(default_factory=list)

    def __post_init__(self):
        # YAML safe_load parses scientific notation (e.g. 15.0e9) as str → coerce
        for fname in ("frequency", "bandwidth", "temperature", "power_dbm",
                       "guard_band_ratio", "sinr_min_db", "sinr_max_db",
                       "dist_min", "dist_max"):
            val = getattr(self, fname, None)
            if isinstance(val, str):
                setattr(self, fname, float(val))
        for fname in ("num_subcarriers", "num_bs", "num_ue", "max_depth",
                       "max_num_paths_per_src", "samples_per_src",
                       "tx_rows", "tx_cols", "rx_rows", "rx_cols"):
            val = getattr(self, fname, None)
            if isinstance(val, str):
                setattr(self, fname, int(val))
        if len(self.bs_orientations) < self.num_bs:
            self.bs_orientations = [[] for _ in range(self.num_bs)]
        self.ue_device_types = [tuple(d) for d in self.ue_device_types]

    @classmethod
    def from_yaml(cls, path: str | Path, **overrides) -> "SceneConfig":
        """Load from a YAML file, with optional field overrides."""
        with open(Path(path)) as f:
            data = yaml.safe_load(f)
        data.update(overrides)
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})

    @classmethod
    def from_preset(cls, name: str = DEFAULT_PRESET, **overrides) -> "SceneConfig":
        """Load a preset by name (filename without .yaml in assets/configs/)."""
        path = CONFIGS_DIR / f"{name}.yaml"
        if not path.exists():
            avail = [p.stem for p in CONFIGS_DIR.glob("*.yaml")]
            raise FileNotFoundError(
                f"Preset '{name}' not found. Available: {avail}"
            )
        return cls.from_yaml(path, **overrides)

    @classmethod
    def list_presets(cls) -> List[str]:
        """List all available preset names."""
        return sorted(p.stem for p in CONFIGS_DIR.glob("*.yaml"))

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
    """Dataset generation & loading config — dataset fields from scene YAML."""
    data_dir: str = ""
    num_snapshots: int = 0
    seed_offset: int = 0

    # Channel estimation
    pilot_ratio: float = 1.0
    snr_range_db: Tuple[float, float] = (0.0, 30.0)

    # Train/val/test split (by BS)
    pretrain_bs_ids: List[int] = field(default_factory=list)
    test_bs_ids: List[int] = field(default_factory=list)

    @classmethod
    def from_scene(cls, scene: SceneConfig, **overrides) -> "DatasetConfig":
        """Derive dataset config from the same YAML that defined the scene."""
        path = CONFIGS_DIR / f"{scene.scene_name}_{scene.deployment}{scene.num_bs}.yaml"
        data = {}
        if path.exists():
            with open(path) as f:
                data = yaml.safe_load(f)
        ds_keys = {f.name for f in cls.__dataclass_fields__.values()}
        ds_fields = {k: v for k, v in data.items() if k in ds_keys}
        ds_fields.update(overrides)
        return cls(**ds_fields)


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
    """Training config — single source of truth for all training scripts."""
    # General
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 80
    patience: int = 25

    # Phase 0: per-BS independent training
    phase0_batch_size: int = 700
    phase0_epochs: int = 80

    # FL (Phase 1-1)
    fl_rounds: int = 50
    local_epochs: int = 5

    # Few-shot (Phase 1-2)
    fewshot_epochs: int = 50
    fewshot_k_shots: List[int] = field(default_factory=lambda: [5, 10, 20, 50, 100, 200])
    fewshot_n_repeats: int = 5

    # MAML
    maml_inner_lr: float = 0.01
    maml_outer_lr: float = 1e-3
    maml_inner_steps: int = 5
    maml_tasks_per_batch: int = 4
    maml_meta_epochs: int = 100

    # Phase 2
    phase2_epochs: int = 100

    # Phase 3 ablation
    phase3_epochs: int = 80
    phase3_adapt_epochs: int = 50

    # Eval
    eval_snrs: List[float] = field(default_factory=lambda: [0, 5, 10, 15, 20, 25, 30])

    # Device
    device: str = "cuda"

    # Logging & output
    log_dir: str = "assets/logs"
    save_dir: str = "assets/checkpoints"
    plots_dir: str = "assets/plots"
    results_dir: str = "assets/results"


def get_plot_dir(experiment: str) -> Path:
    """Return plot directory for an experiment, creating it if needed.

    Usage:
        plot_dir = get_plot_dir("0_baseline")
        fig.savefig(plot_dir / "snr_nmse.png")
    """
    d = Path(TrainConfig().plots_dir) / experiment
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_results_dir(experiment: str) -> Path:
    """Return results directory for an experiment, creating it if needed."""
    d = Path(TrainConfig().results_dir) / experiment
    d.mkdir(parents=True, exist_ok=True)
    return d
