from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from pathlib import Path

import yaml

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "assets" / "configs"
DEFAULT_PRESET = "munich_elaa_s_1k_15g"


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
    numerology_mu: int = -1  # 3GPP NR numerology index (0-6)

    # DMRS config (3GPP TS 38.211)
    dmrs_type: int = 2           # 1=comb-2 (50%), 2=comb-3 (33%)
    dmrs_additional_position: int = 1  # 0→1 DMRS sym, 1→2, 2→3, 3→4
    dmrs_max_length: int = 1     # 1=single symbol, 2=double symbol

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
    synthetic_array: bool = True

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
                       "tx_rows", "tx_cols", "rx_rows", "rx_cols",
                       "numerology_mu"):
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
        """Channel sampling spacing (Hz) = effective_bw / num_subcarriers.
        Used by Sionna subcarrier_frequencies() for channel generation."""
        return self.effective_bandwidth / self.num_subcarriers

    @property
    def scs(self) -> float:
        """3GPP NR subcarrier spacing (Hz) = 2^mu * 15 kHz."""
        assert self.numerology_mu >= 0, "numerology_mu not set"
        return (2 ** self.numerology_mu) * 15e3

    @property
    def slot_duration_s(self) -> float:
        """Slot duration in seconds = 1ms / 2^mu."""
        assert self.numerology_mu >= 0, "numerology_mu not set"
        return 1e-3 / (2 ** self.numerology_mu)

    @property
    def symbols_per_slot(self) -> int:
        return 14

    @property
    def slots_per_subframe(self) -> int:
        assert self.numerology_mu >= 0, "numerology_mu not set"
        return 2 ** self.numerology_mu

    @property
    def dmrs_symbols_per_slot(self) -> int:
        """Number of DMRS OFDM symbols per slot."""
        return 1 + self.dmrs_additional_position  # base 1 + additional

    @property
    def pilot_sc_per_dmrs_symbol(self) -> int:
        """Pilot subcarriers per DMRS OFDM symbol."""
        if self.dmrs_type == 1:
            return self.num_subcarriers // 2  # comb-2
        return self.num_subcarriers * 2 // 6  # comb-3 (Type 2)

    @property
    def total_pilot_res(self) -> int:
        """Total pilot REs per slot."""
        return self.pilot_sc_per_dmrs_symbol * self.dmrs_symbols_per_slot

    @property
    def total_res_per_slot(self) -> int:
        """Total REs per slot."""
        return self.num_subcarriers * self.symbols_per_slot

    @property
    def pilot_density(self) -> float:
        """Fraction of REs that are pilots."""
        return self.total_pilot_res / self.total_res_per_slot

    def pilot_mask(self, as_2d: bool = False):
        """Boolean mask for pilot RE positions.

        If as_2d=False: (num_subcarriers,) — pilot subcarrier mask within a DMRS symbol.
        If as_2d=True: (symbols_per_slot, num_subcarriers) — full slot grid.
        """
        import numpy as np
        # Frequency mask within DMRS symbol
        freq_mask = np.zeros(self.num_subcarriers, dtype=bool)
        if self.dmrs_type == 1:
            freq_mask[::2] = True  # comb-2
        else:
            # Type 2: positions 0,1 out of every 6
            for i in range(0, self.num_subcarriers, 6):
                freq_mask[i] = True
                if i + 1 < self.num_subcarriers:
                    freq_mask[i + 1] = True

        if not as_2d:
            return freq_mask

        # 2D: (14 symbols, num_sc)
        grid = np.zeros((self.symbols_per_slot, self.num_subcarriers), dtype=bool)
        # DMRS symbol positions per TS 38.211 Table 7.4.1.1.2-3 / 6.4.1.1.3-3
        # Mapping type A, dmrs-TypeA-Position=pos2 (l0=2), ld=14
        dmrs_positions = [2]
        if self.dmrs_additional_position >= 1:
            dmrs_positions.append(11)
        if self.dmrs_additional_position >= 2:
            dmrs_positions = [2, 8, 11]
        if self.dmrs_additional_position >= 3:
            dmrs_positions = [2, 5, 8, 11]
        for sym in dmrs_positions:
            if sym < self.symbols_per_slot:
                grid[sym] = freq_mask
        return grid


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
    train_bs_ids: List[int] = field(default_factory=list)
    val_bs_ids: List[int] = field(default_factory=list)
    test_bs_ids: List[int] = field(default_factory=list)
    pretrain_bs_ids: List[int] = field(default_factory=list)  # legacy alias for train_bs_ids

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
    encoder_channels: List[int] = field(default_factory=lambda: [32, 32])
    kernel_size: int = 3

    # Site embedding
    site_embed_dim: int = 32
    site_integration: str = "film"  # "film", "concat", "add", "none"

    # Task head
    task_head_channels: List[int] = field(default_factory=lambda: [32])

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
