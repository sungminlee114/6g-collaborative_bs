"""Unified experiment pipeline: load UEs once into RAM, run experiments individually.

Two-phase design:
  1. load_all_ues() — CIR→CFR once, keep h_real in RAM (~96GB for 6 UEs)
  2. run_s0/s2/s3/s4/s5/s7() — each takes ue_data, returns results instantly

Usage (notebook):
    ue_data = load_all_ues("munich_elaa_m_1k_15g", max_snapshots=4000)
    s0 = run_s0(ue_data)        # instant
    s2 = run_s2(ue_data)        # instant
    # modify s2 params, re-run:
    s2 = run_s2(ue_data, n_tau=50)  # instant, no reload
"""
import numpy as np
import torch
from typing import Optional
from tqdm.auto import tqdm

from src.ce_skip import ExperimentConfig
from src.ce_skip.temporal_dataset import TemporalChannelData
from src.ce_skip.metrics import (
    nmse_per_slot, rate_preservation_ratio,
    rate_loss_per_slot, skip_miss_rate,
    scheduling_overhead, effective_throughput_gain,
)

SPEED_LABELS = {0.0: "static", 1.0: "ped", 8.3: "veh"}


def _rayleigh_distance(cfg: ExperimentConfig) -> float:
    c = 3e8
    wavelength = c / cfg.frequency
    d_spacing = wavelength / 2
    D = max(cfg.tx_rows, cfg.tx_cols) * d_spacing
    return 2 * D**2 / wavelength


# ══════════════════════════════════════════════════════════════
#  Phase 1: Load UEs into RAM
# ══════════════════════════════════════════════════════════════

def load_all_ues(
    preset: str,
    max_snapshots: int = 4000,
    ue_per_speed: int = 2,
    split: dict = None,
    load_ue_ids: list = None,
) -> dict:
    """Load selected UEs into RAM. Returns dict with ue_list + metadata.

    This is the SLOW step (CIR→CFR). Run once, keep in memory.
    ~16GB per UE (4000 snap × 2 × 512 × 1024 × 4 bytes).

    Args:
        split: if provided (from cfg.ue_split()), partitions ue_list into
               train_ues, val_ues, test_ues. Ignores ue_per_speed.
        load_ue_ids: explicit list of UE IDs to load. Ignores ue_per_speed.
    """
    cfg = ExperimentConfig.from_preset(preset)
    data = TemporalChannelData(
        cfg.temporal_dir, trajectory_dir=cfg.trajectory_dir,
        max_snapshots=max_snapshots, preset=preset,
    )
    print(f"Loaded {preset}: {data.num_snapshots} snaps, {data.num_ue} UEs, dt={data.dt_s}s")

    r_rayleigh = _rayleigh_distance(cfg)
    T = min(max_snapshots, data.num_snapshots)

    # Determine which UEs to load
    if load_ue_ids is not None:
        # Explicit UE list takes priority
        target_ue_ids = sorted(load_ue_ids)
    elif split is not None:
        # Load all UEs in the split
        all_ids = split["train"] + split.get("val", []) + split["test"]
        target_ue_ids = sorted(set(all_ids))
    else:
        target_ue_ids = None  # legacy: use ue_per_speed selection

    if target_ue_ids is not None:
        # Resolve (bs_id, uid) pairs — find serving BS for each UE
        all_ue_ids = []
        for uid in target_ue_ids:
            bs_id = int(data.ue_bs_ids[uid]) if data.ue_bs_ids is not None else 0
            h1 = data.get_ue_series(uid, bs_id, snap_range=(0, 1))
            if np.abs(h1).sum() > 1e-12:
                all_ue_ids.append((bs_id, uid))
            else:
                print(f"  [warn] UE{uid} has zero channel, skipping")
    else:
        # Legacy: select ue_per_speed per speed group
        if data.ue_bs_ids is not None:
            bs_ids = sorted(set(data.ue_bs_ids))
        else:
            bs_ids = list(range(cfg.num_bs))
        all_ue_ids = []
        for bs_id in bs_ids:
            ue_ids_for_bs = [i for i, b in enumerate(data.ue_bs_ids) if b == bs_id] if data.ue_bs_ids is not None else []
            if not ue_ids_for_bs:
                continue
            valid = []
            for uid in ue_ids_for_bs:
                h1 = data.get_ue_series(uid, bs_id, snap_range=(0, 1))
                if np.abs(h1).sum() > 1e-12:
                    valid.append(uid)
            counts = {}
            for uid in valid:
                spd = round(float(data.get_ue_speed(uid)), 1)
                counts.setdefault(spd, 0)
                if counts[spd] < ue_per_speed:
                    all_ue_ids.append((bs_id, uid))
                    counts[spd] += 1

    labels = [f"UE{uid}({SPEED_LABELS.get(round(data.get_ue_speed(uid), 1), '?')})" for _, uid in all_ue_ids]
    print(f"Selected {len(all_ue_ids)} UEs: [{', '.join(labels)}]")

    # Pre-load velocity data once
    _traj_vels = None
    traj_path = data.trajectory_dir / "trajectories.npz" if data.trajectory_dir else None
    if traj_path and traj_path.exists():
        _traj = np.load(traj_path)
        if "velocities" in _traj:
            _traj_vels = _traj["velocities"]

    # Load all into RAM
    ue_list = []
    for bs_id, uid in tqdm(all_ue_ids, desc="Loading UEs (CIR→CFR)", unit="ue"):
        h_complex = data.get_ue_series(uid, bs_id, snap_range=(0, T))
        h_real = torch.from_numpy(
            np.stack([h_complex.real, h_complex.imag], axis=1).astype(np.float32)
        )  # (T, 2, n_ant, n_sc) float32 on CPU
        del h_complex

        # Position relative to BS → (r, θ, v_radial, v_tangential)
        ue_pos = data.positions[0, uid] if data.positions is not None else np.zeros(3)
        bs_positions = data.bs_info.get("positions", data.bs_info.get("bs_positions", []))
        bs_pos = np.array(bs_positions[bs_id]) if bs_id < len(bs_positions) else np.zeros(3)
        rel = ue_pos - bs_pos
        r_dist = float(np.linalg.norm(rel[:2]))
        theta_deg = float(np.degrees(np.arctan2(rel[1], rel[0])))

        # Velocity decomposition from pre-loaded trajectory
        v_radial, v_tangential = 0.0, 0.0
        if _traj_vels is not None:
            v_avg = _traj_vels[:, uid, :].mean(axis=0)  # (2,)
            r_hat = rel[:2] / r_dist if r_dist > 1e-6 else np.zeros(2)
            v_radial = float(np.dot(v_avg, r_hat))
            v_mag = float(np.linalg.norm(v_avg))
            v_tangential = float(np.sqrt(max(v_mag**2 - v_radial**2, 0)))

        ue_list.append({
            "uid": int(uid),
            "bs_id": int(bs_id),
            "speed": float(data.get_ue_speed(uid)),
            "dist": float(data.get_ue_distance(uid, bs_id, snap_idx=0)),
            "r": r_dist,
            "theta": theta_deg,
            "v_radial": v_radial,
            "v_tangential": v_tangential,
            "pos_xy": (float(rel[0]), float(rel[1])),
            "h_real": h_real,
        })

    ram_gb = sum(u["h_real"].nbytes for u in ue_list) / 1e9
    print(f"Loaded {len(ue_list)} UEs into RAM ({ram_gb:.1f} GB)")

    data.clear_cache()

    result = {
        "preset": preset,
        "ue_list": ue_list,
        "cfg": cfg,
        "r_rayleigh": r_rayleigh,
        "max_snapshots": T,
        "n_ant": cfg.num_rx_ant * cfg.num_tx_ant,
        "n_sc": cfg.num_subcarriers,
    }

    # Partition into train/val/test if split provided
    if split is not None:
        uid_to_ue = {u["uid"]: u for u in ue_list}
        result["train_ues"] = [uid_to_ue[uid] for uid in split["train"] if uid in uid_to_ue]
        result["val_ues"] = [uid_to_ue[uid] for uid in split.get("val", []) if uid in uid_to_ue]
        result["test_ues"] = [uid_to_ue[uid] for uid in split["test"] if uid in uid_to_ue]
        result["split"] = split
        print(f"  Split: train={len(result['train_ues'])}, "
              f"val={len(result['val_ues'])}, test={len(result['test_ues'])}")

    return result


# ══════════════════════════════════════════════════════════════
#  Phase 1.5: Unified pipeline helpers (DL-CE training + LUT)
# ══════════════════════════════════════════════════════════════

def calibrate_lut(ue_data: dict, snr_db: float = 20.0,
                  nmse_margin_db: float = 1.0,
                  tau_values: list = None,
                  device=None) -> "scipy.interpolate.interp1d":
    """Build speed→optimal_c LUT from train UEs.

    For each train UE:
      1. Compute d_oracle (or d_ls) from h_real/h_ls
      2. Sweep tau values, find empirical optimal c
      3. Record (speed, c_opt)

    Returns scipy interp1d: speed (m/s) → optimal c value.
    """
    from scipy.interpolate import interp1d

    train_ues = ue_data.get("train_ues", ue_data["ue_list"])
    if tau_values is None:
        tau_values = np.linspace(0.001, 0.3, 25).tolist()

    noise_floor = np.sqrt(2.0 / (10 ** (snr_db / 10)))
    speed_c_pairs = []

    # Get pilot mask from config if available
    cfg = ue_data.get("cfg")
    pmask = cfg.scene.pilot_mask() if cfg and hasattr(cfg, 'scene') else None

    for u in tqdm(train_ues, desc="Calibrating LUT", unit="ue"):
        # Compute deltas if not already done
        if "d_oracle" not in u:
            _precompute_deltas_for_ue(u, snr_db, pilot_mask=pmask)

        # Sweep tau and find optimal c
        results = _batch_nmse_multi_tau(
            u["h_real"], u.get("h_ls", u["h_real"]),
            u["d_oracle"], tau_values,
            alpha_mode="binary", device=device,
        )
        sweep_results = [
            {"tau": r["tau"], "nmse": float(np.mean(r["nmse_arr"])),
             "n_skip": r["n_skip"], "total": r["total"]}
            for r in results
        ]
        c_opt = _empirical_optimal_c(sweep_results, noise_floor, nmse_margin_db)
        speed_c_pairs.append((u["speed"], c_opt))

    speed_c_pairs.sort()
    speeds_arr = np.array([s for s, c in speed_c_pairs])
    copt_arr = np.array([c for s, c in speed_c_pairs])

    lut = interp1d(speeds_arr, copt_arr, kind="linear",
                   bounds_error=False, fill_value=(copt_arr[0], copt_arr[-1]))

    print(f"  LUT calibrated from {len(train_ues)} train UEs:")
    for spd in sorted(set(speeds_arr)):
        mask = speeds_arr == spd
        print(f"    {spd:.1f} m/s → c_opt = {copt_arr[mask].mean():.3f} "
              f"(range {copt_arr[mask].min():.3f}–{copt_arr[mask].max():.3f})")

    return lut


def train_dl_ce(ue_data: dict, snr_db: float = 20.0, epochs: int = 80,
                lr: float = 1e-3, batch_size: int = 2,
                n_train: int = 100, n_val: int = 50,
                device: str = "cuda", save_path: str = None,
                tracker=None) -> torch.nn.Module:
    """Train DL-CE model using independent snapshot data (diverse positions).

    Uses existing snapshot data (ChannelEstimationDataset) — each sample is
    a different UE position, providing spatial diversity without temporal overhead.
    Random SNR augmentation per epoch for noise diversity.

    Returns trained model.
    """
    from src.dataset_operation.dataset import ChannelEstimationDataset
    from src.models.estimator import create_model
    from src.training.trainer import train_local
    from torch.utils.data import DataLoader

    cfg = ue_data["cfg"]
    data_dir = cfg.data_dir

    # Use existing snapshot data — train/val split by UE index
    # bs_ids=None loads all BS; filtering by ue_id is sufficient
    train_ds = ChannelEstimationDataset(
        data_dir=data_dir,
        ue_ids=list(range(0, n_train)), fixed_snr_db=snr_db,
    )
    val_ds = ChannelEstimationDataset(
        data_dir=data_dir,
        ue_ids=list(range(n_train, n_train + n_val)), fixed_snr_db=snr_db,
    )

    print(f"Training DL-CE: {len(train_ds)} train, {len(val_ds)} val samples "
          f"(independent snapshots, SNR={snr_db}dB)")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)

    # Create model — same architecture as S1/train_dl_ce.py
    # Fully convolutional: n_ant_pairs/n_subcarriers are defaults (8/1024)
    # but model adapts to any input spatial size via conv layers
    n_ant = cfg.num_rx_ant * cfg.num_tx_ant
    if n_ant > 256:
        batch_size = min(batch_size, 4)
    model = create_model(
        site_integration="none",
        encoder_channels=64, encoder_blocks=3, task_head_blocks=2,
    )

    # Train
    model = model.to(device)
    result = train_local(
        model, train_loader, val_loader,
        epochs=epochs, lr=lr, device=device,
        patience=15, verbose=True,
        save_as=save_path, tracker=tracker,
    )

    best_val = result.get('best_val', result.get('best_val_nmse_db', float('nan')))
    print(f"  DL-CE training done: best_val={best_val:.1f} dB")

    return model


def _precompute_deltas_for_ue(u: dict, snr_db: float = 20.0,
                               pilot_mask: np.ndarray = None):
    """Compute h_ls, d_oracle, d_ls for a single UE dict (in-place).

    Args:
        pilot_mask: (n_sc,) bool — if given, δ is computed on pilot SC only
                    (realistic DMRS observation). NMSE is always on full grid.
    """
    h_real = u["h_real"]
    T = h_real.shape[0]

    sig_pow = h_real.flatten(1).pow(2).mean(1, keepdim=True).sqrt().unsqueeze(-1).unsqueeze(-1)
    h_ls = h_real + (sig_pow / (10 ** (snr_db / 20))) * torch.randn_like(h_real)
    u["h_ls"] = h_ls

    # Select subcarriers for δ computation
    if pilot_mask is not None:
        pm = torch.from_numpy(pilot_mask)
        h_real_p = h_real[:, :, :, pm]
        h_ls_p = h_ls[:, :, :, pm]
    else:
        h_real_p = h_real
        h_ls_p = h_ls

    # Oracle delta (on pilot SC)
    diff = h_real_p[1:] - h_real_p[:-1]
    d_oracle = (diff.flatten(1).norm(dim=1) /
                h_real_p[:-1].flatten(1).norm(dim=1).clamp(min=1e-12)).numpy()
    u["d_oracle"] = d_oracle

    # LS delta (on pilot SC)
    diff_ls = h_ls_p[1:] - h_ls_p[:-1]
    d_ls = (diff_ls.flatten(1).norm(dim=1) /
            h_ls_p[:-1].flatten(1).norm(dim=1).clamp(min=1e-12)).numpy()
    u["d_ls"] = d_ls

    spd = round(u["speed"], 1)
    u["spd"] = str(spd)


# ══════════════════════════════════════════════════════════════
#  Phase 2: Individual experiments (instant, no CIR→CFR)
# ══════════════════════════════════════════════════════════════

def run_s0(ue_data: dict, snr_db: float = 20.0) -> dict:
    """S0: Temporal persistence — δ oracle + LS."""
    per_ue = []
    for u in tqdm(ue_data["ue_list"], desc="S0: UEs", unit="ue", position=0):
        h = u["h_real"]
        T = h.shape[0]

        # LS noise
        sig_pow = h.flatten(1).pow(2).mean(1, keepdim=True).sqrt().unsqueeze(-1).unsqueeze(-1)
        h_ls = h + (sig_pow / (10 ** (snr_db / 20))) * torch.randn_like(h)

        d_oracle, d_ls, nmse_reuse = [], [], []
        for t in tqdm(range(1, T), desc=f"  S0 UE{u['uid']} δ", unit="snap", leave=False, position=1):
            ref = h[t - 1].norm()
            if ref > 1e-12:
                d_oracle.append(float((h[t] - h[t - 1]).norm() / ref))
                if h[t].norm() > 1e-12:
                    nmse_reuse.append(float((h[t] - h[t - 1]).pow(2).sum() / h[t].pow(2).sum()))
            else:
                d_oracle.append(float("inf"))
            ref_ls = h_ls[t - 1].norm()
            if ref_ls > 1e-12:
                d_ls.append(float((h_ls[t] - h_ls[t - 1]).norm() / ref_ls))
            else:
                d_ls.append(float("inf"))

        do = np.array([d for d in d_oracle if np.isfinite(d)])
        dl = np.array([d for d in d_ls if np.isfinite(d)])
        nm = np.array(nmse_reuse)
        if len(do) == 0:
            continue

        per_ue.append({
            "uid": u["uid"], "speed": u["speed"], "dist": u["dist"],
            "median_delta": float(np.median(do)),
            "median_delta_ls": float(np.median(dl)) if len(dl) > 0 else float("nan"),
            "mean_delta": float(np.mean(do)),
            "p90_delta": float(np.percentile(do, 90)),
            "nmse_reuse_db": float(10 * np.log10(max(np.mean(nm), 1e-30))) if len(nm) > 0 else float("nan"),
        })

    all_d = [u["median_delta"] for u in per_ue]
    static_d = [u["median_delta"] for u in per_ue if u["speed"] < 0.1]
    summary = {
        "n_ues": len(per_ue),
        "median_delta": float(np.median(all_d)) if all_d else float("nan"),
        "static_median_delta": float(np.median(static_d)) if static_d else float("nan"),
        "median_nmse_reuse_db": float(np.median([u["nmse_reuse_db"] for u in per_ue
                                                   if not np.isnan(u["nmse_reuse_db"])])) if per_ue else float("nan"),
    }
    return {"per_ue": per_ue, "summary": summary}


def _doppler_scheduling(speed: float, frequency: float, dt_s: float, T: int, n_max: int = 50) -> np.ndarray:
    """Doppler-based scheduling: skip for T_c slots, then full CE.

    T_c = λ/(2v) = coherence time. Skip interval = floor(T_c / dt).
    Returns tiers array (T,) — 0=skip, 2=full.
    """
    c = 3e8
    wavelength = c / frequency
    if speed < 0.01:
        # Static: skip everything except first and safety
        skip_interval = n_max
    else:
        T_c = wavelength / (2 * speed)
        skip_interval = max(1, int(T_c / dt_s))
        skip_interval = min(skip_interval, n_max)

    tiers = np.zeros(T, dtype=np.int32)
    tiers[0] = 2  # first slot always full
    for t in range(1, T):
        if (t % skip_interval) == 0:
            tiers[t] = 2
        else:
            tiers[t] = 0
    return tiers


def _smoothed_ls_deltas(h_ls: torch.Tensor, K: int = 4) -> np.ndarray:
    """Compute δ from K-slot averaged LS estimates. Noise reduces by √K."""
    T = h_ls.shape[0]
    # Running average with window K
    h_avg = torch.zeros_like(h_ls)
    for t in range(T):
        t0 = max(0, t - K + 1)
        h_avg[t] = h_ls[t0:t+1].mean(dim=0)

    diff = h_avg[1:] - h_avg[:-1]
    ref = h_avg[:-1].flatten(1).norm(dim=1).clamp(min=1e-12)
    return (diff.flatten(1).norm(dim=1) / ref).numpy()


def _snr_adaptive_tau(tau_base: float, snr_db: float) -> float:
    """SNR-adaptive threshold: τ_eff = τ_base + noise_floor.

    Noise floor of LS δ ≈ √(2/SNR_linear).
    """
    snr_lin = 10 ** (snr_db / 10)
    noise_floor = np.sqrt(2.0 / snr_lin)
    return tau_base + noise_floor


def run_s1(ue_data: dict, tau: float = 0.2, snr_db: float = 20.0) -> dict:
    """S1: Monitor comparison — which scheduling strategy works best?

    Compares 5 strategies at one τ:
      1. Always Full CE (baseline)
      2. Always Skip (lower bound)
      3. LS monitor (current proposal)
      4. Oracle monitor (upper bound)
      5. Doppler-based (coherence time, no per-slot monitor)
      6. SNR-adaptive τ (LS + noise floor compensation)
      7. Smoothed LS (K-slot averaging)

    Returns per-UE results for all strategies.
    """
    cfg = ExperimentConfig.from_preset(ue_data["preset"])
    frequency = cfg.frequency
    dt_s = cfg.slot_duration_s

    # SNR-adaptive τ
    tau_snr = _snr_adaptive_tau(tau, snr_db)

    per_ue = []
    for u in tqdm(ue_data["ue_list"], desc=f"S1: monitor comparison (τ={tau})", unit="ue"):
        h = u["h_real"]
        T = h.shape[0]
        h_ls, deltas_ls, deltas_oracle, _ = _precompute_deltas(h, snr_db)
        h_real_sq = h.flatten(1).pow(2).sum(1).clamp(min=1e-12)

        def _nmse_from_tiers(tiers):
            nm = _compute_nmse_from_tiers(h, h_ls, tiers)
            return float(10 * np.log10(max(np.mean(nm), 1e-30)))

        def _sr(tiers):
            return float((tiers == 0).sum() / len(tiers))

        strategies = {}

        # 1. Always Full CE
        strategies["full_ce"] = {
            "nmse_db": float(10 * np.log10(max(
                ((h_ls - h).flatten(1).pow(2).sum(1) / h_real_sq).mean().item(), 1e-30))),
            "skip_rate": 0.0,
        }

        # 2. Always Skip
        h_skip = h_ls[0:1].expand_as(h)
        strategies["always_skip"] = {
            "nmse_db": float(10 * np.log10(max(
                ((h_skip - h).flatten(1).pow(2).sum(1) / h_real_sq).mean().item(), 1e-30))),
            "skip_rate": 1.0,
        }

        # 3. LS monitor
        sched_ls = _vectorized_scheduling(deltas_ls, tau_low=tau, tau_high=2 * tau)
        strategies["ls_monitor"] = {
            "nmse_db": _nmse_from_tiers(sched_ls["tiers"]),
            "skip_rate": _sr(sched_ls["tiers"]),
        }

        # 4. Oracle monitor
        sched_oracle = _vectorized_scheduling(deltas_oracle, tau_low=tau, tau_high=2 * tau)
        strategies["oracle_monitor"] = {
            "nmse_db": _nmse_from_tiers(sched_oracle["tiers"]),
            "skip_rate": _sr(sched_oracle["tiers"]),
        }

        # 5. Doppler-based
        tiers_doppler = _doppler_scheduling(u["speed"], frequency, dt_s, T)
        strategies["doppler"] = {
            "nmse_db": _nmse_from_tiers(tiers_doppler),
            "skip_rate": _sr(tiers_doppler),
        }

        # 6. SNR-adaptive τ
        sched_snr = _vectorized_scheduling(deltas_ls, tau_low=tau_snr, tau_high=2 * tau_snr)
        strategies["snr_adaptive"] = {
            "nmse_db": _nmse_from_tiers(sched_snr["tiers"]),
            "skip_rate": _sr(sched_snr["tiers"]),
        }

        # 7. Smoothed LS (K=4)
        deltas_smooth = _smoothed_ls_deltas(h_ls, K=4)
        sched_smooth = _vectorized_scheduling(deltas_smooth, tau_low=tau, tau_high=2 * tau)
        strategies["smoothed_ls_K4"] = {
            "nmse_db": _nmse_from_tiers(sched_smooth["tiers"]),
            "skip_rate": _sr(sched_smooth["tiers"]),
        }

        per_ue.append({
            "uid": u["uid"], "speed": u["speed"], "dist": u["dist"],
            "strategies": strategies,
        })

        # Print summary
        tqdm.write(f"  UE{u['uid']} (v={u['speed']:.1f}):")
        for name, s in strategies.items():
            tqdm.write(f"    {name:20s}: NMSE={s['nmse_db']:+6.1f}dB  SR={s['skip_rate']:.0%}")

    return {"per_ue": per_ue, "tau": tau, "snr_db": snr_db, "tau_snr_adaptive": tau_snr}


def _precompute_deltas(h_real: torch.Tensor, snr_db: float = 20.0,
                       pilot_mask: np.ndarray = None):
    """Precompute LS estimates and δ series for a UE.

    Args:
        pilot_mask: (n_sc,) bool — if given, δ is computed on pilot SC only.
                    NMSE reuse is always computed on full grid.

    Returns:
        h_ls: (T, 2, n_ant, n_sc) noisy LS (full grid)
        deltas_ls: (T-1,) LS-based normalized delta (pilot SC)
        deltas_oracle: (T-1,) oracle normalized delta (pilot SC)
        nmse_reuse: (T-1,) NMSE if reusing previous slot (full grid)
    """
    T = h_real.shape[0]
    sig_pow = h_real.flatten(1).pow(2).mean(1, keepdim=True).sqrt().unsqueeze(-1).unsqueeze(-1)
    h_ls = h_real + (sig_pow / (10 ** (snr_db / 20))) * torch.randn_like(h_real)

    # Select subcarriers for δ computation
    if pilot_mask is not None:
        pm = torch.from_numpy(pilot_mask)
        h_real_p = h_real[:, :, :, pm]
        h_ls_p = h_ls[:, :, :, pm]
    else:
        h_real_p = h_real
        h_ls_p = h_ls

    # LS δ on pilot SC
    diff_ls = h_ls_p[1:] - h_ls_p[:-1]
    deltas_ls = (diff_ls.flatten(1).norm(dim=1) /
                 h_ls_p[:-1].flatten(1).norm(dim=1).clamp(min=1e-12)).numpy()

    # Oracle δ on pilot SC
    diff_oracle = h_real_p[1:] - h_real_p[:-1]
    deltas_oracle = (diff_oracle.flatten(1).norm(dim=1) /
                     h_real_p[:-1].flatten(1).norm(dim=1).clamp(min=1e-12)).numpy()

    # NMSE reuse on FULL grid (demodulation uses all SC)
    diff_full = h_real[1:] - h_real[:-1]
    nmse_reuse = (diff_full.flatten(1).pow(2).sum(1) /
                  h_real[1:].flatten(1).pow(2).sum(1).clamp(min=1e-12)).numpy()

    return h_ls, deltas_ls, deltas_oracle, nmse_reuse


def _precompute_deltas_sparse(h_real: torch.Tensor, snr_db: float = 20.0,
                               pilot_mask: np.ndarray = None):
    """Like _precompute_deltas but with sparse DMRS pilot observation.

    Args:
        h_real: (T, 2, n_ant, n_sc) ground truth channel
        snr_db: SNR for LS noise
        pilot_mask: (n_sc,) bool — True at pilot subcarrier positions.
                    If None, falls back to full observation.

    Returns:
        h_ls: (T, 2, n_ant, n_sc) full noisy LS (all SC, for NMSE eval)
        deltas_ls: (T-1,) δ computed on pilot SC only
        deltas_oracle: (T-1,) oracle δ on pilot SC only
        nmse_reuse: (T-1,) NMSE if reusing previous slot (full grid)
    """
    T = h_real.shape[0]
    sig_pow = h_real.flatten(1).pow(2).mean(1, keepdim=True).sqrt().unsqueeze(-1).unsqueeze(-1)
    h_ls = h_real + (sig_pow / (10 ** (snr_db / 20))) * torch.randn_like(h_real)

    if pilot_mask is None:
        pm = slice(None)
    else:
        pm = torch.from_numpy(pilot_mask)

    h_ls_p = h_ls[:, :, :, pm]
    h_real_p = h_real[:, :, :, pm]

    diff_ls = h_ls_p[1:] - h_ls_p[:-1]
    deltas_ls = (diff_ls.flatten(1).norm(dim=1) /
                 h_ls_p[:-1].flatten(1).norm(dim=1).clamp(min=1e-12)).numpy()

    diff_oracle = h_real_p[1:] - h_real_p[:-1]
    deltas_oracle = (diff_oracle.flatten(1).norm(dim=1) /
                     h_real_p[:-1].flatten(1).norm(dim=1).clamp(min=1e-12)).numpy()

    diff_full = h_real[1:] - h_real[:-1]
    nmse_reuse = (diff_full.flatten(1).pow(2).sum(1) /
                  h_real[1:].flatten(1).pow(2).sum(1).clamp(min=1e-12)).numpy()

    return h_ls, deltas_ls, deltas_oracle, nmse_reuse


# ── Speed estimation from δ history → adaptive τ ────────────────────

def _estimate_doppler_from_deltas(deltas: np.ndarray, dt_s: float,
                                   ema_alpha: float = 0.3) -> np.ndarray:
    """Estimate max Doppler spread f_d from δ time series.

    Uses the relationship: δ ≈ 2π·f_d·Δt for small Δt (first-order Taylor of J₀).
    EMA smoothing removes LS noise from δ before estimation.

    Args:
        deltas: (T-1,) normalized channel delta series
        dt_s: slot duration in seconds
        ema_alpha: EMA smoothing factor (0=heavy smoothing, 1=no smoothing)

    Returns:
        f_d_est: (T-1,) estimated max Doppler frequency (Hz)
    """
    # Smooth δ to remove LS noise
    ema = np.empty_like(deltas)
    ema[0] = deltas[0]
    for t in range(1, len(deltas)):
        ema[t] = ema_alpha * deltas[t] + (1 - ema_alpha) * ema[t - 1]

    # δ ≈ 2π·f_d·Δt → f_d ≈ δ / (2π·Δt)
    f_d_est = np.clip(ema / (2 * np.pi * dt_s), 0.0, None)
    return f_d_est


def _estimate_speed_from_deltas(deltas: np.ndarray, dt_s: float,
                                 frequency_hz: float,
                                 ema_alpha: float = 0.3) -> np.ndarray:
    """Estimate UE speed from δ time series.

    δ → f_d → v = f_d · c / f_c

    Args:
        deltas: (T-1,) normalized channel delta
        dt_s: slot duration in seconds
        frequency_hz: carrier frequency in Hz
        ema_alpha: EMA smoothing factor

    Returns:
        v_est: (T-1,) estimated speed (m/s)
    """
    c = 3e8
    f_d = _estimate_doppler_from_deltas(deltas, dt_s, ema_alpha)
    return f_d * c / frequency_hz


def _empirical_optimal_c(tau_sweep_results: list, noise_floor: float,
                          nmse_margin_db: float = 1.0) -> float:
    """Find optimal c from tau sweep data (empirical, per-UE).

    Scans c values from high to low, returns the largest c where
    NMSE stays within nmse_margin_db of the Full CE baseline (c→0).

    Args:
        tau_sweep_results: list of {tau, nmse_arr/nmse, n_skip, total, ...}
        noise_floor: LS noise floor for c = tau/NF conversion
        nmse_margin_db: max allowed NMSE degradation vs baseline (dB)

    Returns:
        c_opt: optimal c value (float)
    """
    if not tau_sweep_results:
        return 1.0

    # Extract c and NMSE
    c_vals, nmses = [], []
    for r in tau_sweep_results:
        c_vals.append(r["tau"] / noise_floor)
        nmse_lin = r.get("nmse", None)
        if nmse_lin is None:
            nmse_lin = float(np.mean(r["nmse_arr"]))
        nmses.append(nmse_lin)

    c_vals = np.array(c_vals)
    nmses_db = 10 * np.log10(np.clip(nmses, 1e-30, None))
    baseline_db = nmses_db[0]  # smallest τ ≈ Full CE

    # Scan from high c to low c: find largest c within margin
    for i in range(len(c_vals) - 1, -1, -1):
        if nmses_db[i] <= baseline_db + nmse_margin_db:
            return float(c_vals[i])
    return float(c_vals[0])


def _optimal_tau_from_speed(v_est: np.ndarray, frequency_hz: float,
                             dt_s: float, noise_floor: float,
                             target_nmse_db: float = -15.0,
                             c_min: float = 0.5, c_max: float = 5.0) -> np.ndarray:
    """Compute speed-adaptive threshold τ*(v).

    Faster UE → lower τ (trigger CE more often).
    Slower UE → higher τ (skip more aggressively).

    The mapping: τ*(v) = noise_floor × c(v), where c(v) interpolates
    between c_min (at max speed) and c_max (at zero speed) based on
    coherence time in slots.

    Args:
        v_est: (T-1,) estimated speed (m/s)
        frequency_hz: carrier frequency (Hz)
        dt_s: slot duration (s)
        noise_floor: LS noise floor (used as τ unit)
        target_nmse_db: target NMSE quality (not used in simple mapping)
        c_min: minimum c = τ/NF (for fastest UE)
        c_max: maximum c = τ/NF (for static UE)

    Returns:
        tau_adaptive: (T-1,) per-slot adaptive threshold
    """
    c = 3e8
    lam = c / frequency_hz

    # Coherence time in slots: T_c_slots = λ/(2v) / dt_s
    # Avoid div by zero for static UEs
    v_safe = np.clip(v_est, 0.01, None)
    T_c_slots = lam / (2 * v_safe * dt_s)

    # Map T_c_slots to c value: more coherent → higher c (skip more)
    # Sigmoid-like mapping: c = c_min + (c_max - c_min) * (1 - exp(-T_c / k))
    k = 10.0  # characteristic scale in slots
    c_val = c_min + (c_max - c_min) * (1.0 - np.exp(-T_c_slots / k))

    # Static UEs (v < 0.01): max c
    c_val[v_est < 0.01] = c_max

    return noise_floor * c_val


def _adaptive_scheduling(deltas: np.ndarray, dt_s: float, frequency_hz: float,
                          noise_floor: float, ema_alpha: float = 0.3,
                          c_min: float = 0.5, c_max: float = 5.0,
                          alpha_mode: str = "ramp", n_max: int = 50) -> dict:
    """Speed-adaptive CE scheduling.

    Pipeline:
        1. δ history → EMA smoothing → f_d estimation → v estimation
        2. v → T_c → τ*(v) adaptive threshold
        3. δ[t] vs τ*[t] → per-slot alpha scheduling

    Returns dict with: alphas, full_mask, tiers, v_est, tau_adaptive, + counts.
    """
    # Step 1: estimate speed from δ
    v_est = _estimate_speed_from_deltas(deltas, dt_s, frequency_hz, ema_alpha)

    # Step 2: compute adaptive τ per slot
    tau_adaptive = _optimal_tau_from_speed(
        v_est, frequency_hz, dt_s, noise_floor, c_min=c_min, c_max=c_max)

    # Step 3: per-slot scheduling with varying τ
    T = len(deltas) + 1
    alphas = np.zeros(T, dtype=np.float64)
    alphas[0] = 1.0  # slot 0 always full CE

    full_mask = np.zeros(T, dtype=bool)
    full_mask[0] = True

    for t in range(len(deltas)):
        tau_t = tau_adaptive[t]
        delta_min_t = tau_t * 0.5  # ramp starts at half of τ
        if alpha_mode == "binary":
            alphas[t + 1] = 1.0 if deltas[t] > tau_t else 0.0
            full_mask[t + 1] = deltas[t] > tau_t
        elif alpha_mode == "ramp":
            denom = tau_t - delta_min_t
            if denom < 1e-12:
                denom = 1e-12
            alphas[t + 1] = np.clip((deltas[t] - delta_min_t) / denom, 0.0, 1.0)
            full_mask[t + 1] = deltas[t] > tau_t
        else:
            alphas[t + 1] = 1.0 if deltas[t] > tau_t else 0.0
            full_mask[t + 1] = deltas[t] > tau_t

    # Safety: force full CE after n_max consecutive non-Full slots
    if n_max < T:
        consecutive = 0
        for t in range(1, T):
            if not full_mask[t]:
                consecutive += 1
                if consecutive >= n_max:
                    full_mask[t] = True
                    alphas[t] = 1.0
                    consecutive = 0
            else:
                consecutive = 0

    tiers = np.ones(T, dtype=np.int32)
    tiers[alphas < 1e-8] = 0
    tiers[full_mask] = 2

    return {
        "alphas": alphas,
        "full_mask": full_mask,
        "tiers": tiers,
        "n_skip": int((tiers == 0).sum()),
        "n_delta": int((tiers == 1).sum()),
        "n_full": int((tiers == 2).sum()),
        "total": T,
        "v_est": v_est,
        "tau_adaptive": tau_adaptive,
    }


def _vectorized_scheduling(deltas: np.ndarray,
                           tau_low: float = None, tau_high: float = None,
                           tau_full: float = None, delta_min: float = None,
                           alpha_mode: str = "ramp",
                           n_max: int = 50) -> dict:
    """Vectorized scheduling with continuous alpha support.

    Modes:
      - "binary": 2-tier. delta > delta_min → full CE (alpha=1), else skip (alpha=0).
      - "step" (legacy): discrete 3-tier from tau_low/tau_high.
        When tau_low/tau_high are provided without tau_full/delta_min,
        automatically sets alpha_mode="step".
      - "ramp": continuous alpha = clip((delta - delta_min) / (tau_full - delta_min), 0, 1).

    Returns dict with: alphas, full_mask, tiers, n_skip, n_delta, n_full, total.
    """
    T = len(deltas) + 1

    # Legacy compat: tau_low/tau_high → map to delta_min/tau_full
    if tau_low is not None and tau_full is None:
        delta_min = tau_low
        tau_full = tau_high if tau_high is not None else 2 * tau_low
        alpha_mode = "step"

    if delta_min is None or tau_full is None:
        raise ValueError("Must provide either (tau_low, tau_high) or (delta_min, tau_full)")

    # -- Compute continuous alphas (T-1,) for slots 1..T-1 --
    denom = tau_full - delta_min
    if denom < 1e-12:
        denom = 1e-12
    alphas_inner = np.clip((deltas - delta_min) / denom, 0.0, 1.0)  # (T-1,)

    # Full mask: delta exceeds tau_full
    full_inner = deltas > tau_full  # (T-1,) bool

    # For binary mode: skip or full only (no interpolation)
    if alpha_mode == "binary":
        alphas_inner = np.where(deltas > delta_min, 1.0, 0.0)
        full_inner = deltas > delta_min
    # For step mode, quantize alphas to {0, fixed_mid, 1}
    elif alpha_mode == "step":
        # alpha==0 where delta <= delta_min (skip)
        # alpha==1 where delta > tau_full (full)
        # in-between gets 0.5 (legacy EMA default)
        mid_mask = (alphas_inner > 0) & (~full_inner)
        alphas_inner[mid_mask] = 0.5

    # Build full arrays (T,) — slot 0 is always full CE
    alphas = np.zeros(T, dtype=np.float64)
    alphas[0] = 1.0  # placeholder (slot 0 = full)
    alphas[1:] = alphas_inner

    full_mask = np.zeros(T, dtype=bool)
    full_mask[0] = True
    full_mask[1:] = full_inner

    # Safety: force full CE after n_max consecutive non-Full slots
    if n_max < T:
        consecutive = 0
        for t in range(1, T):
            if not full_mask[t]:
                consecutive += 1
                if consecutive >= n_max:
                    full_mask[t] = True
                    alphas[t] = 1.0
                    consecutive = 0
            else:
                consecutive = 0

    # Derive legacy tiers: alpha==0 → 0 (skip), 0<alpha<1 → 1 (delta), full → 2
    # Order matters: set skip first, then full overrides (full_mask slots have alpha=1.0)
    tiers = np.ones(T, dtype=np.int32)  # default: delta (tier 1)
    tiers[alphas < 1e-8] = 0
    tiers[full_mask] = 2

    return {
        "alphas": alphas,
        "full_mask": full_mask,
        "tiers": tiers,
        "n_skip": int((tiers == 0).sum()),
        "n_delta": int((tiers == 1).sum()),
        "n_full": int((tiers == 2).sum()),
        "total": T,
    }


def _compute_nmse_from_alphas(h_real: torch.Tensor, h_ls: torch.Tensor,
                               alphas: np.ndarray, full_mask: np.ndarray,
                               delta_mode: str = "ema",
                               h_ce: torch.Tensor = None) -> np.ndarray:
    """Compute per-slot NMSE using continuous alpha scheduling.

    Per-slot update rule:
      - Full (full_mask[t]=True): h_hat[t] = h_ce[t]
      - alpha < 1e-8 (skip): h_hat[t] = h_hat[t-1]  (fast path for segments)
      - Otherwise: h_hat[t] = alphas[t]*h_ls[t] + (1-alphas[t])*h_hat[t-1]

    Args:
        h_real: (T, 2, n_ant, n_sc) ground truth
        h_ls: (T, 2, n_ant, n_sc) noisy LS estimates
        alphas: (T,) continuous blending weight per slot
        full_mask: (T,) bool — True means use h_ce directly
        delta_mode: "ema" (default), ignored for continuous alpha
        h_ce: optional DL-CE output; if None, uses h_ls

    Returns:
        nmse: (T,) per-slot NMSE
    """
    T = h_real.shape[0]
    if h_ce is None:
        h_ce = h_ls

    # Work in numpy for fast element-wise ops
    h_real_np = h_real.reshape(T, -1).numpy() if h_real.is_contiguous() else h_real.reshape(T, -1).contiguous().numpy()
    h_ls_np = h_ls.reshape(T, -1).numpy() if h_ls.is_contiguous() else h_ls.reshape(T, -1).contiguous().numpy()
    h_ce_np = h_ce.reshape(T, -1).numpy() if h_ce.is_contiguous() else h_ce.reshape(T, -1).contiguous().numpy()

    h_hat = np.empty_like(h_real_np)
    h_hat[0] = h_ce_np[0]

    # Find full CE anchor points — these break dependency chains
    anchors = np.where(full_mask)[0]
    h_hat[anchors] = h_ce_np[anchors]

    # Process segments between consecutive anchors
    seg_starts = anchors
    seg_ends = np.append(anchors[1:], T)

    for start, end in zip(seg_starts, seg_ends):
        if start + 1 >= end:
            continue

        seg_alphas = alphas[start + 1 : end]

        # Fast path: all-skip segment (alpha < 1e-8)
        if (seg_alphas < 1e-8).all():
            h_hat[start + 1 : end] = h_hat[start]
            continue

        # General path: per-slot blending with continuous alpha
        val = h_hat[start]
        for t in range(start + 1, end):
            a = alphas[t]
            if a < 1e-8:
                h_hat[t] = val
            else:
                val = a * h_ls_np[t] + (1.0 - a) * val
                h_hat[t] = val

    # NMSE per slot
    diff = h_hat - h_real_np
    nmse = (diff * diff).sum(axis=1) / np.maximum((h_real_np * h_real_np).sum(axis=1), 1e-12)
    return nmse


def _compute_nmse_from_tiers(h_real: torch.Tensor, h_ls: torch.Tensor,
                              tiers: np.ndarray, alpha: float = 0.5,
                              delta_mode: str = "ema",
                              h_ce: torch.Tensor = None) -> np.ndarray:
    """Wrapper: convert discrete tiers + fixed alpha → continuous alphas, then delegate.

    Tier mapping: 0 (skip) → alpha=0, 1 (delta) → alpha=fixed, 2 (full) → full_mask=True.
    Supports delta_mode="ls_delta" by falling back to legacy per-slot loop.
    """
    T = h_real.shape[0]

    # For ls_delta mode, we need the legacy per-slot loop (different update rule)
    if delta_mode == "ls_delta":
        if h_ce is None:
            h_ce = h_ls
        h_real_np = h_real.reshape(T, -1).numpy() if h_real.is_contiguous() else h_real.reshape(T, -1).contiguous().numpy()
        h_ls_np = h_ls.reshape(T, -1).numpy() if h_ls.is_contiguous() else h_ls.reshape(T, -1).contiguous().numpy()
        h_ce_np = h_ce.reshape(T, -1).numpy() if h_ce.is_contiguous() else h_ce.reshape(T, -1).contiguous().numpy()
        h_hat = np.empty_like(h_real_np)
        h_hat[0] = h_ce_np[0]
        for t in range(1, T):
            tier = tiers[t]
            if tier == 2:
                h_hat[t] = h_ce_np[t]
            elif tier == 0:
                h_hat[t] = h_hat[t - 1]
            else:
                h_hat[t] = h_hat[t - 1] + alpha * (h_ls_np[t] - h_ls_np[t - 1])
        diff = h_hat - h_real_np
        return (diff * diff).sum(axis=1) / np.maximum((h_real_np * h_real_np).sum(axis=1), 1e-12)

    # Convert tiers → continuous alphas + full_mask
    alphas = np.zeros(T, dtype=np.float64)
    full_mask = tiers == 2
    alphas[tiers == 1] = alpha  # delta tier → fixed alpha

    return _compute_nmse_from_alphas(h_real, h_ls, alphas, full_mask,
                                      delta_mode=delta_mode, h_ce=h_ce)


def _batch_nmse_multi_tau(h_real: torch.Tensor, h_ls: torch.Tensor,
                           deltas: np.ndarray, tau_values: list,
                           alpha: float = 0.5, delta_mode: str = "ema",
                           h_ce: torch.Tensor = None,
                           alpha_mode: str = "step",
                           device: Optional[torch.device] = None
                           ) -> list:
    """Compute NMSE for multiple tau values — GPU-accelerated.

    Moves h_real/h_ls to GPU once (~21GB for ELAA), computes scheduling on CPU
    (trivial), then h_hat construction + NMSE on GPU. No RAM copies per tau.
    """
    T = h_real.shape[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device

    # Move to GPU once, flattened: (T, D). Only h_real + h_ls = ~21GB
    h_real_g = h_real.reshape(T, -1).to(device, non_blocking=True)
    h_ls_g = h_ls.reshape(T, -1).to(device, non_blocking=True)
    h_ce_g = h_ls_g if h_ce is None else h_ce.reshape(T, -1).to(device, non_blocking=True)

    # Precompute denominator for NMSE on GPU (tiny: T floats)
    h_real_pow = (h_real_g * h_real_g).sum(dim=1).clamp(min=1e-12)  # (T,)

    results = []
    for tau in tau_values:
        sched = _vectorized_scheduling(deltas, tau_full=2*tau, delta_min=tau,
                                        alpha_mode=alpha_mode)
        alphas_np = sched["alphas"]
        full_mask = sched["full_mask"]

        # Build h_hat on GPU — reuse h_ls_g buffer via clone for anchors
        # Use chunked NMSE to avoid allocating full h_hat (10.4GB)
        anchors = np.where(full_mask)[0]
        seg_starts = anchors
        seg_ends = np.append(anchors[1:], T)

        # Compute NMSE per-slot without storing full h_hat
        nmse = torch.zeros(T, device=device)
        val = h_ce_g[0]
        nmse[0] = ((val - h_real_g[0]).pow(2).sum() / h_real_pow[0])

        anchor_set = set(anchors.tolist())
        for t in range(1, T):
            if t in anchor_set:
                val = h_ce_g[t]
            else:
                a = alphas_np[t]
                if a < 1e-8:
                    pass  # val stays the same (skip)
                else:
                    val = a * h_ls_g[t] + (1.0 - a) * val
            nmse[t] = (val - h_real_g[t]).pow(2).sum() / h_real_pow[t]

        results.append({
            "tau": tau, "nmse_arr": nmse.cpu().numpy(),
            "n_skip": sched["n_skip"], "n_delta": sched["n_delta"],
            "n_full": sched["n_full"], "total": sched["total"],
            "tiers": sched["tiers"], "alphas": alphas_np,
        })

    del h_real_g, h_ls_g, h_ce_g, h_real_pow
    torch.cuda.empty_cache()
    return results


def _batch_scheduling_sweep(h_real: torch.Tensor, h_ls: torch.Tensor,
                             deltas: np.ndarray, tau_values: list,
                             alpha: float = 0.5, delta_mode: str = "ema",
                             n_max: int = 50, alpha_mode: str = "step") -> list:
    """Batch tau sweep: compute scheduling + NMSE for all taus in one pass.

    Shares h_ls and deltas across taus. Uses _compute_nmse_from_alphas internally.

    Args:
        alpha_mode: "step" (legacy 3-tier) or "ramp" (continuous alpha).

    Returns list of dicts: [{tau, nmse_arr, n_skip, n_delta, n_full, total, tiers, alphas}, ...]
    """
    results = []

    for tau in tqdm(tau_values, desc="    τ sweep", unit="τ", leave=False):
        if alpha_mode == "ramp":
            sched = _vectorized_scheduling(deltas, tau_full=2 * tau, delta_min=tau,
                                            alpha_mode="ramp", n_max=n_max)
        else:
            sched = _vectorized_scheduling(deltas, tau_low=tau, tau_high=2 * tau,
                                            n_max=n_max)

        nmse_arr = _compute_nmse_from_alphas(h_real, h_ls, sched["alphas"],
                                              sched["full_mask"], delta_mode=delta_mode)

        results.append({
            "tau": tau, "nmse_arr": nmse_arr,
            "tiers": sched["tiers"], "alphas": sched["alphas"],
            **{k: sched[k] for k in ["n_skip", "n_delta", "n_full", "total"]},
        })

    return results


def run_s2(ue_data: dict, tau_values: list = None, snr_db: float = 20.0) -> dict:
    """S2: Threshold sweep → Pareto front. Vectorized δ, fast tau sweep."""
    if tau_values is None:
        tau_values = np.linspace(0.01, 0.8, 30).tolist()

    per_tau = {tau: {"nmse": [], "n_skip": 0, "n_delta": 0, "n_full": 0, "total": 0}
               for tau in tau_values}

    for u in tqdm(ue_data["ue_list"], desc="S2: UEs", unit="ue"):
        h = u["h_real"]
        h_ls, deltas_ls, deltas_oracle, _ = _precompute_deltas(h, snr_db)
        batch = _batch_scheduling_sweep(h, h_ls, deltas_ls, tau_values)
        tqdm.write(f"  UE{u['uid']}: {len(batch)} τ values swept")
        for r in batch:
            pt = per_tau[r["tau"]]
            pt["nmse"].extend(r["nmse_arr"].tolist())
            pt["n_skip"] += r["n_skip"]
            pt["n_delta"] += r["n_delta"]
            pt["n_full"] += r["n_full"]
            pt["total"] += r["total"]

    sweep = []
    for tau in tau_values:
        pt = per_tau[tau]
        avg = np.mean(pt["nmse"]) if pt["nmse"] else 1.0
        total = pt["total"]
        sweep.append({
            "tau": float(tau),
            "avg_nmse_db": float(10 * np.log10(max(avg, 1e-30))),
            "skip_rate": pt["n_skip"] / max(total, 1),
            "computation_ratio": (pt["n_full"] * 0.5 + (pt["n_delta"] + pt["n_skip"]) * 0.005) / max(total * 0.5, 1e-12),
        })
    return {"sweep": sweep}


def run_s3(ue_data: dict, tau_values: list = None, snr_db: float = 20.0) -> dict:
    """S3: Distance-dependent threshold analysis."""
    if tau_values is None:
        tau_values = np.linspace(0.01, 0.8, 30).tolist()

    r_ray = ue_data["r_rayleigh"]
    per_ue = []

    for u in tqdm(ue_data["ue_list"], desc="S3: UEs", unit="ue"):
        h = u["h_real"]
        dist = u["dist"]
        zone = "NF" if dist < r_ray else ("Transition" if dist < 3 * r_ray else "FF")
        h_ls, deltas_ls, deltas_oracle, _ = _precompute_deltas(h, snr_db)
        batch = _batch_scheduling_sweep(h, h_ls, deltas_ls, tau_values)
        tqdm.write(f"  UE{u['uid']} ({zone}, {dist:.0f}m): {len(batch)} τ swept")

        sweep = {}
        for r in batch:
            sweep[float(r["tau"])] = {"nmse": float(np.mean(r["nmse_arr"])),
                                       "skip_rate": r["n_skip"] / r["total"]}

        per_ue.append({"uid": u["uid"], "dist": dist, "speed": u["speed"], "zone": zone, "sweep": sweep})

    # Aggregate per zone
    zones = {}
    for zone_name in ["NF", "Transition", "FF"]:
        ues = [u for u in per_ue if u["zone"] == zone_name]
        if not ues:
            zones[zone_name] = {"count": 0, "optimal_tau": None}
            continue
        optimal = tau_values[0]
        agg = []
        for tau in tau_values:
            avg_nmse = np.mean([u["sweep"][float(tau)]["nmse"] for u in ues])
            avg_db = 10 * np.log10(max(avg_nmse, 1e-30))
            avg_sr = np.mean([u["sweep"][float(tau)]["skip_rate"] for u in ues])
            agg.append({"tau": float(tau), "nmse_db": avg_db, "skip_rate": avg_sr})
            if avg_db <= -10.0:
                optimal = tau
        zones[zone_name] = {"count": len(ues), "optimal_tau": float(optimal), "sweep": agg}

    return {"zones": zones, "per_ue": per_ue, "r_rayleigh": r_ray}


def run_s4(ue_data: dict, tau_low: float = 0.2, snr_db: float = 20.0,
           delta_modes: list = None, alpha_values: list = None) -> dict:
    """S4: Delta update ablation."""
    if delta_modes is None:
        delta_modes = ["skip", "ema", "ls_delta"]
    if alpha_values is None:
        alpha_values = [0.3, 0.5, 0.7]

    results = {}

    combos = []
    for mode in delta_modes:
        for alpha in (alpha_values if mode != "skip" else [0.0]):
            combos.append((mode, alpha, f"{mode}_a{alpha:.1f}" if mode != "skip" else "skip"))

    for u in tqdm(ue_data["ue_list"], desc="S4: UEs", unit="ue"):
        h = u["h_real"]
        spd_label = SPEED_LABELS.get(round(u["speed"], 1), f"v{u['speed']:.1f}")
        h_ls, deltas_ls, deltas_oracle, _ = _precompute_deltas(h, snr_db)
        sched = _vectorized_scheduling(deltas_ls, tau_low=tau_low, tau_high=2 * tau_low)

        for mode, alpha, key in tqdm(combos, desc=f"  S4 UE{u['uid']} modes", unit="m", leave=False):
            if key not in results:
                results[key] = {}
            if spd_label not in results[key]:
                results[key][spd_label] = {"nmse": [], "skip_rates": []}

            nm = _compute_nmse_from_tiers(h, h_ls, sched["tiers"], alpha=alpha, delta_mode=mode)
            results[key][spd_label]["nmse"].extend(nm.tolist())
            results[key][spd_label]["skip_rates"].append(sched["n_skip"] / sched["total"])

    agg = {}
    for key, sd in results.items():
        agg[key] = {}
        for sl, v in sd.items():
            agg[key][sl] = {
                "nmse_db": float(10 * np.log10(max(np.mean(v["nmse"]), 1e-30))),
                "skip_rate": float(np.mean(v["skip_rates"])),
            }
    return agg


def run_s5(ue_data: dict, snr_list: list = None, tau_list: list = None) -> dict:
    """S5: Beamforming rate impact."""
    if snr_list is None:
        snr_list = [10.0, 15.0, 20.0, 25.0]
    if tau_list is None:
        tau_list = [0.1, 0.2, 0.3, 0.5]

    combo = {}
    for snr in snr_list:
        for tau in tau_list:
            combo[(snr, tau)] = {"rate_loss": [], "rpr": [], "smr": []}

    for u in tqdm(ue_data["ue_list"], desc="S5: UEs", unit="ue"):
        h = u["h_real"]

        for snr in tqdm(snr_list, desc=f"  S5 UE{u['uid']} SNR", unit="snr", leave=False):
            snr_lin = 10 ** (snr / 10)
            h_ls, deltas_ls, _, _ = _precompute_deltas(h, snr)
            batch = _batch_scheduling_sweep(h, h_ls, deltas_ls, tau_list)

            for r in batch:
                tau = r["tau"]
                # h_hat is implicitly in nmse_arr; rebuild for rate metrics
                h_hat = torch.zeros_like(h)
                h_hat[0] = h_ls[0]
                tiers = r["tiers"]
                for t in range(1, h.shape[0]):
                    if tiers[t] == 2:
                        h_hat[t] = h_ls[t]
                    elif tiers[t] == 0:
                        h_hat[t] = h_hat[t - 1]
                    else:
                        h_hat[t] = 0.5 * h_ls[t] + 0.5 * h_hat[t - 1]

                rl = rate_loss_per_slot(h_hat, h, snr_lin)
                combo[(snr, tau)]["rate_loss"].extend(rl.numpy().tolist())
                combo[(snr, tau)]["rpr"].append(rate_preservation_ratio(h_hat, h, snr_lin)["rpr_oracle"])
                combo[(snr, tau)]["smr"].append(skip_miss_rate(h_hat, h, tiers.tolist(), snr_lin, 0.05))

    agg = {}
    for snr in snr_list:
        sk = f"snr{int(snr)}"
        agg[sk] = {}
        for tau in tau_list:
            cd = combo[(snr, tau)]
            rl = np.array(cd["rate_loss"])
            agg[sk][f"tau{tau}"] = {
                "mean_rate_loss": float(np.mean(rl)),
                "frac_above_5pct": float(np.mean(rl > 0.05)),
                "avg_rpr": float(np.mean(cd["rpr"])),
                "avg_smr": float(np.mean(cd["smr"])),
            }
    return agg


def run_s7(ue_data: dict, s2_result: dict = None) -> dict:
    """S7: System overhead & effective throughput (pure computation)."""
    n_ant = ue_data["n_ant"]
    n_sc = ue_data["n_sc"]
    overhead = scheduling_overhead(n_ant, n_sc, t_full_ms=0.5)

    # Find best S2 operating point if available
    if s2_result is not None:
        sweet = [s for s in s2_result["sweep"] if s["avg_nmse_db"] <= -10]
        if sweet:
            best = min(sweet, key=lambda s: s["computation_ratio"])
            eff = effective_throughput_gain(best["skip_rate"], 0.97, best["computation_ratio"])
        else:
            eff = effective_throughput_gain(0.0, 1.0, 1.0)
    else:
        eff = effective_throughput_gain(0.0, 1.0, 1.0)

    return {"overhead": overhead, "effective_throughput": eff}
