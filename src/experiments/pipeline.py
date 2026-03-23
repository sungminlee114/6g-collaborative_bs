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
) -> dict:
    """Load selected UEs into RAM. Returns dict with ue_list + metadata.

    This is the SLOW step (CIR→CFR). Run once, keep in memory.
    ~16GB per UE (4000 snap × 2 × 512 × 1024 × 4 bytes).
    """
    cfg = ExperimentConfig.from_preset(preset)
    data = TemporalChannelData(
        cfg.temporal_dir, trajectory_dir=cfg.trajectory_dir,
        max_snapshots=max_snapshots, preset=preset,
    )
    print(f"Loaded {preset}: {data.num_snapshots} snaps, {data.num_ue} UEs, dt={data.dt_s}s")

    r_rayleigh = _rayleigh_distance(cfg)

    # Find BS with UEs
    if data.ue_bs_ids is not None:
        bs_ids = sorted(set(data.ue_bs_ids))
    else:
        bs_ids = list(range(cfg.num_bs))

    # Pre-filter valid UEs
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
        selected = []
        counts = {}
        for uid in valid:
            spd = round(float(data.get_ue_speed(uid)), 1)
            counts.setdefault(spd, 0)
            if counts[spd] < ue_per_speed:
                selected.append((bs_id, uid))
                counts[spd] += 1
        all_ue_ids.extend(selected)

    labels = [f"UE{uid}({SPEED_LABELS.get(round(data.get_ue_speed(uid), 1), '?')})" for _, uid in all_ue_ids]
    print(f"Selected {len(all_ue_ids)} UEs: [{', '.join(labels)}]")

    # Load all into RAM
    ue_list = []
    T = min(max_snapshots, data.num_snapshots)
    for bs_id, uid in tqdm(all_ue_ids, desc="Loading UEs (CIR→CFR)", unit="ue"):
        h_complex = data.get_ue_series(uid, bs_id, snap_range=(0, T))
        h_real = torch.from_numpy(
            np.stack([h_complex.real, h_complex.imag], axis=1).astype(np.float32)
        )  # (T, 2, n_ant, n_sc) float32 on CPU
        del h_complex

        ue_list.append({
            "uid": int(uid),
            "bs_id": int(bs_id),
            "speed": float(data.get_ue_speed(uid)),
            "dist": float(data.get_ue_distance(uid, bs_id, snap_idx=0)),
            "h_real": h_real,
        })

    ram_gb = sum(u["h_real"].nbytes for u in ue_list) / 1e9
    print(f"Loaded {len(ue_list)} UEs into RAM ({ram_gb:.1f} GB)")

    data.clear_cache()

    return {
        "preset": preset,
        "ue_list": ue_list,
        "cfg": cfg,
        "r_rayleigh": r_rayleigh,
        "max_snapshots": T,
        "n_ant": cfg.num_rx_ant * cfg.num_tx_ant,
        "n_sc": cfg.num_subcarriers,
    }


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
    dt_s = 0.00025  # TODO: get from trajectory_info

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


def _precompute_deltas(h_real: torch.Tensor, snr_db: float = 20.0):
    """Precompute LS estimates and δ series for a UE.

    Returns:
        h_ls: (T, 2, n_ant, n_sc) noisy LS
        deltas_ls: (T-1,) LS-based normalized delta
        deltas_oracle: (T-1,) oracle (ground truth) normalized delta
        nmse_reuse: (T-1,) NMSE if reusing previous slot
    """
    T = h_real.shape[0]
    sig_pow = h_real.flatten(1).pow(2).mean(1, keepdim=True).sqrt().unsqueeze(-1).unsqueeze(-1)
    h_ls = h_real + (sig_pow / (10 ** (snr_db / 20))) * torch.randn_like(h_real)

    # LS δ: ||h_ls(t) - h_ls(t-1)|| / ||h_ls(t-1)||
    diff_ls = h_ls[1:] - h_ls[:-1]
    deltas_ls = (diff_ls.flatten(1).norm(dim=1) /
                 h_ls[:-1].flatten(1).norm(dim=1).clamp(min=1e-12)).numpy()

    # Oracle δ: ||h(t) - h(t-1)|| / ||h(t-1)||
    diff_oracle = h_real[1:] - h_real[:-1]
    deltas_oracle = (diff_oracle.flatten(1).norm(dim=1) /
                     h_real[:-1].flatten(1).norm(dim=1).clamp(min=1e-12)).numpy()

    # Vectorized NMSE(reuse): ||h(t) - h(t-1)||² / ||h(t)||²  (reuse diff_oracle)
    nmse_reuse = (diff_oracle.flatten(1).pow(2).sum(1) /
                  h_real[1:].flatten(1).pow(2).sum(1).clamp(min=1e-12)).numpy()

    return h_ls, deltas_ls, deltas_oracle, nmse_reuse


def _vectorized_scheduling(deltas: np.ndarray,
                           tau_low: float = None, tau_high: float = None,
                           tau_full: float = None, delta_min: float = None,
                           alpha_mode: str = "ramp",
                           alpha_fixed: float = None,
                           n_max: int = 50) -> dict:
    """Vectorized scheduling with continuous alpha support.

    Modes:
      - "step" (legacy): discrete 3-tier from tau_low/tau_high.
        When tau_low/tau_high are provided without tau_full/delta_min,
        automatically sets alpha_mode="step".
      - "ramp": continuous alpha = clip((delta - delta_min) / (tau_full - delta_min), 0, 1).

    Args:
        alpha_fixed: fixed alpha for step mode's mid-tier (default 0.5).

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

    # For step mode, quantize alphas to {0, fixed_mid, 1}
    if alpha_mode == "step":
        # alpha==0 where delta <= delta_min (skip)
        # alpha==1 where delta > tau_full (full)
        # in-between gets alpha_fixed (legacy EMA default 0.5)
        _af = alpha_fixed if alpha_fixed is not None else 0.5
        mid_mask = (alphas_inner > 0) & (~full_inner)
        alphas_inner[mid_mask] = _af

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
    tiers = np.ones(T, dtype=np.int32)  # default: delta (tier 1)
    tiers[full_mask] = 2
    tiers[alphas < 1e-8] = 0
    tiers[full_mask] = 2  # ensure full overrides

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
    alphas[full_mask] = 1.0     # placeholder for full slots

    return _compute_nmse_from_alphas(h_real, h_ls, alphas, full_mask,
                                      delta_mode=delta_mode, h_ce=h_ce)


def _batch_nmse_multi_tau(h_real: torch.Tensor, h_ls: torch.Tensor,
                           deltas: np.ndarray, tau_values: list,
                           alpha: float = 0.5, delta_mode: str = "ema",
                           h_ce: torch.Tensor = None,
                           alpha_mode: str = "step") -> list:
    """Compute NMSE for multiple tau values in parallel.

    Each tau is independent (same h_ls, different scheduling).
    Pre-converts tensors to numpy once, then uses ThreadPool on pure-numpy work
    (no GIL contention since numpy releases GIL for C-level ops).

    Args:
        alpha_mode: "step" (legacy 3-tier) or "ramp" (continuous alpha).
    """
    from concurrent.futures import ThreadPoolExecutor

    # Pre-convert to numpy ONCE (avoid repeated conversion per tau)
    T = h_real.shape[0]
    _h_real_np = h_real.reshape(T, -1).numpy() if h_real.is_contiguous() else h_real.reshape(T, -1).contiguous().numpy()
    _h_ls_np = h_ls.reshape(T, -1).numpy() if h_ls.is_contiguous() else h_ls.reshape(T, -1).contiguous().numpy()
    _h_ce_np = _h_ls_np if h_ce is None else (h_ce.reshape(T, -1).numpy() if h_ce.is_contiguous() else h_ce.reshape(T, -1).contiguous().numpy())

    def _one_tau(tau):
        sched = _vectorized_scheduling(deltas, tau_full=2*tau, delta_min=tau,
                                        alpha_mode=alpha_mode)
        alphas = sched["alphas"]
        full_mask = sched["full_mask"]

        # Inline fast NMSE computation (pure numpy, GIL-free)
        h_hat = np.empty_like(_h_real_np)
        h_hat[0] = _h_ce_np[0]
        anchors = np.where(full_mask)[0]
        h_hat[anchors] = _h_ce_np[anchors]
        seg_starts = anchors
        seg_ends = np.append(anchors[1:], T)

        for start, end in zip(seg_starts, seg_ends):
            if start + 1 >= end:
                continue
            seg_alphas = alphas[start + 1 : end]
            if (seg_alphas < 1e-8).all():
                h_hat[start + 1 : end] = h_hat[start]
                continue
            val = h_hat[start]
            for t in range(start + 1, end):
                a = alphas[t]
                if a < 1e-8:
                    h_hat[t] = val
                else:
                    val = a * _h_ls_np[t] + (1.0 - a) * val
                    h_hat[t] = val

        diff = h_hat - _h_real_np
        nmse = (diff * diff).sum(axis=1) / np.maximum((_h_real_np * _h_real_np).sum(axis=1), 1e-12)

        return {
            "tau": tau, "nmse_arr": nmse,
            "n_skip": sched["n_skip"], "n_delta": sched["n_delta"],
            "n_full": sched["n_full"], "total": sched["total"],
            "tiers": sched["tiers"], "alphas": sched["alphas"],
        }

    with ThreadPoolExecutor(max_workers=min(len(tau_values), 32)) as pool:
        results = list(pool.map(_one_tau, tau_values))
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
        sched = _vectorized_scheduling(deltas, tau_low=tau, tau_high=2 * tau,
                                        n_max=n_max, alpha_mode=alpha_mode)

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
