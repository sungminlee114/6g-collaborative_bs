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


def _precompute_deltas(h_real: torch.Tensor, snr_db: float = 20.0):
    """Precompute LS estimates and δ series for a UE (shared across all tau/mode sweeps).

    Returns:
        h_ls: (T, 2, n_ant, n_sc) noisy LS
        deltas: (T-1,) normalized delta values
        nmse_reuse: (T-1,) NMSE if reusing previous slot
    """
    T = h_real.shape[0]
    sig_pow = h_real.flatten(1).pow(2).mean(1, keepdim=True).sqrt().unsqueeze(-1).unsqueeze(-1)
    h_ls = h_real + (sig_pow / (10 ** (snr_db / 20))) * torch.randn_like(h_real)

    # Vectorized δ: ||h_ls(t) - h_ls(t-1)|| / ||h_ls(t-1)||
    diff = h_ls[1:] - h_ls[:-1]  # (T-1, 2, n_ant, n_sc)
    diff_norm = diff.flatten(1).norm(dim=1)  # (T-1,)
    ref_norm = h_ls[:-1].flatten(1).norm(dim=1).clamp(min=1e-12)  # (T-1,)
    deltas = (diff_norm / ref_norm).numpy()  # (T-1,)

    # Vectorized NMSE(reuse): ||h(t) - h(t-1)||² / ||h(t)||²
    true_diff = h_real[1:] - h_real[:-1]
    nmse_reuse = (true_diff.flatten(1).pow(2).sum(1) /
                  h_real[1:].flatten(1).pow(2).sum(1).clamp(min=1e-12)).numpy()

    return h_ls, deltas, nmse_reuse


def _vectorized_scheduling(deltas: np.ndarray, tau_low: float, tau_high: float,
                           n_max: int = 50) -> dict:
    """Vectorized scheduling decision for a single tau (no Python loop over T).

    Returns dict with n_skip, n_delta, n_full, total, tiers.
    """
    T = len(deltas) + 1
    tiers = np.full(T, 2, dtype=np.int32)  # default: full CE
    tiers[0] = 2  # first slot always full

    # Classify tiers based on delta thresholds
    skip_mask = deltas <= tau_low        # T0
    delta_mask = (deltas > tau_low) & (deltas <= tau_high)  # T1
    full_mask = deltas > tau_high        # T2

    tiers[1:][skip_mask] = 0
    tiers[1:][delta_mask] = 1
    tiers[1:][full_mask] = 2

    # Safety: force full CE after n_max consecutive non-full slots
    if n_max < T:
        consecutive = 0
        for t in range(1, T):
            if tiers[t] != 2:
                consecutive += 1
                if consecutive >= n_max:
                    tiers[t] = 2
                    consecutive = 0
            else:
                consecutive = 0

    return {
        "n_skip": int((tiers == 0).sum()),
        "n_delta": int((tiers == 1).sum()),
        "n_full": int((tiers == 2).sum()),
        "total": T,
        "tiers": tiers,
    }


def _compute_nmse_from_tiers(h_real: torch.Tensor, h_ls: torch.Tensor,
                              tiers: np.ndarray, alpha: float = 0.5,
                              delta_mode: str = "ema") -> np.ndarray:
    """Compute per-slot NMSE given scheduling decisions."""
    T = h_real.shape[0]
    h_hat = torch.zeros_like(h_real)
    h_hat[0] = h_ls[0]

    for t in range(1, T):
        tier = tiers[t]
        if tier == 2:
            h_hat[t] = h_ls[t]
        elif tier == 0:
            h_hat[t] = h_hat[t - 1]
        else:
            if delta_mode == "ema":
                h_hat[t] = alpha * h_ls[t] + (1 - alpha) * h_hat[t - 1]
            elif delta_mode == "ls_delta":
                h_hat[t] = h_hat[t - 1] + alpha * (h_ls[t] - h_ls[t - 1])
            else:
                h_hat[t] = h_hat[t - 1]

    return nmse_per_slot(h_hat, h_real).numpy()


def _batch_scheduling_sweep(h_real: torch.Tensor, h_ls: torch.Tensor,
                             deltas: np.ndarray, tau_values: list,
                             alpha: float = 0.5, delta_mode: str = "ema",
                             n_max: int = 50) -> list:
    """Batch tau sweep: compute scheduling + NMSE for all taus in one pass.

    Shares h_ls and deltas across taus. The sequential h_hat loop runs once per tau
    but uses C-speed numpy for tier classification.

    Returns list of dicts: [{tau, nmse_arr, n_skip, n_delta, n_full, total, tiers}, ...]
    """
    T = h_real.shape[0]
    results = []

    # Pre-flatten for NMSE computation
    h_real_flat = h_real.flatten(1)
    h_real_sq = h_real_flat.pow(2).sum(1)  # (T,)

    for tau in tau_values:
        sched = _vectorized_scheduling(deltas, tau_low=tau, tau_high=2 * tau, n_max=n_max)
        tiers = sched["tiers"]

        # Build h_hat — sequential but single pass
        h_hat = torch.zeros_like(h_real)
        h_hat[0] = h_ls[0]
        for t in range(1, T):
            tier = tiers[t]
            if tier == 2:
                h_hat[t] = h_ls[t]
            elif tier == 0:
                h_hat[t] = h_hat[t - 1]
            else:
                if delta_mode == "ema":
                    h_hat[t] = alpha * h_ls[t] + (1 - alpha) * h_hat[t - 1]
                elif delta_mode == "ls_delta":
                    h_hat[t] = h_hat[t - 1] + alpha * (h_ls[t] - h_ls[t - 1])
                else:
                    h_hat[t] = h_hat[t - 1]

        # Vectorized NMSE
        diff_sq = (h_hat - h_real).flatten(1).pow(2).sum(1)
        nmse_arr = (diff_sq / h_real_sq.clamp(min=1e-12)).numpy()

        results.append({
            "tau": tau, "nmse_arr": nmse_arr, "tiers": tiers,
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
        h_ls, deltas, _ = _precompute_deltas(h, snr_db)
        batch = _batch_scheduling_sweep(h, h_ls, deltas, tau_values)
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
        h_ls, deltas, _ = _precompute_deltas(h, snr_db)
        batch = _batch_scheduling_sweep(h, h_ls, deltas, tau_values)

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
        h_ls, deltas, _ = _precompute_deltas(h, snr_db)
        sched = _vectorized_scheduling(deltas, tau_low=tau_low, tau_high=2 * tau_low)

        for mode, alpha, key in combos:
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

        for snr in snr_list:
            snr_lin = 10 ** (snr / 10)
            h_ls, deltas, _ = _precompute_deltas(h, snr)
            batch = _batch_scheduling_sweep(h, h_ls, deltas, tau_list)

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
