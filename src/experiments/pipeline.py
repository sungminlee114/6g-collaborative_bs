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
from src.ce_skip.ce_algorithms import LSEstimator
from src.ce_skip.scheduling import adaptive_ce_scheduling
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


def run_s2(ue_data: dict, tau_values: list = None, snr_db: float = 20.0) -> dict:
    """S2: Threshold sweep → Pareto front."""
    if tau_values is None:
        tau_values = np.linspace(0.01, 0.8, 30).tolist()

    ce = LSEstimator()
    per_tau = {tau: {"nmse": [], "n_skip": 0, "n_delta": 0, "n_full": 0, "total": 0}
               for tau in tau_values}

    for u in tqdm(ue_data["ue_list"], desc="S2: UEs", unit="ue", position=0):
        h = u["h_real"]
        for tau in tqdm(tau_values, desc=f"  S2 UE{u['uid']} τ", unit="τ", leave=False, position=1):
            h_hat, stats = adaptive_ce_scheduling(h, ce, tau_low=tau, tau_high=2 * tau, snr_db=snr_db)
            nm = nmse_per_slot(h_hat, h).numpy()
            pt = per_tau[tau]
            pt["nmse"].extend(nm.tolist())
            pt["n_skip"] += stats.n_skip
            pt["n_delta"] += stats.n_delta
            pt["n_full"] += stats.n_full
            pt["total"] += stats.total

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
    ce = LSEstimator()
    per_ue = []

    for u in tqdm(ue_data["ue_list"], desc="S3: UEs", unit="ue", position=0):
        h = u["h_real"]
        dist = u["dist"]
        zone = "NF" if dist < r_ray else ("Transition" if dist < 3 * r_ray else "FF")

        sweep = {}
        for tau in tqdm(tau_values, desc=f"  S3 UE{u['uid']} τ", unit="τ", leave=False, position=1):
            h_hat, stats = adaptive_ce_scheduling(h, ce, tau_low=tau, tau_high=2 * tau, snr_db=snr_db)
            nm = nmse_per_slot(h_hat, h).numpy()
            sweep[float(tau)] = {"nmse": float(np.mean(nm)), "skip_rate": stats.skip_rate}

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

    ce = LSEstimator()
    results = {}

    combos = []
    for mode in delta_modes:
        for alpha in (alpha_values if mode != "skip" else [0.0]):
            combos.append((mode, alpha, f"{mode}_a{alpha:.1f}" if mode != "skip" else "skip"))

    for u in tqdm(ue_data["ue_list"], desc="S4: UEs", unit="ue", position=0):
        h = u["h_real"]
        spd_label = SPEED_LABELS.get(round(u["speed"], 1), f"v{u['speed']:.1f}")

        for mode, alpha, key in tqdm(combos, desc=f"  S4 UE{u['uid']}", unit="mode", leave=False, position=1):
                if key not in results:
                    results[key] = {}
                if spd_label not in results[key]:
                    results[key][spd_label] = {"nmse": [], "skip_rates": []}

                h_hat, stats = adaptive_ce_scheduling(
                    h, ce, tau_low=tau_low, tau_high=2 * tau_low,
                    alpha=alpha, snr_db=snr_db, delta_mode=mode,
                )
                nm = nmse_per_slot(h_hat, h).numpy()
                results[key][spd_label]["nmse"].extend(nm.tolist())
                results[key][spd_label]["skip_rates"].append(stats.skip_rate)

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

    ce = LSEstimator()
    combo = {}
    for snr in snr_list:
        for tau in tau_list:
            combo[(snr, tau)] = {"rate_loss": [], "rpr": [], "smr": []}

    s5_combos = [(snr, tau) for snr in snr_list for tau in tau_list]

    for u in tqdm(ue_data["ue_list"], desc="S5: UEs", unit="ue", position=0):
        h = u["h_real"]
        for snr, tau in tqdm(s5_combos, desc=f"  S5 UE{u['uid']} SNR×τ", unit="pt", leave=False, position=1):
            snr_lin = 10 ** (snr / 10)
            h_hat, stats = adaptive_ce_scheduling(h, ce, tau_low=tau, tau_high=2 * tau, snr_db=snr)
            rl = rate_loss_per_slot(h_hat, h, snr_lin)
            combo[(snr, tau)]["rate_loss"].extend(rl.numpy().tolist())
            combo[(snr, tau)]["rpr"].append(rate_preservation_ratio(h_hat, h, snr_lin)["rpr_oracle"])
            combo[(snr, tau)]["smr"].append(skip_miss_rate(h_hat, h, stats.tiers, snr_lin, 0.05))

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
