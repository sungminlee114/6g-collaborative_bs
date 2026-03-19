"""Unified experiment pipeline: load each UE once, run all analyses (S0~S7).

Avoids repeated CIR→CFR by processing all experiments per-UE in a single pass.

Usage:
    from src.experiments.pipeline import run_all_experiments
    results = run_all_experiments("munich_elaa_m_1k_15g", max_snapshots=4000, ue_per_speed=2)
"""
import numpy as np
import torch
from typing import Optional

from src.ce_skip import ExperimentConfig
from src.ce_skip.temporal_dataset import TemporalChannelData
from src.ce_skip.ce_algorithms import LSEstimator
from src.ce_skip.scheduling import adaptive_ce_scheduling, compute_scheduling_cost
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


def run_all_experiments(
    preset: str,
    max_snapshots: int = 4000,
    ue_per_speed: int = 2,
    snr_db: float = 20.0,
    tau_values: Optional[list] = None,
    delta_modes: Optional[list] = None,
    alpha_values: Optional[list] = None,
    snr_list: Optional[list] = None,
) -> dict:
    """Run S0, S2, S3, S4, S5, S7 in a single UE-by-UE pass.

    Returns dict with keys: s0, s2, s3, s4, s5, s7.
    """
    from tqdm.auto import tqdm

    if tau_values is None:
        tau_values = np.linspace(0.01, 0.8, 30).tolist()
    if delta_modes is None:
        delta_modes = ["skip", "ema", "ls_delta"]
    if alpha_values is None:
        alpha_values = [0.3, 0.5, 0.7]
    if snr_list is None:
        snr_list = [10.0, 15.0, 20.0, 25.0]

    cfg = ExperimentConfig.from_preset(preset)
    data = TemporalChannelData(
        cfg.temporal_dir, trajectory_dir=cfg.trajectory_dir,
        max_snapshots=max_snapshots, preset=preset,
    )
    print(f"Loaded {preset}: {data.num_snapshots} snaps, {data.num_ue} UEs, dt={data.dt_s}s")

    r_rayleigh = _rayleigh_distance(cfg)
    print(f"Rayleigh distance: {r_rayleigh:.1f} m")

    # ── Pre-filter valid UEs ──
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

    # ── Accumulators ──
    s0_per_ue = []
    s2_per_tau = {tau: {"nmse": [], "n_skip": 0, "n_delta": 0, "n_full": 0, "total": 0}
                  for tau in tau_values}
    s3_per_ue = []  # per-UE with distance + zone
    s4_results = {}
    s5_combo = {}
    for snr in snr_list:
        for tau in [0.1, 0.2, 0.3, 0.5]:
            s5_combo[(snr, tau)] = {"rate_loss": [], "rpr": [], "smr": []}

    ce = LSEstimator()
    T = max_snapshots
    n_tau = len(tau_values)
    n_s4_combos = len(delta_modes) * max(len(alpha_values), 1)
    n_s5_combos = len(snr_list) * 4

    # ── Main loop ──
    for ue_idx, (bs_id, uid) in enumerate(all_ue_ids):
        speed = data.get_ue_speed(uid)
        dist = data.get_ue_distance(uid, bs_id, snap_idx=0)
        spd_label = SPEED_LABELS.get(round(speed, 1), f"v{speed:.1f}")
        print(f"\n── UE{uid} ({spd_label}, {dist:.0f}m) [{ue_idx+1}/{len(all_ue_ids)}] ──")

        # === LOAD ONCE ===
        print(f"  Loading CIR→CFR ({T} snaps)...")
        h_complex = data.get_ue_series(uid, bs_id, snap_range=(0, T))
        h_real = torch.from_numpy(
            np.stack([h_complex.real, h_complex.imag], axis=1).astype(np.float32)
        )
        T_actual = h_real.shape[0]

        # === S0: δ oracle + LS ===
        print(f"  S0: computing δ...", end="", flush=True)
        noise_std = h_real.flatten(1).pow(2).mean(1, keepdim=True).sqrt().unsqueeze(-1).unsqueeze(-1)
        noise_std = noise_std / (10 ** (snr_db / 20))
        h_ls = h_real + noise_std * torch.randn_like(h_real)

        deltas_oracle, deltas_ls, nmse_reuse = [], [], []
        for t in range(1, T_actual):
            ref = h_real[t - 1].norm()
            if ref > 1e-12:
                deltas_oracle.append(float((h_real[t] - h_real[t - 1]).norm() / ref))
                ht_norm = h_real[t].norm()
                if ht_norm > 1e-12:
                    nmse_reuse.append(float((h_real[t] - h_real[t - 1]).pow(2).sum() / h_real[t].pow(2).sum()))
            else:
                deltas_oracle.append(float("inf"))
            ref_ls = h_ls[t - 1].norm()
            if ref_ls > 1e-12:
                deltas_ls.append(float((h_ls[t] - h_ls[t - 1]).norm() / ref_ls))
            else:
                deltas_ls.append(float("inf"))

        d_o = np.array([d for d in deltas_oracle if np.isfinite(d)])
        d_l = np.array([d for d in deltas_ls if np.isfinite(d)])
        nm_arr = np.array(nmse_reuse)

        if len(d_o) > 0:
            ue_s0 = {
                "uid": int(uid), "speed": float(speed), "dist": float(dist),
                "median_delta": float(np.median(d_o)),
                "median_delta_ls": float(np.median(d_l)) if len(d_l) > 0 else float("nan"),
                "mean_delta": float(np.mean(d_o)),
                "p90_delta": float(np.percentile(d_o, 90)),
                "nmse_reuse_db": float(10 * np.log10(max(np.mean(nm_arr), 1e-30))) if len(nm_arr) > 0 else float("nan"),
            }
            s0_per_ue.append(ue_s0)
            print(f" δ={ue_s0['median_delta']:.4f}, δ_LS={ue_s0['median_delta_ls']:.4f}")
        else:
            print(f" SKIP (all-zero)")
            del h_real, h_ls, h_complex
            continue

        # === S2: Threshold sweep ===
        print(f"  S2: τ sweep ({n_tau} values)...", end="", flush=True)
        for tau in tqdm(tau_values, desc="    S2 τ", leave=False, unit="τ"):
            h_hat, stats = adaptive_ce_scheduling(h_real, ce, tau_low=tau, tau_high=2 * tau, snr_db=snr_db)
            nm = nmse_per_slot(h_hat, h_real).numpy()
            pt = s2_per_tau[tau]
            pt["nmse"].extend(nm.tolist())
            pt["n_skip"] += stats.n_skip
            pt["n_delta"] += stats.n_delta
            pt["n_full"] += stats.n_full
            pt["total"] += stats.total
        print(f" done")

        # === S3: Distance classification ===
        zone = "NF" if dist < r_rayleigh else ("Transition" if dist < 3 * r_rayleigh else "FF")
        # Per-UE tau sweep for distance analysis
        ue_s3_sweep = {}
        for tau in tau_values:
            h_hat, stats = adaptive_ce_scheduling(h_real, ce, tau_low=tau, tau_high=2 * tau, snr_db=snr_db)
            nm = nmse_per_slot(h_hat, h_real).numpy()
            ue_s3_sweep[float(tau)] = {"nmse": float(np.mean(nm)), "skip_rate": stats.skip_rate}
        s3_per_ue.append({
            "uid": int(uid), "dist": float(dist), "speed": float(speed),
            "zone": zone, "sweep": ue_s3_sweep,
        })

        # === S4: Delta update ablation ===
        print(f"  S4: ablation ({n_s4_combos} modes)...", end="", flush=True)
        for mode in delta_modes:
            for alpha in (alpha_values if mode != "skip" else [0.0]):
                key = f"{mode}_a{alpha:.1f}" if mode != "skip" else "skip"
                if key not in s4_results:
                    s4_results[key] = {}
                if spd_label not in s4_results[key]:
                    s4_results[key][spd_label] = {"nmse": [], "skip_rates": []}

                h_hat, stats = adaptive_ce_scheduling(
                    h_real, ce, tau_low=0.2, tau_high=0.4,
                    alpha=alpha, snr_db=snr_db, delta_mode=mode,
                )
                nm = nmse_per_slot(h_hat, h_real).numpy()
                s4_results[key][spd_label]["nmse"].extend(nm.tolist())
                s4_results[key][spd_label]["skip_rates"].append(stats.skip_rate)
        print(f" done")

        # === S5: Beamforming rate impact ===
        print(f"  S5: beamforming ({n_s5_combos} combos)...", end="", flush=True)
        for snr in snr_list:
            snr_lin = 10 ** (snr / 10)
            for tau in [0.1, 0.2, 0.3, 0.5]:
                h_hat, stats = adaptive_ce_scheduling(h_real, ce, tau_low=tau, tau_high=2 * tau, snr_db=snr)
                rl = rate_loss_per_slot(h_hat, h_real, snr_lin)
                s5_combo[(snr, tau)]["rate_loss"].extend(rl.numpy().tolist())
                s5_combo[(snr, tau)]["rpr"].append(
                    rate_preservation_ratio(h_hat, h_real, snr_lin)["rpr_oracle"])
                s5_combo[(snr, tau)]["smr"].append(
                    skip_miss_rate(h_hat, h_real, stats.tiers, snr_lin, 0.05))
        print(f" done")

        del h_real, h_ls, h_complex

    # ══════════════════════════════════════════════════════════════
    #  Aggregate results
    # ══════════════════════════════════════════════════════════════
    print(f"\nAggregating results...")

    # S0
    all_d = [u["median_delta"] for u in s0_per_ue]
    static_d = [u["median_delta"] for u in s0_per_ue if u["speed"] < 0.1]
    s0_summary = {
        "n_ues": len(s0_per_ue),
        "median_delta": float(np.median(all_d)) if all_d else float("nan"),
        "static_median_delta": float(np.median(static_d)) if static_d else float("nan"),
        "median_nmse_reuse_db": float(np.median([u["nmse_reuse_db"] for u in s0_per_ue
                                                   if not np.isnan(u["nmse_reuse_db"])])) if s0_per_ue else float("nan"),
    }

    # S2
    s2_sweep = []
    for tau in tau_values:
        pt = s2_per_tau[tau]
        avg_nmse = np.mean(pt["nmse"]) if pt["nmse"] else 1.0
        total = pt["total"]
        s2_sweep.append({
            "tau": float(tau),
            "avg_nmse_db": float(10 * np.log10(max(avg_nmse, 1e-30))),
            "skip_rate": pt["n_skip"] / max(total, 1),
            "computation_ratio": (pt["n_full"] * 0.5 + (pt["n_delta"] + pt["n_skip"]) * 0.005) / max(total * 0.5, 1e-12),
        })

    # S3
    s3_zones = {}
    for zone_name in ["NF", "Transition", "FF"]:
        ues = [u for u in s3_per_ue if u["zone"] == zone_name]
        if not ues:
            s3_zones[zone_name] = {"count": 0, "optimal_tau": None}
            continue
        optimal_tau = tau_values[0]
        sweep_agg = []
        for tau in tau_values:
            avg_nmse = np.mean([u["sweep"][str(float(tau))]["nmse"] for u in ues])
            avg_nmse_db = 10 * np.log10(max(avg_nmse, 1e-30))
            avg_sr = np.mean([u["sweep"][str(float(tau))]["skip_rate"] for u in ues])
            sweep_agg.append({"tau": float(tau), "nmse_db": avg_nmse_db, "skip_rate": avg_sr})
            if avg_nmse_db <= -10.0:
                optimal_tau = tau
        s3_zones[zone_name] = {"count": len(ues), "optimal_tau": float(optimal_tau), "sweep": sweep_agg}

    # S4
    s4_agg = {}
    for key, speed_data in s4_results.items():
        s4_agg[key] = {}
        for sl, vals in speed_data.items():
            s4_agg[key][sl] = {
                "nmse_db": float(10 * np.log10(max(np.mean(vals["nmse"]), 1e-30))),
                "skip_rate": float(np.mean(vals["skip_rates"])),
            }

    # S5
    s5_agg = {}
    for snr in snr_list:
        sk = f"snr{int(snr)}"
        s5_agg[sk] = {}
        for tau in [0.1, 0.2, 0.3, 0.5]:
            cd = s5_combo[(snr, tau)]
            rl = np.array(cd["rate_loss"])
            s5_agg[sk][f"tau{tau}"] = {
                "mean_rate_loss": float(np.mean(rl)),
                "frac_above_5pct": float(np.mean(rl > 0.05)),
                "avg_rpr": float(np.mean(cd["rpr"])),
                "avg_smr": float(np.mean(cd["smr"])),
            }

    # S7: System overhead (pure computation, no data needed)
    n_ant = cfg.num_rx_ant * cfg.num_tx_ant
    n_sc = cfg.num_subcarriers
    overhead = scheduling_overhead(n_ant, n_sc, t_full_ms=0.5)

    # Find best S2 operating point for S7
    sweet_s2 = [s for s in s2_sweep if s["avg_nmse_db"] <= -10]
    if sweet_s2:
        best = min(sweet_s2, key=lambda s: s["computation_ratio"])
        eff = effective_throughput_gain(best["skip_rate"], 0.97, best["computation_ratio"])
    else:
        eff = effective_throughput_gain(0.0, 1.0, 1.0)

    data.clear_cache()

    print(f"\n{'═'*50}")
    print(f"  Pipeline complete: {len(s0_per_ue)} UEs × {T} snaps")
    print(f"  S0: median δ={s0_summary['median_delta']:.4f}")
    if sweet_s2:
        print(f"  S2: sweet spot τ={best['tau']:.2f}, CR={best['computation_ratio']:.3f}")
    print(f"  S3: zones NF={s3_zones['NF']['count']}, Trans={s3_zones['Transition']['count']}, FF={s3_zones['FF']['count']}")
    print(f"  S7: overhead={overhead['overhead_ratio']:.4%}, capacity={eff['capacity_multiplier']:.1f}x")
    print(f"{'═'*50}")

    return {
        "preset": preset,
        "config": {
            "max_snapshots": max_snapshots, "ue_per_speed": ue_per_speed,
            "snr_db": snr_db, "r_rayleigh": r_rayleigh,
            "n_ant": n_ant, "n_sc": n_sc,
        },
        "s0": {"per_ue": s0_per_ue, "summary": s0_summary},
        "s2": {"sweep": s2_sweep},
        "s3": {"zones": s3_zones, "per_ue": [
            {"uid": u["uid"], "dist": u["dist"], "speed": u["speed"], "zone": u["zone"]}
            for u in s3_per_ue
        ], "r_rayleigh": r_rayleigh},
        "s4": s4_agg,
        "s5": s5_agg,
        "s7": {"overhead": overhead, "effective_throughput": eff},
    }
