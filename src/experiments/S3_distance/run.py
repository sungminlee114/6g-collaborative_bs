"""S3: Distance-Dependent Threshold (H3 — NF-specific contribution).

Groups UEs by distance zone (NF / Transition / FF based on Rayleigh distance)
and finds optimal τ* per zone.

Key insight: τ*_NF < τ*_Transition < τ*_FF
(near-field channels change faster → need tighter threshold)

This effect should be absent in Config C (5G, all FF).

Usage:
    python -m src.experiments.S3_distance.run --preset munich_elaa_l_1k_15g --gpu 0
    python -m src.experiments.S3_distance.run --all
"""
import argparse
import json

import numpy as np
import torch

from src.config import get_results_dir
from src.tracker import Tracker
from src.ce_skip import ExperimentConfig
from src.ce_skip.temporal_dataset import TemporalChannelData
from src.ce_skip.scheduling import adaptive_ce_scheduling
from src.ce_skip.metrics import nmse_per_slot, rate_preservation_ratio


from src.ce_skip import PRIMARY_PRESETS  # noqa: E402 (defined in ce_skip/__init__)
# Presets imported from ce_skip — change there to update all experiments


def rayleigh_distance(cfg: ExperimentConfig) -> float:
    """Compute Rayleigh distance: 2D^2/λ where D = array aperture."""
    c = 3e8
    wavelength = c / cfg.frequency
    # Array aperture (largest dimension)
    d_spacing = wavelength / 2
    D = max(cfg.tx_rows, cfg.tx_cols) * d_spacing
    return 2 * D**2 / wavelength


def classify_zones(distances: np.ndarray, r_rayleigh: float) -> dict:
    """Classify UE distances into NF / Transition / FF zones."""
    zones = {
        "NF": (0, r_rayleigh),
        "Transition": (r_rayleigh, 3 * r_rayleigh),
        "FF": (3 * r_rayleigh, float("inf")),
    }
    assignments = {}
    for zone_name, (d_min, d_max) in zones.items():
        mask = (distances >= d_min) & (distances < d_max)
        assignments[zone_name] = {
            "mask": mask,
            "count": int(mask.sum()),
            "d_min": d_min,
            "d_max": d_max,
        }
    return assignments


def sweep_tau_for_ues(
    h_tensor: torch.Tensor,
    tau_values: np.ndarray,
    snr_db: float = 20.0,
    target_nmse_db: float = -10.0,
) -> dict:
    """Sweep τ for a set of UEs, find optimal τ* at target NMSE.

    Args:
        h_tensor: (N_UE, T, 2, n_ant, n_sc) — channel data
        tau_values: array of τ_low values to sweep
        snr_db: SNR for LS noise
        target_nmse_db: NMSE threshold for finding optimal τ*

    Returns dict with sweep results and optimal τ*.
    """
    from src.ce_skip.ce_algorithms import LSEstimator
    ce = LSEstimator()

    N_ue = h_tensor.shape[0]
    sweep = []

    for tau in tau_values:
        all_nmse = []
        all_skip = 0
        all_total = 0

        for ue_idx in range(N_ue):
            h_hat, stats = adaptive_ce_scheduling(
                h_tensor[ue_idx], ce,
                tau_low=tau, tau_high=2 * tau,
                snr_db=snr_db,
            )
            nm = nmse_per_slot(h_hat, h_tensor[ue_idx]).cpu().numpy()
            all_nmse.extend(nm.tolist())
            all_skip += stats.n_skip
            all_total += stats.total

        avg_nmse = np.mean(all_nmse)
        avg_nmse_db = 10 * np.log10(max(avg_nmse, 1e-30))
        skip_rate = all_skip / max(all_total, 1)

        sweep.append({
            "tau": float(tau),
            "nmse_db": avg_nmse_db,
            "skip_rate": skip_rate,
        })

    # Find optimal τ*: largest τ where NMSE ≤ target
    optimal_tau = tau_values[0]
    for s in sweep:
        if s["nmse_db"] <= target_nmse_db:
            optimal_tau = s["tau"]
        else:
            break

    return {
        "sweep": sweep,
        "optimal_tau": float(optimal_tau),
    }


def run_distance_analysis(preset: str, gpu: int = 0, max_snapshots: int = 200):
    """Run distance-dependent threshold analysis."""
    device = f"cuda:{gpu}"
    torch.cuda.set_device(device)
    cfg = ExperimentConfig.from_preset(preset)

    if not cfg.temporal_dir.exists():
        print(f"  ⚠ Temporal data not found: {cfg.temporal_dir}")
        return None

    data = TemporalChannelData(cfg.temporal_dir, trajectory_dir=cfg.trajectory_dir, max_snapshots=max_snapshots)
    r_rayleigh = rayleigh_distance(cfg)
    print(f"  Rayleigh distance: {r_rayleigh:.1f} m")

    from src.ce_skip.helpers import iter_ue_tensors
    from src.ce_skip.ce_algorithms import LSEstimator
    from src.ce_skip.scheduling import adaptive_ce_scheduling

    test_bs = cfg.test_bs_ids[0]
    tau_values = np.linspace(0.01, 1.0, 40)

    # Collect per-UE data: distance, speed, and per-tau NMSE (UE-by-UE for memory)
    zone_data = {"NF": [], "Transition": [], "FF": []}
    delta_dist = []

    for h_ue, uid, dist, speed in iter_ue_tensors(data, test_bs, device, max_snapshots, max_ue=30):
        # Classify zone
        if dist < r_rayleigh:
            zone = "NF"
        elif dist < 3 * r_rayleigh:
            zone = "Transition"
        else:
            zone = "FF"

        # Sweep tau for this UE
        ce = LSEstimator()
        ue_sweep = {}
        for tau in tau_values:
            h_hat, stats = adaptive_ce_scheduling(h_ue, ce, tau_low=tau, tau_high=2*tau, snr_db=20.0)
            nm = nmse_per_slot(h_hat, h_ue).cpu().numpy()
            ue_sweep[tau] = {"nmse": float(np.mean(nm)), "skip_rate": stats.skip_rate}

        zone_data[zone].append({"uid": uid, "dist": dist, "speed": speed, "sweep": ue_sweep})

        # Delta vs distance
        h_cpu = h_ue.cpu().numpy()
        for t in range(1, h_cpu.shape[0]):
            diff_n = np.linalg.norm(h_cpu[t] - h_cpu[t-1])
            ref_n = np.linalg.norm(h_cpu[t-1])
            if ref_n > 1e-12:
                delta_dist.append({"distance": float(dist), "delta": float(diff_n/ref_n), "speed": float(speed)})

    results = {"r_rayleigh": r_rayleigh, "zones": {}}

    for zone_name in ["NF", "Transition", "FF"]:
        ues = zone_data[zone_name]
        n_ues = len(ues)
        if n_ues == 0:
            print(f"  {zone_name}: no UEs")
            results["zones"][zone_name] = {"count": 0, "optimal_tau": None}
            continue

        print(f"  {zone_name}: {n_ues} UEs")

        # Find optimal tau from aggregated sweep data
        optimal_tau = tau_values[0]
        sweep_results = []
        for tau in tau_values:
            avg_nmse = np.mean([u["sweep"][tau]["nmse"] for u in ues])
            avg_nmse_db = 10 * np.log10(max(avg_nmse, 1e-30))
            avg_sr = np.mean([u["sweep"][tau]["skip_rate"] for u in ues])
            sweep_results.append({"tau": float(tau), "nmse_db": avg_nmse_db, "skip_rate": avg_sr})
            if avg_nmse_db <= -10.0:
                optimal_tau = tau

        results["zones"][zone_name] = {
            "count": n_ues,
            "optimal_tau": float(optimal_tau),
            "sweep": sweep_results,
        }
        print(f"    Optimal τ* = {optimal_tau:.3f}")

    results["delta_vs_distance"] = delta_dist

    data.clear_cache()
    return results


def main():
    parser = argparse.ArgumentParser(description="S3: Distance-Dependent Threshold")
    parser.add_argument("--preset", type=str, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max-snapshots", type=int, default=200)
    args = parser.parse_args()

    presets = PRIMARY_PRESETS if args.all else [args.preset or PRIMARY_PRESETS[0]]
    results_dir = get_results_dir("S3_distance")

    with Tracker(
        "S3/distance",
        config={"presets": presets},
        capture_output=True,
        purpose="Distance-dependent threshold analysis (NF-specific)",
        variables={
            "independent": ["distance_zone", "tau"],
            "dependent": ["optimal_tau", "nmse_db", "skip_rate"],
            "controlled": ["snr=20dB", "target_nmse=-10dB"],
        },
        hypothesis="τ*_NF < τ*_Transition < τ*_FF — unique to NF propagation",
        eval_criteria="Compare optimal τ* across zones; absent in Config C (all FF)",
    ) as run:
        all_results = {}
        for preset in presets:
            print(f"\n{'═'*60}")
            print(f"  Preset: {preset}")
            print(f"{'═'*60}")
            r = run_distance_analysis(preset, gpu=args.gpu, max_snapshots=args.max_snapshots)
            if r is not None:
                all_results[preset] = r
                # Log optimal taus per zone
                for zone_name, zone_data in r["zones"].items():
                    if zone_data.get("optimal_tau") is not None:
                        run.log(**{f"{preset}_{zone_name}_tau": zone_data["optimal_tau"]})

        with open(results_dir / "distance_threshold.json", "w") as f:
            json.dump(all_results, f, indent=2, default=str)

        # Summary
        print(f"\n{'═'*60}")
        print("  Distance-Dependent τ* Summary")
        print(f"{'═'*60}")
        print(f"{'Preset':<30} {'R_ray(m)':<10} {'τ*_NF':<8} {'τ*_Trans':<8} {'τ*_FF':<8}")
        for preset, r in all_results.items():
            short = preset.replace("munich_", "")
            taus = {z: r["zones"].get(z, {}).get("optimal_tau", "-") for z in ["NF", "Transition", "FF"]}
            fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)
            print(f"{short:<30} {r['r_rayleigh']:<10.1f} "
                  f"{fmt(taus['NF']):<8} {fmt(taus['Transition']):<8} {fmt(taus['FF']):<8}")

        run.set_result(n_presets=len(all_results))


if __name__ == "__main__":
    main()
