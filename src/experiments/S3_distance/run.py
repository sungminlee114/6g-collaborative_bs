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
from pathlib import Path

import numpy as np
import torch

from src.config import SceneConfig, get_results_dir
from src.tracker import Tracker
from src.ce_skip.temporal_dataset import TemporalChannelData
from src.ce_skip.scheduling import adaptive_ce_scheduling
from src.ce_skip.metrics import nmse_per_slot, rate_preservation_ratio


PRIMARY_PRESETS = [
    "munich_elaa_l_1k_15g",   # R_Rayleigh ≈ 20.5m
    "munich_elaa_l_4k_28g",   # R_Rayleigh ≈ 10.9m
    "munich_5g_mimo_3g5",     # R_Rayleigh ≈ 0.5m (all FF)
]


def rayleigh_distance(cfg: SceneConfig) -> float:
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
    cfg = SceneConfig.from_preset(preset)

    temporal_dir = Path(cfg.data_dir).parent / f"{Path(cfg.data_dir).name}_temporal"
    if not temporal_dir.exists():
        print(f"  ⚠ Temporal data not found: {temporal_dir}")
        return None

    data = TemporalChannelData(temporal_dir, max_snapshots=max_snapshots)
    r_rayleigh = rayleigh_distance(cfg)
    print(f"  Rayleigh distance: {r_rayleigh:.1f} m")

    test_bs = cfg.test_bs_ids[0] if cfg.test_bs_ids else 1
    h_all, ue_ids = data.get_all_series(test_bs, snap_range=(0, min(data.num_snapshots, max_snapshots)))

    if len(ue_ids) == 0:
        print(f"  ⚠ No UEs for BS {test_bs}")
        return None

    # Get distances
    distances = np.array([data.get_ue_distance(uid, test_bs) for uid in ue_ids])

    # Convert to real tensor
    N_ue, T, n_ant, n_sc = h_all.shape
    h_real = np.stack([h_all.real, h_all.imag], axis=2).astype(np.float32)
    h_tensor = torch.from_numpy(h_real).to(device)

    # Classify zones
    zones = classify_zones(distances, r_rayleigh)
    tau_values = np.linspace(0.01, 1.0, 40)

    results = {"r_rayleigh": r_rayleigh, "zones": {}}

    for zone_name, zone_info in zones.items():
        mask = zone_info["mask"]
        n_ues = zone_info["count"]
        if n_ues == 0:
            print(f"  {zone_name}: no UEs in [{zone_info['d_min']:.1f}, {zone_info['d_max']:.1f}) m")
            results["zones"][zone_name] = {"count": 0, "optimal_tau": None}
            continue

        print(f"  {zone_name}: {n_ues} UEs in [{zone_info['d_min']:.1f}, {zone_info['d_max']:.1f}) m")

        h_zone = h_tensor[mask]
        zone_result = sweep_tau_for_ues(h_zone, tau_values, snr_db=20.0, target_nmse_db=-10.0)
        zone_result["count"] = n_ues
        zone_result["d_range"] = [zone_info["d_min"], zone_info["d_max"]]
        results["zones"][zone_name] = zone_result

        print(f"    Optimal τ* = {zone_result['optimal_tau']:.3f}")

    # Also compute δ statistics per distance bin
    print("  Computing δ vs distance...")
    delta_dist = []
    for ue_idx in range(min(N_ue, 30)):
        dist = distances[ue_idx]
        speed = data.get_ue_speed(ue_ids[ue_idx])
        h_ue = h_all[ue_idx]  # (T, n_ant, n_sc) complex
        for t in range(1, T):
            diff = h_ue[t] - h_ue[t - 1]
            ref_norm = np.linalg.norm(h_ue[t - 1])
            if ref_norm > 1e-12:
                delta = float(np.linalg.norm(diff) / ref_norm)
                delta_dist.append({"distance": float(dist), "delta": delta, "speed": float(speed)})

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
