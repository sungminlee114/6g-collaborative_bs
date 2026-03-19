"""S0 core: shared logic for CLI (run.py) and notebook (analysis.ipynb).

All data loading, delta computation, and aggregation lives here.
"""
import json
from pathlib import Path
from typing import Optional

import numpy as np

from src.ce_skip import ExperimentConfig, PRIMARY_PRESETS
from src.ce_skip.temporal_dataset import TemporalChannelData
from src.ce_skip.helpers import iter_ue_tensors


def load_data(preset: str, max_snapshots: int = 20000) -> TemporalChannelData:
    """Load temporal channel data for a preset. Returns TemporalChannelData instance."""
    import os
    # Ensure we're at project root (notebooks may run from subdirs)
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    os.chdir(project_root)
    cfg = ExperimentConfig.from_preset(preset)
    if not cfg.temporal_dir.exists():
        raise FileNotFoundError(f"Temporal data not found: {cfg.temporal_dir}")
    data = TemporalChannelData(
        cfg.temporal_dir, trajectory_dir=cfg.trajectory_dir,
        max_snapshots=max_snapshots, preset=preset,
    )
    print(f"Loaded {preset}: {data.num_snapshots} snapshots, {data.num_ue} UEs, dt={data.dt_s}s")
    return data


def compute_delta_profile(data: TemporalChannelData, bs_id: int,
                          max_snapshots: int = None, ue_per_speed: int = None) -> dict:
    """Compute δ_oracle(t) for all UEs served by a BS.

    Args:
        max_snapshots: limit snapshots per UE (None = all)
        ue_per_speed: if set, only use N UEs per speed category (0, 1, 8.3 m/s)

    Returns dict with:
        per_ue: list of {uid, speed, dist, median_delta, mean_delta, p90_delta, nmse_reuse_db}
        overall: aggregate statistics
    """
    per_ue = []

    T = max_snapshots or data.num_snapshots

    # Filter UEs: pick N per speed category if requested
    ue_filter = None
    if ue_per_speed is not None and data.ue_bs_ids is not None and data.speeds is not None:
        ue_ids_for_bs = [i for i, b in enumerate(data.ue_bs_ids) if b == bs_id]
        selected = []
        speed_counts = {}
        for uid in ue_ids_for_bs:
            spd = round(float(data.speeds[uid]), 1)
            speed_counts.setdefault(spd, 0)
            if speed_counts[spd] < ue_per_speed:
                selected.append(uid)
                speed_counts[spd] += 1
        ue_filter = set(selected)

    # Load all selected UEs at once (batch CIR→CFR, uses all GPUs)
    import torch
    target_ues = sorted(ue_filter) if ue_filter else None
    h_all, ue_ids_loaded = data.get_all_series(bs_id, snap_range=(0, T))
    # h_all: (N_UE_bs, T, n_ant, n_sc) complex

    if len(ue_ids_loaded) == 0:
        return {"per_ue": [], "overall": {"n_ues": 0, "median_delta_all": float("nan"),
                "mean_delta_all": float("nan"), "static_median_delta": float("nan"),
                "static_n": 0, "frac_below_0.1": 0, "frac_below_0.3": 0}}

    for i, uid in enumerate(ue_ids_loaded):
        if target_ues is not None and uid not in target_ues:
            continue

        h_complex = h_all[i]  # (T, n_ant, n_sc) complex
        h_real = torch.from_numpy(
            np.stack([h_complex.real, h_complex.imag], axis=1).astype(np.float32)
        )  # (T, 2, n_ant, n_sc)

        dist = data.get_ue_distance(uid, bs_id, snap_idx=0)
        speed = data.get_ue_speed(uid)

        T_actual = h_real.shape[0]
        deltas = []
        nmse_reuse = []
        for t in range(1, T_actual):
            h_t = h_real[t]
            h_prev = h_real[t - 1]
            ref_norm = h_prev.norm()
            if ref_norm > 1e-12:
                delta = float((h_t - h_prev).norm() / ref_norm)
                h_t_norm = h_t.norm()
                if h_t_norm > 1e-12:
                    nmse_reuse.append(float((h_t - h_prev).pow(2).sum() / h_t.pow(2).sum()))
            else:
                delta = float("inf")
            deltas.append(delta)

        d_arr = np.array([d for d in deltas if np.isfinite(d)])
        nm_arr = np.array(nmse_reuse)
        if len(d_arr) == 0:
            continue

        per_ue.append({
            "uid": int(uid), "speed": float(speed), "dist": float(dist),
            "median_delta": float(np.median(d_arr)),
            "mean_delta": float(np.mean(d_arr)),
            "p90_delta": float(np.percentile(d_arr, 90)),
            "nmse_reuse_db": float(10 * np.log10(max(np.mean(nm_arr), 1e-30))) if len(nm_arr) > 0 else float("nan"),
        })

    all_deltas = [u["median_delta"] for u in per_ue]
    static_deltas = [u["median_delta"] for u in per_ue if u["speed"] == 0.0]

    overall = {
        "n_ues": len(per_ue),
        "median_delta_all": float(np.median(all_deltas)) if all_deltas else float("nan"),
        "mean_delta_all": float(np.mean(all_deltas)) if all_deltas else float("nan"),
        "static_median_delta": float(np.median(static_deltas)) if static_deltas else float("nan"),
        "static_n": len(static_deltas),
        "frac_below_0.1": float(np.mean(np.array(all_deltas) < 0.1)) if all_deltas else 0.0,
        "frac_below_0.3": float(np.mean(np.array(all_deltas) < 0.3)) if all_deltas else 0.0,
    }
    return {"per_ue": per_ue, "overall": overall}


def run_all_bs(data: TemporalChannelData, preset: str, gpu: str = None,
               max_snapshots: int = None, ue_per_speed: int = None) -> dict:
    """Run delta profile across all BSs with UEs.

    Args:
        gpu: CUDA device string, e.g. "0" or "0,1". None = use all available.
        max_snapshots: limit snapshots per UE
        ue_per_speed: N UEs per speed category (e.g. 1 = 3 UEs total)
    """
    import os
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    cfg = ExperimentConfig.from_preset(preset)
    all_bs = sorted(set(cfg.train_bs_ids + cfg.val_bs_ids + cfg.test_bs_ids))

    # Only process BSs that have UEs
    bs_with_ues = []
    if data.ue_bs_ids is not None:
        for bs_id in all_bs:
            n = sum(1 for b in data.ue_bs_ids if b == bs_id)
            if n > 0:
                bs_with_ues.append(bs_id)
            else:
                print(f"  BS {bs_id}: 0 UEs, skipping")
    else:
        bs_with_ues = all_bs

    all_per_ue = []
    for bs_id in bs_with_ues:
        profile = compute_delta_profile(data, bs_id, max_snapshots=max_snapshots, ue_per_speed=ue_per_speed)
        all_per_ue.extend(profile["per_ue"])
        ov = profile["overall"]
        print(f"  BS {bs_id}: {ov['n_ues']} UEs, median δ={ov['median_delta_all']:.4f}, "
              f"static δ={ov['static_median_delta']:.4f}")

    all_deltas = [u["median_delta"] for u in all_per_ue]
    all_nmse = [u["nmse_reuse_db"] for u in all_per_ue if not np.isnan(u["nmse_reuse_db"])]
    static_deltas = [u["median_delta"] for u in all_per_ue if u["speed"] == 0.0]

    summary = {
        "n_ues": len(all_per_ue),
        "median_delta": float(np.median(all_deltas)) if all_deltas else float("nan"),
        "mean_delta": float(np.mean(all_deltas)) if all_deltas else float("nan"),
        "frac_below_0.1": float(np.mean(np.array(all_deltas) < 0.1)) if all_deltas else 0.0,
        "frac_below_0.3": float(np.mean(np.array(all_deltas) < 0.3)) if all_deltas else 0.0,
        "frac_below_0.5": float(np.mean(np.array(all_deltas) < 0.5)) if all_deltas else 0.0,
        "median_nmse_reuse_db": float(np.median(all_nmse)) if all_nmse else float("nan"),
        "static_median_delta": float(np.median(static_deltas)) if static_deltas else float("nan"),
        "static_n": len(static_deltas),
    }
    return {"preset": preset, "summary": summary, "per_ue": all_per_ue}


def check_go_nogo(results: list) -> bool:
    """Check kill criterion: median δ > 0.5 → no-go."""
    go = True
    for r in results:
        if r is None:
            continue
        s = r["summary"]
        med = s["median_delta"]
        static = s["static_median_delta"]
        if med > 0.5:
            print(f"  ✗ KILL: {r['preset']} median δ = {med:.3f} > 0.5")
            go = False
        else:
            print(f"  ✓ GO: {r['preset']} median δ = {med:.3f} ≤ 0.5")
        if static > 0.01:
            print(f"    ⚠ Static δ = {static:.4f} (simulator noise)")
    return go


def save_results(result: dict, results_dir: str = "assets/results/S0_persistence"):
    """Save per-preset + combined results."""
    rd = Path(results_dir)
    rd.mkdir(parents=True, exist_ok=True)
    out = rd / f"persistence_{result['preset']}.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {out}")
    return out


def load_results(preset: str, results_dir: str = "assets/results/S0_persistence") -> Optional[dict]:
    """Load previously saved results."""
    path = Path(results_dir) / f"persistence_{preset}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None
