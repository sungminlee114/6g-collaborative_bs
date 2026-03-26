"""S0 CLI: Temporal Persistence Profiling.

Thin wrapper around core.py for command-line execution with Tracker.

Usage:
    python -m src.experiments.S0_persistence.run --preset munich_elaa_m_1k_28g
    python -m src.experiments.S0_persistence.run --all --max-snapshots 20000
"""
import argparse
import json

from src.config import get_results_dir
from src.tracker import Tracker
from src.ce_skip import PRIMARY_PRESETS
from .core import load_data, run_all_bs, check_go_nogo, save_results


def main():
    parser = argparse.ArgumentParser(description="S0: Temporal Persistence Profiling")
    parser.add_argument("--preset", type=str, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max-snapshots", type=int, default=20000)
    args = parser.parse_args()

    presets = PRIMARY_PRESETS if args.all else [args.preset or PRIMARY_PRESETS[0]]
    results_dir = get_results_dir("S0_persistence")

    with Tracker(
        "S0/persistence",
        config={"presets": presets, "max_snapshots": args.max_snapshots},
        capture_output=True,
        purpose="Temporal persistence: δ_oracle + NMSE degradation + static sanity check",
        variables={
            "independent": ["preset"],
            "dependent": ["delta_oracle", "nmse_reuse_db", "static_delta"],
            "controlled": [f"max_snapshots={args.max_snapshots}"],
        },
        hypothesis="Overall median δ_oracle ≤ 0.5 and static δ ≈ 0",
        eval_criteria="Kill if median δ > 0.5 or static δ > 0.01",
    ) as run:
        all_results = []
        for preset in presets:
            print(f"\n{'─'*60}")
            print(f"  Preset: {preset}")
            print(f"{'─'*60}")
            data = load_data(preset, max_snapshots=args.max_snapshots)
            result = run_all_bs(data, preset)
            save_results(result, str(results_dir))
            all_results.append(result)
            data.clear_cache()
            run.log(**{f"{preset}_done": True})

        print(f"\n{'═'*60}")
        print("  GO / NO-GO CHECK")
        print(f"{'═'*60}")
        go = check_go_nogo(all_results)
        run.set_result(go=go)

        combined = {r["preset"]: r["summary"] for r in all_results if r}
        with open(results_dir / "persistence_summary.json", "w") as f:
            json.dump(combined, f, indent=2)

        print(f"\n  Final verdict: {'GO ✓' if go else 'NO-GO ✗'}")


if __name__ == "__main__":
    main()
