"""Plot alpha strategies from cached data. No computation needed.

Usage:
    python scripts/plot_alpha_strategies.py [--ylim MIN MAX]
"""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CACHE = "assets/results/paper_cache/alpha_strategies.json"

def main():
    # Parse optional ylim
    ylim = None
    if "--ylim" in sys.argv:
        idx = sys.argv.index("--ylim")
        ylim = (float(sys.argv[idx+1]), float(sys.argv[idx+2]))

    with open(CACHE) as f:
        data = json.load(f)

    c_values = data["c_values"]
    NF = data["noise_floor"]
    strat_names = data["strategies"]

    # 4 representative strategies only
    style = {
        "Full CE (every slot)":       ("black",      "-",  1.5),
        "Binary (skip/full)":         ("tab:gray",   "-",  1.5),
        "δ/τ (proposed)":             ("tab:green",  "-",  2.5),
        "Sigmoid(k=63) [fit]":        ("tab:blue",   "--", 1.5),
    }
    # Filter to only plot these 4
    strat_names = [s for s in strat_names if s in style]

    selected = sorted(data["ues"].keys(), key=int)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharey=False)

    for idx, uid in enumerate(selected):
        ax = axes[idx//3, idx%3]
        ue = data["ues"][uid]

        for sname in strat_names:
            color, ls, lw = style.get(sname, ("tab:gray", "-", 1.0))
            ax.plot(c_values, ue["nmse"][sname], ls, color=color, lw=lw, alpha=0.85, label=sname)

        ax.axvline(1.0, color="orange", ls=":", lw=1.5, alpha=0.6, label="c=1 (τ=NF)")
        ax.axhline(-20.0, color="black", ls=":", lw=0.8, alpha=0.3)
        ax.set_title(f"UE{uid} ({ue['cat']}, {ue['speed']:.1f}m/s)", fontsize=11)
        ax.set_xlabel("c = τ / NF")
        if idx == 0:
            ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.3)
        if idx % 3 == 0:
            ax.set_ylabel("NMSE (dB)")
        # Pseudo-log: map dB so that lower values get more space
        # Use function_scale: y_display = -10^(-dB/10) → log spacing for negative dB
        ax.set_yscale("function", functions=(
            lambda x: -np.power(10.0, -np.asarray(x, dtype=float)/10),  # dB → display
            lambda x: -10*np.log10(-np.asarray(x, dtype=float))          # display → dB
        ))
        if ylim:
            ax.set_ylim(*ylim)

    fig.suptitle("Scheduling Strategies Comparison (ELAA 15GHz, Oracle δ, 200 slots)", fontsize=12)
    plt.tight_layout()
    plt.savefig("assets/plots/paper_alpha_combined.png", dpi=200, bbox_inches="tight")
    plt.savefig("assets/plots/paper_alpha_combined.pdf", bbox_inches="tight")
    plt.close()
    print("Saved: assets/plots/paper_alpha_combined.png")

if __name__ == "__main__":
    main()
