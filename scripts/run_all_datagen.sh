#!/bin/bash
# Generate all channel datasets — independent + temporal modes.
# 7 configs: 3 MIMO baselines + 4 ELAA (FR3/FR2 only).
# Runs ALL configs in parallel (1 GPU per config).
#
# Usage: bash scripts/run_all_datagen.sh [phase]
#   phase: "indep" | "temporal" | "all" (default: all)

set -e

NUM_UE=100
TRAJ_GPU=0  # CPU-bound, any GPU for trajectory gen

# ── Mode config ────────────────────────────────────────────────────
INDEP_SNAPSHOTS=100
TEMPORAL_SNAPSHOTS=1000
DT_MS=10
VELOCITIES="0,1,8.3"
SEED=42

# ── 7 configs × GPU assignments ──────────────────────────────────
# Each config gets 1 dedicated GPU for max parallelism.
# Config name        GPU   Array     Freq     BW    SC
declare -A CONFIG_GPU=(
    [munich_5g_mimo_3g5]=0      # 8×8    3.5GHz  100M  256
    [munich_mimo_15g]=1         # 8×8    15GHz   400M  1024
    [munich_mimo_28g]=2         # 8×8    28GHz   100M  256
    [munich_elaa_s_1k_15g]=3    # 16×16  15GHz   400M  1024
    [munich_elaa_s_1k_28g]=4    # 16×16  28GHz   400M  1024
    [munich_elaa_m_1k_15g]=5    # 32×16  15GHz   400M  1024
    [munich_elaa_m_1k_28g]=6    # 32×16  28GHz   400M  1024
)

PRESETS=(${!CONFIG_GPU[@]})

# ── Helpers ────────────────────────────────────────────────────────
count_snapshots() {
    local dir=$1
    if [ ! -d "$dir" ]; then echo 0; return; fi
    find "$dir" -name "channels.npz" 2>/dev/null | wc -l
}

data_dir_for() {
    local preset=$1
    echo "assets/data/channels_${preset#munich_}"
}

# ── Phase: Independent ─────────────────────────────────────────────
run_independent_all() {
    echo ""
    echo "═══ Phase 1: Independent ($INDEP_SNAPSHOTS snapshots) ═══"
    echo "  Launching ${#PRESETS[@]} configs in parallel..."
    echo ""

    local pids=()
    local names=()

    for preset in "${PRESETS[@]}"; do
        local gpu=${CONFIG_GPU[$preset]}
        local dir=$(data_dir_for "$preset")
        local done=$(count_snapshots "$dir")

        if [ "$done" -ge "$INDEP_SNAPSHOTS" ]; then
            echo "  GPU$gpu ⏭ $preset ($done/$INDEP_SNAPSHOTS exist)"
            continue
        fi

        local log="/tmp/datagen_indep_${preset#munich_}.log"
        echo "  GPU$gpu ▶ $preset ($done→$INDEP_SNAPSHOTS) → $log"

        CUDA_VISIBLE_DEVICES=$gpu uv run python -m src.dataset_operation.generate_parallel \
            --preset "$preset" \
            --num_snapshots "$INDEP_SNAPSHOTS" \
            --num_ue "$NUM_UE" \
            --gpus "$gpu" \
            --start_snapshot "$done" \
            > "$log" 2>&1 &
        pids+=($!)
        names+=("$preset")
    done

    if [ ${#pids[@]} -eq 0 ]; then
        echo "  All independent datasets complete!"
        return 0
    fi

    echo ""
    echo "  Monitor: tail -f /tmp/datagen_indep_*.log"
    echo "  Progress: curl localhost:8765/api/datagen"
    echo ""

    # Wait for all
    local failed=0
    for i in "${!pids[@]}"; do
        if wait ${pids[$i]}; then
            echo "  ✓ ${names[$i]} done"
        else
            echo "  ✗ ${names[$i]} FAILED (rc=$?)"
            ((failed++))
        fi
    done

    return $failed
}

# ── Phase: Temporal ────────────────────────────────────────────────
run_temporal_all() {
    echo ""
    echo "═══ Phase 2: Temporal ($TEMPORAL_SNAPSHOTS snapshots, dt=${DT_MS}ms) ═══"
    echo ""

    # Step 1: Generate trajectories (CPU-bound, sequential is fine)
    echo "── Trajectory generation ──"
    for preset in "${PRESETS[@]}"; do
        local dir="$(data_dir_for "$preset")_temporal"
        local traj="$dir/trajectories.npz"

        if [ -f "$traj" ]; then
            echo "  ⏭ $preset trajectories exist"
            continue
        fi

        echo "  ▶ $preset trajectories..."
        CUDA_VISIBLE_DEVICES=$TRAJ_GPU uv run python -m src.dataset_operation.generate_trajectories \
            --preset "$preset" \
            --num_snapshots "$TEMPORAL_SNAPSHOTS" \
            --num_ue "$NUM_UE" \
            --dt_ms "$DT_MS" \
            --velocities "$VELOCITIES" \
            --seed "$SEED" \
            --data_dir "$dir"
        echo "  ✓ $preset trajectories done"
    done

    # Step 2: Generate channels in parallel
    echo ""
    echo "── Channel generation (parallel) ──"

    local pids=()
    local names=()

    for preset in "${PRESETS[@]}"; do
        local gpu=${CONFIG_GPU[$preset]}
        local dir="$(data_dir_for "$preset")_temporal"
        local traj="$dir/trajectories.npz"
        local done=$(count_snapshots "$dir")

        if [ "$done" -ge "$TEMPORAL_SNAPSHOTS" ]; then
            echo "  GPU$gpu ⏭ $preset ($done/$TEMPORAL_SNAPSHOTS exist)"
            continue
        fi

        local log="/tmp/datagen_temporal_${preset#munich_}.log"
        echo "  GPU$gpu ▶ $preset ($done→$TEMPORAL_SNAPSHOTS) → $log"

        CUDA_VISIBLE_DEVICES=$gpu uv run python -m src.dataset_operation.generate_parallel \
            --preset "$preset" \
            --num_snapshots "$TEMPORAL_SNAPSHOTS" \
            --num_ue "$NUM_UE" \
            --gpus "$gpu" \
            --trajectories "$traj" \
            --data_dir "$dir" \
            --start_snapshot "$done" \
            > "$log" 2>&1 &
        pids+=($!)
        names+=("$preset")
    done

    if [ ${#pids[@]} -eq 0 ]; then
        echo "  All temporal datasets complete!"
        return 0
    fi

    echo ""
    echo "  Monitor: tail -f /tmp/datagen_temporal_*.log"
    echo "  Progress: curl localhost:8765/api/datagen"
    echo ""

    local failed=0
    for i in "${!pids[@]}"; do
        if wait ${pids[$i]}; then
            echo "  ✓ ${names[$i]} done"
        else
            echo "  ✗ ${names[$i]} FAILED (rc=$?)"
            ((failed++))
        fi
    done

    return $failed
}

# ── Main ───────────────────────────────────────────────────────────
PHASE="${1:-all}"

echo "════════════════════════════════════════════════════════════"
echo "  Channel Dataset Generation (parallel)"
echo "  ${#PRESETS[@]} configs, 1 GPU each"
echo "  Phase: $PHASE"
echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "════════════════════════════════════════════════════════════"

case "$PHASE" in
    indep|independent)
        run_independent_all
        ;;
    temporal)
        run_temporal_all
        ;;
    all)
        run_independent_all
        echo ""
        run_temporal_all
        ;;
    *)
        echo "Unknown phase: $PHASE (use: indep|temporal|all)"
        exit 1
        ;;
esac

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Done! $(date '+%Y-%m-%d %H:%M:%S')"
echo "════════════════════════════════════════════════════════════"
