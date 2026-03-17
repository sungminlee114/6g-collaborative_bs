#!/bin/bash
# Generate temporal (trajectory-based) channel datasets.
# Step 1: Generate trajectories (radio map → initial positions → mobility, single GPU)
# Step 2: Generate channels in parallel (multi-GPU, reads pre-computed positions)
#
# Usage: bash scripts/run_temporal_datagen.sh

set -e

NUM_SNAPSHOTS=1000
NUM_UE=100
DT_MS=10
VELOCITIES="0,1,8.3"
SEED=42

GPUS="0 1 2 3 4 5 7"  # 7 GPUs (GPU6 = occupied)
TRAJ_GPU=0              # Single GPU for trajectory generation (radio map)

PRESETS=(
    munich_elaa_s_1k_15g
    munich_elaa_s_1k_28g
    # Add more as needed
)

# ── Helper ─────────────────────────────────────────────────────────
run_temporal() {
    local preset=$1
    local data_dir="assets/data/channels_${preset#munich_}_temporal"
    local traj_file="$data_dir/trajectories.npz"

    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  Temporal: $preset"
    echo "  ${NUM_SNAPSHOTS} snapshots × dt=${DT_MS}ms = $((NUM_SNAPSHOTS * DT_MS))ms"
    echo "  Velocities: $VELOCITIES m/s"
    echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "════════════════════════════════════════════════════════════"

    # Step 1: Generate trajectories (single GPU, ~2 min)
    if [ ! -f "$traj_file" ]; then
        echo "  [1/2] Generating trajectories..."
        CUDA_VISIBLE_DEVICES=$TRAJ_GPU uv run python -m src.dataset_operation.generate_trajectories \
            --preset "$preset" \
            --num_snapshots "$NUM_SNAPSHOTS" \
            --num_ue "$NUM_UE" \
            --dt_ms "$DT_MS" \
            --velocities "$VELOCITIES" \
            --seed "$SEED" \
            --data_dir "$data_dir"
    else
        echo "  [1/2] Trajectories already exist: $traj_file"
    fi

    # Step 2: Generate channels (multi-GPU parallel)
    echo "  [2/2] Generating channels..."
    uv run python -m src.dataset_operation.generate_parallel \
        --preset "$preset" \
        --num_snapshots "$NUM_SNAPSHOTS" \
        --num_ue "$NUM_UE" \
        --gpus $GPUS \
        --trajectories "$traj_file" \
        --data_dir "$data_dir"

    echo "  Finished: $(date '+%Y-%m-%d %H:%M:%S')"
}

# ── Run ────────────────────────────────────────────────────────────
for preset in "${PRESETS[@]}"; do
    run_temporal "$preset"
done

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  All temporal datasets generated!"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "════════════════════════════════════════════════════════════"
