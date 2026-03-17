#!/bin/bash
# Generate all channel datasets — independent + temporal modes.
# Skips presets that already have all snapshots generated.
# Task tracking is automatic (backlog.json).
#
# Usage: bash scripts/run_all_datagen.sh

set -e

NUM_UE=100
GPUS="0 1 2 3 4 5 7"  # 7 GPUs (GPU6 = occupied)
TRAJ_GPU=0

# ── Independent mode config ────────────────────────────────────────
INDEP_SNAPSHOTS=100

# ── Temporal mode config ───────────────────────────────────────────
TEMPORAL_SNAPSHOTS=1000
DT_MS=10
VELOCITIES="0,1,8.3"
SEED=42

# ── All ELAA presets ───────────────────────────────────────────────
ELAA_PRESETS=(
    munich_elaa_s_1k_15g
    munich_elaa_s_1k_28g
    munich_elaa_s_2k_15g
    munich_elaa_s_2k_28g
    munich_elaa_m_1k_15g
    munich_elaa_m_1k_28g
    munich_elaa_m_2k_15g
    munich_elaa_m_2k_28g
    munich_elaa_l_1k_15g
    munich_elaa_l_1k_28g
    munich_elaa_l_2k_15g
    munich_elaa_l_2k_28g
    munich_elaa_s_4k_15g
    munich_elaa_s_4k_28g
    munich_elaa_m_4k_15g
    munich_elaa_m_4k_28g
    munich_elaa_l_4k_15g
    munich_elaa_l_4k_28g
)

FIVEG_PRESETS=(
    munich_5g_mimo_3g5
)

ALL_PRESETS=("${FIVEG_PRESETS[@]}" "${ELAA_PRESETS[@]}")

# ── Skip check ─────────────────────────────────────────────────────
count_snapshots() {
    local dir=$1
    local expected=$2
    if [ ! -d "$dir" ]; then
        echo 0
        return
    fi
    local count=$(find "$dir" -name "channels.npz" 2>/dev/null | wc -l)
    echo "$count"
}

# ── Helpers ────────────────────────────────────────────────────────
run_independent() {
    local preset=$1
    local data_dir=$2
    local done=$(count_snapshots "$data_dir" "$INDEP_SNAPSHOTS")

    if [ "$done" -ge "$INDEP_SNAPSHOTS" ]; then
        echo "  ⏭ Independent SKIP ($done/$INDEP_SNAPSHOTS snapshots exist)"
        return
    fi

    echo "  ▶ Independent: $INDEP_SNAPSHOTS snapshots ($done exist, generating rest)"
    uv run python -m src.dataset_operation.generate_parallel \
        --preset "$preset" \
        --num_snapshots "$INDEP_SNAPSHOTS" \
        --num_ue "$NUM_UE" \
        --gpus $GPUS
}

run_temporal() {
    local preset=$1
    local data_dir=$2
    local traj_file="$data_dir/trajectories.npz"
    local done=$(count_snapshots "$data_dir" "$TEMPORAL_SNAPSHOTS")

    if [ "$done" -ge "$TEMPORAL_SNAPSHOTS" ]; then
        echo "  ⏭ Temporal SKIP ($done/$TEMPORAL_SNAPSHOTS snapshots exist)"
        return
    fi

    # Step 1: Trajectories
    if [ ! -f "$traj_file" ]; then
        echo "  ▶ Temporal [1/2]: Generating trajectories..."
        CUDA_VISIBLE_DEVICES=$TRAJ_GPU uv run python -m src.dataset_operation.generate_trajectories \
            --preset "$preset" \
            --num_snapshots "$TEMPORAL_SNAPSHOTS" \
            --num_ue "$NUM_UE" \
            --dt_ms "$DT_MS" \
            --velocities "$VELOCITIES" \
            --seed "$SEED" \
            --data_dir "$data_dir"
    else
        echo "  ▶ Temporal [1/2]: Trajectories exist"
    fi

    # Step 2: Channels
    echo "  ▶ Temporal [2/2]: Generating channels ($done exist)..."
    uv run python -m src.dataset_operation.generate_parallel \
        --preset "$preset" \
        --num_snapshots "$TEMPORAL_SNAPSHOTS" \
        --num_ue "$NUM_UE" \
        --gpus $GPUS \
        --trajectories "$traj_file" \
        --data_dir "$data_dir"
}

# ── Main loop ──────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════════"
echo "  Channel Dataset Generation"
echo "  ${#ALL_PRESETS[@]} presets × 2 modes (independent + temporal)"
echo "  GPUs: $GPUS"
echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "════════════════════════════════════════════════════════════"

for preset in "${ALL_PRESETS[@]}"; do
    # Derive data dirs
    indep_dir="assets/data/channels_${preset#munich_}"
    temporal_dir="${indep_dir}_temporal"

    echo ""
    echo "──── $preset ────"
    echo "  $(date '+%H:%M:%S')"

    # Independent
    run_independent "$preset" "$indep_dir"

    # Temporal
    run_temporal "$preset" "$temporal_dir"
done

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  All datasets generated!"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "════════════════════════════════════════════════════════════"
