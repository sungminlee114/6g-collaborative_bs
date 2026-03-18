#!/bin/bash
# Generate all channel datasets — independent + temporal modes.
# 7 configs: 3 MIMO baselines + 4 ELAA (FR3/FR2 only).
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

# ── 7-config matrix (see docs/ce-skip/dataset_design.md) ─────────
# Row 1: MIMO baselines (8×8, 3 freq bands)
# Row 2-3: ELAA (16×16, 32×16) at FR3/FR2 only
PRESETS=(
    munich_5g_mimo_3g5      # 8×8,  3.5 GHz, 100M BW, 256 SC
    munich_mimo_15g         # 8×8,  15 GHz,  400M BW, 1024 SC
    munich_mimo_28g         # 8×8,  28 GHz,  100M BW, 256 SC
    munich_elaa_s_1k_15g    # 16×16, 15 GHz, 400M BW, 1024 SC
    munich_elaa_s_1k_28g    # 16×16, 28 GHz, 400M BW, 1024 SC
    munich_elaa_m_1k_15g    # 32×16, 15 GHz, 400M BW, 1024 SC
    munich_elaa_m_1k_28g    # 32×16, 28 GHz, 400M BW, 1024 SC
)

# ── Skip check ─────────────────────────────────────────────────────
count_snapshots() {
    local dir=$1
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
    local done=$(count_snapshots "$data_dir")

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
    local done=$(count_snapshots "$data_dir")

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
echo "  ${#PRESETS[@]} presets × 2 modes (independent + temporal)"
echo "  GPUs: $GPUS"
echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "════════════════════════════════════════════════════════════"

for preset in "${PRESETS[@]}"; do
    # Derive data dirs from preset name (strip munich_ prefix)
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
