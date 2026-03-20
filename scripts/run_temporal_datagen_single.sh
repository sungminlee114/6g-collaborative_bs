#!/bin/bash
# Generate temporal channel data for all 3 presets using SINGLE WORKER per preset.
# Single worker avoids shard boundary spike artifact.
# Each preset: 4000 snapshots, 9 UEs, dt=0.25ms, seed=42
# Shared trajectory: assets/data/shared_trajectories/
#
# Usage:
#   bash scripts/run_temporal_datagen_single.sh          # all 3
#   bash scripts/run_temporal_datagen_single.sh 28g      # only 28g
#   bash scripts/run_temporal_datagen_single.sh 5g       # only 5g
#
# Estimated time: ~13min per preset on single GPU

set -e
TRAJ="assets/data/shared_trajectories/trajectories.npz"
N_SNAP=4000
N_UE=9

run_preset() {
    local PRESET=$1
    local GPU=$2
    local DATA_DIR="assets/data/channels_${PRESET}_temporal"

    echo ""
    echo "════════════════════════════════════════════"
    echo "  $PRESET → $DATA_DIR (GPU $GPU)"
    echo "════════════════════════════════════════════"

    # Clean old data
    rm -f "$DATA_DIR/channels.h5"
    rm -f "$DATA_DIR"/_shard_*.h5
    mkdir -p "$DATA_DIR"

    # Generate with single worker (no shard boundary spikes)
    CUDA_VISIBLE_DEVICES=$GPU uv run python -m src.dataset_operation.generate_worker \
        --preset "munich_$PRESET" \
        --snapshot_start 0 --snapshot_end $N_SNAP --num_ue $N_UE \
        --data_dir "$DATA_DIR" \
        --trajectories "$TRAJ"

    # Rename shard to channels.h5
    SHARD="$DATA_DIR/_shard_0_${N_SNAP}.h5"
    if [ -f "$SHARD" ]; then
        mv "$SHARD" "$DATA_DIR/channels.h5"
        # Update h5 attrs
        uv run python -c "
import h5py
f = h5py.File('$DATA_DIR/channels.h5', 'r+')
f.attrs['n_snapshots'] = $N_SNAP
f.attrs['n_ue'] = $N_UE
if 'snapshot_start' in f.attrs: del f.attrs['snapshot_start']
if 'snapshot_end' in f.attrs: del f.attrs['snapshot_end']
f.close()
print('  Updated h5 attrs')
"
    fi

    # Verify: zero snapshots + spike check
    uv run python -c "
import h5py, numpy as np
f = h5py.File('$DATA_DIR/channels.h5', 'r')
n = f['cir_a'].shape[0]
zeros = sum(1 for t in range(n) if np.all(f['cir_a'][t] == 0))
print(f'  Verified: {n} snapshots, {zeros} all-zero')
if zeros > 0:
    print(f'  ✗ WARNING: {zeros} all-zero snapshots!')
else:
    print(f'  ✓ Clean data')
f.close()
"

    echo "  Done: $PRESET"
}

# Parse argument
TARGET=${1:-all}

case $TARGET in
    28g|elaa_m_1k_28g)
        run_preset "elaa_m_1k_28g" 0
        ;;
    5g|5g_mimo_3g5)
        run_preset "5g_mimo_3g5" 0
        ;;
    15g|elaa_m_1k_15g)
        run_preset "elaa_m_1k_15g" 0
        ;;
    all)
        # Run sequentially on GPU 0 (single worker = no parallelism within preset)
        # But different presets can run on different GPUs in parallel
        run_preset "elaa_m_1k_28g" 0 &
        PID_28G=$!
        run_preset "5g_mimo_3g5" 1 &
        PID_5G=$!

        echo "Running 28g (GPU 0, PID=$PID_28G) and 5g (GPU 1, PID=$PID_5G) in parallel..."
        echo "15g already done (55GB h5 exists)"

        wait $PID_28G
        echo "28g done"
        wait $PID_5G
        echo "5g done"
        ;;
    *)
        echo "Usage: $0 [28g|5g|15g|all]"
        exit 1
        ;;
esac

echo ""
echo "════════════════════════════════════════════"
echo "  All done. Verify with:"
echo "  for d in assets/data/channels_*_temporal; do"
echo "    ls -lh \$d/channels.h5 2>/dev/null"
echo "  done"
echo "════════════════════════════════════════════"
