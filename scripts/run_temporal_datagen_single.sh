#!/bin/bash
# Generate temporal channel data for all 3 presets using SINGLE WORKER per preset.
# Single worker avoids shard boundary spike artifact.
# Each preset: 200 snapshots, 25 UEs (5 speeds × 5 each), dt from config numerology_mu, seed=42
# Shared trajectory: assets/data/shared_trajectories/
#
# Step 1: Generate trajectories (needs GPU for radio map)
# Step 2: Generate CIR + CFR channels (main workload)
#
# Usage:
#   bash scripts/run_temporal_datagen_single.sh          # all 3
#   bash scripts/run_temporal_datagen_single.sh 28g      # only 28g
#   bash scripts/run_temporal_datagen_single.sh traj     # only trajectories
#
# Estimated time: ~5-10min per preset on single GPU (200 snapshots)

set -e
TRAJ_DIR="assets/data/shared_trajectories"
TRAJ="$TRAJ_DIR/trajectories.npz"
N_SNAP=400
N_UE=25
VELOCITIES="0,1,2,5,8.3"

generate_trajectories() {
    echo ""
    echo "════════════════════════════════════════════"
    echo "  Generating trajectories: ${N_UE} UEs, ${N_SNAP} snapshots"
    echo "  Speeds: ${VELOCITIES} m/s (5 UEs per speed)"
    echo "════════════════════════════════════════════"

    CUDA_VISIBLE_DEVICES=0 uv run python -m src.dataset_operation.generate_trajectories \
        --preset munich_elaa_m_1k_28g --num_snapshots $N_SNAP --num_ue $N_UE \
        --velocities "$VELOCITIES" --seed 42 \
        --data_dir "$TRAJ_DIR" --mobility gauss_markov --alpha 0.95 \
        --target_bs 1

    echo "  Trajectories saved to $TRAJ"
}

run_preset() {
    local PRESET=$1
    local GPU=$2
    local DATA_DIR="assets/data/channels_${PRESET}_temporal"

    echo ""
    echo "════════════════════════════════════════════"
    echo "  $PRESET → $DATA_DIR (GPU $GPU)"
    echo "  ${N_SNAP} snapshots × ${N_UE} UEs (CIR + CFR)"
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

    # Verify: zero snapshots + dataset check
    uv run python -c "
import h5py, numpy as np
f = h5py.File('$DATA_DIR/channels.h5', 'r')
print(f'  Datasets: {list(f.keys())}')
for k in f.keys():
    print(f'    {k}: shape={f[k].shape}, dtype={f[k].dtype}')
n = f['cir_a'].shape[0]
zeros = sum(1 for t in range(n) if np.all(f['cir_a'][t] == 0))
has_cfr = 'cfr' in f
cfr_ok = '+ CFR' if has_cfr else 'NO CFR'
print(f'  Verified: {n} snapshots, {zeros} all-zero, {cfr_ok}')
if zeros > 0:
    print(f'  WARNING: {zeros} all-zero snapshots!')
else:
    print(f'  Clean data')
sz_gb = sum(f[k].id.get_storage_size() for k in f.keys()) / 1024**3
print(f'  Data size: {sz_gb:.1f} GB')
f.close()
"

    echo "  Done: $PRESET"
}

# Parse argument
TARGET=${1:-all}

case $TARGET in
    traj|trajectories)
        generate_trajectories
        ;;
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
        # Step 1: Generate shared trajectories
        generate_trajectories

        # Step 2: Generate channels — all 3 presets in parallel on different GPUs
        run_preset "elaa_m_1k_15g" 0 &
        PID_15G=$!
        run_preset "elaa_m_1k_28g" 1 &
        PID_28G=$!
        run_preset "5g_mimo_3g5" 2 &
        PID_5G=$!

        echo ""
        echo "Running 3 presets in parallel:"
        echo "  15g (GPU 0, PID=$PID_15G)"
        echo "  28g (GPU 1, PID=$PID_28G)"
        echo "  5g  (GPU 2, PID=$PID_5G)"

        wait $PID_15G
        echo "15g done"
        wait $PID_28G
        echo "28g done"
        wait $PID_5G
        echo "5g done"
        ;;
    *)
        echo "Usage: $0 [traj|28g|5g|15g|all]"
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
