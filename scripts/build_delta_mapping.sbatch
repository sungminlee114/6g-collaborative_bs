#!/bin/bash
#SBATCH --job-name=delta_map
#SBATCH --partition=a100
#SBATCH --qos=big_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --output=assets/runs/delta_map_%j.log

# Builds per-slot (δ_ls, δ_oracle) mapping from train UEs.
# Streams 1 UE at a time from h5 (no bulk RAM).
# Output: assets/results/paper_cache/train_delta_mapping_{preset}.npz

set -e
cd /home/sungmin/Projects/6g-collaborative_bs

PRESET_SHORT="${1:-28g}"
case "$PRESET_SHORT" in
    28g) PRESET="munich_elaa_m_1k_28g" ;;
    15g) PRESET="munich_elaa_m_1k_15g" ;;
    5g)  PRESET="munich_5g_mimo_3g5" ;;
    *)   echo "Unknown: $PRESET_SHORT"; exit 1 ;;
esac

echo "════════════════════════════════════════════"
echo "  Delta mapping: $PRESET (train UEs, streaming)"
echo "  Job: $SLURM_JOB_ID"
echo "════════════════════════════════════════════"

uv run python3 -c "
import numpy as np, torch, h5py
from pathlib import Path
from tqdm import tqdm
from src.ce_skip import ExperimentConfig
from src.experiments.pipeline import _precompute_deltas

preset = '${PRESET}'
cfg = ExperimentConfig.from_preset(preset)
split = cfg.ue_split(train=3, val=1, test=6)
train_ids = split['train']
SNR_DB = 20.0

h5 = h5py.File(cfg.temporal_dir / 'channels.h5', 'r')
cfr = h5['cfr']  # (T, N_UE, n_rx, n_tx, n_sc) complex64
print(f'CFR shape: {cfr.shape}')

all_d_ls = []
all_d_oracle = []
all_uid = []

for uid in tqdm(train_ids, desc='Train UEs'):
    # Read CFR directly
    cfr_ue = cfr[:, uid]  # (T, n_rx, n_tx, n_sc)
    h_real = torch.from_numpy(
        np.stack([cfr_ue.real, cfr_ue.imag], axis=1).astype(np.float32))
    h_real = h_real.reshape(h_real.shape[0], 2, -1, h_real.shape[-1])

    h_ls, d_ls, d_oracle, _ = _precompute_deltas(h_real, SNR_DB)

    all_d_ls.append(d_ls)
    all_d_oracle.append(d_oracle)
    all_uid.append(np.full(len(d_ls), uid))

    del h_real, h_ls, cfr_ue

h5.close()

d_ls_all = np.concatenate(all_d_ls)
d_oracle_all = np.concatenate(all_d_oracle)
uid_all = np.concatenate(all_uid)

out_dir = Path('assets/results/paper_cache')
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / 'train_delta_mapping_${PRESET_SHORT}.npz'
np.savez_compressed(out_path, d_ls=d_ls_all, d_oracle=d_oracle_all, uid=uid_all)
print(f'Saved: {out_path} ({len(d_ls_all)} slots from {len(train_ids)} train UEs)')
"

echo "════════════════════════════════════════════"
echo "  Done: $PRESET"
echo "════════════════════════════════════════════"
