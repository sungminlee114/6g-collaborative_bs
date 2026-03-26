#!/bin/bash
#SBATCH --job-name=lut_calib
#SBATCH --partition=a100
#SBATCH --qos=big_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=assets/runs/lut_calib_%j.log

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
echo "  Delta→c LUT Calibration: $PRESET (streaming, 1 UE at a time)"
echo "  Job: $SLURM_JOB_ID"
echo "════════════════════════════════════════════"

uv run python3 -c "
import numpy as np, torch, json
from pathlib import Path
from tqdm import tqdm
from src.ce_skip import ExperimentConfig
from src.ce_skip.temporal_dataset import TemporalChannelData
from src.experiments.pipeline import (
    _precompute_deltas, _batch_nmse_multi_tau, _empirical_optimal_c
)

preset = '${PRESET}'
cfg = ExperimentConfig.from_preset(preset)
split = cfg.ue_split(train=3, val=1, test=6)
train_ids = split['train']
SNR_DB = 20.0
NOISE_FLOOR = np.sqrt(2.0 / (10 ** (SNR_DB / 10)))
TAU_ORACLE = np.linspace(0.001, 0.3, 25).tolist()

# Load DL-CE model once (small, ~6MB on GPU)
from src.models.estimator import create_model
from src.training.trainer import load_checkpoint
from src.experiments.S1_ce_profiling.train_dl_ce import ckpt_path_for
ckpt = ckpt_path_for(cfg, target_bs=1)
dl_ce = create_model(site_integration='none', encoder_channels=64, encoder_blocks=3, task_head_blocks=2)
load_checkpoint(dl_ce, f'ce_skip/{ckpt.stem}', device='cpu')
dl_ce = dl_ce.eval().half().to('cuda')

# Open temporal data (no bulk load)
data = TemporalChannelData(
    cfg.temporal_dir, trajectory_dir=cfg.trajectory_dir,
    max_snapshots=800, preset=preset,
)

delta_c_pairs = []
for uid in tqdm(train_ids, desc='Train UE (streaming)'):
    bs_id = int(data.ue_bs_ids[uid]) if data.ue_bs_ids is not None else 0

    # Load 1 UE
    h_complex = data.get_ue_series(uid, bs_id, snap_range=(0, 800))
    h_real = torch.from_numpy(
        np.stack([h_complex.real, h_complex.imag], axis=1).astype(np.float32))
    del h_complex

    # Compute delta
    h_ls, d_ls, d_oracle, _ = _precompute_deltas(h_real, SNR_DB)

    # DL-CE inference (fp16, batch=16)
    T = h_real.shape[0]
    h_ce_list = []
    for t0 in range(0, T, 16):
        batch = h_ls[t0:t0+16]
        rms = batch.flatten(1).pow(2).mean(1, keepdim=True).sqrt().clamp(min=1e-10).unsqueeze(-1).unsqueeze(-1)
        with torch.no_grad(), torch.amp.autocast('cuda'):
            out = dl_ce((batch / rms).half().to('cuda')).float().cpu()
        h_ce_list.append(out * rms)
    h_ce = torch.cat(h_ce_list, dim=0)

    # Tau sweep
    sweep = _batch_nmse_multi_tau(
        h_real, h_ce, d_oracle, TAU_ORACLE,
        alpha_mode='binary', device=torch.device('cuda'))

    c_opt = _empirical_optimal_c(sweep, NOISE_FLOOR, nmse_margin_db=1.0)
    med_delta = float(np.median(d_oracle))
    speed = float(data.get_ue_speed(uid))
    delta_c_pairs.append({
        'uid': int(uid), 'speed': speed,
        'med_delta': med_delta, 'c_opt': c_opt,
    })
    print(f'  UE{uid} ({speed:.1f}m/s): med_delta={med_delta:.4f}, c_opt={c_opt:.2f}')

    # Free this UE immediately
    del h_real, h_ls, h_ce, h_ce_list, sweep, d_ls, d_oracle
    torch.cuda.empty_cache()

del dl_ce
data.clear_cache()
torch.cuda.empty_cache()

# Save
out_path = Path('assets/results/paper_cache')
out_path.mkdir(parents=True, exist_ok=True)
lut_file = out_path / 'train_lut_delta_${PRESET_SHORT}.json'
with open(lut_file, 'w') as f:
    json.dump(delta_c_pairs, f, indent=2)
print(f'\\nSaved: {lut_file} ({len(delta_c_pairs)} train UEs)')
"

echo "════════════════════════════════════════════"
echo "  Done: $PRESET"
echo "════════════════════════════════════════════"
