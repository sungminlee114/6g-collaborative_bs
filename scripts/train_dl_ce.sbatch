#!/bin/bash
#SBATCH --job-name=dl_ce_train
#SBATCH --partition=a100
#SBATCH --qos=big_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=assets/runs/dl_ce_train_%j.log

# Usage:
#   sbatch scripts/train_dl_ce.sbatch           # all presets
#   sbatch scripts/train_dl_ce.sbatch 28g       # single preset

set -e
cd /home/sungmin/Projects/6g-collaborative_bs

PRESET_SHORT="${1:-all}"

if [ "$PRESET_SHORT" = "all" ]; then
    uv run python -m src.experiments.S1_ce_profiling.train_dl_ce --all --epochs 100
else
    case "$PRESET_SHORT" in
        28g) PRESET="munich_elaa_m_1k_28g" ;;
        15g) PRESET="munich_elaa_m_1k_15g" ;;
        5g)  PRESET="munich_5g_mimo_3g5" ;;
        *)   echo "Unknown: $PRESET_SHORT"; exit 1 ;;
    esac
    uv run python -m src.experiments.S1_ce_profiling.train_dl_ce --preset "$PRESET" --epochs 100
fi
