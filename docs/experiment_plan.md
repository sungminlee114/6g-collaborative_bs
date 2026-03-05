# Experiment Verification Plan

## Dataset
- Sionna RT, Munich scene, 15 GHz mmWave, 400 MHz BW, 1024 subcarriers
- 8 BS (fixed positions), 100 UE/snapshot, multiple snapshots (≥100)
- 2×2 TX (4 ant), 1×1 RX cross-pol (2 ant) → 8 antenna pairs
- Pre-train BSs: [0..5], Test BSs: [6, 7]
- Storage: .npz per snapshot + metadata.parquet

## Verification Steps (ALL must be experiments)

### Step 0: Pre-trained E vs From-Scratch
- Pre-train E+θ_task on BS₁~₆ → deploy to BS₇ with new θ_BS
- vs. training from scratch on BS₇
- **Claim**: Pre-trained E converges faster with less data

### Step 1: 3-way vs 2-way vs FedAvg
- 3-way (ours): E shared + θ_task shared + θ_BS local
- 2-way (FedPer): E shared + θ_task local
- FedAvg: everything shared
- Independent: per-BS training, no sharing
- **Claim**: 3-way decomposition achieves best per-site performance

### Step 1b: Few-Shot Adaptation
- Given pre-trained E+θ_task, adapt θ_BS with k={5,10,20,50,100} samples
- Compare vs. fine-tuning all params vs. from-scratch
- **Claim**: θ_BS-only adaptation is most sample-efficient

### Step 2: Task-Agnostic θ_BS (KILLER EXPERIMENT)
- Train on Task A (channel estimation), freeze θ_BS
- Apply to Task B (beam prediction) — only train new θ_task_B
- **Claim**: θ_BS captures site-specific info independent of task

### Step 2b: E as Downstream Backbone
- Freeze pre-trained E, use as feature extractor for new tasks
- **Claim**: E learns transferable wireless physics representations

### Ablations
- θ_BS dimension: {8, 16, 32, 64, 128, 256}
- Number of pre-training BSs: {2, 4, 6}
- Site injection type: FiLM vs concat vs add
- θ_task/θ_BS structural order variants

### Cold Start Analysis
- Plot NMSE vs. number of training samples
- Show crossover point where independent training catches up
- Time-based analysis: adaptation speed comparison

## Baselines
- LS (identity — just returns noisy input)
- LMMSE (diagonal Wiener filter approximation)
- Independent ReEsNet (per-BS, no sharing)
- FedAvg ReEsNet (all params shared)
- FedPer (2-way: shared encoder + local head)
- MAML (meta-learned initialization)
- From-scratch ReEsNet (on target BS only)

## Project Structure
```
src/
├── config.py
├── data/generate.py, dataset.py, utils.py
├── models/estimator.py (3-way), baselines.py
├── training/trainer.py, federated.py, meta_learning.py
└── experiments/ (ipynb per verification step)
```
