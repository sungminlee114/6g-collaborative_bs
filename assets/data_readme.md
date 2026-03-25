# Channel Dataset Summary

Sionna RT ray-tracing channel datasets for 6G O-RAN collaborative BS research.
Scene: Munich city center, 8 BS sites, 100 UE/snapshot.

## OFDM Numerology (3GPP TS 38.211)

Each config specifies `numerology_mu` which determines the 3GPP NR subcarrier spacing and slot timing.

| Parameter | Formula | 15/28 GHz (μ=3) | 3.5 GHz (μ=1) |
|-----------|---------|-----------------|----------------|
| SCS | 2^μ × 15 kHz | 120 kHz | 30 kHz |
| Slot duration | 1 ms / 2^μ | 0.125 ms | 0.5 ms |
| Symbols/slot | (fixed) | 14 | 14 |
| Slots/subframe | 2^μ | 8 | 2 |
| Slots/frame | 10 × 2^μ | 80 | 20 |
| Frame | (fixed) | 10 ms | 10 ms |

Two distinct spacings coexist in each config.

| Spacing | Value | Purpose |
|---------|-------|---------|
| **SCS** (`cfg.scs`) | 120 kHz (μ=3) | 3GPP system numerology. Determines slot duration. |
| **Pilot spacing** (`cfg.subcarrier_spacing`) | ~352 kHz | Channel frequency sampling = effective_bw / N_sc. Used by Sionna `subcarrier_frequencies()`. |

`num_subcarriers` (1024 or 256) is the number of frequency-domain pilot samples across the bandwidth, not the total OFDM subcarriers in the system (which would be BW/SCS ≈ 3333 for 400 MHz @ 120 kHz). This is standard practice in CE simulation. The channel matrix H ∈ ℂ^(N_sc × N_tx × N_rx) uses N_sc = `num_subcarriers`.

## 3×3 Factorial Design

3 antenna scales × 3 frequencies.

### Config Matrix

| Config | Array (UPA) | Nt | Freq | Band | μ | SCS | Slot | BW | N_sc | Power |
|--------|------------|----:|-----:|------|--:|----:|-----:|----:|-----:|------:|
| `5g_mimo_3g5` | 8×8 | 64 | 3.5 GHz | FR1 | 1 | 30 kHz | 0.5 ms | 100 MHz | 256 | 46 dBm |
| `mimo_15g` | 8×8 | 64 | 15 GHz | FR3 | 3 | 120 kHz | 0.125 ms | 400 MHz | 1024 | 40 dBm |
| `mimo_28g` | 8×8 | 64 | 28 GHz | FR2 | 3 | 120 kHz | 0.125 ms | 100 MHz | 256 | 40 dBm |
| `elaa_s_*_15g` | 16×16 | 256 | 15 GHz | FR3 | 3 | 120 kHz | 0.125 ms | 400 MHz | 1024 | 40 dBm |
| `elaa_s_*_28g` | 16×16 | 256 | 28 GHz | FR2 | 3 | 120 kHz | 0.125 ms | 400 MHz | 1024 | 40 dBm |
| `elaa_m_*_15g` | 32×16 | 512 | 15 GHz | FR3 | 3 | 120 kHz | 0.125 ms | 400 MHz | 1024 | 40 dBm |
| `elaa_m_*_28g` | 32×16 | 512 | 28 GHz | FR2 | 3 | 120 kHz | 0.125 ms | 400 MHz | 1024 | 40 dBm |

- **Nr = 2** (cross-polarized UE antenna, 3GPP default)
- **Synthetic array**: physical single-antenna BS, Sionna constructs virtual UPA in post-processing

### Comparison Axes

| Comparison | Configs | What varies | What's controlled |
|-----------|---------|------------|-------------------|
| Frequency effect (MIMO) | `5g_mimo_3g5` vs `mimo_15g` vs `mimo_28g` | freq, BW | Nt=64 |
| Frequency effect (ELAA-S) | `elaa_s_1k_15g` vs `elaa_s_1k_28g` | freq | Nt=256, BW=400M, SC=1024 |
| Frequency effect (ELAA-M) | `elaa_m_1k_15g` vs `elaa_m_1k_28g` | freq | Nt=512, BW=400M, SC=1024 |
| Antenna scaling @15 GHz | `mimo_15g` → `elaa_s_1k_15g` → `elaa_m_1k_15g` | Nt (64→256→512) | freq=15G, BW=400M |
| Antenna scaling @28 GHz | `mimo_28g` → `elaa_s_1k_28g` → `elaa_m_1k_28g` | Nt (64→256→512) | freq=28G |

## Data Modes

### Independent (100 snapshots)
- Random UE drops per snapshot (i.i.d.)
- For cross-sectional channel estimation experiments

### Temporal (configurable snapshots)
- **Gauss-Markov mobility model** (α=0.95) with building collision avoidance
- dt = slot duration from config (μ=3 → 0.125 ms, μ=1 → 0.5 ms)
- `generate_trajectories.py` reads `numerology_mu` from preset config automatically (no `--dt_ms`)
- UE speeds: 0, 1, 2, 5, 8.3 m/s
- Pre-computed trajectories via `generate_trajectories.py`
- Full Sionna RT re-trace per snapshot (PathSolver, not apply_doppler)
- For temporal channel prediction / CE-skip experiments

## Directory Structure

```
assets/data/
├── channels_{config}/                  # Independent mode
│   ├── snapshot_NNNN/
│   │   ├── channels.npz               # CFR: (num_ue, Nr, Nt, num_sc) complex64
│   │   ├── cir.npz                     # CIR: paths, delays, amplitudes
│   │   └── ue_positions.npy            # (num_ue, 3) float32
│   ├── bs_info.json                    # Config metadata
│   └── progress.json                   # Generation progress
├── channels_{config}_temporal/         # Temporal mode
│   ├── snapshot_NNNN/                  # Same structure as above
│   ├── trajectories.npz               # Pre-computed UE trajectories
│   │   ├── positions: (1000, 100, 3)  # Per-snapshot UE positions
│   │   ├── velocities: (1000, 100, 2) # Gauss-Markov time-varying velocities
│   │   ├── speeds: (100,)             # Initial speeds
│   │   ├── bs_ids: (100,)             # Serving BS per UE
│   │   └── device_types: (100,)       # UE type index
│   ├── trajectory_info.json            # Mobility parameters
│   ├── bs_info.json
│   └── progress.json
```

## Disk Usage

| Config | Independent | Temporal | Total |
|--------|----------:|----------:|------:|
| `5g_mimo_3g5` | 2.3 GB | 23 GB | 25 GB |
| `mimo_15g` | 9.2 GB | 93 GB | 102 GB |
| `mimo_28g` | 2.7 GB | 26 GB | 29 GB |
| `elaa_s_1k_15g` | 38 GB | 362 GB | 400 GB |
| `elaa_s_1k_28g` | 38 GB | 377 GB | 415 GB |
| `elaa_m_1k_15g` | 74 GB | 726 GB | 800 GB |
| `elaa_m_1k_28g` | 77 GB | ~700 GB | ~777 GB |
| **Total** | **~241 GB** | **~2.3 TB** | **~2.5 TB** |

## Generation

```bash
# Full generation (independent + temporal, all 7 configs parallel)
bash scripts/run_all_datagen.sh

# Independent only
bash scripts/run_all_datagen.sh indep

# Temporal only (requires trajectories)
bash scripts/run_all_datagen.sh temporal
```

Hardware: 8× A100 40GB, 128-core Xeon Gold 6530, 1TB RAM.
1 GPU per config, 7 configs in parallel.
