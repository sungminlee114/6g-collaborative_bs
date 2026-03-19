# Channel Dataset Summary

Sionna RT ray-tracing channel datasets for 6G O-RAN collaborative BS research.
Scene: Munich city center, 8 BS sites, 100 UE/snapshot.

## 3×3 Factorial Design

3 antenna scales × 3 frequencies, consistent SC spacing (~390.6 kHz).

### Config Matrix

| Config | Array (UPA) | Nt | Freq | BW | SC | SC spacing | Synthetic | Power |
|--------|------------|----:|-----:|----:|-----:|-----------:|:---------:|------:|
| `5g_mimo_3g5` | 8×8 | 64 | 3.5 GHz | 100 MHz | 256 | 390.6 kHz | No | 46 dBm |
| `mimo_15g` | 8×8 | 64 | 15 GHz | 400 MHz | 1024 | 390.6 kHz | Yes | 40 dBm |
| `mimo_28g` | 8×8 | 64 | 28 GHz | 100 MHz | 256 | 390.6 kHz | Yes | 40 dBm |
| `elaa_s_1k_15g` | 16×16 | 256 | 15 GHz | 400 MHz | 1024 | 390.6 kHz | Yes | 40 dBm |
| `elaa_s_1k_28g` | 16×16 | 256 | 28 GHz | 400 MHz | 1024 | 390.6 kHz | Yes | 40 dBm |
| `elaa_m_1k_15g` | 32×16 | 512 | 15 GHz | 400 MHz | 1024 | 390.6 kHz | Yes | 40 dBm |
| `elaa_m_1k_28g` | 32×16 | 512 | 28 GHz | 400 MHz | 1024 | 390.6 kHz | Yes | 40 dBm |

- **Nr = 2** (cross-polarized UE antenna, 3GPP default)
- **Synthetic array**: physical single-antenna BS, Sionna constructs virtual UPA in post-processing
- **SC spacing**: Fixed at ~390.6 kHz across all configs (FR1: 100M/256, FR2/FR3: 400M/1024)

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

### Temporal (1000 snapshots)
- **Gauss-Markov mobility model** (α=0.75) with building collision avoidance
- dt = 10 ms, total = 10 seconds simulated time
- UE speeds: 0 m/s (static), 1 m/s (pedestrian), 8.3 m/s (30 km/h vehicle)
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
