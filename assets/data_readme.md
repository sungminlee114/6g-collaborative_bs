# Channel Dataset Summary

Sionna RT ray-tracing channel datasets for 6G O-RAN collaborative BS research.
Scene: Munich city center, 4 BS sites.

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

Standard references: TS 38.211 Table 4.2-1 (numerology), Table 4.3.2-1 (slots/frame).

### Subcarrier Spacing vs Channel Sampling

`cfg.scs` (120 kHz) is the 3GPP system SCS. `cfg.subcarrier_spacing` (~114 kHz for 200 MHz/1584) is the Sionna channel frequency sampling interval. With μ=3 and BW=200 MHz, we use the full 3GPP subcarrier count (N_RB=132, N_sc=1584), so these two values are approximately equal.

### DMRS Configuration (3GPP TS 38.211 Section 6.4.1.1)

Uplink PUSCH DMRS for channel estimation at BS.

| Parameter | Value | Standard Reference |
|-----------|-------|-------------------|
| DMRS Type | 2 (Configuration Type 2) | Table 6.4.1.1.3-2 |
| Frequency pattern | 2 REs per 6 SC (CDM group 0: k={0,1,6,7,...}) | Table 6.4.1.1.3-2 |
| Pilot SC per DMRS symbol | N_sc × 2/6 = 528 | - |
| Mapping type | A (dmrs-TypeA-Position = pos2, l0 = 2) | Table 6.4.1.1.3-3 |
| Additional position | pos1 (1 additional → 2 total) | Table 6.4.1.1.3-3 |
| DMRS symbol indices | {2, 11} (14-symbol slot) | Table 6.4.1.1.3-3 |
| Total pilot REs/slot | 528 × 2 = 1,056 | - |
| Total REs/slot | 1,584 × 14 = 22,176 | - |
| **Pilot density** | **4.8%** | - |

## Config Matrix (CE-skip paper)

| Config | Array (UPA) | Nt | Freq | Band | μ | SCS | Slot | BW | N_sc | N_RB | Power |
|--------|------------|----:|-----:|------|--:|----:|-----:|----:|-----:|-----:|------:|
| `elaa_m_*_15g` | 32×16 | 512 | 15 GHz | FR3 | 3 | 120 kHz | 0.125 ms | 200 MHz | 1584 | 132 | 40 dBm |
| `elaa_m_*_28g` | 32×16 | 512 | 28 GHz | FR2 | 3 | 120 kHz | 0.125 ms | 200 MHz | 1584 | 132 | 40 dBm |

BW 200 MHz @ μ=3: 3GPP TS 38.101-2 Table 5.3.2-1 (FR2 supported channel bandwidth).

- **Nr = 2** (cross-polarized UE antenna, 3GPP default)
- **Synthetic array**: physical single-antenna BS, Sionna constructs virtual UPA in post-processing

## Temporal Data (CE-skip experiments)

| Parameter | Value |
|-----------|-------|
| Snapshots | 800 |
| UEs | 15 (5 speeds × 3 each) |
| Speeds (m/s) | 0, 1, 2, 5, 8.3 |
| dt | 0.125 ms (= slot duration, from config) |
| Total duration | 100 ms |
| Mobility model | Gauss-Markov (α=0.95) |
| Trajectories | Pre-computed, shared across configs |
| Channel generation | Full Sionna RT re-trace per snapshot |

### Coherence Time Reference

| Frequency | Speed | f_d (Hz) | T_c (ms) | T_c (slots) | Coverage (800 snaps) |
|-----------|-------|----------|----------|-------------|---------------------|
| 15 GHz | 1 m/s | 50 | 8.5 | 68 | 11.8 T_c |
| 15 GHz | 8.3 m/s | 415 | 1.0 | 8 | 100 T_c |
| 28 GHz | 1 m/s | 93 | 4.5 | 36 | 22 T_c |
| 28 GHz | 8.3 m/s | 775 | 0.55 | 4.4 | 182 T_c |

All speed/frequency combinations have ≥10 T_c coverage.

### Estimated Disk Usage (ELAA-M, per config)

H ∈ ℂ^(1584 × 512 × 2) per snapshot per UE = 13 MB (complex64).
800 snapshots × 15 UEs ≈ **156 GB** per config.

## Generation

```bash
# Temporal data (trajectories + channels)
bash scripts/run_temporal_datagen_single.sh

# Single preset
bash scripts/run_temporal_datagen_single.sh 15g
bash scripts/run_temporal_datagen_single.sh 28g
```

Hardware: 8× A100 40GB, 128-core Xeon Gold 6530, 1TB RAM.
