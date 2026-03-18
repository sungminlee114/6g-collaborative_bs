# CE-Skip Dataset Design: 9-Config Matrix

## Design Philosophy

**3x3 factorial design**: Antenna scale x Frequency band, with consistent SC spacing (~390 kHz) across all configs for fair comparison.

---

## Configuration Matrix (9 configs)

|  | **3.5 GHz (FR1)** | **15 GHz (FR3)** | **28 GHz (FR2)** |
|---|---|---|---|
| **8x8 (64 ant)** | `5g_mimo_3g5` | `mimo_15g` | `mimo_28g` |
| **16x16 (256 ant)** | `elaa_s_256_3g5` | `elaa_s_1k_15g` | `elaa_s_1k_28g` |
| **32x16 (512 ant)** | `elaa_m_256_3g5` | `elaa_m_1k_15g` | `elaa_m_1k_28g` |

### SC Spacing Consistency

All configs use **~390.6 kHz** subcarrier spacing:
- FR1 (3.5 GHz): BW = 100 MHz, SC = 256 → 390.6 kHz/SC
- FR2/FR3 (15/28 GHz): BW = 400 MHz, SC = 1024 → 390.6 kHz/SC

### Common Parameters (all 9 configs)

| Parameter | Value |
|-----------|-------|
| Scene | Sionna RT, Munich UMi |
| Num BS | 8 (identical positions) |
| BS split | 2 train / 2 val / 4 test |
| Num UE | 100 per snapshot |
| UE dist | 10–150 m |
| UE antennas | 2 (cross-pol variants) |
| Snapshots | 100 |
| Guard band | 10% |
| Temperature | 293 K |
| Max depth | 5 reflections |

### Per-Config Details

| Config | Array | Freq | BW | SC | Power | Synth Array | data_dir |
|--------|-------|------|-----|-----|-------|-------------|----------|
| `5g_mimo_3g5` | 8×8 | 3.5 GHz | 100 MHz | 256 | 46 dBm | false | `channels_5g_mimo_3g5` |
| `mimo_15g` | 8×8 | 15 GHz | 400 MHz | 1024 | 40 dBm | true | `channels_mimo_15g` |
| `mimo_28g` | 8×8 | 28 GHz | 100 MHz | 256 | 40 dBm | true | `channels_mimo_28g` |
| `elaa_s_256_3g5` | 16×16 | 3.5 GHz | 100 MHz | 256 | 46 dBm | false | `channels_elaa_s_256_3g5` |
| `elaa_s_1k_15g` | 16×16 | 15 GHz | 400 MHz | 1024 | 40 dBm | true | `channels_elaa_s_1k_15g` |
| `elaa_s_1k_28g` | 16×16 | 28 GHz | 400 MHz | 1024 | 40 dBm | true | `channels_elaa_s_1k_28g` |
| `elaa_m_256_3g5` | 32×16 | 3.5 GHz | 100 MHz | 256 | 46 dBm | false | `channels_elaa_m_256_3g5` |
| `elaa_m_1k_15g` | 32×16 | 15 GHz | 400 MHz | 1024 | 40 dBm | true | `channels_elaa_m_1k_15g` |
| `elaa_m_1k_28g` | 32×16 | 28 GHz | 400 MHz | 1024 | 40 dBm | true | `channels_elaa_m_1k_28g` |

---

## Experiment Design: Isolate One Axis at a Time

```
                     FR1 (3.5 GHz)     FR3 (15 GHz)      FR2 (28 GHz)
                     BW=100M SC=256    BW=400M SC=1024   BW=100M SC=256
                                                         (or 400M/1024)
8×8   (64 ant)       5g_mimo_3g5       mimo_15g           mimo_28g
                          |                |                   |
                          ↓ Antenna scale  ↓                   ↓
16×16 (256 ant)      elaa_s_256_3g5    elaa_s_1k_15g      elaa_s_1k_28g
                          |                |                   |
                          ↓                ↓                   ↓
32×16 (512 ant)      elaa_m_256_3g5    elaa_m_1k_15g      elaa_m_1k_28g

                     ←──────── Frequency axis ────────→
```

### Controlled Comparisons

| Comparison | Config pair/set | What varies | What's fixed |
|------------|----------------|-------------|--------------|
| **Freq effect (FR1↔FR2)** | `5g_mimo_3g5` vs `mimo_28g` | Freq (3.5→28 GHz) | Ant 8×8, SC 256 |
| **Freq effect (FR3↔FR2)** | `mimo_15g` vs any FR2 ELAA | Freq (15→28 GHz) | SC 1024 |
| **Freq effect (same ant)** | `elaa_s_1k_15g` vs `elaa_s_1k_28g` | Freq only | Ant 16×16, SC 1024 |
| **Scale effect @ FR1** | `5g_mimo_3g5` → `elaa_s_256_3g5` → `elaa_m_256_3g5` | Ant (64→256→512) | Freq 3.5G, SC 256 |
| **Scale effect @ FR3** | `mimo_15g` → `elaa_s_1k_15g` → `elaa_m_1k_15g` | Ant (64→256→512) | Freq 15G, SC 1024 |
| **Scale effect @ FR2** | `mimo_28g` → `elaa_s_1k_28g` → `elaa_m_1k_28g` | Ant (64→256→512) | Freq 28G |
| **Mobility effect** | Any config × {0, 1, 8.3, 33} m/s | Speed only | Everything else |
| **Multi-BS** | All configs have 8 BS | Per-BS skip pattern | Same scene, positions |

---

## Data Size Estimates

| Config | Ant | SC | Per-snapshot | 100 snap × 8 BS |
|--------|-----|----|-------------|------------------|
| `5g_mimo_3g5` | 64 | 256 | ~6 MB | **~5G** |
| `mimo_15g` | 64 | 1024 | ~25 MB | **~20G** |
| `mimo_28g` | 64 | 256 | ~6 MB | **~5G** |
| `elaa_s_256_3g5` | 256 | 256 | ~12 MB | **~10G** |
| `elaa_s_1k_*` | 256 | 1024 | ~50 MB | **~40G** |
| `elaa_m_256_3g5` | 512 | 256 | ~25 MB | **~20G** |
| `elaa_m_1k_*` | 512 | 1024 | ~100 MB | **~80G** |

Total estimated: **~340G** for all 9 configs

---

## Key Design Decisions

### Why 3x3 factorial?
- Isolates antenna scale effect (rows) and frequency effect (columns)
- Every comparison holds one axis constant, enabling clean ablation
- 5G baselines (row 1) establish performance floor; ELAA rows show scaling

### Why different BW/SC across frequency bands?
- FR1 (3.5 GHz): 100 MHz BW is 3GPP-standard for n78 band
- FR2/FR3 (15/28 GHz): 400 MHz BW matches wider mmWave/FR3 allocations
- SC count adjusted to maintain ~390 kHz spacing for consistent OFDM structure

### Why `synthetic_array: false` for FR1?
- At 3.5 GHz, half-wavelength spacing (~4.3 cm) makes arrays physically large
- 16×16 UPA: ~68 cm; 32×16 UPA: ~137 cm × 68 cm
- Physical modeling matters more at sub-6 GHz (Rayleigh distance within UE range)
- FR2/FR3 arrays are smaller → synthetic approximation is acceptable

### Why `power_dbm` differs?
- FR1: 46 dBm (macro cell standard)
- FR2/FR3: 40 dBm (small cell / mmWave standard)

---

## Changelog

- 2026-03-18: Rationalized to 6 configs (removed legacy/2k/4k/L presets)
- 2026-03-18: Expanded to 9-config 3×3 matrix (added mimo_15g, elaa_s_256_3g5, elaa_m_256_3g5)
- 2026-03-18: Fixed mimo_28g SC 1024→256 for consistent ~390 kHz spacing
