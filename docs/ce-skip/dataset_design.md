# CE-Skip Dataset Design: 7-Config Matrix

## Design Philosophy

**3-row × 3-col design** (with FR1+ELAA excluded): Antenna scale × Frequency band, consistent SC spacing (~390 kHz) across all configs.

FR1 (3.5 GHz) is represented only at MIMO scale (8×8). ELAA scaling experiments use FR3/FR2 where near-field effects are meaningful. See [exclusion rationale](#why-no-fr1--elaa).

---

## Configuration Matrix (7 configs)

|  | **3.5 GHz (FR1)** | **15 GHz (FR3)** | **28 GHz (FR2)** |
|---|---|---|---|
| **8×8 (64 ant)** | `5g_mimo_3g5` | `mimo_15g` | `mimo_28g` |
| **16×16 (256 ant)** | — | `elaa_s_1k_15g` | `elaa_s_1k_28g` |
| **32×16 (512 ant)** | — | `elaa_m_1k_15g` | `elaa_m_1k_28g` |

### SC Spacing Consistency

All configs use **~390.6 kHz** subcarrier spacing:
- FR1 (3.5 GHz): BW = 100 MHz, SC = 256 → 390.6 kHz/SC
- FR2/FR3 (15/28 GHz): BW = 400 MHz, SC = 1024 → 390.6 kHz/SC

### Common Parameters (all 7 configs)

| Parameter | Value |
|-----------|-------|
| Scene | Sionna RT, Munich UMi |
| Num BS | 8 (identical positions) |
| BS split | 2 train / 2 val / 4 test |
| Num UE | 100 per snapshot |
| UE dist | 10–150 m |
| UE antennas | 2 (cross-pol variants) |
| Snapshots | 100 (independent) / 1000 (temporal) |
| Guard band | 10% |
| Temperature | 293 K |
| Max depth | 5 reflections |

### Per-Config Details

| Config | Array | Freq | BW | SC | Power | Synth Array | data_dir |
|--------|-------|------|-----|-----|-------|-------------|----------|
| `5g_mimo_3g5` | 8×8 | 3.5 GHz | 100 MHz | 256 | 46 dBm | false | `channels_5g_mimo_3g5` |
| `mimo_15g` | 8×8 | 15 GHz | 400 MHz | 1024 | 40 dBm | true | `channels_mimo_15g` |
| `mimo_28g` | 8×8 | 28 GHz | 100 MHz | 256 | 40 dBm | true | `channels_mimo_28g` |
| `elaa_s_1k_15g` | 16×16 | 15 GHz | 400 MHz | 1024 | 40 dBm | true | `channels_elaa_s_1k_15g` |
| `elaa_s_1k_28g` | 16×16 | 28 GHz | 400 MHz | 1024 | 40 dBm | true | `channels_elaa_s_1k_28g` |
| `elaa_m_1k_15g` | 32×16 | 15 GHz | 400 MHz | 1024 | 40 dBm | true | `channels_elaa_m_1k_15g` |
| `elaa_m_1k_28g` | 32×16 | 28 GHz | 400 MHz | 1024 | 40 dBm | true | `channels_elaa_m_1k_28g` |

---

## Experiment Design: Isolate One Axis at a Time

```
                     FR1 (3.5 GHz)     FR3 (15 GHz)      FR2 (28 GHz)
                     BW=100M SC=256    BW=400M SC=1024   BW=400M SC=1024

8×8   (64 ant)       5g_mimo_3g5       mimo_15g           mimo_28g
                                            |                   |
                                            ↓ Antenna scale     ↓
16×16 (256 ant)           —            elaa_s_1k_15g      elaa_s_1k_28g
                                            |                   |
                                            ↓                   ↓
32×16 (512 ant)           —            elaa_m_1k_15g      elaa_m_1k_28g

                     ←──────── Frequency axis ────────→
```

### Controlled Comparisons

| Comparison | Config pair/set | What varies | What's fixed |
|------------|----------------|-------------|--------------|
| **Freq effect (FR1↔FR2)** | `5g_mimo_3g5` vs `mimo_28g` | Freq (3.5→28 GHz) | Ant 8×8 |
| **Freq effect (FR3↔FR2)** | `mimo_15g` vs `mimo_28g` | Freq (15→28 GHz) | Ant 8×8 |
| **Freq effect (same ant)** | `elaa_s_1k_15g` vs `elaa_s_1k_28g` | Freq only | Ant 16×16, SC 1024 |
| **Scale effect @ FR3** | `mimo_15g` → `elaa_s_1k_15g` → `elaa_m_1k_15g` | Ant (64→256→512) | Freq 15G, SC 1024 |
| **Scale effect @ FR2** | `mimo_28g` → `elaa_s_1k_28g` → `elaa_m_1k_28g` | Ant (64→256→512) | Freq 28G, SC 1024 |
| **Cross-freq scale** | FR3 ELAA triplet vs FR2 ELAA triplet | Freq (15↔28 GHz) | Same ant progression |
| **Mobility effect** | Any config × {0, 1, 8.3, 33} m/s | Speed only | Everything else |
| **Multi-BS** | All configs have 8 BS | Per-BS skip pattern | Same scene, positions |

---

## Data Size Estimates

| Config | Ant | SC | Per-snapshot | 100 snap × 8 BS |
|--------|-----|----|-------------|------------------|
| `5g_mimo_3g5` | 64 | 256 | ~6 MB | **~5G** |
| `mimo_15g` | 64 | 1024 | ~25 MB | **~20G** |
| `mimo_28g` | 64 | 256 | ~6 MB | **~5G** |
| `elaa_s_1k_*` | 256 | 1024 | ~50 MB | **~40G** (×2) |
| `elaa_m_1k_*` | 512 | 1024 | ~100 MB | **~80G** (×2) |

Total estimated: **~270G** for all 7 configs (independent mode)

---

## Key Design Decisions

### Why no FR1 + ELAA?

FR1 (3.5 GHz) + ELAA (256/512 ant) was excluded from the matrix:
- **Near-field negligible**: At 3.5 GHz, Rayleigh distance for 16×16 UPA ≈ 11m, 32×16 ≈ 44m. Most UEs (10–150m) are in far-field → ELAA near-field advantages don't apply.
- **Physical size impractical**: 32×16 UPA at λ/2 = 4.3cm → **137×69cm** panel. Physically installable but rarely deployed in practice.
- **Literature gap for wrong reason**: Papers don't study this combo because the physics doesn't warrant it, not because it's unexplored territory.
- **CE-skip positioning**: Our contribution is about temporal CE scheduling, not near-field CE itself. Far-field ELAA at FR1 would dilute the narrative without adding insight.

Config files (`munich_elaa_s_256_3g5.yaml`, `munich_elaa_m_256_3g5.yaml`) are retained but not used in experiments.

### Why 3 frequency bands?

- **FR1 (3.5 GHz)**: Established 5G baseline, most literature comparisons available
- **FR3 (15 GHz)**: Emerging 6G band, only 2 papers in our relworks use it — novelty kick
- **FR2 (28 GHz)**: Well-studied mmWave, strong baseline for ELAA comparisons

### Why different BW/SC across frequency bands?
- FR1 (3.5 GHz): 100 MHz BW is 3GPP-standard for n78 band
- FR2/FR3 (15/28 GHz): 400 MHz BW matches wider mmWave/FR3 allocations
- SC count adjusted to maintain ~390 kHz spacing for consistent OFDM structure

### Why `synthetic_array: false` for FR1?
- At 3.5 GHz, half-wavelength spacing (~4.3 cm) makes arrays physically large
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
- 2026-03-18: Reduced to 7-config matrix — excluded FR1+ELAA (near-field negligible at 3.5 GHz)
