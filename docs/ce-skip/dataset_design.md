# CE-Skip Dataset Design: 3GPP Standard Compliance

## OFDM System Parameters

| Parameter | Value | 3GPP Reference |
|-----------|-------|---------------|
| Numerology (μ) | 3 | TS 38.211 Table 4.2-1 |
| SCS | 120 kHz (= 2^3 × 15 kHz) | TS 38.211 Table 4.2-1 |
| Slot duration | 0.125 ms (= 1ms / 2^3) | TS 38.211 Section 4.3.2, Table 4.3.2-1 |
| Symbols/slot | 14 (Normal CP) | TS 38.211 Table 4.3.2-1 |
| Bandwidth | 200 MHz | TS 38.101-2 Table 5.3.2-1 |
| N_RB | 132 | TS 38.101-2 Table 5.3.2-1 |
| N_sc | 1584 (= 132 × 12) | - |
| Carrier freq | 15 GHz (FR3) / 28 GHz (FR2) | - |

### BW 선택 근거
- 200 MHz는 FR2 μ=3에서 3GPP 표준 지원 BW (TS 38.101-2 Table 5.3.2-1)
- 6G FR3에서 200-400 MHz가 industry consensus (Samsung, Nokia, Ericsson)
- 400 MHz도 가능하지만 N_sc=3168로 데이터 2배, 200 MHz가 practical

### μ=3 선택 근거
- 28 GHz (FR2): μ=3 (120 kHz)이 표준 data channel SCS
- 15 GHz (FR3): 미규격이나, FR2에 가까운 upper mid-band → mmWave급 SCS 적용
- 두 config를 동일 numerology로 통일하여 순수 주파수 효과만 비교

## DMRS Pilot Configuration

### 표준 근거

**Uplink CE를 위한 PUSCH DMRS** (TS 38.211 Section 6.4.1.1).
PDSCH DMRS는 Section 7.4.1.1이나, BS에서의 uplink CE이므로 PUSCH 기준.

| RRC Parameter | Value | Effect | Standard Reference |
|---------------|-------|--------|-------------------|
| `dmrs-Type` | type2 | Configuration Type 2 (paired REs, 3 CDM groups) | TS 38.211 Table 6.4.1.1.3-2 |
| `dmrs-TypeA-Position` | pos2 | First DMRS at symbol l0=2 | TS 38.331 |
| `dmrs-AdditionalPosition` | pos1 | 1 additional DMRS symbol | TS 38.211 Table 6.4.1.1.3-3 |
| `maxLength` | len1 | Single-symbol DMRS | TS 38.211 |

### Type 2 주파수 패턴 (Table 6.4.1.1.3-2)

```
k = 6n + k' + Δ

CDM group 0 (Δ=0): k' ∈ {0,1} → SC positions {0, 1, 6, 7, 12, 13, ...}
CDM group 1 (Δ=2): k' ∈ {0,1} → SC positions {2, 3, 8, 9, 14, 15, ...}
CDM group 2 (Δ=4): k' ∈ {0,1} → SC positions {4, 5, 10, 11, 16, 17, ...}
```

각 CDM group은 12 SC 중 4개 사용 (33%). 단일 CDM group (rank-1 UE) 기준 2/6 = 33%.

주의: 3GPP 표준은 "comb-3"이라는 용어를 사용하지 않음. 공식 명칭은 "DMRS configuration type 2".

### 시간축 DMRS 위치 (Table 6.4.1.1.3-3)

Mapping type A, l0=2, ld=14 (full slot), single-symbol DMRS:

| additionalPosition | DMRS symbol indices |
|---|---|
| pos0 | {2} |
| **pos1** | **{2, 11}** |
| pos2 | {2, 8, 11} |
| pos3 | {2, 5, 8, 11} |

**우리 설정: pos1 → symbols {2, 11}**

### Pilot 밀도 계산

```
Per DMRS symbol: 1584 × 2/6 = 528 pilot subcarriers
DMRS symbols per slot: 2
Total pilot REs: 528 × 2 = 1,056
Total REs per slot: 1,584 × 14 = 22,176
Pilot density: 1,056 / 22,176 = 4.76%
```

## Config Matrix (CE-skip paper)

| Config | Array (UPA) | Nt | Freq | Band | μ | SCS | Slot | BW | N_sc | N_RB |
|--------|------------|----:|-----:|------|--:|----:|-----:|----:|-----:|-----:|
| `elaa_m_*_15g` | 32×16 | 512 | 15 GHz | FR3 | 3 | 120 kHz | 0.125 ms | 200 MHz | 1584 | 132 |
| `elaa_m_*_28g` | 32×16 | 512 | 28 GHz | FR2 | 3 | 120 kHz | 0.125 ms | 200 MHz | 1584 | 132 |

- **Nr = 2** (cross-polarized UE antenna)
- **Synthetic array**: physical single-antenna BS, Sionna constructs virtual UPA

## Temporal Data

| Parameter | Value |
|-----------|-------|
| Snapshots | 800 |
| UEs | 15 (5 speeds × 3 each) |
| Speeds (m/s) | 0, 1, 2, 5, 8.3 |
| dt | 0.125 ms (= slot duration) |
| Total duration | 100 ms |
| Mobility model | Gauss-Markov (α=0.95) |

## 시뮬레이션과 실제 시스템의 차이

### 반영된 것 (3GPP compliant)
- TS 38.211 numerology (μ=3, SCS=120 kHz)
- TS 38.101-2 channel bandwidth (200 MHz @ FR2)
- TS 38.211 DMRS Type 2 frequency pattern (Section 6.4.1.1.3, Table 6.4.1.1.3-2)
- TS 38.211 DMRS symbol positions (Table 6.4.1.1.3-3, mapping type A, pos1 → {2, 11})
- Slot 단위 temporal sampling
- Sionna RT ray-tracing 채널 (site-specific)

### 단순화된 것 (acknowledged limitations)

| 단순화 | 설명 | CE-skip 영향 |
|--------|------|-------------|
| Slot 내 time variation 무시 | 14 symbol 간 채널 일정 가정. T_c >> 0.125 ms이므로 유효 | 없음 (slot 간 스케줄링) |
| 단일 CDM group | OCC multiplexing 미반영, rank-1 가정 | δ metric 무관 |
| No inter-cell interference | 단일 셀 시나리오 | δ가 cleaner → optimistic |
| No pilot contamination | 단일 셀이므로 해당 없음 | LS 품질 과대평가 |

## 코드 구현

```python
from src.config import SceneConfig

cfg = SceneConfig.from_preset("munich_elaa_m_1k_15g")

# 3GPP parameters
cfg.scs                    # 120,000 Hz
cfg.slot_duration_s        # 0.000125 s
cfg.numerology_mu          # 3

# DMRS
cfg.dmrs_type              # 2
cfg.pilot_sc_per_dmrs_symbol  # 528
cfg.dmrs_symbols_per_slot     # 2
cfg.total_pilot_res           # 1056
cfg.pilot_density             # 0.0476

# Pilot mask
freq_mask = cfg.pilot_mask()           # (1584,) bool
grid_mask = cfg.pilot_mask(as_2d=True) # (14, 1584) bool
```

## Changelog

- 2026-03-18: Initial 7-config matrix (3×3 - FR1+ELAA)
- 2026-03-25: 3GPP numerology SSOT (μ=3, BW=200MHz, N_sc=1584, DMRS Type 2)
- 2026-03-25: DMRS symbol positions corrected {2,7} → {2,11} per Table 6.4.1.1.3-3
