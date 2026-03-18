# CE-Skip Dataset Design: Configuration Rationale

## Design Philosophy

**"공통 비교 축 + 한 축씩 확장"** — 대부분의 논문과 비교 가능한 baseline을 두고, CE-skip 고유의 차별화 축을 추가.

---

## Active Configurations (실제 사용)

### Tier 0: Literature Baseline (비교 기준)

| Config | `munich_mimo_28g` |
|--------|-------------------|
| Array | **8×8 UPA (64)** |
| Freq | **28 GHz (FR2)** |
| BW | **100 MHz** |
| SC | **1024** |
| Scene | Sionna RT, Munich UMi |
| BS | 8 |
| Role | 논문 비교 기준점 (P41, P56, P63, P70과 동일 스케일) |

### Tier 1: 5G mMIMO Baseline (sub-6 GHz)

| Config | `munich_5g_mimo_3g5` |
|--------|----------------------|
| Array | **8×8 UPA (64)** |
| Freq | **3.5 GHz (FR1)** |
| BW | **100 MHz** |
| SC | **256** |
| Scene | Sionna RT, Munich UMi |
| BS | 8 |
| Role | Sub-6 GHz 비교 (P58, P83, P103과 비교 가능). Far-field only (Rayleigh ~0.6m) |

### Tier 2: ELAA Main (핵심 실험)

| Config | `munich_elaa_s_1k_28g` | `munich_elaa_s_1k_15g` |
|--------|------------------------|------------------------|
| Array | **16×16 UPA (256)** | **16×16 UPA (256)** |
| Freq | **28 GHz (FR2)** | **15 GHz (FR3)** |
| BW | **400 MHz** | **400 MHz** |
| SC | **1024** | **1024** |
| Scene | Sionna RT, Munich UMi | Sionna RT, Munich UMi |
| BS | 8 | 8 |
| Role | ELAA 메인. 대부분 논문보다 큰 스케일 | **FR3 = 6G 핵심 신규 대역, 극소수만 사용** |

### Tier 3: Large ELAA (확장성 검증)

| Config | `munich_elaa_m_1k_28g` | `munich_elaa_m_1k_15g` |
|--------|------------------------|------------------------|
| Array | **32×16 UPA (512)** | **32×16 UPA (512)** |
| Freq | **28 GHz** | **15 GHz** |
| BW | **400 MHz** | **400 MHz** |
| SC | **1024** | **1024** |
| Role | Scalability 검증. P94 (Deep Unrolling)의 512 UPA와 동일 스케일 |

---

## Experiment Design: 한 축씩 변화

```
                        Tier 0          Tier 1          Tier 2          Tier 3
                     (mimo_28g)    (5g_mimo_3g5)    (elaa_s_1k)     (elaa_m_1k)
Antennas               8×8             8×8            16×16           32×16
                        64              64             256             512
                                                        ↑ Scale 축

Frequency             28 GHz         3.5 GHz        28 / 15 GHz    28 / 15 GHz
                                                        ↑ Freq 축 (FR3 킥)

Mobility (temporal)   0 / 1 / 8.3 / 33 m/s  ← 모든 config에 적용
                        ↑ CE-skip 핵심 축: skip 이득이 mobility에 따라 변함

Multi-BS                 8               8              8               8
                        ↑ 논문 대비 압도적 (대부분 1~3)
```

### 비교 가능한 축

| 비교 | Config 쌍 | 변하는 것 | 고정 |
|------|-----------|----------|------|
| Freq 효과 | `elaa_s_1k_28g` vs `elaa_s_1k_15g` | Freq only | Ant, SC, BW |
| Scale 효과 | `mimo_28g` vs `elaa_s_1k_28g` vs `elaa_m_1k_28g` | Ant only (64→256→512) | Freq 28G, SC 1024 |
| Band 효과 | `5g_mimo_3g5` vs `mimo_28g` | Freq (3.5→28 GHz) | Ant 64 |
| Mobility 효과 | Any config × {0, 1, 8.3, 33} m/s | Speed only | Everything else |

---

## Removed Configurations (삭제)

### 삭제 이유: 너무 큰 SC (데이터 비용 >> 노벨티)

| Config | Array | SC | 예상 데이터 크기 | 삭제 이유 |
|--------|-------|----|-----------------|----------|
| `munich_elaa_*_2k_*` (6개) | 16×16 / 32×16 | **2048** | ~150G each | SC 1024 대비 노벨티 미미, 데이터 4배 |
| `munich_elaa_*_4k_*` (6개) | 16×16 / 32×16 / 32×32 | **4096** | ~500G each | 비현실적 크기. L_4k_28g만 568G |

### 삭제 이유: 너무 큰 Array (학습 불가)

| Config | Array | Total | 삭제 이유 |
|--------|-------|-------|----------|
| `munich_elaa_l_*` (6개) | **32×32** | **1024** | GPU 메모리 한계. 1024 SC에서도 336G+. 논문에서 P105만 동일 스케일 |

### 삭제 이유: Legacy (CE-skip 무관)

| Config | 삭제 이유 |
|--------|----------|
| `munich_uma8` | Legacy 2×2 array, CE-skip과 무관 |
| `munich_umi16` | Legacy 2×2 array, 16 BS, CE-skip과 무관 |

---

## Data Size Estimates

| Config | Ant | SC | Per-snapshot | 100 snapshots × 8 BS |
|--------|-----|----|-------------|---------------------|
| `mimo_28g` | 64 | 1024 | ~25 MB | **~20G** |
| `5g_mimo_3g5` | 64 | 256 | ~6 MB | **~5G** (이미 2.3G 있음) |
| `elaa_s_1k_*` | 256 | 1024 | ~50 MB | **~40G** (이미 37-38G 있음) |
| `elaa_m_1k_*` | 512 | 1024 | ~100 MB | **~80G** |

총 예상: ~300G (현재 L_4k 904G 대비 70% 절약)

---

## Key Novelties (논문 기여)

1. **FR3 (15 GHz)**: 전체 논문 중 2편만 사용 → 강한 차별점
2. **ELAA + Temporal**: 0편 → CE-skip 고유 gap
3. **Multi-BS (8)**: 대부분 1~3 → site-specific skip 패턴 분석 가능
4. **Ray-tracing (Sionna RT)**: CDL/TDL 통계 채널 대비 현실적
5. **Mobility sweep**: 0~33 m/s → skip 이득의 mobility 의존성 최초 분석
