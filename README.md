# 6G ELAA Channel Dataset Generation

Sionna RT를 사용한 6G ELAA (Extremely Large Aperture Array) 채널 데이터셋 생성 프로젝트

## 개요

6G ELAA 환경에서의 near-field 채널 특성을 포함한 사이트별 채널 데이터셋을 생성합니다.
Sionna Ray Tracing 기반으로 뮌헨 도시 환경에서 다중 기지국-사용자 간의 CIR/CFR을 추출하며,
3가지 ELAA 스케일(Small/Base/Big) × 3가지 서브캐리어(1024/2048/4096) × 2가지 주파수(15 GHz / 28 GHz) = **18 configs**로 구성됩니다.

## 시뮬레이션 환경

### 시나리오
- **장소**: Munich (Sionna RT 도시 씬)
- **온도**: 20°C (293K)
- **배치**: UMi 16-BS (3GPP TR 38.901, ISD ~130m, 높이 10-24m)

### ELAA Configuration Matrix (2 freq × 3 antenna × 3 SC = 18 configs)

**독립 축:**

| 축 | 옵션 | 설명 |
|----|------|------|
| **Frequency** | 15 GHz (FR3), 28 GHz (FR2) | 주파수 대역 |
| **Antenna (TX)** | Small 16×16 (256), Base 32×16 (512), Big 32×32 (1024) | ELAA 스케일 |
| **Subcarriers** | 1k (1024), 2k (2048), 4k (4096) | OFDM 부반송파 수 |

**Preset 네이밍**: `munich_elaa_{s|m|l}_{1k|2k|4k}_{15g|28g}`

| Scale | SC | Preset (15g / 28g) | TX Array | BW | Elements |
|-------|----|--------------------|----------|----|----------|
| Small | 1k | `munich_elaa_s_1k_15g` / `_28g` | 16×16 | 400 MHz | 256 |
| Small | 2k | `munich_elaa_s_2k_15g` / `_28g` | 16×16 | 800 MHz | 256 |
| Small | 4k | `munich_elaa_s_4k_15g` / `_28g` | 16×16 | 1.6 GHz | 256 |
| Base | 1k | `munich_elaa_m_1k_15g` / `_28g` | 32×16 | 400 MHz | 512 |
| Base | 2k | `munich_elaa_m_2k_15g` / `_28g` | 32×16 | 800 MHz | 512 |
| Base | 4k | `munich_elaa_m_4k_15g` / `_28g` | 32×16 | 1.6 GHz | 512 |
| Big | 1k | `munich_elaa_l_1k_15g` / `_28g` | 32×32 | 400 MHz | 1024 |
| Big | 2k | `munich_elaa_l_2k_15g` / `_28g` | 32×32 | 800 MHz | 1024 |
| Big | 4k | `munich_elaa_l_4k_15g` / `_28g` | 32×32 | 1.6 GHz | 1024 |

**Rayleigh Distance** (near-field 경계, d_R = 2D²/λ):

| Scale | Aperture @15G | d_R @15G | Aperture @28G | d_R @28G |
|-------|---------------|----------|---------------|----------|
| Small (16×16) | 160 mm | ~32 m | 86 mm | ~60 m |
| Base (32×16) | 320 mm | ~128 m | 171 mm | ~238 m |
| Big (32×32) | 320 mm | ~512 m | 171 mm | ~955 m |

> d_R 이내의 UE는 near-field 영역에 위치하여 spherical wave 전파 특성을 경험합니다.

### Legacy Configs (5G mmWave)
기존 세팅도 유지됩니다:
- `munich_uma8` — 8 BS, 2×2 TX (4 ant), 15 GHz, 1024 SC
- `munich_umi16` — 16 BS, 2×2 TX (4 ant), 15 GHz, 1024 SC

## 네트워크 구성

- **기지국(BS) 수**: 8개 (UMi 16-BS 중 중앙 영역 선택)
- **커버리지 영역**: 400×350m focused area (뮌헨 씬 중앙부)
- **사용자(UE) 수**: 100개/snapshot
- **BS 높이**: 10-24m (건물 옥상, UMi 스펙)
- **송신 전력**: 40 dBm
- **Mean NN-ISD**: ~196m
- **Coverage**: @150m = 99%

### BS 배치 및 Split (2 train / 2 val / 4 test)

| BS | 위치 (x, y, z) | 높이 | 방향 | Split |
|----|---------------|------|------|-------|
| BS0 | (3.7, -69.7, 24.0) | 24m | center | **Train** |
| BS1 | (-319.0, -185.9, 15.1) | 15m | SW | Test |
| BS2 | (-123.7, 169.2, 16.7) | 17m | N | Val |
| BS3 | (122.4, -296.0, 13.0) | 13m | SE | Test |
| BS4 | (-140.0, -296.6, 14.8) | 15m | S | Val |
| BS5 | (-186.6, 32.3, 18.1) | 18m | NW | **Train** |
| BS6 | (119.5, 161.5, 17.8) | 18m | NE | Test |
| BS7 | (-325.7, 38.0, 10.0) | 10m | W | Test |

- **Train** (BS0, BS5): 단일 BS 최적화 학습
- **Val** (BS2, BS4): 하이퍼파라미터 튜닝, early stopping
- **Test** (BS1, BS3, BS6, BS7): zero-shot 일반화 평가, 높이/방향 다양

## 안테나 설정

### 기지국 (Transmitter) — ELAA

| Scale | 배열 | 안테나 수 | 안테나 간격 | 편파 | Aperture (0.5λ 간격) |
|-------|------|----------|------------|------|----------------------|
| Small | 16×16 Planar | 256 | 0.5λ | V | ~160 mm (15G) / ~86 mm (28G) |
| Base | 32×16 Planar | 512 | 0.5λ | V | ~320×160 mm (15G) / ~171×86 mm (28G) |
| Big | 32×32 Planar | 1024 | 0.5λ | V | ~320 mm (15G) / ~171 mm (28G) |

- **패턴**: Isotropic
- **SyntheticArray**: Big config에서는 `True` 권장 (메모리 절약)

### 사용자 단말 (Receiver)
- **안테나 배열**: 1×1 (2개 교차편파 안테나)
- **안테나 간격**: 0.5λ
- **편파**: 교차편파(Cross)
- **패턴**: Dipole
- **UE 디바이스 다양성**: [1×1 cross], [1×2 V], [2×2 V]

## 사용자 위치 샘플링

### 샘플링 조건
- **SINR 범위**: 2 ~ 40 dB
- **거리 제약**: 10 ~ 100m (기지국으로부터)
- **TX Association**: 활성화 (각 UE를 최적 기지국에 자동 연결)

### Radio Map 생성 파라미터
- **셀 크기**: 1m × 1m
- **기지국당 샘플**: 10,000,000 rays
- **최대 반사 깊이**: 5회
- **포함 전파 효과**: LoS, Specular Reflection, Diffuse Reflection, Refraction, Diffraction, Edge Diffraction

## 생성되는 데이터셋

### Preset 사용법
```python
from src.config import SceneConfig
cfg = SceneConfig.from_preset("munich_elaa_m_2k_28g")  # Base ELAA, 28 GHz
print(cfg.num_tx_ant)       # 512
print(cfg.num_subcarriers)  # 2048
```

### 1. 채널 임펄스 응답 (CIR)

```python
a, tau = paths.cir(
    normalize_delays=True,
    associated_tx_idxs=associated_tx_idxs,
    out_type="numpy"
)
```

#### 데이터 형태 (Base ELAA 28 GHz 예시)
| 변수 | Shape | 설명 |
|------|-------|------|
| `a` | `(100, 2, 1, 512, N_paths, 1)` | 복소 채널 이득 |
| `tau` | `(100, 2, 1, 512, N_paths)` | 경로별 전파 지연 시간 (초) |
| `valids` | `(100, 2, 1, 512, N_paths)` | 유효 경로 마스크 |

#### Shape 차원 설명
- `100`: RX 수 (사용자 수)
- `2`: RX 안테나 수 (cross-pol)
- `1`: 연결된 TX 수 (각 UE는 1개 BS에만 연결)
- `512`: TX 안테나 수 (ELAA 스케일에 따라 256/512/1024)
- `N_paths`: 최대 경로 수
- `1`: 시간 샘플 수

### 2. 채널 주파수 응답 (CFR)

```python
h_freq = paths.cfr(
    frequencies=frequencies,
    associated_tx_idxs=associated_tx_idxs,
    normalize=True,
    normalize_delays=True,
    out_type="numpy"
)
```

#### 데이터 형태 (Base ELAA 28 GHz 예시)
| 변수 | Shape | 설명 |
|------|-------|------|
| `h_freq` | `(100, 2, 1, 512, 1, 2048)` | 복소 주파수 응답 |

#### Shape 차원 설명
- `100`: RX 수
- `2`: RX 안테나 수
- `1`: 연결된 TX 수
- `512`: TX 안테나 수 (ELAA 스케일에 따라 변동)
- `1`: 시간 샘플 수
- `2048`: OFDM 부반송파 수 (스케일에 따라 1024/2048/4096)

#### OFDM 파라미터 (Base config)
- **유효 대역폭**: 720 MHz (90% × 800 MHz)
- **부반송파 간격**: 351.56 kHz
- **주파수 범위**: ±0.360 GHz (캐리어 기준)

### 3. 메타데이터 (`rx_infos`)

각 UE의 정보를 담은 100개의 딕셔너리 리스트:

```python
{
    "tx_id": int,        # 연결된 기지국 ID (0~3)
    "idx_in_tx": int,    # 해당 기지국 내 UE 인덱스
    "position": ndarray  # 3D 위치 [x, y, z]
}
```

## 전파 경로 계산 설정

```python
paths = PathSolver()(
    scene=scene,
    max_depth=5,
    max_num_paths_per_src=100_000,
    samples_per_src=100_000,
    los=True,
    specular_reflection=True,
    diffuse_reflection=True,
    refraction=True,
    synthetic_array=False,       # Small/Base: False, Big: True 권장
    seed=41
)
```

## 출력 데이터 요약 (스케일별)

### Small ELAA (256 ant, 1024 SC)
| 데이터 | 타입 | Shape |
|--------|------|-------|
| **a** | complex128 | (100, 2, 1, 256, N_paths, 1) |
| **tau** | float64 | (100, 2, 1, 256, N_paths) |
| **h_freq** | complex128 | (100, 2, 1, 256, 1, 1024) |

### Base ELAA (512 ant, 2048 SC)
| 데이터 | 타입 | Shape |
|--------|------|-------|
| **a** | complex128 | (100, 2, 1, 512, N_paths, 1) |
| **tau** | float64 | (100, 2, 1, 512, N_paths) |
| **h_freq** | complex128 | (100, 2, 1, 512, 1, 2048) |

### Big ELAA (1024 ant, 4096 SC)
| 데이터 | 타입 | Shape |
|--------|------|-------|
| **a** | complex128 | (100, 2, 1, 1024, N_paths, 1) |
| **tau** | float64 | (100, 2, 1, 1024, N_paths) |
| **h_freq** | complex128 | (100, 2, 1, 1024, 1, 4096) |

## Near-Field 특성

6G ELAA의 핵심은 **near-field beamfocusing** 효과:

- 기존 far-field (plane wave) 가정이 깨지고 spherical wave 전파
- Rayleigh distance 이내의 UE에서 빔이 특정 **지점**에 집중 (vs far-field: 특정 **방향**)
- 큰 배열 + 높은 주파수일수록 near-field 영역 확대
- Big config (1024 ant, 28 GHz)에서 Rayleigh distance ~955m → 거의 모든 UE가 near-field

## 사용 방법

### 환경 설정
```python
import sionna.rt
import mitsuba as mi
import drjit as dr

mi.set_variant("cuda_ad_mono_polarized", "llvm_ad_mono_polarized")
```

### 씬 로드 및 파라미터 설정
```python
from src.config import SceneConfig

cfg = SceneConfig.from_preset("munich_elaa_m_2k_28g")

scene = load_scene(sionna.rt.scene.munich, merge_shapes=True)
scene.frequency = cfg.frequency
scene.bandwidth = cfg.bandwidth
scene.temperature = cfg.temperature
```

### 데이터 생성 워크플로우
1. Preset 선택 (`SceneConfig.from_preset`)
2. Radio Map 생성 (`compute_radio_map`)
3. 사용자 위치 샘플링 (`sample_user_positions`)
4. 경로 계산 (`PathSolver`)
5. CIR/CFR 추출

## 주요 특징

- **6G ELAA**: 256~1024 안테나 소자의 초대형 배열
- **Near-field Channel**: Spherical wave 전파, beamfocusing 효과
- **Multi-scale**: Small/Base/Big 3단계 스케일로 연구 유연성
- **Dual-band**: 15 GHz (FR3) + 28 GHz (FR2) 지원
- **Realistic Urban**: Sionna RT 뮌헨 도시 레이트레이싱
- **Multi-cell**: 8 BS UMi (focused 400×350m)
- **OFDM Ready**: 1024/2048/4096 부반송파 주파수 응답

## 응용 분야

- 6G ELAA near-field 채널 모델링
- Near-field beamfocusing 알고리즘 검증
- Site-specific 채널 추정 DNN 학습
- FL 기반 multi-cell 협력 학습
- Task-agnostic site representation 연구

## 참고사항

- Big config (1024 ant) 사용 시 `SyntheticArray=True` 권장 (GPU 메모리 절약)
- 메모리 최적화를 위해 `dr.flush_malloc_cache()` 사용
- Preview/Render 기능으로 3D 시각화 가능
- 모든 config preset 목록: `SceneConfig.list_presets()`
