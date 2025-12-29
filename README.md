# mmWave Channel Dataset Generation

Sionna RT를 사용한 15 GHz mmWave 채널 데이터셋 생성 프로젝트

## 📋 개요

이 프로젝트는 Sionna Ray Tracing 라이브러리를 사용하여 도시 환경(뮌헨 씬)에서 5G mmWave 채널 데이터셋을 생성합니다. 다중 기지국과 사용자 간의 채널 임펄스 응답(CIR) 및 채널 주파수 응답(CFR)을 추출합니다.

## 🎯 시뮬레이션 환경

### 시나리오
- **장소**: Munich (Sionna RT 제공 도시 씬)
- **주파수**: 15 GHz (5G FR3 mmWave 대역)
- **대역폭**: 400 MHz
- **온도**: 20°C (293K)
- **파장**: 19.99 mm

### 네트워크 구성
- **기지국(BS) 수**: 8개
- **사용자(UE) 수**: 100개
- **OFDM 부반송파**: 1024개

## 📡 안테나 설정

### 기지국 (Transmitter)
- **안테나 배열**: 2×2 Planar Array (4개 안테나)
- **안테나 간격**: 0.5λ (수직/수평)
- **편파**: 수직(V)
- **패턴**: Isotropic
- **송신 전력**: 40 dBm
- **배치 높이**: 16m ~ 90m (건물 옥상)

### 사용자 단말 (Receiver)
- **안테나 배열**: 1×1 (2개 교차편파 안테나)
- **안테나 간격**: 0.5λ (수직/수평)
- **편파**: 교차편파(Cross)
- **패턴**: Dipole
- **배치 높이**: 지상 레벨

## 🎲 사용자 위치 샘플링

### 샘플링 전략 (`sample_user_positions`)
```python
num_users = 100
```

### 샘플링 조건
- **SINR 범위**: 2 ~ 40 dB
- **거리 제약**: 10 ~ 100m (기지국으로부터)
- **TX Association**: 활성화 (각 UE를 최적 기지국에 자동 연결)
- **기지국별 UE 분배**: 랜덤 분할
  - 예시: `[14, 4, 3, 1, 16, 29, 1, 32]` UE/BS

### Radio Map 생성 파라미터
- **셀 크기**: 1m × 1m
- **기지국당 샘플**: 10,000,000 rays
- **최대 반사 깊이**: 5회
- **포함 전파 효과**:
  - LoS (Line-of-Sight)
  - Specular Reflection (정반사)
  - Diffuse Reflection (산란)
  - Refraction (투과)
  - Diffraction (회절)
  - Edge Diffraction (엣지 회절)

## 📊 생성되는 데이터셋

### 1. 채널 임펄스 응답 (CIR)

```python
a, tau = paths.cir(
    normalize_delays=True,
    associated_tx_idxs=associated_tx_idxs,
    out_type="numpy"
)
```

#### 데이터 형태
| 변수 | Shape | 설명 |
|------|-------|------|
| `a` | `(100, 2, 1, 4, 86, 1)` | 복소 채널 이득 |
| `tau` | `(100, 2, 1, 4, 86)` | 경로별 전파 지연 시간 (초) |
| `valids` | `(100, 2, 1, 4, 86)` | 유효 경로 마스크 |

#### Shape 차원 설명
- `100`: RX 수 (사용자 수)
- `2`: RX 안테나 수
- `1`: 연결된 TX 수 (각 UE는 1개 BS에만 연결)
- `4`: TX 안테나 수
- `86`: 최대 경로 수
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

#### 데이터 형태
| 변수 | Shape | 설명 |
|------|-------|------|
| `h_freq` | `(100, 2, 1, 4, 1, 1024)` | 복소 주파수 응답 |

#### Shape 차원 설명
- `100`: RX 수
- `2`: RX 안테나 수
- `1`: 연결된 TX 수
- `4`: TX 안테나 수
- `1`: 시간 샘플 수
- `1024`: OFDM 부반송파 수

#### OFDM 파라미터
- **유효 대역폭**: 360 MHz (90% × 400 MHz)
- **부반송파 간격**: 351.56 kHz
- **주파수 범위**: -0.180 ~ +0.180 GHz (캐리어 기준 상대 주파수)

### 3. 메타데이터 (`rx_infos`)

각 UE의 정보를 담은 100개의 딕셔너리 리스트:

```python
{
    "tx_id": int,        # 연결된 기지국 ID (0~7)
    "idx_in_tx": int,    # 해당 기지국 내 UE 인덱스
    "position": ndarray  # 3D 위치 [x, y, z]
}
```

## 🔧 전파 경로 계산 설정

```python
paths = PathSolver()(
    scene=scene,
    max_depth=5,                      # 최대 반사 횟수
    max_num_paths_per_src=100_000,    # 기지국당 유지할 최대 경로 수
    samples_per_src=100_000,          # 기지국당 발사할 ray 수
    los=True,
    specular_reflection=True,
    diffuse_reflection=True,
    refraction=True,
    synthetic_array=False,            # 각 안테나 개별 계산
    seed=41
)
```

## 📈 데이터 시각화

### CIR 시각화
- 각 RX당 별도 Figure 생성
- TX × TX 안테나 격자 subplot
- RX 안테나는 색상으로 구분하여 동일 subplot에 표시
- Stem plot으로 경로별 지연(τ)과 크기(|a|) 표시

### CFR 시각화
- 부반송파별 채널 이득 크기 플롯
- 주파수 선택적 페이딩 특성 확인

## 🚀 사용 방법

### 환경 설정
```python
import sionna.rt
import mitsuba as mi
import drjit as dr

mi.set_variant("cuda_ad_mono_polarized", "llvm_ad_mono_polarized")
```

### 씬 로드 및 파라미터 설정
```python
scene = load_scene(sionna.rt.scene.munich, merge_shapes=True)
scene.frequency = 15e9
scene.bandwidth = 400e6
scene.temperature = 293
```

### 데이터 생성 워크플로우
1. Radio Map 생성 (`compute_radio_map`)
2. 사용자 위치 샘플링 (`sample_user_positions`)
3. 경로 계산 (`PathSolver`)
4. CIR/CFR 추출

## 📦 출력 데이터 요약

| 데이터 | 타입 | Shape | 용도 |
|--------|------|-------|------|
| **a** | complex128 | (100, 2, 1, 4, 86, 1) | 채널 이득 (시간 도메인) |
| **tau** | float64 | (100, 2, 1, 4, 86) | 전파 지연 |
| **h_freq** | complex128 | (100, 2, 1, 4, 1, 1024) | 채널 이득 (주파수 도메인) |
| **valids** | bool | (100, 2, 1, 4, 86) | 유효 경로 마스크 |
| **rx_infos** | list[dict] | 100 | UE 위치/연결 정보 |

## 💡 주요 특징

- **Realistic Urban Scenario**: 실제 뮌헨 도시 구조 반영
- **mmWave Propagation**: 15 GHz 고주파 전파 특성 시뮬레이션
- **Multi-path Channel**: 최대 86개 경로의 다중경로 채널
- **MIMO Support**: 4×2 MIMO 채널 데이터
- **OFDM Ready**: 1024 부반송파 주파수 응답 제공
- **TX Association**: 각 UE를 최적 기지국에 자동 연결

## 🔍 응용 분야

- 5G/6G 채널 모델링
- MIMO 빔포밍 알고리즘 검증
- 채널 예측/추정 머신러닝 학습
- 커버리지 분석
- 간섭 분석 및 최적화

## 📝 참고사항

- 메모리 최적화를 위해 `dr.flush_malloc_cache()` 사용
- Preview/Render 기능으로 3D 시각화 가능
