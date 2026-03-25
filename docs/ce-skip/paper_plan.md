# When Not to Estimate: Event-Triggered CE Inference Scheduling for Software-Defined Base Stations

> 최종 정리 (2026.03.18 대화 기반)

---

## 0. 확정된 결정 사항

| 항목 | 결정 | 근거 |
|------|------|------|
| 도구 | Sionna RT only (pyAerial 안 씀) | NF 미지원, 인터페이스 비용 대비 이득 없음 |
| Case | 5G FF (8×8, 3.5GHz) + 6G ELAA (24×24, 15/28GHz) | CE 비용 스케일링 비교 (FF lightweight vs ELAA heavy) |
| 데이터 | temporal trajectory 새로 생성 | independent drops로는 skip 정의 불가 |
| CE 구현 | LS + Genie-LMMSE + DL-CE (3종) | CE-agnostic 프레임워크 증명, Polar-OMP 생략 |
| 기존 데이터 | DL-CE 학습용으로 활용 | independent drops는 CE 정확도 학습에 적합 |
| Contribution | CE scheduling (wrapper), CE 알고리즘 아님 | 어떤 CE든 위에 올릴 수 있음 |

---

## 1. Introduction

### 1.1 배경: CE 비용의 증가

6G ELAA는 수백~수천 개의 안테나 소자를 배치한다. Near-field에서는 구형파(spherical wavefront) 전파로 인해 채널이 각도뿐 아니라 거리에도 의존하며, CE의 탐색 차원이 1D(angle)에서 2D(angle+distance)로 증가한다 [Cui & Dai, TWC 2022]. Polar-domain OMP, Bayesian sparse recovery, DL 기반 refiner 등이 제안되었으나, 이 방법들은 모두 연산 비용이 far-field CE 대비 크게 증가한다.

한편 5G massive MIMO에서도 DL 기반 CE(CNN, Transformer, Diffusion)의 도입으로 CE inference의 연산 비용이 전통적 LS/MMSE 대비 수십~수백 배 증가하는 추세다.

### 1.2 배경: CE의 물리적 위치와 아키텍처 제약

O-RAN 7.2x functional split 기준, CE는 **O-DU의 L1 High (Upper PHY)** 에서 수행된다. O-RU(Radio Unit)는 Lower PHY(FFT/iFFT, CP 제거, beamforming weight 적용)만 담당하고, frequency-domain IQ sample을 eCPRI fronthaul을 통해 O-DU로 전송한다. O-DU가 이를 받아 DMRS 추출 → CE → equalization → demapping → LDPC decoding까지 처리한다.

물리적으로 O-DU는 셀사이트 근처의 엣지 데이터센터(국사 또는 엣지 서버룸)에 위치한다:

```
철탑/옥상 → 안테나 + O-RU (RF 처리, FFT)
  ↓ eCPRI fronthaul (광케이블, 수~20 km)
근처 국사/엣지 → O-DU 서버 ← CE 수행 위치
  ↓ midhaul
중앙 DC → O-CU, 5GC
```

이 구조의 핵심 제약은 **fronthaul 왕복 지연**이다. 업링크에서 O-RU → O-DU로 IQ sample이 올라가고, O-DU에서 CE + decoding 후 HARQ ACK/NACK을 UE에 보내야 하는데, 5G NR의 HARQ 타이밍 budget은 SCS 30kHz 기준 수 ms로 매우 빡빡하다. 현실적으로 O-RU~O-DU 간 전파 지연을 100~250μs 이내로 제한해야 하며, 64T64R MIMO 기준 셀당 수십 Gbps의 fronthaul 대역폭이 필요하다. SCS가 올라갈수록(60/120kHz, mmWave) 슬롯 길이가 짧아져 타이밍이 더 빡빡해진다.

이 fronthaul 지연 부담이 CE scheduling 최적화의 직접적 동기 중 하나다: **매 슬롯마다 full CE inference를 O-DU에서 수행하는 것은 연산 비용뿐 아니라, 이미 빡빡한 L1 처리 파이프라인의 latency budget을 더 압박한다.**

### 1.3 배경: CE가 소프트웨어로 전환되었다

전통적 기지국에서 CE는 ASIC 고정 파이프라인의 일부로, FFT 출력 후 자동 실행되는 하드웨어 블록이었다. Skip 분기가 파이프라인에 존재하지 않았으므로, "CE를 수행할지 말지"는 결정 가능한 변수가 아니었다.

NVIDIA Aerial cuPHY는 CE를 포함한 모든 L1 기능을 GPU CUDA 커널로 구현한다 [NVIDIA, 2025]. NTT DOCOMO는 이를 상용 배포하여 50% 기지국 전력 절감을 보고했다. O-RAN dApp 아키텍처는 DU 내부에서 PHY 데이터에 직접 접근하며 sub-ms 제어 루프를 구현한다 [Lacava et al., Computer Networks 2025]. 기존 xApp은 Near-RT RIC에서 E2 인터페이스를 통해 집계된 KPI만 접근 가능했으나, dApp은 L1 레벨 IQ sample이나 channel estimate에 직접 접근할 수 있다. CE는 더 이상 ASIC의 고정 공정이 아니라 GPU에서 스케줄링 가능한 소프트웨어 태스크가 되었다.

### 1.4 아무도 안 한 질문

> 채널이 충분히 유사한 연속 슬롯에서, 매번 full CE inference를 수행해야 하는가?

### 1.5 이 질문이 정의될 수 없었던 구조적 이유

1. **하드웨어 고정**: ASIC 기반 수신기에서 CE는 FFT→DMRS추출→LS/MMSE가 하드웨어 파이프라인으로 연결되어 있으며, skip 분기가 물리적으로 존재하지 않았다.
2. **표준의 암묵적 가정**: 3GPP TS 38.211은 CE 알고리즘을 규정하지 않지만, antenna port 정의에서 "the channel over which a PDSCH symbol on one antenna port is conveyed can be inferred from the channel over which a DM-RS symbol on the same antenna port is conveyed only if the two symbols are within the same slot"이라고 명시한다. Coherent demodulation의 전제로 per-slot CE를 암묵적으로 요구한다.
3. **평가 프레임의 폐쇄성**: CE 논문의 성능 지표가 per-slot NMSE로 고정되어, "CE를 skip한 슬롯"의 평가 방법 자체가 프레임 안에 존재하지 않았다. 모든 pilot overhead 연구(compressed sensing, superimposed pilot, pilot selection 등)는 "한 번 CE를 수행할 때 파일럿을 줄이는" 문제만 다루며, "CE 수행 빈도를 줄이는" 문제를 다루지 않았다.
4. **커뮤니티 분리**: "CE를 언제 할지"는 본질적으로 자원 스케줄링 문제이나, CE는 Signal Processing 커뮤니티가, 스케줄링은 Networking 커뮤니티가 각각 다룬다. 교차 영역 문제를 다룰 주체가 없었다.

### 1.6 이제 가능한 이유

GPU-native RAN에서 CE는 CUDA 커널이다. 실행 여부를 런타임에 결정할 수 있다. 핵심 구분: **DMRS 수신(3GPP 의무)과 full CE inference(구현 재량)의 분리(decoupling)**. DMRS는 매 슬롯 수신되지만, 수신된 DMRS로부터 full channel reconstruction을 수행하는 것은 표준이 아닌 구현자의 결정이다. GPU에서 CE 커널의 실행 여부를 dApp이 제어할 수 있다.

### 1.7 기존 연구와의 관계

- **ICENet [IEEE 2025]**: CE inference의 depth (반복 횟수)를 적응적으로 조절. 하지만 매 슬롯 반드시 inference를 수행. 우리는 inference **frequency** (수행 여부)를 조절. 직교하는 차원.
- **Temporal prediction (CsiNet-LSTM, CNN-GPT2)**: "다음 슬롯의 채널을 예측"하되 매 슬롯 prediction을 수행. 우리는 "예측이 필요 없을 만큼 채널이 안 변했으면 아예 skip". Prediction의 상위 레이어.
- **Elbir FL-based NF CE [arXiv:2302.04802]**: CE 알고리즘의 학습을 분산화. 우리는 CE 수행 자체의 스케줄링을 최적화. CE 알고리즘에 agnostic — Elbir의 방법 위에도 우리의 scheduling을 적용 가능.

---

## 2. Core Idea

### 2.1 DMRS 수신 ≠ CE Inference

파일럿 수신과 full CE를 분리한다.

- **LS estimate** (매 슬롯 항상 수행): DMRS 위치에서 Y/X. O(N_pilot) 곱셈. 비용 극히 낮음.
- **Full CE inference** (조건부 수행): LS를 입력으로 LMMSE/DL/CS 등을 돌려 전체 채널 복원. 비용 높음.

Skip 대상은 full CE inference이지, 파일럿 수신이 아니다.

### 2.2 Proportional-Alpha Scheduling (확정)

```
매 슬롯 t:
  (1) h_LS(t) ← Y_pilot(t) / X_pilot          [항상 수행, overhead ≈ 0]
  (2) δ(t) ← ‖h_LS(t) − h_LS(t−1)‖_F / ‖h_LS(t−1)‖_F   [Frobenius norm monitor]
  (3) τ = c · √(2/SNR)                         [SNR-adaptive threshold, c는 설계 파라미터]
  (4) α(t) = min(δ(t) / τ, 1)                  [Proportional blending weight]
  (5) ĥ(t) = α(t)·h_LS(t) + (1-α(t))·ĥ(t−1)  [EMA update]
  (6) Safety: 연속 N_max 슬롯 non-full이면 강제 α=1
```

**설계 선택 근거 (2026.03.24 실험 결과 기반):**

- **δ metric**: Frobenius norm (전체 antenna×subcarrier의 L2). Equalization 후 SNR degradation과 직결 (Post-EQ SNR ≈ SNR/(1+SNR·δ²)).
- **τ = c·√(2/SNR)**: LS noise floor = √(2/SNR) ≈ 0.141 @20dB. τ < NF면 LS monitor가 변화를 감지 못해 scheduling 무의미. c=1이 자연스러운 operating point.
- **α = δ/τ (proportional)**: Binary (skip/full) 대비 UE4(118-path ped)에서 **19dB 개선** (-1.9→-20.7dB). 작은 δ에서 α가 작아져 temporal noise averaging 효과 (static UE에서 Full CE보다 10dB 좋음).
- **Oracle upper bound**: greedy per-slot optimal α를 fitting → α = sigmoid(63·(δ-0.054)). Static UE에서 proposed δ/τ 대비 최대 ~8dB 더 좋음 (sharp transition으로 noise averaging 극대화).

**비교한 4가지 전략:**

| Strategy | 수식 | 역할 |
|---|---|---|
| Full CE | α=1 (매 slot) | Baseline (-20dB) |
| Binary | α=0 if δ<τ, else 1 | Naive. UE4에서 catastrophic failure |
| **δ/τ (proposed)** | α=min(δ/τ, 1) | Main proposal. Tunable, practical |
| Oracle fit (sigmoid) | α=1/(1+exp(-63·(δ-0.054))) | Upper bound (h_real 기반 fit, not deployable) |

**Controlled experiment (UE3/UE4 static):**
- UE4를 고정하면 δ: 0.057 → 0.009 (6.4× 감소) → skip 가능 수준
- **Skip 가능 여부 = f(mobility × path diversity)**. 둘 다 있을 때만 skip 불가.

### 2.3 CE-Agnostic Framework

이 scheduling wrapper는 Full CE가 무엇이든 동작한다. 논문에서는 3종의 CE로 검증:

| CE | 역할 | 비용 | 비고 |
|----|------|------|------|
| LS | Lower bound | ~0.01ms | Skip 이득 작음 → "이것도 되나?" |
| Genie-LMMSE | Oracle bound | ~0.5ms | 완벽한 통계 정보 가정 → 이론적 한계 |
| DL-CE (ResNet) | Practical | ~2ms | 현실적 DL CE → 실용적 이득 |

**Polar-OMP 생략 이유**: 구현 2주 + dictionary 메모리 문제(1024 ant × 65K atoms). 우리 contribution은 CE 알고리즘이 아니라 scheduling. CE가 비쌀수록 skip 이득이 크다는 것을 3종 비교로 충분히 보여줌.

---

## 3. Hypotheses

### H1: Temporal Persistence (전제 조건 — go/no-go)

> NF ELAA 채널의 연속 스냅샷 간 normalized LS difference δ(t)는 저속/정적 환경에서 대부분의 시간 동안 threshold 이하이다.

이것이 성립하지 않으면 전체 방향 폐기.

### H2: CE-Agnostic Effectiveness

> Skip scheduling은 CE 알고리즘(LS, LMMSE, DL-CE)에 무관하게 일관된 computation-rate tradeoff를 제공한다. CE가 비쌀수록 skip의 절대적 이득이 크다.

### H3: CE Cost Amplifies Skip Benefit (reframed from NF-specific → computational)

> CE 알고리즘의 연산 비용이 클수록 skip의 절대적 이득이 크다. 6G ELAA에서 CE 비용이 가장 높으므로, scheduling framework의 가치가 극대화된다.

이전 H3 (distance-dependent τ*)는 NF zone에 충분한 UE가 필요하나, 현실적 시나리오(dist_min=10m)에서 R_Rayleigh < 10m이면 NF UE가 없음. CE 비용 기반 argument는 NF 전파 효과 없이도 성립하며, 더 일반적.

### H4: Beamforming Robustness

> Stale channel (skip한 슬롯)로 beamforming하더라도, rate loss가 full CE 대비 5% 이내인 operating point가 존재한다.

---

## 4. Experimental Design

### 4.0 환경

```
Hardware:  NVIDIA A100 (CC 8.0, 80GB)
Software:  Sionna RT 1.x, PyTorch 2.x
Scene:     Munich UMi, 8 BS

[Primary configs — 논문 본문 Figure/Table]
  Config A: 15 GHz FR3, 512 ant (24×24), 1024 SC, 400 MHz BW   (R_Rayleigh ≈ 10.2m, ELAA)
  Config B: 28 GHz FR2, 512 ant (24×24), 1024 SC, 400 MHz BW   (R_Rayleigh ≈ 5.5m,  ELAA)
  Config C: 3.5 GHz FR1, 64 ant (8×8), 256 SC, 100 MHz BW      (R_Rayleigh ≈ 2.7m,  5G mMIMO FF baseline)

[Extended configs — generality 검증, appendix 또는 추가 분석]
  18 ELAA presets: {s,m,l} × {1k,2k,4k} × {15g,28g}
    - 안테나 수 (s=8×8, m=16×16, l=32×32) → Rayleigh distance 변화
    - 대역폭 (1k/2k/4k SC) → CE 비용 스케일링
    - 주파수 (15/28 GHz) → coherence time 차이
  1 5G preset: munich_5g_mimo_3g5

  모든 preset에 대해 independent + temporal 데이터 생성됨 (run_all_datagen.sh).
  Primary configs 외 preset은 일반성 주장 강화에 활용.

UE distance: 10m – 150m
BS split: train {0,5} / val {2,4} / test {1,3,6,7}
Mobility: 0 m/s (static), 1 m/s (pedestrian), 8.3 m/s (30km/h low vehicle)
  — 33 m/s (120km/h) 제외: NF ELAA 주요 use case에 해당하지 않음
```

### 4.1 데이터 생성 (Phase 0)

기존 independent drops: DL-CE 학습용으로 유지.
신규 temporal trajectory: CE scheduling 평가용.

```python
# generate.py 수정 핵심
# 초기 위치: 한 번 샘플링
if snapshot_idx == 0:
    ue_pos_init = sample_positions(num_ue, dist_min, dist_max)
    ue_direction = random_unit_vectors(num_ue)

# 매 snapshot: 이동
ue_pos[t] = ue_pos_init + velocity * dt * t * ue_direction
# 경계 처리: 씬 밖으로 나가면 방향 반전
```

Mobility 세트:

| 시나리오 | 속도 | dt | 용도 |
|----------|------|-----|------|
| Static | 0 m/s | 10ms | FWA, 공장 센서 — skip 최대 |
| Pedestrian | 1 m/s | 10ms | 실내, 캠퍼스 |
| Low vehicle | 8.3 m/s (30km/h) | 10ms | 도심 |
| High vehicle | 33 m/s (120km/h) | 10ms | 고속도로 — skip 최소/불가 예상 |

- **Pilot data** (go/no-go용): 20 UE × 200 snapshots × 3 mobility × primary configs = 9세트
- **Full data** (논문용): 100 UE × 1000 snapshots × 3 mobility × 19 presets (run_all_datagen.sh로 일괄 생성)

### 4.2 Exp 1 — Temporal Persistence Profiling (H1, go/no-go)

```python
for config in [A, B, C]:
    for mobility in [static, ped, low_veh, high_veh]:
        for ue in all_ues:
            for t in range(1, T):
                delta[t] = norm(h_true[t] - h_true[t-1]) / norm(h_true[t-1])

        plot_cdf(delta)                    # 전체 CDF
        plot_cdf_by_distance(delta, dist)  # 거리 구간별 CDF
        plot_delta_vs_distance(delta, dist) # scatter
```

**Kill criterion**: pedestrian에서 delta median > 0.5 → 방향 재검토.

### 4.3 Exp 2 — CE 구현 + A100 프로파일링

3개 CE 구현 후 비용 측정:

```python
import torch.cuda

for ce_method in [ls, genie_lmmse, dl_ce]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    h_hat = ce_method(h_ls)
    end.record()
    torch.cuda.synchronize()

    time_ms = start.elapsed_time(end)

    # + NMSE 측정
    nmse = (norm(h_hat - h_true)**2 / norm(h_true)**2).mean()
```

측정: wall-clock time (ms), NMSE, GPU memory.

**DL-CE 학습**: independent drops 데이터 사용.
- Input: h_ls (SNR별 noise 추가)
- Target: h_true
- Loss: MSE, 학습 epochs: ~100

### 4.4 Exp 3 — Threshold Sweep + Pareto Front (H2)

```python
for ce_method in [ls, lmmse, dl_ce]:
    for config in [A, B, C]:
        for tau in np.linspace(0.01, 1.0, 100):
            h_hat, stats = adaptive_ce_scheduling(
                H_true, snr, tau, ce_method
            )

            skip_rate = stats['skip'] / T
            nmse = compute_nmse(h_hat, H_true)
            rate = compute_achievable_rate(h_hat, H_true, snr)
            cost = stats['full'] * t_full + stats['delta'] * t_delta + stats['skip'] * t_monitor

            record(config, ce_method, tau, skip_rate, nmse, rate, cost)
```

핵심 Figure: X=Computation Ratio, Y=Rate Preservation Ratio. 3 CE × 3 config = 9 curves. Knee point가 recommended τ.

### 4.5 Exp 4 — Distance-Dependent Channel Variation Analysis

UE-BS 거리에 따라 channel variation(δ)이 다른지 분석. NF zone에 UE가 부족하여 NF-specific claim은 제한적이나, 거리 구간별 variation 차이 자체는 scheduling threshold 설계에 유용한 정보.

> **Note (2026.03.19 reframe)**: 원래 H3는 "NF에서 τ*가 거리 의존적"이었으나, R_Rayleigh < dist_min(10m)으로 NF zone이 비어있어 검증 불가. "CE 비용이 클수록 skip 이득이 크다"(H3 reframed, S2에서 검증)로 대체.

### 4.6 Exp 5 — Delta Update Ablation

Skip(reuse) vs Delta update 3종 비교:

```
(a) Pure skip:  ĥ(t) = ĥ(t-1)
(b) EMA:        ĥ(t) = α·h_LS(t) + (1-α)·ĥ(t-1)
(c) LS-delta:   ĥ(t) = ĥ(t-1) + β·(h_LS(t) - h_LS(t-1))
```

### 4.7 Exp 6 — Beamforming Rate Impact (H4)

```python
# MRT beamformer with estimated channel
w = h_hat / norm(h_hat)
rate = log2(1 + snr * abs(w.conj() @ h_true)**2)
rate_oracle = log2(1 + snr * norm(h_true)**2)  # perfect CSI
rate_loss = 1 - rate / rate_oracle
```

CDF of rate_loss. Target: P(rate\_loss > 5%) < 10%.

### 4.8 Exp 7 — Multi-BS Generalization

Train BS {0,5}에서 학습한 τ\* → Test BS {1,3,6,7}에서 성능 측정.
Generalization gap 분석.

---

## 5. Metrics

### 5.1 기존 metric (baseline 보고용)

- **NMSE** = E[‖ĥ−h‖²] / E[‖h‖²] — per-slot, CE 커뮤니티 표준

### 5.2 새로운 metrics (이 논문의 프레임워크)

| Metric | 정의 | 의미 |
|--------|------|------|
| Skip Rate (SR) | N\_skip / N\_total | CE inference 절감률 |
| Rate Preservation Ratio (RPR) | R\_adaptive / R\_fullCE | Throughput 보존율 (1에 가까울수록 좋음) |
| Computation Ratio (CR) | Cost\_adaptive / Cost\_fullCE | 연산 비용 비율 (낮을수록 좋음) |
| Efficiency Score (ES) | RPR / CR | 단위 연산당 throughput (높을수록 좋음) |
| Skip Miss Rate (SMR) | P(skip 했는데 rate\_loss > 10%) | 안전성 지표 |

여기서 Cost는 각 CE의 A100 측정 wall-clock time 기반:

```
Cost_adaptive = N_full × t_full + N_delta × t_delta + N_skip × t_monitor
Cost_fullCE   = N_total × t_full
CR = Cost_adaptive / Cost_fullCE
```

### 5.3 핵심 Figure 목록

| Fig # | 내용 | 검증 대상 |
|-------|------|-----------|
| 1 | System model diagram (dApp + GPU L1 + 2-mode) | — |
| 2 | Temporal delta CDF (mobility별, config별) | H1 |
| 3 | Pareto front: CR vs NMSE (3 CE × 3 config) | H2 핵심 |
| 4 | Component profiling: stacked bar (monitor/LS/EMA/fullCE) | S7 |
| 5 | Delta update ablation (skip vs EMA vs LS-delta) | S4 |
| 6 | RPR heatmap across (SNR, τ) | H4 |
| 7 | Multi-BS generalization gap | S6 |
| 8 | Effective throughput: per-UE loss vs system capacity gain | S7 |

### 5.4 핵심 Table 목록

| Table # | 내용 |
|---------|------|
| 1 | CE 알고리즘별 A100 프로파일링 (time, memory, NMSE) — S7 |
| 2 | Sweet spot 요약: 각 (config, CE)에서의 (SR, CR, NMSE, RPR) — S2+S5 |
| 3 | System overhead breakdown (monitor/LS/EMA vs full CE) — S7 |
| 4 | Multi-BS generalization gap — S6 |

---

## 6. Expected Contributions

| # | Contribution | 유형 | 검증 |
|---|-------------|------|------|
| C1 | CE inference scheduling 문제를 최초 정의. Pilot reception과 CE inference의 decoupling을 형식화 | Problem formulation | Intro + System model |
| C2 | 2-Mode adaptive scheduling (Blend with continuous alpha / Full) 제안 | Algorithm | S2, S4 |
| C3 | CE-agnostic: LS/Genie-LMMSE/DL-CE 3종에서 일관된 효과 실증 | Generality | S2 (Pareto front) |
| C4 | CE 비용이 클수록 skip 이득이 크다 — ELAA에서 가치 극대화 | Computational analysis | S2 (CR: LS 0.57 vs DL-CE 0.10) |
| C5 | Throughput-computation Pareto front + effective throughput metric 제안 | Evaluation framework | S2, S5, S7 |
| C6 | Multi-BS generalization: train BS의 τ*가 unseen BS로 transfer | Practical deployment | S6 (test gap <2dB) |

**Reframe (2026.03.19)**: NF는 contribution이 아닌 motivation으로 재위치. "6G ELAA에서 CE 비용 폭발 → scheduling 필요성" (intro context). Distance-dependent τ*는 현재 실험에서 유의미한 NF/FF 차이 미관측 (R_Rayleigh < dist_min). CE 비용 기반 argument (C4)로 대체.

---

## 7. 리뷰어 예상 공격 및 대응

**Q: "CE skip은 그냥 channel prediction의 특수한 경우 아닌가?"**
→ Channel prediction은 매 슬롯 prediction을 수행하므로 연산량 절감이 없다. 우리는 "prediction이 필요 없는 슬롯을 식별하여 연산 자체를 skip"한다. Prediction 위에 올라가는 meta-decision.

**Q: "고속 이동에서는 안 되잖아?"**
→ 맞다. Limitation으로 명시. 하지만 6G NF ELAA의 주요 use case(FWA, indoor, factory)는 저속/정적. 고속 환경에서는 τ를 낮추면 full CE에 수렴하므로 성능 저하 없이 graceful degradation.

**Q: "LS difference로 trigger하면 noise에 취약하지 않나?"**
→ 맞다. 저SNR에서 delta가 noise-dominated되어 false trigger 발생 가능. EMA smoothing으로 완화. 이것이 Exp 5(delta update ablation)에서 검증됨.

**Q: "Polar-OMP 같은 제대로 된 NF CE와 비교해야 하지 않나?"**
→ 우리 contribution은 CE 알고리즘이 아닌 scheduling framework. LS(최소), LMMSE(최적 선형), DL-CE(비선형) 3종에서 일관된 효과를 보이므로 CE-agnostic함이 증명됨. Polar-OMP도 이 위에 올릴 수 있으며 이는 future work.

**Q: "실제 시스템에서의 latency 절감은?"**
→ 본 논문은 oracle channel 기반 scheduling의 이론적 이득을 분석. 실시간 시스템 구현(cuPHY/dApp integration)은 future work. 단, A100에서의 CE kernel wall-clock time을 보고하여 실제 절감량의 추정치를 제공.

---

## 8. Preliminary Experimental Findings (2026.03.20)

> 아래는 munich_elaa_m_1k_15g (24×24, 15GHz) 기준 초기 실험 결과.
> 9 UE (3 static + 3 ped + 3 veh), 4000 snapshots, dt=0.25ms, BS 1개.

### 8.1 채널 변화의 두 가지 regime (S0)

채널 변화 δ_oracle = ‖h(t) - h(t-1)‖ / ‖h(t-1)‖ 분석 결과, 채널 변화가 **continuous drift + discrete spike** 두 종류로 구분됨:

**Continuous drift** (baseline δ):
| Mobility | Median δ_oracle | LS noise floor (20dB) | LS로 감지? |
|----------|----------------|----------------------|-----------|
| Static   | 0.007          | 0.14                 | **불가** (δ << noise floor) |
| Ped 1m/s | 0.000~0.043    | 0.14                 | **일부만** (UE에 따라 다름) |
| Veh 8.3m/s | 0.26~0.32   | 0.14                 | **가능** (δ >> noise floor) |

**Discrete spike** (path event):
- 모든 UE에서 동시 발생 (t=143, 286, 429... 간격 ~143 slots = 35.75ms)
- δ > 1.0 (baseline 대비 100x 이상)
- 원인: vehicle UE 이동 → scene scatterer 변화 → PathSolver가 path set 재계산 → CFR interference pattern 급변
- CIR energy는 10% 내외 변화이나, phase가 전 antenna에 걸쳐 무작위 재배열 (abs_mean phase diff ≈ π/2)
- **물리적으로 정당**: 현실에서도 scatterer 이동에 의한 급격한 채널 변화 발생

**핵심 insight**: CE scheduling은 drift와 spike를 **다른 메커니즘**으로 처리해야.
- Drift → Doppler/coherence time 기반 예측 가능
- Spike → 예측 불가, reactive monitoring 필요 (LS δ가 유효)

### 8.2 LS Monitor의 Noise Floor 한계 (S0)

SNR 20dB에서 LS δ의 noise floor ≈ √(2/SNR_linear) ≈ 0.14.
- Static/ped UE의 oracle δ (0.00~0.04)가 이 아래 → **LS monitor로는 진짜 채널 변화를 감지 불가**
- 하지만 spike (δ > 1.0)은 noise floor보다 훨씬 크므로 **LS monitor로 감지 가능**
- 이건 LS monitor의 "noise-triggered refresh"가 우연히 channel staleness를 방지하는 아이러니한 효과

### 8.3 Monitor 전략 비교 (S1)

7가지 scheduling strategy를 τ=0.2에서 비교:

| Strategy | Static NMSE | Ped NMSE | Veh NMSE | 평가 |
|----------|------------|----------|----------|------|
| Always Full CE | -20.7dB | -20.7dB | -20.7dB | baseline |
| Always Skip | +57~+85dB | +46~+85dB | +69~+84dB | 실패 (spike 누적) |
| **LS monitor** | **-20.6dB** | -3.4~-20.7dB | **-12~-14dB** | **best practical** |
| Oracle monitor | -20.6dB | -3.4~-20.7dB | -7~-12dB | oracle도 veh에서 LS보다 나쁨 |
| Doppler | +40~+67dB | +39~+77dB | +48~+62dB | **spike 감지 불가 → 실패** |
| SNR-adaptive τ | -20.6dB | -3.4~-20.7dB | -3~-6dB | τ가 너무 높아짐 |
| Smoothed LS (K=4) | +30~+56dB | +19~+57dB | +43~+56dB | **spike smoothing → 실패** |

**핵심 결론**:
1. Doppler-only, smoothed LS → spike 감지 불가로 catastrophic failure
2. LS monitor → spike 감지 가능 (δ_spike >> noise floor), static/ped에서 noise-triggered refresh 효과
3. Oracle monitor가 veh에서 LS보다 나쁜 이유: oracle이 더 많이 skip(40%) → drift 누적. LS의 noise trigger가 더 자주 refresh(SR=15~22%)
4. **LS monitor가 practical best** — but UE4 (ped, δ=0.04)에서 -3.4dB로 실패 (noise floor 문제)

### 8.4 Spike의 물리적 특성

```
UE0 (static), Spike at t=143:
  t=142: ||h|| = 4.608e-02 (stable)
  t=143: ||h|| = 3.916e-02 (15% drop, δ=1.31)  ← spike
  t=144: ||h|| = 3.916e-02 (새 값으로 안정)

  CIR: 115 paths → 115 paths (경로 수 불변)
       energy 0.283 → 0.253 (10% 감소)
  Phase diff: abs_mean = π/2 (전 antenna에 걸쳐 위상 무작위 재배열)
  → MAGNITUDE dominated (UE0) 또는 PHASE dominated (UE1, UE3)
```

- Spike 간격 ≈ 143 slots (35.75ms) — vehicle UE가 0.30m 이동 (15파장)
- Spike 이후 채널이 새 상태로 "정착" (1-shot transition, not gradual)
- **이건 channel prediction으로 대응 불가 — reactive scheduling만 유효**

### 8.5 방향성 수정

**기존 (paper_plan 초기)**:
- CE skip으로 X% 연산 절감
- 2-mode (Blend/Full) continuous-alpha scheduling

**수정 (실험 결과 기반)**:
- 채널 변화는 drift + spike 두 regime
- LS monitor는 spike 감지에 유효하지만, drift 감지에는 noise-limited
- **Hybrid scheduling**: Doppler로 drift 기반 skip interval 결정 + LS로 spike 감지
- 또는: **"언제 CE를 해야 하는가?"** 로 reframing — CE skip이 아니라 CE timing optimization
- Spike frequency/magnitude 분석이 새로운 contribution 후보

### 8.6 남은 검증

- [ ] 28GHz, 3.5GHz config에서 동일 분석 (spike 간격 변화?)
- [ ] SNR 30dB에서 LS noise floor 감소 → ped UE 감지 개선?
- [ ] Hybrid monitor (Doppler + LS spike trigger) 구현 + 비교
- [ ] Spike frequency의 물리적 모델링 (scatterer density, velocity 의존성)
- [ ] Beamforming rate impact (S5) — spike 고려한 RPR

---

## 9. Timeline

```
Week 1:  generate.py 수정 + pilot data 생성 (20 UE, 200 snap, 4 mobility, 3 config)
         Exp 1: temporal persistence → go/no-go
         5G config (8×8, 3.5GHz) 추가

Week 2:  CE 구현 (LS: 0.5일, LMMSE: 1일, DL-CE: 3일)
         DL-CE 학습 (independent drops 데이터)
         Exp 2: A100 프로파일링

Week 3:  Full data 생성 (100 UE, 1000 snap, 선택된 세트)
         Exp 3: threshold sweep + Pareto front
         Exp 5: delta update ablation

Week 4:  Exp 4: distance-dependent threshold
         Exp 6: beamforming rate impact
         Exp 7: multi-BS generalization

Week 5:  논문 작성
         Fig/Table 생성
```

---

## 9. One-Sentence Summary

We decouple mandatory pilot reception from optional CE inference, and propose a two-mode continuous-alpha scheduling framework that is CE-algorithm-agnostic: in Blend mode, channel estimates are updated via EMA with a weight that varies continuously with observed channel variation; in Full mode, triggered when variation exceeds a threshold, the complete CE algorithm is launched. Validated across 5G mMIMO and 6G ELAA configurations with LS, LMMSE, and DL-based CE, it achieves up to 90% computation reduction with less than 3% throughput loss, where the benefit scales with CE complexity — making it most valuable for computationally expensive ELAA systems.
