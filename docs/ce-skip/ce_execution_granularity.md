# CE Execution Granularity Analysis

> CE-skip 논문에서 인용하는 DL-CE 논문들의 실제 CE 실행 단위 분석.
> "Per-slot CE가 default"라는 우리 주장의 정확한 scope를 정의하기 위함.

## 핵심 발견

**대부분의 DL-CE 논문은 per-slot CE가 아님.** 자체 정의한 pilot structure에서 CE를 수행.
우리 논문의 "per-slot" 전제는 3GPP NR DMRS 구조에 한정되며, DL-CE 논문 일반에 대한 주장이 아님.

## 논문별 CE 실행 단위

| 논문 | CE 단위 | Pilot 구조 | NR 표준 준수? | 비고 |
|------|---------|-----------|-------------|------|
| **PACENet (Yang 2025)** | Per pilot block | T=16 시간축 orthogonal pilot 전송 후 한 번 CE. **OFDM 아님, flat fading** (H ∈ ℂ^(Nr×Nt), 주파수 차원 없음) | 아니오 (자체 정의) | Kronecker 채널 모델, narrowband |
| **Channelformer (Luan 2023)** | Per frame | Frame 내 symbol 1,13 (또는 1,5,9,13)에 pilot | 아니오 (자체 정의) | SISO OFDM, 주파수+시간 보간 |
| **ReQuestNet (Pratik 2025)** | Per slot | 14 OFDM symbol slot, DMRS 위치 3GPP 준수 | **예** | T=4 iterative refinement, 가장 NR에 가까움 |
| **NVIDIA NRX (Cammerer 2024)** | Per slot | 5G NR DMRS 그대로 | **예** | 실제 NR 파라미터, A100 실측 |
| **XLCNet (Chen 2024)** | Per snapshot | Single snapshot, static | 아니오 | Narrowband, 시간 개념 없음 |
| **Wideband Unrolling (2025)** | Per snapshot | Single snapshot, static | 아니오 | NF ELAA, 시간 개념 없음 |

## 해석

### "Per-slot CE가 default"가 성립하는 조건
- **NR 표준 구현** (NVIDIA NRX, ReQuestNet). DMRS가 매 scheduled slot에 매핑되므로 CE도 per-slot.
- **상용 기지국**. 3GPP 표준을 따르므로 per-slot DMRS → per-slot CE.

### "Per-slot"이 아닌 경우
- **학술 DL-CE 논문** (PACENet, Channelformer, XLCNet). 자체 pilot 구조를 정의하며 NR slot 개념이 없음.
- **이 논문들에서 CE 실행 간격은 설계 변수**이지 표준에 의해 결정되는 것이 아님.

### CE-skip 논문에 대한 시사점

1. **Intro의 "per-slot CE has been an unquestioned default"는 NR 표준 맥락에서만 정확.**
   학술 DL-CE 논문 일반에 대해서는 과장.

2. **더 정확한 framing.** "CE is conventionally executed at every pilot observation interval,
   whether per-slot in NR or per-frame/per-block in academic models.
   No prior work treats this interval as an adaptive runtime decision."

3. **CE-skip의 contribution은 CE 실행 단위에 무관하게 유효.**
   Per-slot이든 per-frame이든, "이번에 CE를 돌릴까?"는 여전히 유효한 질문.
   단, per-slot에서 CE cost가 더 자주 발생하므로 skip의 절대적 이득이 더 큼.

## Related Works 차별화 문장 (권장)

```latex
DL-based methods improve CE accuracy for various pilot configurations,
from per-frame OFDM~\cite{luan2022} to per-slot NR~\cite{nvidia_nrx},
but all assume CE is executed at every pilot observation without
questioning whether execution is necessary.
```

---
*Created: 2026-03-25 | CE-skip intro/relworks 정확성 검증 과정에서 도출*
