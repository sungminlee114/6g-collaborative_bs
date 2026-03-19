# CE Skip Experiments: Known Issues & Fix Plan

## CRITICAL: Static δ ≠ 0 (Simulator Noise)

**발견일**: 2026-03-19
**증상**: Static UE (speed=0, position 불변)에서 median δ = 0.058~0.49
**원인**: `generate_worker.py:158`에서 `seed = snap_id * 17 + 41` → 매 snapshot마다 다른 seed → Sionna RT PathSolver의 diffuse scattering이 stochastic → 같은 위치라도 다른 path set

**영향**: δ metric이 physical channel variation 외에 simulator noise를 포함. S0의 δ 분포, S2-S6의 scheduling 판단 모두 이 noise에 영향받음.

**수정 방법** (데이터 재생성 필요):
```python
# generate_worker.py:158
# Before:
seed = snap_id * 17 + 41

# After (temporal mode):
seed = 42  # 고정 seed → deterministic paths → static δ = 0
# 또는
diffuse_reflection=False  # stochastic component 제거
```

**우선순위**: 데이터 재생성이 수 시간 소요. 현재 실험 결과는 "simulator noise 포함" disclaimer 필요. 최종 논문 제출 전 재생성 필수.

---

## HIGH: Speed-Position Confounding (S0)

**증상**: ELAA 15GHz에서 v8.3(30km/h)의 median δ가 pedestrian(1m/s)보다 작음
**원인**: 속도별로 다른 UE가 다른 위치에 배치 → 속도 효과와 위치 효과가 분리 불가
**수정**: 같은 initial position에서 속도만 바꾸는 controlled experiment 필요
**임시 대응**: S0에서 속도별 분리 분석 대신 overall feasibility CDF만 제시

---

## HIGH: Genie-LMMSE Underfitted (5G MIMO)

**증상**: 5G MIMO에서 LMMSE(-5.3dB) > LS(-20.2dB) — LMMSE가 15dB 나쁨
**원인**: N=44 train samples, D=256 dimensions → rank-deficient covariance
**수정**: snapshot 수 늘리기 (3→20+) 또는 regularization 강화

---

## MEDIUM: S0 δ vs Quality Gap

**증상**: S0은 δ 분포만 보고 go/no-go 판정. δ → NMSE/rate 영향 미분석
**수정**: S0에 δ_true (ground truth 변화) + δ_LS (noisy 변화) + NMSE degradation 관계 추가

---

## HIGH: dt=10ms is 10x Too Large

**발견일**: 2026-03-19
**증상**: dt=10ms로 시뮬레이션, 하지만 3GPP NR 1 slot = 1ms (15kHz SCS) / 0.5ms (30kHz)
**영향**: dt=10ms에서의 δ는 10-slot 간격의 변화. 실제 per-slot δ는 이보다 ~10배 작을 것 → skip 이득을 과소평가하고 있을 가능성
**수정**: config별 적절한 dt 사용
  - 5G 3.5GHz (SCS 15kHz): dt = 1ms
  - 6G 15GHz (SCS 30kHz): dt = 0.5ms
  - 6G 28GHz (SCS 120kHz): dt = 0.125ms
**주의**: dt 축소 → 같은 duration에 10~80배 snapshot 필요 → 생성 시간 증가

---

## LOW: NF Zone Unpopulated (S3)

**증상**: R_Rayleigh < dist_min(10m) → NF zone에 UE 없음
**상태**: Reframe 완료. NF는 motivation, contribution은 CE cost scaling
**Future work**: dist_min < R_Rayleigh인 시나리오 또는 elaa_l (32×32) 사용
