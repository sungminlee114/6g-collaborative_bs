# Task-Agnostic Site Representation for 6G Channel Estimation

## 1. Core Research Idea

### Concept
3-way model decomposition for wireless channel estimation across multiple BS sites:
- **E (Shared Encoder)**: Learns universal wireless physics features, shared via FL/transfer learning
- **theta_task (Shared Task Head)**: Task-specific decoder, shared across sites
- **theta_BS (Site Embedding)**: ~64-dim learnable vector, zero-initialized (like LoRA), stays local per BS

### Key Insight (Killer Contribution)
theta_BS trained on Task A (channel estimation) should be **task-agnostic** and transfer to Task B (beam prediction).
If this works -> "site foundation representation" -- much stronger than just cold start improvement.

### Framing: Transfer/Meta-Learning (NOT FL during operation)
- **Phase 1 (Pre-train)**: Train E + theta_task on simulated BS_1~6 (Sionna RT digital twin)
- **Phase 2 (Deploy)**: Deploy to unseen BS_7, only adapt theta_BS (few-shot)
- **Motivation**: "Can't simulate every BS location, need fast adaptation to unseen sites"
- FL motivation is WEAK ("if you have data, train independently") -- transfer learning is the right framing

### Novelty vs. Prior Work
- FedPer (2-way): shared encoder + local head -> we add theta_BS as 3rd component
- FedRep, MAML, Per-FedAvg: exist individually, but 3-way physical decomposition for wireless is novel
- "Overlapping observation": multiple BSs observe same UEs -- unique to wireless, not in standard FL
- Individual components exist; **combination with physical meaning is the novelty**

### Architecture Details
- Input: H_LS (noisy LS estimate) -> (B, 2, 8, 1024) -- real/imag, 2x4 antenna pairs, 1024 subcarriers
- Encoder: Conv2D + ResBlocks (spatial dims preserved)
- Site injection variants: **FiLM** (default), concat, add -- experiment to find best
- Task head: ResBlocks + Conv2D output -> residual correction
- Output: H_est = H_LS + residual (ReEsNet-style)
- theta_BS integration order (before/after encoder) needs experimentation

### Channel Estimation Task
- Use established methods: **ReEsNet** (Li et al. 2020, ~200 citations) -- NOT novel architecture
- Task method should NOT be novel to avoid diluting paper's actual contribution
- Input: noisy LS estimate, Output: denoised channel, Metric: NMSE (dB)

### Cold Start Value
- Advantage is in **adaptation speed**, not eternal superiority
- With enough data, independent training catches up -> contribution is the cold-start phase
- theta_BS zero-initialized -> adapts from this starting point

---

## 2. Dataset

- Sionna RT, Munich scene, 15 GHz mmWave, 400 MHz BW, 1024 subcarriers
- 8 BS (fixed positions), 100 UE/snapshot, multiple snapshots (>=100)
- 2x2 TX (4 ant), 1x1 RX cross-pol (2 ant) -> 8 antenna pairs
- Pre-train BSs: [0..5], Test BSs: [6, 7]
- Storage: .npz per snapshot + metadata.parquet

---

## 3. Experiment Phases

### Phase 0.5: Architecture Search
**Goal**: θ_BS adapter 종류/위치/공유 전략을 체계적으로 비교하여 최적 구조 결정

#### Step 0.5-1: Adapter Type & Placement
- [ ] SSF (Scale & Shift) vs LoRA vs none
- [ ] Placement: encoder only / task_head only / both
- [ ] 11 configurations × FL 50 rounds on 8 BSs

#### Step 0.5-2: Sharing Strategy
- [ ] all_except_adapter (E+θ_task shared) vs encoder_only (FedPer-style)

#### Step 0.5-3: Literature Mapping
- [ ] 각 config이 기존 연구(FedAvg, FedPer, FedPerLoRA 등)에 해당하는지 확인

**Phase 0.5 통과 기준**: 최적 config이 FedAvg/FedPer baseline보다 최소 1 dB 개선

---

### Phase 0: Sanity Check
**Goal**: ReEsNet 채널 추정이 우리 Sionna RT 데이터에서 논문 수준으로 동작하는지 확인

#### Step 0-1: 데이터 생성 검증
- [ ] Sionna RT로 multi-snapshot 데이터 생성 (최소 50 snapshots)
- [ ] 데이터 형상 확인: CFR (N_UE, 2, 4, 1024), metadata.parquet
- [ ] CIR/CFR 시각화: 물리적으로 합리적인지 (path loss, delay spread 등)

#### Step 0-2: 단일 BS 채널 추정 학습
- [ ] PlainEstimator (ReEsNet) 1개 BS 데이터로 학습
- [ ] SNR별 NMSE 곡선: SNR={0, 5, 10, 15, 20, 25, 30} dB
- [ ] LS baseline과 비교 -> DNN이 확실히 좋아야 함
- [ ] LMMSE baseline과 비교 -> DNN이 LMMSE보다 좋거나 비슷해야 함
- **성공 기준**: NMSE < -15 dB @ SNR=20 dB (ReEsNet 논문 수준)

#### Step 0-3: 다중 BS 독립 학습
- [ ] 8개 BS 각각 독립적으로 PlainEstimator 학습
- [ ] BS간 성능 차이 확인 (데이터 양, 채널 특성 차이)
- **확인사항**: BS별 성능이 합리적으로 다른지 (위치/환경 차이 반영)

**Phase 0 통과 기준**: 모든 BS에서 NMSE < -10 dB @ SNR=20 dB

---

### Phase 1: 3-way 구조 검증
**Goal**: E + theta_task + theta_BS 분리가 의미있는 성능 차이를 만드는지

#### Step 1-1: 3-way vs 2-way vs FedAvg vs Independent
- [ ] 4가지 방법 FL 학습 (같은 총 epoch)
  - 3-way (ours): E shared + theta_task shared + theta_BS local
  - 2-way (FedPer): E shared + theta_task local
  - FedAvg: everything shared
  - Independent: per-BS training, no sharing
- [ ] Per-BS bar chart + FL convergence curve
- **성공 기준**: 3-way >= FedPer > FedAvg >= Independent (per-BS 성능)

#### Step 1-2: Few-shot Adaptation (k-shot)
- [ ] Pre-train on BS_1~6, adapt to BS_7 with k={5, 10, 20, 50, 100, 200}
- [ ] 방법: theta_BS-only vs fine-tune-all vs from-scratch
- [ ] k-shot curve with error bars (N=5 repeats)
- **성공 기준**: theta_BS-only가 k<50에서 다른 방법보다 NMSE 2+ dB 우세

#### Step 1-3: Pre-trained E vs From-scratch
- [ ] Pre-trained E+theta_task로 새 BS 적응 vs 처음부터 학습
- [ ] Convergence curve (epoch vs NMSE)
- **성공 기준**: Pre-trained가 수렴 속도 3x+ 빠름

**Phase 1 통과 기준**: 3-way가 2-way보다 최소 1 dB 이상 개선

---

### Phase 2: theta_BS Task-Agnostic 검증 (Killer Experiment)
**Goal**: theta_BS가 task-independent한 site representation인지 증명

#### Step 2-1: Task A -> Task B 전이
- [ ] Task A (채널 추정)로 theta_BS 학습 -> freeze
- [ ] Task B (power profile 예측)에 frozen theta_BS 사용 vs 미사용
- [ ] Convergence curve: with theta_BS vs without vs from-scratch
- **성공 기준**: theta_BS 있으면 Task B 성능 10%+ 개선

#### Step 2-2: E as Downstream Backbone
- [ ] Pre-trained E를 freeze하고 새 task의 feature extractor로 사용
- **성공 기준**: frozen E > random E (최소 NMSE 3 dB 차이)

**Phase 2 통과 기준**: theta_BS가 Task B에서 유의미한 개선 보여야 함

---

### Phase 3: Ablation Studies

#### Step 3-1: theta_BS 차원
- [ ] dim = {8, 16, 32, 64, 128, 256} -> sweet spot 찾기 (보통 32~64 예상)

#### Step 3-2: Site injection 방식
- [ ] FiLM vs concat vs add vs none

#### Step 3-3: Pre-training BS 개수
- [ ] {2, 4, 6} BS로 pre-train -> 테스트 BS 성능
- [ ] More BS -> better generalization 보여야 함

#### Step 3-4: F_UE 기여도 (Feature Ablation)
- [ ] Full F_UE (pos + device features) vs pos only vs device only vs no F_UE

#### Step 3-5: Cold Start 분석
- [ ] NMSE vs training sample count 곡선
- [ ] Crossover point 찾기 (from-scratch가 따라잡는 시점)
- **확인사항**: crossover가 실제 배치에서 현실적인 데이터 양 이후인지

#### Step 3-6: Multi-task theta_BS
- [ ] Task A only로 학습한 theta_BS vs Task A+B joint로 학습한 theta_BS
- [ ] 동일한 Task B held-out 데이터로 평가
- **확인사항**: joint가 single-task보다 좋으면 "multi-task → better site representation" 주장 가능

---

## 4. Baselines

- LS (identity -- just returns noisy input)
- LMMSE (diagonal Wiener filter approximation)
- Independent ReEsNet (per-BS, no sharing)
- FedAvg ReEsNet (all params shared)
- FedPer (2-way: shared encoder + local head)
- MAML (meta-learned initialization)
- From-scratch ReEsNet (on target BS only)

---

## 5. 논문 Figure 계획

- Fig 1: Architecture diagram
- Fig 2: Phase 0 SNR-NMSE curve (기본 검증)
- Fig 3: Phase 1 FL comparison bar chart
- Fig 4: Phase 1 few-shot curve
- Fig 5: Phase 2 task-agnostic transfer
- Fig 6: Ablation (dim + injection + cold start)

---

## 6. 실행 우선순위

1. **Phase 0** (기본 동작) -> 안 되면 나머지 의미 없음
2. **Phase 1-1** (3-way vs baselines) -> 핵심 contribution 검증
3. **Phase 1-2** (few-shot) -> 실용적 가치 증명
4. **Phase 2-1** (task-agnostic) -> killer contribution
5. **Phase 3** (ablations) -> 논문 completeness
6. **Figures** -> 최종 정리

---

## 7. Project Structure

```
src/
├── config.py
├── data/generate.py, dataset.py, utils.py
├── models/estimator.py (3-way), baselines.py
├── training/trainer.py, federated.py, meta_learning.py
└── experiments/ (ipynb per verification step)
```

---

## 8. Deployment Scenario 논의: UMa vs UMi

### 배경
현재 시뮬레이션: 8 BS, 뮌헨 씬 (~0.9 km²), BS 높이 16~90m (rooftop)
→ 이는 **UMa (Urban Macro)** 에 해당 (높이 25m+, ISD 500m+, ~8-10 BS/km²)

### 3GPP 배치 기준 (실제 도시)

| 시나리오 | BS 높이 | ISD | 밀도 | 커버리지 반경 |
|---|---|---|---|---|
| UMa | 25m+ | 500m+ | 8-10 BS/km² | 수백m~km |
| UMi | 10-25m | 200m | 40-50 BS/km² | 100-300m |
| mmWave small cell | 3-6m | 100-200m | 더 높음 | 100m 이하 |

- 실제 뮌헨 사례: Huawei MRC 옥상 25m, Oktoberfest에 macro 8개+기존6개 (42만m²)
- 5G mmWave는 4G 대비 BS 수 3-5배 필요

### 우리 연구에 UMi가 더 적합한 이유

1. **O-RAN narrative**: O-RAN은 small cell 중심 disaggregated RAN → UMi가 자연스러운 시나리오
2. **Cold-start motivation 강화**: UMi는 BS 수가 많음 → 새 site 배치 빈번 → "모든 site를 시뮬레이션할 수 없다"는 동기가 더 강해짐
3. **6G forward-looking**: 6G는 더 높은 주파수, 더 dense deployment 예상 → UMi/small cell이 주류
4. **Collaborative BS**: dense deployment에서 커버리지 겹침이 많아 BS 간 협력이 더 필요

### 현실적 선택

**현재 8 BS (UMa) 유지하되, 논문 framing을 조정하는 방안:**

- 8 BS @ rooftop은 UMa로 justify 가능 (실제 도시 밀도 ~9 BS/km²에 부합)
- 논문에서: "UMa 시나리오로 검증하되, UMi dense deployment에서는 cold-start 문제가 더 심각하므로 우리 방법의 가치가 더 커진다" 로 framing
- 또는: BS 높이를 10-25m로 낮추면 UMi로 부를 수 있음 (BS 수는 유지, 커버리지 홀은 motivation으로 활용)

**UMa vs UMi 자체가 6G specific finding:**

- 같은 site representation 방법이 UMa/UMi 모두에서 동작하는지 → ablation study 가능
- UMi에서 cold-start 문제가 더 심각 → 우리 방법의 gain이 더 크다는 것을 보이면 강한 contribution

### 결정 (2026-03-17)
- **데이터셋 재생성 예정** — UMi 배치로 전환
- [ ] BS 높이 10-25m로 조정 (3GPP UMi 기준)
- [ ] BS 수 16개로 증가 (UMi 밀도 ~18 BS/km², 뮌헨 씬 0.9km² 기준)
- [ ] `scripts/bs_coverage_check.ipynb`의 place_bs_umi() 결과 활용
- [ ] 기존 8 BS UMa도 비교 실험으로 유지 가능 (ablation)

---

## Notes
- "비판적으로" -- push back hard, thesis-antithesis-synthesis
- "collaborative BS가 아니어도 됨" -- relaxed project scope
