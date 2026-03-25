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

- Sionna RT, Munich scene, 6G ELAA multi-scale configs
- 16 BS (UMi layout), 100 UE/snapshot, multiple snapshots (>=100)
- ELAA TX: Small 16×16 (256), Base 32×16 (512), Big 32×32 (1024)
- RX: 1×1 cross-pol (2 ant)
- Dual-band: 15 GHz (FR3) + 28 GHz (FR2)
- Subcarriers: 1024 (Small) / 2048 (Base) / 4096 (Big)
- Pre-train BSs: [0..11], Test BSs: [12..15]
- Storage: .npz per snapshot + metadata.parquet
- Legacy: munich_uma8 (8 BS, 4 ant), munich_umi16 (16 BS, 4 ant)

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

## 9. Cross-Cluster 분석 & 연구 갭

> relworks.md 논문 군집 간 연결 관계 및 연구 갭 분석 (2026-03-04 작성)

### 9.1 논문 간 연결 관계

```
핵심 스토리라인 1: "O-RAN에서의 on-device AI 배포"
  [P02:dApps] → [P34:SSBA] → [P36:MAB] → [P41:HW-Hetero]
  실시간 제어 메커니즘 → 사이트별 빔 AI → 초경량 추론 → 일반화 한계
  ⟹ 갭: dApp으로 배포 가능한 빔 관리 AI가 HW 이질성에서도 동작하는 방법?

핵심 스토리라인 2: "분산/협력 학습으로 다중 BS 지능"
  [P14:Edge-LAM] → [P30:SL-6G] → [P09:D2D-FL] → [P10:Robust-FL]
  Edge LAM 채널 예측 → Split Learning → 동적 FL+MAC → FL 보안
  ⟹ 갭: Federated channel prediction에서 동적 환경 + 보안을 동시에 보장하는 통합 프레임워크?

핵심 스토리라인 3: "LLM/SLM의 6G PHY 적용"
  [P19:Multi-task PHY] → [P22:TinyLM-6G] → [P21:Push-LLM] → [P31:SLIDE]
  LLM PHY 다중과제 → 적정 크기 → Edge 배포 → 모델 전달
  ⟹ 갭: 1-3B 규모 SLM을 on-device PHY 과제에 fine-tune하고 O-RAN으로 전달하는 end-to-end 시스템?

핵심 스토리라인 4: "시뮬레이션 → 실제 배포 브릿지"
  [P43:Sionna] → [P44:DeepTelecom] → [P34:SSBA] → [P06:China-Mobile]
  Ray-tracing 시뮬레이터 → 디지털 트윈 데이터셋 → DL 빔 정렬 → 5000+ BS 필드 트라이얼
  ⟹ 갭: Sionna 기반 디지털 트윈에서 학습 → 실제 배포 transfer learning의 체계적 방법론?
```

### 9.2 주요 연구 갭 종합

| 갭 ID | 설명 | 관련 논문 | 난이도 |
|-------|------|----------|-------|
| G1 | On-device AI (dApp/xApp)의 HW 이질성 강건한 빔 관리 | P02,P34,P36,P41 | ★★★★★ |
| G2 | 동적 채널 + 보안 + FL 통합 채널 예측 | P09,P10,P14 | ★★★★☆ |
| G3 | SLM (1-3B) on-device PHY 다중과제 + O-RAN 배포 | P19,P22,P21,P31 | ★★★★★ |
| G4 | 디지털 트윈 → 실세계 전이학습 체계 | P43,P44,P34,P06 | ★★★★☆ |
| G5 | Edge LAM의 마이크로서비스 기반 협력 추론 + 무선 최적화 | P14,P26,P30 | ★★★★☆ |
| G6 | 모델 버전 관리 + 동적 업데이트의 안정성-성능 트레이드오프 | P08,P41 | ★★★☆☆ |
| G7 | ISCC 기반 센싱 지원 협력 빔 관리 | P16,P32 | ★★★★☆ |
| G8 | SE/attention block의 FL aggregation 시 site-specific adaptation 손실 메커니즘 분석 | P78, P80 |
| G9 | Site-adaptive CE에서 FL vs Transfer Learning vs Continual Learning 체계적 비교 | P82, P83, P86 |
| G10 | 경량 모델 (O-DU급)에서 collaborative learning의 이점이 커지는 현상의 이론적 분석 | P78, P85 |

---

## 10. Nature Communications 연구주제 후보

### 후보 1: **"Heterogeneity-Resilient On-Device AI for Cooperative Beam Management in 6G O-RAN"**
> **HW 이질성에 강건한 협력 기지국 빔 관리 on-device AI**

- **근거**: P41(Zeulin)이 HW 이질성 문제를 식별했으나 해법 미제시. P34(SSBA), P36(MAB)은 단일 BS 솔루션. 다벤더 O-RAN 환경에서 이질적 BS들이 협력하여 빔 관리를 수행하는 프레임워크는 부재.
- **방법론**: Sionna RT(P43)로 15GHz 다중 BS 데이터셋 생성 → domain-invariant feature (beamspace/angular-delay profile) 학습 → meta-learning/continual learning으로 HW 변화에 적응 → dApp(P02) 또는 near-RT RIC으로 배포 → FL(P32)로 다중 BS 협력
- **노벨티**: (1) HW 이질성 문제 + 다중 BS 협력의 최초 결합, (2) 디지털 트윈 기반 사전학습 → 실세계 fine-tuning 파이프라인
- **임팩트**: O-RAN Alliance의 다벤더 비전과 직결. 실용적 6G 배포의 핵심 장벽 해결.
- **Nature Comms 적합성**: ★★★★★ — 이질적 시스템의 협력 지능이라는 broad impact, 물리/ML/시스템 crossover

### 후보 2: **"Foundation Model-Enabled Collaborative Intelligence at the 6G Radio Edge"**
> **6G Radio Edge에서의 파운데이션 모델 기반 협력 지능**

- **근거**: P14(Edge LAM), P19(Multi-task PHY LLM), P22(TinyLM-6G)가 각각 edge LAM, PHY 다중과제, 스케일링을 다루지만 통합 없음. P45(SpectrumFM)은 스펙트럼만. **채널 예측 + 빔포밍 + 간섭 관리를 통합하는 6G PHY 파운데이션 모델은 부재**.
- **방법론**: Sionna RT로 다양한 환경/주파수/배열 데이터 대규모 생성 → self-supervised pre-training (masked CIR reconstruction + next-slot CFR prediction, P45 영감) → LoRA fine-tuning per BS → split federated fine-tuning (P14, P30) across collaborative BSs → 1-3B 모델 (P22 가이드라인)
- **노벨티**: (1) 최초 PHY-layer FM 사전학습 + 다중 BS FL fine-tuning, (2) 채널/빔/간섭 통합 과제, (3) 디지털 트윈 생성 데이터 → 실 데이터 bridging
- **임팩트**: AI-native 6G의 핵심 비전 구현. 범용 무선 FM이 특정 환경에 적응하는 패러다임.
- **Nature Comms 적합성**: ★★★★★ — "Foundation model for physical layer"라는 새 패러다임. AI + 통신 + 물리의 crossover.

### 후보 3: **"Split Inference with Simultaneous Model Delivery for Real-Time Cooperative Base Station Intelligence"**
> **실시간 협력 BS 지능을 위한 동시적 모델 전달 + 분할 추론**

- **근거**: P31(SLIDE)이 모델 다운로드+추론 동시화를 제안했으나 단일 BS. P30(SL)이 multi-edge 분할 학습을 다루지만 추론 시 모델 전달 미고려. P26(Splitwise)이 sub-layer 분할을 제안했으나 BS 협력 없음. **다중 BS가 협력하여 UE에 모델을 전달하면서 동시에 분할 추론하는 프레임워크는 부재**.
- **방법론**: CoMP-style 다중 BS 동시 모델 전달 (각 BS가 모델의 다른 부분 전송) → UE에서 수신된 레이어부터 즉시 추론 시작 → Lyapunov 기반 안정성 보장 (P26) → O-RAN near-RT RIC으로 BS 간 모델 파티션 조율
- **노벨티**: (1) CoMP + model delivery + split inference의 최초 통합, (2) 다중 BS 동시 전달로 다운로드 시간 1/N 감소
- **Nature Comms 적합성**: ★★★★☆ — 시스템 수준 혁신, 그러나 이론적 깊이가 충분해야 함

### 후보 4: **"Digital Twin-Aided Self-Evolving Beam Management for 6G O-RAN"**
> **디지털 트윈 기반 자기 진화 빔 관리**

- **근거**: P34(SSBA)가 디지털 트윈 파이프라인을 제안했으나 미구현. P43(Sionna) + P44(DeepTelecom)이 시뮬레이션 도구/데이터셋 제공. P08이 모델 버전 관리 제안. **디지털 트윈으로 지속적으로 모델을 진화시키고, 실세계 피드백으로 calibration하는 closed-loop 시스템은 부재**.
- **방법론**: Sionna RT 디지털 트윈 → 초기 빔 관리 모델 학습 → on-device 배포 → 실세계 RSS/CIR 피드백으로 모델 drift 감지 → 디지털 트윈 자동 업데이트 → 모델 재학습 → RL 기반 업데이트 정책 (P08)으로 안정성 보장 배포
- **노벨티**: (1) Sim-to-real-to-sim 순환 학습, (2) 자율적 모델 진화 (human-out-of-the-loop)
- **Nature Comms 적합성**: ★★★★☆ — 자율 시스템의 self-evolution은 광범위한 관심, 그러나 기존 digital twin 연구와의 차별화 필요

### 후보 5: **"Sensing-Aided Cooperative Channel Prediction via Federated On-Device AI in 6G Networks"**
> **6G에서 센싱 보조 협력 채널 예측을 위한 연합 on-device AI**

- **근거**: P16(ISCC)이 센싱+통신+컴퓨팅 통합을 제안. P32(Beam Survey)가 ISAC 빔 관리 정리. P14(Edge LAM)이 federated 채널 예측. **센싱 데이터 (radar/LiDAR)를 활용한 federated cooperative channel prediction은 미탐구**.
- **방법론**: 다중 BS가 각각 로컬 센싱 (radar) + 파일럿 기반 채널 추정 → 멀티모달 fusion (P37 VBS 영감) → federated learning으로 global prediction model → on-device 추론으로 proactive handover/beam switching
- **노벨티**: (1) ISAC + FL + 채널 예측의 최초 결합, (2) 센싱이 채널 예측 정확도 상한을 높이는 메커니즘 (P16 이론 확장)
- **Nature Comms 적합성**: ★★★★☆ — ISAC은 6G 핵심이나, 실험 검증의 깊이 필요

---

## 11. PACE-Net 실험 발견 & 해석

### 실험적 발견 (E0 Architecture Search에서)

- PACE-Net에서 FL aggregation 시 SE의 implicit site adaptation이 평균화되어 성능 저하 관찰
  - Independent: -18.34 dB vs Ours (FL+LoRA): -17.37 dB
  - ResNet/DWS-ResNet에서는 Ours가 Independent를 이김
- → **SE와 FL의 충돌은 PACE-Net 원논문(P78)에서 미다룸 — 새로운 finding**

### 해석 (P80 ANFR 기반)

- SE의 channel attention은 **입력에 따라 feature를 adaptive하게 re-weight** (P78 PACE-Net)
- FL aggregation은 이 adaptive weight의 기반이 되는 Linear layer를 평균화
- P80 (ANFR)은 channel attention이 FL heterogeneity를 **완화**할 수 있음을 보였으나, 이는 attention을 **shared feature의 일관성 필터**로 쓸 때
- 우리 경우 SE는 **site-specific feature amplifier** 역할 → FL 평균화와 목적이 상충

---

## 12. 세팅 냉정 분석 & 6G ELAA 확장 전략

### 12.1 현재 세팅 객관적 평가 (2026.03 기준)

| 항목 | 현재 세팅 | 실제 5G NR | 6G 방향 | 평가 |
|------|-----------|-----------|---------|------|
| 주파수 | 28 GHz (FR2) | FR1: <6GHz, FR2: 24-52GHz | sub-THz (100-300 GHz) | 5G 수준, 6G 아님 |
| BS 안테나 | 4 Tx | Macro 32-64T, Small cell 4-8T | ELAA 256-1024+ | 5G small cell 수준 |
| UE 안테나 | 2 Rx | 2-4 Rx | 2-8 Rx | **현실적** |
| 서브캐리어 | 1024 OFDM | FR2: 792-3168 SC | 유사 | **현실적** |
| 채널 모델 | Sionna RT (뮌헨) | 3GPP 38.901 + RT | Site-specific RT | **현실적** |
| Multi-cell | 8-16 BS, FL | 수십-수백 cell | O-RAN 협력 | 방향 맞음, 규모 작음 |

**결론**: "6G 시스템 시뮬레이션" 주장은 과장. 정확히는 **5G FR2 small cell** 환경.
프로젝트 가치는 시스템 세팅이 아니라 **FL + site-specific adaptation 프레임워크**에 있음.

### 12.2 PACE-Net vs 우리 — 정직한 비교

| 항목 | PACE-Net (Yang et al., Entropy 2025) | 우리 |
|------|--------------------------------------|------|
| 채널 모델 | Kronecker (통계적, i.i.d.) | Sionna RT (레이트레이싱) |
| OFDM | 없음 (flat channel, H ∈ ℂ^(Nr×Nt), T=16 시간축 pilot) | 1024 frequency-domain samples |
| BS 안테나 | 64 Tx (논문 Fig.1: Nt=64) | 64~512 Tx |
| UE 안테나 | 16 Rx (**비현실적**) | 2 Rx (현실적) |
| MIMO 구조 | 64×16 (학술용) | 2×512 (현실적) |
| 네트워크 채널 | C=256 (antenna 수와 일치) | C=64 (VRAM 제약) |
| Multi-cell | 없음 | 있음 (FL) |
| Feature map 크기 | (B, 256, 16, 16) 작음 | (B, 64, 8, 1024) 큼 |

**C=256 vs C=64 이유**: 논문은 spatial dim (16×16)이 작아 채널을 크게 가져갈 수 있음.
우리는 1024 subcarrier 유지 때문에 feature map이 이미 크므로 (batch=260일 때 activation ~17 GB),
C=256으로 올리면 ~70 GB로 단일 GPU 불가.

### 12.3 FR2 mmWave 채널 추정 연구 현황

FR2 DL 채널 추정은 **성숙했지만 dead는 아님** (2025-2026에도 논문 출판 중):
- CNN/UNet mmWave MIMO CE (arxiv 2506.11714, 2025.06)
- Dual-band massive MIMO extrapolation (arxiv 2601.06858, 2026.01)
- Decentralized FL for GNN CE in O-RAN (IEEE ICMLCN 2025)

**2025-2026 핫토픽 이동**:
1. Near-field ELAA 채널 추정 (arxiv 2507.23526 — comprehensive survey)
2. Sub-THz / THz UM-MIMO (IET Comms 2025, arxiv 2601.18333)
3. RIS-aided 채널 추정
4. Cross-band extrapolation (sub-6GHz → mmWave)
5. ISAC (통합 센싱-통신)

**FL + 채널 추정**은 아직 emerging 분야 — 경쟁 적음.

### 12.4 Sionna 기반 6G ELAA 확장 전략

#### 12.4.1 ELAA 물리 계층 구현

6G ELAA의 핵심인 수천 개 안테나(5,000+) 환경에서 발생하는 **Near-field Beamfocusing** 효과를
Sionna의 GPU 가속 레이 트레이싱(RT)으로 정밀하게 모델링 가능.

- Sionna의 `SyntheticArray` 기능을 활용하여 ELAA 거대 위상 배열 구조 정의
- 고주파수(15 GHz+)에서의 구형파(Spherical Wave) 특성 반영
- Near-field 영역: Rayleigh distance가 수십-수백m로 확대 → far-field 가정 깨짐

#### 12.4.2 dApp 로직 통합 (실시간 Closed-loop)

dApp의 핵심은 **10ms 미만 실시간성**:

1. Sionna RT에서 CIR(Channel Impulse Response) 추출
2. AI 모델(dApp)의 입력으로 투입
3. 추론된 빔포밍 가중치를 물리 계층에 적용
4. Closed-loop 시뮬레이션으로 실시간 제어 지연 시간 정량화

SIMD 가속 개념 반영하여 AI 워크로드 실행 중 latency budget 분석 필요.

### 12.5 데이터셋 정당성 설득 논리 (Scientific Justification)

리뷰어 예상 질문: **"시뮬레이션 데이터가 실제 환경을 대변할 수 있는가?"**

#### Step 1: Differentiable RT를 통한 Sim-to-Real 간극 해소

> "본 연구의 데이터셋은 고정된 값이 아니다. 실세계에서 수집된 소량의 피드백을 통해
> Sionna의 재질 파라미터(반사율 등)를 역전파(Backpropagation)로 캘리브레이션하여
> 실제 환경에 최적화된 디지털 트윈 데이터를 생성한다."

참고: Differentiable Ray Tracing (Hoydis et al., Sionna RT)

#### Step 2: Site-specific 데이터의 가치 강조

> "6G 기지국은 고유한 주변 지형지물에 종속적이다. Sionna로 생성된 고정밀 LoD3
> 시나리오 데이터는 실제 기지국이 직면할 전파 환경을 가장 사실적으로 모사한
> 'Site-specific' 지능의 토대다."

참고: NVIDIA 6G site-specific 연구, Traciverse, DeepTelecom 파이프라인

#### Step 3: 데이터 생성 파이프라인의 표준성

> "본 데이터셋 생성 방식은 최신 6G 연구 트렌드인 Traciverse 및 DeepTelecom
> 파이프라인과 일관성을 유지하며, 3GPP 및 O-RAN 규격에 부합하는 채널 모델을 따른다."

### 12.6 Nature Comms급 완성도를 위한 추가 실험

1. **Heterogeneity Test**: 동일 데이터셋으로 학습된 모델이 이질적 안테나 구성
   (ELAA vs Standard MIMO)에서도 성능 유지 검증
2. **Self-Evolving Proof**: 운영 중 실시간 데이터 수신 → 모델 자가 진화 →
   NMSE 점진적 개선 시뮬레이션 입증

---

## Notes
- "비판적으로" -- push back hard, thesis-antithesis-synthesis
- "collaborative BS가 아니어도 됨" -- relaxed project scope
