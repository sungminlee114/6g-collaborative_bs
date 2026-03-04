# Experiment Roadmap: 단계별 검증 계획

## Phase 0: 기본 동작 검증 (Sanity Check)
**목표**: ReEsNet 채널 추정이 우리 Sionna RT 데이터에서 논문 수준으로 동작하는지 확인

### Step 0-1: 데이터 생성 검증
- [ ] Sionna RT로 multi-snapshot 데이터 생성 (최소 50 snapshots)
- [ ] 데이터 형상 확인: CFR (N_UE, 2, 4, 1024), metadata.parquet
- [ ] CIR/CFR 시각화: 물리적으로 합리적인지 (path loss, delay spread 등)

### Step 0-2: 단일 BS 채널 추정 학습
- [ ] PlainEstimator (ReEsNet) 1개 BS 데이터로 학습
- [ ] **성공 기준**: NMSE < -15 dB @ SNR=20 dB (ReEsNet 논문 수준)
- [ ] SNR별 NMSE 곡선: SNR={0, 5, 10, 15, 20, 25, 30} dB
- [ ] LS baseline과 비교 → DNN이 확실히 좋아야 함
- [ ] LMMSE baseline과 비교 → DNN이 LMMSE보다 좋거나 비슷해야 함

### Step 0-3: 다중 BS 독립 학습
- [ ] 8개 BS 각각 독립적으로 PlainEstimator 학습
- [ ] BS간 성능 차이 확인 (데이터 양, 채널 특성 차이)
- [ ] **확인사항**: BS별 성능이 합리적으로 다른지 (위치/환경 차이 반영)

**Phase 0 통과 기준**: 모든 BS에서 NMSE < -10 dB @ SNR=20 dB

---

## Phase 1: 3-way 구조 검증
**목표**: E + θ_task + θ_BS 분리가 의미있는 성능 차이를 만드는지

### Step 1-1: 3-way vs 2-way vs FedAvg vs Independent
- [ ] 4가지 방법 FL 학습 (같은 총 epoch)
- [ ] **성공 기준**: 3-way ≥ FedPer > FedAvg ≥ Independent (per-BS 성능)
- [ ] Per-BS bar chart + FL convergence curve

### Step 1-2: Few-shot adaptation (k-shot)
- [ ] Pre-train on BS₁~₆, adapt to BS₇ with k={5,10,20,50,100,200}
- [ ] 방법: θ_BS-only vs fine-tune-all vs from-scratch
- [ ] **성공 기준**: θ_BS-only가 k<50에서 다른 방법보다 NMSE 2+ dB 우세
- [ ] k-shot curve with error bars (N=5 repeats)

### Step 1-3: Pre-trained E vs From-scratch
- [ ] Pre-trained E+θ_task로 새 BS 적응 vs 처음부터 학습
- [ ] **성공 기준**: Pre-trained가 수렴 속도 3x+ 빠름
- [ ] Convergence curve (epoch vs NMSE)

**Phase 1 통과 기준**: 3-way가 2-way보다 최소 1 dB 이상 개선

---

## Phase 2: θ_BS Task-Agnostic 검증 (Killer Experiment)
**목표**: θ_BS가 task-independent한 site representation인지 증명

### Step 2-1: Task A → Task B 전이
- [ ] Task A (채널 추정)로 θ_BS 학습 → freeze
- [ ] Task B (power profile 예측)에 frozen θ_BS 사용 vs 미사용
- [ ] **성공 기준**: θ_BS 있으면 Task B 성능 10%+ 개선
- [ ] Convergence curve: with θ_BS vs without vs from-scratch

### Step 2-2: E as downstream backbone
- [ ] Pre-trained E를 freeze하고 새 task의 feature extractor로 사용
- [ ] **성공 기준**: frozen E > random E (최소 NMSE 3 dB 차이)

**Phase 2 통과 기준**: θ_BS가 Task B에서 유의미한 개선 보여야 함

---

## Phase 3: Ablation Studies
**목표**: 각 설계 선택의 기여도 정량화

### Step 3-1: θ_BS 차원
- [ ] dim = {8, 16, 32, 64, 128, 256}
- [ ] Sweet spot 찾기 (보통 32~64 예상)

### Step 3-2: Site injection 방식
- [ ] FiLM vs concat vs add vs none
- [ ] Best 방식 선택 근거

### Step 3-3: Pre-training BS 개수
- [ ] {2, 4, 6} BS로 pre-train → 테스트 BS 성능
- [ ] More BS → better generalization 보여야 함
- [ ] 하지만 너무 많으면 E가 과도하게 일반화? 확인

### Step 3-4: F_UE 기여도 (Feature Ablation)
- [ ] Full F_UE (pos + device features) vs pos only vs device only vs no F_UE
- [ ] 각 feature group의 marginal contribution

### Step 3-5: Cold Start 분석
- [ ] NMSE vs training sample count 곡선
- [ ] Crossover point 찾기 (from-scratch가 따라잡는 시점)
- [ ] **이 crossover가 실제 배치에서 현실적인 데이터 양 이후인지 확인**

---

## Phase 4: 논문 Figure 준비
- [ ] Fig 1: Architecture diagram (이미 있음)
- [ ] Fig 2: Phase 0 SNR-NMSE curve (기본 검증)
- [ ] Fig 3: Phase 1 FL comparison bar chart
- [ ] Fig 4: Phase 1 few-shot curve
- [ ] Fig 5: Phase 2 task-agnostic transfer
- [ ] Fig 6: Ablation (dim + injection + cold start)

---

## 실행 우선순위
1. **Phase 0** (기본 동작) → 안 되면 나머지 의미 없음
2. **Phase 1-1** (3-way vs baselines) → 핵심 contribution 검증
3. **Phase 1-2** (few-shot) → 실용적 가치 증명
4. **Phase 2-1** (task-agnostic) → killer contribution
5. **Phase 3** (ablations) → 논문 completeness
6. **Phase 4** (figures) → 최종 정리
