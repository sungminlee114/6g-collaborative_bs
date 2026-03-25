## Cluster L: CE-Skip — Adaptive CE Inference Scheduling Related Works

> CE inference scheduling (when to skip/reduce CE computation) 관련 논문. 6G ELAA의 GPU-native RAN에서 CE를 소프트웨어 태스크로 처리할 때, temporal channel persistence를 활용하여 CE 수행 빈도를 줄이는 연구 방향. 114편 분석 중 HIGH/MEDIUM 관련도 논문을 6개 하위 클러스터로 분류.

```
Cluster L: CE-Skip — Adaptive CE Inference Scheduling (41편)
  ├─ L1: CE Computational Cost & Motivation (10편)
  │    DL-CE가 비싸다 → skip scheduling 동기 부여
  ├─ L2: Temporal Channel Persistence & Prediction (10편)
  │    채널 시간 상관 → skip 가능성의 물리적 근거
  ├─ L3: GPU-native RAN & dApp Architecture (9편)
  │    CE를 schedulable software로 → skip 실행 플랫폼
  ├─ L4: Adaptive/Event-Triggered Inference (4편)
  │    적응적 계산 → skip 결정 방법론
  ├─ L5: Beamforming with Stale/Imperfect CSI (4편)
  │    skip 결과 stale CSI의 BF 영향
  └─ L6: Near-Field ELAA CE Methods (7편)
       ELAA CE 현황 → 모두 static 가정, temporal gap 확인
```

---

### L1: CE Computational Cost & Motivation

CE가 비싸다는 것을 보여주는 논문들. DL-CE inference cost가 LS 대비 33x 이상이라는 사실이 skip scheduling의 핵심 동기.

#### [P85] ReQuestNet: Foundational Learning Model for Channel Estimation (Qualcomm)
- **CE-skip 관련**: 3.6M 파라미터, T=4 iterative refinement으로 CE 모델 중 가장 무거움. Skip scheduling의 savings가 4x로 증폭됨.
- **CE-skip relevance**: ★★★★★

#### [P82] NVIDIA Neural 5G NR Receiver: Environment-Specific Base Stations
- **CE-skip 관련**: A100에서 TensorRT 기반 CE inference ~1ms 측정. 0.5ms slot에서 CE가 2 slot 소요 — skip 1회로 1ms GPU 시간 절약. **Munich Sionna RT** 사용 (본 프로젝트와 동일 시뮬레이션 환경).
- **CE-skip relevance**: ★★★★★

#### [P79] Channelformer: Attention-Based Neural Solution for Wireless Channel Estimation
- **CE-skip 관련**: Online Channelformer inference **20.5ms** (32K params) vs LS 0.6ms — **33x cost ratio**. 이 격차가 CE-skip의 핵심 동기. 70% pruning (9.6K params)으로도 LS 대비 여전히 고비용.
- **CE-skip relevance**: ★★★★★

#### [P60] WirelessGPT: Multi-Task FM for Wireless
- **CE-skip 관련**: CE inference 2.31ms (Transformer head) — FM-based CE의 구체적 cost 수치. 80M 파라미터 FM이 CE에 사용되면 skip scheduling이 더욱 중요.
- **CE-skip relevance**: ★★★★☆

#### [P93] Lightweight DL-Based Channel Estimation for XL-MIMO (XLCNet)
- **CE-skip 관련**: C-XLCNet 0.078ms inference, 29K params — 10x compute/36x model size 압축. 경량 DL-CE도 ELAA scale에서는 누적 비용 상당. Universal NF+FF 적용 가능.
- **CE-skip relevance**: ★★★☆☆

#### [P94] Channel Estimation for Wideband XL-MIMO: Constrained Deep Unrolling
- **CE-skip 관련**: PGD-Net 3.79 GFLOPs per inference. Wideband (10 GHz BW at 100 GHz)에서 beam split으로 subcarrier별 채널 변동 — skip 결정 시 frequency-domain 고려 필요.
- **CE-skip relevance**: ★★★☆☆

#### [P95] LLM4XCE: Large Language Models for XL-MIMO Channel Estimation
- **CE-skip 관련**: 126M 파라미터 (GPT-2 backbone) — per-slot CE에 부적합할 정도로 무거움. LLM-scale CE의 등장이 skip scheduling 필요성을 극대화.
- **CE-skip relevance**: ★★★☆☆

#### [P78] PACE-Net: Channel Estimation via Polarized Self-Attention
- **CE-skip 관련**: PSA module의 O(DK^2 NtNr) complexity — DL-CE가 antenna 수에 선형 scaling하므로 ELAA에서 skip 이득 증가. **주의: flat fading (no OFDM), Kronecker 모델. OFDM frequency-selective 셋팅과 직접 비교 불가.**
- **CE-skip relevance**: ★★★☆☆

#### [P69] ContraWiMAE: Multi-Task Foundation Model for Wireless Channel
- **CE-skip 관련**: 570K params FM-based CE, 0.057ms (90% masking) to 0.237ms (full) inference. 경량 FM-CE가 always-on quality monitor로 활용 가능.
- **CE-skip relevance**: ★★★☆☆

#### [P58] LWM: Large Wireless Model
- **CE-skip 관련**: 600K params wireless FM. Pre-train once, deploy everywhere 패러다임이 site-adaptive CE scheduling을 지원. Temporal modeling 부재 — CE-skip이 채우는 gap.
- **CE-skip relevance**: ★★★☆☆

---

### L2: Temporal Channel Persistence & Prediction

채널이 시간적으로 천천히 변한다는 물리적 근거와, temporal prediction/correlation을 활용하는 기법들. CE-skip의 skip 가능성을 정당화하는 핵심 근거.

#### [P86] Continual Learning for Wireless Channel Prediction
- **CE-skip 관련**: Coherence time 0.3ms (28 GHz/60 km/h) — CE-skip interval의 물리적 하한. Cross-config NMSE inflation 37.5% — 재추정 없이 방치하면 성능 저하 정량화. Experience Replay (LARS)가 scenario drift에 가장 robust.
- **CE-skip relevance**: ★★★★★

#### [P83] Transfer Learning vs Meta-Learning for MIMO-OFDM Channel Denoising
- **CE-skip 관련**: Fine-tune/inference duty cycle이 CE-skip scheduling과 구조적으로 동형. Optimal adaptation frequency가 velocity/SNR에 의존 — 고정 skip interval이 아닌 adaptive scheduling 정당화. MAML 1-step adaptation으로 overhead 최소화.
- **CE-skip relevance**: ★★★★★

#### [P79] Channelformer *(L1과 중복, temporal 측면)*
- **CE-skip 관련**: 인접 OFDM symbol 간 temporal correlation r_t = 0.94-0.97 (750-972 Hz Doppler). Channel이 충분히 느리게 변해 최근 CE 출력 재사용 시 <6% 오차.
- **CE-skip relevance**: ★★★★★

#### [P59] WiFo: Wireless Foundation Model for Channel Prediction
- **CE-skip 관련**: 3D STF patch로 temporal prediction, zero-shot cross-scenario generalization. T/2 future time step 예측 (0.25-1ms interval). CE-skip의 "미래 CE 필요 여부" 예측과 직결. 0-300 km/h velocity coverage.
- **CE-skip relevance**: ★★★★☆

#### [P68] DL CSI Feedback with Temporal Correlation
- **CE-skip 관련**: Angle-difference feedback으로 인접 CSI frame 간 변화가 작고 압축 가능함을 입증. T-frame refinement (과거 T개 CSI 활용)이 CE-skip의 cached estimate 기반 interpolation과 직접 유사.
- **CE-skip relevance**: ★★★★☆

#### [P90] Channel Estimation for 6G Near-Field: Comprehensive Survey
- **CE-skip 관련**: D-STiCE (LSTM 기반 time-varying CE) 언급 — temporal sparse channel parameter tracking. Doppler factor가 Rician model에 명시적 포함. FL-based CE로 12x pilot overhead 감소.
- **CE-skip relevance**: ★★★★☆

#### [P98] A Novel Pilot Scheme for Sub-array Structured ELAA in XL-MIMO
- **CE-skip 관련**: "CIR support patterns remain relatively stable over time" — CE-skip의 핵심 물리적 가정을 명시적으로 언급. Sub-6G (2.6 GHz)에서 더 긴 coherence time = 더 많은 skip 기회.
- **CE-skip relevance**: ★★★★☆

#### [P70] CSI-MAE: Masked Autoencoder-based Channel Foundation Model
- **CE-skip 관련**: 3GPP statistical model 기반 학습으로 cross-scenario zero-shot generalization. 8x8 antenna/256 SC가 본 프로젝트 config와 정확히 일치. UE velocity 포함하나 temporal dimension은 없음.
- **CE-skip relevance**: ★★★☆☆

#### [P19] LLM-Enabled Multi-Task Physical Layer Network
- **CE-skip 관련**: Channel prediction task에서 velocity-dependent NMSE 측정 (10 km/h: -20 dB, 100 km/h: -8 dB). CE-skip의 adaptive trigger 설계 직접 지원: 저속에서 공격적 skip, 고속에서 보수적 skip.
- **CE-skip relevance**: ★★★☆☆

#### [P34] Site-Specific Beam Alignment in 6G via Deep Learning (SSBA)
- **CE-skip 관련**: Site-specific BA가 beam measurement overhead를 16-32x 감소. CE-skip은 동일 철학의 CE 버전: site-specific 학습으로 CE inference overhead 감소. Digital twin pipeline이 CE-skip architecture와 유사.
- **CE-skip relevance**: ★★★☆☆

---

### L3: GPU-native RAN & dApp Architecture

CE를 GPU 커널로 실행하는 software-defined BS 아키텍처. CE-skip의 monitor + scheduler가 배포되는 플랫폼.

#### [P02] dApps: Enabling Real-Time AI-Based Open RAN Control
- **CE-skip 관련**: CE-skip의 architectural home. dApp이 DU에 co-located, E3 interface로 CE KPM 접근, **450us 제어 루프** 실측. "Augmented Sensing and Channel Estimation"을 dApp use case로 명시적 언급. PRB masking control action이 CE-skip의 "skip CE" action과 구조적 동일.
- **CE-skip relevance**: ★★★★★

#### [P07] XAI-on-RAN: Explainable, AI-native, GPU-Accelerated RAN
- **CE-skip 관련**: NVIDIA Aerial A100 testbed에서 **GPU utilization 63%** — CE-skip monitor를 위한 37% compute headroom 존재. LSTM inference 5.1ms, attention 추가 비용 0.6ms. RAN DSP + AI 공존 검증.
- **CE-skip relevance**: ★★★★★

#### [P01] Beyond Connectivity: Open Architecture for AI-RAN Convergence in 6G
- **CE-skip 관련**: AI-RAN Site에서 GPU MIG partitioning (40GB RAN + 20GB LLM + 10GB CNN). RAN DSP가 non-elastic workload — CE-skip이 존중해야 할 timing constraint. AI-and-RAN coexistence 검증 (throughput/CRC 일관).
- **CE-skip relevance**: ★★★★★

#### [P62] AI/ML Lifecycle Management for Interoperable AI Native RAN
- **CE-skip 관련**: 3GPP Rel-16~20 LCM framework. CE-skip의 monitor = LCM Management block의 (de)activation control. SGCS threshold 기반 performance monitoring이 CE-skip의 delta-NMSE threshold와 기능적 동일. "Scheduling more monitoring reduces overhead savings" — CE-skip의 핵심 trade-off.
- **CE-skip relevance**: ★★★★★

#### [P06] Towards AI-Native RAN: An Operator's Perspective
- **CE-skip 관련**: "RS overhead reduction"을 6G use case로 명시 (Table I). AI Node + 6gNB 구조가 CE-skip 배포 모델과 일치. 3D monitoring (model perf + network perf + resource perf) framework 제공.
- **CE-skip relevance**: ★★★★☆

#### [P87] Distributed AI Platform for the 6G RAN (Microsoft)
- **CE-skip 관련**: Far-edge runtime (<1ms), eBPF probe로 RAN NF 계측. "vRAN is largely underutilized (<50%)" — CE-skip의 compute headroom 추가 증거. Inference parameters as orchestrator knobs.
- **CE-skip relevance**: ★★★★☆

#### [P40] Accelerating vRAN and O-RAN with SIMD
- **CE-skip 관련**: CPU-based PHY timing data (4x4 MIMO LMMSE detection 0.03ms). GPU와 대비되는 CPU vRAN 참조점. SIMD vs GPU vs FPGA 비교표 (Table I).
- **CE-skip relevance**: ★★★☆☆

#### [P08] Self-Learning Model Versioning for AI-native O-RAN Edge
- **CE-skip 관련**: RL policy가 "dApp은 stability를 accuracy보다 우선" — CE-skip의 설계 원칙 (CE model은 안정, skip scheduler가 적응). Model versioning으로 LS/LMMSE/DL-CE 전환 관리.
- **CE-skip relevance**: ★★★☆☆

#### [P04] MX-AI: Agentic Observability and Control Platform for Open and AI-RAN
- **CE-skip 관련**: Multi-timescale control hierarchy 검증: dApp RT(<1ms) → xApp(10ms-1s) → rApp(>1s). Per-slice CE policy 가능 (URLLC: skip 금지, eMBB: 공격적 skip).
- **CE-skip relevance**: ★★★☆☆

---

### L4: Adaptive/Event-Triggered Inference

CE-skip의 방법론적 선행 연구. Event-triggered 패러다임, adaptive computation, threshold 기반 결정.

#### [P29] Communication Efficient Cooperative Edge AI via Event-Triggered Offloading
- **CE-skip 관련**: CE-skip의 가장 가까운 방법론적 유사체. **Dual-threshold early-exit** (confidence < beta_l → routine exit, > beta_u → offload, 중간 → 다음 block). Channel-adaptive threshold optimization via lookup table. Missing-target-offloading tradeoff가 CE-skip의 skip-vs-recompute tradeoff와 구조적 동일.
- **CE-skip relevance**: ★★★★★

#### [P35] DL-Based Beam Management for mmWave Vehicular (DeepBT)
- **CE-skip 관련**: **Prediction-aided measurement substitution** — 3회 중 2회를 prediction으로 대체하여 66.7% overhead 감소. Beam domain에서의 CE-skip 정확한 유사체. MAFD metric (beam dynamics 특성화)이 CE-skip의 channel dynamics metric에 영감.
- **CE-skip relevance**: ★★★★★

#### [P56] 5G-Advanced AI/ML Beam Management (3GPP, Nokia)
- **CE-skip 관련**: 3GPP Rel-18 호환 SBP/TBP 평가. MOR (Measurement Overhead Reduction) metric — CE-skip의 평가 지표로 직접 적용 가능. TBP가 속도 2-4x에서 6.7% accuracy loss만 보임 — graceful degradation 검증.
- **CE-skip relevance**: ★★★★☆

#### [P41] Rethinking Beam Management: Generalization Under HW Heterogeneity
- **CE-skip 관련**: ML beam predictor가 antenna/codebook/environment heterogeneity에서 >50% SE drop — per-site model 필요성의 가장 강력한 근거. 15 GHz, 8x8 UPA, 9.5 m/s 차량 환경이 본 프로젝트 config와 유사.
- **CE-skip relevance**: ★★★★☆

---

### L5: Beamforming with Stale/Imperfect CSI

CE-skip으로 인한 stale CSI가 beamforming 성능에 미치는 영향. CE-skip의 QoS guarantee 설계에 필수적.

#### [P38] Data and Model-Driven DL Beamforming (GNN Robust BF)
- **CE-skip 관련**: CSI uncertainty를 Gaussian error (variance 0.075)로 모델링한 robust BF. Interference feature s_k가 CE error 영향을 명시적 캡처. DAQE로 channel error augmented training → stale CSI에 robust한 BF 달성. 5% outage probability constraint로 QoS 보장.
- **CE-skip relevance**: ★★★★★

#### [P49] FL Strategies for Coordinated Beamforming in Multicell ISAC
- **CE-skip 관련**: Multi-cell coordinated BF에서 ICI 관리. HFL의 interference leakage control이 VFL보다 CSI staleness에 robust (local CSI만 사용). CE-skip의 한 BS skip 결정이 인접 BS ICI에 영향 → HFL 접근법이 적합.
- **CE-skip relevance**: ★★★★☆

#### [P10] Robust FL for Wireless Channel Estimation
- **CE-skip 관련**: "Outdate mode" attack (의도적 outdated CSI 제공)이 모델 정확도에 minimal impact — **strong temporal correlation** 때문. CE-skip의 전제 (skip해도 괜찮다)를 실험적으로 지지하는 증거.
- **CE-skip relevance**: ★★★★☆

#### [P50] Personalized FL-Driven Beamforming for ISAC
- **CE-skip 관련**: Multi-BS PFL에서 BS별 heterogeneous sensing/comm trade-off. EM-based adaptive aggregation이 site-specific adaptation과 직결. MATLAB ray-tracing 채널 → Sionna RT와 유사한 방법론.
- **CE-skip relevance**: ★★★☆☆

---

### L6: Near-Field ELAA CE Methods

ELAA-specific CE 방법론. 대부분 static channel 가정 → temporal scheduling gap 확인. CE-skip이 채우는 연구 공백.

#### [P92] Distributed Signal Processing for ELAA Systems
- **CE-skip 관련**: Fronthaul cost가 antenna 수에 선형, LMMSE complexity가 cubic scaling. DCE framework로 spatial cost 절감 — CE-skip은 temporal 차원의 추가 절감. "CE must be done per coherence interval" 언급 — coherence interval이 곧 skip interval.
- **CE-skip relevance**: ★★★★☆

#### [P97] Channel Estimation for XL-MIMO with Decentralized Baseband Processing
- **CE-skip 관련**: SBL-GNNs로 centralized 대비 comparable accuracy at lower complexity (0.004s vs 2.379s). Decentralized architecture에서 per-subarray skip 결정 가능 (heterogeneous scheduling).
- **CE-skip relevance**: ★★★★☆

#### [P99] Integrated Channel Estimation and Sensing for Near-Field ELAA
- **CE-skip 관련**: Joint CE + sensing으로 user position 추정 (mm accuracy above 5 dB). Position 정보가 CE-skip trigger로 활용 가능: 위치 불변 → channel 불변 → skip CE. THz LoS 시나리오에서 매우 안정적 채널 → 높은 skip ratio.
- **CE-skip relevance**: ★★★★☆

#### [P91] Recent Advances in Near-Field Beam Training and CE for XL-MIMO
- **CE-skip 관련**: Comprehensive survey confirming no existing work on temporal/scheduling aspects of CE in XL-MIMO. FR3 (7-24 GHz) identified as open direction. Sensing-aided CE could provide side info for skip decisions.
- **CE-skip relevance**: ★★★☆☆

#### [P89] A Tutorial on Near-Field XL-MIMO Communications Towards 6G
- **CE-skip 관련**: 3.5 GHz/256 antennas에서 Rayleigh distance ~100m — 대부분 UE가 near-field. Near-field channel model (NUSW)의 기본 이해 제공.
- **CE-skip relevance**: ★★★☆☆

#### [P105] U6G XL-MIMO Radiomap Prediction: Multi-Config Dataset
- **CE-skip 관련**: Sionna RT 기반 XL-MIMO dataset (up to 32x32 UPA, 1.8-6.7 GHz, 800 scenes). 3GPP TR 38.901 antenna model — 본 프로젝트와 동일. 고주파(6.7 GHz) coverage 31%로 하락 → 신뢰할 수 있는 CE가 더 중요.
- **CE-skip relevance**: ★★★☆☆

#### [P99b] Near-Field Beamfocusing with Modular Linear Arrays
- **CE-skip 관련**: Modular array에서 per-ULA MUSIC + triangulation = very low complexity CE. CE 자체가 저비용이면 skip 이득 감소. 15 GHz mid-band = moderate coherence time.
- **CE-skip relevance**: ★★★☆☆

---

### Additional Papers (Supporting Context)

이하 논문들은 CE-skip에 LOW-MEDIUM 관련도이나, 특정 인용 포인트가 있어 참고 기록.

#### [P14] Edge Large AI Models: Revolutionizing 6G Networks
- "End-to-end design bypasses explicit CE" — CE overhead 감소가 인식된 문제임을 확인. CE-skip의 대안적 접근 (full bypass vs selective skip).
- **CE-skip relevance**: ★★☆☆☆

#### [P36] Meta-Learning MAB Beam Tracking
- 1 beam probe/timestep으로 최소 measurement budget — CE-skip의 최소 CE inference 목표와 유사. Meta-learning으로 cross-environment transfer.
- **CE-skip relevance**: ★★☆☆☆

#### [P63] Compression of Site-Specific DNNs for MIMO Precoding
- Site-specific compression Pareto front가 사이트마다 다름 → CE-skip threshold도 site-specific이어야 함. 8x8 UPA config 일치.
- **CE-skip relevance**: ★★☆☆☆

#### [P55] Elastic FL over O-RAN Architecture
- Multi-time-scale O-RAN control (non-RT system descriptor, near-RT FL controller, FL MAC scheduler). CE-skip scheduler의 O-RAN 배포 참조.
- **CE-skip relevance**: ★★☆☆☆

#### [P09] Dynamic D2D-Assisted FL over O-RAN
- "D-Events" (discrete-time channel changes) + dynamic model drift metric — CE-skip의 event-triggered paradigm과 개념적 유사.
- **CE-skip relevance**: ★★☆☆☆

#### [P52] Coalition Formation for Heterogeneous FL Channel Estimation
- FL-based CE로 CE-skip이 wrapping할 수 있는 또 다른 CE method. Coalition formation이 channel correlation 기반 — site adaptation과 유사.
- **CE-skip relevance**: ★★☆☆☆

#### [P77] AI/ML for Beam Management: A Standardization Perspective
- Model LCM framework (data collection, training, inference, monitoring, fallback) — CE-skip deployment pipeline template. Lightweight model (<1 MB) 강조.
- **CE-skip relevance**: ★★☆☆☆

#### [P84] Domain Adaptation-Enabled Realistic Map-Based Channel Estimation
- DL-CE model이 domain-specific → environment drift 시 skip threshold를 보수적으로 조정해야 함. Ray-tracing MBCM이 Sionna RT setup과 관련.
- **CE-skip relevance**: ★★☆☆☆

#### [P88] REAL: RL-Enabled xApps for Closed-Loop Optimization in O-RAN
- xApp-level control (KPI sampling 500ms) is too slow for PHY-level CE-skip → dApp 필요성 확인.
- **CE-skip relevance**: ★★☆☆☆

#### [P32] A Survey of Beam Management for mmWave and THz Towards 6G
- Beam measurement periodicity가 CE inference periodicity와 직접 유사. 3GPP P1/P2/P3 procedures 참조.
- **CE-skip relevance**: ★★☆☆☆

---

### Dataset Configuration Comparison Table

HIGH/MEDIUM relevance 논문의 시뮬레이션 환경 비교. **Bold** = 본 프로젝트 config와 유사.

| Paper | P## | Antennas | Freq (GHz) | BW (MHz) | SC | Channel Model | Mobility | Scene |
|-------|-----|----------|-----------|----------|-----|--------------|----------|-------|
| ReQuestNet | P85 | 2x2 MIMO | 5G NR | variable | 4-272 RBs | 3GPP TDL/CDL | 0-450 Hz Doppler | Synthetic |
| NVIDIA NRX | P82 | 2x4 MU-MIMO | **2.14** | 47.5 | 132 PRBs | 3GPP UMi | 0-8 m/s | **Munich (Sionna RT)** |
| Channelformer | P79 | SISO | 2.1 | 1.08 | 72 | 3GPP EPA/EVA/ETU | 0-100 km/h | Synthetic |
| WirelessGPT | P60 | 4x4 UPA | **2.4** | - | 32 | WINNER II | 40-100 km/h | Multi-scenario |
| XLCNet | P93 | 256 ULA | **28** | NB | NB | Hybrid NF+FF | Static | Synthetic |
| Deep Unrolling | P94 | 512 ULA / 2048 UPA | 100 | 10 GHz | 256 | NF + SnS | Static | Synthetic |
| LLM4XCE | P95 | 256 ULA | **28** | NB | NB | Hybrid NF+FF | Static | Synthetic |
| Continual Learning | P86 | - | 5.0 | 100 | 18 RBs | QuaDRiGa | 0-60 km/h | UMi |
| Transfer/Meta CE | P83 | (2,16) | **3.5** | - | 512 | CDL-B/E | 60-120 km/h | Synthetic |
| WiFo | P59 | 4-32 UPA | 1.5-**28** | - | 32-128 | QuaDRiGa | 0-300 km/h | Multi-scenario |
| Temporal CSI FB | P68 | (8,2,2) | 2.4/5.0 | 20 | 256/64 | DeepMIMO | 0.4-2.8 m/s | Indoor |
| NF CE Survey | P90 | Various | mmW/THz | - | - | Multiple NF | Doppler mentioned | Survey |
| Sub-Array Pilot | P98 | 1024 (128 sub) | **2.6** | 15 kHz/SC | 1024 | COST2100 | Static* | Semi-urban |
| dApps | P02 | - | - | - | 384-2048 IQ | Real 5G | - | OAI Testbed |
| XAI-on-RAN | P07 | 4T4R | 5G SA | - | - | Real 5G | Robot UE | **NVIDIA Aerial Testbed** |
| AI-RAN Conv. | P01 | - | - | - | - | Real 5G | - | X5G Testbed (A100) |
| Event-Triggered | P29 | - | - | 30 MHz | - | Rayleigh fading | - | Medical imaging |
| DeepBT | P35 | - | mmWave | - | OFDM | Ray-tracing | Vehicular | Marseille/Rosslyn |
| 5G-Adv BM | P56 | UPA | FR2 | - | - | 3GPP UMa | 3-120 km/h | 3GPP evaluation |
| Robust BF | P38 | 4 TX | - | 10 | N/A | Rayleigh | Static | Single cell |
| FL ISAC BF | P49 | 6+6 | - | - | N/A | Rician(3) | Static | 500m cell |
| Robust FL CE | P10 | CNN input | mmWave | - | 612 | MATLAB 5G | - | 10 SBS + 1 MBS |
| Distributed SP | P92 | 128-1024 | - | 80 | 192 | ELAA cluster | - | Framework |
| Decentralized CE | P97 | 128 ULA (4 sub) | **28** | 1.6 GHz | 16 | NF + dual-WB | Static | Synthetic |
| ISAC NF ELAA | P99 | 256/128 ULA | 100/**28** | 0.1 GHz | 64 | NF spherical | Static | Synthetic |
| **Ours (CE-skip)** | - | **8x8/16x16/32x32 UPA** | **3.5/15/28** | **-** | **256-4096** | **Sionna RT** | **0-33 m/s** | **Munich UMi, 8 BS** |

*Static but notes "CIR support temporally stable"

---

### CE-Skip Gap Analysis

114편 분석 결과, 아래의 조합을 수행한 논문은 **단 한 편도 없음** — 이것이 CE-skip paper의 contribution.

#### Gap 1: Temporal CE Scheduling (When to Run CE)
- **모든 ELAA CE 논문 (L6)이 static channel 가정.** Temporal channel evolution을 고려한 CE 방법은 D-STiCE [P90 survey에서 인용]뿐이며, 그것도 CE "scheduling"이 아닌 CE "prediction."
- 기존 연구는 **how** to estimate (더 좋은 CE 알고리즘)에 집중. **When** to estimate (CE 수행 시점 결정)을 다룬 논문 없음.

#### Gap 2: CE as Schedulable Software Task
- GPU-native RAN 논문 (L3)은 플랫폼 아키텍처를 제공하지만, CE 커널의 adaptive scheduling을 구현하지 않음.
- dApps [P02]는 CE를 use case로 언급하지만 구체적 scheduling algorithm 없음.
- XAI-on-RAN [P07]은 GPU utilization 측정하지만 CE kernel 수준 scheduling 없음.

#### Gap 3: Event-Triggered CE (Not Periodic)
- Beam management의 prediction-aided substitution [P35, P56]은 beam domain에서 유사한 철학이나, CE domain에 적용한 사례 없음.
- Event-triggered inference [P29]는 image classification에 적용; PHY-layer CE에 적용한 사례 없음.

#### Gap 4: Multi-Tier Adaptive CE
- 기존 CE는 단일 method 고정 (LS 또는 LMMSE 또는 DL-CE). 상황에 따라 CE method를 전환하는 multi-tier 접근 없음.
- CE-skip의 3-tier (Skip / Lite-LS / Full DL-CE)는 새로운 조합.

#### Gap 5: Stale CSI에 대한 BF 영향의 CE Scheduling 관점 분석
- Robust BF [P38]는 CSI error 모델링을 하지만 CE scheduling과 연결하지 않음.
- CE skip으로 인한 specific CSI aging pattern (not random error, but temporally correlated staleness)을 BF 성능과 연결한 분석 없음.

#### Our Contribution (CE-skip이 채우는 공백):
1. **When Not to Estimate**: 최초로 CE inference scheduling 문제를 공식화
2. **3-Tier Adaptive CE**: Skip / Lite / Full의 event-triggered 전환
3. **PSA Monitor**: Lightweight temporal stationarity detector (<<450us budget)
4. **GPU-native 배포**: dApp framework 내 CE scheduler 설계
5. **Stale CSI → BF Rate 분석**: Skip pattern이 SE에 미치는 영향 정량화
6. **Multi-frequency/Multi-antenna 검증**: 3.5/15/28 GHz, 64-1024 antenna elements
