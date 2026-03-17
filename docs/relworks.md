# Related Works: 6G O-RAN + On-device AI for Collaborative Base Stations

> 105편 논문 목록 및 요약 (2023–2026.03). CE-skip 관련 41편 cross-reference (Cluster L). 연구 갭 분석/연구 방향 제안은 `research_and_experiments.md` §9-12 참조.

---

## Table of Contents
1. [논문 군집 (Clusters) 및 스토리라인](#1-논문-군집-및-스토리라인)
2. [논문 요약표](#2-논문-요약표)
3. [Cluster A: O-RAN / AI-RAN 아키텍처](#cluster-a-o-ran--ai-ran-아키텍처)
4. [Cluster B: Edge AI 추론 최적화](#cluster-b-edge-ai-추론-최적화)
5. [Cluster C: LLM/SLM for 6G Networks](#cluster-c-llmslm-for-6g-networks)
6. [Cluster D: 협력/분할 추론 (Split & Collaborative Inference)](#cluster-d-협력분할-추론)
7. [Cluster E: 빔 관리 & 채널 추정 (Beam Management & Channel Estimation)](#cluster-e-빔-관리--채널-추정)
8. [Cluster F: 시뮬레이션 / 데이터셋 / 디지털 트윈](#cluster-f-시뮬레이션--데이터셋--디지털-트윈)
9. [추가 발굴 논문](#4-추가-발굴-논문-18편-웹서치-기반)
10. [UE Feature Vector / Representation 추출 관련 연구](#5-ue-feature-vector--representation-추출-관련-연구)
11. [Site-Adaptive Channel Estimation & Attention in FL](#6-site-adaptive-channel-estimation--attention-in-fl-8편)
12. [Cluster J: 6G ELAA / XL-MIMO 채널 추정](#cluster-j-6g-elaa--xl-mimo-채널-추정)
13. [Cluster K: Differentiable Ray Tracing & Physics-Informed Optimization](#cluster-k-differentiable-ray-tracing--physics-informed-optimization)
14. [Cluster L: CE-Skip — Adaptive CE Inference Scheduling](#cluster-l-ce-skip--adaptive-ce-inference-scheduling-related-works)

---

## 1. 논문 군집 및 스토리라인

```
Cluster A: O-RAN / AI-RAN 아키텍처 (11편)
  ├─ A1: AI-RAN 융합 아키텍처 (Polese, Chatzistefanidis, Li-Toshiba)
  ├─ A2: O-RAN Native AI & 표준화 (Feng, Li-ChinaMobile, Basaran)
  ├─ A3: O-RAN 위의 FL/분산학습 (Abdisarabshali, Fang, Bensalem)
  └─ A4: dApp/xApp 실시간 제어 (Lacava, Han)

Cluster B: Edge AI 추론 최적화 (8편)
  ├─ B1: On-device AI 서베이 (Wang-survey, Wang-cognitive)
  ├─ B2: 6G Edge LAM (Wang-HKUST, Lyu, Yao)
  └─ B3: Edge GenAI 배포 (Nezami, Vadlamani-MIWEN)

Cluster C: LLM/SLM for 6G (6편)
  ├─ C1: LLM PHY-layer 다중과제 (Zheng, Mehmood)
  ├─ C2: Edge LLM 배포 (Lin, Ferrag)
  └─ C3: Tiny LLM / IoT (Kandala, Chen-IoT)

Cluster D: 협력/분할 추론 (7편)
  ├─ D1: LLM Split Inference (Chen-adaptive, Younesi-Splitwise)
  ├─ D2: CNN/DNN 분할 추론 (Wang-Hecofer, Fang-MAE)
  ├─ D3: 이벤트 기반 협력 추론 (Zhou)
  └─ D4: SL/SLIDE (Lin-SL, Qu-SLIDE)

Cluster E: 빔 관리 & 채널 추정 (13편)
  ├─ E1: 빔 관리 서베이 & 기초 (Xue, Bjornson)
  ├─ E2: DL 기반 빔 관리 (Heng-SSBA, Oliveira-DeepBT, Mattick-MAB, Bian-VBS)
  ├─ E3: 빔포밍 & 채널 추정 (Liang-GNN, New-FAS, Chae-vRAN-SIMD)
  └─ E4: HW 이질성 & 일반화 (Zeulin)

Cluster F: 시뮬레이션 & 데이터셋 (3편)
  ├─ F1: Sionna RT (NVIDIA)
  ├─ F2: DeepTelecom (Wang-ZJU)
  └─ F3: SpectrumFM Foundation Model (Zhou-Chae)

Cluster J: 6G ELAA / XL-MIMO 채널 추정 (11편)
  ├─ J1: 서베이 & 튜토리얼 (Lu, Long, Zeng, Xu-Larsson)
  ├─ J2: DL 기반 CE (XLCNet, Deep Unrolling, LLM4XCE)
  ├─ J3: Near-field 빔 훈련 (Ning-TMC)
  ├─ J4: 분산/구조적 CE (Decentralized BB, Sub-array)
  └─ J5: ISAC + ELAA (Wang-UESTC)

Cluster K: Differentiable RT & Physics-Informed Optimization (6편)
  ├─ K1: Diff RT 핵심 기법 (Implicit diff, Discontinuity smoothing)
  ├─ K2: Sim-to-Real 캘리브레이션 (VLM+DiffRT, RIS deployment)
  ├─ K3: Diff RT vs DL 비교 (13도시 실측)
  └─ K4: ELAA + Diff Radiomap (U6G Beam Map)

Cluster L: CE-Skip — Adaptive CE Inference Scheduling (41편 cross-ref)
  ├─ L1: CE Computational Cost & Motivation (10편)
  ├─ L2: Temporal Channel Persistence & Prediction (10편)
  ├─ L3: GPU-native RAN & dApp Architecture (9편)
  ├─ L4: Adaptive/Event-Triggered Inference (4편)
  ├─ L5: Beamforming with Stale/Imperfect CSI (4편)
  └─ L6: Near-Field ELAA CE Methods (7편)
```

**스토리라인 흐름:**
```
[F: 시뮬레이션/데이터] → [E: 빔/채널] → [B: Edge AI 최적화] → [D: 분할 추론]
         ↓                    ↓                    ↓                    ↓
     데이터셋 생성        PHY-layer AI          모델 압축/배포        BS-Edge 분할
         ↓                    ↓                    ↓                    ↓
                    [A: O-RAN 아키텍처] ←──── [C: LLM/SLM for 6G]
                    통합 플랫폼 & 표준화        네트워크 지능화
```

---

## Cluster A: O-RAN / AI-RAN 아키텍처

### A1: AI-RAN 융합 아키텍처

#### [P01] Beyond Connectivity: An Open Architecture for AI-RAN Convergence in 6G
- **저자**: Polese et al. (Northeastern University)
- **연도**: 2025, arXiv:2507.06911
- **핵심 기여**: O-RAN 확장 AI-RAN 아키텍처 제안. AI-SMO, AI-O-Cloud, AI-O2 인터페이스 도입. GPU MIG 파티셔닝으로 RAN + AI 워크로드 공존 시연.
- **방법론**: X5G 테스트베드에서 NVIDIA A100 GPU에 NVIDIA Aerial (PHY) + OAI + LLM(Ollama) + CNN/ResNet 동시 구동
- **주요 결과**: RAN 처리량 40-60 Mbps 유지하면서 LLM 배포 5.8-34.4초, GPU 40/20/10GB 분할
- **강점**: 실제 5G 테스트베드 검증, 기존 O-RAN 표준과의 호환성, 운영적 고려 (수익화, 에너지, 보안)
- **한계**: 단일 노드 실험만, 오케스트레이션 알고리즘 미구현
- **관련도**: ★★★★☆ — AI-for-RAN + AI-on-RAN 이중 패러다임, 협력 BS 인프라 수준에서 직접 관련

#### [P02] dApps: Enabling Real-Time AI-Based Open RAN Control
- **저자**: Lacava et al. (Northeastern/Sapienza/EURECOM)
- **연도**: 2025, Elsevier Computer Networks
- **핵심 기여**: O-RAN 최초 실시간(sub-10ms) AI 제어 루프 dApp 아키텍처. E3 인터페이스/E3AP 프로토콜 제안. CU/DU에 공존하는 경량 마이크로서비스.
- **방법론**: OAI 기반 구현, E3SM 서비스 모델, Colosseum/Arena 테스트베드. 스펙트럼 공유 + UE 위치추정 (CIR 기반) use case.
- **주요 결과**: 평균 제어 루프 지연 ~400μs (10ms 이하), 스펙트럼 공유 처리량 유지, CIR 기반 위치추정 sub-meter 정확도
- **강점**: xApp 불가능한 sub-ms 실시간 제어, 오픈소스 구현, 기존 O-RAN과 호환
- **한계**: 단순 휴리스틱 알고리즘만 시연, 다중 dApp 충돌 해결 미완
- **관련도**: ★★★★★ — **프로젝트의 CIR/CFR 추출과 직접 연결. on-device AI 추론의 O-RAN 구현 메커니즘.**

#### [P03] Large GenAI Models meet Open Networks for 6G
- **저자**: Li et al. (Toshiba Europe)
- **연도**: 2025, arXiv:2410.18790
- **핵심 기여**: API 기반 텔레콤 GAI 마켓플레이스 플랫폼. BEACON-5G 테스트베드에서 edge LLM vs cloud LLM 비교.
- **주요 결과**: Edge Llama 3.1 8B가 Cloud GPT-3.5 Turbo보다 낮은 TFT 달성
- **강점**: MNO 수익화 관점 제시, 실제 H100 GPU 테스트
- **한계**: 생성 AI 중심 (PHY 무관), 초 단위 지연 (실시간 RAN 부적합)
- **관련도**: ★★★☆☆ — Edge AI 배포 시연은 관련, 그러나 PHY-layer와는 간접적

#### [P04] MX-AI: Agentic Observability and Control Platform for Open and AI-RAN
- **저자**: Chatzistefanidis et al. (EURECOM/BubbleRAN/Khalifa)
- **연도**: 2025, arXiv:2508.09197
- **핵심 기여**: 최초 실제 5G O-RAN에서 LLM 멀티에이전트 제어 시스템. 5개 전문 에이전트 (Orchestrator, Monitoring, Deployment, Validator, Executor). RAG + push-based delta-aware watchers.
- **주요 결과**: GPT-4.1: 4.1/5.0 관찰 일관성, 100% 행동 정확도, 8.8초 지연. 로컬 70B: 3.8/5.0, 100%, 12-14초. 3B: 1.9/5.0, 100%, 1.3초.
- **강점**: 실제 라이브 O-RAN 시연, 다양한 LLM 비교, Pareto 분석
- **한계**: 비실시간 (1-14초), 제한된 제어 액션 (10개), 안전성 미해결
- **관련도**: ★★★★☆ — 비실시간 관리 수준이지만, near-RT RIC에 SLM 배포 방향이 on-device AI와 연결

### A2: O-RAN Native AI & 표준화

#### [P05] Towards 6G Native-AI Edge Networks
- **저자**: Feng et al. (Exeter/Sony/Southeast Univ/SUTD)
- **연도**: 2025, arXiv:2512.04405
- **핵심 기여**: Semantic Communication + Agentic Intelligence 통합 6G Native-AI 프레임워크. 3축 분류법: 시맨틱 추상화, 에이전트 자율성, RAN 제어 배치. 2-timescale 확률적 최적화 (수렴 증명 포함).
- **방법론**: 빠른 루프(MARL 에이전트), 느린 루프(시맨틱 인코더/디코더). Jetson/Xavier edge + A100 GPU + USRP SDR 테스트베드.
- **주요 결과**: 2-timescale 설계가 모든 SNR에서 최고 TSR, 대역폭 1.6-2.3x 효율, 지연 25-40% 감소
- **강점**: 최초 수렴 분석 (Proposition 1), O-RAN E2/A1/O1 매핑, 50+ 논문 체계적 분류
- **한계**: 대부분 기존 결과 재현, 확장성 미검증
- **관련도**: ★★★★★ — **on-device MARL + semantic comm의 O-RAN 통합. 프로젝트의 협력 BS 위치 설정에 이상적 프레임워크.**

#### [P06] Towards AI-Native RAN: An Operator's Perspective
- **저자**: Li et al. (China Mobile Research)
- **연도**: 2025, arXiv:2507.08403
- **핵심 기여**: 6G Day 1 AI-Native RAN 표준화 관점. AI Node + 6gNB 아키텍처. **5000+ BS 필드 트라이얼 (31개 도시, 중국)**.
- **주요 결과**: 짧은 영상 지연 25.6% 감소, QR 스캔 21.9% 감소, 에너지 절약 34.16%, 1000+ 앱 유형 >95% 정확도 분류
- **강점**: **대규모 실증 (5000+ BS)**, 배포 준비된 아키텍처, 표준화 로드맵 (3GPP/O-RAN/ETSI/ITU)
- **한계**: 멀티벤더 인터페이스 미표준화, on-device 학습 불가 (추론만)
- **관련도**: ★★★★★ — **중앙 학습 + 분산 추론 패러다임이 협력 BS와 직접 부합. 산업 관점 필수 레퍼런스.**

#### [P07] XAI-on-RAN: Explainable, AI-native, and GPU-Accelerated RAN
- **저자**: Basaran & Dressler (TU Berlin)
- **연도**: 2025, NeurIPS 2025 AI4NextG Workshop
- **핵심 기여**: O-RAN RIC에서 실시간 XAI. Attention + IG 하이브리드 기법으로 투명성-지연-GPU 활용률 균형.
- **주요 결과**: 하이브리드 8.1ms (near-RT 10ms 이내), 피델리티 +0.41 vs SHAP
- **관련도**: ★★★☆☆ — 설명가능성은 보조적 관심사이나, AI 신뢰성 측면에서 참고 가치

### A3: O-RAN 위의 FL/분산학습

#### [P08] Efficient Self-Learning and Model Versioning for AI-native O-RAN Edge
- **저자**: Bensalem et al. (TU Braunschweig / NTUST)
- **연도**: 2026, arXiv:2601.17534
- **핵심 기여**: O-RAN edge ML 모델 버전 관리 프레임워크. RL(Q-learning) 기반 업데이트 정책으로 정확도-안정성-지연 균형. dApp/xApp/rApp 3계층 제어 루프 대응.
- **주요 결과**: RL이 dApp 안정성 유지하면서 xApp/rApp 정확도 개선. 항상 업데이트 → 최고 정확도/최저 안정성, RL이 최적 균형.
- **한계**: 시뮬레이션만, Q-learning 확장성 제한
- **관련도**: ★★★★☆ — **협력 BS의 on-device 모델 관리 (업데이트 vs 안정성) 직접 해당**

#### [P09] Dynamic D2D-Assisted FL over O-RAN
- **저자**: Abdisarabshali et al. (UB-SUNY / Purdue)
- **연도**: 2024, arXiv:2404.06324
- **핵심 기여**: Dynamic wireless + dynamic datasets에서의 FL. DCLM 프레임워크: O-RAN xApp/rApp으로 전용 FL MAC 스케줄러. "Dynamic model drift" 개념 도입 (편미분 부등식).
- **방법론**: D2D 보조 계층적 FL (DPU→CHU→O-RU), 세밀한 시간 단위(FGTI) 리소스 할당
- **주요 결과**: 수렴 바운드 (Theorem 1) — 동적 무선 제어 결정, 데이터셋 역학, 사용자 선택, FL 정확도의 관계 명시
- **강점**: 최초 동적 무선채널 + 동적 데이터 FL, 엄밀한 이론 분석, O-RAN MAC 통합
- **한계**: 매우 복잡한 수학, 시뮬레이션만, 확정적 D-Event
- **관련도**: ★★★★☆ — **mmWave 채널 추정에서 시변 데이터 분포에 대한 FL 직접 적용 가능**

#### [P10] Robust FL for Wireless Channel Estimation
- **저자**: Fang et al. (RPTU Kaiserslautern / DFKI)
- **연도**: 2024, IEEE WCNC 2024
- **핵심 기여**: FL 기반 채널 추정의 적대적 공격 취약성 분석. StoMedian (BME + median) 및 LLPF (손실 기반 사전 필터링) 방어 기법.
- **방법론**: SBS→MBS FL 아키텍처, CNN 채널 추정, Reverse/Collusion/Outdate 공격 모드
- **주요 결과**: StoMedian이 공격 시 FedMedian 수준 방어 + 공격 없을 때 FedBE 수준 수렴
- **관련도**: ★★★★☆ — **SBS/MBS FL 아키텍처가 프로젝트의 협력 BS와 정확히 일치. 보안 관점 필수.**

### A4: E2E 지능화

#### [P11] Toward E2E Intelligence in 6G: AI Agent-Based RAN-CN Framework
- **저자**: Han et al. (Kyung Hee / ETRI / Ruhr Bochum)
- **연도**: 2026, arXiv:2602.23623
- **핵심 기여**: LLM + ReAct 패러다임의 RAN-CN 통합 지능 프레임워크. 이중 메모리 (단기+장기), MCP 기반 도구 호출.
- **주요 결과**: RSRP 추론 MAE=1.74, 위치 추론 MAE=0.49, E2E 슬라이싱 27.5 vs 26.6 사용자 만족
- **관련도**: ★★★★☆ — RAN-CN 교차 도메인 조정은 관련, 그러나 관리 평면 수준 (on-device 아님)

---

## Cluster B: Edge AI 추론 최적화

### B1: On-device AI 서베이

#### [P12] Empowering Edge Intelligence: A Comprehensive Survey on On-Device AI Models
- **저자**: Wang et al. (Beijing Normal / HKBU)
- **연도**: 2025, ACM Computing Surveys
- **핵심 기여**: On-device AI 파이프라인 종합 서베이 (데이터→모델→시스템). MobileNet, EfficientNet, Deep Compression 등 288개 레퍼런스 커버.
- **주요 결과**: MobileNetV3 75.2% ImageNet/219M MACs, Deep Compression 35-49x 압축, edgeBERT 7x 에너지 절약
- **한계**: 무선통신/6G/O-RAN 미다룸
- **관련도**: ★★☆☆☆ — 기초 레퍼런스 (on-device 기법 개요)

#### [P13] Cognitive Edge Computing: Optimizing Large Models for Pervasive Deployment
- **저자**: Wang et al. (Beijing Normal / HK PolyU)
- **연도**: 2025, arXiv:2501.03265
- **핵심 기여**: "Cognitive Edge Computing" 개념. LLM/SLM의 추론 능력 보존하면서 엣지 배포. INT2/INT3 양자화, MoE, speculative decoding 커버.
- **주요 결과**: Sub-10B 모델 on-device 가능, 10-100ms 로컬 추론 vs 100ms-2s 클라우드, pFL-SBPM 업링크 96.875% 절약
- **관련도**: ★★★☆☆ — Cloud-Edge 협력 패턴이 O-RAN CU/DU/RU 매핑에 유용

### B2: 6G Edge Large AI Models

#### [P14] Edge Large AI Models: Revolutionizing 6G Networks ⭐
- **저자**: Wang et al. (HKUST / ShanghaiTech)
- **연도**: 2025, arXiv:2505.00321
- **핵심 기여**: **6G edge LAM 종합 프레임워크: (1) Split FedFT + LoRA + Renyi DP 학습, (2) 마이크로서비스 기반 추론, (3) 에어인터페이스 적용 (채널 예측 + 빔포밍)**. Federated LAM 채널 예측 + Graph LAM 빔포밍.
- **방법론**: Split FedFT (LoRA 5% 파라미터만), QuaDRiGa 3GPP 채널, GNN precoder
- **주요 결과**: Federated LAM이 LLM4CP/GRU/RNN 대비 낮은 훈련 손실, Edge 서버 수 증가 시 성능 향상 후 안정화
- **강점**: 학습-추론-응용 전체 수명주기 커버, 채널 예측 + 빔포밍 직접 다룸
- **한계**: 매거진 수준 (~8p) 실험 깊이 제한, NMSE 미보고, O-RAN 미언급
- **관련도**: ★★★★★ — **프로젝트의 채널 예측 + 빔포밍 + FL과 정확히 일치하는 가장 직접적 논문**

#### [P15] The Larger the Merrier? Efficient LAIM Inference in Wireless Edge
- **저자**: Lyu et al. (KTH / CUHK-SZ / Paris-Saclay)
- **연도**: 2025, arXiv:2505.09214
- **핵심 기여**: LAIM pruning-aware co-inference의 최초 이론적 기반. Output distortion ≤ parameter distortion 증명. Rate-distortion 이론으로 pruning 하한 유도. Laplacian weight 분포 검증 (ResNet-152, BERT, BART, GPT-3).
- **주요 결과**: Pruning ratio > 0.5-0.7에서 신뢰할 수 있는 상한, 관절 최적화가 모든 베이스라인 능가
- **관련도**: ★★★★☆ — 무선 채널에서의 LAIM 압축 이론, 협력 BS fronthaul 용량 제한과 관련

#### [P16] Energy-Efficient Edge Inference in ISCC Networks
- **저자**: Yao et al. (Southeast Univ / SRIBD / HKU / CUHK-SZ)
- **연도**: 2025, arXiv:2503.00298
- **핵심 기여**: **최초 ISCC (통합 센싱-통신-컴퓨팅) 에너지 효율 edge AI 추론 프레임워크**. 센싱 품질이 정확도 상한 결정, pruning/quantization은 additive noise.
- **주요 결과**: 기존 대비 40% 에너지 절감, DNN 50% pruning 가능 (정확도 유지)
- **관련도**: ★★★★☆ — ISCC는 6G 핵심 패러다임, 센싱+통신+컴퓨팅 공동 최적화가 BS 수준 AI에 적용 가능

### B3: Edge GenAI 배포

#### [P17] Generative AI on the Edge: Architecture and Performance Evaluation
- **저자**: Nezami et al. (University of Leeds)
- **연도**: 2024, arXiv:2411.17712
- **핵심 기여**: **최초 O-RAN edge에서 LLM 벤치마크** — Raspberry Pi 5 클러스터 (K3s). 4-bit GGUF 양자화 LLM 8종 평가.
- **주요 결과**: 경량 모델 5-12 tokens/s, CPU/RAM 50% 미만, Yi(1.48B) 47초/InternLM(7.74B) 252초 지연
- **강점**: 실제 하드웨어 + O-RAN 맥락, 오픈소스 재현 가능
- **한계**: 대화 AI 태스크만 (무선 태스크 미검증), CPU 전용 → 느림
- **관련도**: ★★★★☆ — O-RAN edge 하드웨어에서의 AI 실현 가능성 기준선

#### [P18] Machine Intelligence on Wireless Edge Networks (MIWEN)
- **저자**: Vadlamani et al. (MIT / Duke)
- **연도**: 2025, arXiv:2506.12210
- **핵심 기여**: **패러다임 전환적 개념** — BS가 모델 가중치를 RF 파형으로 브로드캐스트, 클라이언트 RF 수신 체인(mixer)에서 직접 추론. 로컬 저장/ADC 변환 불필요.
- **방법론**: 주파수 다중화 인코딩, diode ring mixer로 내적 연산, 차별화 가능 RF 체인 모델 학습
- **주요 결과**: ENOB 10비트, MNIST 95% @ ~100 pJ, 추가 실리콘 불필요
- **강점**: 기존 RF 하드웨어만으로 zero-storage 추론, 에너지 초효율
- **한계**: MNIST/소형 네트워크만, 이상적 채널 가정, LayerNorm 에너지 지배적
- **관련도**: ★★★★☆ — 비전적 논문. BS→UE 모델 브로드캐스트는 6G native 개념, OFDM 호환

---

## Cluster C: LLM/SLM for 6G Networks

### C1: LLM PHY-layer 다중과제

#### [P19] Large Language Model Enabled Multi-Task Physical Layer Network ⭐
- **저자**: Zheng et al. (Tsinghua)
- **연도**: 2024, arXiv:2412.20772
- **핵심 기여**: **단일 LLAMA2-7B로 precoding + signal detection + channel prediction 동시 수행**. Multi-task instruction template + task-specific encoder/decoder + LoRA + LoftQ 양자화.
- **방법론**: MU-MISO-OFDM, 128 BS 안테나, QuaDRiGa 3GPP UMa NLOS, 4-bit 양자화
- **주요 결과**: 채널 예측: Transformer/RNN/LSTM/GRU 모두 능가. Precoding: near-optimal. 75% 저장 절감 (16→4bit) / 성능 저하 무시 가능.
- **강점**: 최초 다중 PHY 과제 통합 LLM, LoRA-aware 양자화, 확장 가능 (추가 과제)
- **한계**: 7B 모델은 on-device 어려움, 시뮬레이션만, 과제 간 의존성 미탐구
- **관련도**: ★★★★☆ — BS 측 PHY 연산 (DU)에 직접 적용 가능, 그러나 모델 크기 장벽

#### [P20] Bridging 6G IoT and AI: LLM-Based PHY Optimization
- **저자**: Mehmood et al. (NTNU / LUMS)
- **연도**: 2026, arXiv:2602.06819
- **핵심 기여**: PE-RTFV 프레임워크 — LLM 2개 (Optimizer + Agent)를 prompt engineering만으로 PHY 최적화 (재학습 없음). SWIPT IoT constellation 설계 시연.
- **주요 결과**: 15 반복으로 유전 알고리즘 수준 성능, 비선형 에너지 하베스팅 암묵적 학습
- **한계**: ChatGPT 5.2 (클라우드 의존), 단일 과제만, 지연 분석 없음
- **관련도**: ★★★☆☆ — 재학습 없는 PHY 최적화 개념은 흥미, 그러나 클라우드 LLM 의존

### C2: Edge LLM 배포

#### [P21] Pushing LLMs to the 6G Edge ⭐
- **저자**: Lin et al. (HKU)
- **연도**: 2023-2025, IEEE Communications Magazine
- **핵심 기여**: **End-edge cooperation 패러다임으로 6G MEC에 LLM 배포 비전**. SplitLoRA (최초 SL+LoRA), SplitMoE, SLM-LLM speculative decoding.
- **주요 결과**: SplitLoRA: GPT-2 medium 2.2-3.8시간 학습, 25.8% 컴퓨팅으로 LLM 수준 성능
- **강점**: ITU 6G 표준 정렬, SplitLoRA 원본 기여, 실행 가능한 오픈 문제 정리
- **한계**: 비전 논문 (실험 제한), O-RAN 미언급
- **관련도**: ★★★★★ — **end-edge 협력이 O-RAN CU/DU/RU에 직접 매핑. SL/split inference가 협력 BS 핵심 기술.**

#### [P22] How Small Can 6G Reason? Scaling Tiny Language Models ⭐
- **저자**: Ferrag et al. (UAE University / Khalifa)
- **연도**: 2026, arXiv:2603.02156
- **핵심 기여**: **최초 AI-native 6G용 소형 LM 스케일링 연구**. 6G-Bench (3GPP/O-RAN 정렬 30개 과제). 135M~7B 10개 모델 평가. Edge Score = 정확도/(지연×메모리).
- **주요 결과**: 1-1.5B에서 안정성 급변 (z=13.9), 3B 이후 수확 체감 (+0.064), 350M Edge Score 최고 (191x10⁴)
- **강점**: O-RAN Alliance 직접 정렬, 재현 가능 엄밀 방법론, 배포 가이드라인 (RAN/MEC/control plane 계층)
- **한계**: Zero-shot MCQ만, fine-tuning 미검증, 실제 edge HW 미배포
- **관련도**: ★★★★★ — **O-RAN 계층별 모델 크기 가이드라인. 1.5-3B가 near-RT RIC에 적합하다는 직접적 근거.**

### C3: Tiny LLM / IoT

#### [P23] TinyLLM: Training and Deploying Language Models at the Edge
- **저자**: Kandala et al. (NUS)
- **연도**: 2024, arXiv:2412.15304
- **핵심 기여**: 30-120M 파라미터 커스텀 모델을 도메인 특화 데이터로 사전학습, SBC에 배포. **소형 모델이 특정 과제에서 수B 모델 매칭** 증명.
- **주요 결과**: 124M 모델 87-98% 정확도 (Phi-2/3, Llama 대비), 70x 빠른 추론, 2GB RAM SBC 구동
- **관련도**: ★★★☆☆ — 도메인 특화 소형 모델 패러다임이 BS 수준 채널 AI에 적용 가능

#### [P24] LLM-Empowered IoT for 6G: Architecture, Challenges, and Solutions
- **저자**: Chen et al. (South China Univ / Pengcheng Lab)
- **연도**: 2025, IEEE IoT Magazine
- **핵심 기여**: "LLM for IoT" + "LLM on IoT" 이중 구조. **메모리 효율 SFL**: 서버가 단일 LLM 유지 + 순차적 LoRA 어댑터 로딩.
- **주요 결과**: 표준 SFL 대비 79% 메모리 절감, 6% 학습 시간 단축, 40% 빠른 수렴 (SL 대비)
- **강점**: 이종 디바이스 (Jetson Nano~Apple M3) 실제 실험
- **관련도**: ★★★★☆ — **이종 BS에서의 collaborative fine-tuning에 SFL 직접 적용 가능**

---

## Cluster D: 협력/분할 추론

### D1: LLM Split Inference

#### [P25] Adaptive Layer Splitting for Wireless LLM Inference in Edge Computing
- **저자**: Chen et al. (Zhejiang Univ)
- **연도**: 2024
- **핵심 기여**: MBRL로 LLM 최적 분할점 동적 결정. Reward surrogate model로 학습 시간 3000x 단축 (24일→7.7분).
- **방법론**: PPO + DNN reward surrogate, Nakagami-m 채널, LLaMA2-7B/13B, Mistral-7B 등
- **주요 결과**: 채널 노이즈 증가 시 분할점이 input에서 멀어짐 (직관 부합), PPO가 A2C/DQN 능가
- **관련도**: ★★★★☆ — 채널 적응형 분할이 O-RAN UE-Edge 시나리오에 직접 적용

#### [P26] Splitwise: Collaborative Edge-Cloud LLM via Lyapunov-Assisted DRL
- **저자**: Younesi et al. (Innsbruck / Sharif)
- **연도**: 2025, IEEE UCC '25
- **핵심 기여**: **Sub-layer 수준 LLM 파티셔닝** (attention head + FFN 블록). Lyapunov 안정성 보장 + PPO dual critics. 24층/16헤드 모델에서 10³¹ 가능한 구성.
- **주요 결과**: 1.4-2.8x 지연 감소, 41% 에너지 절약, <4% 정확도 저하, Edge 10GB (13B 모델)
- **강점**: Sub-layer 세분성, 이론적 안정성 보장, 실제 edge HW 검증 (Jetson, Galaxy S23, RPi5)
- **관련도**: ★★★★☆ — O-RAN disaggregated 아키텍처에 직접 매핑 (far edge/near edge/cloud)

### D2: CNN/DNN 분할 추론

#### [P27] CNN Collaborative Inference for Heterogeneous Edge Devices (Hecofer)
- **저자**: Wang et al.
- **연도**: 2024, Sensors (MDPI)
- **핵심 기여**: Hecofer: 이종 edge 디바이스 간 CNN 사전 파티셔닝. Micro-shifting 최적화 + 파이프라인 큐.
- **주요 결과**: VGG19 170% 속도향상 (7 디바이스), ResNet50 124.6% (6 디바이스)
- **한계**: CNN만, 유선 LAN 100Mbps (무선 미고려), 정확도 영향 미분석
- **관련도**: ★★☆☆☆ — 이종 디바이스 협력 개념만 관련, 무선/O-RAN 무관

#### [P28] MAE: Collaborative Inference with Efficient DNN Partitioning
- **저자**: Fang et al.
- **연도**: 2025, Elsevier Computer Networks
- **핵심 기여**: MoE 패러다임을 CNN 채널에 적용한 sparse expert 기반 분할 추론. 고정 파티션 포인트 (post-conv1) + 룩업 테이블.
- **주요 결과**: VGG16 45.7% 지연 감소, InceptionNet 69.4% 감소, <2% 정확도 저하
- **관련도**: ★★★☆☆ — MoE 기반 효율은 대역폭 제한 무선 edge에 유용하나 무선 미고려

### D3: 이벤트 기반 협력 추론

#### [P29] Communication Efficient Cooperative Edge AI via Event-Triggered Offloading
- **저자**: Zhou et al. (CUHK / HKU)
- **연도**: 2025, arXiv:2501.02001
- **핵심 기여**: 채널 적응 이벤트 트리거 edge 추론. **이중 임계값 다중 출구 아키텍처**: 로컬에서 희귀 이벤트 조기 감지, 복잡한 경우 edge 오프로딩.
- **방법론**: ShuffleNetV2/MobileNetV2 로컬 + ResNet50 서버, Lipschitz 연속성 증명, 가속 경사 하강
- **주요 결과**: 이중 임계값이 단일 임계값/로컬 전용 능가, 불균형 데이터 (9:1)에서 더 큰 이점
- **관련도**: ★★★★☆ — 6G 미션크리티컬 앱 + 채널 적응 오프로딩이 mmWave 환경에 적합

### D4: Split Learning & SLIDE

#### [P30] Split Learning in 6G Edge Networks ⭐
- **저자**: Lin et al. (HKU / Tsinghua)
- **연도**: 2024, arXiv:2306.12194
- **핵심 기여**: **6G 무선 edge에서의 Split Learning 종합 아키텍처**. Split Edge Learning (SEL): 서버가 주요 훈련, 디바이스 데이터 프라이버시 보존. Multi-edge 협력, 모델 배치/마이그레이션.
- **주요 결과**: Inter-server 협력 SEL: 15 라운드에 90% (비협력 22, 로컬 35), 클라이언트 선택+cut layer 최적화 30-40% 수렴 개선
- **강점**: 6G edge 전용, single-edge~multi-edge 전체 스펙트럼 커버, 모빌리티 지원
- **관련도**: ★★★★★ — **Multi-edge SEL = 협력 BS 패러다임. 모델 배치/마이그레이션이 O-RAN 리소스 관리와 직접 대응.**

#### [P31] SLIDE: Simultaneous Model Downloading and Inference ⭐
- **저자**: Qu et al. (HKU / Tsinghua)
- **연도**: 2026, arXiv:2512.20946
- **핵심 기여**: **모델 다운로드와 추론을 동시 수행** (기존 Download-and-Inference 대비). 다중 사용자 OFDMA에서 모델 제공 + 대역 할당 + 컴퓨팅 자원 공동 최적화. **다항 시간 최적 알고리즘**.
- **주요 결과**: 기존 DAI 대비 32.5% E2E 지연 감소, 디스크 로딩 대비 0.2x만 느림, 40-60% 더 많은 사용자 서비스
- **강점**: CNN/ViT/LLM/RNN 지원, Jetson Orin 검증, 최적성 증명
- **한계**: 단일 BS, 캐시 미스 미고려
- **관련도**: ★★★★★ — **BS→UE 모델 전달의 핵심 메커니즘. 협력 BS가 다중 사용자에게 모델 전달하는 시나리오에 직접 적용.**

---

## Cluster E: 빔 관리 & 채널 추정

### E1: 빔 관리 서베이 & 기초

#### [P32] A Survey of Beam Management for mmWave and THz Communications Towards 6G ⭐
- **저자**: Xue et al.
- **연도**: 2023-2024, IEEE Communications Surveys & Tutorials
- **핵심 기여**: mmWave + THz 빔 관리 최초 종합 서베이. **AI (DL/RL/DRL/FL/TL), RIS, ISAC** 3대 기술 축 통합. **다중 에이전트 협력 빔 관리 (FL/TL/split learning)** 최초 리뷰.
- **관련도**: ★★★★★ — **협력 BS 빔 관리의 레퍼런스 프레임워크. FL 기반 다중 BS 빔 관리가 프로젝트 핵심.**

#### [P33] Towards 6G MIMO: Massive Spatial Multiplexing, Dense Arrays
- **저자**: Bjornson, **Chae**, Heath, Marzetta et al.
- **연도**: 2024, arXiv:2401.02844
- **핵심 기여**: 6G UM-MIMO 튜토리얼. Near-field beamfocusing, 공간 DoF, 채널 추정 (LS/MMSE/RS-LS/OMP), EM/회로 이론 통합.
- **주요 결과**: 5000안테나 @ 30GHz → Fraunhofer 250m, 1000 UE 다중화 가능, half-λ 간격이 모든 공간 DoF 캡처
- **관련도**: ★★★★☆ — 프로젝트의 물리 계층 기초. 채인 교수(연세대) 공저. 대규모 다중화가 지능형 빔 관리 필요성의 근거.

### E2: DL 기반 빔 관리

#### [P34] Site-Specific Beam Alignment in 6G via Deep Learning ⭐
- **저자**: Heng et al. (UT Austin / ASU)
- **연도**: 2024, IEEE Communications Magazine
- **핵심 기여**: **Site-Specific Beam Alignment (SSBA)**: 셀 별로 probing codebook + beam selection DNN 공동 최적화. Ray-tracing 기반 학습 → 디지털 트윈 파이프라인.
- **주요 결과**: 8 측정만으로 genie 대비 1dB 이내 (32x 탐색 감소), site-specific이 site-agnostic 대비 3dB 이점
- **강점**: O-RAN xApp 자연 적합, 디지털 트윈 파이프라인 제안
- **관련도**: ★★★★★ — **O-RAN on-device AI의 가장 자연스러운 적용. non-RT RIC(학습) + near-RT RIC(배포) 매핑. 다중 셀 FL이 협력 BS와 직접 연결.**

#### [P35] DL-Based Beam Management for mmWave Vehicular (DeepBT)
- **저자**: Oliveira et al.
- **연도**: 2025, arXiv:2511.02260
- **핵심 기여**: LSTM 기반 빔 추적 (DeepBT-C/R) + autoregressive 추론으로 66.7% 측정 오버헤드 감소. 50% NLOS 환경에서도 강건.
- **주요 결과**: Top-10 정확도 ~99% (LOS), R50% NLOS에서도 유지. 모델 2.04MB/입력 1.25KB.
- **관련도**: ★★★★☆ — 초경량 모델 (2MB)이 BS/UE on-device 배포에 이상적. 협력 BS V2I 관련.

#### [P36] Meta-Learning MAB Beam Tracking ⭐
- **저자**: Mattick et al. (Fraunhofer IIS / Ruhr Bochum)
- **연도**: 2025, arXiv:2512.05680
- **핵심 기여**: **빔 추적을 Restless MAB/POMDP로 정형화**. 소형 확률적 NN으로 RSS 피드백만 사용 online 빔 선택. 위치/3D 모델/ray-tracing 불필요.
- **주요 결과**: 단일 빔 측정으로 최대 RSS의 74-75%, GP 대비 975x 효율적 추론, O(1)/timestep
- **강점**: **초경량 on-device 추론**, 환경 변화에 강건, 불확실성 정량화 (Bayesian)
- **관련도**: ★★★★★ — **On-device AI 빔 추적의 이상적 사례. O-RAN near-RT RIC 배포 직접 가능. 4 BS 설정이 협력 BS와 연결.**

#### [P37] Multi-modal Virtual BS for MIMO Beam Alignment (VBS)
- **저자**: Bian et al. (HKUST / Southeast Univ)
- **연도**: 2026, arXiv:2602.22796
- **핵심 기여**: 3D LiDAR + BS 위치로 Virtual BS (반사면의 거울상) 구축. VBS 기반 거친 채널 복원 → Top-S 부분 빔 훈련. ML 학습 없이 기하학적 방법.
- **주요 결과**: Top-5 부분 훈련으로 최적의 98% SE 달성, 탐색 오버헤드 수천→5로 감소
- **강점**: 물리 기반/해석 가능, LoS+NLoS 모두 지원, **Sionna ray-tracing 사용** (40 GHz)
- **관련도**: ★★★★☆ — Sionna 사용, 기하학 기반 빔 정렬이 협력 BS 환경 지식 공유에 보완적

### E3: 빔포밍 & 채널 추정

#### [P38] Data and Model-Driven DL Beamforming (Chae)
- **저자**: Liang et al. (incl. **Chae**)
- **연도**: 2024, arXiv:2406.03098
- **핵심 기여**: 비지도학습 + model-driven 강건 빔포밍. Bipartite GNN (BGNN)으로 안테나-사용자 그래프 추론. Modified optimal beamforming structure (3K 출력으로 차원 축소).
- **주요 결과**: BTI 대비 14% 높은 rate, **1000x 빠른 실행 (10-15ms)**, N=6→7,8,9 일반화
- **강점**: 성능 + 속도 동시 달성 (드묾), GNN 확장성, 채인 교수 공저
- **관련도**: ★★★★☆ — 10-15ms 추론이 near-RT RIC on-device에 적합. 채널 불확실성 하 강건 빔포밍.

#### [P39] Channel Estimation in Fluid Antenna System (FAS)
- **저자**: New et al. (incl. **Chae**)
- **연도**: 2025, IEEE TWC
- **핵심 기여**: FAS 채널 복원에 oversampling 필수 증명. Half-wavelength 불충분 (spectral leakage). 불완전 CSI FAS > 완전 CSI TAS.
- **관련도**: ★★☆☆☆ — FAS 특화, 채널 샘플링 기본 원리는 참고

#### [P40] Accelerating vRAN and O-RAN with SIMD (Chae)
- **저자**: Park, **Chae**, Heath (UCSD / Yonsei)
- **연도**: 2025, arXiv:2510.07843
- **핵심 기여**: SIMD (AVX2)로 vRAN PHY-layer 가속. LMMSE MIMO 검출 50% 속도향상. 4x4 MIMO 검출 TTI의 3%만 사용 → AI 워크로드 여유.
- **주요 결과**: 4x4 MIMO 0.03ms (1ms TTI의 3%), 139.4-279 Mbps 처리량
- **강점**: **PHY 가속으로 on-device AI를 위한 컴퓨팅 헤드룸 확보**, 채인 교수/EIS Lab 직접 관련
- **관련도**: ★★★★☆ — **SIMD PHY 가속이 동일 COTS 서버에서 AI 추론 가능하게 하는 실용적 enabler**

### E4: HW 이질성 & 일반화

#### [P41] Rethinking Beam Management: Generalization Under HW Heterogeneity ⭐
- **저자**: Zeulin et al. (Tampere / UCSD)
- **연도**: 2026, arXiv:2602.18151
- **핵심 기여**: **HW 이질성 (안테나 구성, 코드북, 컴퓨팅)이 ML 빔 관리의 일반화를 근본적으로 제한**함을 주장/입증. **15 GHz** (프로젝트와 동일 주파수).
- **주요 결과**: 안테나 미스매치 → 90th %ile SE 50%+ 저하 (기존 HS/ES보다 나쁨), 코드북 미스매치 → 높은 발산
- **강점**: 간과된 핵심 문제 식별, O-RAN 다벤더 배포에서 실용적 분류법
- **한계**: 문제 식별 위주 (구체적 해법 미구현)
- **관련도**: ★★★★★ — **on-device AI 빔 관리의 근본 한계. 15 GHz 동일 주파수. O-RAN 다벤더 환경의 핵심 도전.**

#### [P42] DL Beam Management for mmWave Vehicular (DeepBT) — [P35와 동일]

---

## Cluster F: 시뮬레이션 & 데이터셋

#### [P43] Sionna RT Technical Report
- **저자**: Ait Aoudia et al. (NVIDIA)
- **연도**: 2025, arXiv:2504.21719
- **핵심 기여**: Sionna RT 공식 기술 보고서. GPU 가속 ray tracer, **완전 미분가능**, SBR + Image Method, 중복 제거 해싱, Fibonacci lattice (10⁶ rays), TF/PyTorch 호환.
- **주요 결과**: Importance sampling 100x 효율, linear scaling, synthetic array 대규모 안테나 지원
- **관련도**: ★★★★☆ — **프로젝트의 핵심 도구. 미분가능성 → on-device 학습/fine-tuning 활용 가능.**

#### [P44] DeepTelecom: Digital-Twin DL Dataset
- **저자**: Wang et al. (Zhejiang Univ / Khalifa)
- **연도**: 2025, arXiv:2508.14507
- **핵심 기여**: LLM 보조 LoD3 씬 모델링 + Sionna ray-tracing. 멀티모달 출력 (CIR/CFR/AoA/AoD/coverage map/video). RIS 지원.
- **강점**: LLM 기반 material annotation 자동화, 다양한 시나리오
- **한계**: 데이터셋 미완전 공개, 실측 검증 없음
- **관련도**: ★★★★☆ — 프로젝트와 유사한 Sionna 기반 파이프라인. 벤치마크/보완 데이터셋 가능성.

#### [P45] SpectrumFM: Foundation Model for Spectrum Management (Chae)
- **저자**: Zhou et al. (incl. **Chae**)
- **연도**: 2025, arXiv:2505.06256
- **핵심 기여**: 최초 스펙트럼 관리 파운데이션 모델. CNN + multi-head self-attention 하이브리드 인코더. Self-supervised (masked reconstruction + next-slot prediction) → AMC/WTC/SS/AD 다운스트림.
- **주요 결과**: AMC F1 73.46% (베이스라인 +2.65-9.61%p), SS AUC 0.97 @ -4dB, 저 SNR에서 강건
- **관련도**: ★★★☆☆ — 파운데이션 모델 패러다임의 무선 적용. O-RAN 스펙트럼 관리에 적용 가능. 채널/빔 도메인 확장 방향.

#### [P46] Training ML at the Edge: A Survey
- **저자**: Khouas et al. (Deakin / TII)
- **연도**: 2024, arXiv:2403.02619
- **핵심 기여**: Edge 학습 종합 서베이 (803편). FL/SL/swarm/gossip, transfer/incremental/meta, KD/quantization/pruning, BNN/SNN/forward-forward. 6개 메트릭 비교 프레임워크.
- **주요 결과**: FL이 edge 학습 지배적, SL 두 번째 성장세, 단일 기법으로 모든 요구 충족 불가 → 결합 필요
- **관련도**: ★★★☆☆ — Edge 학습 기법 선택 가이드. FL+SL 결합이 협력 BS에 적합하다는 근거.

---

## 2. 논문 요약표

| # | 논문 | 군집 | 관련도 | 연도 |
|---|------|------|-------|------|
| P02 | dApps (Lacava) | A1 | ★★★★★ | 2025 |
| P05 | 6G Native-AI Edge (Feng) | A2 | ★★★★★ | 2025 |
| P06 | AI-Native RAN Operator (Li-CM) | A2 | ★★★★★ | 2025 |
| P14 | Edge LAM 6G (Wang-HKUST) | B2 | ★★★★★ | 2025 |
| P19 | Multi-task PHY LLM (Zheng) | C1 | ★★★★☆ | 2024 |
| P21 | Pushing LLMs to 6G Edge (Lin) | C2 | ★★★★★ | 2025 |
| P22 | How Small Can 6G Reason (Ferrag) | C2 | ★★★★★ | 2026 |
| P30 | Split Learning 6G (Lin) | D4 | ★★★★★ | 2024 |
| P31 | SLIDE (Qu) | D4 | ★★★★★ | 2026 |
| P32 | Beam Mgmt Survey (Xue) | E1 | ★★★★★ | 2024 |
| P34 | SSBA (Heng) | E2 | ★★★★★ | 2024 |
| P36 | MAB Beam Tracking (Mattick) | E2 | ★★★★★ | 2025 |
| P41 | HW Heterogeneity (Zeulin) | E4 | ★★★★★ | 2026 |

> ★★★★★ = 13편, ★★★★☆ = 17편, ★★★☆☆ = 10편, ★★☆☆☆ = 5편, ★☆☆☆☆ = 1편

---

*Initial analysis: 2026-03-04 | 46 papers analyzed by 8 parallel agents*

---

## 4. 추가 발굴 논문 (18편, 웹서치 기반)

> 5개 병렬 검색 에이전트로 각 후보군 관련 누락 논문 탐색. arXiv 다운로드 및 텍스트 추출 완료.

### Cluster G1: CoMP + AI Model Delivery / Multi-BS 협력 추론

#### [P47] Fine-Grained AI Model Caching and Downloading With CoMP Broadcasting
- **저자**: Yang Fu, Peng Qin, et al.
- **연도**: 2025, arXiv:2509.19341, IEEE TWC
- **핵심 기여**: CoMP 브로드캐스팅으로 **BS→UE** AI 모델 전달. 모델을 Parameter Block(PB) 단위로 분해, BS 간 PB 마이그레이션 후 CoMP로 사용자에게 전송. MADRL(MAASN-DA) 기반 캐싱/전달 최적화.
- **주요 결과**: 29.74-67.86% 지연 감소 (vs baselines), Llama2-7B/13B 확장 실험
- **한계**: ⚠️ **BS→UE 방향만 다룸** (UE 측 inference). BS-side 추론이나 BS 간 협력 추론 없음. O-RAN 아키텍처 매핑 없음. Split inference 미고려.
- **관련도**: ★★★★☆ — 후보 #3의 직접적 선행연구이나, 문제 방향이 다름 (UE-side vs BS-side)

#### [P48] Collaborative Edge AI Inference over Cloud-RAN
- **저자**: Pengfei Zhang, et al.
- **연도**: 2024, arXiv:2404.06007, IEEE Trans. Commun.
- **핵심 기여**: Cloud-RAN에서 분산 디바이스 → RRH → 중앙으로 특징 벡터 Over-the-Air 집계. AirComp 기반 다중 RRH 협력 추론.
- **한계**: RRH는 relay 역할만, BS-side 자체 추론 없음. 빔 관리나 채널 예측 미다룸.
- **관련도**: ★★★☆☆

### Cluster G2: ISAC + FL + Beamforming/Channel

#### [P49] FL Strategies for Coordinated Beamforming in Multicell ISAC
- **저자**: Lai Jiang, et al. (UCL)
- **연도**: 2025, arXiv:2501.16951
- **핵심 기여**: 다중 셀 ISAC에서 VFL/HFL 기반 협력 빔포밍. VFL은 중앙 서버 기반, HFL은 완전 분산. 간섭 누출 기반 손실함수로 로컬 CSI만으로 학습 가능.
- **주요 결과**: 3BS×6안테나, M=2 UE/BS 시나리오에서 최적 빔포밍에 근접
- **한계**: ⚠️ **채널 예측 없음** — 현재 CSI→빔포밍 매핑만 (static snapshot). 합성 Rician 채널 사용, ray-tracing 없음. 시계열 모델링 없음.
- **관련도**: ★★★★☆ — FL+ISAC 결합 가장 근접 연구이나, channel forecasting 아닌 beamforming optimization

#### [P50] Personalized FL-Driven Beamforming for ISAC
- **저자**: Zhou Ni, et al.
- **연도**: 2025, arXiv:2510.06709, IEEE CCNC 2026
- **핵심 기여**: EM 기반 Personalized FL로 BS별 적응적 모델 가중치. 통신/센싱 트레이드오프 BS별 최적화.
- **주요 결과**: FedAvg/FedPer 대비 우수, MATLAB ray-tracing 활용
- **한계**: P49와 동일 — 채널 예측 없음, snapshot 빔포밍만
- **관련도**: ★★★★☆

#### [P51] FL with Integrated Sensing, Communication, and Computation
- **저자**: Yipeng Liang, et al.
- **연도**: 2024, arXiv:2409.11240
- **핵심 기여**: FL+ISCC 통합 프레임워크. 센싱 노이즈 + OTA 집계 오류가 FL 수렴에 미치는 영향 분석.
- **한계**: 이론적 프레임워크 위주, 채널 예측 구체적 적용 없음
- **관련도**: ★★★☆☆

#### [P52] Coalition Formation for Heterogeneous FL Channel Estimation
- **저자**: Nan Qi, et al.
- **연도**: 2025, arXiv:2502.05538
- **핵심 기여**: RIS 보조 cell-free MIMO에서 이질적 FL 채널 추정. 분산 DRL 기반 coalition 형성.
- **관련도**: ★★★☆☆

#### [P53] Sensing-Aided Beam Prediction with Transfer Learning
- **저자**: Yuan Feng, et al.
- **연도**: 2024, arXiv:2405.15339
- **핵심 기여**: 환경 센싱 데이터로 빔 예측, transfer learning으로 30% 라벨 데이터만으로 적응
- **관련도**: ★★★☆☆

### Cluster G3: HW 이질성 + 빔 관리 + FL

#### [P54] ProtoBeam: Generalizing Beam Prediction to Unseen Antennas
- **저자**: Mashaal et al. (Calgary)
- **연도**: 2025, arXiv:2501.03435
- **핵심 기여**: Prototypical Networks로 안테나 HW 이질성 극복. 미학습 안테나 16-shot에서 74.11% 정확도 (398% 향상).
- **한계**: ⚠️ **단일 BS, FL 없음, O-RAN 없음**. 60 GHz 단일 링크, 안테나 RF 이질성만 다룸 (컴퓨팅 이질성 미다룸).
- **관련도**: ★★★★☆ — HW 이질성 문제 해결 시도하나 scope가 좁음

#### [P55] Elastic FL over O-RAN Architecture
- **저자**: Abdisarabshali et al. (Buffalo)
- **연도**: 2025, arXiv:2305.02109, IEEE IoT Magazine
- **핵심 기여**: O-RAN에서 다중 FL 서비스 동시 실행 아키텍처. 3단계: non-RT RIC(eApp), near-RT RIC(FL Controller), O-DU(MAC Scheduler). 150 차량 UE, Porto 실제 궤적.
- **한계**: ⚠️ **빔 관리 미적용** — CIFAR-10/MNIST 등 일반 분류만. 모델/HW 이질성 미고려.
- **관련도**: ★★★★☆ — O-RAN+FL 인프라는 좋으나, 빔 관리에 적용 안 됨

#### [P56] 5G-Advanced AI/ML Beam Management (3GPP 관점)
- **저자**: Jayaweera et al.
- **연도**: 2024, arXiv:2404.15326
- **핵심 기여**: 3GPP 5G-Advanced AI/ML 빔 관리 성능 평가. ML 통합 모델 기반.
- **관련도**: ★★★☆☆

#### [P57] CRKD: Resource-Efficient Beam Prediction via Knowledge Distillation
- **저자**: Park et al.
- **연도**: 2025, arXiv:2504.05187, IEEE Trans. Mobile Computing
- **핵심 기여**: 멀티모달 교사→경량 학생 모델 지식 증류. 교사 모델 10% 파라미터로 빔 예측.
- **관련도**: ★★★☆☆

### Cluster G4: Foundation Models + Differentiable RT

#### [P58] LWM: Large Wireless Model
- **저자**: Alikhani, Charan, Alkhateeb
- **연도**: 2024, arXiv:2411.08872
- **핵심 기여**: 최초 무선 채널 Foundation Model. Masked Channel Modeling으로 self-supervised 사전학습. DeepMIMO 15개 시나리오, 100만+ 채널 샘플.
- **관련도**: ★★★★☆ — 후보 #2 (PHY FM)의 직접 경쟁자

#### [P59] WiFo: Wireless Foundation Model for Channel Prediction
- **저자**: Liu et al.
- **연도**: 2024, arXiv:2412.08908, Science China Info Sciences
- **핵심 기여**: STF(Space-Time-Frequency) 무선 FM. MAE 아키텍처, 160K 샘플, 16개 CSI 구성에서 zero-shot 일반화.
- **관련도**: ★★★★☆ — 후보 #2 직접 경쟁자

#### [P60] WirelessGPT: Multi-Task FM for Wireless
- **저자**: Yang et al.
- **연도**: 2025, arXiv:2502.06877
- **핵심 기여**: 80M GPT 스타일 FM. Traciverse(300GB, 27도시) + SionnaRT + DeepMIMO 사전학습. 채널 추정, 빔 예측, 신호 검출, 센싱 통합.
- **관련도**: ★★★★☆

#### [P61] Learning Radio Environments by Differentiable Ray Tracing
- **저자**: Hoydis et al. (NVIDIA)
- **연도**: 2024, arXiv:2311.18558, IEEE TMLCN
- **핵심 기여**: Sionna RT의 differentiable ray tracing 캘리브레이션. 재질, 산란, 안테나 패턴을 역전파로 학습. Sim-to-real gap 해소.
- **관련도**: ★★★★☆ — 후보 #4 (DT self-evolving)의 핵심 빌딩블록

### Cluster G5: AI LCM + On-device 최적화

#### [P62] AI/ML Lifecycle Management for Interoperable AI Native RAN
- **저자**: Huang, Wen, Li
- **연도**: 2025, arXiv:2507.18538
- **핵심 기여**: 3GPP Rel-16~20 AI/ML LCM 진화 정리. 모델 페어링, 활성화, 폴백, 버전 동기화 프로토콜.
- **관련도**: ★★★★☆

#### [P63] Compression of Site-Specific DNNs for MIMO Precoding
- **저자**: Kasalaee et al.
- **연도**: 2025, arXiv:2502.08758, IEEE ICMLCN 2025
- **핵심 기여**: Ray-tracing 기반 사이트별 DNN 압축. 혼합 정밀도 양자화 + NAS. WMMSE 대비 35배 에너지 효율.
- **관련도**: ★★★☆☆

#### [P64] Integrated Sensing and Edge AI: Survey for 6G
- **저자**: Liu et al.
- **연도**: 2025, arXiv:2501.06726, IEEE COMST
- **핵심 기여**: ISEA(Integrated Sensing and Edge AI) 서베이. Task-oriented ISAC + edge AI 추론 통합.
- **관련도**: ★★★★☆

---

*Updated: 2026-03-04 | 64 papers (46 initial + 18 web-search) analyzed*

---

## 5. UE Feature Vector / Representation 추출 관련 연구

> UE(User Equipment) 또는 디바이스 측에서 특징 벡터(feature vector), 잠재 표현(latent representation), 임베딩(embedding)을 추출하는 연구 동향 정리. CSI feedback autoencoder, wireless foundation model, split inference, semantic communication 4가지 맥락으로 분류.

### 수학적 표기 권장사항

| 요소 | 권장 표기 | 비고 |
|------|----------|------|
| UE feature vector | **z_u** 또는 **z** | 가장 보편적. subscript u로 UE 구분 |
| Encoder (UE-side) | **f_φ(·)** 또는 **f_enc(·)** | φ는 encoder 파라미터 |
| Decoder (BS-side) | **g_θ(·)** 또는 **f_dec(·)** | θ는 decoder 파라미터 |
| 채널 입력 | **H ∈ ℂ^{M×N}** | M: 안테나, N: 서브캐리어 |
| 양자화된 representation | **z_q** | VQ-VAE 계열 |
| 전체 representation 집합 | **Z = {z_u}_{u=1}^U** | U명 UE |
| 기본 형태 | **z_u = f_φ(H_u) ∈ ℝ^d** | d: representation 차원 |

### Cluster H1: CSI Feedback Autoencoder (3GPP Rel-18/19 표준화)

> UE가 encoder를 돌려 CSI를 압축된 latent vector로 변환 → gNB decoder가 복원하는 two-sided model. 3GPP에서 표준화 진행 중.

#### [P65] Universal Auto-encoder Framework for MIMO CSI Feedback
- **저자**: Jinhyun So, Hyukjoon Kwon
- **연도**: 2024, IEEE ICASSP 2024 / arXiv:2403.00299
- **핵심 기여**: 가변 입력 크기 및 다중 압축률 지원 Universal AE. 마스킹 기반 가변 압축.
- **수학적 표기**: Encoder `f_φ(·)`, Decoder `g_θ(·)`, Latent `z ∈ ℝ^λ`, `z = f_φ(H)`, CSI `H ∈ ℝ^{2×K×N_BS×N_UE}`, CR = λ/(2·K·N_UE·N_BS)
- **관련도**: ★★★★☆ — UE-side encoder가 latent z를 생성하는 가장 기본적 프레임워크

#### [P66] Vector Quantization for Deep-Learning-Based CSI Feedback in Massive MIMO
- **저자**: Junyong Shin, Yujin Kang, Yo-Seb Jeon
- **연도**: 2024, arXiv:2403.07355
- **핵심 기여**: VQ-VAE 기반 유한 비트 CSI 피드백. Grassmannian codebook으로 방향-크기 분리 양자화.
- **수학적 표기**: Encoder `f_enc(·)`, Latent `z = f_enc(H̃_ad)`, Quantized `z_q`, Codebook `B = {b_k}_{k=1}^{2^B}`, Shape-gain: `z_{q,i} = Q_mag(‖z_i‖)·Q_dir(z_i/‖z_i‖)`
- **관련도**: ★★★★☆ — 양자화된 representation z_q 전송의 대표적 방법

#### [P67] Precoding-Oriented CSI Feedback with MI-Regularized VQ-VAE
- **연도**: 2026, arXiv:2602.02508
- **핵심 기여**: Noisy pilot → learned codebook → discrete latent representation으로 매핑. Codeword index를 BS로 전송.
- **수학적 표기**: `z_q = Q(f_enc(y))`, y: pilot observation
- **관련도**: ★★★★☆ — Precoding 목적 최적화된 UE representation

#### [P68] Deep Learning-Based CSI Feedback for Wi-Fi with Temporal Correlation
- **연도**: 2025, arXiv:2505.23198
- **수학적 표기**: `z = f_enc(X)`, `X̂ = f_dec(z_q)`
- **관련도**: ★★★☆☆ — Wi-Fi 도메인이나 temporal correlation 활용이 참고 가치

### Cluster H2: Wireless Channel Foundation Models (채널 임베딩)

> 채널 데이터에서 task-agnostic representation을 뽑는 foundation model. Self-supervised pre-training 후 다양한 downstream task에 활용.

#### [P58] LWM: Large Wireless Model ⭐ (기존 목록)
- **저자**: Alikhani, Charan, Alkhateeb
- **연도**: 2024, arXiv:2411.08872
- **핵심 기여**: 최초 무선 채널 Foundation Model. Masked Channel Modeling으로 self-supervised 사전학습.
- **수학적 표기**: Patch embedding `e_i^emb = W_i^emb·p_i^m + b_i ∈ ℝ^D`, Output `E^LWM ∈ ℝ^{(P+1)×D}`, CLS embedding `C ∈ ℝ^D` (aggregated channel summary), Pre-training loss `L_MCM = (1/|M|)·Σ‖W_i^dec·e_i^LWM − p_i‖²`
- **관련도**: ★★★★★ — CLS token이 채널의 global representation으로 기능

#### [P69] ContraWiMAE: Multi-Task Foundation Model for Wireless Channel Representation ⭐
- **저자**: Berkay Guler, Giovanni Geraci, Hamid Jafarkhani
- **연도**: 2025, IEEE JSAC (submitted) / NeurIPS 2025 AI4NextG / arXiv:2505.09160
- **핵심 기여**: Masked reconstruction + contrastive learning 통합. 채널 representation의 의미적 유사성 학습.
- **수학적 표기**: Encoder `f_θ(·)`, Output `Z_enc ∈ ℝ^{2N_v×d_e}`, Contrastive loss `L_contra = −E[log(exp(z_i·z_i⁺/τ) / Σ_j exp(z_i·z_j/τ))]`, Reconstruction loss `L_recon = E[‖M'(H) − M'(g_φ(f_θ(H_m)))‖_F²]`
- **관련도**: ★★★★★ — Contrastive learning으로 채널 representation의 의미적 구조 학습

#### [P59] WiFo: Wireless Foundation Model for Channel Prediction (기존 목록)
- **저자**: Boxun Liu et al.
- **연도**: 2025, Science China Info Sciences / arXiv:2412.08908
- **수학적 표기**: 3D STF CSI `H ∈ ℂ^{T×K×N}`, Encoder output `H_enc ∈ ℝ^{D_enc×L_vis}`, Embedding `H_emb ∈ ℝ^{D_enc×L}`
- **관련도**: ★★★★☆ — Space-Time-Frequency 3차원 representation

#### [P70] CSI-MAE: Masked Autoencoder-based Channel Foundation Model
- **연도**: 2026, arXiv:2601.03789
- **핵심 기여**: Cross-scenario 일반화를 위한 masked channel modeling. 3GPP 데이터셋 기반.
- **수학적 표기**: `H_emb = f_enc([CLS; H_vis] + P_emb)`, [CLS] token as global representation, Masking ratio 75%, Loss `L_MSE = (1/N)·Σ‖h_i − ĥ_i‖²`
- **관련도**: ★★★★☆ — [CLS] 토큰 기반 global channel representation

### Cluster H3: Split Inference / Split Learning (중간 표현 전송)

> UE에서 DNN의 앞부분(head model)만 돌리고 intermediate representation을 edge/BS로 전송.

#### [P71] Semantic Edge Computing and Semantic Communications in 6G Networks
- **연도**: 2024, Computer Networks (Elsevier) / arXiv:2411.18199
- **핵심 기여**: DNN split across device (head) and edge (tail). Rate-distortion 프레임워크.
- **수학적 표기**: Markov chain `X → L_1 → L_2 → ... → L_i → ... → Y`, Head model `H: X → L_i`, Tail model `T: L_i → Y`, Optimization `min_{L_i} T^cm(L_i) + T^ce(L_i) + T_n(L_i)`
- **관련도**: ★★★★☆ — 중간 레이어 출력 L_i가 UE representation으로 기능

#### [P72] Dynamic Encoding and Decoding for Split Learning in MEC
- **연도**: 2024, arXiv:2309.02787
- **핵심 기여**: Information bottleneck theory 기반 split learning. UE encoder가 전송 비용 vs 정보량 균형 최적화.
- **수학적 표기**: IB framework: `min I(Z;X) − β·I(Z;Y)`, Z: UE-side latent representation
- **관련도**: ★★★★☆ — Information bottleneck으로 representation 최적 차원 결정

### Cluster H4: Semantic / Task-Oriented Communication (디바이스 측 특징 추출)

> 디바이스에서 의미론적 특징을 추출하여 전송. 원본 데이터 대신 task-relevant representation만 전송.

#### [P73] SAFE: Semantic Adaptive Feature Extraction with Rate Control for 6G
- **저자**: Yuna Yan et al.
- **연도**: 2024, IEEE Globecom 2024 Workshop / arXiv:2410.01597
- **핵심 기여**: 입력을 sub-semantic 단위로 분해, 채널 상태에 따라 적응적 rate 할당.
- **수학적 표기**: Semantic encoder `g(I, ω)`, Sub-semantic extraction `S_i = c(I_i, φ)`, Bandwidth ratio `k_i/n = (H/8·W/8·d_i)/(H·W·3)`
- **관련도**: ★★★★☆ — Sub-semantic S_i가 적응적 UE representation

#### [P74] Robust Deep JSCC for Task-Oriented Semantic Communications
- **연도**: 2025, arXiv:2503.12907
- **핵심 기여**: JSCC 인코딩된 representation의 채널 노이즈에 대한 강건성 정규화.
- **관련도**: ★★★☆☆ — 전송된 representation의 robustness 보장

#### [P75] Distributed Generative AI in 6G: Mobile Edge Generation
- **연도**: 2024, arXiv:2409.05870
- **핵심 기여**: Latent feature 압축 후 전송. DRL 기반 동적 전력 할당.
- **관련도**: ★★★☆☆ — Generative model의 latent feature 전송

### Cluster H5: End-to-End Autoencoder / 3GPP 표준

#### [P76] A Review on DL Autoencoder in Next-Generation Communication Systems
- **연도**: 2024, arXiv:2412.13843
- **핵심 기여**: 120+ 논문 리뷰. 통신 AE의 표준 표기법 정리.
- **수학적 표기 (표준화)**: Encoder (TX) `f_{θ_t}: x → w`, Decoder (RX) `f_{θ_r}: y → x̂`, Combined `θ = {θ_t, θ_r}`, MSE loss `L_MSE(θ) = (1/N)·Σ|f(x;θ) − x|²`
- **관련도**: ★★★☆☆ — 표기법 표준 레퍼런스

#### [P77] AI/ML for Beam Management: A Standardization Perspective
- **저자**: Qing Xue et al.
- **연도**: 2024, arXiv:2309.10575
- **핵심 기여**: 3GPP Rel-18 UE-side model이 L1-RSRP 예측. Set B 측정 → Set A 빔 예측.
- **수학적 표기**: Beam sets A (prediction), B (measurement), UE input: measured L1-RSRP from Set B, Output: predicted beam ID / L1-RSRP
- **관련도**: ★★★★☆ — 3GPP 표준화된 UE-side AI의 공식 프레임워크

### 표기법 종합 비교표

| 연구 맥락 | Encoder 표기 | Latent/Feature 표기 | Decoder 표기 | 대표 논문 |
|-----------|-------------|-------------------|-------------|----------|
| CSI Feedback AE | f_φ(·), f_enc(·) | z = f_enc(H), z ∈ ℝ^λ | g_θ(·), f_dec(·) | P65, P66 |
| VQ-VAE CSI | f_enc(·) | z, z_q (quantized) | f_dec(·) | P66, P67 |
| Foundation Models | f_θ(·) | E^LWM, Z_enc, [CLS] ∈ ℝ^D | g_φ(·) | P58, P69, P70 |
| Split Inference | Head H(·) | L_i (i-th layer latent) | Tail T(·) | P71, P72 |
| Semantic Comm | g(·, ω), c(·, φ) | S_i (sub-semantics) | Decoder | P73 |
| End-to-End AE | f_{θ_t}(·) | w (latent signal) | f_{θ_r}(·) | P76 |
| 3GPP Beam Mgmt | ML model f_W(·) | R (RSRP vector) | Prediction î | P77 |

> **권장**: `z_u = f_φ(H_u) ∈ ℝ^d` — CSI feedback AE, foundation model, split learning 문헌 모두와 일관. VAE 문헌의 표준 관례(z)와도 부합.

---

## 6. Site-Adaptive Channel Estimation & Attention in FL (8편)

> PACE-Net 실험에서 발견한 "SE attention이 FL aggregation과 충돌" 현상 관련 논문 조사. Site-specific adaptation, domain adaptation, attention + FL 조합 연구 정리.

### Cluster I1: Attention 기반 채널 추정 (SE block의 원본 참조)

#### [P78] PACE-Net: Channel Estimation for Massive MIMO via Polarized Self-Attention (PACE-Net 원본 참조)
- **저자**: Yang, Li, Liu, Xia, Wang, Li
- **연도**: 2025, Entropy (MDPI) 27(3):220
- **PDF**: `papers/PACE_Net_DL_Channel_Estimation_Massive_MIMO.pdf`
- **핵심 기여**: Polarized Self-Attention (PSA) 기반 채널 추정 네트워크. Channel attention (SE 방식) + Spatial attention 직교 결합. 채널 추정을 image denoising으로 변환.
- **방법론**: ResNet backbone + PSA 모듈. Kronecker 채널 모델, 64×16 massive MIMO.
- **주요 결과**: MMSE 대비 우수한 NMSE, 계산복잡도 대폭 감소 (MMSE의 행렬 역산 제거)
- **관련도**: ★★★★☆ — **PACE-Net의 직접적 참조. SE block이 채널 특성에 따라 feature를 adaptive하게 re-weight한다는 핵심 메커니즘 출처.**

#### [P79] Channelformer: Attention-Based Neural Solution for Wireless Channel Estimation
- **저자**: Luan, Thompson
- **연도**: 2023, IEEE TWC / arXiv:2302.04368
- **PDF**: `papers/Channelformer_2302.04368.pdf`
- **핵심 기여**: Multi-head self-attention 인코더 + CNN 디코더 하이브리드 CE. 70% 파라미터 pruning 가능.
- **관련도**: ★★★☆☆ — Attention 기반 CE의 기초 연구. 단일 BS, FL 없음.

### Cluster I2: Channel Attention과 FL Heterogeneity

#### [P80] ANFR: Adaptive Normalization-Free Feature Recalibration in Federated Learning ⭐
- **저자**: Siomos, Naval-Marimont, Passerat-Palmbach, Tarroni
- **연도**: 2025, ICLR 2025 / arXiv:2410.02006
- **PDF**: `papers/ANFR_FL_channel_attention_2410.02006.pdf`
- **핵심 기여**: **FL에서 channel attention으로 heterogeneity 해결.** Weight Standardization + Channel Attention으로 BN 제거. **클라이언트 간 불일치 feature를 suppress, 일관된 feature를 emphasize.**
- **주요 결과**: Global FL과 personalized FL 모두에서 SOTA. BN 없이 heterogeneous data에서 안정적 수렴.
- **강점**: **SE block과 FL의 상호작용을 직접 다룬 가장 가까운 논문.** Channel attention이 FL에서 feature inconsistency를 완화하는 메커니즘 제시.
- **한계**: Vision 도메인 (CIFAR, ImageNet). 무선 채널 추정에는 적용 안 됨.
- **관련도**: ★★★★★ — **핵심 참고문헌. 우리 PACE-Net 실험 결과를 해석하는 이론적 근거. "SE가 client-specific feature를 학습하면 FL aggregation 시 상충"이라는 우리 가설을 뒷받침.**

#### [P81] FedAttn: Federated Attention for Distributed LLM Inference
- **저자**: Deng, Xiong, Chen, Kim, Debbah, Poor
- **연도**: 2025, arXiv:2511.02647
- **PDF**: `papers/FedAttn_2511.02647.pdf`
- **핵심 기여**: Transformer self-attention 자체를 federate. 로컬 self-attention + KV matrix 주기적 교환/aggregation. Token relevance heterogeneity 분석.
- **관련도**: ★★★☆☆ — Attention을 분산하는 개념적 참고. 채널 추정과는 다른 도메인.

### Cluster I3: Site-Specific / Environment-Adaptive 채널 추정

#### [P82] NVIDIA Neural 5G NR Receiver: Environment-Specific Base Stations ⭐
- **저자**: Cammerer et al. (NVIDIA / Rohde & Schwarz)
- **연도**: 2024-2025, arXiv:2409.02912
- **PDF**: `papers/NVIDIA_env_specific_BS_2409.02912.pdf`
- **핵심 기여**: **Site-specific neural receiver.** Ray-tracing 디지털 트윈으로 사전학습 → 사이트별 fine-tuning. Classical PHY layer (CE, equalization, demapping)를 trainable NN으로 대체.
- **주요 결과**: 3GPP-compliant 5G NR 실시간 구동. Site-specific fine-tuning이 generic 모델 대비 유의미한 성능 향상.
- **강점**: **"각 BS는 고유한 전파 환경 → site-specific model 필요"라는 우리 프로젝트 핵심 가정의 산업적 검증.**
- **관련도**: ★★★★★ — **Site-specific adaptation의 필요성을 산업 수준에서 입증. 우리의 per-BS LoRA adapter가 이 방향과 정확히 일치.**

#### [P83] Transfer Learning vs Meta-Learning for MIMO-OFDM Channel Denoising
- **저자**: Ha, Jeon et al.
- **연도**: 2025, arXiv:2508.09751
- **PDF**: `papers/Transfer_vs_Meta_CE_2508.09751.pdf`
- **핵심 기여**: 표준호환 온라인 학습 데이터 생성 + Transfer Learning (fine-tuning) vs Meta-Learning (MAML) 비교. 새로운 환경에 빠른 적응.
- **관련도**: ★★★★☆ — Site adaptation 방법론 비교. 우리의 FL + LoRA는 이 두 접근의 중간점.

#### [P84] Domain Adaptation-Enabled Realistic Map-Based Channel Estimation
- **저자**: Hoang et al.
- **연도**: 2025, arXiv:2507.08974
- **PDF**: `papers/Domain_Adapt_CE_2507.08974.pdf`
- **핵심 기여**: QSCM (준정적 채널 모델) → MBCM (맵기반 채널 모델) 간 domain gap을 domain adaptation으로 해소. Simulation-to-realistic 전이.
- **관련도**: ★★★★☆ — Site 간 채널 통계 차이를 domain adaptation으로 해결. 우리의 LoRA per-BS가 implicit domain adaptation 역할.

#### [P85] ReQuestNet: Foundational Learning Model for Channel Estimation (Qualcomm)
- **저자**: Pratik, Sadeghi, Cesa et al. (Qualcomm AI Research)
- **연도**: 2025, IEEE Globecom 2025 / arXiv:2508.08790
- **PDF**: `papers/ReQuestNet_Qualcomm_2508.08790.pdf`
- **핵심 기여**: Recurrent equivariant 아키텍처로 채널 추정 foundation model. 다양한 delay-Doppler profile에서 10 dB gain over genie MMSE.
- **관련도**: ★★★★☆ — 대형 foundation model 접근. 우리의 경량 per-BS 접근과 대비되는 방향.

### Cluster I4: Continual Learning for Channel Prediction

#### [P86] Continual Learning for Wireless Channel Prediction ⭐
- **저자**: Mohsin, Umer, Bilal, Jamshed, Cioffi
- **연도**: 2025, ICML Workshop ML4Wireless / arXiv:2506.22471
- **PDF**: `papers/Continual_Learning_Channel_2506.22471.pdf`
- **핵심 기여**: **Cross-cell handover 시 채널 통계 변화에 대한 continual learning.** Replay, synaptic importance regularization (EWC/SI), LwF 비교. 다른 안테나/주파수/산란 환경 간 적응.
- **주요 결과**: 최고 방법 2 dB NMSE 개선 (~35%). SI가 가장 효과적.
- **강점**: **"각 BS/cell은 다른 채널 통계" 문제를 continual learning으로 접근 — 우리의 FL + per-BS adapter와 상보적 관점.**
- **관련도**: ★★★★★ — **우리와 같은 문제 (BS별 채널 이질성)를 다른 방법 (CL vs FL)으로 해결. Related work에 반드시 인용.**

---

---

## Cluster J: 6G ELAA / XL-MIMO 채널 추정

> 6G의 핵심 기술인 ELAA(Extremely Large Aperture Array) / XL-MIMO의 근거리장(near-field) 채널 추정, 빔 훈련, 분산 처리 관련 논문. 수백~수천 안테나 배열에서 구형파(spherical wave) 전파, 공간 비정상성(spatial non-stationarity), polar-domain 표현 등 새로운 도전과제를 다룸.

### J1: ELAA/XL-MIMO 서베이 & 튜토리얼

#### [P89] A Tutorial on Near-Field XL-MIMO Communications Towards 6G ⭐
- **저자**: Lu, Zeng, You, Han, Zhang, Wang, Dong, Jin, Wang, Jiang, You, Zhang
- **연도**: 2024, IEEE Communications Surveys & Tutorials / arXiv:2310.11044
- **PDF**: `papers/XL_MIMO_Tutorial_Near_Field_6G_2310.11044.pdf`
- **핵심 기여**: **XL-MIMO 근거리장 통신 종합 튜토리얼**. Near-field beam codebook, beam training, channel estimation, DAM (Delay Alignment Modulation) 전송 설계 포괄. Rayleigh distance가 수십~수백m로 확대 → far-field 가정 깨짐.
- **주요 결과**: XL-MIMO는 5G massive MIMO 대비 10배+ 안테나 → beamfocusing (3D 초점), near-field 영역에서 angle+distance 동시 추정 필요
- **강점**: IEEE COMST 게재, XL-MIMO 분야 표준 참조 논문
- **관련도**: ★★★★★ — **프로젝트의 ELAA 확장 방향의 기초 레퍼런스. P33 (Bjornson/Chae 6G MIMO 튜토리얼)의 근거리장 확장판.**

#### [P90] Channel Estimation for 6G Near-Field Wireless Communications: A Comprehensive Survey
- **저자**: Long, Ye, Moretti, Morelli, Sanguinetti, Chen, Wang
- **연도**: 2025, arXiv:2507.23526
- **PDF**: `papers/NF_CE_6G_Survey_2507.23526.pdf`
- **핵심 기여**: **근거리장 채널 추정 기법 종합 서베이**. EM 파동 관점에서 근거리장/원거리장 경계 정의, 주류 근거리장 채널 모델 소개, 단일/다중 사용자 + 단일/다중 캐리어 시스템별 추정 기법 체계적 분류.
- **강점**: 20+ 대표 논문 비교, 추정 정확도-복잡도-파일럿 오버헤드 trade-off 분석
- **관련도**: ★★★★★ — **ELAA 채널 추정의 최신 종합 레퍼런스. 프로젝트가 ELAA로 확장 시 필수 인용.**

#### [P91] Recent Advances in Near-Field Beam Training and Channel Estimation for XL-MIMO Systems
- **저자**: Zeng, Wang, Li, Hao, Chu, Xie, Wang, Pham
- **연도**: 2025, arXiv:2504.05578
- **PDF**: `papers/NF_Beam_Training_CE_XL_MIMO_Survey_2504.05578.pdf`
- **핵심 기여**: XL-MIMO 빔 훈련 + 채널 추정 최신 기법 종합 리뷰. Polar-domain codebook, hierarchical beam training, CS-based/DL-based CE 분류. 미해결 과제 제시.
- **강점**: Beam training과 CE를 동시에 다루는 체계적 분류법
- **관련도**: ★★★★☆ — 빔 훈련 + CE 통합 관점에서 프로젝트 확장 방향 참고

#### [P92] Distributed Signal Processing for ELAA Systems: State-of-the-Art and Future Directions
- **저자**: Xu, Larsson, Jorswieck, Li, Jin, Chang
- **연도**: 2024-2025, IEEE JSTSP / arXiv:2407.16121
- **PDF**: `papers/Distributed_SP_ELAA_2407.16121.pdf`
- **핵심 기여**: **ELAA 분산 신호처리 종합 개관**. 안테나 수 증가에 따른 interconnection 비용 + 계산 복잡도 병목 → 분산 SP 알고리즘 필요. 분산 CE, 분산 빔포밍, 분산 검출 포괄.
- **강점**: Larsson (Linköping) 공저, ELAA의 분산 처리 필요성을 체계적으로 정리
- **관련도**: ★★★★★ — **ELAA에서 분산/협력 처리의 필요성 = 프로젝트의 FL + per-BS 프레임워크와 직접 연결. Sub-array 단위 로컬 처리 → 전역 집계 패턴이 FL aggregation과 구조적으로 동일.**

### J2: DL 기반 XL-MIMO 채널 추정

#### [P93] Lightweight DL-Based Channel Estimation for XL-MIMO (XLCNet)
- **저자**: Dong et al. (NUAA)
- **연도**: 2024, IEEE TVT / arXiv:2402.08916
- **PDF**: `papers/XLCNet_Lightweight_CE_XL_MIMO_2402.08916.pdf`
- **핵심 기여**: **XLCNet — near-field + far-field 모두 지원하는 경량 CE 네트워크**. 2D Conv + shortcut (ReEsNet 유사 구조). C-XLCNet: pruning + quantization으로 10x 복잡도 감소, 36x 모델 크기 감소.
- **주요 결과**: XLCNet이 NMSE 및 spectral efficiency에서 기존 방법 능가, C-XLCNet은 제한적 성능 저하로 경량화
- **관련도**: ★★★★★ — **ReEsNet 유사 구조 + near-field 대응 = 프로젝트의 estimator를 ELAA로 확장하는 직접적 참고. 경량화 기법도 on-device 배포에 활용 가능.**

#### [P94] Channel Estimation for Wideband XL-MIMO: A Constrained Deep Unrolling Approach
- **저자**: Zheng et al.
- **연도**: 2025, arXiv:2505.07717
- **PDF**: `papers/Wideband_XL_MIMO_CE_Deep_Unrolling_2505.07717.pdf`
- **핵심 기여**: MAP 문제로 CE 정형화 → PGD 알고리즘을 deep unrolling. Learnable step sizes + NN proximal mapping으로 채널 prior 암묵적 학습. Wideband + near-field + beam squint 동시 처리.
- **관련도**: ★★★★☆ — Model-driven DL 접근이 프로젝트의 data-driven 접근과 보완적

#### [P95] LLM4XCE: Large Language Models for XL-MIMO Channel Estimation
- **저자**: Li, Li, Dong (NUAA)
- **연도**: 2025, arXiv:2512.08955
- **PDF**: `papers/LLM4XCE_XL_MIMO_CE_2512.08955.pdf`
- **핵심 기여**: **LLM의 semantic modeling 능력을 XL-MIMO CE에 활용**. Hybrid-field (near+far) 시나리오에서 spatial-channel representation 복원. Foundation model → CE 적용의 최신 사례.
- **관련도**: ★★★★☆ — Foundation model 기반 CE의 최신 방향. P19 (Multi-task PHY LLM)의 XL-MIMO 확장판.

### J3: Near-Field 빔 훈련

#### [P96] Near-Field Beam Training for XL-MIMO Based on Deep Learning
- **저자**: Ning et al.
- **연도**: 2024, IEEE Transactions on Mobile Computing / arXiv:2406.03249
- **PDF**: `papers/NF_Beam_Training_XL_MIMO_DL_2406.03249.pdf`
- **핵심 기여**: CNN 기반 근거리장 빔 훈련. Polar-domain codebook에서 angle+distance 동시 추정. Historical data 활용으로 beam training overhead 대폭 감소.
- **관련도**: ★★★★☆ — DL 빔 훈련이 P34 (SSBA)의 near-field 확장에 해당

### J4: ELAA 분산/구조적 CE

#### [P97] Channel Estimation for XL-MIMO with Decentralized Baseband Processing
- **저자**: Tang, Wang, Pan, Zeng, Chen, Yu, Xiao, de Lamare, Wang (Southeast Univ / KTH / PUC-Rio)
- **연도**: 2025, arXiv:2501.17059
- **PDF**: `papers/Decentralized_CE_XL_MIMO_2501.17059.pdf`
- **핵심 기여**: **Hybrid analog-digital XL-MIMO에서 분산 baseband 처리 기반 CE**. 2단계: (1) Local sparse reconstruction (SBL-GNN) per sub-array, (2) Global fusion + refinement (variational message passing). 로컬 추정 → 전역 집계 구조.
- **주요 결과**: SBL-GNN이 기존 centralized/decentralized 방법 대비 우수한 추정 성능 + 낮은 복잡도
- **강점**: **GNN으로 채널 계수 간 dependency 캡처, Bayesian 프레임워크**
- **관련도**: ★★★★★ — **"Local reconstruction → Global refinement" = FL의 "Local training → Global aggregation"과 구조적으로 동일. 프로젝트의 FL 프레임워크를 ELAA sub-array 구조로 확장하는 핵심 참고.**

#### [P98] A Novel Pilot Scheme for Sub-array Structured ELAA in XL-MIMO
- **저자**: (arXiv:2512.10478)
- **연도**: 2025, arXiv:2512.10478
- **PDF**: `papers/Sub_Array_ELAA_Pilot_2512.10478.pdf`
- **핵심 기여**: Sub-array 구조 ELAA에서의 다중 사용자 UL CE 파일럿 설계. 공간 비정상성 (spatial non-stationarity) — 각 sub-array가 서로 다른 user subset을 "볼" 수 있음 (visibility region).
- **관련도**: ★★★★☆ — Sub-array별 다른 채널 통계 = BS별 다른 채널 통계와 유사한 구조

### J5: ISAC + ELAA

#### [P99] Integrated Channel Estimation and Sensing for Near-Field ELAA Systems
- **저자**: Wang, Fang, Li, Ning (UESTC / Stevens Institute)
- **연도**: 2026, arXiv:2601.18333
- **PDF**: `papers/Integrated_CE_Sensing_NF_ELAA_2601.18333.pdf`
- **핵심 기여**: ELAA 근거리장 OFDM에서 **CE + 센싱 통합**. Non-orthogonal pilot으로 다중 사용자 수용. Tensor decomposition (CPD/BTD)으로 채널 파라미터 + 사용자 위치 동시 추정.
- **주요 결과**: 파일럿 수가 사용자 수보다 적어도 uniqueness 보장, CS 대비 우수한 CE 정확도
- **관련도**: ★★★★☆ — ISAC + ELAA의 최신 연구. 후보 #5 (Sensing-Aided Cooperative CE)의 ELAA 확장 방향.

### J3: Modular Array & Near-Field Beamfocusing

#### [P99b] Near-Field Beamfocusing, Localization, and Channel Estimation with Modular Linear Arrays
- **저자**: Kosasih, Demir, Björnson (Linköping University)
- **연도**: 2025, arXiv:2505.07991
- **PDF**: `papers/NF_Beamfocusing_Modular_Arrays_2505.07991.pdf`
- **핵심 기여**: **Modular Linear Array (MLA) — 여러 소규모 ULA를 수 미터 간격으로 배치하여 aperture 확장**. 안테나 수 증가 없이 near-field beamfocusing 달성. 시뮬레이션: 2×25=50 ~ 4×36=144 elements, 15 GHz.
- **주요 결과**: MLA가 동일 안테나 수의 단일 ULA 대비 beamfocusing 해상도 우수, 적은 안테나로 NF 효과 달성
- **관련도**: ★★★★☆ — **"적은 안테나로 NF"이라는 관점이 프로젝트의 ELAA 실험 세팅 (S=256, M=512, L=1024)의 합리적 범위를 뒷받침. Björnson 공저.**

### J 종합: 프로젝트와의 연결

**ELAA/XL-MIMO가 프로젝트에 주는 시사점:**
1. **Sub-array = Sub-BS**: ELAA의 sub-array 단위 로컬 처리 + 전역 집계 (P92, P97) ↔ FL의 per-BS 로컬 학습 + 전역 집계 — **구조적으로 동일한 프레임워크**
2. **Spatial non-stationarity = Site heterogeneity**: ELAA에서 각 sub-array가 다른 채널 통계를 경험 (P98) ↔ 각 BS가 다른 전파 환경 — **같은 문제의 다른 스케일**
3. **Near-field CE**: 기존 far-field LS/LMMSE가 깨짐 → DL 기반 CE의 가치가 더 커짐 (P93, P94)
4. **경량화**: C-XLCNet (P93)의 pruning/quantization이 on-device 배포 (O-DU급)에 직접 활용 가능

---

## Cluster K: Differentiable Ray Tracing & Physics-Informed Optimization

> Differentiable RT(미분 가능 레이 트레이싱)를 이용한 물리 기반 최적화 — Sim-to-Real 캘리브레이션, site-specific 배치 최적화, implicit differentiation. 기존 P61 (Hoydis et al., Learning Radio Environments)의 확장. 2025-2026년 6G 연구에서 "Diff" 기술이 핵심 enabler로 부상.

### K1: Differentiable RT 핵심 기법

#### [P100] Fast, Differentiable, GPU-Accelerated Ray Tracing via Implicit Differentiation ⭐
- **저자**: Eertmans, Lequeu, Legat, Jacques, Oestges (UCLouvain, ICTEAM)
- **연도**: 2025, EuCAP 2026 채택 / arXiv:2510.16172
- **PDF**: `papers/Fast_Diff_GPU_RT_Implicit_2510.16172.pdf`
- **핵심 기여**: **Implicit Differentiation으로 솔버 반복을 직접 미분하지 않고 그래디언트 계산**. Fermat 원리 기반 경로 길이 최소화 → reflection + diffraction 통합 처리. 기존 AD(자동 미분) 대비 압도적 메모리/속도 이점.
- **방법론**: 경로 탐색을 총 경로 길이 최소화 문제로 정형화 → GPU 병렬 실행 → implicit diff로 그래디언트 효율 계산
- **주요 결과**: Newton 방법 수준 수렴 + 대규모 확장성 우수. JAX/DrJIT 통합, 오픈소스.
- **강점**: **Sionna RT의 핵심 한계 (반사/회절 분리 처리) 해결. ELAA 수천 안테나 환경에서 실시간 최적화 가능성.**
- **관련도**: ★★★★★ — **프로젝트의 Sionna RT 파이프라인에 implicit diff 통합 시, 재질/안테나 파라미터 역전파 최적화 속도 대폭 개선. dApp 실시간성의 핵심 enabler.**

#### [P101] Fully Differentiable Ray Tracing via Discontinuity Smoothing for Radio Network Optimization
- **저자**: Eertmans et al. (UCLouvain)
- **연도**: 2024, arXiv:2401.11882
- **PDF**: `papers/Fully_Diff_RT_Discontinuity_2401.11882.pdf`
- **핵심 기여**: RT의 불연속점 (ray obstruction에 의한 급격한 변화)을 smoothing function으로 처리하여 **모든 scene parameter에 대해 미분 가능한 loss function** 제공. 기존 diff RT의 gradient=0 문제 해결.
- **관련도**: ★★★★☆ — P100의 선행 연구. 불연속 처리는 site-specific 환경에서 건물 가림 등에 필수.

### K2: Sim-to-Real 캘리브레이션

#### [P102] VLM-Guided Differentiable RT for Multi-Material RF Parameter Estimation ⭐
- **저자**: (arXiv:2601.18242)
- **연도**: 2026, arXiv:2601.18242
- **PDF**: `papers/VLM_Guided_Diff_RT_RF_Param_2601.18242.pdf`
- **핵심 기여**: **Vision-Language Model (VLM)로 씬 이미지에서 재질 추론 → ITU-R 테이블 기반 초기값 설정 → Differentiable RT로 gradient-based refinement**. VLM이 TX/RX 배치도 최적화 (material-discriminative paths 선택).
- **주요 결과**: 2-4x 빠른 수렴, 10-100x 낮은 최종 파라미터 오류 (random init 대비). Sionna 기반 실험.
- **강점**: **AI (VLM) + Physics (Diff RT) 결합의 최신 사례. 프로젝트의 Sionna 데이터셋 정당성 방어에 직접 활용 가능 ("단순 시뮬이 아니라 VLM + diff RT로 캘리브레이션").**
- **관련도**: ★★★★★ — **Sim-to-Real gap 해소의 최신 방법론. 리뷰어의 "시뮬레이션 데이터 신뢰성" 공격 방어 논거.**

#### [P103] Site-Specific RIS Deployment via Calibrated Ray Tracing
- **저자**: (arXiv:2510.09478)
- **연도**: 2025, arXiv:2510.09478
- **PDF**: `papers/Site_Specific_RIS_Calibrated_RT_2510.09478.pdf`
- **핵심 기여**: Sionna RT + 실측 데이터 캘리브레이션 기반 RIS 배치 최적화. RIS 위치/방향/구성 + BS 빔포밍을 joint optimization. 4G/5G/6G 주파수에서 검증.
- **관련도**: ★★★★☆ — Calibrated RT → site-specific optimization 파이프라인. 프로젝트의 디지털 트윈 활용 방향과 일치.

### K3: Differentiable RT vs Deep Learning 비교

#### [P104] Radio Propagation Modelling: To Differentiate or To Deep Learn?
- **저자**: (arXiv:2509.19337)
- **연도**: 2025, arXiv:2509.19337
- **PDF**: `papers/Diff_vs_DL_Radio_Propagation_2509.19337.pdf`
- **핵심 기여**: **Diff RT vs DL 실세계 대규모 비교** — 13개 도시, 10,000+ 안테나 실측 데이터. DL이 diff RT 대비 최대 3 dB 정확도 우위 + 빠른 적응. Diff RT는 대규모 일반화에서 한계.
- **주요 결과**: DL > Diff RT (정확도, 적응 속도) in production-scale. 단, diff RT는 물리 해석 가능성에서 우위.
- **강점**: **냉정한 현실 확인 — diff RT가 만능이 아님. DL과의 결합이 필요하다는 논거 (= 프로젝트의 DL + Sionna RT 접근이 정당).**
- **관련도**: ★★★★★ — **"Diff RT만으로는 부족, DL과 결합해야" → 프로젝트의 "Sionna RT 데이터 + DL 채널 추정" 접근의 정당성 직접 뒷받침.**

### K4: ELAA + Differentiable Radiomap

#### [P105] U6G XL-MIMO Radiomap Prediction: Multi-Config Dataset and Beam Map Approach
- **저자**: Li et al.
- **연도**: 2026, arXiv:2603.06401
- **PDF**: `papers/U6G_XL_MIMO_Radiomap_BeamMap_2603.06401.pdf`
- **핵심 기여**: **최초 XL-MIMO radiomap 데이터셋** — 78,400 radiomaps, 800 도시 씬, 5개 주파수대 (1.8-6.7 GHz), 9개 array config (최대 32x32 UPA). **Beam Map 접근**: array 구성을 scalar가 아닌 beam radiation pattern으로 입력 → **재학습 없이 새 array config에 일반화**.
- **주요 결과**: Beam map이 기존 scalar encoding 대비 unseen array config에서 대폭 성능 향상
- **강점**: **미분 가능한 beam map → 빔 최적화 알고리즘에 즉시 통합 가능. Array config가 바뀌어도 재학습 불필요 = O-RAN 다벤더 환경의 핵심 요구.**
- **관련도**: ★★★★★ — **ELAA + differentiable 접근의 최신 사례. HW 이질성 (P41) 문제에 대한 해법 방향. 프로젝트의 ELAA 확장 시 radiomap 기반 사전학습 데이터로 활용 가능.**

### K 종합: 프로젝트에 주는 시사점

**Differentiable RT 기술의 3가지 역할:**

1. **Sim-to-Real 캘리브레이션** (P61, P102, P103): Sionna RT의 재질 파라미터를 실측 데이터로 역전파 최적화 → 리뷰어의 "시뮬 데이터 신뢰성" 질문에 대한 방어 논거
2. **Site-specific 배치 최적화** (P100, P101): Implicit diff로 BS/안테나 위치/방향을 gradient-based 최적화 → dApp의 실시간 빔 제어에 필수
3. **Retraining-free 일반화** (P105): Beam map 입력으로 array config 변경 시 재학습 불필요 → O-RAN 다벤더 환경 지원

**냉정한 현실** (P104): Diff RT 단독으로는 production-scale에서 DL에 뒤짐 → **DL + Physics (Diff RT) 결합이 최적** = 프로젝트의 접근 방향이 정당.

---

## 7. 미분류 추가 논문 (PDF 보유, 본문 미정리)

#### [P87] Distributed AI Platform for the 6G RAN
- **저자**: Ananthanarayanan et al. (Microsoft Research)
- **연도**: 2024, arXiv:2410.03747
- **PDF**: `papers/Distributed_AI_Platform_6G_RAN_2410.03747.pdf`
- **핵심 기여**: 6G AI-native RAN을 위한 분산 AI 플랫폼 아키텍처 제안. 기존 접근의 한계 분석 및 CU/DU/RU disaggregation 기반 AI 배포 비전.
- **관련도**: ★★★★☆ — AI-native RAN 플랫폼 관점. O-RAN disaggregation + AI 워크로드 공존.

#### [P88] REAL: RL-Enabled xApps for Closed-Loop Optimization in O-RAN with srsRAN
- **저자**: Barker et al. (Clemson University)
- **연도**: 2025, arXiv:2502.00715
- **PDF**: `papers/REAL_RL_xApps_ORAN_2502.00715.pdf`
- **핵심 기여**: OSC RIC + srsRAN에서 RL 기반 xApp으로 near-RT 네트워크 슬라이싱 (eMBB/URLLC/mMTC). GNU Radio 채널 모델링 (FSPL, multipath, AWGN, Doppler). 실시간 closed-loop 최적화 시연.
- **주요 결과**: RL xApp이 다양한 트래픽 수요에서 동적 자원 할당 및 QoS 유지
- **관련도**: ★★★★☆ — O-RAN 실시간 AI 제어 시연. srsRAN 기반 경량 테스트베드.

> **참고**: `Decentralized_FL_GNN_ORAN_CE_2404.03088.pdf`는 P10 (Robust FL, Fang et al.)과 동일 논문 (arXiv ID 일치).

---

*Updated: 2026-03-18 | 105 papers (88 + 11 ELAA/XL-MIMO + 6 Diff RT) + 1 duplicate analyzed*

---

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
- **CE-skip 관련**: PSA module의 O(DK^2 NtNr) complexity — DL-CE가 antenna 수에 선형 scaling하므로 ELAA에서 skip 이득 증가.
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
- **CE-skip 관련**: CE-skip의 architectural home. dApp이 DU에 co-located, E3 interface로 CE KPM 접근, **450us 제어 루프** 실측. "Augmented Sensing and Channel Estimation"을 dApp use case로 명시적 언급.
- **CE-skip relevance**: ★★★★★

#### [P07] XAI-on-RAN: Explainable, AI-native, GPU-Accelerated RAN
- **CE-skip 관련**: NVIDIA Aerial A100 testbed에서 **GPU utilization 63%** — CE-skip monitor를 위한 37% compute headroom 존재. LSTM inference 5.1ms, attention 추가 비용 0.6ms.
- **CE-skip relevance**: ★★★★★

#### [P01] Beyond Connectivity: Open Architecture for AI-RAN Convergence in 6G
- **CE-skip 관련**: AI-RAN Site에서 GPU MIG partitioning (40GB RAN + 20GB LLM + 10GB CNN). AI-and-RAN coexistence 검증 (throughput/CRC 일관).
- **CE-skip relevance**: ★★★★★

#### [P62] AI/ML Lifecycle Management for Interoperable AI Native RAN
- **CE-skip 관련**: 3GPP Rel-16~20 LCM framework. CE-skip의 monitor = LCM Management block의 (de)activation control. "Scheduling more monitoring reduces overhead savings" — CE-skip의 핵심 trade-off.
- **CE-skip relevance**: ★★★★★

#### [P06] Towards AI-Native RAN: An Operator's Perspective
- **CE-skip 관련**: "RS overhead reduction"을 6G use case로 명시 (Table I). AI Node + 6gNB 구조가 CE-skip 배포 모델과 일치.
- **CE-skip relevance**: ★★★★☆

#### [P87] Distributed AI Platform for the 6G RAN (Microsoft)
- **CE-skip 관련**: Far-edge runtime (<1ms). "vRAN is largely underutilized (<50%)" — CE-skip의 compute headroom 추가 증거.
- **CE-skip relevance**: ★★★★☆

#### [P40] Accelerating vRAN and O-RAN with SIMD
- **CE-skip 관련**: CPU-based PHY timing data (4x4 MIMO LMMSE detection 0.03ms). GPU와 대비되는 CPU vRAN 참조점.
- **CE-skip relevance**: ★★★☆☆

#### [P08] Self-Learning Model Versioning for AI-native O-RAN Edge
- **CE-skip 관련**: RL policy가 "dApp은 stability를 accuracy보다 우선" — CE-skip의 설계 원칙. Model versioning으로 LS/LMMSE/DL-CE 전환 관리.
- **CE-skip relevance**: ★★★☆☆

#### [P04] MX-AI: Agentic Observability and Control Platform for Open and AI-RAN
- **CE-skip 관련**: Multi-timescale control hierarchy: dApp RT(<1ms) → xApp(10ms-1s) → rApp(>1s). Per-slice CE policy 가능.
- **CE-skip relevance**: ★★★☆☆

---

### L4: Adaptive/Event-Triggered Inference

CE-skip의 방법론적 선행 연구. Event-triggered 패러다임, adaptive computation, threshold 기반 결정.

#### [P29] Communication Efficient Cooperative Edge AI via Event-Triggered Offloading
- **CE-skip 관련**: CE-skip의 가장 가까운 방법론적 유사체. **Dual-threshold early-exit** (confidence < β_l → routine exit, > β_u → offload, 중간 → 다음 block). Missing-target-offloading tradeoff가 CE-skip의 skip-vs-recompute tradeoff와 구조적 동일.
- **CE-skip relevance**: ★★★★★

#### [P35] DL-Based Beam Management for mmWave Vehicular (DeepBT)
- **CE-skip 관련**: **Prediction-aided measurement substitution** — 3회 중 2회를 prediction으로 대체하여 66.7% overhead 감소. Beam domain에서의 CE-skip 정확한 유사체.
- **CE-skip relevance**: ★★★★★

#### [P56] 5G-Advanced AI/ML Beam Management (3GPP, Nokia)
- **CE-skip 관련**: 3GPP Rel-18 호환 SBP/TBP 평가. MOR (Measurement Overhead Reduction) metric — CE-skip의 평가 지표로 직접 적용 가능. TBP가 속도 2-4x에서 6.7% accuracy loss만 보임.
- **CE-skip relevance**: ★★★★☆

#### [P41] Rethinking Beam Management: Generalization Under HW Heterogeneity
- **CE-skip 관련**: ML beam predictor가 heterogeneity에서 >50% SE drop — per-site model 필요성. 15 GHz, 8x8 UPA가 본 프로젝트 config와 유사.
- **CE-skip relevance**: ★★★★☆

---

### L5: Beamforming with Stale/Imperfect CSI

CE-skip으로 인한 stale CSI가 beamforming 성능에 미치는 영향. CE-skip의 QoS guarantee 설계에 필수적.

#### [P38] Data and Model-Driven DL Beamforming (GNN Robust BF)
- **CE-skip 관련**: CSI uncertainty를 Gaussian error로 모델링한 robust BF. DAQE로 channel error augmented training → stale CSI에 robust한 BF. 5% outage probability constraint로 QoS 보장.
- **CE-skip relevance**: ★★★★★

#### [P49] FL Strategies for Coordinated Beamforming in Multicell ISAC
- **CE-skip 관련**: Multi-cell coordinated BF에서 HFL이 CSI staleness에 robust (local CSI만 사용). CE-skip의 한 BS skip 결정이 인접 BS ICI에 영향 → HFL 접근법이 적합.
- **CE-skip relevance**: ★★★★☆

#### [P10] Robust FL for Wireless Channel Estimation
- **CE-skip 관련**: "Outdate mode" — outdated CSI 제공이 minimal impact ← **strong temporal correlation**. CE-skip의 전제를 실험적으로 지지.
- **CE-skip relevance**: ★★★★☆

#### [P50] Personalized FL-Driven Beamforming for ISAC
- **CE-skip 관련**: Multi-BS PFL에서 BS별 adaptive aggregation. MATLAB ray-tracing 채널 → Sionna RT와 유사 방법론.
- **CE-skip relevance**: ★★★☆☆

---

### L6: Near-Field ELAA CE Methods

ELAA-specific CE 방법론. 대부분 static channel 가정 → temporal scheduling gap 확인.

#### [P92] Distributed Signal Processing for ELAA Systems
- **CE-skip 관련**: LMMSE complexity cubic scaling. "CE must be done per coherence interval" — coherence interval이 곧 skip interval.
- **CE-skip relevance**: ★★★★☆

#### [P97] Channel Estimation for XL-MIMO with Decentralized Baseband Processing
- **CE-skip 관련**: SBL-GNNs, 0.004s vs 2.379s (centralized). Per-subarray skip 결정 가능.
- **CE-skip relevance**: ★★★★☆

#### [P99] Integrated Channel Estimation and Sensing for Near-Field ELAA
- **CE-skip 관련**: Position 정보가 CE-skip trigger로 활용 가능: 위치 불변 → skip CE.
- **CE-skip relevance**: ★★★★☆

#### [P91] Recent Advances in Near-Field Beam Training and CE for XL-MIMO
- **CE-skip 관련**: No existing work on temporal/scheduling aspects — gap 확인.
- **CE-skip relevance**: ★★★☆☆

#### [P89] A Tutorial on Near-Field XL-MIMO Communications Towards 6G
- **CE-skip 관련**: 3.5 GHz/256 ant → Rayleigh distance ~100m. Near-field channel model 기초.
- **CE-skip relevance**: ★★★☆☆

#### [P105] U6G XL-MIMO Radiomap Prediction: Multi-Config Dataset
- **CE-skip 관련**: Sionna RT 기반 XL-MIMO dataset (up to 32x32 UPA, 1.8-6.7 GHz).
- **CE-skip relevance**: ★★★☆☆

#### [P99b] Near-Field Beamfocusing with Modular Linear Arrays
- **CE-skip 관련**: Per-ULA parametric CE = low complexity → skip 이득 감소. 15 GHz mid-band.
- **CE-skip relevance**: ★★★☆☆

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
| NF CE Survey | P90 | Various | mmW/THz | - | - | Multiple NF | Doppler | Survey |
| Sub-Array Pilot | P98 | 1024 (128 sub) | **2.6** | 15 kHz/SC | 1024 | COST2100 | Static* | Semi-urban |
| dApps | P02 | - | - | - | 384-2048 IQ | Real 5G | - | OAI Testbed |
| XAI-on-RAN | P07 | 4T4R | 5G SA | - | - | Real 5G | Robot UE | **NVIDIA Aerial** |
| AI-RAN Conv. | P01 | - | - | - | - | Real 5G | - | X5G (A100) |
| Event-Triggered | P29 | - | - | 30 MHz | - | Rayleigh fading | - | Medical imaging |
| DeepBT | P35 | - | mmWave | - | OFDM | Ray-tracing | Vehicular | Marseille/Rosslyn |
| 5G-Adv BM | P56 | UPA | FR2 | - | - | 3GPP UMa | 3-120 km/h | 3GPP eval |
| Robust BF | P38 | 4 TX | - | 10 | N/A | Rayleigh | Static | Single cell |
| FL ISAC BF | P49 | 6+6 | - | - | N/A | Rician(3) | Static | 500m cell |
| Robust FL CE | P10 | CNN input | mmWave | - | 612 | MATLAB 5G | - | 10 SBS + 1 MBS |
| Distributed SP | P92 | 128-1024 | - | 80 | 192 | ELAA cluster | - | Framework |
| Decentralized CE | P97 | 128 ULA (4 sub) | **28** | 1.6 GHz | 16 | NF + dual-WB | Static | Synthetic |
| ISAC NF ELAA | P99 | 256/128 ULA | 100/**28** | 0.1 GHz | 64 | NF spherical | Static | Synthetic |
| **Ours (CE-skip)** | - | **8x8/16x16/32x32 UPA** | **3.5/15/28** | **100-1600** | **256-4096** | **Sionna RT** | **0-33 m/s** | **Munich UMi, 8 BS** |

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

#### Gap 3: Event-Triggered CE (Not Periodic)
- Beam management의 prediction-aided substitution [P35, P56]은 beam domain에서 유사한 철학이나, CE domain에 적용한 사례 없음.
- Event-triggered inference [P29]는 image classification에 적용; PHY-layer CE에 적용한 사례 없음.

#### Gap 4: Multi-Tier Adaptive CE
- 기존 CE는 단일 method 고정 (LS 또는 LMMSE 또는 DL-CE). 상황에 따라 CE method를 전환하는 multi-tier 접근 없음.
- CE-skip의 3-tier (Skip / Delta / Full)는 새로운 조합.

#### Gap 5: Stale CSI에 대한 BF 영향의 CE Scheduling 관점 분석
- Robust BF [P38]는 CSI error 모델링을 하지만 CE scheduling과 연결하지 않음.
- CE skip으로 인한 specific CSI aging pattern (temporally correlated staleness)을 BF 성능과 연결한 분석 없음.

---

*Updated: 2026-03-18 | 105 papers + Cluster L (41 CE-skip cross-references) analyzed*
