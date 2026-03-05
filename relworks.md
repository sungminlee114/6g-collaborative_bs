# Related Works: 6G O-RAN + On-device AI for Collaborative Base Stations

> 46 papers (2023–2026.03) 종합 분석. Sionna RT 기반 15 GHz mmWave 채널 데이터셋, 8 BS 협력, on-device AI 관점.

---

## Table of Contents
1. [논문 군집 (Clusters) 및 스토리라인](#1-논문-군집-및-스토리라인)
2. [Cluster A: O-RAN / AI-RAN 아키텍처](#cluster-a-o-ran--ai-ran-아키텍처)
3. [Cluster B: Edge AI 추론 최적화](#cluster-b-edge-ai-추론-최적화)
4. [Cluster C: LLM/SLM for 6G Networks](#cluster-c-llmslm-for-6g-networks)
5. [Cluster D: 협력/분할 추론 (Split & Collaborative Inference)](#cluster-d-협력분할-추론)
6. [Cluster E: 빔 관리 & 채널 추정 (Beam Management & Channel Estimation)](#cluster-e-빔-관리--채널-추정)
7. [Cluster F: 시뮬레이션 / 데이터셋 / 디지털 트윈](#cluster-f-시뮬레이션--데이터셋--디지털-트윈)
8. [Cross-Cluster 분석 & 연구 갭](#2-cross-cluster-분석--연구-갭)
9. [Nature Communications 연구주제 후보](#3-nature-communications-연구주제-후보)

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

## 2. Cross-Cluster 분석 & 연구 갭

### 2.1 논문 간 연결 관계

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

### 2.2 주요 연구 갭 종합

| 갭 ID | 설명 | 관련 논문 | 난이도 |
|-------|------|----------|-------|
| G1 | On-device AI (dApp/xApp)의 HW 이질성 강건한 빔 관리 | P02,P34,P36,P41 | ★★★★★ |
| G2 | 동적 채널 + 보안 + FL 통합 채널 예측 | P09,P10,P14 | ★★★★☆ |
| G3 | SLM (1-3B) on-device PHY 다중과제 + O-RAN 배포 | P19,P22,P21,P31 | ★★★★★ |
| G4 | 디지털 트윈 → 실세계 전이학습 체계 | P43,P44,P34,P06 | ★★★★☆ |
| G5 | Edge LAM의 마이크로서비스 기반 협력 추론 + 무선 최적화 | P14,P26,P30 | ★★★★☆ |
| G6 | 모델 버전 관리 + 동적 업데이트의 안정성-성능 트레이드오프 | P08,P41 | ★★★☆☆ |
| G7 | ISCC 기반 센싱 지원 협력 빔 관리 | P16,P32 | ★★★★☆ |

---

## 3. Nature Communications 연구주제 후보

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

## Impact Score 요약표

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
