# Unit 7: Remaining Papers Scan for CE-Skip Relevance

**Scope:** 44 papers not covered by Units 1-6 (edge AI, LLM deployment, split inference, CSI feedback autoencoders, semantic communication, differentiable ray tracing, O-RAN architecture, misc).

**CE-skip context:** "Event-Triggered CE Inference Scheduling for Software-Defined Base Stations" -- adaptive skip of CE inference in 6G ELAA GPU-native RAN. Key topics: temporal channel persistence, adaptive computation, GPU kernel scheduling, beamforming with stale CSI.

---

## Papers WITH CE-Skip Relevance

### 1. LLM_MultiTask_Physical_Layer_2412.20772 -- MODERATE RELEVANCE

**Why relevant:** LLM-based multi-task PHY network performing channel prediction, signal detection, and multi-user precoding simultaneously. Channel prediction component explicitly handles temporal CSI prediction (predict T2=4 future slots from T1=16 historical slots), which is conceptually adjacent to CE-skip's temporal persistence assumption.

**Key insight for CE-skip:** The channel prediction task demonstrates that temporal dependencies in CSI can be captured by attention-based models. The paper shows NMSE degrades at higher velocities (channel aging), directly validating that temporal coherence is velocity-dependent -- exactly what CE-skip's adaptive trigger should exploit.

**Dataset config (QuaDRiGa, 3GPP UMa NLOS):**
- BS: UPA 16x8 = 128 antennas, single-antenna UEs
- K = 4-8 users, fc = 2.4 GHz, BW = 8.64 MHz, M = 48 subcarriers
- TDD, velocity 10-100 km/h uniform, SNR 5-20 dB
- T1 = 16 historical slots, T2 = 4 predicted slots
- 50k train / 10k test samples per task
- Achieves ~-20 dB NMSE at 10 km/h, degrades to ~-8 dB at 100 km/h

**Citable as:** Baseline for temporal channel prediction; evidence that channel aging rate is the key factor determining when CE must be refreshed.

---

### 2. Edge_Large_AI_Models_6G_2505.00321 -- MODERATE RELEVANCE

**Why relevant:** Proposes federated LAM for channel prediction with explicit discussion of bypassing explicit channel estimation. The beamforming section states: "end-to-end design that bypasses explicit channel estimation and directly utilizes noisy pilots can be further integrated into the graph LAM-based beamforming design framework, thereby mitigating the performance degradation due to the objective mismatch between the beamforming design and channel estimation modules."

**Key insight for CE-skip:** This "bypass CE" philosophy is complementary to CE-skip. Where they bypass CE entirely via end-to-end learning, CE-skip selectively skips CE when channel is temporally stable. Both address the same underlying problem: CE overhead is excessive in ELAA.

**Dataset config (QuaDRiGa, 3GPP-compliant):**
- LLM4CP base model, up to 10 edge servers
- 5% IID channel samples per server
- Federated LAM with LoRA (5% of parameters)

**Citable as:** Motivation that CE overhead reduction is a recognized problem; alternative approach (end-to-end bypass vs. selective skip).

---

### 3. Beyond_Connectivity_AI_RAN_Convergence_6G_2507.06911 -- MODERATE RELEVANCE

**Why relevant:** This is the most architecturally relevant paper for CE-skip's GPU coexistence story. It presents an AI-RAN convergence architecture where RAN DSP and AI workloads share the same GPU infrastructure, with concrete experimental profiling on NVIDIA A100 with GPU-accelerated physical layer (NVIDIA Aerial + OpenAirInterface).

**Key findings for CE-skip:**
- "RAN DSP represents a non-elastic workload, where tasks need to be executed within specific timing constraints" -- exactly the constraint CE-skip must respect
- GPU partitioned via MIG: 40 GB for RAN, 20 GB for LLM, 10 GB for CNN
- AI-RAN coexistence validated: throughput and CRC error rate consistent even with simultaneous AI workloads
- Workload deployment latency: RAN + ResNet in 1-4s, LLM in 5.8-34.4s
- "careful scheduling to ensure energy efficiency" for AI accelerators

**Citable as:** Evidence that GPU-native RAN is real (NVIDIA Aerial on A100); RAN DSP timing constraints that CE-skip must satisfy; GPU resource partitioning precedent.

---

### 4. Towards_AI_Native_RAN_Operator_Perspective_2507.08403 -- LOW-MODERATE RELEVANCE

**Why relevant:** China Mobile's 6G AI-Native RAN paper with 5000+ 5G-A BS field trial. Mentions "AI-driven RAN processing" including "low-complexity signal processing for massive MIMO" and "AI-based RS overhead reduction" as 6G use cases. Also discusses GPU/NPU resources within RAN and scheduling tasks.

**Key for CE-skip:** Validates the operator-side motivation for CE-skip: RS (reference signal) overhead reduction is an explicit 6G standardization goal. CE-skip directly achieves this by reducing how often CE inference runs.

**Citable as:** Operator/standardization motivation; RS overhead reduction as 6G priority.

---

### 5. AI_ML_Lifecycle_Mgmt_AI_Native_RAN_2507.18538 -- LOW RELEVANCE

**Why relevant:** 3GPP Rel-16-20 AI/ML lifecycle management for RAN. Discusses model drift detection, KPI-driven monitoring, and two-sided CSI compression. The LCM framework (model training, transfer, execution, monitoring, control) could apply to CE-skip's trigger model lifecycle.

**Citable as:** Standardization context for deploying AI models (like CE-skip trigger) in RAN.

---

### 6. Dynamic_Encoding_Decoding_Split_Learning_2309.02787 -- LOW RELEVANCE

**Why relevant:** Information Bottleneck (IB) theory applied to split learning with adaptive encoding depth based on network conditions. The concept of "multiple modes of complexity-relevance tradeoffs" maps loosely to CE-skip's adaptive computation idea. Applied to mmWave throughput prediction.

**Key insight:** IB-theoretic justification for adaptive computation depth -- could inspire theoretical framing of CE-skip as an information bottleneck problem (when does stale CSI contain enough information?).

**Citable as:** Theoretical precedent for adaptive computation depth in wireless.

---

## Papers NOT Relevant

### Edge AI / LLM / Split Inference:
- **Efficient_Large_AI_Inference_Wireless_Edge_2505.09214** -- NOT RELEVANT (pruning-aware LAIM co-inference for LLM deployment; no CE/channel topics)
- **Generative_AI_on_the_Edge_2411.17712** -- NOT RELEVANT (LLM benchmarking on Raspberry Pi for O-RAN; no channel estimation)
- **How_Small_Can_6G_Reason_Tiny_LM_2603.02156** -- NOT RELEVANT (tiny LLM scaling for network-level semantic reasoning; no PHY layer CE)
- **LLM_Empowered_IoT_6G_2503.13819** -- NOT RELEVANT (LLM architecture for IoT; split federated learning for LLM fine-tuning, not CE)
- **On_Device_AI_Models_Survey_2503.06027** -- NOT RELEVANT (general survey on on-device AI deployment; no channel estimation specifics)
- **Optimizing_Edge_AI_Survey_2501.03265** -- NOT RELEVANT (cognitive edge computing survey; LLM optimization techniques, no CE)
- **Pushing_LLMs_to_6G_Edge_2309.16739** -- NOT RELEVANT (LLM deployment at 6G edge via split learning/inference; no CE topics)
- **TinyLLM_Edge_Deployment_2412.15304** -- NOT RELEVANT (training 30-120M LLMs for edge sensing applications; no CE)
- **Adaptive_Layer_Splitting_Wireless_LLM** -- NOT RELEVANT (RL-based LLM split point optimization; about NLP inference, not CE)
- **Split_Learning_6G_Edge_Networks_2306.12194** -- NOT RELEVANT (split learning survey for 6G; focuses on FL alternative, not CE)
- **Splitwise_Collaborative_Edge_Cloud_LLM_2512.23310** -- NOT RELEVANT (Lyapunov-DRL for LLM edge-cloud partitioning; no CE)
- **SLIDE_Simultaneous_Model_Download_Inference_2512.20946** -- NOT RELEVANT (simultaneous model downloading and inference; resource allocation, no CE)
- **CNN_Collaborative_Inference_Heterogeneous_Edge** -- NOT RELEVANT (CNN partitioning across heterogeneous edge devices; no wireless channel)
- **MAE_Collaborative_Inference_DNN_Partitioning** -- NOT RELEVANT (MoE-based DNN partitioning for edge inference; no CE)

### O-RAN Architecture:
- **Towards_6G_Native_AI_Edge_Networks_2512.04405** -- NOT RELEVANT (semantic communication + agentic intelligence for 6G; no CE specifics)
- **Toward_E2E_Intelligence_6G_AI_Agent_RAN_CN_2602.23623** -- NOT RELEVANT (LLM-based AI agent for RAN-CN control; no CE inference scheduling)
- **Large_GenAI_Models_Open_Networks_6G_2410.18790** -- NOT RELEVANT (GAI marketplace platform for O-RAN; monetization focus, no CE)
- **MX_AI_Agentic_Platform_Open_AI_RAN_2508.09197** -- NOT RELEVANT (LLM agent for O-RAN observability/control; no PHY layer CE)

### CSI Feedback / Semantic Communication:
- **Universal_AE_MIMO_CSI_Feedback_2403.00299** -- NOT RELEVANT (universal autoencoder for CSI compression with variable input sizes; FDD feedback, not CE inference scheduling)
- **VQ_VAE_CSI_Feedback_Massive_MIMO_2403.07355** -- NOT RELEVANT (vector quantization for CSI feedback; codebook design, not CE)
- **Precoding_Oriented_CSI_VQ_VAE_2602.02508** -- NOT RELEVANT (VQ-VAE for precoding-oriented CSI feedback; mutual information regularization, no temporal/skip)
- **SAFE_Semantic_Adaptive_Feature_6G_2410.01597** -- NOT RELEVANT (semantic communication with adaptive sub-semantic selection; image transmission, not CE)
- **Robust_JSCC_Task_Oriented_Semantic_2503.12907** -- NOT RELEVANT (KL-divergence regularization for robust JSCC; semantic comm, not CE)
- **DL_Autoencoder_Review_NextGen_Comm_2412.13843** -- NOT RELEVANT (comprehensive AE survey for comm systems; general review, no CE-skip concepts)
- **Mobile_Edge_Generation_Distributed_GenAI_6G_2409.05870** -- NOT RELEVANT (GenAI deployment at edge for text-to-image; no CE)
- **Semantic_Edge_Computing_6G_2411.18199** -- NOT RELEVANT (survey unifying semantic edge computing and semantic comms; no CE inference scheduling)

### Differentiable Ray Tracing:
- **Differentiable_Ray_Tracing_Learning_Radio_Env_2311.18558** -- NOT RELEVANT (gradient-based RT calibration for material/antenna properties; no CE)
- **Fast_Diff_GPU_RT_Implicit_2510.16172** -- NOT RELEVANT (GPU-accelerated differentiable RT for diffraction/reflection paths; algorithmic, no CE)
- **Fully_Diff_RT_Discontinuity_2401.11882** -- NOT RELEVANT (discontinuity smoothing for differentiable RT; optimization technique, no CE)
- **Diff_vs_DL_Radio_Propagation_2509.19337** -- NOT RELEVANT (comparing diff RT vs DL for radio propagation on real MNO data; no CE scheduling)
- **VLM_Guided_Diff_RT_RF_Param_2601.18242** -- NOT RELEVANT (VLM-guided RF material parameter estimation via inverse RT; no CE)
- **Site_Specific_RIS_Calibrated_RT_2510.09478** -- NOT RELEVANT (RIS deployment optimization via calibrated Sionna RT; no CE)
- **U6G_XL_MIMO_Radiomap_BeamMap_2603.06401** -- NOT RELEVANT (XL-MIMO radiomap prediction dataset with beam maps; coverage prediction, not CE inference)

### Miscellaneous:
- **Integrated_Sensing_Edge_AI_6G_Survey_2501.06726** -- NOT RELEVANT (ISEA survey; mentions CE tangentially but focuses on sensing-AI integration, no CE-skip)
- **CoMP_AI_Model_Caching_Downloading_2509.19341** -- NOT RELEVANT (AI model caching/downloading with CoMP broadcasting; no CE)
- **Bridging_6G_IoT_LLM_Physical_Layer_2602.06819** -- NOT RELEVANT (prompt engineering for PHY optimization; constellation design, no CE)
- **Training_ML_at_Edge_Survey_2403.02619** -- NOT RELEVANT (edge ML training survey; federated/split learning methods, no CE)
- **Machine_Intelligence_Wireless_Edge_2506.12210** -- NOT RELEVANT (in-physics computation via RF analog inner products; novel but unrelated to CE scheduling)

---

## Summary of Findings

### Hidden Gems Found: 3 papers with meaningful CE-skip relevance

1. **Beyond_Connectivity (Polese et al., 2507.06911)** -- The strongest find. Provides concrete GPU-native RAN profiling data (NVIDIA Aerial on A100, MIG partitioning) that directly validates CE-skip's architectural assumptions. The RAN DSP timing constraint discussion and AI-RAN coexistence experiments are directly citable for our GPU scheduling story.

2. **LLM_MultiTask_Physical_Layer (Zheng & Dai, 2412.20772)** -- Channel prediction dataset with velocity-dependent NMSE degradation (10 km/h: -20 dB, 100 km/h: -8 dB). This velocity-NMSE curve directly supports CE-skip's adaptive trigger design: skip more aggressively at low velocity (high temporal coherence), skip less at high velocity.

3. **Edge_Large_AI_Models (Wang et al., 2505.00321)** -- Explicit discussion of "bypassing explicit CE" in beamforming design. Positions CE overhead as a recognized problem in ELAA systems, providing motivation for CE-skip as an alternative (selective skip vs. full bypass).

### Secondary finds:
- **Towards_AI_Native_RAN (2507.08403)** -- RS overhead reduction as explicit 6G standardization goal (China Mobile field trial)
- **AI_ML_Lifecycle_Mgmt (2507.18538)** -- 3GPP LCM framework applicable to CE-skip model deployment
- **Dynamic_Encoding_Decoding (2309.02787)** -- IB-theoretic adaptive computation concept

### Overall assessment:
The remaining 44 papers are overwhelmingly not relevant to CE-skip. The three hidden gems are useful but not transformative -- they provide supporting evidence (GPU architecture validation, velocity-dependent temporal coherence, CE overhead motivation) rather than direct methodological contributions. The core CE-skip literature remains in Units 1-6.
