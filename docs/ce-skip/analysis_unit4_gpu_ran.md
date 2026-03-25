# Unit 4: GPU-Native RAN, dApps, and Adaptive Inference

Analysis of 9 papers for the CE-skip paper: "Event-Triggered CE Inference Scheduling for Software-Defined Base Stations"

---

## Paper-by-Paper Analysis

---

### 1. dApps: Enabling Real-Time AI-Based Open RAN Control
**Lacava et al., Computer Networks 2025 (arXiv 2501.16502)**

**CE-skip Relevance: HIGH**

This is the single most important paper for CE-skip's architectural positioning. It defines dApps as sub-10ms control applications co-located with DU/CU -- exactly the deployment model CE-skip's monitor tier needs.

**Architecture Details:**
- dApps deploy directly on CU/DU as lightweight microservices (contrast with RT-RIC approaches that require a new platform)
- New E3 interface connects dApps to RAN functions via IPC (shared memory, Unix domain sockets)
- E3 Agent within DU/CU exposes I/Q samples, CSI, CQI, BSR, SRS data to dApps
- E2SM-DAPP bridges dApp outputs to xApps on Near-RT RIC
- Timescale: < 1 ms (vs xApps 10ms-1s, rApps >1s)

**Latency Measurements (critical for CE-skip):**
- Average control loop latency: ~400 us, consistently below 450 us
- Breakdown (ZMQ + IPC, best config):
  - Collect Data (I/Q extraction via T-tracer): ~50-80 us
  - Process Data (ASN.1 decode + inference logic): ~200-250 us (dominant)
  - Create Control (generate E3 Control Message): ~30-50 us
  - Deliver Control (send to RAN): ~30-50 us
- Tested with 384-2048 I/Q samples (1536-8192 bytes indication payload)
- IPC eliminates protocol overhead entirely (vs TCP 20%, SCTP 42%)

**CE-skip use case mapping:**
- The paper explicitly lists "Augmented Sensing and Channel Estimation" as a dApp use case (Section 3):
  > "dApps can enable dynamic CSI compression... as well as custom AI/ML models for channel estimation that can be tailored to specific user conditions"
- Also lists "channel equalization" in the nGRG research report use cases
- PRB masking control action (spectrum sharing use case) is analogous to CE-skip's "skip CE for these subframes" control action

**Key quotes for CE-skip paper:**
- "dApps access data otherwise unavailable to RICs due to privacy or timing constraints"
- "Control actions must be applied within a short interval (e.g., 0.5 ms)"
- "Real-time control loops via dApps are feasible, achieving average control latency below 450 microseconds"

**What CE-skip can cite:**
- dApp architecture as the deployment vehicle for the monitor+scheduler tiers
- 450 us control loop as evidence that sub-ms CE skip decisions are feasible
- E3 interface for exposing channel estimation KPMs to the skip scheduler
- Channel estimation explicitly listed as a dApp-eligible function

---

### 2. Accelerating vRAN and O-RAN with SIMD (Chae et al.)
**Park, Chae, Heath; arXiv 2510.07843 (Oct 2025)**

**CE-skip Relevance: MEDIUM**

Provides PHY processing timing data for CPU-based vRAN, which is the alternative to GPU-native RAN. Useful as a comparison point.

**Architecture Details:**
- Focus: SIMD (AVX2/AVX-512) acceleration for PHY on x86-64 COTS CPUs
- Software-defined DU with LMMSE MIMO detection as primary workload
- No GPU -- purely CPU-based processing

**PHY Processing Timing (key data):**
- LMMSE MIMO detection processing time per 1 TTI (1 ms):
  - 2x2 MIMO: ~0.005 ms total (SIMD-PS), dominated by matrix inversion
  - 4x4 MIMO: ~0.03 ms total (SIMD-PS), 50% speedup over scalar baseline
  - 8x8 MIMO: ~5-16 ms total (baseline), needs SIMD to fit in TTI
- Breakdown: COV (covariance), INV (inversion), W_mmse (weight calc), EQ (equalization)
- Matrix inversion is the computational bottleneck
- SIMD-PS (single precision) achieves comparable accuracy to PD (double precision)

**Comparison table (Table I): SIMD vs GPU vs FPGA**
| Metric | SIMD (CPU) | SIMT (GPU) | FPGA |
|--------|-----------|------------|------|
| Latency | Low | Medium | Very Low |
| Flexibility | High | Medium | Low |
| Deployment | Low | High | Very High |
| Energy Eff. | Medium | Low | High |

**CE-skip relevance:**
- Positions GPU as "medium latency" for PHY -- CE-skip argues GPU latency is predictable and profiled, enabling skip scheduling
- SIMD paper focuses on MIMO detection, not CE specifically, but the same LMMSE computation applies to CE
- CE is explicitly mentioned as a SIMD-accelerable PHY function but not benchmarked
- The 4x4 MIMO detection at 0.03 ms shows that CE computations are fast -- the scheduling overhead of CE-skip must be even smaller

---

### 3. Self-Learning Model Versioning for AI-native O-RAN Edge
**Bensalem et al., arXiv 2601.17534 (Jan 2026)**

**CE-skip Relevance: MEDIUM**

Addresses ML model lifecycle management across O-RAN's three control loops. Relevant for CE-skip's model update/versioning story.

**Architecture Details:**
- Multi-layer: Cell-Site (<1ms), Edge Cloud (<10ms), Regional Cloud (10ms-1s), Central Cloud (>1s)
- Explicitly includes dApps at Cell-Site layer for <1ms real-time loop
- Update Manager + Version Repository + RL-driven update policy
- Container orchestrator handles deployment across heterogeneous workers

**Key insight for CE-skip:**
- RL policy "favors stability over accuracy improvement for dApps" -- because dApp model updates cause the highest disruption (real-time constraints)
- This validates CE-skip's design choice: the CE model itself should be stable, while the skip scheduler adapts
- Model versioning framework could manage CE model updates (e.g., switching between LS/LMMSE/DL-CE)

**Inference scheduling relevance:**
- Not directly about adaptive computation frequency
- But the update decision (when to swap model versions) is analogous to CE-skip's decision (when to run CE vs skip)
- Both are event-triggered: update when conditions change

---

### 4. Communication Efficient Cooperative Edge AI via Event-Triggered Computation Offloading
**Zhou, You, Huang; arXiv 2501.02001 (Jan 2025)**

**CE-skip Relevance: HIGH** (methodological)

The closest methodological analog to CE-skip. Event-triggered inference with dual-threshold early-exit architecture.

**Architecture Details:**
- Edge device + edge server split inference
- Dual-threshold multi-exit CNN architecture
- Channel-adaptive offloading policy

**Event-triggered mechanism:**
- Binary tail-event detection at device, rare events offloaded to server
- Dual thresholds create uncertainty region: confident events exit early, uncertain events continue processing
- Channel-adaptive: offloading decision depends on real-time channel state
- Optimization: maximize rare-event classification accuracy under energy/communication constraints
- Non-convex problem reformulated as strongly convex

**CE-skip mapping:**
- CE-skip's "skip vs compute" decision is analogous to "local exit vs offload"
- CE-skip's channel stability metric maps to confidence threshold
- CE-skip's event trigger (mobility, SNR change) maps to rare-event detection
- The dual-threshold idea could enhance CE-skip: low-confidence = must recompute CE, high-confidence = safe to skip, middle = use interpolated CE
- Channel-adaptive policy: CE-skip should also adapt skip aggressiveness to channel conditions

**Differences:**
- This paper: device-server split inference for classification tasks
- CE-skip: within-BS decision about CE computation frequency for PHY processing
- No RAN/O-RAN architecture mapping
- No GPU kernel considerations

---

### 5. Energy-Efficient Edge Inference in ISCC Networks
**Yao, Xu, Zhu, Huang, Cui; arXiv 2503.00298 (Mar 2025)**

**CE-skip Relevance: LOW**

General edge inference optimization (sensing + communication + computation). No direct RAN/PHY relevance.

**Architecture Details:**
- Split inference with flexible splitting point
- Model pruning + feature quantization
- Joint resource allocation for ISCC (integrated sensing, communication, computation)

**Potentially useful concepts:**
- Explicit inference accuracy characterization as function of SCC resources
- Energy minimization under accuracy + latency constraints
- The idea that sensing quality determines achievable accuracy -- analogous to CE-skip: channel coherence determines how long you can skip CE

**CE-skip relevance:**
- The accuracy-vs-compute tradeoff formulation could inspire CE-skip's theoretical framework
- But the paper operates at a fundamentally different layer (edge device classification, not PHY processing)

---

### 6. XAI-on-RAN: Explainable, AI-native, and GPU-Accelerated RAN Towards 6G
**Basaran, Dressler; NeurIPS AI4NextG Workshop 2025 (arXiv 2511.17514)**

**CE-skip Relevance: HIGH**

First-hand GPU-accelerated RAN testbed with NVIDIA Aerial. Provides concrete GPU utilization and latency data.

**Architecture Details:**
- Hardware: NVIDIA A100 GPU, Gigabyte E251-U70 servers, Foxconn RPQN7801 4T4R RU
- Software: NVIDIA Aerial (cuPHY) for L1/L2 on GPU, OAI for higher layers
- O-RAN SC RIC (E-release) with E2 interface to DU
- Split 7.2 fronthaul between DU and RU
- xApps: Traffic Predictor (TP), KPM monitor, XAI-Native

**GPU Processing Data (Table 2, critical for CE-skip):**
| Model | AI Inference T(inf) | XAI T(xai) | Total T(total) | GPU Util |
|-------|-------------------|-----------|--------------|----------|
| Non-XAI (Baseline) | 5.1 ms | -- | 5.3 ms | ~63% |
| XAI (SHAP, m=16) | 5.2 ms | ~15 ms | ~20.4 ms | ~86% |
| XAI (Attention only) | 5.2 ms | 0.6 ms | 5.9 ms | ~70% |
| Ours (Attention+IG, k=5) | 5.2 ms | 2.8 ms | 8.1 ms | ~73% |

**Key findings for CE-skip:**
- **GPU utilization baseline is only ~63%** -- significant headroom exists for CE-skip's monitor tier
- LSTM inference on A100: 5.1 ms per cycle
- Adding lightweight computation (attention) costs only 0.6 ms extra
- The GPU is "not fully saturated" even with XAI -- validates CE-skip's premise that CE kernels can be selectively scheduled without starving other PHY tasks
- Communication overhead between RIC components: negligible due to co-location (~0.2 ms)

**CE-skip can cite:**
- A100 GPU utilization data as evidence of available compute headroom
- NVIDIA Aerial as the concrete cuPHY platform CE-skip targets
- The latency decomposition methodology (T_inf + T_xai + T_comm) maps directly to CE-skip's (T_CE + T_monitor + T_control)
- Real-time constraint: 10 ms near-RT RIC cycle is feasible with GPU acceleration

**Testbed details:**
- COTS UE on remote-controlled robot (mobility testing)
- Live KPM monitoring via Grafana dashboard
- 5G SA mode, split 7.2

---

### 7. Distributed AI Platform for the 6G RAN
**Ananthanarayanan et al., Microsoft Research; arXiv 2410.03747 (Oct 2024)**

**CE-skip Relevance: MEDIUM**

Platform architecture for AI-native RAN. Strong on orchestration concepts but no PHY-level timing data.

**Architecture Details:**
- Three-tier: far edge (<1ms, CPU-only), near edge (1-10ms, CPU+few GPUs), cloud (>10ms, CPU+many GPUs)
- Programmable probes (eBPF-based) for dynamic data collection from RAN NFs
- AI processor runtime with shared-memory fast IO, WASM execution environment
- Distributed AI orchestrator for placement decisions

**Key concepts for CE-skip:**
- "Sub-millisecond reaction times" required for far-edge AI runtime
- eBPF probes for instrumenting RAN NFs -- could expose CE kernel execution state
- AI application graphs: chained models with reducing data volume left-to-right
- Inference parameters as orchestrator knobs (e.g., data sampling rates, model accuracy tradeoffs)
- GPU co-location with RAN workloads: "AI-and-RAN" paradigm (share compute between RAN and AI apps)
- "vRAN is largely underutilized (<50%)" -- supports CE-skip's compute headroom argument

**CE-skip mapping:**
- CE-skip's monitor tier = a far-edge AI app block
- CE-skip's skip scheduler = another AI app block that chains from the monitor
- The orchestrator concept maps to CE-skip's scheduling policy management
- Inference parameters as knobs: CE-skip's skip threshold is exactly such a parameter

---

### 8. REAL: Reinforcement Learning-Enabled xApps for Experimental Closed-Loop Optimization in O-RAN
**Barker et al., Clemson University; arXiv 2502.00715 (Feb 2025)**

**CE-skip Relevance: LOW-MEDIUM**

Full-stack O-RAN testbed with OSC RIC + srsRAN for RL-based network slicing. Demonstrates practical challenges of real-time AI control in O-RAN.

**Architecture Details:**
- OSC Near-RT RIC + srsRAN 5G stack + Open5GS core
- E2 interface for PRB allocation control
- GNU Radio for channel modeling (FSPL, single-tap fading, AWGN, Doppler)
- PPO (Proximal Policy Optimization) actor-critic RL for slice resource allocation
- Up to 12 UEs across 3 slices (URLLC, eMBB, mMTC)

**Practical constraints revealed:**
- KPI sampling rate: 500 ms (far too slow for CE-skip's sub-ms needs)
- ZeroMQ saturation with >3 simultaneous UEs
- CPU-only environment (24-core i9-14900K) -- no GPU
- Online training: RL agent trains during simulation, not offline

**CE-skip relevance:**
- Demonstrates that xApp-level control (10ms-1s timescale) is too slow for PHY-level decisions like CE skip
- Validates the need for dApp-level (<1ms) control for CE-skip
- The reward function design (per-slice QoS) could inspire CE-skip's reward: per-UE CE accuracy vs compute savings
- Scalability issues (ZeroMQ saturation) highlight practical deployment challenges

---

### 9. Collaborative Edge AI Inference over Cloud-RAN
**Zhang et al., ShanghaiTech; arXiv 2404.06007 (Apr 2024)**

**CE-skip Relevance: LOW**

Cloud-RAN based edge inference with AirComp feature aggregation. Purely communication-theoretic, no PHY processing or scheduling relevance.

**Architecture Details:**
- Cloud-RAN: RRHs connected to central processor (CP) via capacity-limited fronthaul
- Over-the-air computation (AirComp) for feature vector aggregation
- Task-oriented design using discriminant gain metric
- Joint optimization of transmit precoding, receive beamforming, quantization

**Minimal CE-skip connection:**
- Cloud-RAN fronthaul quantization relates tangentially to split 7.2 fronthaul in O-RAN
- The discriminant gain metric (importance-weighted feature elements) conceptually relates to CE-skip's idea that some subcarriers/UEs need CE more than others
- But no direct architecture, timing, or scheduling relevance

---

## Summary Table

| # | Paper | CE-skip Relevance | Key Contribution to CE-skip |
|---|-------|-------------------|---------------------------|
| 1 | dApps (Lacava et al.) | **HIGH** | dApp architecture, E3 interface, 450us control loop, CE as use case |
| 2 | SIMD vRAN (Chae et al.) | MEDIUM | PHY timing data, CPU vs GPU comparison, LMMSE benchmark |
| 3 | Model Versioning (Bensalem) | MEDIUM | ML lifecycle across O-RAN loops, dApp stability preference |
| 4 | Event-Triggered Edge AI (Zhou) | **HIGH** | Dual-threshold event-triggered inference, channel-adaptive offloading |
| 5 | Energy-Efficient ISCC (Yao) | LOW | Accuracy-vs-compute tradeoff formulation only |
| 6 | XAI-on-RAN (Basaran) | **HIGH** | A100 GPU utilization (63%), NVIDIA Aerial testbed, latency data |
| 7 | Distributed AI Platform (MS) | MEDIUM | Far-edge runtime design, eBPF probes, compute headroom (<50% util) |
| 8 | REAL RL xApps (Barker) | LOW-MEDIUM | xApp too slow for PHY, validates dApp need |
| 9 | Cloud-RAN Edge AI (Zhang) | LOW | No direct relevance |

---

## Cross-Paper Synthesis for CE-skip

### 1. Architecture Validation
The dApps paper (#1) provides the definitive architectural home for CE-skip. The monitor tier and skip scheduler are dApps co-located with the DU, accessing CE-related KPMs via E3 and issuing skip/compute control actions within 450 us. The XAI-on-RAN paper (#6) confirms this is feasible on NVIDIA Aerial with A100 GPUs, and the GPU utilization headroom (~37% idle at baseline) is sufficient for CE-skip's lightweight monitoring.

### 2. Timing Budget
Combining data from papers #1, #2, and #6:
- CE computation itself (LMMSE-like): 0.03-5 ms depending on MIMO config (paper #2)
- dApp control loop overhead: ~450 us (paper #1)
- GPU inference cycle: ~5 ms, GPU util ~63% (paper #6)
- **CE-skip's monitor must run in <<450 us to not bottleneck the control loop**
- Total budget per TTI: 1 ms (for 15 kHz SCS), CE-skip decision must fit within

### 3. Event-Triggered Design
Paper #4 provides the closest methodological framework:
- CE-skip = event-triggered CE inference, not periodic
- Dual-threshold: confident-stable -> skip CE, uncertain -> compute CE, middle -> interpolate
- Channel-adaptive: skip aggressiveness adapts to channel coherence time
- The non-convex optimization in paper #4 can be adapted for CE-skip's accuracy-compute tradeoff

### 4. Control Loop Hierarchy
Across papers #1, #3, #7, #8:
- Non-RT RIC (>1s): CE model selection/versioning (paper #3)
- Near-RT RIC (10ms-1s): too slow for per-slot CE skip (paper #8 confirms)
- dApp (<1ms): CE skip scheduling (paper #1)
- Far-edge runtime (<1ms): CE monitoring (paper #7)

### 5. Compute Headroom Evidence
- vRAN CPU utilization <50% (paper #7, Microsoft)
- GPU utilization ~63% baseline (paper #6, XAI-on-RAN)
- Both confirm CE-skip's core premise: modern software-defined BS has enough compute slack to run a monitor that decides whether CE is needed

### 6. Papers to Cite in CE-skip Manuscript

**Must cite (architecture + feasibility):**
- dApps (Lacava et al.) -- primary architectural reference
- XAI-on-RAN (Basaran, Dressler) -- GPU-native RAN testbed, Aerial SDK reference

**Should cite (methodology):**
- Event-Triggered Edge AI (Zhou et al.) -- event-triggered inference framework
- SIMD vRAN (Chae et al.) -- PHY processing timing reference

**Can cite (supporting):**
- Model Versioning (Bensalem et al.) -- ML lifecycle in O-RAN
- Distributed AI Platform (MS Research) -- platform architecture, compute utilization data
- REAL RL xApps (Barker et al.) -- xApp limitations, validates dApp need

**Skip:**
- Energy-Efficient ISCC (Yao et al.) -- too tangential
- Cloud-RAN Edge AI (Zhang et al.) -- no architectural relevance
