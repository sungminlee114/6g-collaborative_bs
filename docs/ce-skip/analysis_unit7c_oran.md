# Unit 7c: O-RAN Architecture, AI Lifecycle, and Agentic Platforms

Analysis of 7 papers for the CE-skip paper: "Event-Triggered CE Inference Scheduling for Software-Defined Base Stations"

Focus areas: AI lifecycle management, model versioning, dApp/xApp control loops, GPU resource sharing.

---

## Paper-by-Paper Analysis

---

### 1. Beyond Connectivity: An Open Architecture for AI-RAN Convergence in 6G
**Polese et al., arXiv 2507.06911 (Northeastern / zTouch Networks)**

**CE-skip Relevance: HIGH**

This paper provides the strongest architectural justification for CE-skip among the seven. It proposes a converged O-RAN + AI-RAN architecture with two key components directly relevant to CE-skip: AI-RAN Sites with GPU-accelerated infrastructure and dApp-level real-time control.

**Key Architectural Elements:**

- **AI-RAN Site (Fig. 3):** Edge cloud with GPU accelerators managed by AI-O-Cloud, running containerized workloads for both RAN DSP and AI inference. CE-skip's monitor dApp would deploy as an AI-for-RAN workload on this exact infrastructure.
- **dApps as AI-for-RAN:** Explicitly listed as "real-time programmable components that extend the O-RAN architecture to provide intelligence within the protocol stack" (Sec. IV-A). This is CE-skip's deployment model -- a dApp controlling CE kernel execution.
- **MIG partitioning:** Experiments partition an A100 GPU into 3 slices (40GB RAN, 20GB LLM, 10GB CNN). This demonstrates the GPU resource sharing model CE-skip assumes: CE kernels and the skip-decision logic coexist on the same GPU with resource isolation.
- **AI-and-RAN coexistence validation:** Fig. 5 shows RAN throughput and CRC error rate are consistent with/without AI workloads running, validating that AI inference (like CE-skip's monitor) does not degrade RAN DSP performance.
- **Batch vs. real-time deployment (Fig. 4):** CE-skip's monitor is a real-time AI-for-RAN workload. The paper's real-time deployment workflow (pre-authentication -> policy -> MEC interface -> DMS) maps to how an operator would deploy the CE-skip dApp.

**CE-skip citation use:**
- Primary architectural reference for "dApp as CE controller" in GPU-native RAN sites
- MIG experiment as evidence that skip-decision inference overhead is negligible alongside RAN DSP
- AI-O2 interface for centralized policy delivery (e.g., skip aggressiveness threshold) from AI-SMO to edge site

---

### 2. AI/ML Life Cycle Management for Interoperable AI Native RAN
**Huang, Wen & Li, arXiv 2507.18538 (NTU / NSYSU / Imperial College)**

**CE-skip Relevance: HIGH**

This is the definitive reference for 3GPP AI/ML lifecycle management (LCM), covering Rel-16 through Rel-20. It provides the standardized framework within which CE-skip's models would be managed, monitored, and updated.

**Key LCM Elements for CE-skip:**

- **Five-block LCM architecture (Fig. 2):** Data Collection -> Model Training/Adaptation -> Model Storage -> Inference -> Management. CE-skip has a DL-CE model that undergoes this exact lifecycle. The Management block's "selection/(de)activation/switching/fallback" operations map directly to CE-skip's skip/execute decision, which is fundamentally a model activation control.
- **Performance monitoring modes (Table II):** The three monitoring modes for CSI prediction (Type 1: UE-side SGCS with perf-bad flag, Type 2: ground-truth reporting, Type 3: quantized SGCS) provide a template for CE-skip's own monitoring. CE-skip's monitor checks "is the cached channel estimate still accurate?" -- this is functionally equivalent to the SGCS-threshold mechanism but applied at the gNB side.
- **KPI-driven triggers:** TS 28.567's DriftDetection policy triggers Retrain/Rollback when KPIs fall below thresholds. CE-skip's trigger is analogous: when delta-NMSE exceeds the threshold, the system transitions from "skip" to "execute" state.
- **Delta-model updates (Rel-19 LCM Profile):** "hot-swap triggers, KPI-driven rollback; RRC IE set (ml-ModelId, ml-ModelVersion, integrity hash)" -- this is how the DL-CE model in CE-skip would be updated when environment drift is detected.
- **Model-ID-based approach:** modelId + modelVersion tracking is exactly what CE-skip needs for managing multiple CE model versions across different BS sites.

**CE-skip citation use:**
- Frame CE-skip's monitor as an instance of 3GPP's LCM Management block
- The skip/execute decision is a model "(de)activation" control action within the standardized LCM framework
- Performance monitoring overhead trade-off (Sec. IV-A-i) directly parallels CE-skip's core insight: monitoring resources have diminishing returns, so skip when monitoring says "no change"

**Critical insight from this paper:** The paper notes "scheduling more monitoring resources enables better tracking but reduces the overhead savings achieved by AI/ML models" (Sec. IV-A-i). This is exactly CE-skip's trade-off -- the monitoring cost of the skip-decision must be less than the saved CE computation.

---

### 3. Towards AI-Native RAN: An Operator's Perspective of 6G Day 1 Standardization
**Li, Sun, Wang et al. (China Mobile Research Institute), arXiv 2507.08403**

**CE-skip Relevance: MEDIUM-HIGH**

This paper provides the operator's perspective on AI-Native RAN standardization, with three elements relevant to CE-skip: the AI Node architecture, performance monitoring metrics, and RS overhead reduction as a target use case.

**Key Elements:**

- **AI Node architecture (Fig. 9):** Proposes a centralized "AI Node" connected to distributed 6gNBs via a new RAN interface. The AI Node handles model training, storage, and L3 inference; 6gNBs handle L1/L2 inference locally. CE-skip's monitor runs at the 6gNB (L1 inference), while model retraining happens at the AI Node. This matches CE-skip's assumed deployment.
- **Use case table (Table I):** Explicitly lists "RS overhead reduction (incl. superimposed pilot)" as a 6G candidate use case for AI-for-air-interface, with BS-side inference using CNN/LSTM/Transformers (<10M params). CE-skip is directly related: rather than reducing RS symbols, it reduces CE inference frequency -- a complementary dimension of overhead reduction.
- **CSI prediction listed:** "outdated and limited accuracy" as the key challenge of non-AI solutions, with MLP-Mixer/2D-FCN/CNN (<10M params). CE-skip leverages CSI temporal correlation to decide when prediction/re-estimation is unnecessary.
- **Three-dimensional monitoring (Table II):** Model Performance (accuracy, inference latency) + Network Performance (BLER, throughput) + Resource Performance (CPU/GPU utilization, energy). CE-skip's skip decisions affect all three: inference accuracy (potentially degraded on skip slots), throughput (from stale CSI), and GPU utilization (reduced by skipping).
- **Collaborative AI computing (Sec. III-D, Fig. 7):** Collaborative training/inference/fine-tuning across nodes. CE-skip's DL-CE model could be collaboratively trained across multiple BSs, with skip thresholds adapted per site.
- **Dedicated AI radio bearer (Fig. 3):** New radio bearer for AI/ML data delivery. Could carry CE-skip's model updates or performance monitoring data between AI Node and 6gNBs.

**CE-skip citation use:**
- Operator validation that RS overhead reduction is a recognized 6G use case
- AI Node/6gNB split aligns with CE-skip's deployment model (lightweight monitor at BS, model management centralized)
- Multi-dimensional monitoring framework as evaluation methodology for CE-skip

---

### 4. Towards 6G Native-AI Edge Networks: A Semantic-Aware and Agentic Intelligence Paradigm
**Feng et al. (Exeter / Sony / Southeast Univ.), arXiv 2512.04405**

**CE-skip Relevance: LOW**

This paper focuses on semantic communication and agentic intelligence for 6G -- task-oriented meaning exchange and multi-agent reinforcement learning for RAN control. It operates at a different abstraction level than CE-skip.

The paper discusses O-RAN control placement (PHY/MAC, near-RT RIC, non-RT RIC) and xApp/rApp deployment, but only at a taxonomic level without concrete architectural details relevant to CE kernel scheduling. The semantic communication paradigm (transmitting compact task-relevant representations instead of raw bits) is philosophically aligned with CE-skip's "don't re-estimate when the old estimate suffices" but is technically unrelated.

**One-line dismissal:** Survey of semantic communication and agentic AI for 6G RAN; no concrete architectural mechanisms for PHY-layer inference scheduling.

---

### 5. Toward E2E Intelligence in 6G Networks: An AI Agent-Based RAN-CN Converged Intelligence Framework
**Han et al. (Kyung Hee Univ. / ETRI / Ruhr Univ.), arXiv 2602.23623**

**CE-skip Relevance: LOW**

This paper proposes an LLM+ReAct-based AI agent for unified RAN-CN decision-making. The agent queries a monitoring database to generate control policies for network slicing, handover, and capacity planning. It operates at the management plane (seconds-to-minutes timescale), far above CE-skip's sub-slot PHY-layer decision timescale.

The paper's critique of "task-specific ML models" and call for unified reasoning frameworks is tangentially relevant -- CE-skip's monitor is itself a task-specific model. However, the LLM-based approach targets network-level orchestration, not real-time PHY processing.

**One-line dismissal:** LLM-based RAN-CN unified reasoning agent operating at management-plane timescales; no relevance to sub-ms PHY-layer CE scheduling.

---

### 6. Large Generative AI Models meet Open Networks for 6G: Integration, Platform, and Monetization
**Li et al. (Toshiba Europe), arXiv 2410.18790**

**CE-skip Relevance: NONE**

This paper proposes an API-centric GAI marketplace for deploying and monetizing generative AI services (LLMs, image generators) within 6G networks. It is about AI-on-RAN (external AI workloads running on RAN infrastructure), not AI-for-RAN (AI enhancing RAN functions). The Open RAN testbed experiments measure LLM token generation latency, which has no connection to channel estimation.

**One-line dismissal:** GAI marketplace and monetization strategy for LLM services on 6G infrastructure; entirely AI-on-RAN, zero overlap with CE-skip.

---

### 7. MX-AI: Agentic Observability and Control Platform for Open and AI-RAN
**Chatzistefanidis et al. (EURECOM / BubbleRAN / Aalto / Khalifa Univ.), arXiv 2508.09197**

**CE-skip Relevance: MEDIUM**

MX-AI is a multi-agent LLM system deployed at the SMO R1 interface for intent-driven RAN observability and control on a live OAI+FlexRIC testbed. While it operates at the non-real-time SMO level (8.8s end-to-end latency), it validates the programmable control stack that CE-skip's ecosystem relies on.

**Relevant Elements:**

- **dApps reference (Table I):** Explicitly cites dApps [Lacava et al.] as prior art for "on-device micro-services able to execute control logic inside the RAN stack itself," distinguishing them from MX-AI's higher-level reasoning. This reinforces the layered control model: dApps for sub-ms CE-skip decisions, xApps/rApps for slower optimization, LLM agents for intent-level management.
- **FlexRIC E2 integration:** The paper demonstrates closed-loop control via E2 interface on a real OAI gNB. While CE-skip would use E3/dApp interface (faster), the E2 path could deliver policy updates (skip threshold parameters) from the near-RT RIC to the DU.
- **Slice reconfiguration use case:** MX-AI reconfigures PRB allocations per slice through LLM reasoning. CE-skip could be exposed as a "per-slice CE policy" knob -- e.g., URLLC slices never skip CE, eMBB slices skip aggressively.
- **Control hierarchy validation:** The paper's architecture (Fig. 1) cleanly shows RT (<1ms, dApps) -> near-RT (10ms-1s, xApps) -> non-RT (>1s, rApps/LLM agents), confirming the control timescale taxonomy CE-skip relies on.

**CE-skip citation use:**
- Validates the multi-timescale control hierarchy where CE-skip's dApp operates at the RT layer while receiving policy from upper layers
- The "per-slice policy" concept could extend CE-skip (different skip aggressiveness per slice)

---

## Summary Table

| Paper | Relevance | Key CE-skip Contribution |
|-------|-----------|-------------------------|
| Beyond Connectivity (Polese) | **HIGH** | AI-RAN Site with GPU+dApps, MIG partitioning, coexistence validation |
| AI/ML LCM (Huang, Wen, Li) | **HIGH** | 3GPP LCM framework for CE model management, monitoring trade-offs |
| AI-Native RAN Operator (Li, CMRI) | **MED-HIGH** | AI Node architecture, RS overhead reduction use case, monitoring metrics |
| 6G Native-AI Edge (Feng) | LOW | Semantic communication taxonomy, no PHY-layer inference scheduling |
| E2E Intelligence (Han) | LOW | LLM agent for management-plane RAN-CN orchestration |
| Large GenAI Open Networks (Li, Toshiba) | NONE | GAI marketplace, AI-on-RAN monetization |
| MX-AI Agentic Platform (Chatzistefanidis) | **MEDIUM** | Multi-timescale control hierarchy validation, per-slice policy concept |

## Recommended Citations

**Must-cite (paper body):**
1. **Polese et al. (Beyond Connectivity)** -- architectural evidence that dApps + GPU sites enable CE-skip
2. **Huang, Wen & Li (AI/ML LCM)** -- CE-skip's monitor as LCM Management block instance; monitoring overhead trade-off

**Should-cite (related work / discussion):**
3. **Li et al. (AI-Native RAN Operator)** -- RS overhead reduction as recognized 6G use case; AI Node/6gNB deployment model
4. **MX-AI** -- control hierarchy validation (dApp RT layer for CE-skip, upper layers for policy)

**Do not cite:**
5-7. The remaining three papers (semantic communication, LLM agent, GAI marketplace) are outside CE-skip's scope.
