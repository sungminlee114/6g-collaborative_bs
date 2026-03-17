# Unit 7b: Split/Collaborative Inference

Analysis of 7 papers for the CE-skip paper: "Event-Triggered CE Inference Scheduling for Software-Defined Base Stations"

---

## Paper-by-Paper Analysis

---

### 1. Communication Efficient Cooperative Edge AI via Event-Triggered Computation Offloading
**You Zhou, Changsheng You, Kaibin Huang; arXiv 2501.02001 (Jan 2025)**

**CE-skip Relevance: HIGH**

This is the most relevant paper in this batch. It proposes an event-triggered cooperative inference framework with a dual-threshold, multi-exit architecture that selectively skips computation for "routine" events while offloading complex rare events to an edge server. The core idea -- using confidence thresholds to decide whether to compute further or exit early -- is structurally analogous to CE-skip's decision of whether to run full CE inference or reuse a cached estimate.

**Architecture Details:**
- Two-stage dynamic co-inference: lightweight CNN on device for binary tail/head detection, deep CNN (ResNet50) on edge server for multi-class classification
- Dual-threshold early-exiting: each CNN block has an intermediate classifier; if confidence C_n(m) < beta_l -> head event (exit early, no offload); if C_n(m) > beta_u -> tail event (offload to server); if beta_l <= C_n(m) <= beta_u -> pass to next block
- Channel-adaptive threshold optimization: thresholds are dynamically adjusted based on SNR to balance classification accuracy vs. communication/energy cost
- Feasibility condition links SNR to whether offloading is energetically viable (Lemma 1)

**Key Mechanism -- Missing-Target-Offloading Tradeoff:**
- P_off = (1 - P_miss) * P_tail + P_false * P_head
- Widening the uncertainty region (lowering beta_l, raising beta_u) forces more blocks to be traversed locally -- better accuracy but more computation
- Narrowing it causes more early exits -- less computation but more misses
- This tradeoff is directly analogous to CE-skip's skip-vs-recompute tradeoff: skipping more saves GPU cycles but risks stale estimates

**Optimization:**
- Non-convex problem transformed to strongly convex via proximal-point penalty method
- Lipschitz-continuous gradients proven for objective and constraints
- Lookup table of optimal (beta_l*, beta_u*) precomputed for different SNR values -- real-time decisions are table lookups, not online optimization
- Convergence rate improves with better channel conditions (higher SNR -> faster convergence)

**Experimental Setup:**
- Models: ShuffleNetV2 and MobileNetV2 (device), ResNet50 (server)
- Dataset: 25,000 retinal images (medical), binary + multi-class classification
- Imbalance ratios: 4:1 and 9:1 (head:tail)
- Communication: 30 dBm transmit power, 30 MHz bandwidth, varying SNR via fading coefficient
- Results: dual-threshold outperforms single-threshold and terminal detection across all offloading constraints; performance gain increases with SNR

**CE-skip Parallels and Citations:**
1. **Event-triggered paradigm**: Both CE-skip and this paper avoid always-on computation. CE-skip skips CE inference when channel is stable; this paper skips offloading when events are routine. Cite as prior art for event-triggered inference in 6G edge AI.
2. **Dual-threshold mechanism**: CE-skip's PSA monitor uses a persistence metric with a threshold to trigger recomputation. This paper's dual-threshold on confidence scores is a richer variant. CE-skip could cite this as motivation for exploring multi-threshold skip policies.
3. **Channel-adaptive decisions**: Both systems adapt to channel conditions -- CE-skip monitors temporal channel persistence, this paper adapts offloading thresholds to SNR. The lookup-table approach for precomputed optimal thresholds is directly applicable to CE-skip's scheduler design.
4. **Energy-accuracy tradeoff formalization**: The paper's formulation of accuracy vs. offloading probability under energy constraints (P1) mirrors CE-skip's formulation of NMSE vs. skip ratio under latency constraints.

**Key quotes:**
- "By selectively focusing on critical, high-impact events, computational workload and network congestion can be significantly reduced"
- "This work is the first to explore event-triggered cooperative inference systems with a novel dual-threshold based architecture design"

**Dataset config:** N/A (medical imaging, not wireless channels)

---

### 2. Dynamic Encoding and Decoding of Information for Split Learning in Mobile-Edge Computing
**Alhussein, Wei, Akhavain; arXiv 2309.02787 (Sep 2023)**

**CE-skip Relevance: LOW-MEDIUM**

The paper proposes an adaptive split encoder-decoder that adjusts the complexity-relevance tradeoff of transmitted latent representations based on network conditions, using Information Bottleneck (IB) theory. Applied to mmWave throughput prediction on the Lumos5G dataset.

**Key Ideas:**
- Encoder on UE, decoder on edge server; encoder has multiple modes (shallow z vs. deeper z') that trade off compression vs. informativeness
- Cascaded training: train optimal encoder-decoder first, then add bottleneck layer for more compressed mode
- Orchestrator monitors network conditions and instructs encoder which mode to use
- IB theory applied to sequential (temporal) models -- finds compression occurs across temporal hidden states, not just across training epochs
- Use case: mmWave throughput prediction in dual-connectivity (macro BS + mmWave micro BS)

**CE-skip Connection:**
- The idea of dynamically switching between "full computation" and "compressed/cheap computation" based on conditions maps loosely onto CE-skip's skip/compute decision
- The temporal compression finding (compression across hidden temporal states) is tangentially related to CE-skip's premise that channel estimates have temporal persistence
- However, this paper is about training-time split learning, not inference-time scheduling; the adaptive switching is about representation fidelity, not whether to run inference at all

**What CE-skip can cite:** Tangential reference for the general principle that temporal redundancy in sequential data enables adaptive computation reduction. Not a primary citation.

**Dataset config:** Lumos5G dataset (throughput prediction, not channel estimation)

---

### 3. Split Learning in 6G Edge Networks
**Zheng Lin, Guanqiao Qu, Xianhao Chen, Kaibin Huang; arXiv 2306.12194 (Jan 2024)**

**CE-skip Relevance: IRRELEVANT** -- Survey of split learning for model training (not inference). Covers FL vs. SL, resource management, model placement. No connection to adaptive inference scheduling or channel estimation.

---

### 4. Splitwise: Collaborative Edge-Cloud Inference for LLMs via Lyapunov-Assisted DRL
**Younesi et al.; UCC 2025 (arXiv 2512.23310, Dec 2025)**

**CE-skip Relevance: IRRELEVANT** -- Dynamic partitioning of LLM transformer layers across edge and cloud using DRL. Optimizes latency/energy/accuracy for NLP workloads (GPT-2, LLaMA). No connection to channel estimation, temporal persistence, or inference skipping. The Lyapunov queue stability framework is interesting but targets LLM serving, not PHY-layer processing.

---

### 5. SLIDE: Simultaneous Model Downloading and Inference at the Wireless Network Edge
**Qu, Li, Chen, Chen, Zhou; arXiv 2512.20946 (Jan 2026)**

**CE-skip Relevance: IRRELEVANT** -- Overlaps model downloading with layer-by-layer inference to reduce end-to-end latency. Focuses on model provisioning and bandwidth allocation for on-device AI model delivery. No relevance to CE scheduling, temporal channel dynamics, or adaptive inference.

---

### 6. CNN Collaborative Inference Mechanism for Heterogeneous Edge Devices
**Wang et al.; Sensors 2024, 24, 4176**

**CE-skip Relevance: IRRELEVANT** -- Pipeline-parallel CNN inference across heterogeneous IoT edge devices (Raspberry Pi, Jetson). Pre-partitioning based on critical operator layers with micro-shifting and dual compression to balance pipeline stages. Pure systems optimization for multi-device CNN execution; no connection to wireless channel estimation or adaptive/event-triggered computation.

---

### 7. MAE: Collaborative Inference Acceleration with Efficient DNN Partitioning
**Fang et al.; Computer Networks 278 (2026) 112073**

**CE-skip Relevance: IRRELEVANT** -- Mixture of Adaptive Experts (MAE) framework for edge-device collaborative DNN inference. Reconfigures convolutional channels as sparse "experts" to reduce intermediate feature transmission. Focuses on partition-point selection and resource allocation for multi-DNN workloads. No connection to channel estimation, temporal persistence, or inference scheduling.

---

## Summary Table

| # | Paper | Relevance | Key Takeaway for CE-skip |
|---|-------|-----------|--------------------------|
| 1 | Communication Efficient Cooperative Edge AI (Zhou+, 2501.02001) | **HIGH** | Event-triggered dual-threshold co-inference; closest structural analog to CE-skip's skip/compute decision. Channel-adaptive threshold optimization via lookup tables. |
| 2 | Dynamic Encoding/Decoding Split Learning (Alhussein+, 2309.02787) | LOW-MEDIUM | Adaptive computation modes based on network conditions; temporal compression in IB theory. Tangential. |
| 3 | Split Learning in 6G Edge Networks (Lin+, 2306.12194) | IRRELEVANT | Split learning survey (training, not inference) |
| 4 | Splitwise (Younesi+, 2512.23310) | IRRELEVANT | LLM edge-cloud partitioning |
| 5 | SLIDE (Qu+, 2512.20946) | IRRELEVANT | Model downloading overlapped with inference |
| 6 | CNN Collaborative Inference (Wang+, Sensors 2024) | IRRELEVANT | Pipeline-parallel CNN on heterogeneous IoT |
| 7 | MAE (Fang+, CompNet 2026) | IRRELEVANT | MoE-based DNN partitioning for edge |

## Key Insight for CE-skip

The event-triggered cooperative inference paper (Zhou et al., 2501.02001) is the only paper in this batch with strong structural relevance. Its dual-threshold early-exit mechanism provides a well-formalized precedent for CE-skip's adaptive skip decision:

- **Shared principle**: Not all inputs require full computation. Routine/stable inputs can be handled cheaply (early exit / skip), while complex/changing inputs get full processing (deep inference / fresh CE).
- **Shared mechanism**: Threshold-based decision on a confidence/persistence metric, adapted to channel conditions.
- **Difference**: Zhou et al. operate at the application layer (image classification with offloading), while CE-skip operates at the PHY layer (channel estimation with GPU scheduling). CE-skip's "event" is temporal channel change, not a data sample.

The remaining 6 papers address split/collaborative inference for general ML workloads (LLMs, CNNs, IoT) with no connection to channel estimation or temporal scheduling. They belong to a different branch of the edge AI literature.
