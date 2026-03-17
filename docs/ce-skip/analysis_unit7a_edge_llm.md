# Unit 7a: Edge AI / LLM / Split Inference

Analysis of 11 papers for the CE-skip paper: "Event-Triggered CE Inference Scheduling for Software-Defined Base Stations"

**Verdict: 0 HIGH relevance, 2 LOW relevance, 9 NOT relevant.**

This cluster covers LLM deployment on edge devices (split inference, pruning, quantization, federated fine-tuning). The core concern is how to run billion-parameter language models under resource constraints -- a fundamentally different problem from CE-skip's sub-ms PHY-layer inference scheduling. However, two papers touch on channel prediction/estimation tangentially and provide minor contextual value.

---

## Paper-by-Paper Analysis

---

### 1. Edge Large AI Models: Revolutionizing 6G Networks
**Wang, Shi, Zhou, Zhu, Letaief; arXiv 2505.00321 (May 2025)**

**CE-skip Relevance: LOW**

Survey/vision paper on deploying large AI models at the 6G edge. Covers collaborative fine-tuning, microservice-assisted inference, and air-interface applications. Section V-A proposes federated LAM for channel prediction using autoregressive encoders with LoRA fine-tuning. Mentions that "channel estimation overhead can be huge because of the short channel coherence time" -- acknowledging the problem CE-skip addresses but not proposing any scheduling or skip mechanism.

**Minor citable point:** The paper notes temporal/frequency correlation in successive channel states as a resource for prediction, which aligns with CE-skip's temporal persistence assumption. However, the paper's focus is on training large foundation models for prediction, not on when to invoke inference.

**Use in CE-skip paper:** At most a single citation in the introduction to note that large AI models are being proposed for PHY tasks, increasing per-inference computational cost and thus strengthening the motivation for selective inference scheduling.

---

### 2. Efficient Large AI Inference in Wireless Edge Networks
**Lyu, Xiao, Xu, Skoglund, Di Renzo; arXiv 2505.09214 (May 2025)**

**NOT relevant.** Pruning-aware split inference of generic LAIMs between edge device and server. Optimizes pruning ratio, transmit power, and computation frequency. No PHY-layer channel estimation content.

---

### 3. Generative AI on the Edge: Architecture and Performance Evaluation
**Nezami, Hafeez, Djemame, Zaidi; arXiv 2411.17712 (Nov 2024)**

**NOT relevant.** Benchmarks LLM inference (Yi, Phi, Llama3) on Raspberry Pi 5 cluster with K3s for O-RAN edge. Measures token throughput (5-12 tok/s) and CPU/RAM usage. No channel estimation or PHY-layer content; focuses on conversational AI deployment feasibility.

---

### 4. How Small Can 6G Reason? Scaling Tiny Language Models for AI-Native Networks
**Ferrag, Lakas, Debbah; arXiv 2603.02156 (Mar 2026)**

**NOT relevant.** Evaluates 135M-7B parameter language models on 6G-Bench (30 network management decision tasks). Finds 1.5-3B models optimal for edge deployment. Mentions ELAA and near-field in the 6G technology overview (line 69) but only as general context; no channel estimation or temporal scheduling content.

---

### 5. LLM-Empowered IoT for 6G Networks
**Chen, Wu, Li, Ji; arXiv 2503.13819 (Jun 2025)**

**NOT relevant.** Proposes LLM-empowered IoT architecture with memory-efficient split federated learning for LoRA fine-tuning on heterogeneous IoT devices. Pure LLM deployment paper; no PHY-layer or channel estimation content.

---

### 6. LLM-Enabled Multi-Task Physical Layer Network
**Zheng, Dai; arXiv 2412.20772 (Mar 2025)**

**CE-skip Relevance: LOW**

Most relevant paper in this cluster. Proposes a single LLM (LLAMA2-based) fine-tuned with LoRA to perform channel prediction, multi-user precoding, and signal detection simultaneously. The channel prediction component predicts T2=4 future slots from T1=16 historical slots using CSI attention modules with temporal patching.

**Channel prediction details:**
- System: 32 BS antennas, 4 users, 32 subcarriers, 180 kHz spacing
- Velocity: 10-100 km/h uniform, SNR: 5-20 dB
- Metric: NMSE, compared against Transformer, RNN, LSTM, GRU, LLM4CP baselines
- Key insight: "channel coherence time is shorter than the channel estimation period" in high-mobility -- this is exactly the channel aging problem CE-skip monitors

**Connection to CE-skip:** The paper demonstrates that LLM-scale models are being proposed for PHY tasks including channel prediction. This increases per-inference FLOP cost dramatically (LLAMA2-7B backbone), making selective scheduling even more important. If such models replace lightweight CE networks, the computational savings from CE-skip become proportionally larger.

**Use in CE-skip paper:** Cite as motivation -- as PHY-layer AI models grow toward LLM scale, the cost of running every CE inference becomes prohibitive, strengthening the case for event-triggered scheduling. One sentence in introduction or related work at most.

---

### 7. Empowering Edge Intelligence: Survey on On-Device AI Models
**Wang et al.; ACM Computing Surveys, arXiv 2503.06027 (Mar 2025)**

**NOT relevant.** Comprehensive survey on on-device AI covering model compression, hardware acceleration, and edge deployment across IoT/mobile/autonomous vehicles. General-purpose survey with no PHY-layer or channel estimation focus.

---

### 8. Cognitive Edge Computing: Optimizing Large Models and AI Agents
**Wang, Li, Jia; arXiv 2501.03265 (Nov 2025)**

**NOT relevant.** Survey on deploying reasoning-capable LLMs and AI agents at edge. Covers quantization, sparsity, distillation, elastic offloading, federated personalization. No wireless PHY-layer content.

---

### 9. Pushing LLMs to the 6G Edge: Vision, Challenges, and Opportunities
**Lin, Qu, Chen, Chen, Chen, Huang; arXiv 2309.16739 (Jun 2025)**

**NOT relevant.** Vision paper on deploying LLMs at 6G MEC. Discusses split learning/inference, parameter-efficient fine-tuning, small-large LM cooperation. Focuses on LLM service delivery, not PHY-layer signal processing.

---

### 10. TinyLLM: Training and Deploying Language Models at Edge Computers
**Kandala, Medaranga, Varshney; NUS (Dec 2024)**

**NOT relevant.** Framework for training 30-120M parameter language models for sensor data analysis on edge devices. Demonstrates small models can outperform larger ones for specific tasks. No wireless communication or channel estimation content.

---

### 11. Adaptive Layer Splitting for Wireless LLM Inference in Edge Computing
**Chen, Li, Yu, Zhao, Zhang; Zhejiang University**

**NOT relevant.** Uses model-based RL to determine optimal LLM split point between UE and edge node under varying network conditions. Optimizes inference latency/performance tradeoff. Generic LLM inference optimization; no channel estimation or PHY-layer content.

---

## Summary Table

| # | Paper | Relevance | Reason |
|---|-------|-----------|--------|
| 1 | Edge Large AI Models 6G | LOW | Mentions CE overhead in channel prediction context; citable for motivation |
| 2 | Efficient Large AI Inference | NONE | Generic LAIM split inference optimization |
| 3 | Generative AI on Edge | NONE | LLM benchmarking on Raspberry Pi |
| 4 | How Small Can 6G Reason | NONE | LM scaling for network management tasks |
| 5 | LLM-Empowered IoT | NONE | Split federated learning for IoT LLMs |
| 6 | LLM MultiTask PHY Layer | LOW | LLM for channel prediction; motivates CE-skip for expensive models |
| 7 | On-Device AI Survey | NONE | General on-device AI survey |
| 8 | Optimizing Edge AI Survey | NONE | Cognitive edge computing survey |
| 9 | Pushing LLMs to 6G Edge | NONE | LLM edge deployment vision |
| 10 | TinyLLM Edge | NONE | Small LM training framework for sensors |
| 11 | Adaptive Layer Splitting | NONE | RL-based LLM split point optimization |

## Key Takeaway for CE-skip Paper

This entire cluster is **not directly relevant** to CE-skip. The papers address a different problem (deploying large language models on resource-constrained edge devices) rather than optimizing when to run PHY-layer inference.

The only usable connection is **motivational**: Papers 1 and 6 show that the trend toward using LLM-scale models for channel prediction/estimation will dramatically increase per-inference cost. This strengthens CE-skip's value proposition -- if CE costs 10x more FLOPS due to LLM-based estimators, skipping unnecessary invocations saves 10x more compute. This deserves at most 1-2 sentences in the introduction, not a dedicated related work subsection.

**Recommended citation count from this cluster: 0-1 papers (Paper 6 if any).**
