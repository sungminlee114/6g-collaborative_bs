# Unit 5: FL for CE / Beamforming / ISAC -- Paper Analysis

CE-skip paper context: CE-skip is CE-algorithm-agnostic and works on top of any CE method.
FL-based CE is another CE method that CE-skip could wrap. Key concerns: beamforming with stale CSI from skipped slots.

---

## Per-Paper Analysis

### 1. Robust FL for Wireless CE (Fang et al., 2404.03088)

**CE-skip relevance: MEDIUM**
- FL-based CE with SBS-level training and MBS aggregation. Directly relevant as a CE method CE-skip could wrap.
- The "outdate mode" attack (providing random outdated CSI) is conceptually related to stale CSI from CE-skip, though framed as an adversarial attack rather than intentional scheduling.
- Shows outdated CSI has minimal impact due to strong temporal correlation -- supports CE-skip's premise that skipping CE is feasible.

**Dataset config:**
| Parameter | Value |
|-----------|-------|
| Antennas | Not specified (CNN input 612x14x1) |
| Frequency | Not specified (mmWave mentioned) |
| BW | Not specified |
| Subcarriers | 612 (from input dimension) |
| Channel model | MATLAB 5G Toolbox synthetic |
| Scene | Synthetic |
| Mobility | Not specified |
| Distance | Not specified |
| Num BS | 10 SBS + 1 MBS |
| SNR | Not specified |

**CE/BF methods:** CNN-based CE (3-layer CNN), FL with FedAvg/FedMedian/StoMedian aggregation, LLPF pre-filtering.

**Multi-BS setup:** 10 SBS + 1 MBS. SBSs train local CE models on cached UE data; MBS aggregates. Hierarchical FL.

**Stale CSI handling:** Yes -- "outdate mode" attack uses random outdated CSI. Finding: outdated CSI has minimal impact on model accuracy due to strong temporal correlation. This is favorable evidence for CE-skip.

---

### 2. Coalition FL for CE in RIS-assisted Cell-free MIMO (Qi et al., 2502.05538)

**CE-skip relevance: MEDIUM**
- FL-based CE for RIS-assisted cell-free MIMO with cascaded channels. Relevant as another FL-CE method CE-skip could wrap.
- Coalition formation optimizes FL user grouping based on channel correlation -- similar to our multi-BS site-adaptation concept.
- HFL (heterogeneous FL) reduces local model size via knowledge distillation, relevant for deployment on resource-constrained BS.

**Dataset config:**
| Parameter | Value |
|-----------|-------|
| Antennas | 4x4=16 per BS |
| Frequency | Not explicitly stated (DeepMIMO, mmWave implied) |
| BW | 100 MHz |
| Subcarriers | Not specified |
| Channel model | DeepMIMO ray-tracing dataset |
| Scene | Street canyon (600m x 440m, buildings) |
| Mobility | Static (fixed UE positions) |
| Distance | ~600m street length |
| Num BS | 4 BS + 1 RIS |
| SNR | 0-15 dB |

**CE/BF methods:** DNN-based CE (3 conv layers + FC), FL with transfer learning (distance + RSRP similarity), HFL with knowledge distillation, coalition formation via DQN/Qmix.

**Multi-BS setup:** 4 BS in cell-free MIMO with 1 RIS. Coalition formation groups UEs with similar channel characteristics for FL.

**Stale CSI handling:** No explicit treatment. Static scenario assumed.

---

### 3. FL Coordinated Beamforming for Multicell ISAC (Jiang et al., 2501.16951)

**CE-skip relevance: HIGH**
- Multi-cell coordinated beamforming with ICI management. Directly relevant to CE-skip because beamforming with imperfect/stale CSI is a central concern.
- VFL framework: BSs collaboratively design BF matrices under central server guidance, implicitly managing ICI without global CSI exchange. If CSI is stale, ICI management degrades.
- HFL framework: fully decentralized with novel loss function controlling interference leakage power -- more robust to CSI staleness since each BS uses only local CSI.
- Multi-BS cooperation model closely matches our 8-BS Munich setup.

**Dataset config:**
| Parameter | Value |
|-----------|-------|
| Antennas | NT=6 TX + NR=6 RX per BS (also tested NT=NR=16) |
| Frequency | Not specified |
| BW | Not specified |
| Subcarriers | N/A (beamforming, not OFDM CE) |
| Channel model | Rician fading (factor=3) |
| Scene | Multi-cell with 500m cell radius |
| Mobility | Static (random user/target placement) |
| Distance | Within 500m cell radius |
| Num BS | M=3 |
| SNR | 0-25 dB |

**CE/BF methods:** DNN-based beamforming (MLP with 4 hidden layers, 512 neurons), VFL-based coordinated BF, HFL-based coordinated BF, benchmarks: WMMSE, MRT, IMT, CBF.

**Multi-BS setup:** 3 BSs with central server. VFL: server computes global loss from local features. HFL: fully decentralized with FedAvg aggregation. Each BS serves K=2 UEs + senses 1 target.

**Stale CSI handling:** Not explicitly addressed. Assumes perfect local CSI available at each BS. However, the HFL interference leakage control approach could be more robust to CSI staleness than VFL since it does not require global CSI.

---

### 4. FL with Integrated Sensing, Communication, and Computation (Liang et al., 2409.11240)

**CE-skip relevance: LOW**
- Theoretical FL framework analysis (FedAVG-ISCC vs FedSGD-ISCC). Not about CE or beamforming design.
- Focuses on convergence analysis of FL when sensing data collection is integrated with communication and computation.
- Uses MNIST/Fashion-MNIST image classification, not wireless channel tasks.

**Dataset config:**
| Parameter | Value |
|-----------|-------|
| Antennas | Not specified |
| Frequency | Not specified |
| BW | Not specified |
| Subcarriers | N/A |
| Channel model | IID Rayleigh fading (for FL communication) |
| Scene | N/A |
| Mobility | N/A |
| Distance | N/A |
| Num BS | 1 server + 10 devices |
| SNR | Pmax=10W |

**CE/BF methods:** None (image classification task).

**Multi-BS setup:** Single server + 10 edge devices. Standard FL topology.

**Stale CSI handling:** Addresses communication errors (quantization noise in FL model exchange) but not CSI staleness for wireless system design.

---

### 5. Personalized FL Beamforming for ISAC (Ni et al., 2510.06709)

**CE-skip relevance: MEDIUM-HIGH**
- PFL for multi-cell ISAC beamforming optimization. EM-based adaptive aggregation weights per BS.
- Multi-BS heterogeneity (different sensing/comm trade-offs) directly relates to our site-specific adaptation.
- DNN maps channel samples to beamforming matrices -- if input CSI is stale, BF performance degrades.
- Uses MATLAB ray-tracing for realistic channel generation -- similar methodology to our Sionna RT approach.

**Dataset config:**
| Parameter | Value |
|-----------|-------|
| Antennas | NT=8 TX + NR=8 RX per BS |
| Frequency | Not specified |
| BW | Not specified |
| Subcarriers | N/A (beamforming optimization) |
| Channel model | Ray-tracing (MATLAB) + Rician fading (factor=3) for ICI |
| Scene | 3km x 3km semi-urban (OpenStreetMap + CellMapper) |
| Mobility | Not specified |
| Distance | Within cell coverage |
| Num BS | M=3 |
| SNR | Not specified |

**CE/BF methods:** ISAC-DNN (two parallel FC branches for comm/sensing channels, 256 neurons each), PFL with EM-based aggregation, baselines: FedAvg, FedPer, pFedMe, fixed-weight PFL.

**Multi-BS setup:** 3 BSs with central server. Each BS serves different number of UEs (2/3/4) and senses 1 UAV target. EM-based PFL allows each BS to adaptively weight global vs local model.

**Stale CSI handling:** No explicit treatment. Assumes perfect CSI available for DNN input. However, the PFL framework's ability to adapt to local conditions suggests potential robustness to varying CSI quality across BSs.

---

### 6. FedAttn: Federated Attention for Collaborative LLM Inference (Deng et al., 2511.02647)

**CE-skip relevance: LOW**
- Distributed LLM inference framework using federated self-attention. Not about wireless CE or beamforming.
- Interesting theoretical framework: duality between FL (model training) and FedAttn (collaborative inference). Could inspire federated inference scheduling concepts.
- Sparse attention and adaptive KV aggregation concepts loosely parallel CE-skip's selective computation idea.

**Dataset config:** N/A (LLM inference with Qwen2.5 on GSM8K benchmark)

**CE/BF methods:** None. Self-attention mechanism in Transformers.

**Multi-BS setup:** N/A. Multiple edge devices collaborating on LLM inference.

**Stale CSI handling:** N/A.

---

### 7. Elastic FL over O-RAN Architecture (Abdisarabshali et al., 2305.02109)

**CE-skip relevance: MEDIUM**
- O-RAN architecture for multi-service FL execution. Highly relevant to CE-skip's deployment context (software-defined BS in O-RAN).
- Introduces elastic resource provisioning for FL services -- CE-skip's scheduling could be implemented as an O-RAN xApp.
- Multi-time-scale control: non-RT system descriptor, near-RT FL controller, FL MAC scheduler -- aligns with CE-skip's event-triggered scheduling across different time scales.
- Addresses dynamic wireless channels and client mobility during FL execution.

**Dataset config:**
| Parameter | Value |
|-----------|-------|
| Antennas | Not specified |
| Frequency | Not specified |
| BW | 5 MHz |
| Subcarriers | N/A |
| Channel model | Not specified (real Porto city taxi dataset for client mobility) |
| Scene | Porto city (real-world taxi trajectories, 150 clients) |
| Mobility | Real taxi trajectories |
| Distance | City-scale |
| Num BS | Multiple O-RAN DUs |
| SNR | N/A |

**CE/BF methods:** None. Uses CIFAR-10, Fashion-MNIST, MNIST classification tasks to demonstrate FL orchestration.

**Multi-BS setup:** O-RAN with O-CU, O-DU, O-RU hierarchy. Multiple FL services competing for resources.

**Stale CSI handling:** Addresses dynamic channel conditions affecting FL communication overhead, but not CSI staleness for wireless system design.

---

### 8. Dynamic D2D FL over O-RAN (Abdisarabshali et al., 2404.06324)

**CE-skip relevance: MEDIUM**
- Extension of paper #7 with D2D-assisted hierarchical FL and dynamic resource allocation.
- Introduces "D-Events" for discrete-time channel changes -- similar to CE-skip's event-triggered paradigm.
- Dynamic model drift metric captures impact of time-varying datasets on FL accuracy -- analogous to CSI aging in CE-skip.
- MAC scheduler design for FL-specific resource allocation in O-RAN -- CE-skip scheduler could follow similar architecture.

**Dataset config:**
| Parameter | Value |
|-----------|-------|
| Antennas | Not specified |
| Frequency | Not specified |
| BW | Not specified |
| Subcarriers | N/A |
| Channel model | Time-varying channels modeled as D-Events |
| Scene | Not specified |
| Mobility | Dynamic (time-varying datasets) |
| Distance | Not specified |
| Num BS | Multiple O-RAN nodes |
| SNR | N/A |

**CE/BF methods:** None. MNIST/CIFAR-10 classification tasks.

**Multi-BS setup:** Hierarchical D2D clusters within O-RAN. Users form D2D groups for local model aggregation before global aggregation.

**Stale CSI handling:** Addresses dynamic wireless channels via D-Events and fine-granular time instants (FGTIs). Models temporal variation of datasets via ODE. Introduces dynamic model drift to measure impact on FL accuracy. Conceptually parallel to CSI aging.

---

### 9. Decentralized FL GNN O-RAN CE (2404.03088)

**DUPLICATE of Paper #1** (Robust FL for Wireless CE, Fang et al., 2404.03088). Same arXiv ID, same authors, same content. The repository has two copies (v1 and v2 of the same paper).

---

### 10. Data and Model-Driven DL Beamforming (Liang, Zheng, Li, Wong, Chae, 2406.03098)

**CE-skip relevance: HIGH**
- Robust beamforming under channel estimation errors using GNN. Directly addresses imperfect CSI impact on BF.
- Proposes modified optimal BF structure with trainable interference features from CE errors -- models the exact problem CE-skip faces when using stale CSI for beamforming.
- Data augmentation based quantile estimation (DAQE) samples channel errors during training to achieve robust BF -- CE-skip could use similar approach to train BF robust to CSI aging.
- Outage probability constraint (5%) ensures QoS even with imperfect CSI -- directly applicable to CE-skip's QoS guarantees.

**Dataset config:**
| Parameter | Value |
|-----------|-------|
| Antennas | N=4 TX (MU-MISO) |
| Frequency | Not specified |
| BW | 10 MHz |
| Subcarriers | N/A (narrowband MISO) |
| Channel model | Rayleigh fading (CN(0,1)) |
| Scene | Single cell |
| Mobility | Static (channel error model) |
| Distance | Not specified |
| Num BS | 1 |
| SNR | 5-30 dBm transmit power |

**CE/BF methods:** GNN-based robust BF (bipartite GNN with S-Net + Power-Net), modified optimal BF structure with interference features, DAQE for rate quantile estimation, bisection for power minimization. Benchmarks: BTI optimization (CVX), RZF-DNN.

**Multi-BS setup:** Single BS, K=4 users. No multi-BS cooperation.

**Stale CSI handling:** YES -- core contribution. Models CSI uncertainty as Gaussian error (variance 0.075). Optimizes beamforming to maximize minimum rate quantile under outage probability constraint. Achieves 14% higher robust rate than traditional optimization methods. The interference feature s_k explicitly captures channel estimation error impact.

---

## Dataset Comparison Table

| Paper | Ant | Freq | BW | SC | Channel | Scene | Mobility | BS | SNR range |
|-------|-----|------|----|----|---------|-------|----------|-----|-----------|
| #1 Robust FL CE | ? | mmWave | ? | 612 | MATLAB 5G | Synth | ? | 10+1 | ? |
| #2 Coalition FL RIS | 4x4=16 | mmWave* | 100MHz | ? | DeepMIMO RT | Street | Static | 4+1RIS | 0-15dB |
| #3 FL ISAC BF | 6+6 | ? | ? | N/A | Rician(3) | 500m cell | Static | 3 | 0-25dB |
| #4 FL ISCC | ? | ? | ? | N/A | Rayleigh | N/A | N/A | 1+10dev | ? |
| #5 PFL ISAC BF | 8+8 | ? | ? | N/A | RT+Rician | 3km semi-urban | ? | 3 | ? |
| #6 FedAttn | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| #7 Elastic FL ORAN | ? | ? | 5MHz | N/A | ? | Porto city | Taxi mobility | Multi-DU | N/A |
| #8 D2D FL ORAN | ? | ? | ? | N/A | Time-varying | ? | Dynamic | Multi | N/A |
| #9 | DUPLICATE of #1 | | | | | | | | |
| #10 Robust DL BF | 4 TX | ? | 10MHz | N/A | Rayleigh | Single cell | Static | 1 | 5-30dBm |
| **Ours** | **8x8/16x16/32x32** | **3.5/15/28 GHz** | **N/A** | **64-1024** | **Sionna RT** | **Munich UMi** | **0-33 m/s** | **8** | **varies** |

*mmWave implied from DeepMIMO reference

---

## Summary and Key Findings

### Relevance Ranking for CE-skip

| Relevance | Papers | Reason |
|-----------|--------|--------|
| **HIGH** | #3 (FL ISAC BF), #10 (Robust DL BF) | Direct BF optimization under imperfect/stale CSI; multi-BS coordination |
| **MEDIUM-HIGH** | #5 (PFL ISAC BF) | Multi-BS PFL for BF with ray-tracing channels; site-specific adaptation |
| **MEDIUM** | #1/#9 (Robust FL CE), #2 (Coalition FL CE), #7 (Elastic FL ORAN), #8 (D2D FL ORAN) | FL-CE methods CE-skip could wrap; O-RAN deployment context |
| **LOW** | #4 (FL ISCC), #6 (FedAttn) | Theoretical FL analysis or LLM inference; no CE/BF relevance |

### Key Insights for CE-skip Paper

1. **Stale CSI evidence from Paper #1:** The "outdate mode" attack result showing minimal degradation from outdated CSI due to temporal correlation directly supports CE-skip's premise. Can cite as empirical evidence that CE skipping is viable.

2. **Robust BF under CSI errors from Paper #10:** The modified optimal BF structure with trainable interference features (s_k) from channel estimation errors provides a principled framework for beamforming with stale CSI. The DAQE approach could be adapted for CE-skip: train BF networks with augmented CSI aging samples to achieve robust performance under skip scheduling.

3. **Multi-BS coordination from Papers #3 and #5:** The VFL/HFL coordinated BF frameworks show how multiple BSs can cooperate for BF design. CE-skip's scheduling decision at one BS affects ICI management at neighboring BSs -- the HFL interference leakage control approach is more robust since each BS only needs local CSI.

4. **O-RAN deployment from Papers #7 and #8:** The elastic FL and D2D FL frameworks provide concrete O-RAN architectures (xApps, MAC schedulers, RAN slicing) for deploying intelligent scheduling. CE-skip's event-triggered scheduler maps naturally to an O-RAN xApp in the near-RT RIC.

5. **Event-triggered parallel from Paper #8:** The "D-Events" concept for discrete-time channel changes and dynamic model drift metric are conceptually parallel to CE-skip's event triggers. CE-skip could adapt this formalism.

6. **FL as a CE method CE-skip wraps:** Papers #1, #2 show FL-based CE as a viable CE method. CE-skip is agnostic to the underlying CE method, so FL-CE is simply another option in the CE method portfolio (alongside LS, LMMSE, DL-CE). The key question for CE-skip is not how CE is done, but when to re-run it.

### Gap Analysis vs. Our Project Config

- **Antenna scale gap:** Most papers use small arrays (4-16 antennas). Our ELAA setup (up to 32x32=1024) is significantly larger. Only Paper #5 uses 8x8 arrays.
- **Frequency gap:** Most papers do not specify frequency or use generic models. Our multi-frequency setup (3.5/15/28 GHz) is more comprehensive.
- **Multi-BS gap:** Papers use 1-4 BSs. Our 8-BS Munich deployment is denser.
- **Mobility gap:** Most papers assume static scenarios. Our mobility range (0-33 m/s) is more realistic.
- **Channel model gap:** Most use simplified Rayleigh/Rician or DeepMIMO. Our Sionna RT with real Munich geometry is more realistic.
- **None of these papers address CE scheduling (when to run CE).** They all assume CE is performed every slot. This confirms CE-skip's novelty.
