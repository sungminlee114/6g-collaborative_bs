# Unit 6: Beam Management & Site-Specific Papers Analysis

**Purpose**: Analyze 12 beam management and site-specific papers for CE-skip paper relevance.
CE-skip's Exp 6 evaluates beamforming rate impact with stale CSI. Beam management overhead reduction is related -- both reduce PHY-layer processing frequency.

## Reference Project Config
| Parameter | Values |
|-----------|--------|
| Frequency | 3.5 / 15 / 28 GHz |
| Antennas  | 8x8 / 16x16 / 32x32 UPA |
| BS count  | 8 BS |
| Scene     | Munich UMi (Sionna RT) |
| UE range  | 10--150 m |
| Mobility  | 0 / 1 / 8.3 / 33 m/s |

---

## Per-Paper Analysis

### 1. A Survey of Beam Management for mmWave and THz Communications Towards 6G
**Authors**: Xue et al. (2023)
**CE-skip relevance**: **MEDIUM** -- Comprehensive survey of beam management overhead; directly relevant framing for why CE/beam scheduling matters.

**Dataset config**: Survey paper -- no single dataset. Reviews 150+ papers across mmWave (30--100 GHz) and THz (0.1--10 THz) bands. Covers 3GPP NR beam management procedures (P1--P3), IEEE 802.11ad/ay BFT.

**Beam/CE methods**: Exhaustive beam sweeping (3GPP NR SSB/CSI-RS), hierarchical codebooks, AI-empowered (DL, FL, TL), sensing-aided (ISAC), RIS-enhanced. Key overhead: periodic beam sweeping with RSRP measurement/reporting cycles.

**Site-specific aspects**: Discusses FL and TL for collaborative beam management across BSs. Notes site-specific propagation determines beam distributions. No per-site adaptation framework proposed.

**Temporal aspects**: Beam tracking for mobile UEs is a core topic. Discusses beam coherence time, temporal prediction, and beam failure recovery. TBP (temporal beam prediction) is reviewed.

**Key takeaway for CE-skip**: The survey frames beam management overhead as a critical bottleneck -- same motivation as CE-skip. Beam measurement periodicity directly parallels CE inference periodicity. The 3GPP P1/P2/P3 procedures define when beam measurements happen, analogous to when CE inference should run.

---

### 2. Site-Specific Beam Alignment in 6G via Deep Learning
**Authors**: Heng, Zhang, Alkhateeb, Andrews (2024)
**CE-skip relevance**: **HIGH** -- Core site-specific BA concept directly parallels CE-skip's per-site CE scheduling. Both argue that site-specific tuning (codebook vs. CE frequency) dramatically reduces overhead.

**Dataset config**:
| Parameter | Value |
|-----------|-------|
| Antennas | 64x1 ULA (MISO) |
| Frequency | 28 GHz |
| Bandwidth | 50 MHz |
| Codebook | 256 DFT beams |
| Scene | Boston downtown (ray-tracing, DeepMIMO) |
| UEs | 77,597 (52% LOS, 48% NLOS) |
| Tx Power | 40 dBm |

**Beam/CE methods**: End-to-end learned probing codebook + beam selection. Two approaches: Codebook-Based (CB) and Grid-Free (GF). With 8--16 probing beams, achieves near-optimal SNR (within 0.5--1 dB of genie), reducing measurement overhead by 16--32x.

**Site-specific aspects**: **Central thesis**: site-specific learning for BA. Probing codebook and beam selection are both adapted per-site. Emphasizes that "the price for this data acquisition and site-specific modeling is well-justified." Proposes digital twin pipeline for continuous model update.

**Temporal aspects**: Discusses beam tracking as future direction. Notes CB approaches are more robust for quick environmental changes. Self-training and auto-updating models discussed as R4 requirement.

**Key takeaway for CE-skip**: The SSBA concept is a direct analogue to CE-skip: both use site-specific learned models to reduce PHY-layer overhead. SSBA reduces beam measurement frequency; CE-skip reduces CE inference frequency. The digital-twin pipeline (Fig. 4) mirrors CE-skip's architecture where site-specific models are continuously updated.

---

### 3. DL-Based Beam Management for mmWave Vehicular Networks Exploring Temporal Correlation
**Authors**: Oliveira et al. (2025)
**CE-skip relevance**: **HIGH** -- Proposes prediction-aided measurement substitution (replacing beam measurements with predictions to reduce overhead by 66.7%), directly analogous to CE-skip's inference skipping.

**Dataset config**:
| Parameter | Value |
|-----------|-------|
| Scene | Marseille (M10%), Rosslyn (R10%, R50%) -- ray-tracing |
| NLOS ratio | 10--50% |
| BS height | 10--25 m |
| Modulation | OFDM |
| Tracking | Vehicular V2I, DFT codebook |
| Measurement interval | 80 ms (prediction), 240 ms (actual measurement) |

**Beam/CE methods**: RNN (LSTM)-based beam tracking. Classification (DeepBT-C) and regression (DeepBT-R). Autoregressive inference: replaces 2 out of 3 measurements with predictions, reducing overhead by ~66.7%. Sliding window of 4 time steps.

**Site-specific aspects**: Not explicitly site-specific. Models trained/tested per-dataset. Notes site-specific LIDAR models have generalization concerns.

**Temporal aspects**: **Core contribution**: temporal beam tracking with RSRP prediction. Introduces MAFD (Mean Absolute First Difference) metric for beam dynamics characterization. Studies LOS/NLOS transition impact on temporal stability.

**Key takeaway for CE-skip**: The measurement substitution strategy (Fig. 5) is the beam-domain equivalent of CE-skip. Every N-th time slot uses actual measurement; intermediate slots use predictions. The 66.7% overhead reduction with marginal accuracy loss validates the CE-skip philosophy. The MAFD metric could inspire a CE-skip equivalent for channel dynamics.

---

### 4. Meta-Learning Multi-armed Bandits for Beam Tracking in 5G and 6G Networks
**Authors**: Mattick et al. (2025)
**CE-skip relevance**: **MEDIUM** -- RL-based beam tracking with minimal measurements per time step. Generalizes across unseen trajectories/environments via meta-learning.

**Dataset config**:
| Parameter | Value |
|-----------|-------|
| Frequency | 28 GHz |
| Bandwidth | 100 MHz |
| Subcarriers | 64 active OFDM |
| Antennas | URA panels, half-wavelength spacing |
| Codebook | 60 beams (main experiments), up to 240 |
| BS count | 1--4 BS |
| Channel model | QuaDRiGa |
| Scene | Custom urban with blockers/reflectors |
| UE | Omni-directional, mobile |

**Beam/CE methods**: Restless MAB (RMAB) formulation. Only probes k=1 beam per time step. Decomposes into "goal prediction" (where optimal beam will be) and "search" (exploration-exploitation). GRU-based stochastic neural network. Meta-learned offline, deployed online with single forward pass.

**Site-specific aspects**: Tests cross-environment transfer (no-blocker -> blocker -> blocker+reflector). Shows 36--70% top-1 accuracy across unseen environments (vs. 2% random). Not explicitly per-site adapted.

**Temporal aspects**: Tracks moving UE through time. Models distribution shift from UE movement. Sequential POMDP formulation with belief state updates.

**Key takeaway for CE-skip**: The RMAB framework's principle of minimizing measurement budget (1 beam probe per timestep) directly parallels CE-skip's goal of minimizing CE inference calls. Meta-learning for environment transfer is relevant to CE-skip's site-specific model initialization.

---

### 5. Multi-modal Data Driven Virtual Base Station Construction for Massive MIMO Beam Alignment
**Authors**: Bian et al. (2026)
**CE-skip relevance**: **LOW** -- Geometry-based beam alignment using LiDAR/VBS construction. Not about temporal scheduling or measurement frequency reduction.

**Dataset config**:
| Parameter | Value |
|-----------|-------|
| Antennas | ULA (BS: N_BS, UE: N_UE) |
| Frequency | Not explicitly stated (mmWave/sub-THz implied) |
| Scene | OpenStreetMap + ray-tracing |
| Codebook | Near-field polar-domain |
| Modality | 3D LiDAR + BS location |

**Beam/CE methods**: Virtual BS (VBS) construction from LiDAR point clouds. Coarse channel reconstruction from VBS geometry -> partial beam training. Reduces search space from M_BS * M_UE to small candidate set S.

**Site-specific aspects**: Inherently site-specific -- VBS construction depends on physical building geometry. VBS locations are computed per-site from LiDAR data. More interpretable than DL approaches.

**Temporal aspects**: Static method -- no temporal beam tracking. VBS set is pre-computed for a given environment.

**Key takeaway for CE-skip**: Low direct relevance. The VBS concept of extracting environment-specific structure is philosophically related to site representation, but the method is geometry-based rather than learning-based, and does not address temporal scheduling.

---

### 6. 5G-Advanced AI/ML Beam Management: Performance Evaluation with Integrated ML Models
**Authors**: Jayaweera et al. (Nokia, 2024)
**CE-skip relevance**: **HIGH** -- 3GPP Rel-18 compliant SBP/TBP evaluation with system-level simulator. Directly shows overhead reduction with measurement overhead reduction (MOR) metric.

**Dataset config**:
| Parameter | Value |
|-----------|-------|
| Scenario | 3GPP Urban Macro (Rel-18) |
| Frequency | FR2 (mmWave) |
| Antennas | UPA (gNB + UE), lambda/2 spacing |
| Codebook | 64-beam CSI-RS (SBP), 32-beam CSI-RS (TBP), 8-beam SSB |
| UE speed | 3 km/h (SBP), 30 km/h (TBP), tested at 60/120 km/h |
| Channel | 3GPP model |
| UEs | 42,000 total across 200 drops |
| Observation/Prediction window | l_o=5, l_p=1 |

**Beam/CE methods**:
- SBP: DNN (SSB->CSI-RS) and CNN-DNN (subset CSI-RS->full CSI-RS). Top-1 accuracy 63--82%.
- TBP: LSTM-CNN with historical RSRP. Top-1 accuracy 56--78%.
- MOR: SBP up to 87.5%, TBP up to 80%.
- Model size: <1 MB, <2M FLOPS.

**Site-specific aspects**: Studies antenna configuration generalization. Shows significant degradation when NW antenna panel changes (cell-edge UEs penalized). Models need retraining for new configurations.

**Temporal aspects**: TBP predicts future beams from historical RSRP measurements. Sample-and-hold (SnH) baseline. TBP with Set B=16, Set A=32 achieves 11% Top-1 accuracy gain over SnH. Generalization across UE speeds: 3--7% degradation at 2--4x speed.

**Key takeaway for CE-skip**: The MOR metric and system-level throughput evaluation methodology are directly applicable to CE-skip. The TBP finding that models degrade gracefully with speed (6.7% accuracy loss at 4x speed) supports CE-skip's hypothesis that temporal prediction can replace frequent measurements. Performance monitoring and fallback to legacy procedures parallel CE-skip's event-triggered design.

---

### 7. AI/ML for Beam Management in 5G-Advanced: A Standardization Perspective
**Authors**: Xue et al. (2024)
**CE-skip relevance**: **MEDIUM** -- Industrial/standardization perspective on AI/ML BM. Discusses model generalization, life cycle management, and UE-gNB collaboration.

**Dataset config**: Uses Qualcomm and Huawei evaluation results from 3GPP contributions. SBP Top-1 accuracy: 63.5--75.4% (AI) vs. 10.7--29.4% (non-AI). TBP: 56.4--81.6% (AI) vs. 63.3--78.3% (non-AI).

**Beam/CE methods**: SBP (spatial) and TBP (temporal) as two core BM use cases. Emphasizes measurement overhead reduction as primary motivation. Discusses signaling for model transfer between gNB and UE.

**Site-specific aspects**: Highlights model generalization as key open issue. Proposes dataset mixing and online learning. Discusses specialized model catalogs per environment. Recommends meta/transfer learning for environment adaptation.

**Temporal aspects**: TBP predicts future beams from historical measurements. Discusses model life cycle management (LCM) including data collection, training, inference, and monitoring.

**Key takeaway for CE-skip**: The standardization framework for AI/ML BM (data collection, training, inference, monitoring, fallback) provides a template for CE-skip's deployment pipeline. The model LCM concept (Section B.4) directly maps to CE-skip's site-specific model management. The emphasis on lightweight models (<1 MB) is relevant to CE-skip's inference overhead considerations.

---

### 8. Compression of Site-Specific Deep Neural Networks for Massive MIMO Precoding
**Authors**: Kasalaee et al. (2025)
**CE-skip relevance**: **MEDIUM** -- Site-specific DNN compression for precoding. Demonstrates that site-specific conditions affect optimal model compression differently.

**Dataset config**:
| Parameter | Value |
|-----------|-------|
| Antennas | 8x8 UPA (64 elements), half-wavelength |
| Frequency | 2 GHz |
| BS height | 20 m |
| Tx power | 20 W |
| Users | 4 single-antenna |
| Distance | 50--350 m at 10-degree intervals |
| Scene | Montreal (OpenStreetMap + MATLAB ray-tracing) |
| Channel | LOS + NLOS (up to 10 reflections) |

**Beam/CE methods**: DNN-based digital precoder (CNN + 3 FCL). Mixed-precision quantization-aware training (QAT) with NAS. Achieves 35x higher energy efficiency than WMMSE at equal sum rate. Site-specific compression: different sites need different quantization configs.

**Site-specific aspects**: **Key finding**: optimal compression strategy varies across sites. "UdeM-LOS" vs. "UdeM-NLOS" vs. "Laval" sites show different Pareto fronts for energy-efficiency vs. sum-rate. Site-specific ray-tracing datasets used.

**Temporal aspects**: Static analysis -- no temporal modeling. Single-snapshot precoding.

**Key takeaway for CE-skip**: The site-specific compression finding validates CE-skip's hypothesis that optimal inference parameters (frequency, skip patterns) should vary per-site. Different sites having different Pareto-optimal configurations directly parallels CE-skip's per-site threshold tuning. The 8x8 UPA at 64 elements matches our project's antenna configuration.

---

### 9. Rethinking Beam Management: Generalization Limits Under Hardware Heterogeneity
**Authors**: Zeulin et al. (2026)
**CE-skip relevance**: **HIGH** -- Systematically demonstrates that ML-based beam predictors fail under heterogeneity (antenna, codebook, environment). Directly motivates per-site models.

**Dataset config**:
| Parameter | Value |
|-----------|-------|
| Frequency | 15 GHz |
| Subcarriers | 24 |
| SC spacing | 30 kHz |
| Antennas | 8x8 UPA (BS), 4x4 to 8x8 UPA (UE) |
| BS height | 15 m |
| Codebook | DFT, 16--64 beams |
| Tx power | 20 dBm |
| Mobility | SUMO simulator, avg 9.5 m/s, mixed cars/buses |
| Channel | Ray-tracing (urban vehicular) |

**Beam/CE methods**: ResNet-based DNN for beam direction prediction. Compared to hierarchical search (HS) and exhaustive search (ES). Under matched conditions, DNN matches or beats baselines. Under heterogeneity, DNN shows >50% SE drop at 90th percentile.

**Site-specific aspects**: **Central theme**: ML models trained on one configuration fail on another. Three failure modes:
1. Antenna mismatch: 4x4 -> 8x8 causes high variance in SE
2. Codebook mismatch: different random DFT codebooks cause high divergence
3. Environment mismatch: different spatial quadrants cause degradation

**Temporal aspects**: Mobile UEs with SUMO traffic simulation. Not focused on temporal prediction.

**Key takeaway for CE-skip**: This paper provides the strongest motivation for per-site CE-skip models. If beam predictors trained on one site fail on another, CE scheduling models will too. The taxonomy of heterogeneity dimensions (antenna, codebook, environment) applies directly to CE-skip. The recommendation for "site-specific model catalogs" and performance monitoring with fallback aligns perfectly with CE-skip's event-triggered design. The 15 GHz frequency and 8x8 UPA match our project config.

---

### 10. Environment Sensing-aided Beam Prediction with Transfer Learning for Smart Factory
**Authors**: Feng et al. (2024)
**CE-skip relevance**: **MEDIUM** -- Transfer learning for beam prediction across environments. Shows 70% data and 75% time reduction with fine-tuning.

**Dataset config**:
| Parameter | Value |
|-----------|-------|
| Frequency | 28 GHz |
| Modulation | OFDM |
| Antennas | UPA (Nt = Nt_x * Nt_y at BS, Nr at UE) |
| BS count | 2 |
| Scene | Smart factory (Blender 3D + Wireless InSite ray-tracing) |
| Factory size | 60m x 40m x 20m |
| UE speed | 0.8--1.2 m/s (engineering vehicles) |
| Channel model | X3D ray model |

**Beam/CE methods**: Environment sensing framework: RGB cameras (semantic segmentation via DeepLabv3+) + LiDAR (3D point cloud) + user location. LSTM network for temporal beam prediction. Pre-training + transfer learning: fine-tune with 30% labeled data from new environment for 94% Top-10 accuracy.

**Site-specific aspects**: Explicitly handles environment transfer. Static features (3D point cloud of factory layout) are separated from dynamic features (moving objects). Transfer learning freezes early CNN layers and LSTM, fine-tunes later layers.

**Temporal aspects**: Temporal prediction using LSTM with continuous frame sequences (l frames input). Dynamic scatterer detection from video frames.

**Key takeaway for CE-skip**: The separation of static (site) and dynamic features is philosophically aligned with CE-skip's site representation concept. Transfer learning with 30% data is relevant to CE-skip's few-shot site adaptation. The indoor factory scenario differs from our outdoor urban setting but the methodology applies.

---

### 11. Resource-Efficient Beam Prediction with Multimodal Realistic Simulation Framework (CRKD)
**Authors**: Park et al. (2025)
**CE-skip relevance**: **LOW** -- Knowledge distillation from multimodal teacher to radar-only student. Focuses on sensor modality reduction, not temporal scheduling.

**Dataset config**:
| Parameter | Value |
|-----------|-------|
| System | mmWave beamformer with URA |
| Modality | LiDAR, radar, GPS, RGB (teacher); radar-only (student) |
| Scene | CARLA + MATLAB (V2I urban) |
| Codebook | B beams with weight vectors |

**Beam/CE methods**: Cross-modal Relational Knowledge Distillation (CRKD). Teacher: transformer-based multimodal fusion. Student: radar-only with 10% of teacher parameters, achieving 94.62% of teacher performance.

**Site-specific aspects**: Multi-lane scenarios studied but not explicitly per-site adapted. Dataset analysis shows skewed beam distributions across scenarios.

**Temporal aspects**: Observation window of P sampling intervals for sequential prediction. Not focused on temporal scheduling optimization.

**Key takeaway for CE-skip**: The model compression angle (10% parameters, 94% performance) is relevant to CE-skip's inference cost considerations. Demonstrates that resource-efficient models can maintain high performance, supporting CE-skip's lightweight inference argument.

---

### 12. ProtoBeam: Generalizing Deep Beam Prediction to Unseen Antennas using Prototypical Networks
**Authors**: Mashaal et al. (2025)
**CE-skip relevance**: **MEDIUM** -- Domain adaptation for beam classification across antenna hardware. Relevant to CE-skip's cross-site generalization.

**Dataset config**:
| Parameter | Value |
|-----------|-------|
| Frequency | 60 GHz |
| Antennas | 24-beam codebook, SiBeam frontends (TX0--TX3) |
| Data | I/Q samples (2048 per block) |
| SNR range | -15 to 20 dB |
| Dataset | DeepBeam real-world measurement |

**Beam/CE methods**: Prototypical Networks (PN) for cross-antenna beam classification. DenseNet encoder (3 dense blocks, output dim 128). Achieves 74.11% accuracy in Training-On-One-Testing-On-Another (TOTA) vs. 16.97% baseline (398% improvement). No model retraining needed for new antennas.

**Site-specific aspects**: Addresses antenna hardware heterogeneity, not environment specificity. PN creates class prototypes that generalize across antenna configurations without retraining.

**Temporal aspects**: No temporal modeling. Single-snapshot beam classification.

**Key takeaway for CE-skip**: The Prototypical Network approach for zero-shot domain adaptation could be applied to CE-skip for cross-site generalization without fine-tuning. The 398% improvement in cross-domain accuracy (from 17% to 74%) demonstrates PN's potential for site-agnostic initialization of CE-skip models.

---

## Dataset Comparison Table

| # | Paper | Freq | Ant (BS) | BW | SC | Channel | Scene | Mobility | Dist | #BS | SNR |
|---|-------|------|----------|-----|-----|---------|-------|----------|------|-----|-----|
| 1 | Survey | 30-300 GHz | various | -- | -- | various | -- | -- | -- | -- | -- |
| 2 | SSBA | 28 GHz | 64x1 ULA | 50 MHz | -- | RT | Boston | static | -- | 1 | -/var |
| 3 | DeepBT | mmWave | -- | -- | -- | RT | Marseille/Rosslyn | vehicular | -- | 1 | var |
| 4 | Meta-MAB | 28 GHz | URA | 100 MHz | 64 | QuaDRiGa | custom urban | mobile | -- | 1-4 | -- |
| 5 | VBS | mmWave | ULA | -- | -- | RT | OSM | static | -- | 1 | -- |
| 6 | 5G-Adv | FR2 | UPA | -- | -- | 3GPP | UMa | 3-120 km/h | -- | multi | -- |
| 7 | Std. Persp. | mmWave | UPA | -- | -- | 3GPP | -- | -- | -- | -- | -- |
| 8 | Compression | 2 GHz | 8x8 UPA | -- | -- | RT | Montreal | static | 50-350m | 1 | 15 dB |
| 9 | Rethinking | **15 GHz** | **8x8 UPA** | -- | 24/30kHz | RT | urban vehicular | **9.5 m/s** | -- | 1 | -- |
| 10 | Sensing-TL | 28 GHz | UPA | -- | OFDM | RT | smart factory | 0.8-1.2 m/s | indoor | 2 | -- |
| 11 | CRKD | mmWave | URA | -- | -- | MATLAB | CARLA V2I | vehicular | -- | 1 | -- |
| 12 | ProtoBeam | 60 GHz | 24-beam | -- | -- | real meas. | indoor | static | -- | 1 | -15-20 |
| **Ours** | -- | **3.5/15/28** | **8-32x8-32** | -- | **52-1024** | **Sionna RT** | **Munich UMi** | **0-33 m/s** | **10-150m** | **8** | -- |

**Bold** = close match to our project config. Paper 9 (Rethinking) is the closest match with 15 GHz, 8x8 UPA, and urban vehicular scenario.

---

## Summary and CE-skip Integration

### High Relevance Papers (cite in CE-skip paper)

1. **SSBA (Paper 2)**: Site-specific beam alignment is the beam-domain analogue of CE-skip. Both argue that per-site optimization of PHY-layer procedures dramatically reduces overhead. Cite for motivation: "just as site-specific beam alignment reduces beam measurement overhead by 16-32x, event-triggered CE scheduling can reduce CE inference overhead."

2. **DeepBT (Paper 3)**: Prediction-aided measurement substitution (2/3 predictions, 1/3 actual) is the exact beam-domain parallel of CE-skip. The 66.7% overhead reduction with marginal accuracy loss validates our approach. Cite for methodology comparison.

3. **5G-Adv (Paper 6)**: 3GPP-compliant evaluation methodology with MOR metric, system-level throughput, and model generalization assessment. Cite for evaluation framework and TBP results showing graceful degradation with speed.

4. **Rethinking (Paper 9)**: Strongest motivation for per-site models. Shows ML beam predictors fail under antenna/codebook/environment heterogeneity (>50% SE drop). Cite for: "per-site CE scheduling thresholds are necessary because generalization across sites fails."

### Medium Relevance (related work section)

5. **Survey (Paper 1)**: Background on beam management overhead problem.
6. **Meta-MAB (Paper 4)**: RL-based minimal-measurement beam tracking with meta-learning transfer.
7. **Std. Perspective (Paper 7)**: Model LCM framework applicable to CE-skip deployment.
8. **Compression (Paper 8)**: Site-specific model compression -> site-specific CE-skip parameters.
9. **Sensing-TL (Paper 10)**: Transfer learning across environments with static/dynamic separation.
10. **ProtoBeam (Paper 12)**: Zero-shot domain adaptation via prototypical networks.

### Low Relevance (skip or brief mention)

11. **VBS (Paper 5)**: Geometry-based, no temporal scheduling.
12. **CRKD (Paper 11)**: Knowledge distillation for sensor reduction, not scheduling.

### Key Insights for CE-skip Paper

1. **Beam-CE parallel**: Beam management overhead reduction (via ML prediction) and CE inference scheduling (via skip prediction) share the same fundamental trade-off: accuracy vs. overhead. Multiple beam papers show 50-80% overhead reduction is achievable with ML-aided prediction.

2. **Per-site necessity**: Papers 2, 8, 9 collectively demonstrate that site-specific models are necessary -- generic models fail across configurations. This validates CE-skip's per-site threshold design.

3. **Measurement substitution**: Paper 3's strategy of replacing N-1 out of N measurements with predictions while keeping every Nth actual measurement is exactly CE-skip's approach. Their 66.7% reduction is achievable in CE as well.

4. **Evaluation methodology**: Paper 6's MOR metric, system-level throughput evaluation, and speed generalization testing provide a template for CE-skip's Exp 6 evaluation.

5. **Model LCM**: Papers 6, 7 describe data collection, training, inference, monitoring, and fallback procedures that CE-skip should adopt for practical deployment.
