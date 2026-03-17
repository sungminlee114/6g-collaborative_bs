# Unit 2b: DL-based Channel Estimation -- PDF Paper Analysis

Analysis of 5 DL-CE papers for CE-skip paper context.
CE-skip motivation: DL-CE is the most expensive CE method (~2ms inference), making event-triggered skip scheduling most beneficial.

---

## 1. PACE-Net (Yang et al., Entropy 2025)

**Full title:** Channel Estimation for Massive MIMO Systems via Polarized Self-Attention-Aided Channel Estimation Neural Network

| Aspect | Detail |
|--------|--------|
| **Antennas** | 16 Tx, 16 Rx (uniform rectangular arrays) |
| **Frequency** | 15 GHz carrier |
| **Subcarriers** | Not OFDM-based; flat channel matrix H in C^{Nt x Nr} |
| **Channel model** | Kronecker model with Rayleigh fading (not 3GPP, not ray-tracing) |
| **CE architecture** | CNN (DnCNN-inspired) + Polarized Self-Attention (PSA). 4 layers: input (3x3 conv, 256 filters), hidden (1x3 + 3x1 conv), PSA attention, output (3x3 conv). Residual learning for noise extraction. |
| **Parameters** | Not explicitly stated; lightweight (D=4 layers, small kernels) |
| **Complexity** | O(D * K_l^2 * Nt * Nr) where D=4, sum(K_l^2)=24. For 16x16: DK^2=24 << NtNr=256, so cheaper than LS. Much cheaper than MMSE O(Nt^3 * Nr^3). |
| **Inference time** | Not reported |
| **Temporal** | Static snapshots only (no time-varying channel) |
| **NMSE** | ~-21 dB at SNR=20 dB (from Fig. 8); ~-3 dB at SNR=0 dB. Approaches MMSE, beats LS by ~4 dB at low SNR. |
| **Training** | 10K train, 3K val, 1K test. Keras, Adam, lr=1e-5, 200 epochs, batch=256, Titan 3080 |

**CE-skip relevance:** Lightweight CNN model. Low complexity relative to MMSE. If deployed, inference cost is moderate -- skip scheduling provides moderate benefit. The PSA attention adds minimal overhead but is a per-sample operation that could be skipped when channel is stable.

---

## 2. Channelformer (Luan & Thompson, arXiv 2302.04368, 2023)

**Full title:** Channelformer: Attention based Neural Solution for Wireless Channel Estimation and Effective Online Training

| Aspect | Detail |
|--------|--------|
| **Antennas** | SISO (single antenna Tx, single antenna Rx) |
| **Frequency** | 2.1 GHz (sub-6 GHz band), SCS = 15 kHz |
| **Bandwidth** | 6 RBs = 72 subcarriers, 14 OFDM symbols per slot |
| **Channel model** | 3GPP TS 36.101: EPA, EVA, ETU power delay profiles. Rayleigh fading with exact Doppler method. |
| **CE architecture** | Encoder-decoder: Encoder = Multi-head attention (Nheads=Npilot=2) + Pre-Network (2-layer CNN with GeLU). Decoder = Residual convolutional architecture (K=3 res blocks for offline, K=1 for online) + FC resize module. |
| **Parameters** | Offline: 117,659 (21,358 encoder + 96,301 decoder). Online: 32,069 (21,358 encoder + 10,711 decoder). Pruned online (70%): 9,620 params. |
| **Inference time** | **Online Channelformer: 20.5 ms**, Offline: 31.0 ms. For comparison: LS=0.6 ms, 1D FD-MMSE=1.2 ms, 2D FD-MMSE=3.96 ms, ReEsNet=8.95 ms, InterpolateNet=10.4 ms, HA02=23.4 ms, TR=11.6 ms. (Table III, MATLAB 2020a on RTX 2080 Super) |
| **Online training time** | 99.8 ms per iteration (full), 29.9 ms (70% pruned) |
| **Temporal** | Time-varying channels explicitly handled. Doppler shift 0-194 Hz (0-100 km/h). Online training adapts to dynamic channels. |
| **NMSE/MSE** | Offline Channelformer best across all SNR range on ETU. MSE ~1e-5 at 30 dB SNR. Online version competitive with 1D FD-MMSE. Denoising gain up to 35 dB. |
| **Training** | 125K samples (95% train, 5% val), SNR 5-25 dB, Doppler 0-97 Hz. Huber loss. |
| **Pruning** | 70% weight-level pruning retains near-identical performance with 9,620 params. |

**CE-skip relevance:** CRITICAL PAPER. Channelformer inference takes **20.5 ms** (online) or **31.0 ms** (offline) in MATLAB -- orders of magnitude slower than LS (0.6 ms). Even the pruned version is expensive. This is exactly the scenario where CE-skip scheduling is most beneficial: if the channel is slowly varying, skipping Channelformer inference and reusing the previous estimate saves ~20 ms per slot. The online training latency (99.8 ms) further motivates intelligent scheduling of when to run vs. skip CE.

---

## 3. Transfer vs. Meta Learning for CE (Ha et al., arXiv 2508.09751, 2025)

**Full title:** Online Data Generation for MIMO-OFDM Channel Denoising: Transfer Learning vs. Meta Learning

| Aspect | Detail |
|--------|--------|
| **Antennas** | Nt Tx, Nr Rx (general MIMO-OFDM). 5G NR with DM-RS config type 1, mapping type A. |
| **Frequency** | 5G NR standard (not explicitly specified; likely sub-6 GHz from DM-RS config) |
| **Subcarriers** | K consecutive subcarriers per slot |
| **Channel model** | 3GPP CDL-B (delay spread 300 ns) for optimization analysis. General 3GPP power delay profiles for training. |
| **CE architecture** | Generic DnNN (denoising neural network) -- framework-agnostic. Can be CNN, ResNet, or Transformer. Operates on sub-sampled CFR maps (Mt x Mf sub-CFRs). |
| **Parameters** | Not specified (framework paper, not architecture paper) |
| **Inference time** | Not reported directly |
| **Adaptation approaches** | (1) Transfer learning: pre-train offline on diverse channels, fine-tune online with generated data. (2) Meta learning (MAML): meta-train across channel tasks, adapt with few gradient updates. |
| **Online data generation** | Key contribution: data-aided CE using detected data symbols as virtual DM-RS. Optimal time-frequency window (P*, Q*) derived analytically. |
| **Temporal** | Explicitly time-varying: velocity 60-120 km/h, Doppler effects central to the work. Optimal window P* decreases with increasing velocity. |
| **MSE** | Data-aided sub-CFR estimates serve as practical labels for online training. Performance comparable to ideal scenario with true CFR labels. |

**CE-skip relevance:** Highly relevant to CE-skip. This paper addresses the domain mismatch problem -- when to re-train/adapt the CE model because the channel has changed. CE-skip scheduling is the complementary inference-side problem: when to re-run the CE model vs. reuse previous estimates. The optimal window size (P*, Q*) analysis shows that channel temporal correlation determines how aggressively one can reuse information -- directly analogous to CE-skip's delta threshold. Transfer learning adapts in ~few gradient steps while MAML in ~1 step, both much cheaper than full retraining.

---

## 4. Domain Adaptation for CE (Hoang et al., arXiv 2507.08974, 2025)

**Full title:** Domain Adaptation-Enabled Realistic Map-Based Channel Estimation for MIMO-OFDM

| Aspect | Detail |
|--------|--------|
| **Antennas** | Point-to-point: M_BS BS antennas (ULA), single UE antenna. In QSCM: BS16, in MBCM: varies. |
| **Frequency** | QSCM (source): 3.4 GHz (DeepMIMO Outdoor 1). MBCM (target): 5G NR, numerology mu=1, SCS=30 kHz. |
| **Subcarriers** | K=612 subcarriers, M=14 OFDM symbols per slot, 10 slots/subframe, 2 subframes/frame |
| **Channel model** | Source: QSCM (Quasi-Static Channel Model) from DeepMIMO. Target: MBCM (Map-Based Channel Model) from MATLAB RayTracing + OpenStreetMap (ETS neighborhood). CDL model for time-domain. |
| **CE architecture** | Two approaches: (1) CNN: 8 conv layers, kernels 9x9 then 5x5, filters [64,64,64,32,16,8,1]. BN + Tanh activation. (2) Pix2Pix GAN: UNet generator (7 encoder + 7 decoder layers), 5-layer discriminator. |
| **Parameters** | Not explicitly counted |
| **Inference time** | Not reported |
| **Domain adaptation** | Pre-train on QSCM (3077 samples), fine-tune on MBCM (300 samples for fine-tuning, 700 for validation). Freeze early layers, train last 3 layers (CNN) or last 2 encoder + generator layers (GAN). |
| **Temporal** | Quasi-static: f_D=0 within each slot (block fading). Channel varies slot-to-slot. |
| **NMSE** | Source domain: CNN/GAN improve significantly over LS-LI at low SNR (0-10 dB). At high SNR (>15 dB): LS-LI sufficient. Target domain: fine-tuned LS-LI-CNN and LS-LI-GAN outperform non-fine-tuned versions substantially. |
| **Key insight** | Wasserstein-1 distance between QSCM and MBCM = 0.4597 -- domains are significantly different. Transfer learning bridges this gap with only 300 samples. |

**CE-skip relevance:** Moderate. This paper focuses on sim-to-real transfer (QSCM to map-based), not on inference efficiency. However, it highlights that DL-CE models are domain-specific and need adaptation when the environment changes. For CE-skip, this means the skip threshold should be aware of domain shifts -- if the model is operating in a new domain without adaptation, skip scheduling should be more conservative (run CE more often). The ray-tracing based MBCM is relevant to our Sionna RT setup.

---

## 5. ReQuestNet (Pratik et al., Qualcomm, arXiv 2508.08790, 2025)

**Full title:** ReQuestNet: A Foundational Learning Model for Channel Estimation

| Aspect | Detail |
|--------|--------|
| **Antennas** | 2x2 MIMO (2 Rx x 2 Tx). Handles variable transmit layers (1 or 2). |
| **Frequency** | 5G NR, SCS = 15 or 30 kHz |
| **Subcarriers** | Variable: 4-272 RBs (48-3264 subcarriers), 14 OFDM symbols/slot |
| **Channel model** | 3GPP TDL (TDL-A/B/C/D/E) for training. CDL (CDL-A/B/C) for OOD generalization. Delay spread 30-300 ns. Doppler 0-450 Hz. |
| **CE architecture** | Two-stage: (1) **CoarseNet**: 2D U-Net CNN with gated convolutions. Per-PRG, per-Tx-Rx SISO processing. Input: 8 channels (received DMRS, transmitted DMRS, LS estimate -- real/imag, binary mask, noise std). (2) **RefinementNet**: T=4 recurrent refinement modules (RMs), each with: Likelihood Module (gradient of log-likelihood), Encoder (Fusion-CNN + Intra-PRG attention + Inter-PRG attention + Cross-MIMO attention + MLP), Decoder (U-Net 2D CNN). |
| **Parameters** | **~3.6M total**: CoarseNet ~1.2M, each RM ~0.6M (x4 RMs, non-shared weights = 2.4M) |
| **Inference time** | Not explicitly reported, but "complexity can be optimized further for on-device deployment" |
| **Temporal** | Handles time-varying channels implicitly through Doppler shift range (0-450 Hz). No explicit temporal tracking. |
| **NMSE** | Consistently outperforms genie-aided MMSE across all scenarios. Up to **10 dB gain** at high SNR. Generalizes to OOD CDL channels without retraining. Plots show -NMSE from ~12 dB (low SNR) to ~45 dB (high SNR). |
| **Training** | Online data generation (sampled from config space in Table I). Batch=32, Adam lr=4e-4, 40K steps, ReduceLROnPlateau. SNR perturbation + signal amplification augmentation. |
| **Key features** | Permutation equivariant (handles MIMO stream reordering). Handles variable RB count, PRG bundling (2 or 4), DMRS patterns, precoding types. Joint CE across PRGs (unlike MMSE which is limited to single PRG). |

**CE-skip relevance:** MOST RELEVANT. ReQuestNet is a **3.6M parameter** model with **4 recurrent refinement steps** -- each step involves attention mechanisms and U-Net processing. This is by far the most expensive CE architecture analyzed. The iterative refinement (T=4 steps) means inference cost scales linearly with T. For CE-skip:
1. The high computational cost makes skip scheduling extremely valuable -- avoiding even one unnecessary ReQuestNet inference saves significant compute.
2. The 4-step refinement suggests a natural "early exit" CE-skip variant: use fewer refinement steps when the channel is slowly varying.
3. The foundational model aspect (single model for all configs) means it's always deployed, making scheduling critical.
4. At 3.6M params, this would take significantly longer than the 20.5 ms measured for the 32K-param Channelformer.

---

## Comparative Summary Table

| Paper | Architecture | Params | Inference Time | Channel Model | Antennas | Best NMSE | Temporal |
|-------|-------------|--------|---------------|---------------|----------|-----------|---------|
| PACE-Net | CNN+PSA (4 layers) | ~few K (est.) | Not reported | Kronecker/Rayleigh | 16x16 | -21 dB @20dB SNR | Static |
| Channelformer | Transformer+ResCNN | 32K (online), 118K (offline) | **20.5 ms** (online), 31 ms (offline) | 3GPP EPA/EVA/ETU | SISO | MSE ~1e-5 @30dB | Time-varying, Doppler 0-194Hz |
| Transfer vs Meta | Generic DnNN | Framework-agnostic | Not reported | 3GPP CDL-B | MIMO | Comparable to ideal labels | Time-varying, 60-120 km/h |
| Domain Adapt | CNN (8-layer) / Pix2Pix GAN | ~100K+ (est.) | Not reported | DeepMIMO + MATLAB RT | MISO | Significant gain at low SNR | Quasi-static (block fading) |
| ReQuestNet | CoarseNet + 4x RefineNet | **3.6M** | Not reported (very high) | 3GPP TDL/CDL | 2x2 MIMO | -45 dB @40dB SNR (beats MMSE by 10dB) | Doppler 0-450Hz |

---

## Key Takeaways for CE-skip Paper

### 1. DL-CE computational cost is substantial and varies widely
- Channelformer: **20.5 ms** for a tiny SISO system with 32K params
- ReQuestNet: **3.6M params** with iterative refinement -- estimated >>20 ms
- Even the lightweight PACE-Net adds overhead on top of LS
- Compare: LS takes 0.6 ms, LMMSE takes 1.2-4.0 ms

### 2. Time-varying channels are the norm, not the exception
- All papers except PACE-Net address time-varying channels
- Doppler ranges: 0-450 Hz (ReQuestNet), 0-194 Hz (Channelformer), 60-120 km/h (Transfer vs Meta)
- Channel coherence time determines how often CE needs to run -- the core CE-skip question

### 3. Online adaptation adds further computational burden
- Channelformer online training: 99.8 ms per iteration
- Transfer learning fine-tuning: few gradient steps but still expensive
- MAML: 1 gradient step for adaptation
- CE-skip can reduce both inference and adaptation frequency

### 4. Complexity hierarchy motivates tiered scheduling
For CE-skip, the three CE methods form a natural cost hierarchy:
- **LS**: O(Nt^2 * Nr^2), ~0.6 ms -- always cheap to run
- **LMMSE**: O(Nt^3 * Nr^3), ~1.2-4.0 ms -- moderate cost
- **DL-CE**: O(params * input_size) or iterative, **20-100+ ms** -- skip scheduling most beneficial here

### 5. Pruning/compression can reduce but not eliminate the cost gap
- Channelformer 70% pruning: 9,620 params, but still much slower than LS
- The architectural overhead (attention, iterative refinement) is the bottleneck
- CE-skip is complementary to model compression -- both reduce total compute

### 6. Specific numbers for CE-skip paper Table 1 (DL-CE method complexity)
| Method | Params | Inference (ms) | Source |
|--------|--------|---------------|--------|
| LS | N/A | 0.6 | Channelformer Table III |
| 1D FD-MMSE | N/A | 1.2 | Channelformer Table III |
| 2D FD-MMSE | N/A | 4.0 | Channelformer Table III |
| InterpolateNet | 9,442 | 10.4 | Channelformer Table I/III |
| ReEsNet | 53,000 | 9.0 | Channelformer Table I/III |
| Online Channelformer | 32,069 | 20.5 | Channelformer Table I/III |
| Offline Channelformer | 117,659 | 31.0 | Channelformer Table I/III |
| HA02 | 105,607 | 23.4 | Channelformer Table I/III |
| ReQuestNet | 3,600,000 | >>30 (est.) | ReQuestNet paper |

The **33x cost ratio** (20.5 ms DL-CE vs. 0.6 ms LS) is the key motivation for CE-skip: even skipping 50% of DL-CE inferences would save ~10 ms per slot on average.
