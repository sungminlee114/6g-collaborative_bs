# Unit 7d: CSI Feedback / Semantic Comm / Autoencoder Papers

> CE-skip paper context: "Event-Triggered CE Inference Scheduling for Software-Defined Base Stations"
> Reference config: 3.5/15/28 GHz, 8x8 to 32x32 antennas, 256-4096 subcarriers, Munich UMi (Sionna RT), 8 BS, UE 10-150m

---

## Summary Verdict

**Overall CE-skip relevance of this batch: LOW.** These papers focus on CSI compression/feedback (UE-to-BS uplink), semantic communication for images/text, and general AE survey material. None directly address temporal CSI correlation, adaptive CE inference scheduling, or channel prediction -- the core concerns of CE-skip. However, a few offer tangential conceptual parallels (adaptive rate control, channel-adaptive inference).

---

## 1. Universal AE MIMO CSI Feedback [2403.00299]

**CE-skip Relevance: NONE** -- Addresses CSI feedback compression at UE, not BS-side CE inference scheduling.

### What It Does
- Proposes a universal AE encoder that supports **variable input sizes** (different antenna/subcarrier configs) and **multiple compression ratios** via a single encoder with a masking layer.
- Key trick: partition CSI tensor per antenna element, apply same encoder to each part (reduces params from 24M to 37K).
- Two-step training: universal training then sequential fine-tuning per compression ratio.

### Dataset Config
| Parameter | Value |
|-----------|-------|
| Antennas (BS) | up to N_BS = 32 |
| Antennas (UE) | up to N_UE = 4 |
| Subcarriers (K) | 68, 72, ..., 128 (16 settings) |
| Channel model | 3GPP TDL (EPA, EVA, TDL) |
| SNR | 11-40 dB |
| Frequency | Not specified (3GPP model-based) |
| Samples | 5,898,240 total (12 per setting) |

### Why Not Relevant
- Pure FDD feedback problem: UE compresses DL CSI and sends to BS.
- No temporal dimension, no CE at BS, no inference scheduling.
- Static snapshot CSI only.

---

## 2. VQ-VAE CSI Feedback Massive MIMO [2403.07355]

**CE-skip Relevance: NONE** -- Quantization of CSI feedback latent vectors; no temporal or scheduling component.

### What It Does
- Applies VQ-VAE with shape-gain vector quantization to CSI feedback.
- Separates latent vector into magnitude (scalar quantizer with mu-law) and direction (Grassmannian codebook).
- Multi-rate codebook design via nested codebooks for supporting multiple feedback rates.

### Dataset Config
| Parameter | Value |
|-----------|-------|
| Antennas (BS) | N_t = 32 (ULA) |
| Antennas (UE) | 1 |
| Subcarriers | N_c = 1024 (truncated to N_c_bar = 32 in angular-delay domain) |
| Channel model | COST2100 |
| Scenarios | Indoor picocell (5.3 GHz), Outdoor rural (300 MHz) |
| Frequency | 5.3 GHz / 300 MHz |
| Samples | 100K train, 30K val, 20K test |

### Why Not Relevant
- Focuses on bit-level quantization of feedback vectors -- orthogonal to CE-skip.
- No time series, no channel prediction, no adaptive scheduling.

---

## 3. Precoding-Oriented CSI VQ-VAE [2602.02508]

**CE-skip Relevance: NONE** -- Precoder design via VQ-VAE feedback with MI regularization.

### What It Does
- End-to-end precoding-oriented CSI feedback using VQ-VAE.
- Novel MI lower bound regularizer to prevent codebook collapse (uniform codeword usage).
- Directly optimizes sum achievable rate rather than CSI reconstruction NMSE.
- Shows learned codewords correlate with channel AoD structure.

### Dataset Config
| Parameter | Value |
|-----------|-------|
| Antennas (BS) | M = 64 (ULA) |
| UEs | K = 2 (single antenna each) |
| Paths | L_p = 2 per UE, pilots/UE L = 8 |
| Channel model | mmWave sparse scattering (parametric, not ray-tracing) |
| SNR | 10 dB |
| AoD | Uniform in [-pi/6, pi/6] |
| Frequency | Not specified (mmWave implied) |
| Samples | 10K per mini-batch, freshly generated |

### Why Not Relevant
- Precoder design problem, not CE inference scheduling.
- No temporal modeling, no BS-side computation concern.

---

## 4. SAFE: Semantic Adaptive Feature Extraction [2410.01597]

**CE-skip Relevance: LOW** -- Adaptive rate control concept has distant parallel to CE-skip's adaptive scheduling, but applied to image semantic communication, not channel estimation.

### What It Does
- Proposes adaptive bandwidth allocation for semantic image transmission.
- Decomposes image into sub-semantics with a U-Net architecture; users select different numbers of sub-semantics based on channel bandwidth.
- Three training strategies for multi-level semantic feature extraction.
- Bandwidth ratio per sub-semantic: k_i/n = 1/24 (per sub-semantic), total 1/12 with L=2.

### Dataset & Channel
- **Data**: ImageNet100 (100 categories, 1000 images each, 224x224)
- **Channel**: AWGN and Rayleigh fading (no MIMO channel simulation)
- **SNR range**: -5 to 15 dB

### Tangential Interest
- The idea of "transmit fewer sub-semantics when bandwidth is limited" is conceptually analogous to "skip CE when channel is stable" -- both are adaptive resource allocation.
- However, the domains are entirely different (image semantics vs. channel estimation at BS).

---

## 5. Robust JSCC Task-Oriented Semantic [2503.12907]

**CE-skip Relevance: LOW** -- KL-divergence regularization for channel robustness has a distant connection to CE-skip's concern about channel variation affecting inference quality.

### What It Does
- Proposes Fisher Information-based regularization for JSCC robustness against channel noise.
- KL divergence between noisy and noise-free posteriors approximated as (sigma^2 / 2) * Tr(I(z)).
- Smooths log-posterior curvature, making decoder robust to channel perturbations.
- Architecture-agnostic: applies to DeepJSCC, VL-VFE, DT-JSCC.

### Dataset & Channel
- **Data**: CIFAR-10 (10 classes), CIFAR-100 (100 classes) -- image classification
- **Channel**: AWGN, Rayleigh fading
- **PSNR range**: 5-25 dB (train), 5-25 dB (test, including mismatch)
- **No MIMO channel simulation**

### Tangential Interest
- The regularization adapts to noise variance (sigma^2) -- reminiscent of CE-skip's need to decide based on channel quality whether to run inference.
- But this is about making a fixed model robust, not about scheduling when to run inference.

---

## 6. DL Autoencoder Review NextGen Comm [2412.13843]

**CE-skip Relevance: NONE** -- Broad survey of AE in communications. No specific technique relevant to CE-skip.

### What It Covers
- Comprehensive survey: AE architectures (classical, CNN, RNN, VAE, sparse, adversarial), loss functions (MSE, CE, BCE), activation functions, physical constraints.
- Domain coverage: wireless, optical fiber, free-space optical, VLC, semantic comm, quantum comm.
- FLOP analysis for AE complexity.
- Tables comparing 20+ AE systems by model, dataset, performance, limitations.

### Why Not Relevant
- Survey paper with no specific temporal/adaptive CE scheduling content.
- Mentions channel variability as a challenge for AE-based systems but does not address inference scheduling.
- Could cite as general AE background but adds nothing specific to CE-skip argument.

---

## 7. Mobile Edge Generation [2409.05870]

**CE-skip Relevance: NONE** -- Distributed GenAI (text-to-image) at network edge; no CE or CSI component.

### What It Does
- Deploys Latent Diffusion Model split across edge server (inference/denoising) and UE (final generation).
- Transmits compressed latent "seeds" instead of full images.
- DRL-based dynamic power allocation across fading time slots for seed transmission quality.

### Why Not Relevant
- Application domain is image generation, not channel estimation.
- No MIMO, no CSI, no CE -- purely about efficient AI content delivery over wireless.

---

## 8. Semantic Edge Computing 6G [2411.18199]

**CE-skip Relevance: LOW** -- Survey unifying SEC and SemCom; mentions O-RAN and DNN partitioning for edge inference, but no CE-specific content.

### What It Covers
- Unifies Semantic Edge Computing (split DNN inference) and Semantic Communication (JSCC) under one framework.
- Covers DNN partitioning, latent feature compression at split points, collaborative inference.
- Mentions O-RAN RIC for integrating DL solutions.
- Discusses latency-accuracy tradeoffs in distributed inference.

### Tangential Interest
- The SEC concept of "selectively offloading computation while transmitting only semantic features" is broadly related to CE-skip's philosophy of avoiding unnecessary computation.
- Mentions that intermediate DNN representations can be 5x larger than input (ResNet early layers), motivating compression -- analogous to CE inference being computationally heavy.
- However, no specific CE, channel prediction, or inference scheduling content.

---

## Dataset Configs Summary (MIMO Channel Simulation Only)

Only papers 1-3 use actual MIMO channel simulation. Papers 4-8 use image datasets with simplified AWGN/Rayleigh channels.

| Paper | Antennas (BS) | Antennas (UE) | Freq | Subcarriers | Channel Model | Bandwidth |
|-------|--------------|---------------|------|-------------|---------------|-----------|
| #1 Universal AE | up to 32 | up to 4 | N/S (3GPP) | 68-128 | 3GPP TDL (EPA/EVA) | N/S |
| #2 VQ-VAE | 32 (ULA) | 1 | 5.3 GHz / 300 MHz | 1024 (trunc 32) | COST2100 | N/S |
| #3 Precoding VQ-VAE | 64 (ULA) | 1 per UE (K=2) | mmWave (implied) | N/A (parametric) | Sparse scattering | N/S |

None of these configs are close to our reference setup (Sionna RT, Munich UMi, 8x8 to 32x32 ELAA, 256-4096 SC, 28 GHz). All use statistical channel models or simple parametric models rather than ray-tracing.

---

## Conclusion

**No papers in this batch warrant citation in the CE-skip paper.** The CSI feedback papers (#1-3) solve UE-side compression (opposite direction from BS-side CE inference). The semantic comm papers (#4-5, #7-8) and the AE review (#6) operate in entirely different application domains (image/text transmission) with no channel estimation or inference scheduling content.

The only conceptual parallel worth noting (but not citing) is SAFE's (#4) adaptive sub-semantic selection based on channel conditions, which loosely mirrors CE-skip's adaptive inference decision. However, this analogy is too distant to be useful in the related work section.
