# Unit 2: DL-based Channel Estimation & Temporal Prediction

10 papers analyzed for CE-skip relevance (Event-Triggered CE Inference Scheduling).

**Reference project config:** 3.5/15/28 GHz, 8x8 to 32x32 antennas, 256-4096 SC, Munich UMi Sionna RT.

---

### PACE-Net: DL Channel Estimation for Massive MIMO (2024)

- **CE-skip relevance**: MEDIUM -- PSA (Polarized Self-Attention) module shows how DL-CE complexity scales; O(DK^2 NtNr) per forward pass provides concrete baseline for skip scheduling cost-benefit analysis
- **Dataset config**: 16x16 Tx/Rx antennas, 15 GHz, Rayleigh/Kronecker channel model, 1000m distance, T=16 time-domain orthogonal pilot sequences (NOT frequency-domain subcarriers — PACENet is narrowband flat fading, no OFDM), SNR range [-5, 25] dB
- **CE methods**: LS (O(Nt^2 Nr^2)), MMSE (O(Nt^3 Nr^3)), PACE-Net (O(DK^2 NtNr) where D=4 attention heads, DK^2=24). PACE-Net uses Polarized Self-Attention for spatial-channel feature extraction. Input: H ∈ ℂ^(Nr × Nt) flat channel matrix
- **Temporal aspects**: None. Static channel snapshots only, no mobility or time-varying modeling
- **Key finding for CE-skip**: DL-CE complexity scales linearly with antenna count (NtNr), making skip scheduling increasingly valuable for ELAA. The PSA mechanism could serve as a representative DL-CE architecture when benchmarking inference cost

### Channelformer (arXiv:2302.04368)

- **CE-skip relevance**: HIGH -- Online training algorithm + temporal coherence analysis directly supports adaptive CE scheduling; shows Doppler coherence r_t=0.94-0.97 between adjacent OFDM symbols even at high Doppler
- **Dataset config**: SISO, 2.1 GHz, 72 subcarriers, 15 kHz SCS, 14 OFDM symbols/slot, EPA/EVA/ETU 3GPP channel models, SNR [-5, 30] dB
- **CE methods**: LS baseline, Channelformer (Transformer encoder + CNN decoder). 32K params (online) to 117K params (offline). Up to 70% pruning possible without significant degradation. Online version uses reference signals as self-supervised labels
- **Temporal aspects**: Time-domain interpolation within OFDM slot. Doppler coherence analysis: r_t = J0(2*pi*f_d*T_s) = 0.94-0.97 for adjacent symbols at 750-972 Hz Doppler spread. Online training exploits temporal stationarity windows
- **Key finding for CE-skip**: Temporal correlation coefficient r_t > 0.94 between adjacent symbols validates CE-skip premise -- channel changes slowly enough that reusing recent CE outputs is viable. Online training with 32K params shows lightweight models can adapt in real-time

### Transfer Learning vs Meta-Learning for CE (arXiv:2508.09751)

- **CE-skip relevance**: HIGH -- Fine-tuning period + inference period architecture is structurally analogous to CE-skip scheduling; shows optimal time-frequency window depends on velocity and SNR
- **Dataset config**: 3.5 GHz, 15 kHz SCS, 512 subcarriers, (Nt,Nr)=(2,16) MIMO-OFDM, CDL-B/CDL-E models, 60-120 km/h UE velocity, SNR [0, 20] dB. POSTECH/Samsung collaboration
- **CE methods**: CNN-based channel denoiser. Transfer learning (freeze+fine-tune layers) vs MAML (meta-learned initialization). Data-aided CE for generating fine-tuning labels. Pre-training on CDL-B, adaptation to CDL-E
- **Temporal aspects**: Explicit fine-tuning period (P,Q pilot observations) followed by inference period. Optimal (P,Q) depends on SNR and velocity -- higher velocity requires more frequent adaptation. At 120 km/h, performance degrades significantly without re-adaptation
- **Key finding for CE-skip**: The fine-tune/inference duty cycle directly maps to CE-skip scheduling. MAML achieves good adaptation with fewer shots (lower P), reducing overhead. Velocity-dependent optimal adaptation frequency validates adaptive (not fixed) CE scheduling

### Domain Adaptation for CE (arXiv:2507.08974)

- **CE-skip relevance**: LOW -- Focuses on cross-domain generalization (QSCM to MBCM), not temporal scheduling. Quasi-static block fading assumption ignores temporal dynamics
- **Dataset config**: 3.4 GHz, 30 kHz SCS, 612 subcarriers, 14 OFDM symbols, 1 BS antenna, CDL model. DeepMIMO I3 + MATLAB RayTracing. ETS Montreal
- **CE methods**: CNN and Pix2Pix GAN with transfer learning (freeze early convolutional layers, fine-tune later layers). Source: QSCM (QuaDRiGa SCM), Target: MBCM (map-based channel model via ray-tracing)
- **Temporal aspects**: Quasi-static block fading assumed. Future work mentions extending to time-varying channels but not addressed
- **Key finding for CE-skip**: Domain adaptation (freeze early layers) shows which DL-CE components are site-generic vs site-specific. Relevant for multi-site CE-skip where a single scheduling policy must generalize, but no direct temporal insights

### Continual Learning for Channel Prediction (arXiv:2506.22471)

- **CE-skip relevance**: HIGH -- Directly quantifies temporal prediction accuracy degradation across deployment scenarios; coherence time analysis and cross-config NMSE inflation (37.5%) motivate adaptive scheduling
- **Dataset config**: QuaDRiGa, 5 GHz, 100 MHz BW, 18 OFDM RBs, 500 time instants per scenario. UMi compact -> dense -> standard handover sequence. Stanford/ICML 2025
- **CE methods**: LSTM, GRU, Transformer backbones for channel prediction. Continual learning: Experience Replay (LARS), EWC, SI, LwF. Sequence-to-sequence prediction (T_in past -> T_out future)
- **Temporal aspects**: Channel coherence time: 0.3ms at 28 GHz / 60 km/h; 4ms feedback delay causes 50% sum-rate reduction at 3.5 GHz / 30 km/h. Cross-config NMSE inflation of 37.5% without continual adaptation. Experience Replay (LARS) best overall
- **Key finding for CE-skip**: Coherence time of 0.3ms at mmWave/high-mobility sets the floor for CE-skip interval. NMSE inflation of 37.5% across scenarios quantifies the cost of not re-running CE. Experience Replay enables prediction models to survive deployment changes -- critical for long-skip scenarios

### ReQuestNet: Foundational DL CE for 5G NR (arXiv:2508.08790)

- **CE-skip relevance**: HIGH -- Comprehensive 5G NR DL-CE with 3.6M params; iterative refinement (T=4 steps) means inference cost is 4x a single forward pass, making skip scheduling highly impactful for real-time deployment
- **Dataset config**: 2x2 MIMO, TDL/CDL profiles, 15/30 kHz SCS, 4-272 RBs, 0-450 Hz Doppler, SNR [0, 40] dB. Qualcomm. Tested across broad 5G NR parameter space
- **CE methods**: CoarseNet (U-Net 2D CNN, 1.2M params) + RefinementNet (T=4 iterative steps, 0.6M params each, total 3.6M). Likelihood Module injects noise statistics. Permutation equivariant across antenna pairs. Outperforms genie-aided MMSE by up to 10 dB at high SNR
- **Temporal aspects**: Processes full slot (14 OFDM symbols) jointly. Doppler range up to 450 Hz tested. No explicit temporal prediction, but robustness across Doppler spread implicitly captures temporal variation within a slot
- **Key finding for CE-skip**: At 3.6M params with T=4 iterative refinement, inference cost is substantial. Skip scheduling would multiply savings by 4x compared to single-pass models. The broad Doppler/SNR coverage makes this a realistic target architecture for CE-skip cost-benefit analysis

### NVIDIA Environment-Specific NRX for 5G NR (arXiv:2409.02912)

- **CE-skip relevance**: HIGH -- Provides exact GPU inference latency numbers: ~350us/iteration on A100 with TensorRT, 1ms total budget for N_it=2. This is the most concrete real-time CE cost data available
- **Dataset config**: 2x4 MU-MIMO UL, 3GPP UMi, 2.14 GHz, 47.5 MHz BW, 132 PRBs, 30 kHz SCS, UE velocity [0, 8] m/s. Sionna RT Munich map (same as our project). NVIDIA/ETH collaboration
- **CE methods**: CGNN (Conditional Graph Neural Network) with N_it iterations. RT variant: N_it=2, 1.4x10^5 weights, ~1ms on A100. Large variant: N_it=8, higher accuracy but >1ms. Site-specific fine-tuning with Sionna RT data
- **Temporal aspects**: UE velocity up to 8 m/s (low mobility). Catastrophic forgetting observed during site-specific fine-tuning -- performance on original distribution degrades. No explicit temporal prediction
- **Key finding for CE-skip**: **The 1ms inference budget on A100 is the key number.** At 0.5ms slot duration (30 kHz SCS), DL-CE takes ~2 slots -- skipping even one CE inference saves 1ms of GPU time. Catastrophic forgetting during site-specific fine-tuning means CE-skip must account for model staleness

### DL CSI Feedback with Temporal Correlation (arXiv:2505.23198)

- **CE-skip relevance**: MEDIUM -- Angle-difference feedback exploiting temporal CSI correlation demonstrates that temporal redundancy in channel state is exploitable; CSI refinement module using T previous reconstructions is analogous to CE-skip with interpolation
- **Dataset config**: DeepMIMO I3: (8,2,2) antennas, 2.4 GHz, 20 MHz, 256 SC / 64 sampled, 0.4-2.8 m/s. Wi-MIR: (3,3,3) antennas, 5 GHz, 20 MHz, 64 SC / 30 sampled. POSTECH/Samsung
- **CE methods**: Autoencoder-based CSI feedback (STA encoder + AP decoder). Angle-difference feedback: encode delta between consecutive CSI frames. CSI refinement module: attention-based fusion of T previous reconstructed CSIs
- **Temporal aspects**: Explicit temporal correlation exploitation. Angle-difference representation reduces entropy of CSI feedback. T-frame refinement at AP side. Processing time: STA ~4.4-6.5x10^-4s, AP refinement ~1.2-8.2x10^-2s
- **Key finding for CE-skip**: Angle-difference feedback shows that channel changes between frames are small and compressible. The T-frame refinement concept (using past T CSI reconstructions to improve current estimate) directly parallels CE-skip with interpolation/prediction from cached estimates

### ANFR: FL with Channel Attention (arXiv:2410.02006)

- **CE-skip relevance**: NONE -- Not about wireless channel estimation. Purely a federated learning method (weight standardization + channel attention) for image classification (CIFAR-10, CelebA, medical imaging)
- **Dataset config**: CIFAR-10, CelebA, Fed-ISIC2019, FedChest. ResNet-50 backbone. TMLR publication
- **CE methods**: N/A (computer vision, not CE)
- **Temporal aspects**: N/A
- **Key finding for CE-skip**: No relevance. The "channel" in the title refers to CNN feature channels, not wireless channels

### Channel Estimation and Reconstruction in Fluid Antenna Systems (IEEE TWC, Jan 2025)

- **CE-skip relevance**: LOW -- Traditional (non-DL) CE for fluid antenna systems. Oversampling and MLE-based estimation. No DL inference cost to skip
- **Dataset config**: Fluid antenna system with N_port movable ports. Rayleigh fading, spatial correlation modeled by Bessel function J0(2*pi*d/lambda). MLE-based estimation
- **CE methods**: LS, MLE with oversampling. Nyquist sampling analysis for spatial channel reconstruction. Closed-form expressions for estimation error. No DL methods
- **Temporal aspects**: None. Static spatial correlation analysis only
- **Key finding for CE-skip**: Spatial oversampling requirements in FAS show that more antenna ports increase CE overhead linearly -- motivation for skip scheduling in ELAA, but from a classical estimation perspective rather than DL inference cost

---

## Summary Comparison Table

| Paper | CE-skip Relevance | Freq (GHz) | Antennas | Subcarriers | DL-CE Params | Inference Cost | Temporal |
|-------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| PACE-Net | MEDIUM | 15 | 16x16 | N/A (flat, no OFDM) | ~moderate | O(DK^2 NtNr) | None |
| Channelformer | HIGH | 2.1 | SISO | 72 | 32K-117K | Low | r_t=0.94-0.97 |
| Transfer vs Meta CE | HIGH | 3.5 | (2,16) | 512 | ~moderate | Moderate | Fine-tune/infer cycle |
| Domain Adapt CE | LOW | 3.4 | 1 BS | 612 | CNN/GAN | Moderate | Quasi-static |
| Continual Learning | HIGH | 5.0 | - | 18 RBs | LSTM/GRU/Transformer | Moderate | T_c=0.3ms @28GHz |
| ReQuestNet | HIGH | varied | 2x2 | 4-272 RBs | 3.6M | High (4x iterative) | Within-slot |
| NVIDIA NRX | HIGH | 2.14 | 2x4 | 132 PRBs | 140K | **~1ms on A100** | [0,8] m/s |
| Temporal CSI FB | MEDIUM | 2.4/5.0 | (8,2,2) | 256/64 | Autoencoder | ~0.5-82ms | T-frame refinement |
| ANFR (FL) | NONE | - | - | - | - | - | - |
| Fluid Antenna | LOW | - | N ports | - | None (MLE) | - | None |

## Key Insights for CE-Skip Paper

1. **Inference cost baseline**: NVIDIA NRX provides the most concrete number -- ~1ms per CE inference on A100 with TensorRT (N_it=2). ReQuestNet's 4-step iterative refinement (3.6M params) would cost even more. These numbers justify skip scheduling.

2. **Temporal correlation validates skipping**: Channelformer shows r_t > 0.94 between adjacent OFDM symbols even at high Doppler (750-972 Hz). This means reusing recent CE outputs introduces < 6% correlation error.

3. **Adaptive scheduling is necessary**: Transfer vs Meta CE shows optimal adaptation frequency depends on velocity and SNR. Continual Learning shows 37.5% NMSE inflation across scenarios. Fixed skip intervals will underperform -- event-triggered scheduling is the right approach.

4. **Coherence time floor**: At 28 GHz / 60 km/h, coherence time is ~0.3ms (Continual Learning paper). At 30 kHz SCS (0.5ms slot), this means CE-skip interval should not exceed 1 slot at high mobility. Low mobility (NVIDIA, [0,8] m/s) allows much longer skip intervals.

5. **Model staleness vs skip savings**: NVIDIA NRX shows catastrophic forgetting during fine-tuning. CE-skip must track not only channel staleness but also model staleness -- a dual scheduling problem.

6. **Applicable DL-CE architectures for benchmarking**: ReQuestNet (iterative, 3.6M) and NVIDIA CGNN (graph-based, 140K) bracket the realistic inference cost range. Channelformer (32K online) represents the lightweight end.
