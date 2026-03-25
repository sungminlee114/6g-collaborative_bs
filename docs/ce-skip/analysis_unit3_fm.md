# Unit 3: Wireless Foundation Models & Channel Dataset Analysis

> CE-skip paper context: "Event-Triggered CE Inference Scheduling for Software-Defined Base Stations"
> Reference config: 3.5/15/28 GHz, 8x8 to 32x32 antennas, 256-4096 subcarriers, Munich UMi (Sionna RT), 8 BS, UE 10-150m

---

## 1. LWM (Large Wireless Model) [2411.08872]

**CE-skip Relevance: HIGH** -- First wireless channel FM; its 600K-param encoder with ~2ms inference validates that FM inference cost matters, directly motivating CE skip scheduling.

### Architecture & Training
- **Architecture**: Transformer encoder-only (ViT-style), 12 layers, 12 heads, D=64
- **Parameters**: 600K (encoder only; relatively small by FM standards)
- **Pre-training**: Masked Channel Modeling (MCM), 15% masking ratio (80/10/10 mask/random/keep)
- **Loss**: MSE (regression, not cross-entropy -- continuous channel values)
- **Optimizer**: Adam, lr=1e-4, batch=64

### Dataset Config
| Parameter | Value |
|-----------|-------|
| Antennas (BS) | 32 |
| Antennas (UE) | 1 |
| Subcarriers | 32 |
| Frequency | 3.5 GHz (pre-train); 28 GHz (downstream beam pred) |
| Channel model | Ray-tracing (DeepMIMO) |
| Scenes | 15 scenarios (O1, Boston5G, ASU Campus, 12 US cities) |
| Training samples | 820K train + 200K val (~1M total) |
| Downstream test | 6 unseen cities (Denver, Fort Worth, etc.), 14,840 samples |
| Mobility | Not explicitly modeled (static snapshots) |

### CE/Prediction Methods & Cost
- Downstream tasks: beam prediction (sub-6 to mmWave), LoS/NLoS classification
- Downstream model: Residual 1D-CNN (500K params) on top of LWM embeddings
- No channel estimation task explicitly, but embeddings improve CE downstream models
- Inference time not reported for LWM itself, but downstream overhead is minimal

### Temporal Modeling
- **No explicit temporal modeling**. Processes single-snapshot spatial-frequency CSI (32 ant x 32 SC)
- No time dimension in patches; no prediction horizon
- Captures spatial-spectral dependencies only

### CE-skip Implications
- LWM demonstrates that even a small FM (600K) provides significant representation gains
- The pre-train once, deploy everywhere paradigm directly supports site-adaptive CE scheduling
- Lack of temporal modeling is a gap -- CE skip explicitly needs temporal awareness

---

## 2. WiFo (Wireless Foundation Model) [2412.08908]

**CE-skip Relevance: HIGH** -- Directly addresses temporal channel prediction with zero-shot generalization; the most relevant FM for CE skip scheduling.

### Architecture & Training
- **Architecture**: MAE-based encoder-decoder (ViT backbone), asymmetric design
- **Sizes**: WiFo-Tiny (0.3M) to WiFo-Large (86.1M); default WiFo-Base = 21.6M
- **WiFo-Base**: Enc: 6 layers, 512 width, 8 heads; Dec: 4 layers, 512 width, 8 heads
- **3D Patching**: (t, k, n) = (4, 4, 4) covering time-frequency-space
- **STF Positional Encoding**: Separate sincos PE for time, frequency, space dimensions
- **Pre-training tasks**: 3 masked reconstruction tasks (random 85%, time 50%, frequency 50%)
- **Optimizer**: AdamW, lr=5e-4, cosine decay, 200 epochs, batch=128

### Dataset Config
| Parameter | Value |
|-----------|-------|
| Antennas (BS) | 4 to 32 (UPA: 1x4 to 4x8) |
| Antennas (UE) | 1 (MISO) |
| Subcarriers | 32, 64, 128 |
| Frequency | 1.5, 2.5, 4.9, 5.9 GHz (pre-train); 3.5, 6.7, 28 GHz (test) |
| Subcarrier spacing | 90, 180, 360 kHz |
| Channel model | QuaDRiGa (3GPP compliant) |
| Scenarios | UMi, UMa, RMa, Indoor (LoS/NLoS) |
| User speed | 0-300 km/h |
| Time samples | T = 16 or 24 RBs |
| Time intervals | Delta_t = 0.25-1 ms |
| Training samples | 160K total (16 datasets x 10K each) |
| Test datasets | D17 (3.5 GHz), D18 (6.7 GHz), D19 (28 GHz) |
| SNR | 20 dB noise added |

### CE/Prediction Methods & Cost
- **Time-domain prediction**: Predict future T/2 RBs from historical T/2 RBs
- **Frequency-domain prediction**: Predict last K/2 subcarriers from first K/2
- **Zero-shot inference**: No fine-tuning needed for new scenarios
- Inference cost scales with model size (0.3M to 86.1M params)
- Outperforms LLM4CP (84.6M params, 17.23ms inference) with faster inference

### Temporal Modeling
- **Full temporal modeling** via 3D space-time-frequency patches
- Prediction horizon: T/2 RBs (8-12 future time steps at 0.25-1ms intervals)
- Time-masked reconstruction explicitly trains causal temporal relationships
- Handles velocities from 0 to 300 km/h

### CE-skip Implications
- WiFo's temporal prediction is essentially the core capability CE skip needs: predict whether future CE is necessary
- Zero-shot generalization across scenarios eliminates per-site retraining
- Model cost (21.6M for Base) is non-trivial -- CE skip scheduling could decide when to invoke this vs. cheaper LS
- WiFo-Tiny (0.3M) shows a scalable cost-performance tradeoff relevant to scheduling decisions
- **Key gap**: WiFo predicts CSI, not CE quality metrics; CE skip needs channel stationarity detection

---

## 3. WirelessGPT [2502.06877]

**CE-skip Relevance: MEDIUM** -- Multi-task FM covering CE, prediction, and sensing; demonstrates computational cost-performance tradeoffs across tasks.

### Architecture & Training
- **Architecture**: Transformer-based, multi-domain encoder with cross-domain self-attention
- **Parameters**: 80M (initial), scalable to 800M
- **Pre-training**: Masked patch reconstruction (unsupervised), 3D space-time-frequency patches
- **Positional encoding**: 3-domain PE (temporal PE noted as unnecessary)
- **Datasets**: Traciverse (self-developed, 300GB, 27 cities, 100+ scenarios), SionnaRT, DeepMIMO

### Dataset Config
| Parameter | Value |
|-----------|-------|
| Antennas (BS) | 4x4 (dual pol, 32 elements for prediction) |
| Antennas (UE) | 1 (MISO) |
| Subcarriers | 32 (CE), 32 (prediction), 114 (HAR) |
| Frequency | 2.4 GHz (CE, prediction), 5 GHz (HAR), 3.5-60 GHz (reconstruction) |
| Channel model | WINNER II (CE), QuaDRiGa (prediction), Sionna (reconstruction) |
| Scenarios | Multiple (indoor, outdoor, urban, rural) |
| Velocity | 40-100 km/h (prediction task) |
| Training samples | 60K/10K/20K (CE), 160K/32K/128K (prediction) |
| Pilot length | 64 (CE task) |
| SNR | -5 to 10 dB (CE), 0-30 dB (prediction) |

### CE/Prediction Methods & Cost
- **Channel estimation**: OFDM with 64-length pilot, QPSK, WirelessGPT + Transformer/ResCNN downstream
  - Inference time: 2.31ms (w. Transformer), 1.53ms (w. ResCNN)
  - Training: 188.3ms (w. Transformer), 115.2ms (w. ResCNN)
  - Foundation model itself: 79.6M params
- **Channel prediction**: Predict 4 future slots from 16 historical slots
  - Inference: 2.26ms (w. Transformer), 5.12ms (w. LSTM)
  - Compared to LLM4CP: 17.23ms inference -- WirelessGPT is ~7.5x faster
- **Key cost insight**: Pre-trained representations reduce downstream training/inference time vs. raw channel input

### Temporal Modeling
- Captures temporal dependencies via causal self-attention
- Prediction horizon: 4 future slots from 16 historical (16->4 temporal extrapolation)
- Handles 40-100 km/h velocity range

### CE-skip Implications
- **2.31ms CE inference** with Transformer head directly validates CE skip paper's DL-CE cost assumption (~2ms)
- Multi-task design shows that sensing + CE can share computation -- CE skip could coordinate
- 80M params is expensive; the cost hierarchy (LS < LMMSE < DL-CE < FM-CE) is well-illustrated
- WirelessGPT's inference cost (2.26ms) comparable to DL-CE, but with richer representations

---

## 4. ContraWiMAE [2505.09160]

**CE-skip Relevance: MEDIUM** -- Introduces channel estimation as a downstream task of self-supervised FM; directly measures CE NMSE from pretrained representations.

### Architecture & Training
- **Architecture**: Asymmetric encoder-decoder MAE + contrastive head (InfoNCE)
- **Parameters**: ~570K total (CWiMAE 12,4); Enc: 410K, Dec: 143K, Contrastive head: 16.5K
- **Encoder**: 12 layers, d_e=64, M_enc=16 heads
- **Decoder**: 4 layers, d_e=64, M_dec=8 heads
- **Mask ratio**: m_r = 0.90 (90% masking -- very aggressive)
- **Pre-training**: 3000 epochs, batch=8096, AdamW, lr=3e-4, cosine decay
- **Contrastive learning**: Positive pairs via AWGN noise augmentation (5-40 dB), tau=0.2

### Dataset Config
| Parameter | Value |
|-----------|-------|
| Antennas (BS) | 32 (ULA) |
| Antennas (UE) | 1 |
| Subcarriers | 32 |
| Subcarrier spacing | 30 kHz |
| Bandwidth | 960 kHz |
| Frequency | 3.5 GHz (pre-train), 28 GHz (downstream) |
| Channel model | Ray-tracing (DeepMIMO) |
| Scenes | 56 scenarios worldwide (pre-train), 10 unseen (downstream) |
| Max propagation paths | 20 |
| Training samples | 2.5M (pre-train), 0.55M (downstream) |
| SNR (CE task) | 0-40 dB |
| Polarization | Single |

### CE/Prediction Methods & Cost
- **Channel estimation**: Uplink, pilot on single subcarrier, reconstruct full channel from partial noisy observation
  - Pretrained ContraWiMAE used as-is (no fine-tuning) achieves competitive NMSE
  - Fine-tuning with 2.5-100% data budget further improves
  - At 0 dB SNR: ~0 dB NMSE; at 40 dB: ~-27 dB NMSE (finetuned)
- **Computational cost** (Table V):
  - Encoder-decoder (m_r=0.90): Latency=0.057ms, Throughput=17534 samples/s, GFLOPs=0.033
  - Encoder-only (m_r=0): Latency=0.237ms, Throughput=4221 samples/s, GFLOPs=0.126
  - High masking ratio dramatically reduces compute

### Temporal Modeling
- **No temporal modeling**. Operates on single-snapshot spatial-frequency CSI (Ns x Nf)
- No time dimension considered
- Static channel snapshots only

### CE-skip Implications
- Demonstrates FM-based CE achievable at 0.057-0.237ms -- potentially cheaper than traditional DL-CE
- 90% masking dramatically reduces computational cost while maintaining quality
- Contrastive + reconstructive learning makes representations robust to noise -- relevant for deciding CE skip threshold
- 570K params is very lightweight; could serve as always-on CE quality monitor
- AWGN-based positive pair generation: directly models the noise conditions CE skip must handle

---

## 5. CSI-MAE [2601.03789]

**CE-skip Relevance: MEDIUM** -- Trained on 3GPP statistical channel models (not scene-specific ray-tracing), demonstrating cross-scenario generalization for CE tasks.

### Architecture & Training
- **Architecture**: MAE with ViT-Base encoder, lightweight decoder
- **Masking ratio**: 75%
- **Pre-training**: MSE loss on masked patches only
- **Positional embedding**: 2D sine-cosine (antenna x subcarrier)
- **CLS token**: Prepended for global representation

### Dataset Config
| Parameter | Value |
|-----------|-------|
| Antennas (BS) | 8x8 = 64 |
| Antennas (UE) | 2x2 = 4 |
| Subcarriers | 256 |
| Subcarrier spacing | 15, 30, 60 kHz |
| Frequency | 0.7, 2.4, 3.5, 4.9, 5 GHz |
| Channel model | 3GPP TR 38.901 (statistical) |
| Scenarios | UMi, UMa, RMa |
| BS height | 10m (UMi), 25m (UMa), 35m (RMa) |
| UE velocity | 0-27.78 m/s (0-100 km/h) |
| UE height | 1.5m |
| Training samples | ~1.45M total |
| Sionna RT | Used for data generation |

### CE/Prediction Methods & Cost
- **Channel extrapolation**: Antenna-domain and subcarrier-domain (50% masking for prediction)
- **Channel feedback**: CSI compression and reconstruction
- **User positioning**: CLS token -> linear regression to 2D coordinates
- **Zero-shot**: Fine-tuned on RMa-2.4, tested on RMa-0.7 and RMa-3.5 without retraining
  - Zero-shot outperforms supervised baseline trained on target data

### Temporal Modeling
- **No explicit temporal modeling**. Spatial-frequency only (antenna x subcarrier)
- UE velocity included in data generation but not as a model dimension
- Channel extrapolation is spatial/spectral, not temporal

### CE-skip Implications
- 3GPP statistical models (vs. ray-tracing) provide broader generalization -- relevant for real deployments
- 8x8 antenna config matches our project's smallest config exactly
- 256 subcarriers matches our smallest subcarrier config
- Zero-shot cross-frequency generalization directly supports CE skip: a model trained at one frequency can inform scheduling at another
- Scaling law validated: more data and larger models consistently help

---

## 6. SpectrumFM [2505.06256]

**CE-skip Relevance: LOW** -- Focuses on spectrum management (AMC, WTC, spectrum sensing, anomaly detection) using IQ-level signals, not CSI/channel estimation.

### Architecture & Training
- **Architecture**: Hybrid CNN + Multi-head self-attention encoder, L layers
- **Pre-training tasks**: Masked reconstruction + next-slot signal prediction
- **Next-slot prediction**: LSTM aggregation of encoder output -> predict N-th symbol from N-1 history
- **Input**: IQ signals converted to Amplitude-Phase (AP) representation, sample-level normalization

### Dataset Config
| Parameter | Value |
|-----------|-------|
| Input type | IQ signals (not CSI) |
| Datasets | RML2018.01A (2.55M samples), TechRec (9.7M samples), Self-collected (~9M) |
| Sample shape | 1024x2 or 128x2 (IQ pairs) |
| SNR range | -20 to 30 dB (RML), -6 to 12 dB (TechRec) |
| Modulation types | 24 (OOK, BPSK, QPSK, QAM variants, FM, etc.) |
| Total pre-train data | ~25 GB |
| Frequency bands | Various (licensed + unlicensed) |

### CE/Prediction Methods & Cost
- No channel estimation -- operates at signal level
- Downstream: AMC, WTC, spectrum sensing, anomaly detection
- Next-slot prediction provides temporal modeling but at IQ level, not channel level

### Temporal Modeling
- Next-slot signal prediction: predict IQ sample N from samples 1..N-1
- LSTM used for temporal aggregation
- But this is signal-level temporal, not channel temporal coherence

### CE-skip Implications
- Limited direct relevance to CE skip
- Next-slot prediction concept transferable: CE skip could similarly predict "next CE quality"
- Spectrum sensing (occupied/free detection at -4 dB) demonstrates robust detection under noise -- analogous to channel stationarity detection
- Foundation model paradigm validated in different wireless domain

---

## 7. DeepTelecom [2508.14507]

**CE-skip Relevance: LOW** -- Dataset paper (not a model); provides LoD3 digital-twin channel data generation framework using Sionna RT. Relevant as data infrastructure.

### Architecture & Training
- **Not a model paper** -- dataset generation pipeline
- LLM-assisted scene modeling (material annotation, XML generation)
- GPU-accelerated ray-tracing via Sionna + OptiX

### Dataset Config
| Parameter | Value |
|-----------|-------|
| Scene type | Indoor (LiDAR) + Outdoor (Google 3D Tiles) |
| Level of detail | LoD3 (per-surface material annotation) |
| Ray tracer | Sionna RT (GPU-accelerated, differentiable) |
| Physics | Reflection, refraction, diffraction (UTD) |
| Output | CIR, CFR, AoA, AoD, path-loss, delay, Doppler |
| Antenna configs | Configurable MIMO arrays (e.g., 4x4 UPA) |
| Frequency | Multi-band (3.5 GHz to 60 GHz mentioned) |
| RIS support | Yes (single-beam focusing + multi-beam optimization) |
| Mobility | MT velocity vectors for time-varying CIR |
| Data format | HDF5 (channel tensors), CSV, MP4 videos, images |

### CE/Prediction Methods & Cost
- Not applicable (dataset generation, not inference)

### Temporal Modeling
- Supports time-varying channels via MT velocity vectors
- Generates temporal CIR sequences
- But no prediction model -- just data

### CE-skip Implications
- Provides complementary dataset for CE skip experiments
- LoD3 material-aware scenes produce higher-fidelity channels than typical LoD1
- Multi-modal output (images + channels) could enable vision-aided CE scheduling
- Sionna RT integration matches our project's toolchain

---

## 8. Sionna RT Technical Report [2504.21719]

**CE-skip Relevance: LOW** -- Technical documentation of the ray-tracing engine we use for data generation. Not a CE method, but foundational infrastructure.

### Key Features (v1.0+)
- Fully differentiable ray tracer (Dr.Jit + Mitsuba backend)
- SBR + image method hybrid for CIR computation
- Hashing-based path deduplication
- Supports: reflection, refraction, diffraction (UTD), diffuse reflection, RIS, mobility
- Doppler shift computation for moving TX/RX
- Interoperable with TensorFlow and PyTorch
- Radio map computation (coverage maps)

### Dataset Generation Capabilities
| Parameter | Value |
|-----------|-------|
| Path solver | SBR + image method |
| Interactions | Reflection, refraction, diffraction, diffuse |
| Mobility | Doppler shifts computed from velocity vectors |
| Output | CIR h(tau) = sum_n a_n * delta(tau - tau_n) |
| Differentiable | Yes (gradients w.r.t. materials, geometry, antennas) |
| GPU acceleration | NVIDIA OptiX backend |
| Scene format | Mitsuba XML, OpenStreetMap integration |

### CE-skip Implications
- Our data generation engine -- understanding its capabilities informs experimental design
- Differentiable RT could enable gradient-based optimization of CE scheduling policies
- Mobility support enables generating temporal channel sequences for skip/no-skip training
- Doppler computation provides physical velocity feature for stationarity detection

---

## Summary Comparison Table

| Paper | Type | Params | Pre-train Data | Antennas | Subcarriers | Frequency | Channel Model | Temporal | CE Task | CE-skip Relevance |
|-------|------|--------|---------------|----------|-------------|-----------|--------------|----------|---------|------------------|
| LWM | FM | 600K | 1M (DeepMIMO) | 32 | 32 | 3.5 GHz | Ray-trace | No | No | HIGH |
| WiFo | FM | 0.3-86M | 160K (QuaDRiGa) | 4-32 | 32-128 | 1.5-28 GHz | 3GPP stat. | **Yes** | No | HIGH |
| WirelessGPT | FM | 80M | 300GB (Traciverse+) | 4-32 | 32-114 | 2.4-60 GHz | WINNER/QuaDRiGa | **Yes** | **Yes** | MEDIUM |
| ContraWiMAE | FM | 570K | 2.5M (DeepMIMO) | 32 | 32 | 3.5/28 GHz | Ray-trace | No | **Yes** | MEDIUM |
| CSI-MAE | FM | ViT-Base | 1.45M (Sionna+3GPP) | 64 | 256 | 0.7-5 GHz | 3GPP stat. | No | Extrap. | MEDIUM |
| SpectrumFM | FM | - | 25GB (IQ data) | N/A | N/A | Various | N/A (IQ) | Partial | No | LOW |
| DeepTelecom | Dataset | N/A | - | Config. | Config. | 3.5-60 GHz | Sionna RT | Supported | N/A | LOW |
| Sionna RT | Engine | N/A | - | Config. | Config. | Any | Deterministic | Supported | N/A | LOW |

---

## Key Findings for CE-skip Paper

### 1. FM Computational Cost Hierarchy Validated
The papers collectively establish a clear inference cost hierarchy:
- **ContraWiMAE**: 0.057ms (encoder+decoder with 90% masking) -- cheapest FM
- **ContraWiMAE**: 0.237ms (encoder-only, full processing)
- **WirelessGPT w. ResCNN**: 1.53ms
- **WirelessGPT w. Transformer**: 2.31ms (matches our DL-CE ~2ms assumption)
- **LLM4CP**: 17.23ms (most expensive)

This validates the CE-skip paper's core premise: FM-based CE is expensive enough to warrant intelligent scheduling.

### 2. Temporal Modeling Gap
Only **WiFo** and **WirelessGPT** handle temporal channels:
- WiFo: 3D STF patches, predicts T/2 future time steps, zero-shot across scenarios
- WirelessGPT: Causal self-attention, 16->4 slot prediction, 40-100 km/h

Most FMs (LWM, ContraWiMAE, CSI-MAE) operate on single snapshots. CE skip scheduling fills this gap by using lightweight temporal stationarity detection to decide when expensive FM/DL-CE inference is needed.

### 3. Pre-training Data Scale
| Model | Samples | Data Size | Scenarios |
|-------|---------|-----------|-----------|
| LWM | ~1M | - | 15 |
| WiFo | 160K | - | 16 configs |
| WirelessGPT | Millions | 300GB | 100+ |
| ContraWiMAE | 2.5M | - | 56 |
| CSI-MAE | 1.45M | - | 15 (3 models x 5 freq) |

Our project: 8 BS x 100 snapshots x 100 UE = 80K per config -- comparable to WiFo's per-dataset scale.

### 4. Config Overlap with Our Project
| Config | Our Project | Best Match |
|--------|------------|------------|
| 3.5 GHz | Yes | LWM, ContraWiMAE, CSI-MAE, WiFo(D17) |
| 15 GHz | Yes | No FM covers this (gap!) |
| 28 GHz | Yes | ContraWiMAE, WiFo(D19), LWM(downstream) |
| 8x8 ant | Yes | CSI-MAE (exact match) |
| 16x16 ant | Yes | No exact (LWM=32 ULA closest) |
| 32x32 ant | Yes | No exact (LWM=32 ULA, but 1D not 2D) |
| 256 SC | Yes | CSI-MAE (exact match) |
| 1024+ SC | Yes | No FM covers large SC counts |
| Munich UMi | Yes | No FM uses Munich; DeepMIMO cities closest |

**Notable gap**: No FM covers 15 GHz (FR2 lower band) or large antenna arrays (32x32 = 1024 elements). Our project uniquely addresses these scales.

### 5. Argumentation for CE-skip Paper
These FM papers strengthen the CE-skip argument in three ways:

1. **Cost motivation**: FMs are even more expensive than DL-CE (80M params vs. typical CE nets). As the wireless community moves toward FMs, intelligent scheduling becomes critical.

2. **Temporal prediction**: WiFo and WirelessGPT show FMs can predict future channels, but at significant compute cost. CE skip offers a lightweight alternative: instead of predicting future channels, detect when the channel hasn't changed enough to require re-estimation.

3. **Zero-shot generalization**: WiFo's zero-shot capability suggests FM-based CE could be deployed without per-site training. But inference cost remains -- CE skip scheduling is complementary, reducing how often even a zero-shot FM needs to run.

### 6. Recommended Citations
- **WiFo** [2412.08908]: Most relevant -- temporal STF prediction, zero-shot generalization
- **WirelessGPT** [2502.06877]: Multi-task FM with explicit CE task and 2.31ms inference cost
- **ContraWiMAE** [2505.09160]: Lightweight FM-based CE with measured latency (0.057-0.237ms)
- **LWM** [2411.08872]: First wireless FM, establishes paradigm
- **CSI-MAE** [2601.03789]: 3GPP-based training, cross-scenario zero-shot
