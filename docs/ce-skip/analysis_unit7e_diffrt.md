# Unit 7e: Differentiable RT and Misc Papers

Analysis of 12 papers for the CE-skip paper: "Event-Triggered CE Inference Scheduling for Software-Defined Base Stations"

---

## Paper-by-Paper Analysis

---

### 1. Learning Radio Environments by Differentiable Ray Tracing
**Hoydis, Ait Aoudia, Cammerer, Euchner, Nimier-David, ten Brink, Keller; IEEE Trans. MLCN 2024 (arXiv 2311.18558)**

**CE-skip Relevance: MEDIUM (shared simulation environment)**

This is the foundational Sionna RT differentiable ray tracing paper from NVIDIA. It introduces gradient-based calibration of scene parameters (materials, antenna patterns, scattering) using differentiable RT -- the same Sionna RT engine our CE-skip paper uses for channel generation.

**Simulation Configs:**
- Frequency: 3.438 GHz center, 50 MHz bandwidth
- Antennas: 2x distributed uniform planar 8x4 panels (32 elements each), lambda/2 spacing, vertical polarization
- Single-antenna dipole transmitter (remote-controlled)
- Scene: Indoor hallway + entrance hall at University of Stuttgart (DICHASUS channel sounder dataset "dichasus-dc01")
- 1024 OFDM subcarriers
- ~32.5k measurement positions (10k used: 5k train, 100 val, 4.9k test)
- Specular reflections up to 5th order, 1st order diffraction
- Synthetic validation: 256 random TX positions, up to 3 specular reflections + 1st-order diffuse scattering + diffraction
- Material model: ITU-R parameters (conductivity, permittivity, scattering coefficient, XPD)
- Training: Adam optimizer, lr=0.01, ~3000 steps (~10 min on RTX 3090)

**Key Technical Contributions:**
- Trainable antenna patterns via mixture of spherical Gaussians (von Mises-Fisher)
- Trainable scattering patterns (hemispherical Gaussian model)
- Trainable material embeddings: over-parametrized representation with read-out vectors (L=30 dims)
- "Neural materials": MLP with positional encoding that maps 3D coordinates to material properties
- Loss function: SMAPE on RMS delay spread + total channel gain
- Scaling factor estimation via EMA for uncalibrated measurements

**CE-skip connection:**
- Validates that Sionna RT produces realistic CIRs when calibrated against measurements
- Our CE-skip paper uses the same Sionna RT for synthetic channel generation; this paper provides the calibration methodology that ensures simulation fidelity
- The trainable material/antenna parametrizations could be cited to justify simulation realism
- Indoor scenario (not directly matching our outdoor Munich scene), but validates the RT engine itself

---

### 2. Fast, Differentiable, GPU-Accelerated Ray Tracing for Multiple Diffraction and Reflection Paths
**Eertmans, Lequeu, Legat, Jacques, Oestges; EuCAP 2026 (arXiv 2510.16172)**

**CE-skip Relevance: LOW**

A computational methods paper. Proposes a unified differentiable solver for ray path finding using BFGS optimization with implicit differentiation -- handles arbitrary reflection/diffraction sequences in a single framework. Implemented in JAX.

**Simulation Configs:**
- No wireless-specific simulation (purely geometric ray path finding)
- Benchmarks: 1000 paths in parallel, NVIDIA RTX 3070 (8 GB)
- Interaction depth n = 1 to 5 (reflections + diffractions)
- Planar objects only (d=2 for reflections, d=1 for diffractions)

**CE-skip connection:**
- Purely algorithmic contribution to RT path finding efficiency
- No channel model, no MIMO, no CE -- tangential at best
- Could be cited in passing as advancing the computational tools underlying Sionna-class RT engines

---

### 3. Fully Differentiable Ray Tracing via Discontinuity Smoothing for Radio Network Optimization
**Eertmans, Jacques, Oestges; arXiv 2401.11882**

**CE-skip Relevance: LOW**

Proposes smoothing techniques to handle gradient discontinuities in differentiable RT (caused by object occlusions, visibility transitions). Provides DiffeRT2d, a 2D Python library.

**Simulation Configs:**
- 2D toy scenarios only (no 3D, no frequency/antenna specs)
- Demonstrates antenna location optimization for 1-TX, 2-RX setup
- LOS paths only in the optimization example
- Smoothing functions: sigmoid and hard-sigmoid parametrized by alpha

**CE-skip connection:**
- Addresses a fundamental DRT limitation (vanishing/discontinuous gradients)
- No direct relevance to CE or channel estimation
- The method is compatible with 3D RT but only demonstrated in 2D

---

### 4. Radio Propagation Modelling: To Differentiate or To Deep Learn, That Is The Question
**Bakirtzis, Almasan, Suarez-Varela, Ferreira, Kalntis, Zanella, Wassell, Lutu; arXiv 2509.19337**

**CE-skip Relevance: MEDIUM (large-scale Sionna validation)**

First large-scale experimental comparison of differentiable RT vs DL models on real commercial MNO data. Uses Sionna 1.1.0 for RT simulations. Key finding: DL models outperform calibrated Sionna in accuracy while matching its speed.

**Simulation Configs:**
- Network: 10,000+ antennas across 13 areas (urban, suburban, rural) of a commercial MNO
- Frequency: Operating frequencies of real commercial antennas (not specified as a single value)
- Ground truth: 6 months of crowdsourced RSRP measurements (~300M total)
- Coverage areas: ~3,000 km^2 total, ~200 m^2 average per area
- Grid resolution: 512x512 tensors, resolutions of 2-5 m per cell depending on area group
- Max distances: 512-1280 m from antenna depending on group
- Sionna RT: version 1.1.0, launches 5x10^7 rays, up to 7 reflections
  - Also benchmarked Sionna 0.19 (previous version, 86.02 sec on 128 CPUs vs 0.1 sec on GPU)
- Scene construction: GIS data -> PLY meshes, material types (concrete, brick, glass)
- 3D antenna radiation pattern G(phi, theta), with steering angles and hardware losses
- DL models: U-Net, MaxViT, GAN MaxViT, MPNN, D-MPNN
  - All trained with AdamW, lr=10^-3, weight decay 5x10^-2
  - Input: 4-channel tensor (building height, land use, road network, FSPL-based baseline L)
  - 5-fold cross-validation across 13 areas
  - Hardware: 24 GB NVIDIA RTX A5000

**Key Results:**
- R2R (RT-to-RT): MaxViT/GAN variants best, RMSE ~10.74 dB, MAE ~7.03 dB
- R2M (RT-to-Measurement): Sionna achieves RMSE 15.77 (excl. no-coverage), MAE 12.92
- M2M (Measurement-to-Measurement): GAN MaxViT best at RMSE 9.85, MAE 7.76
- Site-specific calibration: Sionna-AMv (vectorized calibration) achieves ~7.7 dB MAE but DL still wins (~5 dB)
- Sionna's no-coverage points (79-91% coverage) are a significant practical limitation
- Speed: Sionna 1.1.0 GPU ~0.1 sec/scene vs DL ~1-7 sec

**CE-skip connection:**
- Validates Sionna RT as the state-of-the-art simulation tool for channel generation
- The finding that "DL outperforms calibrated RT" supports our approach of using DL-based CE rather than relying on RT alone
- Large-scale real-world validation across 13 cities provides credibility to Sionna-based synthetic data
- The 512x512 radio map resolution and coverage analysis methodology could inform our dataset design
- Can cite as evidence that Sionna-generated channels are realistic enough for ML training

---

### 5. VLM-Guided Differentiable Ray Tracing for Fast and Accurate Multi-Material RF Parameter Estimation
**Kang, Lim, Gu, Ko, Quek, Park; arXiv 2601.18242**

**CE-skip Relevance: LOW**

Uses a vision-language model (Gemini 2.5 Pro) to initialize material parameters and select measurement positions for inverse RT, accelerating convergence of gradient-based material calibration in Sionna.

**Simulation Configs:**
- Scene: 10m x 10m x 3m indoor room (Isaac Sim rendered)
- K = 9 objects (1 brick floor, 4 brick walls, 4 boxes of different materials)
- Frequency: 3.5 GHz
- TX power: 44 dBm
- Antennas: Single-element planar arrays, isotropic radiation pattern, vertical polarization
- U_ray = 5000 rays
- RT depth D = 4
- M = 3 measurement trials, N = 2-8 receivers per trial
- Adam optimizer, lr = 10^-3, max 1000 iterations
- Convergence: alpha = 10^-5, beta = 10^-4, patience J = 50
- GPU: GeForce RTX 5090
- Materials: ITU-R P.2040 conductivity values with Gaussian perturbation

**CE-skip connection:**
- Demonstrates Sionna's differentiable capabilities for inverse problems
- VLM-guided approach is orthogonal to CE-skip
- Indoor-only, single-element antennas -- very different from our ELAA outdoor scenario

---

### 6. Site-Specific RIS Deployment in Cellular Networks via Calibrated Ray Tracing
**Beyraghi, Shabanpour, Geraci, Almasan, Lozano; arXiv 2510.09478**

**CE-skip Relevance: MEDIUM (multi-band Sionna deployment)**

Automated RIS deployment framework using Sionna RT calibrated with real MNO measurement data. Covers 4G/5G/6G across three frequency bands. Validated on a UK city digital twin.

**Simulation Configs:**
- Scene: UK city, 1340m x 1390m, 12 BSs (18-56m height), 3 sectors each = 36 cells
- Frequencies: 4G at 2 GHz (FR1), 5G at 3.5 GHz (FR1), 6G at 10 GHz (FR3)
- Bandwidth: 20 MHz (4G), 100 MHz (5G), 200 MHz (6G)
- BS antenna arrays:
  - 4G: 2x2 planar, vertical polarization
  - 5G: 4x8 planar, vertical polarization
  - 6G: 4x16 planar, vertical polarization
- 3GPP TR 38.901 radiation pattern, HPBW 65 deg azimuth, 10 deg elevation
- TX power per cell: 43 dBm (4G), 49 dBm (5G), 44 dBm (6G)
- Subcarriers: 1200 (4G), 3276 (5G), 3276 (6G)
- UE grid: 2m x 2m tiles at 1.5m height
- RT: Fibonacci shoot-and-bounce, 10^7 rays/cell, up to 4 bounces
- Includes specular reflection, diffraction, scattering
- Beamforming: 2D DFT codebook per system
- Material calibration: permittivity, conductivity, surface scattering as learnable variables
  - Adam optimizer, 600 steps per cell
  - Calibrated over 10x10 m^2 regions with >=25 samples
  - Post-calibration error: mean -0.32 dB, median -0.13 dB, std 2.57 dB
- RIS: very large aperture (11.24 x 11.24 m), lambda/2 spacing

**CE-skip connection:**
- Multi-band (2/3.5/10 GHz) Sionna deployment with calibrated materials provides a reference for simulation realism
- The 4x16 array at 10 GHz is closer to our ELAA scenario than most papers
- Outage analysis (2.85%-6.07% across bands) provides context for where CE matters most
- Material calibration methodology (Adam optimizer, 600 steps) is directly applicable
- Can cite for multi-band simulation credibility and the observation that higher frequencies have more coverage holes (6.07% outage at 10 GHz vs 2.85% at 2 GHz)

---

### 7. U6G XL-MIMO Radiomap Prediction: Multi-Config Dataset and Beam Map Approach
**Li, Han, Lu, Jin, Wen; IEEE Trans. Wireless Commun. (arXiv 2603.06401)**

**CE-skip Relevance: HIGH (closest dataset to our scenario)**

Constructs the first large-scale XL-MIMO radiomap dataset with Sionna RT. 78,400 radiomaps across 800 urban scenes, 5 frequency bands, 9 array configurations (up to 32x32 UPA). Proposes "beam map" as a physics-informed spatial feature for cross-configuration generalization.

**Simulation Configs:**
- Scenes: 800 urban scenes from Nanjing metropolitan area (OpenStreetMap)
  - Each 1.28 x 1.28 km^2, min 5 buildings
  - Building heights from OSM "building:levels" metadata (5m/level, default 20m)
  - 3D modeling via Blender, Mitsuba XML format
  - Material properties: ITU-R P.2040 (roofs metallic, surfaces building-specific, ground concrete)
  - EM parameters frequency-dependent
- TX position: (0, 0, 40) m (macro BS deployment)
- Observation plane: 1280 x 1280 m^2, discretized to 128x128 grid (10 x 10 m^2 per cell)
- RX height: 1.5 m
- Frequency bands: 1.8, 2.6, 3.5, 4.9, 6.7 GHz
- Array configurations (UPAs):
  - At 6.7 GHz: 32x32 (64 beams), 16x16 (16 beams), 8x8 (8 beams), 16x32, 8x16
  - At 4.9 GHz: 8x16, 8x8
  - At 3.5 GHz: 8x8
  - At 2.6 GHz: 8x8, 4x8
  - At 1.8 GHz: 4x4, 2x4, 2x2
- Antenna element: 3GPP TR 38.901, G_E,max=8 dBi, A_max=30 dB, HPBW 65 deg, SLA_V=30 dB
- Beam steering: azimuth-only (theta_beam=90 deg), sweeping range depends on array size
  - 32x32: [-32, 31] deg at 1 deg interval (64 beams)
  - 16x16: [-28, 24.5] deg at 3.5 deg interval (16 beams)
  - 8x8: [-28, 21] deg at 7 deg interval (8 beams)
- RT: max interaction depth 3 (up to 3 reflections/diffractions per path)
  - 10^6 rays per TX-RX pair
  - LoS, specular reflection, edge diffraction (no diffuse scattering)
  - Adaptive receiver batching for GPU memory management
- Power normalization: P_t = 1 mW (0 dBm), radiomap = path loss + beamforming gain
- Total dataset: 800 scenes x 98 configs = 78,400 radiomaps, ~1.28 billion spatial measurements
- Building height maps: 256x256 at 5m resolution

**Beam Map (key contribution):**
- Physics-informed spatial feature encoding array-dependent LoS radiation pattern
- Decouples array-dependent radiation (computed analytically via beam map) from environment-dependent multipath (learned by NN)
- Enables cross-configuration generalization without retraining
- Integration: concatenated as additional input channel to RadioUNet/RME-GAN

**Key Results:**
- Task 1 (Blind prediction): RadioUNet+BeamMap MAE improvement up to 60% for cross-config
- Task 3a (Cross-config): beam map reduces MAE by 51.5% (RME-GAN) and 60.0% (RadioUNet)
- Task 3b (Cross-env): beam map reduces MAE by up to 50.5%
- Coverage ratio drops from 57% (2.6 GHz) to 31% (6.7 GHz) at -120 dBm threshold

**CE-skip connection:**
- **Most relevant dataset paper**: Uses Sionna RT with 3GPP antenna models and multi-band configs very similar to our setup
- The 3GPP TR 38.901 antenna element pattern is the same one we use
- XL-MIMO (up to 1024 elements) is directly relevant to our ELAA scenario
- Multi-frequency (1.8-6.7 GHz) spans our operating range
- The beam map concept (decoupling array radiation from environment multipath) is philosophically related to our site representation idea
- Coverage drop at higher frequencies (31% at 6.7 GHz) motivates the need for reliable CE
- **Dataset publicly available**: https://lxj321.github.io/MulticonfigRadiomapDataset/
- Can cite for: (1) Sionna RT dataset construction methodology, (2) multi-config XL-MIMO simulation parameters, (3) 3GPP antenna model implementation

---

### 8. Integrated Sensing and Edge AI: Realizing Intelligent Perception in 6G
**Liu, Chen, Wu, Wang, Chen, Niyato, Huang; arXiv 2501.06726**

**CE-skip Relevance: LOW-MEDIUM (survey, architectural context)**

Comprehensive survey on Integrated Sensing and Edge AI (ISEA) in 6G. Covers communication-AI-sensing convergence, digital air interface, over-the-air computation, and advanced signal processing.

**Key Points for CE-skip:**
- Defines ISEA as task-oriented paradigm unifying communication, AI computation, and sensing
- Edge AI provisioning: MEC paradigm delivering AI to resource-limited devices with low latency
- Latency targets: 30 ms inference latency, near 100% accuracy for autonomous driving
- Over-the-air computation for distributed sensing/inference
- Foundation models integration as future direction

**CE-skip connection:**
- Provides 6G architectural context for where CE-skip fits (edge AI for PHY processing)
- The survey's "IAAC" (Integrated AI and Communications) usage scenario aligns with CE-skip's vision of AI-driven PHY
- Can cite for general 6G edge AI positioning, but no specific CE or channel estimation focus
- No simulation configs relevant to our work

---

### 9. Fine-Grained AI Model Caching and Downloading with CoMP Broadcasting
**Fu, Qin, Zhang, Cheng, Lu, Wang; IEEE Trans. Wireless Commun. (arXiv 2509.19341)**

**CE-skip Relevance: LOW-MEDIUM (model distribution for edge AI)**

Proposes fine-grained AI model caching at edge nodes exploiting parameter reusability (shared pre-trained backbone + task-specific heads). Uses CoMP broadcasting to deliver shared parameter blocks to multiple users simultaneously.

**Key Technical Points:**
- Exploits parameter reusability: fine-tuned models share frozen pre-trained parameters
- Parameter Block (PB) caching: stores model fragments, not whole models
- CoMP broadcasting: delivers shared PBs to multiple users
- Multi-agent learning framework for distributed caching decisions
- 3GPP 6G specs: model downloading within seconds (general) or 10-100 ms (time-sensitive)

**Simulation Configs:**
- Multi-cell edge network (abstract, no specific RT/channel model)
- AI models: vision models (ResNet variants, ViT variants) from HuggingFace
- Storage capacity: 2-8 GB per edge node
- Model sizes: 44M-632M parameters

**CE-skip connection:**
- The "shared backbone + task-specific heads" concept parallels our 3-way model (E + theta_task + theta_BS)
- CoMP for model delivery is relevant if CE-skip models need to be distributed across BSs
- 3GPP latency requirements (10-100 ms for time-sensitive) provide context for CE-skip's sub-ms scheduling
- However, this paper is about model distribution, not model execution -- different layer of the stack

---

### 10. Bridging 6G IoT and AI: LLM-Based Efficient Approach for Physical Layer Optimization
**Mehmood, Hassan, Kraidy; arXiv 2602.06819**

**CE-skip Relevance: LOW**

Proposes PE-RTFV (Prompt Engineering - Real-Time Feedback and Verification) framework using LLMs for physical layer optimization. Tested on wireless-powered IoT constellation design.

**Key Points:**
- O-LLM generates task-specific prompts, A-LLM produces solutions
- Iterative refinement using real-time system feedback (gradient-descent-like)
- Tested on rate-energy region optimization for modulation design
- Achieves near-genetic-algorithm performance in few iterations
- No retraining required

**CE-skip connection:**
- Demonstrates LLM-based physical layer optimization -- conceptually interesting but completely different approach from our DL-based CE
- The "real-time feedback" concept loosely relates to CE-skip's event-triggered monitoring
- No simulation configs, no channel model, no CE relevance

---

### 11. Training Machine Learning at the Edge: A Survey
**Khouas, Bouadjenek, Hacid, Aryal; arXiv 2403.02619**

**CE-skip Relevance: LOW-MEDIUM (edge training context)**

Comprehensive survey on edge ML training: federated learning, knowledge distillation, transfer learning, on-device training. Covers frameworks, metrics, and applications.

**Key Points for CE-skip:**
- Defines edge learning taxonomy: distributed (FL, split learning) vs local (on-device, transfer learning)
- Edge constraints: limited compute, memory, data availability
- FL dominates edge learning research
- Frameworks: TensorFlow Lite, PyTorch Mobile, ONNX Runtime, FedML, Flower
- Metrics: convergence speed, communication efficiency, privacy guarantees

**CE-skip connection:**
- Our CE-skip model could potentially be trained/fine-tuned at the edge (BS-level adaptation)
- FL techniques are relevant for our collaborative BS training (E1 experiments)
- Knowledge distillation could compress CE models for faster skip decisions
- Survey nature -- no specific configs or experimental results to extract
- Can cite for general edge training landscape if needed

---

### 12. Machine Intelligence on Wireless Edge Networks
**Vadlamani, Sulimany, Gao, Chen, Englund; arXiv 2506.12210**

**CE-skip Relevance: LOW**

Proposes broadcasting model weights as RF waveforms and performing inference using in-physics computation inside the radio receive chain (RF mixers and filters). A radical approach to edge inference.

**Key Technical Points:**
- Weights transmitted as frequency-multiplexed RF signals from BS
- Client encodes activations, computes via RF diode ring mixer (analog MVM)
- No model storage on client device
- ENOB (Effective Number of Bits) as precision metric
- Optimal energy window for accurate analog inner products
- Hardware-tailored training through differentiable RF chain model

**CE-skip connection:**
- Orthogonal to CE-skip: this is about how to execute NN inference using analog RF hardware
- The "broadcasting weights" concept is interesting but our CE-skip operates at a higher abstraction level
- Could theoretically enable ultra-fast CE inference if the CE model weights were broadcast -- but this is speculative
- No relevant simulation configs

---

## Summary Table

| # | Paper | CE-skip Relevance | Sionna RT? | Key Sim Configs | Primary Value for CE-skip |
|---|-------|-------------------|------------|-----------------|--------------------------|
| 1 | Hoydis - Diff RT Calibration | MEDIUM | Yes (core) | 3.438 GHz, 8x4 UPA, indoor, DICHASUS | Validates Sionna RT calibration methodology |
| 2 | Eertmans - Fast GPU RT | LOW | No (JAX) | Geometric only | Advancing RT computational tools |
| 3 | Eertmans - Discontinuity Smoothing | LOW | No (DiffeRT2d) | 2D only | DRT gradient challenges |
| 4 | Bakirtzis - Diff RT vs DL | MEDIUM | Yes (1.1.0) | 10k+ antennas, 13 areas, real MNO | DL outperforms calibrated RT at scale |
| 5 | Kang - VLM-Guided DRT | LOW | Yes | 3.5 GHz, indoor 10x10m | VLM for material init |
| 6 | Beyraghi - RIS + Calibrated RT | MEDIUM | Yes | 2/3.5/10 GHz, multi-band, UK city | Multi-band calibrated Sionna deployment |
| 7 | Li - U6G XL-MIMO Radiomap | **HIGH** | Yes | 1.8-6.7 GHz, up to 32x32 UPA, 800 scenes | Closest dataset to our ELAA scenario |
| 8 | Liu - ISEA Survey | LOW-MEDIUM | No | Survey | 6G edge AI architectural context |
| 9 | Fu - CoMP Model Caching | LOW-MEDIUM | No | Abstract network | Model distribution for edge AI |
| 10 | Mehmood - LLM PHY Optimization | LOW | No | IoT testbed | LLM-based PHY optimization |
| 11 | Khouas - Edge Training Survey | LOW-MEDIUM | No | Survey | Edge ML training landscape |
| 12 | Vadlamani - RF Edge Intelligence | LOW | No | Analog RF | In-physics inference concept |

---

## Key Takeaways for CE-skip Paper

### Simulation Environment Validation
Papers #1, #4, #6, #7 collectively validate Sionna RT as the standard tool for 6G channel simulation:
- **Calibration methodology** (Hoydis): Trainable materials + antenna patterns with SMAPE loss
- **Scale validation** (Bakirtzis): 10k+ real antennas across 13 areas show Sionna is realistic but DL models can match/exceed it
- **Multi-band credibility** (Beyraghi): 2/3.5/10 GHz with calibrated materials on a real UK city
- **XL-MIMO dataset** (Li): 32x32 UPA at 6.7 GHz with 3GPP antenna model -- closest to our ELAA setup

### Dataset Config Comparison with Our Work
Our CE-skip paper uses Sionna RT for Munich urban scene with ELAA configs. The closest reference is Li et al. (#7):
- **Shared**: 3GPP TR 38.901 antenna element, UPA arrays, urban scenes, Sionna RT
- **Different**: They use Nanjing (800 scenes) vs our Munich (1 scene); they focus on radiomaps (RSRP) vs our CFR/CIR

### Citable Claims
1. "Sionna RT produces realistic channels validated against real-world measurements" -- cite #1, #4, #6
2. "XL-MIMO with 1024+ elements at upper mid-band frequencies is the target for 6G" -- cite #7
3. "DL models trained on RT-generated data achieve accuracy comparable to real measurements" -- cite #4
4. "Higher frequencies (6+ GHz) suffer significantly more coverage holes, making reliable CE more critical" -- cite #6 (6.07% outage at 10 GHz), #7 (31% coverage at 6.7 GHz)
5. "Edge AI inference must operate within sub-ms latency for real-time PHY functions" -- cite #8, #9

### Not Directly Relevant but Interesting
- VLM-guided material calibration (#5): Could improve our scene setup but is not about CE
- In-physics RF inference (#12): Radical approach to edge inference, speculative connection to CE
- LLM for PHY optimization (#10): Different paradigm entirely
