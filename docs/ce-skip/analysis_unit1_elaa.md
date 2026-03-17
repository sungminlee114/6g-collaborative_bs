# Unit 1: ELAA/XL-MIMO Channel Estimation — CE-Skip Relevance Analysis

## Summary

These 13 papers cover the ELAA/XL-MIMO CE landscape from practical DL-based estimators to comprehensive surveys. Most focus on the spatial domain (near-field spherical wavefront, spatial non-stationarity) rather than the temporal domain, making direct CE-skip relevance generally LOW to MEDIUM. The key insight for CE-skip is that ELAA near-field channels have higher spatial complexity (angle + distance coupling) but no paper explicitly addresses temporal persistence or when CE inference can be skipped. The Wideband XL-MIMO Deep Unrolling paper and the Distributed SP survey are most relevant as they deal with computational cost reduction — the same motivation behind CE-skip scheduling.

## Paper Analysis

### 1. XLCNet: Lightweight DL-Based CE for XL-MIMO (arXiv:2402.08916)
- **CE-skip relevance**: MEDIUM — Lightweight DL-CE architecture directly applicable as the "Full tier" CE method in CE-skip; compression (pruning + quantization) reduces inference cost, complementary to skip scheduling
- **Dataset config**:
  - Antennas: 256 (ULA)
  - Frequency: 30 GHz (lambda = 0.01m)
  - Bandwidth: not specified (narrowband)
  - Subcarriers: not specified (single-carrier)
  - Channel model: Hybrid-field (mixed near+far-field paths), parametric steering vector model
  - Scene: Synthetic (parametric)
  - Mobility: not explicitly specified (static snapshots implied)
  - UE distance: 10-80 m (uniform)
  - Num BS: 1
  - SNR range: -10 to 20 dB
- **CE methods**: LS (initial), XLCNet (2D CNN residual refinement), LMMSE, HOMP (hybrid-field OMP), V-MRDN
- **Temporal aspects**: None. Static channel snapshots only. No time-varying channel consideration.
- **Key finding for CE-skip**: C-XLCNet achieves 10x complexity reduction and 36x model size reduction with limited performance loss — demonstrates that lightweight CE inference is feasible, supporting the CE-skip framework where Full tier CE must be computationally affordable.

---

### 2. Wideband XL-MIMO CE via Deep Unrolling (arXiv:2505.07717)
- **CE-skip relevance**: MEDIUM — Proposes constrained deep unrolled PGD network for wideband XL-MIMO CE; addresses computational cost which is the key motivation for CE-skip
- **Dataset config**:
  - Antennas: 512 (ULA) / 2048 (UPA, 256x8)
  - Frequency: 100 GHz
  - Bandwidth: 10 GHz
  - Subcarriers: 256
  - Channel model: Near-field spherical wavefront with spatial non-stationarity (visibility regions), multipath L=3
  - Scene: Synthetic (parametric with VR modeling)
  - Mobility: not specified (static)
  - UE distance: 5-30 m (scatterer distance)
  - Num BS: 1
  - SNR range: -5 to 10 dB
- **CE methods**: LMMSE, OMP (polar-domain), ISTA-Net+, AMP-SBL, D2-CNN, Constrained unrolled PGD network (proposed)
- **Temporal aspects**: None. Purely spatial estimation, no temporal channel evolution.
- **Key finding for CE-skip**: Deep unrolling achieves -20 dB NMSE at SNR=10 dB with UPA — significantly outperforms traditional methods. The constrained formulation ensures monotonic descent, relevant for reliable CE in the Full tier.

---

### 3. LLM4XCE: XL-MIMO CE via LLM (arXiv:2512.08955)
- **CE-skip relevance**: LOW — Uses GPT-2 backbone for CE; interesting but extremely heavy for real-time inference, opposite of CE-skip's goal of reducing compute
- **Dataset config**:
  - Antennas: 256 (ULA)
  - Frequency: 30 GHz (lambda = 0.01m)
  - Bandwidth: not specified (narrowband)
  - Subcarriers: not specified (single-carrier)
  - Channel model: Hybrid-field (mixed near+far-field paths), same as XLCNet
  - Scene: Synthetic (parametric)
  - Mobility: not specified (static)
  - UE distance: 10-80 m (uniform)
  - Num BS: 1
  - SNR range: -5 to 20 dB
- **CE methods**: LS, LMMSE, HY-OMP, XLCNet, MAT-CENet, LLM4XCE (GPT-2 based, 17M trainable + 109M frozen)
- **Temporal aspects**: None. Mentions mobility causes near/far-field switching but does not model temporal dynamics.
- **Key finding for CE-skip**: LLM4XCE has 126M parameters total — far too heavy for per-slot CE inference. This motivates CE-skip: if Full-tier CE is expensive (like LLM-based), skipping it when channel is stable saves massive compute.

---

### 4. NF Beam Training for XL-MIMO via DL (arXiv:2406.03249)
- **CE-skip relevance**: LOW — Beam training (not CE), codebook-free beamforming from CSI; different problem domain
- **Dataset config**:
  - Antennas: 256 (ULA, half-wave linear array)
  - Frequency: 50 GHz
  - Bandwidth: 0.03 * fc = 1.5 GHz
  - Subcarriers: 256
  - Channel model: Near-field mmWave channel model
  - Scene: Synthetic
  - Mobility: not specified (static, T=1000 time frames for data generation but not temporal)
  - UE distance: 5-50 m
  - Num BS: 1
  - SNR range: -20 to 20 dB
- **CE methods**: Not a CE paper — focuses on beam training. Uses near-field hierarchical, far-field hierarchical, exhaustive search as baselines
- **Temporal aspects**: None. Static beam training per snapshot.
- **Key finding for CE-skip**: Shows codebook-free DL beam training achieves near-exhaustive-search performance — relevant context that beam management overhead also benefits from intelligent scheduling.

---

### 5. NF Beam Training & CE for XL-MIMO: Survey (arXiv:2504.05578)
- **CE-skip relevance**: MEDIUM — Comprehensive survey of XL-MIMO beam training and CE techniques; useful reference for CE method taxonomy
- **Dataset config**:
  - Antennas: Various (survey covers 128 to 512+)
  - Frequency: mmWave/THz (survey scope)
  - Bandwidth: not specified (survey)
  - Subcarriers: not specified (survey)
  - Channel model: Near-field spherical wavefront models (survey covers multiple)
  - Scene: not specified (survey)
  - Mobility: not specified (survey)
  - UE distance: Within Rayleigh distance (near-field)
  - Num BS: 1 (survey scope)
  - SNR range: not specified (survey)
- **CE methods**: Polar-domain CS (SOMP, P-SIGW), distance-parameterized angular-domain methods, DFT-based beam training, neural network-assisted CE, hierarchical codebooks
- **Temporal aspects**: None explicitly. Mentions real-world channel data validation as open challenge. Notes FR3 (7-24 GHz) beam training as emerging area.
- **Key finding for CE-skip**: Identifies that most existing studies use synthetic near-field channel models — highlights gap in real-world validation. Notes that practical XL-MIMO needs hybrid beam architectures where CE overhead is even higher (FDD systems need pilot overhead proportional to antenna count).

---

### 6. CE for 6G Near-Field: Comprehensive Survey (arXiv:2507.23526)
- **CE-skip relevance**: MEDIUM — Most comprehensive CE survey covering parametric, CS-based, and DL methods for near-field; useful taxonomy and Table IV/V summarize all methods
- **Dataset config**:
  - Antennas: Various (survey: ULA, UPA, UCA configurations discussed)
  - Frequency: mmWave to THz (survey scope)
  - Bandwidth: not specified (survey)
  - Subcarriers: not specified (survey)
  - Channel model: Near-field LoS, multipath, Rayleigh, Rician models for MISO/MIMO
  - Scene: not specified (survey)
  - Mobility: Mentions Doppler factor exp(j2pi*f_D*t) in channel model (Eq. 32)
  - UE distance: Within Rayleigh distance
  - Num BS: 1 (with RIS-aided scenarios)
  - SNR range: not specified (survey)
- **CE methods**: LS, LMMSE, MUSIC, ESPRIT, SAGE, OMP/SOMP variants, polar-domain CS, SBL, CNN-based, deep unfolding, A-RCE+TFIST, D-STiCE (LSTM-based time-varying CE), FL-based CE
- **Temporal aspects**: YES — Mentions D-STiCE method [124] using LSTM for time-varying THz LoS CE that captures temporally correlated sparse channel parameters. Also mentions spatially non-stationary multipath channels [125] with SAGE algorithm. Doppler explicitly included in Rician model.
- **Key finding for CE-skip**: D-STiCE (deep sparse time-varying CE) is the most relevant work — it uses LSTM to track temporal channel evolution, directly supporting the premise that channels have temporal persistence that can be exploited. The FL-based CE [147] achieves 12x lower pilot overhead, relevant to CE-skip's pilot reduction goal.

---

### 7. XL-MIMO Tutorial: Near-Field 6G (arXiv:2310.11044)
- **CE-skip relevance**: LOW — Tutorial/overview covering modeling, performance analysis, and design; no specific CE algorithms or temporal aspects
- **Dataset config**:
  - Antennas: Various theoretical examples (128 ULA, UPA configurations)
  - Frequency: 3.5, 28, 73 GHz (Rayleigh distance examples)
  - Bandwidth: not specified (tutorial)
  - Subcarriers: not specified (tutorial)
  - Channel model: NUSW (non-uniform spherical wave), USW, PBW, UPW models; LoS and multipath
  - Scene: not specified (tutorial)
  - Mobility: not specified
  - UE distance: Discussed theoretically (Rayleigh distance analysis)
  - Num BS: 1 (collocated, sparse, modular, distributed architectures)
  - SNR range: not specified (tutorial)
- **CE methods**: Discusses LS, LMMSE, polar-domain CS, near-field beam codebook design, DAM (delay alignment modulation) conceptually
- **Temporal aspects**: None. Purely spatial/static analysis.
- **Key finding for CE-skip**: Provides fundamental understanding of near-field channel models (NUSW vs UPW) and Rayleigh distance calculations at different frequencies. At 3.5 GHz with 256 antennas, Rayleigh distance ~100m — most UEs in near-field for ELAA.

---

### 8. Distributed Signal Processing for ELAA (arXiv:2407.16121)
- **CE-skip relevance**: MEDIUM — Distributed CE (DCE) algorithms for single-BS ELAA with decentralized baseband processing; computational cost reduction is directly relevant to CE-skip motivation
- **Dataset config**:
  - Antennas: 128 to 1024 (analysis range), typical example M=256 with C clusters
  - Frequency: not specified (general framework)
  - Bandwidth: 80 MHz mentioned for fronthaul cost analysis
  - Subcarriers: 192 (16 resource blocks) in complexity analysis
  - Channel model: Multi-carrier ELAA with antenna clustering, angle-delay domain sparsity
  - Scene: not specified (general framework)
  - Mobility: not specified
  - UE distance: not specified
  - Num BS: 1 (single BS with distributed nodes) + multi-cell CoMP and cell-free scenarios
  - SNR range: not specified
- **CE methods**: Centralized DMMSE, AGE (aggregate-then-estimate), EAG (estimate-then-aggregate), distributed LMMSE, LC-MUE (linear compression)
- **Temporal aspects**: None explicitly. Notes that channel estimation must be done per coherence interval.
- **Key finding for CE-skip**: Fronthaul cost scales linearly with antenna count (Fig 2a); computational complexity scales cubically for LMMSE. DCE framework reduces both — CE-skip adds another dimension by reducing temporal frequency of CE. The paper's sparse aggregation DCE achieves comparable MSE to centralized at much lower cost.

---

### 9. Decentralized CE for XL-MIMO with DBP (arXiv:2501.17059)
- **CE-skip relevance**: MEDIUM — Two-stage local sparse reconstruction + global refinement for XL-MIMO with hybrid beamforming; SBL-GNNs algorithm reduces complexity
- **Dataset config**:
  - Antennas: N_t total in ULA (ELAA), divided into M subarrays of N antennas each; typical N_t = 256-512 implied
  - Frequency: mmWave (implied by half-wavelength spacing d = lambda_c/2)
  - Bandwidth: Wideband (OFDM with K subcarriers)
  - Subcarriers: K (not specific number given in system model)
  - Channel model: Uniform spherical wavefront with G multipath, near-field with quadratic phase terms, wideband dual-wideband effect
  - Scene: Synthetic (parametric)
  - Mobility: not specified (static)
  - UE distance: Near-field regime (within Rayleigh distance)
  - Num BS: 1 (with star-topology DBP: M LPUs + 1 CPU)
  - SNR range: not specified in system model section
- **CE methods**: Centralized SBL, fully decentralized SBL, proposed two-stage: local SBL-GNNs + global Bayesian refinement with Markov chain hierarchical prior
- **Temporal aspects**: None. Static channel estimation per time slot.
- **Key finding for CE-skip**: SBL-GNNs achieves comparable accuracy to centralized at much lower complexity — important because decentralized architectures are where CE-skip would be most beneficial (each subarray can independently decide to skip).

---

### 10. Sub-Array ELAA Pilot Scheme (arXiv:2512.10478)
- **CE-skip relevance**: MEDIUM — Novel pilot scheme exploiting spatial non-stationarity to reduce pilot overhead; directly related to CE-skip's overhead reduction goal
- **Dataset config**:
  - Antennas: M = M_bar x M_tilde (sub-arrays x antennas per sub-array), example: 256 sub-arrays
  - Frequency: 2.6 GHz (sub-6G)
  - Bandwidth: not specified
  - Subcarriers: N = 512 (OFDM)
  - Channel model: Sub-array structured ELAA with spatial non-stationarity, visibility regions, COST2100 model (SemiUrban_VLA_2_6GHz)
  - Scene: Semi-urban (COST2100)
  - Mobility: not specified (static)
  - UE distance: Example: User1 at (0,0,10)m, User2 at (-50,100,1.5)m
  - Num BS: 1
  - SNR range: not specified in detail
- **CE methods**: O-CDM, SR-FDM, NO-CDM (baselines), proposed N-FD-CDM with Turbo-MRF Bayesian inference; group-wise LMMSE
- **Temporal aspects**: Notes "statistical sparsity of the environment tends to remain relatively stable over time" — directly supports CE-skip premise that channel support structure is temporally persistent.
- **Key finding for CE-skip**: The key observation that "CIR support patterns remain stable over time" is a critical assumption for CE-skip. If the channel support (which sub-arrays see which scatterers) changes slowly, CE can be skipped when the support pattern hasn't changed. Uses sub-6G frequency (2.6 GHz), closest to our 3.5 GHz config.

---

### 11. Integrated CE and Sensing for NF ELAA (arXiv:2601.18333)
- **CE-skip relevance**: MEDIUM — Tensor decomposition for joint CE and sensing (localization) in OFDM ELAA; sensing provides geometric info that could trigger CE updates
- **Dataset config**:
  - Antennas: N = 128-256 (ULA), half-wavelength spacing
  - Frequency: 100 GHz (THz, LoS scenario) / 30 GHz (NLoS scenario)
  - Bandwidth: B = 0.1 GHz
  - Subcarriers: P = 64
  - Channel model: Near-field spherical wavefront, LoS (CPD model) and NLoS (BTD model), OFDM
  - Scene: Synthetic (parametric)
  - Mobility: not specified (static users in near-field)
  - UE distance: 20-80 m (U(20,80)), near-field region
  - Num BS: 1 (with M = 32-64 RF chains)
  - SNR range: -10 to 20 dB (varied)
- **CE methods**: CPD (canonical polyadic decomposition) + ALS for LoS, BTD (block term decomposition) + NLS for NLoS; baselines: SOMP, SIGW
- **Temporal aspects**: None explicitly. But integrated sensing provides user position estimates that could serve as CE-skip triggers (if user hasn't moved, skip CE).
- **Key finding for CE-skip**: Joint CE+sensing recovers user positions with millimeter accuracy above 5 dB SNR — position information from sensing can serve as a monitor for CE-skip: if estimated position hasn't changed, channel likely hasn't changed either.

---

### 12. NF Beamfocusing with Modular Linear Arrays (arXiv:2505.07991)
- **CE-skip relevance**: LOW — Beamfocusing analysis and localization for modular arrays; theoretical, no CE algorithm
- **Dataset config**:
  - Antennas: N per ULA, L ULAs (modular); examples: 2x16=32, 2x25=50, 2x64=128
  - Frequency: 15 GHz (upper mid-band)
  - Bandwidth: not specified (narrowband, single-carrier)
  - Subcarriers: not specified
  - Channel model: Free-space LoS (Fresnel approximation), exact near-field
  - Scene: Synthetic (free-space)
  - Mobility: not specified (static)
  - UE distance: 30 m (broadside example), within Fraunhofer distance
  - Num BS: 1 (modular array)
  - SNR range: not specified (beampattern analysis, not CE evaluation)
- **CE methods**: Parametric CE via MUSIC angle estimation + triangulation for localization; matched filter (MF) beamfocusing
- **Temporal aspects**: None. Static beamfocusing analysis.
- **Key finding for CE-skip**: MLAs achieve beamfocusing with significantly fewer antennas than dense ULAs (128 vs 200 for 2m aperture). Per-subarray angle estimation + triangulation enables low-complexity localization — modular architecture naturally supports distributed CE-skip decisions.

---

### 13. Towards 6G MIMO: Massive Spatial Multiplexing (arXiv:2401.02844)
- **CE-skip relevance**: LOW — Tutorial/vision paper on UM-MIMO covering near-field propagation, DoF, CE, EM theory, antenna design; no specific CE algorithms for temporal scheduling
- **Dataset config**:
  - Antennas: Various examples (10x10=100 to 100x50=5000)
  - Frequency: 3.5 GHz, 30 GHz (examples)
  - Bandwidth: not specified (tutorial)
  - Subcarriers: not specified (tutorial)
  - Channel model: Exact near-field (Green's function), Fresnel approximation, beamfocusing analysis
  - Scene: Free-space (theoretical)
  - Mobility: not specified
  - UE distance: Within Rayleigh distance (250m for 1x0.5m array at 30 GHz)
  - Num BS: 1
  - SNR range: not specified (tutorial)
- **CE methods**: Discusses LS, LMMSE conceptually; focuses on beamfocusing, spatial DoF, EM theory
- **Temporal aspects**: None. Static analysis.
- **Key finding for CE-skip**: At 30 GHz with 5000-antenna array (1x0.5m), Rayleigh distance is 250m — entire cell in near-field. Beamwidth (eq 17) is 1.772/N radians, meaning narrow beams that are sensitive to small position changes — argues for frequent CE updates (against skipping) in near-field.

---

## Dataset Config Comparison Table

| Paper | Antennas | Freq (GHz) | BW | SC | Model | Mobility | Dist (m) |
|-------|----------|-----------|-----|-----|-------|----------|------|
| XLCNet | 256 ULA | 30 | n/s | n/s | Hybrid-field parametric | Static | 10-80 |
| Wideband Unrolling | 512 ULA / 2048 UPA | 100 | 10 GHz | 256 | NF spherical + VR | Static | 5-30 |
| LLM4XCE | 256 ULA | 30 | n/s | n/s | Hybrid-field parametric | Static | 10-80 |
| NF Beam DL | 256 ULA | 50 | 1.5 GHz | 256 | NF mmWave | Static | 5-50 |
| NF BT+CE Survey | Various | mmW/THz | n/s | n/s | Survey (multiple) | n/s | NF region |
| NF CE Survey | Various | mmW/THz | n/s | n/s | Survey (multiple) | Doppler mentioned | NF region |
| XL-MIMO Tutorial | Various | 3.5/28/73 | n/s | n/s | NUSW/USW/PBW/UPW | n/s | Rayleigh dist |
| Distributed SP | 128-1024 | n/s | 80 MHz | 192 | Multi-carrier ELAA | n/s | n/s |
| Decentralized CE | ~256-512 ULA | mmWave | Wideband | K | Spherical waveform + DBP | Static | NF region |
| Sub-Array Pilot | 256 sub-arrays | 2.6 | n/s | 512 | COST2100 SemiUrban | Static* | 10-100+ |
| Integrated CE+Sens | 128-256 ULA | 100/30 | 0.1 GHz | 64 | NF spherical LoS/NLoS | Static | 20-80 |
| NF Beamfocusing MLA | 32-128 modular | 15 | n/s | n/s | Free-space LoS | Static | 30 |
| Chae 6G MIMO | 100-5000 | 3.5/30 | n/s | n/s | Exact NF (tutorial) | n/s | Rayleigh dist |

*n/s = not specified*
*\* = notes channel support is temporally stable*

## Key Takeaways for CE-Skip Paper

1. **Temporal gap**: Nearly all papers assume static channels — none address when to re-estimate. This is the gap CE-skip fills.
2. **Computational motivation**: Papers consistently show CE complexity scales cubically with antenna count (LMMSE) or requires heavy DL models (LLM4XCE: 126M params). CE-skip directly addresses this by reducing CE frequency.
3. **Spatial non-stationarity supports CE-skip**: The Sub-Array paper explicitly notes CIR support patterns are "relatively stable over time" — a key physical justification for CE-skip.
4. **Integrated sensing enables monitoring**: The CE+Sensing paper shows position can be estimated alongside CE — position stability can serve as a CE-skip trigger.
5. **Config alignment**: Our 3.5 GHz config is underrepresented (only Chae tutorial and Sub-Array at 2.6 GHz). Most papers use 30-100 GHz. Our 256-antenna config matches XLCNet/LLM4XCE. Our OFDM subcarrier counts (256-4096) span the range used in these papers.
6. **D-STiCE is key reference**: The only temporal CE method found (referenced in NF CE Survey [124]) uses LSTM to track time-varying THz LoS channels — directly validates CE-skip's premise of temporal correlation.
