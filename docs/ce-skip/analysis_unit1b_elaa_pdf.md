# Unit 1b: ELAA/XL-MIMO CE Papers (PDF-only extraction)

Extracted from 8 papers focusing on antenna configs, CE methods, and simulation setups.
Relevance to CE-skip: these define the ELAA channel estimation landscape our scheduling operates over.

---

## 1. XLCNet: Lightweight DL-based CE for XL-MIMO (2402.08916)

**Gao, Dong, Pan, You (Southeast Univ.) -- Feb 2024**

| Parameter | Value |
|-----------|-------|
| **Antennas** | M = 256 ULA, d = lambda/2 |
| **Frequency** | 30 GHz (lambda = 0.01 m) |
| **Bandwidth/Subcarriers** | Not specified (narrowband, single subcarrier) |
| **Channel model** | Hybrid near-field + far-field; unified H = AG formulation; Rayleigh distance D_Ray = M^2 * lambda / 2 = 200 m at M=200 |
| **CE methods compared** | LS, LMMSE, HOMP (hybrid-field OMP), V-MRDN, **XLCNet** (2D CNN, residual), **C-XLCNet** (pruned + quantized) |
| **CE cost** | XLCNet: 263K weights, 0.2ms; C-XLCNet (kappa=0.9, b=8): 29K weights, 0.078ms. Complexity O(M * N_w). Reduces 10x compute, 36x model size |
| **Temporal** | Static (no time-varying) |
| **Mobility** | Not specified |
| **SNR range** | -10 to 20 dB |
| **Training** | 90K train / 10K val / 2K test; Adam, lr=1e-3, 200 epochs + 50 fine-tune; batch=128 |

**CE-skip relevance:**
- C-XLCNet's 0.078ms inference = extremely fast, making skip scheduling overhead-sensitive
- 10x-36x compression shows DL-CE cost can vary dramatically -- skip threshold should adapt to model complexity
- Universal NF+FF applicability means single estimator works across Rayleigh boundary

---

## 2. Wideband XL-MIMO CE via Constrained Deep Unrolling (2505.07717)

**Zheng, Lyu, Wang, Gong (Pengcheng Lab / SUSTech) -- May 2025**

| Parameter | Value |
|-----------|-------|
| **Antennas** | ULA: N = 512, d = lambda/2; UPA: N1 x N2 = 256 x 8 = 2048, d = lambda/2 |
| **Frequency** | f_c = 100 GHz |
| **Bandwidth** | B = 10 GHz |
| **Subcarriers** | M = 256 |
| **RF chains** | N_RF = 4 |
| **Channel model** | Near-field spherical wavefront + spatial non-stationarity (visibility regions) + frequency-dependent beam split. L = 3 paths |
| **CE methods compared** | LMMSE, OMP (polar-domain), ISTA-Net+, AMP-SBL, D2-CNN (U-Net), **PGD-Net** (constrained unrolled PGD, T=5 layers) |
| **CE cost** | FLOPs (x10^9): ISTA-Net+ 2.54, AMP-SBL 2.16, D2-CNN 3.77, PGD-Net 3.79 |
| **Temporal** | Static |
| **Mobility** | Not specified |
| **SNR range** | -5 to 10 dB |
| **Key results** | PGD-Net at SNR=10dB: -19.21 dB NMSE (ULA), -20.04 dB (UPA). Monotonic descent constraint ensures layer-wise improvement |

**CE-skip relevance:**
- 3.79 x 10^9 FLOPs per inference = very heavy; skip scheduling saves significant compute
- Wideband (10 GHz BW at 100 GHz) = extreme beam split; channel varies across subcarriers even if static
- MAP formulation with learned proximal operator -- skip decision could monitor proximal residual

---

## 3. LLM4XCE: LLM-based CE for XL-MIMO (2512.08955)

**Li, Li, Dong (Nanjing Univ. Aeronautics) -- Dec 2025**

| Parameter | Value |
|-----------|-------|
| **Antennas** | M = 256 ULA, d = lambda/2 |
| **Frequency** | 30 GHz (lambda = 0.01 m) |
| **Bandwidth/Subcarriers** | Not specified (narrowband) |
| **Channel model** | Hybrid near-field + far-field (same as XLCNet); D_R = M^2*lambda/2 = 327 m |
| **CE methods compared** | LS, LMMSE, HY-OMP, XLCNet, MAT-CENet (mixed attention transformer), **LLM4XCE** (GPT-2 backbone) |
| **Architecture** | Parallel Feature-Spatial Attention + GPT-2 (d=768); freeze layers 1-10, fine-tune layers 11-12. 17M trainable + 109M frozen params |
| **Temporal** | Static; notes "user channels may shift between near-field and far-field due to mobility" |
| **Mobility** | Mentioned but not simulated |
| **SNR range** | -5 to 20 dB |
| **Training** | 45K train / 5K val / 2K test; batch=64, epochs=200, lr=0.001, Adam |

**CE-skip relevance:**
- 126M total params (17M trainable) = heaviest CE model in survey; ideal candidate for skip scheduling
- Semantic representation approach -- could enable skip decisions based on semantic channel similarity
- Explicitly mentions mobility causing NF/FF transitions -- our CE-skip targets exactly this scenario

---

## 4. NF Beam Training & CE for XL-MIMO: Survey (2504.05578)

**Zeng, Wang, Li, Hao, Chu, Xie, Wang, Pham -- Apr 2025**

This is a **survey paper** (no own simulations). Key taxonomic findings:

| Category | Methods Surveyed |
|----------|-----------------|
| **Beam training** | Polar-domain codebook, hierarchical codebook, DFT-based 2-phase, multi-beam (sparse array), off-grid |
| **CE in uplink TDD (hybrid precoding)** | Polar-domain SOMP, P-SIGW (off-grid), distance-parameterized angular-domain, hybrid-field 2-phase, SBL with near-field codebook |
| **CE in downlink FDD** | Pattern-coupled SBL (block-sparse), polar-domain dictionary with reduced coherence |
| **CE for point-to-point** | Multidimensional OMP, unified LoS/NLoS model |
| **Open challenges** | Real measurement validation, sensing-aided CE, FR3 band (7-24 GHz) CE |

**Antenna configs mentioned in reviewed papers:**
- 256 ULA at 30 GHz (D_Ray ~ 200 m)
- 512 ULA at 100 GHz (survey example)
- UPA configurations not deeply covered

**CE-skip relevance:**
- Confirms no existing work on temporal/scheduling aspects of CE in XL-MIMO
- FR3 band (7-24 GHz) identified as open direction -- intermediate NF/FF regime
- Sensing-aided CE could provide side information for skip decisions

---

## 5. Decentralized CE for XL-MIMO with DBP (2501.17059)

**Tang, Wang, Pan, Zeng et al. (Southeast Univ. / KTH / PUC-Rio) -- Jan 2025**

| Parameter | Value |
|-----------|-------|
| **Antennas** | N_t = 128 ULA (ELAA), divided into M = 4 subarrays of N = 32 each |
| **Frequency** | f_c = 30 GHz |
| **Bandwidth** | f_s = 1.6 GHz |
| **Subcarriers** | K = 16 pilot carriers |
| **RF chains** | N_RF = 1 per subarray |
| **Channel model** | Near-field spherical wavefront + dual-wideband effect; G = 4 paths; distance-dependent + frequency-dependent steering vectors; spatial non-stationarity |
| **CE methods compared** | StdSBL, PC-SBL, VSP, UAMP-MRF, **SBL-GNNs** (proposed, GNN replaces M-step); + centralized Bayesian refinement with Markov chain prior |
| **Architecture** | DBP star topology: LPUs (local) + CPU (global fusion). Two-stage: local sparse reconstruction then global Bayesian refinement |
| **CE cost** | SBL-GNNs: O(K^3 N^2 P N_RF T + ...) per subarray. Runtime: SBL-GNNs 0.004s (LPU), overall 0.442s-0.471s. Much faster than centralized (2.379s) |
| **Temporal** | Static |
| **Mobility** | Distance [10, 50] m |
| **SNR range** | 0 to 10 dB |
| **Training** | SBL-GNNs: 50 epochs, 320 batches x 64 samples; Adam, lr=1e-4 |

**CE-skip relevance:**
- Decentralized architecture = per-subarray skip decisions possible (heterogeneous scheduling)
- Two-stage (local + global refinement) = natural skip granularity (skip refinement stage when channel stable)
- GNN-based SBL converges in T=5 iterations vs hundreds for traditional -- skip threshold differs by method

---

## 6. Sub-Array ELAA Pilot Scheme (2512.10478)

**Zhang, Guo, Lau (HKUST) -- Dec 2025**

| Parameter | Value |
|-----------|-------|
| **Antennas** | M = 1024 total = M_bar (128 sub-arrays) x M_tilde (8 per sub-array); sub-array spacing 0.21 m (5x half-wavelength); vertical half-wavelength spacing |
| **Frequency** | f_c = 3.5 GHz (sub-6G) |
| **Bandwidth** | 15 kHz subcarrier spacing |
| **Subcarriers** | N = 1024 OFDM, 72 cyclic prefix |
| **Channel model** | COST 2100 model; spatial non-stationarity (limited visibility regions); 2D clustered sparsity in antenna-delay domain; planar wavefront per sub-array |
| **CE methods compared** | LMMSE, LMMSE-genie, VAMP-BG (Bernoulli-Gaussian), OMP, **Turbo-MRF** (proposed, 2D Markov Random Field prior) |
| **Pilot scheme** | N-FD-CDM: non-orthogonal frequency-division + code-division multiplexing; users grouped by visibility region |
| **Temporal** | Static |
| **Mobility** | Users at [20,10,1.5]m and [-50,100,1.5]m |
| **SNR range** | 0 to 20 dB |
| **Users** | K = 24, 32, 36 tested |

**CE-skip relevance:**
- Sub-6G frequency = longer coherence time, more opportunities for CE skip
- Sub-array structured ELAA = practical deployment model; spatial non-stationarity means different sub-arrays see different channels
- Pilot overhead reduction (N-FD-CDM saves 2/3 vs NR orthogonal) -- CE skip provides complementary temporal overhead reduction
- 2D MRF prior captures clustered sparsity -- could inform skip: if sparsity pattern unchanged, skip CE

---

## 7. Integrated CE and Sensing for NF ELAA (2601.18333)

**Wang, Fang, Li, Ning (UESTC / Stevens Inst.) -- Jan 2026**

| Parameter | Value |
|-----------|-------|
| **Antennas (LoS/THz)** | N = 256 ULA, M = 32 RF chains, d = lambda_c/2 |
| **Antennas (NLoS/mmWave)** | N = 128 ULA, M = 64 RF chains |
| **Frequency (LoS)** | f_c = 100 GHz (THz), B = 0.1 GHz, P = 64 subcarriers |
| **Frequency (NLoS)** | f_c = 30 GHz, B = 0.1 GHz |
| **Channel model** | Near-field spherical wavefront; LoS-dominated (THz) or multi-path NLoS (mmWave); molecular absorption (K(f) = 0.01) |
| **CE methods compared** | SOMP, SIGW (simultaneous iterative gridless weighted), **CPD-based** and **CPD-based (delay-aided)** for LoS; **BTD-based** (block term decomposition) + NLS for NLoS |
| **Key innovation** | Tensor decomposition of received signal Y in C^{PxMxT}; joint CE + user localization from estimated channel parameters |
| **Temporal** | Static (T pilot symbols, but not time-varying channel) |
| **Mobility** | Users in near-field: angle U(-60,60 deg), distance U(20,80) m |
| **SNR range** | 0 to 30 dB |
| **Users** | K = 8 |

**CE-skip relevance:**
- Integrated sensing + CE = localization info available as side information for skip decisions
- If user position known (from sensing), can predict channel stability and decide to skip
- CPD uniqueness with T >= 2 pilots -- minimum pilot requirement sets lower bound on skip interval
- THz scenario (100 GHz, LoS-dominated) = potentially very stable channel, high skip ratio possible

---

## 8. NF Beamfocusing with Modular Linear Arrays (2505.07991)

**Kosasih, Demir, Bjornson (KTH / Nokia) -- May 2025**

| Parameter | Value |
|-----------|-------|
| **Antennas** | MLA: L ULAs, each with N antennas (e.g., L=2, N=25-64; or L=4, N=16-36). Spacing Delta between ULAs (e.g., 5m). Total aperture D_array = 1-2 m |
| **Frequency** | 15 GHz (upper mid-band) |
| **Bandwidth** | 400 MHz |
| **Subcarriers** | Not specified (single-carrier analysis) |
| **Channel model** | Near-field spherical wavefront (Green's function based); Fresnel approximation; free-space propagation. Fraunhofer distance d_FA = 2*D_array^2/lambda |
| **CE methods** | MUSIC per sub-array (1D angular) + least-squares triangulation for localization; then parametric channel reconstruction |
| **Temporal** | Static (T=100 time samples for covariance estimation) |
| **Mobility** | User at angle U(-60,60 deg), distance U(4,40) m |
| **SNR** | P = 20 dBm transmit, sigma^2 = -78 dBm noise (10 dB noise figure); effective SNR depends on distance |

**CE-skip relevance:**
- Modular array = practical 6G deployment (multiple 5G-like panels on rooftop)
- Per-ULA MUSIC + triangulation = very low complexity CE; skip less beneficial when CE is already cheap
- 15 GHz mid-band = moderate coherence time; different skip regime than mmWave/THz
- Beamfocusing enables distance-domain multiplexing -- channel changes even with small radial movement

---

## Summary Table

| Paper | Antennas | Freq (GHz) | BW | Subcarriers | Channel | CE Methods | SNR (dB) | Temporal |
|-------|----------|------------|-----|-------------|---------|------------|----------|---------|
| XLCNet | 256 ULA | 30 | NB | 1 | Hybrid NF+FF | LS, LMMSE, HOMP, CNN | -10~20 | Static |
| Wideband Unrolling | 512 ULA / 2048 UPA | 100 | 10 GHz | 256 | NF+SnS+beam split | LMMSE, OMP, ISTA-Net+, PGD-Net | -5~10 | Static |
| LLM4XCE | 256 ULA | 30 | NB | 1 | Hybrid NF+FF | LS, LMMSE, HOMP, GPT-2 | -5~20 | Static |
| NF Survey | various | 30-100 | various | various | NF spherical | taxonomy only | - | - |
| Decentralized DBP | 128 ULA (4 sub) | 30 | 1.6 GHz | 16 | NF+dual-WB+SnS | SBL variants, SBL-GNNs | 0~10 | Static |
| Sub-Array Pilot | 1024 (128x8 sub) | 3.5 | 15 kHz/sc | 1024 | COST2100, SnS | LMMSE, VAMP-BG, OMP, Turbo-MRF | 0~20 | Static |
| ISAC NF ELAA | 256/128 ULA | 100/30 | 0.1 GHz | 64 | NF spherical, LoS/NLoS | CPD, BTD, SOMP, SIGW | 0~30 | Static |
| Modular Array | MLA (L x N) | 15 | 400 MHz | 1 | NF Fresnel | MUSIC + triangulation | ~10 eff. | Static |

---

## Key Observations for CE-Skip Paper

### 1. All papers assume static channels
Every single paper uses static channel models. **None address temporal channel variation or CE scheduling.** This confirms a clear research gap for our CE-skip work.

### 2. CE computational cost varies by 4 orders of magnitude
- Cheapest: C-XLCNet 0.078ms (29K params), MUSIC per-subarray
- Mid: SBL-GNNs ~0.004s per subarray, PGD-Net 3.79 GFLOPs
- Heaviest: LLM4XCE 126M params, centralized SBL 2.379s

Skip scheduling benefit scales with CE cost. Our paper should evaluate skip savings across this cost spectrum.

### 3. Dominant antenna configs for simulation
- **256 ULA at 30 GHz** (most common, 3 papers)
- **512 ULA at 100 GHz** (wideband)
- **128 ULA at 30 GHz** (decentralized)
- **1024 sub-array at 3.5 GHz** (sub-6G ELAA)
- **2048 UPA at 100 GHz** (largest)

### 4. CE methods to include in our comparison
From these papers, the relevant CE methods for our CE-skip experiments:
- **LS** (universal baseline, cheapest)
- **LMMSE** (standard, moderate cost)
- **DL-based** (XLCNet/PGD-Net class, high accuracy but costly)

This aligns with our CE-skip plan of 3 CE methods (LS/LMMSE/DL-CE).

### 5. Near-field spatial non-stationarity
Multiple papers (Wideband Unrolling, Decentralized, Sub-Array) model spatial non-stationarity via visibility regions. This means different antenna sub-arrays see different channel components. Implication for CE-skip: **per-subarray skip decisions** could be more efficient than array-wide decisions.

### 6. Sensing-aided CE as skip enabler
Paper 7 (ISAC ELAA) shows channel parameters can be extracted for localization. If user position is tracked via sensing, this provides direct input for skip decisions without running full CE.
