# Research Idea: Task-Agnostic Site Representation for 6G Channel Estimation

## Core Concept
3-way model decomposition for wireless channel estimation across multiple BS sites:
- **E (Shared Encoder)**: Learns universal wireless physics features, shared via FL/transfer learning
- **θ_task (Shared Task Head)**: Task-specific decoder, shared across sites
- **θ_BS (Site Embedding)**: ~64-dim learnable vector, zero-initialized (like LoRA), stays local per BS

## Key Insight (Killer Contribution)
θ_BS trained on Task A (channel estimation) should be **task-agnostic** and transfer to Task B (beam prediction).
If this works → "site foundation representation" — much stronger than just cold start improvement.

## Framing: Transfer/Meta-Learning (NOT FL during operation)
- **Phase 1 (Pre-train)**: Train E + θ_task on simulated BS₁~₆ (Sionna RT digital twin)
- **Phase 2 (Deploy)**: Deploy to unseen BS₇, only adapt θ_BS (few-shot)
- **Motivation**: "Can't simulate every BS location, need fast adaptation to unseen sites"
- FL motivation is WEAK ("if you have data, train independently") — transfer learning is the right framing

## Novelty vs. Prior Work
- FedPer (2-way): shared encoder + local head → we add θ_BS as 3rd component
- FedRep, MAML, Per-FedAvg: exist individually, but 3-way physical decomposition for wireless is novel
- "Overlapping observation": multiple BSs observe same UEs — unique to wireless, not in standard FL
- Individual components exist; **combination with physical meaning is the novelty**

## Architecture Details
- Input: H_LS (noisy LS estimate) → (B, 2, 8, 1024) — real/imag, 2×4 antenna pairs, 1024 subcarriers
- Encoder: Conv2D + ResBlocks (spatial dims preserved)
- Site injection variants: **FiLM** (default), concat, add — experiment to find best
- Task head: ResBlocks + Conv2D output → residual correction
- Output: H_est = H_LS + residual (ReEsNet-style)
- θ_BS integration order (before/after encoder) needs experimentation

## Channel Estimation Task
- Use established methods: **ReEsNet** (Li et al. 2020, ~200 citations) — NOT novel architecture
- Task method should NOT be novel to avoid diluting paper's actual contribution
- Input: noisy LS estimate, Output: denoised channel, Metric: NMSE (dB)

## Cold Start Value
- Advantage is in **adaptation speed**, not eternal superiority
- With enough data, independent training catches up → contribution is the cold-start phase
- θ_BS zero-initialized → adapts from this starting point

## User Communication Style
- "비판적으로" — push back hard, thesis-antithesis-synthesis
- Don't just agree — find weaknesses and counter-argue
- "collaborative BS가 아니어도 됨" — relaxed project scope
