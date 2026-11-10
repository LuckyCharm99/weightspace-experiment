# Changelog

All milestones below were **measured**, not estimated; raw JSONs live in
`results/`. Versions refer to the experiment ladder in `src/`.

## 2026-11-10 — v15: six experts + shipped deterministic gate (current)

**New capability experts (batch-1), same frozen base, same proof protocol:**

| expert | dataset | loss off → on | gate-rezero purity | swap-restore |
|---|---|---|---|---|
| chess | Lichess/chess-puzzles | 11.1713 → 3.7844 | bit-exact | bit-exact |
| chem | maykcaldas/smiles-transformers | 12.5142 → 7.3020 | bit-exact | bit-exact |
| regex | qgallouedec/guess-the-regex | 12.5395 → 1.7448 | bit-exact | bit-exact |

Together with the earlier math (11.15 → 7.17), code (9.54 → 6.68) and sql
(11.75 → 5.44) modules, the library is now **6 named, file-addressable
experts**, each independently attachable/detachable with zero training.

**Gate architecture change (v13 → v15, shipped):**
- circuit retrained from scratch for 7 classes: `nn.Linear(512, 7)` on the
  final-norm hidden state at the answer-decision slot, collected across all
  7 module worlds; held-out accuracy **0.9845**, sha256 `4f5d4e24ebb7746c`.
- commit margin **re-calibrated on held-out data only** (never on the eval
  grid): true-positive domain margins p1 = 1.96 → **M = 2.0**. The previous
  M = 4 was a 4-class artifact; at 7-way arity it silently blocked correct
  commits (chem never answered).
- decision rule unchanged: `ABSTAIN ⟺ needed ≠ general ∧ ¬present(needed)`
  — presence is a bit-exact boolean of the weight assembly (detached tanh
  gates measure exactly 0.0), so refusal is detection, not estimation.

**Measured results:** grid **126/133** (7 assemblies, replayed 3× bit-identical);
confusing-prompt stress **20/20 abstain** on the empty assembly including 8 new
cross-domain traps; 5 risky commitments with experts attached (documented trade
of M 4 → 2). Known residual: hand-written SQL prompt templates drift from the
training-pool shape and read as `code` → safe over-abstention; pool-shaped held-out
SQL routes 100 %.

**Repo packaging:** the merged 6-expert bundle (151 MB) exceeds GitHub's
100 MB file limit — shipped instead as `checkpoints/module_bundles_merged.pt`
(math/code/sql) + `lib10_chess.pt` + `lib10_chem.pt` + `lib10_regex.pt`;
the library is their union (`abstention_v15.py` composes them automatically).

## 2026-11-09 — v13 hard gate + v14 negative result

- **v13** replaced the soft per-layer score + threshold τ with the
  circuit-argmax × presence-boolean hard gate: **40/40** grid (3 experts),
  determinism 3×, auditable 16-row truth table; OOD cost honestly logged
  (prose-shaped prompts can mislabel `needed`).
- Margin sweep on 12 confusing prompts → shipped **v13 + M4** at the time
  (13 risky commitments remained; risky vs correct margin distributions
  overlap).
- **v14** (router-free expert self-verification, no circuit): **failed, 18/40**
  — kept as a negative result. Lessons: self-confidence is fluency, not
  verification; and zeroing the v11 soft layer breaks the code module, so
  capability modules trained in-stack depend on the assembly they live in.

## 2026-11-08/09 — v11 soft gate and ablations

- **v11**: per-layer abstention heads reading hidden state + attached-module
  activity; two-world byte-identical decision spans; **39/40** grid,
  40/40 instrumented held-out (`<abs>` p ≈ 1.0000 when firing), τ = 0.49.
- **v12/v12b** (not shipped): layer-uniform "confident" consensus ablation —
  spread penalty collapses separation (0.55) or, as a hinge, costs 2-4 grid
  cells; verdict is hoarded in the max-margin layer.

## 2026-11-07/08 — foundation + surgery

- Mini Qwen3.8-Flash-Next (~196.5 M): 3:1 GDN/attention hybrid, 4-branch
  gated residuals (HyperConnections-style), fine-grained MoE + shared expert;
  TinyStories 2000 steps, loss **11.93 → 3.28**.
- Weight-space surgery: router-entropy profiling, structural expert pruning,
  clone-injection, task-vector (λ = 0.3) experiments.
- **v3**: first swappable side-expert bundles (math/code), unit-RMS output +
  tanh-bounded gates; swap-restore bit-exactness established (7.1725);
  median-of-demos robustness; expert cosine distance 0.61 (real divergence).
- sql module added (5.44/11.75, both directions bit-exact).

## Architecture evolution (summary)

| era | refusal mechanism | ground truth |
|---|---|---|
| v5–v11 | soft per-layer head + τ | learned score (calibrated) |
| v12/v12b | + uniformity penalty | learned consensus (rejected) |
| v13 | linear circuit argmax × presence | assembly = deterministic half |
| v13+M4 | + top1−top2 margin | held-out calibration (4-class) |
| **v15** | 7-class circuit, M=2.0 held-out p1 | **assembly + margin (shipped)** |
