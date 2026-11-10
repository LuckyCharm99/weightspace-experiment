# Weight-Space Experiments on a Mini Qwen3.8-Flash-Next

A miniature (~197M-parameter) reproduction of Alibaba's Qwen3.8-Flash-Next blueprint
(3:1 Gated-DeltaNet/Attention hybrid, 4-branch gated residual streams, fine-grained
MoE with shared expert), trained from scratch on real Hugging Face data — used as a
testbed for **weight-space surgery**: expert profiling, pruning, injection, task
vectors, swappable domain-capability modules, and a trained **abstention layer that
gates on whether the competent expert is physically attached**.

Existing abstention work estimates *what a model knows* (confidence, refusal tuning).
This experiment implements something different: *which model is currently assembled*.
Remove an expert module and the model declines to answer; attach it and the same
frozen weights answer. The refusal decision is a boolean of two things: a trained
linear circuit on the final hidden state (*needed*) and the bit-exact assembly
manifest (*present*) — deterministic, auditable, and now scaled to **ten experts**
([CHANGELOG](CHANGELOG.md)).

![architecture vs market MoE](results/figures/architecture_vs_market_moe.png)

## Results (all measured, held-out splits)

### 1. Foundation training
`roneneldan/TinyStories`, 2000 steps, bf16, Radeon 8060S (ROCm): loss **11.93 → 3.28**,
~587 tok/s, 196.5M params, Qwen2.5-Coder tokenizer (151,665).

### 2. Weight-space surgery (post-training, in-place)
- **Profiling:** layer-0 routing entropy 2.6128 nats (max 2.7726) → specialists exist.
- **Pruning:** least-used expert (11, 285 allocations) removed structurally, router row sliced.
- **Injection:** cloned + perturbed expert appended, router row zero-init.
- **Task vectors:** τ = W_busiest − W_leastused applied λ=0.3 to all 3 SwiGLU projections.

### 3. Swappable capability modules (frozen base)
One side-expert per MoE layer (8 layers, ~6.3M trainable, bounded gate), domain-trained:

| module ↓ / eval → | story | math | code | sql |
|---|---|---|---|---|
| none | 3.65 | 11.15 | 9.54 | 11.75 |
| math | 5.54 | **7.17** | 8.28 | — |
| code | 4.83 | 9.24 | **6.68** | — |
| sql | 6.81 | — | — | **5.44** |

Batch-1 (2026-11-10) adds **chess** 11.17→**3.78**, **chem** 12.51→**7.30**,
**regex** 12.54→**1.74** — same protocol, same bit-exact purity/swap proofs
(see *Batch-1: six experts* below).

**Swap proofs (zero training, pure weight file I/O):**
- restoring the math bundle reproduced **7.1725** bit-exactly;
- the sql module's gate, re-zeroed, reproduced base loss **11.7497** exactly (nothing
  leaks outside the module) and snapshot-restore gave **5.4408** exactly.
- math vs code expert cosine distance 0.61 — genuinely different weight regions.

![module loss matrix](results/figures/module_loss_matrix.png)

### 4. Capability-gated abstention — 3 experts, grid **39/40** *(soft-gate era, superseded by v13/v15 below)*
A trained head per MoE layer reads hidden state **and the attached module's activity
vector**; averaged score `z` vs held-out-calibrated threshold τ ⇒ `⟦ABSTAINED⟧`
suppresses generation at the answer-decision point.

Training discipline (what made it work after many failed attempts):
decision spans **byte-identical across worlds** — only the module attachment flips
the label; row-aligned fresh-start windows; explicit label-0 supervision at
marker-end answer points; both decision classes get equal loss mass; **train
displays have the eval prompts' shape** (bare `def` stubs, SQL questions).

| module ↓ / eval → | math | code | sql | story |
|---|---|---|---|---|
| none | ABST (+8.4) | ABST (+8.2) | ABST (+8.3) | answers (−7.4) |
| math | answers (−0.2)¹ | ABST (+7.2) | ABST (+6.8) | answers (−6.7) |
| code | ABST (+6.0) | answers (−3.6) | ABST (+5.8) | answers (−7.5) |
| sql | ABST (+6.3) | ABST (+6.5) | answers (−1.6) | answers (−5.3) |

¹ one of three math prompts (z=4.66 > τ=0.49) still abstains with its expert online —
the miss is a single near-τ cell, not a mechanism failure. τ = 0.49; held-out
separation abstain-p05 +5.04 vs answer-p95 −4.06.

![abstention grid](results/figures/abstention_grid.png)
![calibration](results/figures/calibration.png)

**Full instrumented evaluation** (`eval_final.py`, `results/final_eval_v11.json`):
**40/40 on fresh held-out prompts** — a forward pass with zero training records per
layer the hidden RMS, router entropy, side-module gate, abstention score, and the
vocabulary softmax at the decision slot. When abstention fires, `<abs>` holds
**p≈1.0000** (the 151,664 other rows are nullified); answering states place mass on
real content (`sql → 'SELECT'`, `code → '   '`, `story → 'The'`).

Generations (`abstain_v11_results.json`): every no-module/wrong-module prompt outputs
`⟦ABSTAINED⟧`; each prompt with its expert attached produces output text. Answer
*quality* is weak — the base was trained on TinyStories — the contribution is the
gate, honestly reported.

## Repository layout

```
CHANGELOG.md                # measured milestone history (v3 -> v15)
src/
  train_and_experiment.py   # stage 1+2: foundation training + profiling/prune/inject/task-vector
  train_module_experts.py   # stage 3: frozen base + swappable math/code modules
  train_sql_module.py       # stage 3b: third domain (nl -> SQL)
train_batch1_modules.py   # stage 3c: chess, chem, regex experts (batch-1)
train_batch2b.py          # stage 3d: med, units, music experts (batch-2, lang skipped)
train_logics.py           # stage 3e: 10th expert (propositional logics)
abstention_v16.py         # 9-expert hard gate + 280-cell grid + 29-probe stress (M=1.23)
abstention_v17.py         # 10-expert hard gate + 341-cell grid + 30-probe stress (shipped, M=1.09)
abstention_v15.py         # 6-expert hard gate + 133-cell grid + 20-prompt stress (shipped, M=2)
  abstention_v5.py          # stage 4 core: vocab, two-world machinery, head, trainer
  abstention_v9.py v9b.py v10.py v11.py v12.py  # fixes -> 17/17 -> 39/40 -> uniformity ablation
abstention_v13.py             # HARD GATE: deterministic+auditable abstention, 40/40 (shipped, +M4 margin)
eval_final.py               # instrumented full eval (40/40 held-out, layer traces)
test_v13_confusing.py         # 12 confusing prompts x 4 assemblies stress suite
test_v13_mitigated.py         # margin sweep -> shipped M=4 gate
abstention_v14.py             # router-free comparator attempt (NEGATIVE result)
  make_figures.py           # regenerate README figures
checkpoints/                # module bundles + abstention bundles (base ckpt retrainable, 750MB)
results/                    # raw JSONs measured on held-out splits + figures
```

## Reproduce

```bash
python -m venv venv && venv/bin/pip install -r requirements.txt
# ROCm (Strix Halo gfx1151): pip install torch --index-url https://rocm.nightlies.amd.com/v2/gfx1151/
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
venv/bin/python -u src/train_and_experiment.py   # ~4 h GPU, writes checkpoints/ckpt_final.pt
venv/bin/python -u src/train_module_experts.py   # ~25 min
venv/bin/python -u src/train_sql_module.py       # ~15 min
venv/bin/python -u src/train_batch1_modules.py   # ~2.5 h: chess, chem, regex experts
venv/bin/python -u src/abstention_v15.py         # six-expert gate + 133-cell grid + stress (~45 min)
venv/bin/python -u src/abstention_v11.py         # ~45 min (resumes shipped bundles)
venv/bin/python -u src/make_figures.py
```

Skip stages by using the shipped `checkpoints/` (all bundles verified by swap-reproduce tests).

## Limitations (deliberately honest)
- 197M miniature; mechanism-level evidence, not a benchmark claim.
- Answer text quality poor (TinyStories foundation under math/SQL/SQL domains).
- Abstention is an inference gate (`z > τ` suppresses generation), not `<abs>`
  out-logiting the 151K softmax.
- One grid cell misses at τ-margin; τ is calibrated per domain-library state.
- Datasets: TinyStories (CDLA-ish per HF page), hendrycks_math, code_contests,
  sql-create-context — check each upstream license before commercial reuse.

## Adversarial stress test and the shipped configuration: v13 + margin 4
12 deliberately confusing prompts (mixed shapes, misspellings, nonsense, bare
`Solution:`, out-of-domain facts) × 4 assemblies. Empty assembly: **12/12 abstain**
at every setting — detach remains the perfect mitigation. With an expert attached,
the circuit is *confidently wrong* on OOD text (a history question reads as
`math`, bare `Solution:` claims everything): 16 risky commitments. Sweeping a
top1−top2 circuit-margin gate (no retraining — adversarial labels would memorize
the annotation policy) shows the risky and correct-margin distributions
**overlap**: margin 4 removes 3 commitments free (control grid stays 40/40);
margin ≥6 starts eating correct cells (39/40, 37/40). **Shipped: margin M = 4** —
all results 2× bit-identical.

## v14 negative result: router-free expert self-verification fails
Replacing the circuit entirely with comparator-style verification (attached expert
must CLAIM the input: domain-format regex on its own greedy continuation +
repetition-discounted confidence threshold calibrated on train splits only).
Result: **grid 18/40, risky 24/36, empty-assembly safety 7/12** — worse than v13 on
every axis. Two lessons:
1. *Self-confidence is not verification.* A language model is equally fluent on
   garbage ("nonsense tokens" → a fluent fairy-tale continuation at base-conf −1.1).
   Comparator verification needs a **reference to reproduce**; free-form prompts
   have none, so claim-vs-hallucination collapses into fluency.
2. *The soft layer is not decoration.* Zeroing v11's abstention side-experts and
   `<abs>` embedding row made the code module emit prose (format claims 0/40):
   capability modules were trained inside the full stack and their behavior
   partially depends on it. "Abstention with zero learned gate" is empirically
   false at this scale.

## v13: deterministic, auditable abstention
The soft per-layer score + calibrated τ is replaced by two separate halves:
**needed** = argmax of a 512×4 linear circuit on the foundation's final hidden
state at the decision slot (trained across all 4 module worlds; 99.5% held-out
accuracy), and **present** = the bit-exact assembly manifest (detached gates are
*exactly* zero through the tanh envelope — machine-verified). `ABSTAIN ⟺ needed
≠ general ∧ ¬present[needed]`: a 16-row boolean truth table, printed in
`results/abstain_v13_results.json`, with the circuit hashed (sha256) and the
full grid replayed 3× **bit-identical**. Result: **40/40 grid** (beats v11's
39/40 — near-τ borderline cells cannot exist without a τ). Known cost: on
prose-shaped OOD probes the circuit can mislabel the needed domain (SQL prose →
code); with the true expert absent the verdict is still correct, but a
misrouted domain with the wrong expert attached can answer with the wrong
capability — the soft gate could hedge there, the hard gate cannot.

## Batch-1: six experts + v15 six-expert stress test

Three more domains, same protocol (frozen base, ~6.3M side-expert bundle per
domain, 600 steps, deterministic streaming slices disjoint from eval):

| expert | data | loss off → on | purity | swap-restore |
|---|---|---|---|---|
| chess | Lichess/chess-puzzles (FEN → best move) | 11.17 → **3.78** | bit-exact ✓ | bit-exact ✓ |
| chem | maykcaldas/smiles-transformers (formula → SMILES) | 12.51 → **7.30** | bit-exact ✓ | bit-exact ✓ |
| regex | qgallouedec/guess-the-regex (description → pattern) | 12.54 → **1.74** | bit-exact ✓ | bit-exact ✓ |

`abstention_v15.py` rebuilds the shipped hard gate for all 6 experts + story:
the 512×7 circuit is retrained from scratch on decision-slot hidden states
collected across all 7 module worlds (held-out **0.9845**, sha256 `4f5d4e24…`),
and the commit margin was **re-calibrated on held-out data only** (never on the
eval grid): true-positive domain margins have p1 = 1.96, so **M = 2.0** keeps
98.9 % of correct commits and blocks ~half of wrong-domain routings. The old
M = 4 was a 4-class artifact and is wrong for 7 classes.

Results — grid **126/133** (7 assemblies × 19 prompts + 2 story cells × 7),
3× bit-identical, empty assembly abstains on **20/20** confusing probes
(including 8 new cross-domain traps: FEN-shaped regex asks, chemistry in
fairy-tale prose, bare patterns). All six experts now correctly answer their
own domain. Honest residuals: (1) the three hand-written SQL eval prompts read
as `code` (margin ≈ 1) and abstain under the sql assembly — a **template-drift**
gap: pool-shaped held-out SQL routes 100 % correctly, so this is prompt-shape
generalization of a linear circuit at 7-way arity, not a mechanism failure, and
the failure direction is over-abstention (safe); (2) 5 risky commitments —
loosening M from 4 → 2 trades 1 risky (3 experts) for 5 (6 experts) alongside
+2 grid cells and full chem recognition. Wrong-domain margins on held-out reach
33, so no single margin separates the two populations at this scale; a richer
circuit (nonlinear, or h from multiple layers) is the open lever.

The 6-expert bundle `module_bundles_lib10.pt` exceeds GitHub's 100 MB file
limit, so the repo ships `checkpoints/lib10_{chess,chem,regex}.pt` (25 MB each)
plus `module_bundles_merged.pt` (math, code, sql); the library is their union.

## Batch-2 + logics: ten experts + v17 (current shipped configuration)

Three more domains (med = MedMCQA single-answer, units = English unit
arithmetic, music = ABC-notation theory) on the same protocol:

| expert | loss off → on | purity | swap-restore |
|---|---|---|---|
| med | 11.45 → **6.07** | bit-exact ✓ | bit-exact ✓ |
| units | 9.83 → **3.72** | bit-exact ✓ | bit-exact ✓ |
| music | 11.54 → **5.46** | bit-exact ✓ | bit-exact ✓ |

A fourth candidate (en→de translation) trained cleanly (10.60→6.02, proofs
exact) but was **dropped by operator decision** before evaluation — the
library is nine experts, and the skipped bundle simply isn't loaded.

`abstention_v16.py`: 512×10 circuit retrained across all 10 module worlds
(held-out **0.9722**, sha `9135580a…`); commit margin re-derived from held-out
true-positive margins (p1 = **1.23**), not tuned on the grid. Results: grid
**273/280** at M = 1.23 (1 risky commitment; M = 4 buys zero-risk on the grid
at −2 cells), **29/29 abstain** on the empty assembly across the whole legacy
+ new confusing-probe corpus, determinism 3× bit-identical. On 81 adversarial
trap cells × expert assemblies, 8 commit at shipped M — and the pattern is
instructive: **MCQ-format traps are claimed by the med expert regardless of
content** (a music-theory MCQ reads as medicine). Format dominance over
semantics is the documented failure mode of a linear circuit at 10-way arity;
raising M to 3 costs one grid cell and removes one trap.

**10th expert (same day):** propositional logics — NL argument → numbered
symbolic proof (`ergotts/propositional-logic`). 9.63 → **3.94**, the strongest
off→on delta in the program and the lowest story cost of all ten experts;
purity + swap bit-exact. The gate was rebuilt at 11 classes (`abstention_v17.py`,
held-out 0.9530, M = 1.09 from the same held-out p1 calibration): grid
**331/341**, risky 2, **30/30** empty-assembly safety, 3× bit-identical. The
circuit's held-out accuracy declines smoothly with arity (0.995 → 0.984 →
0.972 → 0.953 for 4 → 7 → 10 → 11 classes) while the end-to-end gate stays
≈ 97 % — the boolean composition with the exact presence manifest absorbs
most of the classifier's loss.

## Ablation: layer-uniform "confident" abstention (v12, v12b)

Supervising each layer's verdict separately with a spread penalty (v12: uncapped —
collapsed abstain/answer separation 11→0.55 nats, grid 35/40; v12b: hinge
`0.1·max(0,spread−3)` — spread settled at 2.4, separation 4.2, but matching-module
cells became borderline votes and the grid read 37/40 vs v11's 39/40).
Finding: the network hoards the verdict in whichever layer maximizes margin (L6,
±20 nats); forcing per-layer consensus is bought with margin — agreement and
accuracy are not free to trade. **v11 is the shipped artifact.**

## License
MIT
