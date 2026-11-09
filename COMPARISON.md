# Modular Capabilities vs. Today's Models — Comparison & Use Cases

## 1. How production models handle "capability" today

| Aspect | Monolithic MoE (DeepSeek-V3, Mixtral, Qwen3) | LoRA/adapter hubs | RLHF/confidence abstention | **This design** |
|---|---|---|---|---|
| What an expert is | Anonymous content-specialization discovered by routing | Task adapter, selected by prompt embedding similarity | — (no module concept) | **Named capability slot** (math/code/sql), trained & shipped as a file |
| Capability ground truth | Not verifiable; only behavioral probes | Router guess, unverified | Statistical estimate | **Physically checkable**: module attached or not |
| Adding a capability | Retraining/continued pretrain (forgetting risk) | Fine-tune adapter + retrain router | Align again | Train side-expert (6.3M params); foundation frozen |
| Removing a capability | Practically impossible without retraining | Delete adapter; router may misfire | Unlearn? | Detach file; abstention head then *proves* absence to itself |
| Refusal behavior | Hallucinate when knowledge absent (router always fires) | Answers with wrong adapter | "Probably unsure" (threshold on logits/verbalized confidence) | **Consensus-detected refusal**: 8 per-layer judges vote, `<abs>` holds p≈1.0 of softmax at the answer-decision slot |
| Transfer to unseen prompts | n/a | n/a | Degrades under distribution shift | **40/40** held-out on first-seen prompts (gate reads assembly, not content) |
| Auditability | Introspect nothing | Registry of adapters (trust-based) | None | **Loss-matrix receipts + bit-exact swap reproduction** (11.7497 / 5.4408 measured) |

Key mechanistic difference: market models *estimate* competence from the same weights that produce answers. Here competence is a **property of the machine's current configuration**, read by a separate detector stack — a detection task with deterministic ground truth, not an estimation task.

## 2. Distributed training possibilities this unlocks

Because the foundation is frozen and each domain expert is a small independently-trained weight file:

1. **Federated domain training** — a hospital/bank trains a `medical`/`finance` expert on premises; raw data never leaves; the hub ships the foundation. The module can be *verified by swap-reproduce* before attachment (a checksum-able capability claim).
2. **Parallel capability development** — N experts train concurrently on N workers (they never touch shared weights — no gradient races, no catastrophic forgetting by construction). Wall-clock per capability ≈ single-module time.
3. **Capability marketplace / assembly-time models** — users compose a model = foundation + chosen experts + abstention head retrained for that library. Unattached domains *cannot* answer, even adversarially prompted, because the gate reads slots, not text.
4. **Governance & runtime kill-switch** — "what this model can do" becomes a *bill of materials*. Regulators inspect slot state instead of behavior probing; removing a capability is a file operation, and the model's own abstention layer attests the gap.
5. **Edge economics** — ship 750 MB foundation once, stream only the 25 MB experts a tenant needs; per-request cost = active-module set.
6. **Continual learning without forgetting** — new domain = new slot; the frozen foundation and existing modules are bit-exact unchanged (measured: gate-zero purity restores base loss to 4 decimal places).

## 3. Honest limits (as of this experiment)

- Proven at **197M params, 3 domain experts, single GPU**; scaling claims are structural, not measured.
- Expert injection *does* re-route the frozen base slightly (final-layer routing entropy shifts ±0.4 nats) — capability is ~95% in the module, not 100%.
- The abstention head must be retrained/calibrated when the library changes (τ moves per library state; v11 τ=0.49).
- Answer quality limited by the TinyStories foundation — the *gate* is the contribution.
- Federated/marketplace points (2.x) are design consequences demonstrated at toy scale, not deployed systems.
