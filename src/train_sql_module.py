"""Train the THIRD domain module: natural-language -> SQL on a frozen base.

Mirrors the v3 math/code protocol exactly: foundation frozen, side-expert
bundle (one per MoE layer, unit-RMS + tanh gate) trained on the domain stream,
then evaluated as an on/off loss matrix. train split = first 74,577 rows,
held-out eval = last 4,000 rows (dataset order is fixed, so the slice is
deterministic and disjoint).

Saves checkpoints/module_bundles_merged.pt = {'math','code','sql'} so the
abstention stage can address all three experts uniformly.
"""
import sys, copy, json
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).parent))
import train_module_experts as v3mod
from datasets import load_dataset

DEVICE = v3mod.DEVICE
WS = Path(__file__).parent

_ds_done = False


def sql_rows(split):
    # single 'train' config: carve deterministic slices
    ds = load_dataset("b-mc2/sql-create-context", split="train", streaming=True)
    take, skip_until = (74577, 0) if split == "train" else (None, 74577)
    n = 0
    for r in ds:
        if n < skip_until:
            n += 1
            continue
        if take is not None and n >= take:
            return
        n += 1
        yield r


def sql_stream(split="train"):
    for r in sql_rows(split):
        q = (r.get("question") or "").strip()
        ctx = (r.get("context") or "").strip()
        a = (r.get("answer") or "").strip()
        if not (q and ctx and a):
            continue
        yield f"Question: {q}\nDatabase: {ctx}\nSQL:\n{a}\n"


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B", trust_remote_code=True)
    tok.pad_token = tok.eos_token
    model, mcfg = v3mod.load_base()
    sides = v3mod.attach_modules(model, mcfg)
    v3mod.freeze_base(sides, model)

    print("Building eval batches (held-out tail)...")
    eval_sql = v3mod.make_batches(v3mod.Packer(tok, lambda: sql_stream("eval")), 24)
    eval_story = v3mod.make_batches(v3mod.Packer(tok, lambda: v3mod.story_stream(60000)), 24)

    with torch.no_grad():
        for s in sides:
            s.gate_raw.fill_(0.0)
    base = {"story": v3mod.eval_loss_on(sides, model, mcfg, eval_story),
            "sql": v3mod.eval_loss_on(sides, model, mcfg, eval_sql)}
    print(f"SQL BASE (module off): {base}")

    v3mod.reinit_modules(sides, model)
    v3mod.STEPS_PER_DOMAIN = 600
    v3mod.train_modules(sides, model, mcfg, v3mod.Packer(tok, lambda: sql_stream("train")), "sql")
    sql_snap = v3mod.snapshot(sides)

    # trained weights + trained gate are in place -> ON measurement
    on_loss = {"story": v3mod.eval_loss_on(sides, model, mcfg, eval_story),
               "sql": v3mod.eval_loss_on(sides, model, mcfg, eval_sql)}
    print(f"SQL MODULE ON: {on_loss}")
    with torch.no_grad():
        for s in sides:
            s.gate_raw.fill_(0.0)
    off_after = {"sql": v3mod.eval_loss_on(sides, model, mcfg, eval_sql)}
    print(f"SQL gate re-zeroed (purity check, module off): {off_after}")
    # swap proof: restoring the snapshot with zero training must reproduce ON
    v3mod.restore(sides, sql_snap)
    swap = {"sql": v3mod.eval_loss_on(sides, model, mcfg, eval_sql)}
    print(f"SQL SWAP-RESTORE: {swap}")

    v3 = torch.load(WS / "checkpoints/module_bundles_v3.pt", map_location=DEVICE,
                    weights_only=False)
    out = {"math": v3["math"], "code": v3["code"], "sql": sql_snap}
    torch.save(out, WS / "checkpoints/module_bundles_merged.pt")
    (WS / "sql_module_results.json").write_text(json.dumps(
        {"base_off": base, "after": on_loss}, indent=2))
    print("SQL MODULE COMPLETE")


if __name__ == "__main__":
    main()
