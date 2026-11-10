"""BATCH-1 module expansion: chess, chemistry (formula->SMILES), regex.

Mirrors the SQL protocol per domain: deterministic streaming slice of the train
split (first N train / next M eval rows, disjoint), frozen base, side-expert
bundle trained 600 steps, then the proof ladder:
  base-off -> on -> gate-rezero purity (equals base-off) -> swap-restore
  (equals on).
Saves checkpoints/module_bundles_lib10.pt = {math,code,sql,chess,chem,regex}
and batch1_module_results.json.
"""
import sys, json
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).parent))
import train_module_experts as v3mod
from datasets import load_dataset

DEVICE = v3mod.DEVICE
WS = Path(__file__).resolve().parent.parent


def fmt_chess(r):
    fen, mv = (r.get("FEN") or "").strip(), (r.get("Moves") or "").strip()
    if not fen or not mv or fen.count("/") != 7:
        return None
    return f"FEN: {fen}\nBest move: {mv.split()[0]}\n"


def fmt_chem(r):
    f = (r.get("formula") or "").strip()
    s = (r.get("text") or "").strip()
    if not f or not s or len(s) > 160:
        return None
    return f"Formula: {f}\nSMILES:\n{s}\n"


def fmt_regex(r):
    d = (r.get("public_description") or "").strip()
    x = (r.get("regex") or "").strip()
    if not d or not x:
        return None
    return f"Description: {d}\nRegex:\n{x}\n"


DOMS = {
    # name, n_train, n_eval, formatter, native_eval_key
    "chess": ("Lichess/chess-puzzles", 60000, 2000, fmt_chess, None),
    "chem":  ("maykcaldas/smiles-transformers", 60000, 2000, fmt_chem, None),
    "regex": ("qgallouedec/guess-the-regex", None, None, fmt_regex, "test"),
}


def texts(dom, split):
    name, ntr, nev, fmt, native = DOMS[dom]
    ds = load_dataset(name, streaming=True)
    if native and split == "eval" and native in ds:
        src, n = ds[native], None
    else:
        src, n = ds["train"], (ntr if split == "eval" else None)
    seen = taken = 0
    for r in src:
        seen += 1
        if split == "train" and ntr is not None and seen > ntr and native is None:
            return
        if split == "eval" and native is None:
            if seen <= ntr:
                continue
            if taken >= nev:
                return
        t = fmt(r)
        if t:
            taken += 1
            yield t


def run_domain(tok, sides, model, mcfg, dom, eval_story):
    eval_dom = v3mod.make_batches(v3mod.Packer(tok, lambda: texts(dom, "eval")), 24)
    with torch.no_grad():
        for s in sides:
            s.gate_raw.fill_(0.0)
    base = {"story": v3mod.eval_loss_on(sides, model, mcfg, eval_story),
            dom: v3mod.eval_loss_on(sides, model, mcfg, eval_dom)}
    print(f"{dom.upper()} BASE (module off): {base}", flush=True)

    v3mod.reinit_modules(sides, model)
    v3mod.STEPS_PER_DOMAIN = 600
    v3mod.train_modules(sides, model, mcfg, v3mod.Packer(tok, lambda: texts(dom, "train")), dom)
    snap = v3mod.snapshot(sides)
    on = {"story": v3mod.eval_loss_on(sides, model, mcfg, eval_story),
          dom: v3mod.eval_loss_on(sides, model, mcfg, eval_dom)}
    print(f"{dom.upper()} MODULE ON: {on}", flush=True)

    with torch.no_grad():
        for s in sides:
            s.gate_raw.fill_(0.0)
    off2 = {dom: v3mod.eval_loss_on(sides, model, mcfg, eval_dom)}
    ok_pure = abs(off2[dom] - base[dom]) < 1e-3
    v3mod.restore(sides, snap)
    swap = {dom: v3mod.eval_loss_on(sides, model, mcfg, eval_dom)}
    ok_swap = abs(swap[dom] - on[dom]) < 1e-3
    print(f"{dom.upper()} purity {ok_pure} ({off2[dom]:.4f} vs {base[dom]:.4f}) | "
          f"swap {ok_swap} ({swap[dom]:.4f} vs {on[dom]:.4f})", flush=True)
    with torch.no_grad():
        for s in sides:
            s.gate_raw.fill_(0.0)
    return {"base_off": base, "on": on, "purity": off2, "swap": swap,
            "purity_ok": ok_pure, "swap_ok": ok_swap}, snap


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B", trust_remote_code=True)
    tok.pad_token = tok.eos_token
    model, mcfg = v3mod.load_base()
    sides = v3mod.attach_modules(model, mcfg)
    v3mod.freeze_base(sides, model)
    eval_story = v3mod.make_batches(v3mod.Packer(tok, lambda: v3mod.story_stream(60000)), 24)

    results, snaps = {}, {}
    for dom in ["chess", "chem", "regex"]:
        res, snap = run_domain(tok, sides, model, mcfg, dom, eval_story)
        results[dom], snaps[dom] = res, snap

    lib = torch.load(WS / "checkpoints/module_bundles_merged.pt", map_location=DEVICE,
                     weights_only=False)
    lib.update(snaps)
    torch.save(lib, WS / "checkpoints/module_bundles_lib10.pt")
    (WS / "results/batch1_module_results.json").write_text(json.dumps(results, indent=2))
    print("BATCH1 MODULES COMPLETE:",
          {d: f"{r['base_off'][d]:.2f}->{r['on'][d]:.2f}" for d, r in results.items()})


if __name__ == "__main__":
    main()
