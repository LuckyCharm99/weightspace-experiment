"""BATCH-2 module expansion: med (MedMCQA), lang (en->de), units, music (ABC).

Same protocol as batch-1: frozen base, 600 steps, gate-rezero purity +
swap-restore proofs; per-domain streaming slices disjoint from eval pools.
lang uses the dataset's own dev split for eval (disjoint). Outputs:
checkpoints/lib2_{med,lang,units,music}.pt (individual 6.3M bundles) +
module_bundles_final10.pt (merged, local) + batch2_module_results.json.
"""
import sys, json
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).parent))
import train_module_experts as v3mod
from datasets import load_dataset

DEVICE = v3mod.DEVICE
WS = Path(__file__).resolve().parent.parent
LETTERS = "ABCD"


def fmt_med(r):
    if r.get("choice_type") != "single":
        return None
    try:
        k = int(r.get("cop", "-1"))
    except (TypeError, ValueError):
        return None
    q = (r.get("question") or "").strip()[:600]
    opts = [str(r.get("op" + c) or "").strip()[:200] for c in LETTERS.lower()]
    if not q or k not in (0, 1, 2, 3) or not all(opts):
        return None
    body = "\n".join(f"{L}) {o}" for L, o in zip(LETTERS, opts))
    return f"Question: {q}\n{body}\nAnswer:\n{LETTERS[k]}\n"


def fmt_lang(r):
    en = (r.get("english") or "").strip()
    de = (r.get("non_english") or "").strip()
    if not en or not de or len(en) > 200 or len(de) > 220:
        return None
    return f"English: {en}\nGerman:\n{de}\n"


def fmt_units(r):
    if r.get("lang") != "en":
        return None
    q = (r.get("question") or "").strip()
    a = (r.get("answer_text") or "").strip()
    if not q or not a or len(q) > 200:
        return None
    return f"{q}\nAnswer:\n{a}\n"


def fmt_music(r):
    ins = (r.get("instruction") or "").replace("Human:", "").strip()
    inp = (r.get("input") or "").strip()[:200]
    out = (r.get("output") or "").replace("Assistant:", "").strip()[:500]
    if not out.startswith("X:") and "Music" not in ins:
        return None
    if not ins or not out:
        return None
    return f"{ins}\nInput: {inp}\nScore:\n{out}\n"


# name: (hf_id, config, formatter, n_train_valid, n_eval_valid, native_eval_split)
DOMS = {
    "med":    ("openlifescienceai/medmcqa", None, fmt_med, 20000, 2000, None),
    "lang":   ("sentence-transformers/parallel-sentences-tatoeba", "en-de", fmt_lang, None, None, "dev"),
    "units":  ("roby1222/Unit-Conversion", "qa", fmt_units, 1800, 400, None),
    "music":  ("Seeker38/music_abc_notation_with_music_theory", None, fmt_music, 8000, 1000, None),
}


def texts(dom, split):
    """Deterministic disjoint slices indexed by VALID (post-filter) rows:
    train = valid rows 1..ntr, eval = valid rows ntr+1..ntr+nev."""
    name, cfg, fmt, ntr, nev, native = DOMS[dom]
    ds = load_dataset(name, cfg) if cfg else load_dataset(name)
    if native and split == "eval" and native in ds:
        for r in ds[native]:
            t = fmt(r)
            if t:
                yield t
        return
    gen = ds["train"] if hasattr(ds["train"], "__iter__") else ds["train"]
    count = 0
    for r in gen:
        t = fmt(r)
        if not t:
            continue
        count += 1
        if split == "train":
            if native is not None:  # lang: all train rows are fair game
                yield t
                continue
            if count > ntr:
                return
            yield t
        else:
            if native is not None and count > 20000:
                return
            if count <= ntr:
                continue
            if count > ntr + nev:
                return
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

    results = {}
    for dom in ["med", "lang", "units", "music"]:
        res, snap = run_domain(tok, sides, model, mcfg, dom, eval_story)
        results[dom] = res
        torch.save({dom: snap}, WS / f"checkpoints/lib2_{dom}.pt")
        print(f"saved lib2_{dom}.pt", flush=True)

    lib = torch.load(WS / "checkpoints/module_bundles_lib10.pt", map_location=DEVICE,
                     weights_only=False)
    for dom in ["med", "lang", "units", "music"]:
        lib.update(torch.load(WS / f"checkpoints/lib2_{dom}.pt", map_location=DEVICE,
                              weights_only=False))
    torch.save(lib, WS / "checkpoints/module_bundles_final10.pt")
    (WS / "results/batch2_module_results.json").write_text(json.dumps(results, indent=2))
    print("BATCH2 MODULES COMPLETE:",
          {d: f"{r['base_off'][d]:.2f}->{r['on'][d]:.2f}" for d, r in results.items()})


if __name__ == "__main__":
    main()
