"""Batch-2b: verify med from its saved bundle, train units + music, ship final9.

(lang was dropped by user decision before its bundle was included.)
Outputs: module_bundles_final10.pt (math,code,sql,chess,chem,regex,med,units,
music), batch2_module_results.json (with a lang_skipped note), lib2_{units,music}.pt.
"""
import sys, json
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).parent))
import train_module_experts as v3mod
from train_batch2_modules import texts, run_domain

DEVICE = v3mod.DEVICE
WS = Path(__file__).resolve().parent.parent


def verify_saved(tok, sides, model, mcfg, dom, eval_story):
    snap = torch.load(WS / f"checkpoints/lib2_{dom}.pt", map_location=DEVICE,
                      weights_only=False)[dom]
    with torch.no_grad():
        for s in sides:
            s.gate_raw.fill_(0.0)
    eval_dom = v3mod.make_batches(v3mod.Packer(tok, lambda: texts(dom, "eval")), 24)
    base = {"story": v3mod.eval_loss_on(sides, model, mcfg, eval_story),
            dom: v3mod.eval_loss_on(sides, model, mcfg, eval_dom)}
    v3mod.restore(sides, snap)
    on = {"story": v3mod.eval_loss_on(sides, model, mcfg, eval_story),
          dom: v3mod.eval_loss_on(sides, model, mcfg, eval_dom)}
    with torch.no_grad():
        for s in sides:
            s.gate_raw.fill_(0.0)
    off2 = {dom: v3mod.eval_loss_on(sides, model, mcfg, eval_dom)}
    ok_pure = abs(off2[dom] - base[dom]) < 1e-3
    v3mod.restore(sides, snap)
    swap = {dom: v3mod.eval_loss_on(sides, model, mcfg, eval_dom)}
    ok_swap = abs(swap[dom] - on[dom]) < 1e-3
    print(f"{dom.upper()} (saved bundle) base {base} on {on} | purity {ok_pure} | swap {ok_swap}",
          flush=True)
    with torch.no_grad():
        for s in sides:
            s.gate_raw.fill_(0.0)
    return {"base_off": base, "on": on, "purity": off2, "swap": swap,
            "purity_ok": ok_pure, "swap_ok": ok_swap}


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B", trust_remote_code=True)
    tok.pad_token = tok.eos_token
    model, mcfg = v3mod.load_base()
    sides = v3mod.attach_modules(model, mcfg)
    v3mod.freeze_base(sides, model)
    eval_story = v3mod.make_batches(v3mod.Packer(tok, lambda: v3mod.story_stream(60000)), 24)

    results = {"lang_skipped": "user decision 2026-11-10; bundle remains on disk"}
    results["med"] = verify_saved(tok, sides, model, mcfg, "med", eval_story)
    snaps = {"med": torch.load(WS / "checkpoints/lib2_med.pt", map_location=DEVICE,
                               weights_only=False)["med"]}
    for dom in ["units", "music"]:
        res, snap = run_domain(tok, sides, model, mcfg, dom, eval_story)
        results[dom] = res
        torch.save({dom: snap}, WS / f"checkpoints/lib2_{dom}.pt")
        snaps[dom] = snap

    lib = torch.load(WS / "checkpoints/module_bundles_merged.pt", map_location=DEVICE,
                     weights_only=False)
    for f in ("lib10_chess", "lib10_chem", "lib10_regex"):
        lib.update(torch.load(WS / f"checkpoints/{f}.pt", map_location=DEVICE,
                              weights_only=False))
    lib.update(snaps)
    torch.save(lib, WS / "checkpoints/module_bundles_lib9.pt")
    (WS / "results/batch2_module_results.json").write_text(json.dumps(results, indent=2))
    print("FINAL LIBRARY:", sorted(lib.keys()),
          {d: f"{results[d]['base_off'][d]:.2f}->{results[d]['on'][d]:.2f}"
           for d in ("med", "units", "music")})


if __name__ == "__main__":
    main()
