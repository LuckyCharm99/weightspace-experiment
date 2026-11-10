"""10th expert: propositional LOGICS (ergotts/propositional-logic, 1600 rows:
NL argument -> numbered symbolic proof). Registers the domain into the batch-2
harness and reuses run_domain's exact 600-step protocol + purity/swap proofs.
Outputs: checkpoints/lib3_logics.pt, module_bundles_final10.pt,
logics_module_results.json.
Run: env -u PYTHONPATH OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 venv/bin/python -u train_logics.py"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import train_module_experts as v3mod
import train_batch2_modules as b2
from train_batch2_modules import run_domain

WS = Path(__file__).resolve().parent.parent
DEVICE = v3mod.DEVICE


def fmt_logics(r):
    nl = (r.get("natural_language_argument") or "").strip()[:1200]
    pf = (r.get("propositional_logic_proof") or "").strip()[:900]
    if not nl or not pf:
        return None
    return f"{nl}\nFormalize and prove.\nProof:\n{pf}\n"


# train = valid rows 1..1400, eval = valid rows 1401..1600 (tail-slice convention)
b2.DOMS["logics"] = ("ergotts/propositional-logic", None, fmt_logics, 1400, 200, None)


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B", trust_remote_code=True)
    tok.pad_token = tok.eos_token
    model, mcfg = v3mod.load_base()
    sides = v3mod.attach_modules(model, mcfg)
    v3mod.freeze_base(sides, model)
    eval_story = v3mod.make_batches(v3mod.Packer(tok, lambda: v3mod.story_stream(60000)), 24)

    res, snap = run_domain(tok, sides, model, mcfg, "logics", eval_story)
    torch.save({"logics": snap}, WS / "checkpoints/lib3_logics.pt")
    lib = torch.load(WS / "checkpoints/module_bundles_merged.pt", map_location=DEVICE,
                 weights_only=False)
    for f in ("lib10_chess", "lib10_chem", "lib10_regex",
              "lib2_med", "lib2_units", "lib2_music"):
        lib.update(torch.load(WS / f"checkpoints/{f}.pt", map_location=DEVICE,
                              weights_only=False))
    lib["logics"] = snap
    torch.save(lib, WS / "checkpoints/module_bundles_final10.pt")
    (WS / "results/logics_module_results.json").write_text(json.dumps({"logics": res}, indent=2))
    print(f"LOGICS: {res['base_off']['logics']:.2f} -> {res['on']['logics']:.2f} | "
          f"purity {res['purity_ok']} swap {res['swap_ok']}")
    print("FINAL10:", sorted(lib.keys()))


if __name__ == "__main__":
    main()
