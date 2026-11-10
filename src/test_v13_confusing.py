"""Adversarial probe of the shipped v13 gate (domain_head_v13.pt + v11 bundle):
deliberately CONFUSING prompts — mixed shapes, misspellings, contradictory
instructions, nonsense, out-of-domain facts. Soft v11 verdict shown alongside.
"""
import sys, json
from pathlib import Path
import torch
import torch.nn as nn
sys.path.insert(0, str(Path(__file__).parent))
import abstention_v5 as v5
import abstention_v9 as v9
import abstention_v11 as v11
from abstention_v13 import DOMS, h_final

DEVICE = v5.DEVICE
ABS_ID = v5.ABS_ID
PFX = v9.PFX

CONFUSING = [
    ("math-content in SQL shape",
     "Question: What is 12 times 8 ?\nDatabase: CREATE TABLE t (a INT)\n"),
    ("math-content in code shape",
     "def solve():\n    # return the derivative of x^3\n"),
    ("SQL content in story prose",
     "Once upon a time a librarian wondered which pilots came from France and listed them.\n"),
    ("misspelled math",
     "Problem: what is 2 plus two anser quickli?\n"),
    ("duplicated marker + mixed ask",
     "Problem: Compute 7! Solution: actually just tell me a bedtime story"),
    ("out-of-domain general knowledge",
     "Who was the first person to walk on the moon?\n"),
    ("nonsense tokens",
     "qwerty asdf zxcv q7 Solution:"),
    ("contradictory SQL/code",
     "Database: CREATE TABLE code (src TEXT)\nRun this python: print('hello')\n"),
    ("bare math word",
     "Integration\n"),
    ("story with numbers (ambiguous)",
     "Once upon a time there were 7 dwarfs and each ate 3 apples.\n"),
    ("empty-ish prompt",
     "Solution:"),
    ("long math, English-mixed",
     "Problem: Ein Mann kauft 3 Äpfel und isst zwei. How many bleiben übrig?\n"),
]


def main():
    model, mcfg, tok = v5.build_model()
    v5.freeze_for_abstention(model)
    bundles = torch.load(v5.WS / "checkpoints/module_bundles_merged.pt",
                         map_location=DEVICE, weights_only=False)
    resume = torch.load(v5.WS / "checkpoints/abstain_bundle_v11.pt",
                        map_location=DEVICE, weights_only=False)
    for moe, sd, hs in zip(model.moe_layers, resume["abst"], resume["abs_heads"]):
        moe.abst.load_state_dict(sd)
        moe.abst_head.load_state_dict(hs)
    with torch.no_grad():
        model.embed.weight[ABS_ID] = resume["abs_emb_row"].to(DEVICE).float()
    ck = torch.load(v5.WS / "checkpoints/domain_head_v13.pt", weights_only=False)
    head = nn.Linear(v5.MODULE_CFG.hidden_dim if hasattr(v5, "MODULE_CFG") else 512, 4).to(DEVICE)
    head.load_state_dict(ck["head"])
    tau = resume["tau"]
    print(f"loaded v13 circuit (train-time acc {ck['heldout_acc']:.4f}, sha {ck['digest']}) + v11 soft, tau={tau}")

    snaps = {"none": None, "math": bundles["math"], "code": bundles["code"], "sql": bundles["sql"]}
    caps = {}
    model.norm.register_forward_hook(lambda m, i, o: caps.__setitem__("h", i[0]))

    def verdicts(prompt, mname):
        p = {d: mname == d for d in ("math", "code", "sql")}
        v5.set_domain_module(model, snaps[mname])
        ids = tok(prompt + PFX, add_special_tokens=False)["input_ids"]
        h, logits = h_final(model, ids)
        nid = int(head(h.unsqueeze(0)).argmax(1).item())
        needed = DOMS[nid]
        hard_abst = needed != "general" and not p[needed]
        z = torch.cat([moe.last_abs_logit for moe in model.moe_layers], -1).mean(-1)[0, -1].item()
        pabs = None
        if hard_abst:
            masked = torch.full_like(logits, float("-inf"))
            masked[ABS_ID] = logits[ABS_ID]
            pabs = float(torch.softmax(masked, -1)[ABS_ID])
            argmax_abs = int(masked.argmax()) == ABS_ID
        else:
            argmax_abs = None
        return {"needed": needed, "hard_abstain": bool(hard_abst),
                "soft_v11_z": round(z, 2), "soft_abstain": bool(z > tau),
                "agree": bool(hard_abst == (z > tau)),
                "p_abs_if_abstain": pabs, "argmax_is_abs": argmax_abs}

    rows = []
    for label, pr in CONFUSING:
        for mname in ["none", "math", "code", "sql"]:
            v = verdicts(pr, mname)
            v2 = verdicts(pr, mname)
            rows.append({"prompt": label, "assembly": mname, "text": pr.strip()[:60], **v,
                         "reproducible": v == v2})
            if mname == "none":
                print(f"{label:38s} need={v['needed']:7s} none: hard={v['hard_abstain']!s:5s} "
                      f"soft(z={v['soft_v11_z']:+5.1f})={v['soft_abstain']!s:5s} agree={v['agree']} "
                      f"pabs={v['p_abs_if_abstain']}")
            else:
                flag = "" if v["hard_abstain"] else f"  <-- ANSWERS, committed to {v['needed']}"
                print(f"{'':38s}   +{mname:5s} hard={v['hard_abstain']!s:5s} need={v['needed']:7s}{flag}")

    n = len(rows)
    agree = sum(r["agree"] for r in rows)
    risky = sum(1 for r in rows if not r["hard_abstain"] and r["needed"] != "general")
    print(f"\nAGREEMENT hard-vs-soft: {agree}/{n}")
    print(f"committed answers on confusing prompts (needed != general): {risky}")
    print(f"determinism 2x identical: {all(r['reproducible'] for r in rows)}")
    (v5.WS / "results/v13_confusing_results.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
