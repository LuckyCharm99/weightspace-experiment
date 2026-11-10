"""v13 + margin mitigation: needed = argmax, but a low top1-top2 circuit margin
means 'unsure WHICH expert' -> abstain even if one is attached. Sweep margins;
regression control = the shipped 40-cell grid (mitigation must cost nothing
there); target = the 16 risky commitments from the confusing-prompt test."""
import sys, json
from pathlib import Path
import torch
import torch.nn as nn
sys.path.insert(0, str(Path(__file__).parent))
import abstention_v5 as v5
import abstention_v9 as v9
import abstention_v11 as v11
from abstention_v13 import DOMS, h_final
from test_v13_confusing import CONFUSING

DEVICE = v5.DEVICE
ABS_ID = v5.ABS_ID
PFX = v9.PFX
MARGINS = [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0]


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
    head = nn.Linear(512, 4).to(DEVICE)
    head.load_state_dict(ck["head"])
    snaps = {"none": None, "math": bundles["math"], "code": bundles["code"], "sql": bundles["sql"]}

    def probe(prompt, mname, prefix=PFX):
        p = {d: mname == d for d in ("math", "code", "sql")}
        v5.set_domain_module(model, snaps[mname])
        ids = tok(prompt + prefix, add_special_tokens=False)["input_ids"]
        h, _ = h_final(model, ids)
        lg = head(h.unsqueeze(0))[0]
        top2 = torch.topk(lg, 2)
        needed = DOMS[int(top2.indices[0])]
        margin = float(top2.values[0] - top2.values[1])
        out = {"needed": needed, "margin": round(margin, 2)}
        for M in MARGINS:
            commit = needed == "general" or (p[needed] and margin >= M)
            out[f"abst_M{M}"] = not commit
        return out

    # ---- regression control: shipped 40-cell grid
    doms = [("math", v5.MATH_PROMPTS), ("code", v5.CODE_PROMPTS), ("sql", v11.SQL_PROMPTS)]
    ctrl = []
    for mname in ["none", "math", "code", "sql"]:
        for d, prods in doms:
            for pr in prods:
                ctrl.append((d, mname, probe(pr, mname)))
        for pr in v5.STORY_PROMPTS:
            ctrl.append(("story", mname, probe(pr, mname, prefix="\n")))
    print("CONTROL 40-cell grid margins:")
    mn_by_dom = {}
    for d, m, v in ctrl:
        mn_by_dom.setdefault(d, []).append(v["margin"])
        if d == m or (d == "story" and v["needed"] == "general"):
            pass
    for d, ms in mn_by_dom.items():
        print(f"  {d}: min {min(ms):.2f} max {max(ms):.2f}")
    ctrl_ok = {}
    for M in MARGINS:
        ok = 0
        for d, m, v in ctrl:
            expect_abstain = not (d == m or d == "story")
            ok += int(v[f"abst_M{M}"] == expect_abstain)
        ctrl_ok[M] = ok
    print("control grid pass by margin:", {str(k): f"{v}/40" for k, v in ctrl_ok.items()})

    # ---- confusing prompts x 4 assemblies, mitigation effect
    rows = []
    for label, pr in CONFUSING:
        for m in ["none", "math", "code", "sql"]:
            rows.append({"prompt": label, "assembly": m, **probe(pr, m)})
    risky = {M: sum(1 for r in rows if not r[f"abst_M{M}"] and r["needed"] != "general")
             for M in MARGINS}
    safe_none = {M: sum(1 for r in rows if r["assembly"] == "none" and r[f"abst_M{M}"])
                 for M in MARGINS}
    over = {M: sum(1 for r in rows if r[f"abst_M{M}"] and r["needed"] == "general")
            for M in MARGINS}
    print("\nM   risky-commits(mitigated)  none-abstain  over-abstain  control")
    for M in MARGINS:
        print(f"{M:>3}  {risky[M]:>2}/16                    {safe_none[M]:>2}/12       {over[M]:>2}         {ctrl_ok[M]}/40")

    # choose smallest M killing all risky commits without touching the control grid
    chosen = next((M for M in MARGINS if risky[M] == 0 and ctrl_ok[M] == 40), None)
    print(f"\nCHOSEN MARGIN: {chosen}")
    if chosen is not None:
        for r in rows:
            if r["assembly"] != "none" and not r[f"abst_M{MARGINS[0]}"] and r["needed"] != "general" \
               and r[f"abst_M{chosen}"]:
                print(f"  neutralized: [{r['prompt'][:36]:36s}] +{r['assembly']:4s} "
                      f"need={r['needed']} margin={r['margin']}")
    (v5.WS / "results/v13_mitigated_results.json").write_text(json.dumps(
        {"control_by_margin": {str(k): v for k, v in ctrl_ok.items()},
         "risky_by_margin": {str(k): v for k, v in risky.items()},
         "chosen": chosen, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
