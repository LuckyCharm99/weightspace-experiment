"""v11 — three-expert abstention: add SQL domain to the module library.

Same proven geometry stack (byte-identical spans, row-aligned fresh starts,
output-frame-correct labels, dual decision-class supervision, equal loss mass,
train==eval content shape), extended from 2 experts to 3:

  worlds: off / math-on / code-on / sql-on (phase % 4)
  every world shows the OTHER domains' eval-shaped displays in abstain mode,
  the matching domain's answers in 'answer' mode (label 0 at the marker end).

Resumes from checkpoints/abstain_bundle_v10.pt at refinement LR; recalibrates
tau on decision-geometry buckets; auto-scores the grid against the expectation
table; saves abstain_v11_results.json + checkpoints/abstain_bundle_v11.pt.
"""
import sys, json, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).parent))
import abstention_v5 as v5
import abstention_v9 as v9
import abstention_v9b as v9b
import abstention_v10 as v10
import abstention_v7 as v7
import train_module_experts as v3mod
import train_sql_module as sqlmod

DEVICE = v5.DEVICE
ABS_ID = v5.ABS_ID
PFX = v9.PFX

SQL_PROMPTS = [
    "Question: How many pilots come from France ?\n"
    "Database: CREATE TABLE pilot (Country TEXT, Age INTEGER)\n",
    "Question: What are the names of all countries and their total populations ?\n"
    "Database: CREATE TABLE country (Name TEXT, Population INTEGER)\n",
    "Question: Show the airlines that fly to both Paris and London .\n"
    "Database: CREATE TABLE flight (Airline TEXT, Dest TEXT)\n",
]


def sql_pairs(split):
    """(display, answer_full) — display in eval shape, ends right before marker.
    'train' slice or anything else = held-out tail (sqlmod handles both)."""
    for r in sqlmod.sql_rows(split if split == 'train' else 'eval'):
        q = (r.get("question") or "").strip()
        ctx = (r.get("context") or "").strip()
        a = (r.get("answer") or "").strip()
        if not (q and ctx and a):
            continue
        disp = f"Question: {q}\nDatabase: {ctx}\n"
        yield disp, disp + PFX + a + "\n"


def sa_answer(pairs):
    for disp, full in pairs:
        sol = full.split(PFX, 1)[1] if PFX in full else ""
        if sol:
            yield (disp, sol)


def worlds11(tok, split):
    MA = v5._cycle(lambda: (d for d, _ in v9.math_pairs(split)))
    MB = v5._cycle(lambda: v10.stub_display_answer(v9.math_pairs(split)))
    CA = v5._cycle(lambda: (d for d, _ in sqlmod_code_pairs(split)))
    CB = v5._cycle(lambda: v10.stub_display_answer(sqlmod_code_pairs(split)))
    SA = v5._cycle(lambda: (d for d, _ in sql_pairs(split)))
    SB = v5._cycle(lambda: sa_answer(sql_pairs(split)))
    story = lambda s: (v5.story_stream(s), 'plain')
    off = v9b.RowPacker(tok, [story(60000 if split == 'train' else 300000),
                              (MA, 'abstain'), (CA, 'abstain'), (SA, 'abstain')])
    math_on = v9b.RowPacker(tok, [story(70000 if split == 'train' else 310000),
                                  (MB, 'answer'), (CA, 'abstain'), (SA, 'abstain')])
    code_on = v9b.RowPacker(tok, [story(80000 if split == 'train' else 320000),
                                  (CB, 'answer'), (MA, 'abstain'), (SA, 'abstain')])
    sql_on = v9b.RowPacker(tok, [story(90000 if split == 'train' else 330000),
                                 (SB, 'answer'), (MA, 'abstain'), (CA, 'abstain')])
    return off, math_on, code_on, sql_on


def sqlmod_code_pairs(split):
    return v10.code_pairs_stub(split)


def train4(model, mix_off, mix_math, mix_code, mix_sql, bundles):
    print("\n" + "=" * 60)
    print("TRAINING ABSTENTION (4 worlds: off / math-on / code-on / sql-on)")
    print("=" * 60)
    params_mod, params_gate, params_head = [], [], []
    for moe in model.moe_layers:
        for nm, p in moe.abst.named_parameters():
            (params_gate if nm == "gate_raw" else params_mod).append(p)
        params_head += list(moe.abst_head.parameters())
    opt = torch.optim.AdamW([
        {"params": params_mod, "lr": v5.LR_AB},
        {"params": params_gate, "lr": v5.LR_GATE_AB},
        {"params": params_head, "lr": v5.LR_AB},
        {"params": [model.embed.weight], "lr": v5.LR_EMB},
        {"params": [model.lm_head.boost], "lr": 5e-3},
    ], weight_decay=0.0)
    names = ["off", "math-on", "code-on", "sql-on"]
    snaps = [None, bundles["math"], bundles["code"], bundles["sql"]]
    packs = [mix_off, mix_math, mix_code, mix_sql]
    model.train()
    t0 = time.time()
    for step in range(v5.AB_STEPS):
        phase = (step // v5.SWAP_EVERY) % 4
        v5.set_domain_module(model, snaps[phase])
        opt.zero_grad()
        av = ab = 0.0
        for _ in range(v5.AB_GRAD_ACCUM):
            inp, y, w, a = packs[phase].batch()
            l, vl, bl = v5.abst_loss(model, inp, y, w, a)
            (l / v5.AB_GRAD_ACCUM).backward()
            av += vl; ab += bl
        torch.nn.utils.clip_grad_norm_(
            params_mod + params_gate + params_head + [model.embed.weight], 1.0)
        opt.step()
        if (step + 1) % 50 == 0:
            print(f"  step {step+1}/{v5.AB_STEPS} | world {names[phase]} | "
                  f"vocab {av/v5.AB_GRAD_ACCUM:.3f} | abstain-bce {ab/v5.AB_GRAD_ACCUM:.4f} | "
                  f"{time.time()-t0:.0f}s")


def main():
    print("V11: three-expert abstention (math/code/sql)")
    v9b.RowPacker.ABST_MAX = 48
    model, mcfg, tok = v5.build_model()
    v5.freeze_for_abstention(model)
    bundles = torch.load(v5.WS / "checkpoints/module_bundles_merged.pt",
                         map_location=DEVICE, weights_only=False)
    assert set(bundles) >= {"math", "code", "sql"}

    resume = torch.load(v5.WS / "checkpoints/abstain_bundle_v10.pt",
                        map_location=DEVICE, weights_only=False)
    for moe, sd, hs in zip(model.moe_layers, resume["abst"], resume["abs_heads"]):
        moe.abst.load_state_dict(sd)
        moe.abst_head.load_state_dict(hs)
    with torch.no_grad():
        model.embed.weight[ABS_ID] = resume["abs_emb_row"].to(DEVICE).float()
    print("RESUMED from abstain_bundle_v10.pt")

    packs = worlds11(tok, 'train')
    x, y, w, a = packs[0].batch()
    print(f"DECDEPTH sample: {(a > 0.5).nonzero()[:, 1].tolist()[:10]}")

    v5.AB_STEPS = 400
    v5.LR_AB = 1e-4
    train4(model, *packs, bundles)

    t_off, t_math, t_code, t_sql = worlds11(tok, 'test')

    def grab2(mix, snap, n=20):
        v5.set_domain_module(model, snap)
        dec, ans = [], []
        model.eval()
        for _ in range(n):
            x, y, w, a = mix.batch()
            with torch.no_grad():
                model(x)
            hz = torch.cat([moe.last_abs_logit for moe in model.moe_layers],
                           -1).mean(-1).squeeze(-1)
            hi_w = w >= 20
            dec += hz[hi_w & (a > 0.5)].tolist()
            ans += hz[hi_w & (a == 0)].tolist()
        return dec, ans

    dA, aA = grab2(t_off, None)
    dM, aM = grab2(t_math, bundles['math'])
    dC, aC = grab2(t_code, bundles['code'])
    dS, aS = grab2(t_sql, bundles['sql'])
    st_a, st_ans = v7.stats(dA + dM + dC + dS), v7.stats(aA + aM + aC + aS)
    tau = round((st_a["p05"] + st_ans["p95"]) / 2, 2)
    print(f"DIAG11 abstain={st_a} answer={st_ans} tau={tau}")

    def gated(prompt, snap, prefix=PFX):
        v5.set_domain_module(model, snap)
        ids = tok(prompt + prefix, add_special_tokens=False)['input_ids']
        xx = torch.tensor([ids], device=DEVICE)
        model.eval()
        with torch.no_grad():
            model(xx)
        hz = torch.cat([moe.last_abs_logit for moe in model.moe_layers],
                       -1).mean(-1)[0, -1].item()
        return {"head_z": round(hz, 2), "abstain": bool(hz > tau)}

    doms = [("math", v5.MATH_PROMPTS), ("code", v5.CODE_PROMPTS), ("sql", SQL_PROMPTS)]
    grid, ok, tot = {}, 0, 0
    for mname, snap in [("none", None), ("math", bundles["math"]),
                        ("code", bundles["code"]), ("sql", bundles["sql"])]:
        grid[f"module_{mname}"] = {
            d: [gated(p, snap) for p in prods] for d, prods in doms}
        grid[f"module_{mname}"]["story"] = [
            gated(p, snap, prefix="\n") for p in v5.STORY_PROMPTS]
        row = grid[f"module_{mname}"]
        for d, cells in row.items():
            for c in cells:
                expect_abstain = not (d == mname or d == "story")
                tot += 1; ok += int(c["abstain"] == expect_abstain)
        ab = {d: [c["abstain"] for c in cells] for d, cells in row.items()}
        zz = {d: [c["head_z"] for c in cells] for d, cells in row.items()}
        print(f"module={mname}: {json.dumps(ab)}\n   z: {json.dumps(zz)}")
    print(f"GRID PASS {ok}/{tot}")

    gens = {}
    for name, snap, pr, pre in [
            ("math_no_module", None, v5.MATH_PROMPTS[0], PFX),
            ("code_no_module", None, v5.CODE_PROMPTS[0], PFX),
            ("sql_no_module", None, SQL_PROMPTS[0], PFX),
            ("sql_wrong_module", bundles['math'], SQL_PROMPTS[0], PFX),
            ("math_with_module", bundles['math'], v5.MATH_PROMPTS[0], PFX),
            ("code_with_module", bundles['code'], v5.CODE_PROMPTS[0], PFX),
            ("sql_with_module", bundles['sql'], SQL_PROMPTS[0], PFX),
            ("story_no_module", None, v5.STORY_PROMPTS[0], "\n")]:
        g = gated(pr, snap, prefix=pre)
        out = pr + " ⟦ABSTAINED⟧" if g["abstain"] else \
            pr + v3mod.nucleus(model, tok, pr, max_new=80, temp=0.8, top_p=0.9)
        gens[name] = out
        print(f"{name} [z={g['head_z']}]: {out[:160]}")

    torch.save({"abst": [moe.abst.state_dict() for moe in model.moe_layers],
                "abs_heads": [moe.abst_head.state_dict() for moe in model.moe_layers],
                "abs_emb_row": model.embed.weight[ABS_ID].detach().cpu(),
                "tau": tau}, v5.WS / "checkpoints/abstain_bundle_v11.pt")
    (v5.WS / "abstain_v11_results.json").write_text(json.dumps(
        {"diag": {"abstain": st_a, "answer": st_ans, "tau": tau},
         "grid": grid, "grid_pass": f"{ok}/{tot}", "generations": gens}, indent=2))
    print("V11 COMPLETE")


if __name__ == "__main__":
    main()
