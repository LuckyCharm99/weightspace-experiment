"""v10 — continue v9c2: close the code-with-module miss.

v9c2 state: 15/17 grid cells correct. Abstain fires hard on missing/wrong module
(7-8), story always answers (-7), math-with-module flips to answering (-1.4).
Two misses, one root cause: code problem displays in training are prose desc +
signature, but eval code prompts are bare stubs (def fib(n): / if n<2:). The
head never saw stub-shaped answer-decisions under the code module, so code-on
still scored +6 there.

v10: RESUME from checkpoints/abstain_bundle_v9b.pt (bundle+heads+abs emb row),
code abstain/answer windows rebuilt as stub-shaped displays (first lines of the
solution itself, matching eval prompt shape), ABST_MAX 30->48, 300 refinement
steps at LR_AB 1e-4 (was 3e-4). Then recalibrate tau and rerun the full grid.
"""
import sys, json
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).parent))
import abstention_v5 as v5
import abstention_v9 as v9
import abstention_v9b as v9b
import abstention_v7 as v7
import train_module_experts as v3mod

DEVICE = v5.DEVICE
ABS_ID = v5.ABS_ID


def code_pairs_stub(split):
    """(abstain_display, answer_full) with STUB-shaped display: the first two
    lines of the python solution (def header + first body line) — same shape as
    eval CODE_PROMPTS. Span stays byte-identical across worlds (contract kept)."""
    for row in v9._code_ds(split):
        sols = row.get("solutions") or {}
        items = sols.get("solution") or []
        blobs = []
        for s in items:
            if isinstance(s, dict):
                c = (s.get("content") or "").strip()
                lang = str(s.get("language", "")).lower()
                if c and ("python" in lang or lang in ("none", "")):
                    blobs.append(c)
            elif isinstance(s, str):
                c = s.strip()
                if c and ("def " in c or "import " in c or "print(" in c):
                    blobs.append(c)
        if not blobs:
            continue
        lines = [ln for ln in blobs[0].splitlines() if ln.strip()][:2]
        if not lines[0].startswith("def "):
            continue                       # want a stub header, like eval prompts
        disp = "\n".join(lines) + "\n"
        yield disp, disp + v9.PFX + blobs[0][len("\n".join(lines)):].lstrip("\n")


def stub_display_answer(pairs):
    """answer column -> (display, solution_body) tuples for 'answer' mode."""
    for disp, full in pairs:
        sol = full.split("Solution:\n", 1)[1] if "Solution:\n" in full else ""
        if sol:
            yield (disp, sol)


def worlds10(tok, split):
    MA = v5._cycle(lambda: (d for d, _ in v9.math_pairs(split)))
    CS = v5._cycle(lambda: (d for d, _ in code_pairs_stub(split)))
    MB = v5._cycle(lambda: stub_display_answer(v9.math_pairs(split)))
    CB = v5._cycle(lambda: stub_display_answer(code_pairs_stub(split)))
    # duplicate code-stub column so its eval-shaped answer-decisions appear 2x
    # per row budget vs the desc-shape abstain column that used to dominate code
    off = v9b.RowPacker(tok, [(v5.story_stream(60000 if split == 'train' else 300000), 'plain'),
                              (MA, 'abstain'), (CS, 'abstain'), (CS, 'abstain')])
    math_on = v9b.RowPacker(tok, [(v5.story_stream(70000 if split == 'train' else 310000), 'plain'),
                                  (MB, 'answer'), (CS, 'abstain'), (CS, 'abstain')])
    code_on = v9b.RowPacker(tok, [(v5.story_stream(80000 if split == 'train' else 320000), 'plain'),
                                  (CB, 'answer'), (CB, 'answer'), (MA, 'abstain')])
    return off, math_on, code_on


def main():
    print("V10: continue-train v9c2, stub-shaped code windows")
    v9b.RowPacker.ABST_MAX = 48
    model, mcfg, tok = v5.build_model()
    v5.freeze_for_abstention(model)
    bundles = torch.load(v5.BUNDLES if v5.BUNDLES.exists() else v5.BUNDLES_FALLBACK,
                         map_location=DEVICE, weights_only=False)

    resume = torch.load(v5.WS / "checkpoints/abstain_bundle_v9b.pt",
                        map_location=DEVICE, weights_only=False)
    for moe, sd, hs in zip(model.moe_layers, resume["abst"], resume["abs_heads"]):
        moe.abst.load_state_dict(sd)
        moe.abst_head.load_state_dict(hs)
    with torch.no_grad():
        model.embed.weight[ABS_ID] = resume["abs_emb_row"].to(DEVICE).float()
    print("RESUMED from abstain_bundle_v9b.pt (bundle+heads+abs row)")

    off, mo, co = worlds10(tok, 'train')
    x, y, w, a = off.batch()
    dec_idx = (a > 0.5).nonzero()[:, 1]
    print(f"DECDEPTH sample: {dec_idx.tolist()[:10]} (eval prompts sit at ~12-25)")

    v5.AB_STEPS = 300
    v5.LR_AB = 1e-4                      # refinement, not relearn
    v5.train(model, mcfg, off, mo, co, bundles)

    toff, tmo, tco = worlds10(tok, 'test')

    def grab2(mix, snap, n=25):
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

    dA, aA = grab2(toff, None)
    dM, aM = grab2(tmo, bundles['math'])
    dC, aC = grab2(tco, bundles['code'])
    st_a, st_ans = v7.stats(dA + dM + dC), v7.stats(aA + aM + aC)
    tau = round((st_a["p05"] + st_ans["p95"]) / 2, 2)
    print(f"DIAG10 abstain={st_a} answer={st_ans} tau={tau}")

    def gated(prompt, snap, prefix="Solution:\n"):
        v5.set_domain_module(model, snap)
        ids = tok(prompt + prefix, add_special_tokens=False)['input_ids']
        xx = torch.tensor([ids], device=DEVICE)
        model.eval()
        with torch.no_grad():
            model(xx)
        hz = torch.cat([moe.last_abs_logit for moe in model.moe_layers],
                       -1).mean(-1)[0, -1].item()
        return {"head_z": round(hz, 2), "abstain": bool(hz > tau)}

    grid = {}
    for mname, snap in [("none", None), ("math", bundles["math"]), ("code", bundles["code"])]:
        grid[f"module_{mname}"] = {
            "math": [gated(p, snap) for p in v5.MATH_PROMPTS],
            "code": [gated(p, snap) for p in v5.CODE_PROMPTS],
            "story": [gated(p, snap, prefix="\n") for p in v5.STORY_PROMPTS],
        }
        ab = {d: [r["abstain"] for r in rows] for d, rows in grid[f"module_{mname}"].items()}
        zz = {d: [r["head_z"] for r in rows] for d, rows in grid[f"module_{mname}"].items()}
        print(f"module={mname}: {json.dumps(ab)}\n   z: {json.dumps(zz)}")

    gens = {}
    for name, snap, pr, pre in [
            ("math_no_module", None, v5.MATH_PROMPTS[0], "Solution:\n"),
            ("code_no_module", None, v5.CODE_PROMPTS[0], "Solution:\n"),
            ("story_no_module", None, v5.STORY_PROMPTS[0], "\n"),
            ("math_wrong_module", bundles["code"], v5.MATH_PROMPTS[0], "Solution:\n"),
            ("code_wrong_module", bundles["math"], v5.CODE_PROMPTS[0], "Solution:\n"),
            ("math_with_module", bundles["math"], v5.MATH_PROMPTS[0], "Solution:\n"),
            ("code_with_module", bundles["code"], v5.CODE_PROMPTS[0], "Solution:\n")]:
        g = gated(pr, snap, prefix=pre)
        out = pr + " ⟦ABSTAINED⟧" if g["abstain"] else \
            pr + v3mod.nucleus(model, tok, pr, max_new=80, temp=0.8, top_p=0.9)
        gens[name] = out
        print(f"{name} [z={g['head_z']}]: {out[:200]}")

    torch.save({"abst": [moe.abst.state_dict() for moe in model.moe_layers],
                "abs_heads": [moe.abst_head.state_dict() for moe in model.moe_layers],
                "abs_emb_row": model.embed.weight[ABS_ID].detach().cpu(),
                "tau": tau}, v5.WS / "checkpoints/abstain_bundle_v10.pt")
    (v5.WS / "abstain_v10_results.json").write_text(json.dumps(
        {"diag": {"abstain": st_a, "answer": st_ans, "tau": tau},
         "grid": grid, "generations": gens}, indent=2))
    print("V10 COMPLETE")


if __name__ == "__main__":
    main()
