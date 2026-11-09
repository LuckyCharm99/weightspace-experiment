"""v9 — abstention retrain: TRAIN==EVAL decision span, ONLY the gate differs.

v8's fatal leak: the OFF-world math text was "Problem: X\\n" but the ON-world math
text was raw "X\\nSolution:\\n<solution>" (no 'Problem: ' prefix, different marker
placement). So 'abstain' correlated with the "Problem: " string + a doubled marker
that eval never produces -> head fired on the artifact (train z=+5) but was dead on
real prompts (z=-8.5). That is why every prior run showed p_abs=0.

v9 contract (the whole point):
  For a given example, the token sequence from start THROUGH the final
  'Solution:\\n' is BYTE-IDENTICAL across all three worlds. Worlds differ ONLY by:
     off          : module gate OFF  -> abstain target (label 1) at decision pt
     right-module : gate ON (matching bundle) -> answer: plain next-token CE
     wrong-module : gate ON (other bundle)    -> abstain (label 1) again
  Because text is identical, the ONLY way the head can separate label-0 (answer)
  from label-1 (abstain) is to read d = the domain module's activity/identity.
  That is the intended concept and it is format-free, so it transfers to eval.

Cross-domain abstain (math problem shown while CODE module is on -> abstain)
teaches the head to distinguish WHICH module is online, not merely "some module".
"""
import sys, json
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).parent))
import abstention_v5 as v5
import train_module_experts as v3mod
import abstention_v7 as v7

WS = v5.WS
DEVICE = v5.DEVICE
ABS_ID = v5.ABS_ID
PFX = "Solution:\n"


def _math_ds(split):
    for subject in v3mod.MATH_SUBJECTS:
        yield from v3mod.load_dataset("EleutherAI/hendrycks_math", subject,
                                      split=split, streaming=True)


def _code_ds(split):
    yield from v3mod.load_dataset("deepmind/code_contests", split=split, streaming=True)


def math_pairs(split):
    """(abstain_display, answer_full) with IDENTICAL decision span."""
    for row in _math_ds(split):
        prob = "Problem: " + row["problem"].strip() + "\n"
        yield prob, prob + PFX + row["solution"].strip()


def code_pairs(split):
    for row in _code_ds(split):
        desc = (row.get("description") or "").strip()
        if not desc:
            continue
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
                    blobs.append(c)          # plain-str rows (test split): python heuristic
            if len(blobs) >= 2:
                break
        if not blobs:
            continue
        sig = "def solution(n):\n"
        for line in blobs[0].splitlines():
            if line.strip().startswith("def "):
                sig = line.strip() + "\n"
                break
        disp = desc + "\n" + sig            # abstain side (packer appends PFX)
        yield disp, disp + PFX + blobs[0]   # answer side (PFX inline, byte-identical span)


def cyc(pairs, idx):
    """Infinite generator over the chosen column of a pair stream."""
    def g():
        for tup in pairs:
            for _ in range(1):
                yield tup[idx]
    return v5._cycle(g)


def v8pack(tok, specs):
    """specs: list of (infinite_text_gen, mode). abstain mode appends PFX; plain
    is verbatim. This is v5.MixPacker's exact behaviour -> reuse it, no override."""
    return v5.MixPacker(tok, specs)


def build_worlds(tok, split):
    mp, ma = [], []
    # We build 4 infinite columns: math-abstain-display, math-answer, code-abstain, code-answer.
    p = math_pairs(split); q = code_pairs(split)
    def m_abst():
        for a, _ in p: yield a
    p = math_pairs(split); q = code_pairs(split)
    def m_ans():
        for _, b in p: yield b
    def c_abst():
        for a, _ in q: yield a
    def c_ans():
        for _, b in q: yield b
    MA, MB, CA, CB = (v5._cycle(m_abst), v5._cycle(m_ans),
                      v5._cycle(c_abst), v5._cycle(c_ans))
    # off: everything abstains, story answers
    off = v8pack(tok, [(v5.story_stream(60000), 'plain'), (MA, 'abstain'), (CA, 'abstain')])
    # math-on: math answers, code abstains, story answers
    math_on = v8pack(tok, [(v5.story_stream(70000), 'plain'),
                           (MB, 'plain'), (CA, 'abstain')])
    # code-on: code answers, math abstains, story answers
    code_on = v8pack(tok, [(v5.story_stream(80000), 'plain'),
                           (CB, 'plain'), (MA, 'abstain')])
    return off, math_on, code_on


def main():
    print("V9: paired identical-span abstention retrain")
    model, mcfg, tok = v5.build_model()
    v5.freeze_for_abstention(model)   # fresh abst/heads, EOS-seeded <abs> row, learnable boost
    bundles = torch.load(v5.BUNDLES if v5.BUNDLES.exists() else v5.BUNDLES_FALLBACK,
                         map_location=DEVICE, weights_only=False)

    off, math_on, code_on = build_worlds(tok, "train")
    v5.AB_STEPS = 600
    v5.train(model, mcfg, off, math_on, code_on, bundles)

    # sanity: prove the decision span is byte-identical across worlds (no hidden leak)
    def span(mode, txt):
        ids = tok(txt, truncation=True, max_length=4096)["input_ids"]
        pfx = tok(PFX, add_special_tokens=False)["input_ids"]
        if mode == "abstain":
            return ids + pfx[:len(pfx)]        # decision pt = end of appended PFX
        return ids
    probe = v5._cycle(lambda: math_pairs("test"))
    one = next(probe)
    a_span = span("abstain", one)               # one = display "Problem: X\n"
    b_txt = tok(one, add_special_tokens=False)["input_ids"]
    print(f"SPANCHK abstain-span len={len(a_span)}  head={tok.decode(a_span[-8:])!r}")

    # ---- calibration on TEST-split identical-span streams ----
    toff, tmath_on, tcode_on = build_worlds(tok, "test")
    def grab(mix, snap, n=25, want="both"):
        v5.set_domain_module(model, snap)
        d, p = v7.head_z_batch(model, mix, n_batches=n)
        return d, p
    # abstain class: off math+code AND cross-domain (math shown w/ code module, etc.)
    decA, _ = grab(toff, None)
    _, ansM = grab(tmath_on, bundles['math'])       # answers when matching module on
    _, ansC = grab(tcode_on, bundles['code'])
    decX, _ = grab(tmath_on, bundles['math'], want='abst')  # placeholder; real cross below
    # cross-domain abstain: math display while code module online (label 1 in training)
    cd = v8pack(tok, [(v5._cycle(lambda: [a for a, _ in math_pairs("test")]), 'abstain')])
    decCross, _ = grab(cd, bundles['code'])
    abstain_z = decA + decCross
    answer_z = ansM + ansC
    st_a, st_ans = v7.stats(abstain_z), v7.stats(answer_z)
    sep = round(st_a["p05"] - st_ans["p95"], 2)
    tau = round((st_a["p05"] + st_ans["p95"]) / 2, 2)
    print(f"DIAG9 abstain={st_a} answer={st_ans} sep={sep} tau={tau}")

    # ---- verdict grid on the REAL eval prompts ----
    def gated(prompt, snap, prefix=PFX):
        v5.set_domain_module(model, snap)
        ids = tok(prompt + prefix, add_special_tokens=False)['input_ids']
        x = torch.tensor([ids], device=DEVICE)
        model.eval()
        with torch.no_grad():
            model(x)
        hz = torch.cat([moe.last_abs_logit for moe in model.moe_layers], -1).mean(-1)[0, -1].item()
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

    # ---- generations ----
    gens = {}
    for name, snap, pr, pre in [
            ("math_no_module", None, v5.MATH_PROMPTS[0], PFX),
            ("code_no_module", None, v5.CODE_PROMPTS[0], PFX),
            ("story_no_module", None, v5.STORY_PROMPTS[0], "\n"),
            ("math_wrong_module", bundles["code"], v5.MATH_PROMPTS[0], PFX),
            ("math_with_module", bundles["math"], v5.MATH_PROMPTS[0], PFX),
            ("code_with_module", bundles["code"], v5.CODE_PROMPTS[0], PFX)]:
        g = gated(pr, snap, prefix=pre)
        out = pr + " ⟦ABSTAINED⟧" if g["abstain"] else \
            pr + v3mod.nucleus(model, tok, pr, max_new=80, temp=0.8, top_p=0.9)
        gens[name] = out
        print(f"{name} [z={g['head_z']}]: {out[:200]}")

    torch.save({"abst": [moe.abst.state_dict() for moe in model.moe_layers],
                "abs_heads": [moe.abst_head.state_dict() for moe in model.moe_layers],
                "abs_emb_row": model.embed.weight[ABS_ID].detach().cpu(),
                "tau": tau}, WS / "checkpoints/abstain_bundle_v9.pt")
    (WS / "abstain_v9_results.json").write_text(json.dumps(
        {"diag": {"abstain": st_a, "answer": st_ans, "sep": sep, "tau": tau},
         "grid": grid, "generations": gens}, indent=2))
    print("V9 COMPLETE")


if __name__ == "__main__":
    main()
