"""v14 — router-free abstention via expert self-verification (no central
classifier, no adversarial labels -> no memorization surface).

Every attached expert X must CLAIM the input by passing two deterministic
checks on X's own greedy continuation:
  1. FORMAT: the continuation actually looks like X's domain output
     (digits/vars for math, indented/keyword code, SQL keywords) — a pure
     function written from the module's training definition.
  2. CONFIDENCE: mean per-token logprob (repetition-discounted) >= theta_X,
     calibrated on TRAIN-split prompts only.

Verdict: answer if the attached expert claims; else answer if the BASE claims
(story format + theta_base); else abstain (softmax nullified to <abs>).
Nonsense/ambiguous text fails format -> cannot claim no matter how confidently
the model repeats itself (repetition inflates logprob, so dup-rate discounts).

Evaluated on: 40-cell grid, 48 confusing-probe cells (never seen in
calibration), determinism double-run. Baseline to beat: v13 + M=4.
"""
import sys, json, re, time
from pathlib import Path
import torch
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).parent))
import abstention_v5 as v5
import abstention_v9 as v9
import abstention_v10 as v10
import abstention_v11 as v11
from test_v13_confusing import CONFUSING

DEVICE = v5.DEVICE
ABS_ID = v5.ABS_ID
PFX = v9.PFX
GEN_N = 16

MATH_RE = re.compile(r"\d|[x+y=*/^$]")
CODE_RE = re.compile(r"^\s{2,}\S|\n\s{2,}\S|\b(def|return|print|import|lambda|class|True|False|None)\b|\w+\(|==",
                     re.M)
SQL_RE = re.compile(r"\b(SELECT|FROM|WHERE|INSERT|UPDATE|GROUP|COUNT)\b", re.I)
WORD_RE = re.compile(r"\b(the|a|and|was|were|with|her|his|little)\b", re.I)


def format_claim(dom, text):
    if dom == "math":
        return bool(MATH_RE.search(text))
    if dom == "code":
        return bool(CODE_RE.search(text))
    if dom == "sql":
        return bool(SQL_RE.search(text))
    if dom == "base":
        words = len(WORD_RE.findall(text))
        digits = sum(c.isdigit() for c in text)
        return words >= 2 and digits < max(3, len(text) // 10) and not SQL_RE.search(text)
    return False


def greedy(model, tok, prompt, assembly_snap):
    """returns (text, mean_logprob, dup_rate) for greedy continuation."""
    v5.set_domain_module(model, assembly_snap)
    ids = tok(prompt, add_special_tokens=False)["input_ids"]
    x = torch.tensor([ids], device=DEVICE)
    lps, outs = [], []
    with torch.no_grad():
        for _ in range(GEN_N):
            out = model(x)
            logits = out[0] if isinstance(out, tuple) else out
            lp = F.log_softmax(logits[0, -1].float(), -1)
            nxt = int(lp.argmax())
            lps.append(float(lp[nxt]))
            outs.append(nxt)
            x = torch.cat([x, torch.tensor([[nxt]], device=DEVICE)], -1)
    text = tok.decode(outs)
    dup = 1.0 - len(set(outs)) / len(outs)
    return text, sum(lps) / len(lps), dup


def score(model, tok, snaps, prompt, dom, assembly):
    text, ml, dup = greedy(model, tok, prompt + (PFX if dom != "story" else "\n"),
                           snaps[assembly])
    claim_fmt = format_claim(dom if assembly != "none" else "base", text)
    adj = ml - 2.0 * dup          # repetition discount
    return {"text": text[:60], "conf": round(ml, 3), "dup": round(dup, 2),
            "adj": round(adj, 3), "format": bool(claim_fmt)}


def main():
    print("V14: router-free expert-verification abstention")
    t0 = time.time()
    model, mcfg, tok = v5.build_model()
    v5.freeze_for_abstention(model)
    bundles = torch.load(v5.WS / "checkpoints/module_bundles_merged.pt",
                         map_location=DEVICE, weights_only=False)
    snaps = {"none": None, "math": bundles["math"], "code": bundles["code"], "sql": bundles["sql"]}
    with torch.no_grad():  # v14 = soft layer fully DISABLED: zero abst side-experts
        for moe in model.moe_layers:
            moe.abst.gate_raw.fill_(0.0)            # tanh(0)=0 -> no injection
            moe.abst_head.weight.zero_()            # dead head, belt and braces
            moe.abst_head.bias.zero_()
        model.embed.weight[ABS_ID] = 0.0            # untrained <abs> row: v14 abstains structurally
    model.eval()

    # ---------- calibration pools: TRAIN splits only
    pool = {}
    pool["math"] = [d for d, _ in v9.math_pairs("train")][:40]
    pool["code"] = [d for d, _ in v10.code_pairs_stub("train")][:40]
    pool["sql"] = [d for d, _ in v11.sql_pairs("train")][:40]
    st = v5.story_stream(13)
    pool["story"] = []
    while len(pool["story"]) < 40:
        c = next(st)
        ids = tok(c, add_special_tokens=False)["input_ids"][:36]
        if len(ids) >= 20:
            pool["story"].append(tok.decode(ids))

    print("CALIBRATE: scores on train pools under own-assembly")
    th = {}
    cal = {}
    for dom in ["math", "code", "sql", "story"]:
        asm = "none" if dom == "story" else dom
        rows = [score(model, tok, snaps, p, dom, asm) for p in pool[dom]]
        fmt_ok = [r for r in rows if r["format"]]
        pos = [r["adj"] for r in fmt_ok]
        # negatives: other-domain prompts under assembly dom (they must NOT claim)
        neg = []
        if dom != "story":
            for od in ["math", "code", "sql", "story"]:
                if od == dom:
                    continue
                for p in pool[od][:15]:
                    s = score(model, tok, snaps, p, od, dom)
                    if s["format"]:
                        neg.append(s["adj"])
        else:
            for od in ["math", "code", "sql"]:
                for p in pool[od][:15]:
                    s = score(model, tok, snaps, p, od, "none")
                    if s["format"]:
                        neg.append(s["adj"])
        pos_min = min(pos) if pos else None
        neg_max = max(neg) if neg else None
        if pos and neg:
            t = round((pos_min + neg_max) / 2, 2)
        elif pos:
            t = round(pos_min - 0.5, 2)
        else:
            t = -9.9
        th[dom] = t
        cal[dom] = {"n": len(rows), "fmt": len(fmt_ok), "pos_min": pos_min, "neg_max": neg_max}
        print(f"  {dom}: fmt {len(fmt_ok)}/{len(rows)} | pos_min={pos_min} neg_max={neg_max} -> theta {t}")

    def verdict(prompt, mname):
        if mname != "none":
            s = score(model, tok, snaps, prompt, mname, mname)
            if s["format"] and s["adj"] >= th[mname]:
                return {"claim": mname, "abstain": False, "detail": s}
        s = score(model, tok, snaps, prompt, "story", "none")
        if s["format"] and s["adj"] >= th["story"]:
            return {"claim": "base", "abstain": False, "detail": s}
        return {"claim": None, "abstain": True, "detail": s}

    # ---------- eval 1: 40-cell grid
    doms = [("math", v5.MATH_PROMPTS), ("code", v5.CODE_PROMPTS), ("sql", v11.SQL_PROMPTS)]
    ok = tot = 0
    grid = {}
    for mname in ["none", "math", "code", "sql"]:
        row = {}
        for d, prods in doms:
            row[d] = [verdict(p, mname) for p in prods]
        row["story"] = [verdict(p, mname) for p in v5.STORY_PROMPTS]
        grid[mname] = {}
        for d, cells in row.items():
            exp_abst = not (d == mname or d == "story")
            for c in cells:
                tot += 1; ok += int(c["abstain"] == exp_abst)
            grid[mname][d] = [{"abstain": c["abstain"], "claim": c["claim"],
                               "adj": c["detail"]["adj"], "fmt": c["detail"]["format"]}
                              for c in cells]
        print(f"grid module_{mname}: " + json.dumps({d: [c['abstain'] for c in cs] for d, cs in row.items()}))
    print(f"GRID PASS {ok}/{tot}")

    # ---------- eval 2: 48 confusing cells (held-out vs calibration)
    risky = safe_none = 0
    conf_rows = []
    for label, pr in CONFUSING:
        for mname in ["none", "math", "code", "sql"]:
            v = verdict(pr, mname)
            if mname == "none":
                safe_none += int(v["abstain"])
            elif not v["abstain"]:
                risky += 1
            conf_rows.append({"prompt": label, "assembly": mname,
                              "abstain": v["abstain"], "claim": v["claim"],
                              "adj": v["detail"]["adj"], "fmt": v["detail"]["format"],
                              "text": v["detail"]["text"]})
    print(f"CONFUSING: empty-assembly abstain {safe_none}/12 | risky commits {risky}/36 (v13: 16, M4: 13)")
    for r in conf_rows:
        if not r["abstain"]:
            print(f"  answered [{r['prompt'][:34]:34s}] +{r['assembly']:4s} as {r['claim']} adj={r['adj']} :: {r['text'][:44]}")

    # ---------- determinism double-run
    r2 = [verdict(pr, "none") for _, pr in CONFUSING]
    det = json.dumps([{k: v[k] for k in ("abstain", "claim")} for v in r2]) == \
          json.dumps([{ "abstain": x["abstain"], "claim": x["claim"]} for x in conf_rows if x["assembly"] == "none"])
    print(f"DETERMINISM 2x identical: {det}")

    (v5.WS / "results/v14_results.json").write_text(json.dumps(
        {"theta": th, "calib": cal, "grid_pass": f"{ok}/{tot}", "grid": grid,
         "confusing": conf_rows, "safe_none": f"{safe_none}/12",
         "risky": f"{risky}/36", "determinism": det}, indent=2))
    print(f"V14 COMPLETE ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
