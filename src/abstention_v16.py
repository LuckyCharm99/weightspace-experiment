"""v16 — nine-expert gate: 10-class circuit, grid, extended confusing suite.

Domains: math, code, sql, chess, chem, regex, med, units, music (+general/story).
Circuit: nn.Linear(512, 10) on final-norm h at the decision slot, trained on
cached h collected across all 10 module worlds. Margin chosen from HELD-OUT
true-positive margin distribution (print p1; grid JSON keeps every margin so
the shipped M can be recomputed offline without re-running).
"""
import sys, json, time, hashlib, itertools
from pathlib import Path
import torch
import torch.nn as nn
sys.path.insert(0, str(Path(__file__).parent))
import abstention_v5 as v5
import abstention_v9 as v9
import abstention_v10 as v10
import abstention_v11 as v11
import train_batch1_modules as b1
import train_batch2_modules as b2

DEVICE = v5.DEVICE
PFX = v9.PFX
DOMS = ["general", "math", "code", "sql", "chess", "chem", "regex",
        "med", "units", "music"]
D_ID = {d: i for i, d in enumerate(DOMS)}
MARK = {"math": PFX, "code": PFX, "sql": PFX, "chess": "\nBest move: ",
        "chem": "\nSMILES:\n", "regex": "\nRegex:\n", "med": "\nAnswer:\n",
        "units": "\nAnswer:\n", "music": "\nScore:\n"}
EXPERTS = DOMS[1:]
N_POOL = 250

MED_PROMPTS = [
    "Question: Which vitamin is supplied only from animal sources?\nA) Vitamin C\nB) Vitamin B12\nC) Vitamin D\nD) Vitamin A\nAnswer:\n",
    "Question: Chronic urethral obstruction leads to which kidney change?\nA) Hyperplasia\nB) Hypertrophy\nC) Atrophy\nD) Dysplasia\nAnswer:\n",
    "Question: Kupffer cells of the adult liver are derived from?\nA) Endoderm\nB) Mesoderm\nC) Ectoderm\nD) Neuroectoderm\nAnswer:\n"]
UNITS_PROMPTS = [
    "Convert 3 kilometers to meters.\nAnswer:\n",
    "How many centimeters is 0.25 meters?\nAnswer:\n",
    "Convert 2 hours to seconds.\nAnswer:\n"]
MUSIC_PROMPTS = [
    "Write the ABC notation for a simple tune in G major, 4/4.\nScore:\n",
    "What does the ABC field 'M:3/4' specify?\nScore:\n",
    "Compose an ABC fragment with a repeat sign in D minor.\nScore:\n"]

CONFUSING3 = [
    ("units dressed as math", "Problem: A car covers 5 meters per second for 90 meters. Find x.\nAnswer:\n"),
    ("bare Answer: slot", "Answer:\n"),
    ("med prose story", "The doctor explained to the patient that vitamins matter, and the patient listened carefully.\n"),
    ("chess MCQ", "Question: What is the best move in this position?\nA) e4\nB) Re8#\nC) O-O\nD) Qxd5\nAnswer:\n"),
    ("regex asks units", "Description: convert and match kilometers per hour\nRegex:\n"),
    ("ABC as SMILES", "Formula: X:1\nT:Reel\nSMILES:\n"),
    ("music MCQ", "Question: Which field sets the key signature in ABC notation?\nA) M:\nB) K:\nC) L:\nD) T:\nAnswer:\n"),
    ("units bare numbers", "3 km = "),
    ("medical FEN hybrid", "FEN: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1\nDiagnosis:\n"),
]
OLD_LABELS = [  # (label, prompt) — 12 old + 8 v15 traps, re-run against 9-expert library
]


import abstention_v15 as v15mod
from test_v13_confusing import CONFUSING as OLD12


def display_for(dom, full):
    mk = MARK[dom]
    if mk in full:
        return full.split(mk)[0] + mk
    return full


def collect9(model, tok, lib):
    cache = v5.WS / "checkpoints/v16_hcache.pt"
    if cache.exists():
        c = torch.load(cache, weights_only=False)
        print("using cached v16 collection")
        return c["Xtr"], c["Ytr"], c["Xte"], c["Yte"]
    caps = {}
    model.norm.register_forward_hook(lambda m, i, o: caps.__setitem__("h", i[0]))
    is2 = lambda g: [x for x in itertools.islice(g, N_POOL * 2)]
    pools = {}
    pools["math"] = [d if PFX.strip() in d else d + PFX for d, _ in is2(v9.math_pairs("train"))]
    pools["code"] = [d if PFX.strip() in d else d + PFX for d, _ in is2(v10.code_pairs_stub("train"))]
    pools["sql"] = [display_for("sql", d) for d, _ in is2(v11.sql_pairs("train"))]
    for d in ("chess", "chem", "regex"):
        pools[d] = [display_for(d, t) for t in is2(b1.texts(d, "train"))]
    for d in ("med", "units", "music"):
        pools[d] = [display_for(d, t) for t in is2(b2.texts(d, "train"))]
    st = v5.story_stream(909)
    pools["general"] = []
    while len(pools["general"]) < N_POOL * 2:
        ids = tok(next(st), add_special_tokens=False)["input_ids"][:40]
        if len(ids) >= 20:
            pools["general"].append(tok.decode(ids) + "\n")
    worlds = [("none", None)] + [(d, lib[d]) for d in EXPERTS]
    Xtr, Ytr, Xte, Yte = [], [], [], []
    t0 = time.time()
    for wname, snap in worlds:
        v5.set_domain_module(model, snap)
        for dom, texts in pools.items():
            for i, txt in enumerate(texts[:N_POOL]):
                ids = tok(txt, add_special_tokens=False)["input_ids"][:96]
                with torch.no_grad():
                    model(torch.tensor([ids], device=DEVICE))
                h = caps["h"][0, -1].float().cpu()
                y = D_ID[dom]
                (Xte if i % 5 == 4 else Xtr).append(h)
                (Yte if i % 5 == 4 else Ytr).append(y)
        print(f"  collected world {wname} | {time.time()-t0:.0f}s", flush=True)
    out = dict(Xtr=torch.stack(Xtr), Ytr=torch.tensor(Ytr),
               Xte=torch.stack(Xte), Yte=torch.tensor(Yte))
    torch.save(out, cache)
    return out["Xtr"], out["Ytr"], out["Xte"], out["Yte"]


def main():
    print("V16: nine-expert gate")
    model, mcfg, tok = v5.build_model()
    v5.freeze_for_abstention(model)
    lib = torch.load(v5.WS / "checkpoints/module_bundles_merged.pt",
                     map_location=DEVICE, weights_only=False)
    for f in ("lib10_chess", "lib10_chem", "lib10_regex",
              "lib2_med", "lib2_units", "lib2_music"):
        lib.update(torch.load(v5.WS / f"checkpoints/{f}.pt",
                              map_location=DEVICE, weights_only=False))
    snaps = {"none": None}
    snaps.update(lib)
    Xtr, Ytr, Xte, Yte = collect9(model, tok, lib)
    D = Xtr.shape[1]
    head = nn.Linear(D, len(DOMS)).to(DEVICE)
    Xtr, Ytr, Xte, Yte = Xtr.to(DEVICE), Ytr.to(DEVICE), Xte.to(DEVICE), Yte.to(DEVICE)
    opt = torch.optim.AdamW(head.parameters(), lr=3e-3, weight_decay=0.0)
    for ep in range(1000):
        perm = torch.randperm(Xtr.shape[0], device=DEVICE)
        for j in range(0, Xtr.shape[0], 256):
            idx = perm[j:j+256]
            opt.zero_grad()
            nn.functional.cross_entropy(head(Xtr[idx]), Ytr[idx]).backward()
            opt.step()
        if (ep + 1) % 250 == 0:
            print(f"  head ep {ep+1} | heldout {(head(Xte).argmax(1)==Yte).float().mean():.4f}", flush=True)
    acc = (head(Xte).argmax(1) == Yte).float().mean().item()
    digest = hashlib.sha256(head.weight.detach().cpu().numpy().tobytes()).hexdigest()[:16]
    # margin calibration from held-out only
    lg = head(Xte)
    top2 = torch.topk(lg, 2, dim=1)
    pred, marg, corr = top2.indices[:, 0], (top2.values[:, 0] - top2.values[:, 1]), top2.indices[:, 0] == Yte
    dom_mask = (Yte != 0) & (pred != 0)
    mt = marg[dom_mask & corr].detach().cpu().numpy()
    import numpy as np
    p1 = float(np.percentile(mt, 1))
    print(f"CIRCUIT9 {D}x{len(DOMS)} heldout {acc:.4f} sha {digest} | TP margin p1 {p1:.2f} med {np.median(mt):.2f}")

    caps_store = {}
    if not hasattr(model, "_v16_caps"):
        model._v16_caps = caps_store
        model.norm.register_forward_hook(lambda m, i, o: caps_store.__setitem__("h", i[0]))
    caps = model._v16_caps

    def verdict(prompt, mname, suffix):
        v5.set_domain_module(model, snaps.get(mname))
        ids = tok(prompt + suffix, add_special_tokens=False)["input_ids"][:96]
        with torch.no_grad():
            model(torch.tensor([ids], device=DEVICE))
        lg = head(caps["h"][0, -1].unsqueeze(0).to(DEVICE))[0]
        t2 = torch.topk(lg, 2)
        needed = DOMS[int(t2.indices[0])]
        margin = float((t2.values[0] - t2.values[1]).detach())
        return {"needed": needed, "margin": round(margin, 2),
                "attached": mname != "none" and mname == needed}

    grid_defs = [("math", v5.MATH_PROMPTS, PFX), ("code", v5.CODE_PROMPTS, PFX),
                 ("sql", v11.SQL_PROMPTS, PFX), ("chess", v15mod.CHESS_PROMPTS, "\nBest move: "),
                 ("chem", v15mod.CHEM_PROMPTS, "\nSMILES:\n"),
                 ("regex", v15mod.REGEX_PROMPTS, "\nRegex:\n"),
                 ("med", MED_PROMPTS, ""), ("units", UNITS_PROMPTS, ""),
                 ("music", MUSIC_PROMPTS, ""), ("story", v5.STORY_PROMPTS, "\n")]
    assemblies = ["none"] + EXPERTS

    def grid():
        g, rows = {}, 0
        for a in assemblies:
            row = {}
            for d, prods, suf in grid_defs:
                row[d] = [verdict(p, a, suf) for p in prods]
                rows += len(row[d])
            g[a] = row
        return g

    g1, g2, g3 = grid(), grid(), grid()
    det = json.dumps(g1, sort_keys=True) == json.dumps(g2, sort_keys=True) == json.dumps(g3, sort_keys=True)
    for M in [0, p1, 2.5, 4]:
        ok = tot = risky = 0
        for a, row in g1.items():
            for d, cells in row.items():
                for c in cells:
                    exp_ab = not (d == a or d == "story")
                    commit = c["needed"] == "general" or (c["attached"] and c["margin"] >= M)
                    ok += int((not commit) == exp_ab)
                    tot += 1
                    if commit and d != a and d != "story":
                        risky += 1
        print(f"GRID(M={M:.2f}) {ok}/{tot} risky {risky}")
    print(f"determinism 3x: {det}")

    crows, safe, risky = [], 0, 0
    for label, pr in OLD12:
        c = verdict(pr, "none", PFX)
        safe += int(c["needed"] != "general")
        crows.append({"suite": "v13-old", "prompt": label, "assembly": "none", **c})
    for label, pr in v15mod.CONFUSING2:
        c = verdict(pr, "none", PFX)
        safe += int(c["needed"] != "general")
        crows.append({"suite": "v15", "prompt": label, "assembly": "none", **c})
    nprobes = 0
    for label, pr in CONFUSING3:
        for a in assemblies:
            c = verdict(pr, a, PFX)
            commit = c["needed"] == "general" or (c["attached"] and c["margin"] >= p1)
            nprobes += 1
            if a == "none":
                safe += int(not commit)
            else:
                risky += int(commit)
            crows.append({"suite": "v16", "prompt": label, "assembly": a, "commit": commit, **c})
    print(f"CONFUSING: empty-assembly abstain {safe}/{12+8+len(CONFUSING3)} | risky {risky} across {nprobes} probed cells")

    torch.save({"head": head.state_dict(), "digest": digest, "acc": acc, "D": D,
                "tp_margin_p1": p1}, v5.WS / "checkpoints/domain_head_v16.pt")
    (v5.WS / "results/v16_results.json").write_text(json.dumps(
        {"heldout_acc": acc, "circuit_sha": digest, "tp_margin_p1": p1,
         "determinism": det, "grid": g1, "confusing": crows}, indent=2))
    print("V16 COMPLETE")


if __name__ == "__main__":
    main()
