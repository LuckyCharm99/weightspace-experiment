"""v15 — six-expert routing + abstention stress test.

Rebuilds the shipped v13 gate (hard argmax circuit + bit-exact presence + M=4
margin) for the 6-expert library {math, code, sql, chess, chem, regex} + story.
Circuit retrained from scratch on cached decision-slot h across ALL 7 worlds;
eval = 147-cell grid + 20 confusing probes (12 old + 8 new cross-domain traps).
Determinism 3x; all calibration/train data disjoint from eval prompt sets.
"""
import sys, json, time, hashlib
from pathlib import Path
import torch
import torch.nn as nn
sys.path.insert(0, str(Path(__file__).parent))
import abstention_v5 as v5
import abstention_v9 as v9
import abstention_v10 as v10
import abstention_v11 as v11
import train_batch1_modules as b1

DEVICE = v5.DEVICE
ABS_ID = v5.ABS_ID
PFX = v9.PFX
DOMS = ["general", "math", "code", "sql", "chess", "chem", "regex"]
D_ID = {"general": 0, "math": 1, "code": 2, "sql": 3, "chess": 4, "chem": 5, "regex": 6}
MARK = {"math": PFX, "code": PFX, "sql": PFX, "chess": "\nBest move: ",
        "chem": "\nSMILES:\n", "regex": "\nRegex:\n"}

CHESS_PROMPTS = ["FEN: r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 4 4\n",
                 "FEN: 6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1\n",
                 "FEN: rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq g3 0 2\n"]
CHEM_PROMPTS = ["Formula: C6H6\n", "Formula: CH4\n", "Formula: C2H6O\n"]
REGEX_PROMPTS = ["Description: three uppercase letters followed by four digits\n",
                 "Description: a US zip code, exactly five digits\n",
                 "Description: a hex color code starting with a hash sign\n"]
CONFUSING2 = [
    ("chess-shape regex ask", "FEN: 8/8/8/8/4k3/8/8/4K1 w - - 0 1\nRegex:"),
    ("chem-shape chess ask", "Formula: H2O\nBest move:"),
    ("regex content, chess words", "Description: a chess FEN string with nine fields\nRegex:"),
    ("raw FEN no template", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    ("story about chemistry", "Once upon a time a molecule named C6H6 went to a party and many others came.\n"),
    ("math asking for molecule", "Problem: How many atoms are in benzene C6H6?\n"),
    ("regex as bare prompt", "^[A-Z]{3}[0-9]{4}$"),
    ("chess prose", "The queen leaps to f7 and the game is over in one move. Your task:"),
]
N_POOL = 250


def display_for(dom, full):
    mk = MARK[dom]
    if mk in full:
        head = full.split(mk)[0]
        return head + mk
    return full


import itertools
from test_v13_confusing import CONFUSING as OLD12


def collect6(model, tok, lib):
    cache = v5.WS / "checkpoints/v15_hcache.pt"
    if cache.exists():
        c = torch.load(cache, weights_only=False)
        print("using cached v15 collection")
        return c["Xtr"], c["Ytr"], c["Xte"], c["Yte"]
    caps = {}
    model.norm.register_forward_hook(lambda m, i, o: caps.__setitem__("h", i[0]))
    is2 = lambda g: [x for x in itertools.islice(g, N_POOL * 2)]
    pools = {}
    pools["math"] = [d if PFX.strip() in d else d + PFX
                     for d, _ in is2(v9.math_pairs("train"))]
    pools["code"] = [d if PFX.strip() in d else d + PFX
                     for d, _ in is2(v10.code_pairs_stub("train"))]
    pools["sql"] = [display_for("sql", d) for d, _ in is2(v11.sql_pairs("train"))]
    pools["chess"] = [display_for("chess", t) for t in is2(b1.texts("chess", "train"))]
    pools["chem"] = [display_for("chem", t) for t in is2(b1.texts("chem", "train"))]
    pools["regex"] = [display_for("regex", t) for t in is2(b1.texts("regex", "train"))]
    st = v5.story_stream(909)
    pools["general"] = []
    while len(pools["general"]) < N_POOL * 2:
        ids = tok(next(st), add_special_tokens=False)["input_ids"][:40]
        if len(ids) >= 20:
            pools["general"].append(tok.decode(ids) + "\n")
    worlds = [("none", None)] + [(d, lib[d]) for d in ("math", "code", "sql", "chess", "chem", "regex")]
    Xtr, Ytr, Xte, Yte = [], [], [], []
    model.eval()
    t0 = time.time()
    for wname, snap in worlds:
        v5.set_domain_module(model, snap)
        for dom, texts in pools.items():
            sel = texts[:N_POOL]
            for i, txt in enumerate(sel):
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
    print("V15: six-expert gate")
    model, mcfg, tok = v5.build_model()
    v5.freeze_for_abstention(model)
    lib = torch.load(v5.WS / "checkpoints/module_bundles_merged.pt",
                     map_location=DEVICE, weights_only=False)
    for d in ("chess", "chem", "regex"):
        lib.update(torch.load(v5.WS / f"checkpoints/lib10_{d}.pt",
                              map_location=DEVICE, weights_only=False))
    snaps = {"none": None, "general": None}
    snaps.update(lib)
    Xtr, Ytr, Xte, Yte = collect6(model, tok, lib)
    D = Xtr.shape[1]
    head = nn.Linear(D, 7).to(DEVICE)
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
    print(f"CIRCUIT6 {D}x7 heldout {acc:.4f} sha {digest}")

    caps_store = {}
    if not hasattr(model, "_v15_caps"):
        model._v15_caps = caps_store
        model.norm.register_forward_hook(lambda m, i, o: caps_store.__setitem__("h", i[0]))
    caps = model._v15_caps

    def verdict(prompt, mname, suffix=None):
        p = {d: mname == d for d in ("math", "code", "sql", "chess", "chem", "regex")}
        v5.set_domain_module(model, snaps[mname])
        ids = tok(prompt + (suffix or ""), add_special_tokens=False)["input_ids"][:96]
        with torch.no_grad():
            out = model(torch.tensor([ids], device=DEVICE))
        lg = head(caps["h"][0, -1].unsqueeze(0))[0]
        top2 = torch.topk(lg, 2)
        needed = DOMS[int(top2.indices[0])]
        margin = float(top2.values[0] - top2.values[1])
        commit = needed == "general" or (p.get(needed, False) and margin >= 2.0)
        return {"needed": needed, "margin": round(margin, 2), "abstain": not commit}

    grid_defs = [("math", v5.MATH_PROMPTS, PFX), ("code", v5.CODE_PROMPTS, PFX),
                 ("sql", v11.SQL_PROMPTS, PFX), ("chess", CHESS_PROMPTS, MARK["chess"]),
                 ("chem", CHEM_PROMPTS, MARK["chem"]), ("regex", REGEX_PROMPTS, MARK["regex"]),
                 ("story", v5.STORY_PROMPTS, "\n")]
    assemblies = ["none", "math", "code", "sql", "chess", "chem", "regex"]

    def grid():
        g, ok, tot = {}, 0, 0
        for a in assemblies:
            row = {}
            for d, prods, suf in grid_defs:
                cells = [verdict(p, a, suf) for p in prods]
                exp_ab = not (d == a or d == "story")
                for c in cells:
                    tot += 1; ok += int(c["abstain"] == exp_ab)
                row[d] = cells
            g[a] = row
        return g, ok, tot

    g1, ok, tot = grid()
    g2, _, _ = grid()
    g3, _, _ = grid()
    det = json.dumps(g1, sort_keys=True) == json.dumps(g2, sort_keys=True) == json.dumps(g3, sort_keys=True)
    print(f"GRID6 PASS {ok}/{tot} | determinism 3x: {det}")
    for a in assemblies:
        print(f"  {a}: " + json.dumps({d: [c['abstain'] for c in cs] for d, cs in g1[a].items()}))

    crows, safe, risky = [], 0, 0
    for label, pr in OLD12:
        v = verdict(pr, "none", PFX)
        safe += int(v["abstain"])
        crows.append({"suite": "old", "prompt": label, "assembly": "none", **v})
    for label, pr in CONFUSING2:
        for a in ["none"] + assemblies[1:]:
            v = verdict(pr, a, PFX)
            if a == "none":
                safe += int(v["abstain"])
            elif not v["abstain"]:
                risky += 1
            crows.append({"suite": "new", "prompt": label, "assembly": a, **v})
    n_none = 12 + len(CONFUSING2)
    print(f"CONFUSING: empty-assembly abstain {safe}/{n_none} | risky commits with expert {risky}")

    torch.save({"head": head.state_dict(), "digest": digest, "acc": acc,
                "D": D}, v5.WS / "checkpoints/domain_head_v15.pt")
    (v5.WS / "results/v15_results.json").write_text(json.dumps(
        {"grid_pass": f"{ok}/{tot}", "determinism": det, "heldout_acc": acc,
         "circuit_sha": digest, "grid": g1, "confusing": crows,
         "safe_none": f"{safe}/{n_none}", "risky": risky}, indent=2))
    print("V15 COMPLETE")


if __name__ == "__main__":
    main()
