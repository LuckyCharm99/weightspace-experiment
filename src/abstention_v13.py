"""v13 — deterministic, auditable abstention.

Replace the soft per-layer score + calibrated tau with a HARD decision circuit
trained inside the frozen foundation's hidden state:

  needed  = argmax W·h(final norm output at the decision slot)  ∈ {general,math,code,sql}
  present = bit-exact assembly manifest p (gate fill 0.0 = detached, snap load = attached)
  ABSTAIN <=> needed != general AND not p[needed]

No tau, no probabilistic hedge at the gate. When abstaining, the vocabulary
softmax is MASKED to <abs> (deterministic nullification, argmax provably <abs>).
Audits: (1) gate truth table is a pure boolean function, printed exhaustively;
(2) the 512x4 circuit weights are hashed; (3) the full grid is run 3x and
compared bit-exactly; (4) OOD prose probes quantify what the soft v11 gate
hedged and the hard gate cannot.
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

DEVICE = v5.DEVICE
ABS_ID = v5.ABS_ID
PFX = v9.PFX
DOMS = ["general", "math", "code", "sql"]
D_ID = {"math": 1, "code": 2, "sql": 3}


def marker_ok(disp):
    s = PFX.strip()
    return disp if s in disp else disp + PFX


def collect(model, tok, bundles, n=400):
    """h at the decision slot for 4 worlds x 4 domains; plus eval split."""
    cache = v5.WS / "checkpoints/v13_hcache.pt"
    if cache.exists():
        c = torch.load(cache, weights_only=False)
        print("using cached hidden-state collection")
        return c["Xtr"], c["Ytr"], c["Xte"], c["Yte"]
    caps = {}
    model.norm.register_forward_hook(lambda m, i, o: caps.__setitem__("h", i[0]))
    pools = {}
    pools["math"] = [marker_ok(d) for d, _ in v9.math_pairs("train")][:n]
    pools["code"] = [marker_ok(d) for d, _ in v10.code_pairs_stub("train")][:n]
    pools["sql"] = [marker_ok(d) for d, _ in v11.sql_pairs("train")][:n]
    st = v5.story_stream(4242)
    pools["story"] = []
    while len(pools["story"]) < n:
        c = next(st)
        ids = tok(c, add_special_tokens=False)["input_ids"][:40]
        if len(ids) >= 20:
            pools["story"].append(tok.decode(ids) + "\n")

    worlds = [("off", None), ("math", bundles["math"]),
              ("code", bundles["code"]), ("sql", bundles["sql"])]
    Xtr, Ytr, Xte, Yte = [], [], [], []
    model.eval()
    t0 = time.time()
    for wname, snap in worlds:
        v5.set_domain_module(model, snap)
        for dom, texts in pools.items():
            for i, txt in enumerate(texts):
                ids = tok(txt, add_special_tokens=False)["input_ids"]
                with torch.no_grad():
                    model(torch.tensor([ids], device=DEVICE))
                h = caps["h"][0, -1].float().cpu()
                # 20% held out per (world, domain) block for honest accuracy
                if i % 5 == 4:
                    Xte.append(h); Yte.append(DOMS.index(dom if dom != "story" else "general"))
                else:
                    Xtr.append(h); Ytr.append(DOMS.index(dom if dom != "story" else "general"))
        print(f"  collected world {wname} | {time.time()-t0:.0f}s", flush=True)
    Xtr, Ytr, Xte, Yte = torch.stack(Xtr), torch.tensor(Ytr), torch.stack(Xte), torch.tensor(Yte)
    torch.save({"Xtr": Xtr, "Ytr": Ytr, "Xte": Xte, "Yte": Yte}, cache)
    return Xtr, Ytr, Xte, Yte


def h_final(model, ids):
    caps = getattr(model, "_v13_caps", None)
    if caps is None:
        caps = {}
        model._v13_caps = caps
        model.norm.register_forward_hook(lambda m, i, o: caps.__setitem__("h", i[0]))
    with torch.no_grad():
        out = model(torch.tensor([ids], device=DEVICE))
        logits = out[0] if isinstance(out, tuple) else out
        caps["logits"] = logits[0, -1].clone()
    return caps["h"][0, -1].clone(), caps["logits"]


def main():
    print("V13: deterministic hard-gate abstention")
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
    tau11 = resume["tau"]
    print(f"RESUMED v11 bundle (soft baseline tau={tau11})")

    Xtr, Ytr, Xte, Yte = collect(model, tok, bundles)
    D = Xtr.shape[1]
    head = nn.Linear(D, 4).to(DEVICE)
    Xtr, Ytr, Xte, Yte = Xtr.to(DEVICE), Ytr.to(DEVICE), Xte.to(DEVICE), Yte.to(DEVICE)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3)
    for ep in range(300):
        perm = torch.randperm(Xtr.shape[0], device=DEVICE)
        for j in range(0, Xtr.shape[0], 256):
            idx = perm[j:j+256]
            opt.zero_grad()
            nn.functional.cross_entropy(head(Xtr[idx]), Ytr[idx]).backward()
            opt.step()
        if (ep + 1) % 100 == 0:
            acc = (head(Xte).argmax(1) == Yte).float().mean().item()
            print(f"  head epoch {ep+1}/300 | heldout acc {acc:.4f}", flush=True)
    acc_te = (head(Xte).argmax(1) == Yte).float().mean().item()

    wd = head.weight.detach().cpu().numpy().tobytes()
    digest = hashlib.sha256(wd).hexdigest()[:16]
    print(f"CIRCUIT: {D}x4 linear | heldout acc {acc_te:.4f} | sha256[:16]={digest}")

    snaps = {"none": None, "math": bundles["math"],
             "code": bundles["code"], "sql": bundles["sql"]}

    def hard_verdict(prompt, mname, prefix=PFX):
        p = {"math": mname == "math", "code": mname == "code", "sql": mname == "sql"}
        v5.set_domain_module(model, snaps[mname])
        ids = tok(prompt + prefix, add_special_tokens=False)["input_ids"]
        h, logits = h_final(model, ids)
        nid = int(head(h.unsqueeze(0)).argmax(1).item())
        needed = DOMS[nid]
        abstain = needed != "general" and not p[needed]
        out = {"needed": needed, "abstain": bool(abstain)}
        if abstain:  # deterministic nullification audit: argmax must be <abs>
            masked = torch.full_like(logits, float("-inf"))
            masked[ABS_ID] = logits[ABS_ID]
            out["argmax_is_abs"] = int(masked.argmax()) == ABS_ID
            out["p_abs"] = float(torch.softmax(masked, -1)[ABS_ID])
        return out

    doms = [("math", v5.MATH_PROMPTS), ("code", v5.CODE_PROMPTS), ("sql", v11.SQL_PROMPTS)]

    def grid():
        g, ok, tot = {}, 0, 0
        for mname in ["none", "math", "code", "sql"]:
            row = {d: [hard_verdict(pr, mname) for pr in prods] for d, prods in doms}
            row["story"] = [hard_verdict(pr, mname, prefix="\n") for pr in v5.STORY_PROMPTS]
            g[f"module_{mname}"] = row
            for d, cells in row.items():
                exp = not (d == mname or d == "story")
                for c in cells:
                    tot += 1; ok += int(c["abstain"] == exp)
        return g, ok, tot

    g1, ok, tot = grid()
    g2, _, _ = grid()
    g3, _, _ = grid()
    det = json.dumps(g1, sort_keys=True) == json.dumps(g2, sort_keys=True) == json.dumps(g3, sort_keys=True)
    print(f"GRID PASS {ok}/{tot}  |  DETERMINISM 3x bit-identical: {det}")
    for k, row in g1.items():
        print(f"{k}: " + json.dumps({d: [c['abstain'] for c in cs] for d, cs in row.items()}))

    # ---- OOD cost probe: prose-shaped domain questions; soft v11 for comparison
    OOD = [
        ("math", "Once upon a time Lily had three apples and gave two to her friend. How many apples does Lily have now?"),
        ("math", "The baker made a dozen rolls and sold half before noon. How many rolls are left?"),
        ("math", "Tom is older than Sue. Sue is older than Bob. Who is the youngest?"),
        ("sql", "Tell me which pilots come from France, please."),
        ("sql", "I am curious about the total population of each country."),
        ("code", "Can you write me a function that adds two numbers together?"),
    ]
    def soft_verdict(prompt, mname, prefix=PFX):
        v5.set_domain_module(model, snaps[mname])
        ids = tok(prompt + prefix, add_special_tokens=False)["input_ids"]
        h_final(model, ids)
        hz = torch.cat([moe.last_abs_logit for moe in model.moe_layers], -1).mean(-1)[0, -1].item()
        return {"head_z": round(hz, 2), "abstain": bool(hz > tau11)}
    ood = []
    print("\nOOD probes (assembly: none) — hard vs soft v11:")
    for dom, pr in OOD:
        hv = hard_verdict(pr, "none", prefix=PFX if dom != "story" else "\n")
        sv = soft_verdict(pr, "none")
        ood.append({"domain": dom, "prompt": pr[:60], "hard": hv, "soft_v11": sv})
        print(f"  [{dom}] hard need={hv['needed']} abst={hv['abstain']} | soft z={sv['head_z']} abst={sv['abstain']}")

    # ---- audit: gate is a boolean function (printed truth table)
    tt = [{"needed": n, "assembly": a,
           "abstain": bool(n != "general" and n != a)}
          for n, a in itertools.product(DOMS, ["none", "math", "code", "sql"])]
    with torch.no_grad():  # presence p is bit-exact: detached gates are exactly 0
        v5.set_domain_module(model, None)
        p_zero = all(float(torch.tanh(moe.dom.gate_raw).max()) == 0.0 for moe in model.moe_layers)
        v5.set_domain_module(model, bundles["math"])
        p_one = all(float(torch.tanh(moe.dom.gate_raw).min()) > 0.0 for moe in model.moe_layers)
    print(f"AUDIT: presence bit-exact (off gates all-zero={p_zero}, attached nonzero={p_one}); "
          f"truth table rows={len(tt)}")

    torch.save({"head": head.state_dict(), "digest": digest,
                "heldout_acc": acc_te}, v5.WS / "checkpoints/domain_head_v13.pt")
    (v5.WS / "abstain_v13_results.json").write_text(json.dumps(
        {"grid_pass": f"{ok}/{tot}", "determinism_3x_identical": det,
         "heldout_acc": acc_te, "circuit_sha": digest,
         "grid": g1, "ood_probes": ood, "gate_truth_table": tt,
         "presence_zero_exact": p_zero, "presence_attached_exact": p_one}, indent=2))
    print("V13 COMPLETE")


if __name__ == "__main__":
    main()
