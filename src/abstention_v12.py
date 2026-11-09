"""v12 — uniform per-layer verdicts + k-of-8 majority readout.

v11 finding: the verdict lived almost entirely in layer 6 (+/-20) while L0-L3
were blind, because the loss supervised only the MEAN of the 8 layer scores —
individual layers were underdetermined, and mean-voting stayed fragile
(math-on/sql: L4,L5 negative, mean +2.4 barely cleared tau).

v12 changes exactly two things, everything else (worlds, geometry, data,
resume-from-v11) identical:
  1. LOSS: BCE on EACH layer's z separately, plus a std-across-layers penalty
     at decision slots -> every layer must independently make the right call
     (uniform confidence).
  2. READOUT: abstain iff z_i > tau in >= K of 8 layers (majority vote).
     The mean readout is kept and compared side by side.

Outputs uniformity stats (mean per-layer |z| spread before/after) so the fix
is measurable, not just asserted. Saves abstain_v12_results.json +
checkpoints/abstain_bundle_v12.pt.
"""
import sys, json, time
from pathlib import Path
import torch
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).parent))
import abstention_v5 as v5
import abstention_v9 as v9
import abstention_v9b as v9b
import abstention_v10 as v10
import abstention_v11 as v11
import abstention_v7 as v7
import train_module_experts as v3mod

DEVICE = v5.DEVICE
ABS_ID = v5.ABS_ID
PFX = v9.PFX
K_VOTE = 6          # k-of-8 majority
VAR_W = 0.1         # hinge-only uniformity weight (v12b: was 1.0 hard -> collapsed margins)
SPREAD_CAP = 3.0    # penalty only beyond 3 nats of disagreement


def layer_stack(model):
    return torch.cat([moe.last_abs_logit for moe in model.moe_layers],
                     dim=-1).squeeze(-1)          # [B,S,8]


def abst_loss12(model, inp, y, w, a):
    logits, _ = model(inp)
    vloss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                            reduction="none").view(y.shape)
    vterm = (vloss * w).sum() / w.sum().clamp_min(1.0)
    hz = layer_stack(model)                        # [B,S,8]
    pos_w = torch.where(w >= 20, 8.0, torch.where(a > 0, 8.0, 0.1))
    dec = (w >= 20)
    # per-layer BCE, each layer supervised independently
    bterm = 0.0
    for i in range(hz.shape[-1]):
        bce = F.binary_cross_entropy_with_logits(hz[..., i], a, reduction="none")
        bterm = bterm + (bce * pos_w).sum() / pos_w.sum().clamp_min(1.0)
    bterm = bterm / hz.shape[-1]
    # uniformity: std across layers AT decision slots (both classes)
    if dec.any():
        spread = hz[dec].std(dim=-1).mean()
    else:
        spread = torch.zeros((), device=DEVICE)
    # uniformity HINGE (v12b): penalize only spread beyond SPREAD_CAP so raw
    # margins stay free to grow; v12's uncapped penalty collapsed abstain/answer
    # separation to 0.55 nats by shrinking all layers toward a shared middle.
    hinge = torch.clamp(spread - SPREAD_CAP, min=0.0) if dec.any() else torch.zeros((), device=DEVICE)
    total = vterm + v5.BCE_WEIGHT * bterm + VAR_W * hinge
    return total, vterm.item(), bterm.item(), float(spread)


def train12(model, mixes, bundles):
    print("\n" + "=" * 60)
    print("TRAINING v12 (per-layer BCE + uniformity; 4 worlds)")
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
    model.train()
    t0 = time.time()
    sp_hist = []
    for step in range(v5.AB_STEPS):
        phase = (step // v5.SWAP_EVERY) % 4
        v5.set_domain_module(model, snaps[phase])
        opt.zero_grad()
        av = ab = asp = 0.0
        for _ in range(v5.AB_GRAD_ACCUM):
            inp, y, w, a = mixes[phase].batch()
            l, vl, bl, sp = abst_loss12(model, inp, y, w, a)
            (l / v5.AB_GRAD_ACCUM).backward()
            av += vl; ab += bl; asp += sp
        sp_hist.append(asp / v5.AB_GRAD_ACCUM)
        torch.nn.utils.clip_grad_norm_(
            params_mod + params_gate + params_head + [model.embed.weight], 1.0)
        opt.step()
        if (step + 1) % 50 == 0:
            print(f"  step {step+1}/{v5.AB_STEPS} | world {names[phase]} | "
                  f"vocab {av/v5.AB_GRAD_ACCUM:.3f} | bce {ab/v5.AB_GRAD_ACCUM:.4f} | "
                  f"spread {asp/v5.AB_GRAD_ACCUM:.2f} | {time.time()-t0:.0f}s")
    return sp_hist


def main():
    print("V12: uniform per-layer verdicts + majority readout")
    v9b.RowPacker.ABST_MAX = 48
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
    print("RESUMED from abstain_bundle_v11.pt (tau_v11=%s)" % resume["tau"])

    # pre-fix spread on train worlds (baseline uniformity metric)
    packs = v11.worlds11(tok, 'train')
    v5.LR_AB = 1e-4
    v5.AB_STEPS = 400
    sp_hist = train12(model, packs, bundles)
    print(f"SPREAD first50={sum(sp_hist[:50])/50:.2f} last50={sum(sp_hist[-50:])/50:.2f}")

    t_off, t_math, t_code, t_sql = v11.worlds11(tok, 'test')

    def grab2(mix, snap, n=20):
        v5.set_domain_module(model, snap)
        dec, ans = [], []
        model.eval()
        for _ in range(n):
            x, y, w, a = mix.batch()
            with torch.no_grad():
                model(x)
            hz = layer_stack(model)
            hi_w = w >= 20
            dec += hz[hi_w & (a > 0.5)].mean(-1).tolist()
            ans += hz[hi_w & (a == 0)].mean(-1).tolist()
        return dec, ans

    dA, aA = grab2(t_off, None)
    dM, aM = grab2(t_math, bundles['math'])
    dC, aC = grab2(t_code, bundles['code'])
    dS, aS = grab2(t_sql, bundles['sql'])
    st_a, st_ans = v7.stats(dA + dM + dC + dS), v7.stats(aA + aM + aC + aS)
    tau = round((st_a["p05"] + st_ans["p95"]) / 2, 2)
    print(f"DIAG12 abstain={st_a} answer={st_ans} tau={tau}")

    def probe(prompt, snap, prefix=PFX):
        v5.set_domain_module(model, snap)
        ids = tok(prompt + prefix, add_special_tokens=False)['input_ids']
        xx = torch.tensor([ids], device=DEVICE)
        model.eval()
        with torch.no_grad():
            model(xx)
        hz = layer_stack(model)[0, -1]            # [8]
        votes = int((hz > tau).sum())
        return {"z_layers": [round(float(v), 2) for v in hz],
                "spread": round(float(hz.std()), 2),
                "z_mean": round(float(hz.mean()), 2),
                "votes": votes,
                "abstain_mean": bool(hz.mean() > tau),
                "abstain_vote": bool(votes >= K_VOTE)}

    doms = [("math", v5.MATH_PROMPTS), ("code", v5.CODE_PROMPTS), ("sql", v11.SQL_PROMPTS)]
    grid, ok_m, ok_v, tot = {}, 0, 0, 0
    for mname, snap in [("none", None), ("math", bundles["math"]),
                        ("code", bundles["code"]), ("sql", bundles["sql"])]:
        grid[f"module_{mname}"] = {d: [probe(p, snap) for p in prods]
                                   for d, prods in doms}
        grid[f"module_{mname}"]["story"] = [probe(p, snap, prefix="\n")
                                            for p in v5.STORY_PROMPTS]
        for d, cells in grid[f"module_{mname}"].items():
            for c in cells:
                exp = not (d == mname or d == "story")
                tot += 1; ok_m += int(c["abstain_mean"] == exp); ok_v += int(c["abstain_vote"] == exp)
        ab = {d: [c["votes"] for c in cells] for d, cells in grid[f"module_{mname}"].items()}
        sp = {d: [c["spread"] for c in cells] for d, cells in grid[f"module_{mname}"].items()}
        print(f"module={mname}: votes {json.dumps(ab)}\n   spread {json.dumps(sp)}")
    print(f"GRID12 mean-readout {ok_m}/{tot} | VOTE readout {ok_v}/{tot}")

    gens = {}
    for name, snap, pr, pre in [
            ("math_no_module", None, v5.MATH_PROMPTS[0], PFX),
            ("sql_wrong_module", bundles['math'], v11.SQL_PROMPTS[0], PFX),
            ("sql_with_module", bundles['sql'], v11.SQL_PROMPTS[0], PFX),
            ("story_no_module", None, v5.STORY_PROMPTS[0], "\n")]:
        p = probe(pr, snap, prefix=pre)
        out = pr + " ⟦ABSTAINED⟧" if p["abstain_vote"] else \
            pr + v3mod.nucleus(model, tok, pr, max_new=80, temp=0.8, top_p=0.9)
        gens[name] = out
        print(f"{name} votes={p['votes']}: {out[:130]}")

    torch.save({"abst": [moe.abst.state_dict() for moe in model.moe_layers],
                "abs_heads": [moe.abst_head.state_dict() for moe in model.moe_layers],
                "abs_emb_row": model.embed.weight[ABS_ID].detach().cpu(),
                "tau": tau, "k_vote": K_VOTE},
               v5.WS / "checkpoints/abstain_bundle_v12b.pt")
    (v5.WS / "abstain_v12b_results.json").write_text(json.dumps(
        {"diag": {"abstain": st_a, "answer": st_ans, "tau": tau},
         "spread_curve": {"first50": sp_hist[0], "last50": sp_hist[-1]},
         "grid": grid, "grid_mean": f"{ok_m}/{tot}", "grid_vote": f"{ok_v}/{tot}",
         "generations": gens}, indent=2))
    print("V12 COMPLETE")


if __name__ == "__main__":
    main()
