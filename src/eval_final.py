"""Full evaluation of the FINAL model (v11) with internal monitoring.

For held-out prompts in math/code/sql/story, under each module state
(none/math/code/sql), one forward pass per prompt captures:

  LAYER PROPAGATION   : per-layer hidden-state RMS of the last token,
                        MoE routing entropy per layer, side-module gate values
  ABSTENTION HEAD     : per-layer head z + mean vs tau
  SOFTMAX             : at the answer-decision slot: top-5 vocab tokens with
                        probabilities, and the <abs> row's logit/probability

Writes final_eval_v11.json + a compact printed report.
"""
import sys, json, math
from pathlib import Path
import torch
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).parent))
import abstention_v5 as v5
import abstention_v9 as v9
import abstention_v10 as v10
import abstention_v11 as v11

DEVICE = v5.DEVICE
ABS_ID = v5.ABS_ID
N = 5   # prompts per domain


def holdout_prompts(tok):
    m = [d for d, _ in v9.math_pairs('test')][:N]
    c = [d for d, _ in v10.code_pairs_stub('test')][:N]
    s = [d for d, _ in v11.sql_pairs('test')][:N]
    stories = []
    for t in v5.story_stream(400000):
        txt = t.replace("\n", " ").strip()
        if len(txt) > 40:
            stories.append(txt[:120] + "\n")
        if len(stories) >= N:
            break
    return {"math": m, "code": c, "sql": s, "story": stories}


def probe_forward(model, tok, prompt, prefix):
    ids = tok(prompt + prefix, add_special_tokens=False)["input_ids"]
    x = torch.tensor([ids], device=DEVICE)
    caps = {}

    def mk_hook(i, moe):
        def hook(mod, inp, out):
            idx = moe.last_topk if hasattr(moe, "last_topk") else None
            h = moe.last_abs_logit
            caps[i] = {"hz": float(h.mean(-1)[0, -1]) if h is not None else None}
        return hook

    handles = [moe.register_forward_hook(mk_hook(i, moe))
               for i, moe in enumerate(model.moe_layers)]
    hs_hook_handles = []
    hstore = {}

    def hs_hook(i):
        def hook(mod, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            hstore[i] = float(o[0, -1].float().pow(2).mean().sqrt())
        return hook
    for i, moe in enumerate(model.moe_layers):
        hs_hook_handles.append(moe.register_forward_hook(hs_hook(i)))

    model.eval()
    with torch.no_grad():
        logits, routing = model(x)
    for h in handles + hs_hook_handles:
        h.remove()

    z_layers = [caps.get(i, {}).get("hz") for i in range(len(model.moe_layers))]
    z = sum(v for v in z_layers if v is not None) / max(1, len([v for v in z_layers if v is not None]))
    ent_layers = []
    for topk in routing:
        counts = torch.bincount(topk.flatten(), minlength=16).float()
        p = counts / counts.sum().clamp_min(1)
        ent_layers.append(float(-(p[p > 0] * p[p > 0].log()).sum()))
    rms_layers = [hstore.get(i) for i in range(len(model.moe_layers))]

    lg = logits[0, -1].float()
    probs = F.softmax(lg, dim=-1)
    top = torch.topk(probs, 6)
    tops = []
    for pr, ti in zip(top.values.tolist(), top.indices.tolist()):
        if ti == ABS_ID:
            continue
        tops.append({"tok": tok.decode([ti]), "p": round(pr, 4)})
        if len(tops) >= 5:
            break
    return {
        "z_layers": [round(v, 2) if v is not None else None for v in z_layers],
        "z_mean": round(z, 2),
        "routing_entropy": [round(e, 2) for e in ent_layers],
        "rms": [round(r, 2) if r is not None else None for r in rms_layers],
        "abs_p": round(float(probs[ABS_ID]), 6),
        "abs_logit": round(float(lg[ABS_ID]), 2),
        "top": tops,
        "abstain": bool(z > model.tau),
    }


def main():
    print("FINAL EVAL: model + internals")
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
    model.tau = resume["tau"]
    print(f"tau={model.tau}")

    prompts = holdout_prompts(tok)
    report, grid = {}, {}
    for mname, snap in [("none", None), ("math", bundles["math"]),
                        ("code", bundles["code"]), ("sql", bundles["sql"])]:
        v5.set_domain_module(model, snap)
        # module gate values as attached right now
        gates = []
        for moe in model.moe_layers:
            g = moe.abst.gate
            gates.append(round(float(g.mean()) if torch.is_tensor(g) else float(g), 3))
        report[f"module_{mname}"] = {"side_gates": gates, "domains": {}}
        grid[mname] = {}
        for d, prs in prompts.items():
            prefix = "\n" if d == "story" else "Solution:\n"
            rows = [probe_forward(model, tok, p, prefix) for p in prs]
            za = sum(r["z_mean"] for r in rows) / len(rows)
            ab = sum(r["abstain"] for r in rows)
            abs_p = sum(r["abs_p"] for r in rows) / len(rows)
            grid[mname][d] = {"z_mean": round(za, 2), "abstained": f"{ab}/{len(rows)}",
                              "abs_p_mean": round(abs_p, 6)}
            report[f"module_{mname}"]["domains"][d] = rows
            print(f"module={mname:4s} {d:5s}: z={za:+6.2f}  abstain {ab}/{len(rows)}  "
                  f"abs_p={abs_p:.4f}  top1={rows[0]['top'][0]['tok']!r}")
    (v5.WS / "final_eval_v11.json").write_text(json.dumps(
        {"tau": model.tau, "summary": grid, "detail": report}, indent=2))
    print("EVAL COMPLETE -> final_eval_v11.json")


if __name__ == "__main__":
    main()
