"""v9b — kill the last confound: row-aligned fresh-start windows.

v9 verdict: train-span separation was real (abstain +4.2 vs answer -7.7, sep 9.2)
but eval prompts still scored -8.5. Cause: v9 windows are a CONCATENATED stream,
so every decision point sits at depth ~60-128 inside rich preceding context,
while eval prompts are 10-25 tokens long starting at position 0. The head keyed
on context depth, not just the gate.

v9b: each batch ROW is assembled from independent examples starting at index 0
(same shape eval has). Decision depths now span 5..128 uniformly, INCLUDING the
shallow depths eval uses. Span contract from v9 is unchanged (byte-identical
across worlds; only the gate flips the label).
"""
import sys, json
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).parent))
import abstention_v5 as v5
import abstention_v9 as v9
import abstention_v7 as v7
import train_module_experts as v3mod

DEVICE = v5.DEVICE
ABS_ID = v5.ABS_ID
SEQ_LEN, BATCH = v5.SEQ_LEN, v5.BATCH


class RowPacker(v5.MixPacker):
    """Each row = independent short segments starting at index 0 (eval-like).
    Guarantees >=1 abstain decision per row, first at shallow depth so the head
    sees eval-like context lengths. Plain segments are capped so they don't
    consume the whole row (that was the bug that left ~0 decisions/batch)."""
    ABST_MAX = 30          # cap on problem-display tokens (shallow depth like eval)
    PLAIN_MAX = 40         # cap so multiple segments fit per row

    def batch(self):
        eos = self.tok.eos_token_id
        pfx = self.tok(self.prefix, add_special_tokens=False)["input_ids"]
        T, Y, W, A = [], [], [], []
        for _ in range(BATCH):
            t, y, w, a = [], [], [], []
            first_done = False
            for _seg in range(40):                     # enough segments to fill a row
                if len(t) >= SEQ_LEN + 1:
                    break
                rem = SEQ_LEN + 1 - len(t)
                mode, text = self._next_text()
                if mode == "answer":
                    disp, sol = text
                    ids = self.tok(disp, truncation=True, max_length=4096)["input_ids"]
                    sol_ids = self.tok(sol, truncation=True, max_length=4096)["input_ids"]
                else:
                    ids = self.tok(text, truncation=True, max_length=4096)["input_ids"]
                    sol_ids = None
                if mode == "abstain":
                    ids = ids[:self.ABST_MAX]
                    need = len(ids) + len(pfx) + 1
                    if need > rem:
                        if first_done:
                            break                      # row full; already have a decision
                        pad = rem - len(pfx) - 1       # force one decision even if row tight
                        if pad < 3:
                            break
                        ids = ids[:pad]
                    o = len(t)
                    t += ids + pfx + [eos]
                    y += ids + pfx + [ABS_ID]
                    w += [0.0] * (len(ids) + len(pfx) + 1)
                    a += [0.0] * (len(ids) + len(pfx) + 1)
                    # FRAME FIX: batch() shifts W/A by one ([:,1:]), so the label
                    # must sit on the EOS-slot raw index to land on the decision
                    # slot (input = last pfx token, target = ABS) in the output frame.
                    dec = o + len(ids) + len(pfx)      # raw eos position
                    w[dec] = 30.0; a[dec] = 1.0
                    first_done = True
                elif mode == "answer":
                    # NEW (v9c): explicit ANSWER-decision supervision. The matching
                    # module is ON, so at the marker-end decision the target is the
                    # FIRST SOLUTION token (not <abs>), with abstain label 0. Without
                    # this the head only ever sees marker->abstain and abstains on any
                    # marker even when the right expert is online.
                    ids = ids[:self.ABST_MAX]
                    need = len(ids) + len(pfx) + len(sol_ids) + 1
                    if need > rem:
                        sol_ids = sol_ids[:max(2, rem - len(ids) - len(pfx) - 1)]
                        need = len(ids) + len(pfx) + len(sol_ids) + 1
                        if need > rem:
                            break
                    o = len(t)
                    t += ids + pfx + sol_ids + [eos]
                    y += ids + pfx + sol_ids + [eos]      # next-token (real answer)
                    w += [0.0] * (len(ids) + len(pfx) + len(sol_ids) + 1)
                    a += [0.0] * (len(ids) + len(pfx) + len(sol_ids) + 1)
                    dec = o + len(ids) + len(pfx)          # marker-end -> predicts sol[0]
                    w[dec] = 30.0; a[dec] = 0.0            # label 0 = answer, not abstain
                    first_done = True
                else:
                    take = ids[:min(len(ids), self.PLAIN_MAX, rem - 1)]
                    if not take:
                        break
                    t += take + [eos]
                    y += take + [eos]
                    w += [1.0] * len(take) + [0.0]
                    a += [0.0] * (len(take) + 1)
            while len(t) < SEQ_LEN + 1:
                t.append(eos); y.append(eos); w.append(0.0); a.append(0.0)
            T.append(t); Y.append(y); W.append(w); A.append(a)
        T = torch.tensor(T); Y = torch.tensor(Y)
        W = torch.tensor(W, dtype=torch.float32); A = torch.tensor(A, dtype=torch.float32)
        return (T[:, :-1].to(DEVICE), Y[:, 1:].to(DEVICE),
                W[:, 1:].to(DEVICE), A[:, 1:].to(DEVICE))


def worlds(tok, split):
    # columns: 0=display(abstain-shape), 1=full answer text
    MA = v5._cycle(lambda: (d for d, _ in v9.math_pairs(split)))
    CA = v5._cycle(lambda: (d for d, _ in v9.code_pairs(split)))
    # answer columns yield (display, solution) so packer can build the marker-end
    # 'answer' decision with label 0
    def answer_pairs(pairs):
        for disp, full in pairs:
            sol = full.split("Solution:\n", 1)[1] if "Solution:\n" in full else ""
            if sol:
                yield (disp, sol)
    MB = v5._cycle(lambda: answer_pairs(v9.math_pairs(split)))
    CB = v5._cycle(lambda: answer_pairs(v9.code_pairs(split)))
    off = RowPacker(tok, [(v5.story_stream(60000 if split == 'train' else 300000), 'plain'),
                          (MA, 'abstain'), (CA, 'abstain')])
    math_on = RowPacker(tok, [(v5.story_stream(70000 if split == 'train' else 310000), 'plain'),
                              (MB, 'answer'), (CA, 'abstain')])
    code_on = RowPacker(tok, [(v5.story_stream(80000 if split == 'train' else 320000), 'plain'),
                              (CB, 'answer'), (MA, 'abstain')])
    return off, math_on, code_on


def main():
    print("V9B: row-aligned fresh-start abstention")
    model, mcfg, tok = v5.build_model()
    v5.freeze_for_abstention(model)
    bundles = torch.load(v5.BUNDLES if v5.BUNDLES.exists() else v5.BUNDLES_FALLBACK,
                         map_location=DEVICE, weights_only=False)

    # quick structural proof: decision depth distribution includes eval-like depths
    off, mo, co = worlds(tok, 'train')
    x, y, w, a = off.batch()
    dec_idx = (a > 0.5).nonzero()[:, 1]
    print(f"DECDEPTH sample: {dec_idx.tolist()[:10]} (eval prompts sit at ~12-25)")

    v5.AB_STEPS = 600
    v5.train(model, mcfg, off, mo, co, bundles)

    toff, tmo, tco = worlds(tok, 'test')
    def grab2(mix, snap, n=25):
        """Bucket by decision geometry: abstain-dec (a=1,w=30) vs answer-dec
        (a=0,w=30) at the identical marker-end slot. Continuation tokens ignored."""
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
    dA, aA = grab2(toff, None)                       # off: everything abstains
    dM, aM = grab2(tmo, bundles['math'])             # math-on: math answers, code abstains
    dC, aC = grab2(tco, bundles['code'])             # code-on: code answers, math abstains
    abstain_z = dA + dM + dC
    answer_z = aA + aM + aC
    st_a, st_ans = v7.stats(abstain_z), v7.stats(answer_z)
    tau = round((st_a["p05"] + st_ans["p95"]) / 2, 2)
    print(f"DIAG9B abstain={st_a} answer={st_ans} tau={tau}")

    def gated(prompt, snap, prefix="Solution:\n"):
        v5.set_domain_module(model, snap)
        ids = tok(prompt + prefix, add_special_tokens=False)['input_ids']
        xx = torch.tensor([ids], device=DEVICE)
        model.eval()
        with torch.no_grad():
            model(xx)
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

    gens = {}
    for name, snap, pr, pre in [
            ("math_no_module", None, v5.MATH_PROMPTS[0], "Solution:\n"),
            ("code_no_module", None, v5.CODE_PROMPTS[0], "Solution:\n"),
            ("story_no_module", None, v5.STORY_PROMPTS[0], "\n"),
            ("math_wrong_module", bundles["code"], v5.MATH_PROMPTS[0], "Solution:\n"),
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
                "tau": tau}, v5.WS / "checkpoints/abstain_bundle_v9b.pt")
    (v5.WS / "abstain_v9b_results.json").write_text(json.dumps(
        {"diag": {"abstain": st_a, "answer": st_ans, "tau": tau},
         "grid": grid, "generations": gens}, indent=2))
    print("V9B COMPLETE")


if __name__ == "__main__":
    main()
