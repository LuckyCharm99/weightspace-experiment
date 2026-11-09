"""
Experiment v5 — ABSTENTION MODULE.
Goal: the model replies with an abstention response when the domain expert
(module) is missing. Mechanism: a learned <abs> action token whose logit
competes inside the vocabulary softmax; when <abs> wins, the softmax over
the 151,666 vocab is effectively nullified — the model declines rather than
hallucinates.

Layout (foundation 100% frozen):
  - moe.dom   : domain module slot (frozen weights from v4 bundle; gate=0 = absent)
  - moe.abst  : TRAINABLE calibration side-expert bundle (~6.3M) + abs_head
  - embed[<abs>] : trainable single row (lm_head tied)

Two-world training (this is what makes abstention CONTEXT-GATED, not memorized):
  World OFF (no module): story plain, math -> 'Solution:' then <abs>, code -> <abs>
  World ON  (module loaded): story plain, math answered normally (never <abs>), code likewise

Verification grid: for each domain prompt set, measure greedy P(<abs>) under
module ∈ {absent, math, code}; generations must show ⟦ABSTAINED⟧ exactly when
the matching module is absent.
"""

import copy
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_module_experts as v3mod
from train_module_experts import (
    ModelConfig, MiniFlashNextLM, FlashNextMoE, Packer,
    DEVICE, SEQ_LEN, BATCH, GATE_MAX, GATE_INIT,
    math_stream, code_stream, story_stream,
)

WS = Path(__file__).resolve().parent.parent
CKPT = WS / "checkpoints/ckpt_final.pt"
BUNDLES = WS / "checkpoints/module_bundles_v4.pt"
BUNDLES_FALLBACK = WS / "checkpoints/module_bundles_v3.pt"
ABS_ID = 151665

AB_STEPS = 600
AB_GRAD_ACCUM = 4
LR_AB = 3e-4
LR_GATE_AB = 1e-2
LR_EMB = 3e-2   # <abs> row only (grad-hook isolated) — needs to travel far
BCE_POS_W = 8.0
ABS_HEAD_K = 6.0
BCE_WEIGHT = 2.0
SWAP_EVERY = 5          # alternate world every N steps (equal mix per world over a cycle)


# ==========================================
# Model with <abs> token + dual side slots
# ==========================================

def build_model():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B", trust_remote_code=True)
    tok.pad_token = tok.eos_token
    tok.add_special_tokens({"additional_special_tokens": ["<abs>"]})
    assert tok.convert_tokens_to_ids("<abs>") == ABS_ID

    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    mcfg = ModelConfig(**{k: v for k, v in ck["config"].items()
                          if k in ModelConfig.__dataclass_fields__})
    mcfg.vocab_size = ABS_ID + 1
    model = MiniFlashNextLM(mcfg)
    sd = ck["model"]
    old_emb = sd["embed.weight"]
    pad = torch.zeros(mcfg.vocab_size - old_emb.shape[0], old_emb.shape[1])
    sd["embed.weight"] = torch.cat([old_emb, pad], dim=0)
    sd["lm_head.weight"] = sd["embed.weight"]      # tied
    model.load_state_dict(sd, strict=False)
    with torch.no_grad():
        model.embed.weight[ABS_ID] = old_emb[tok.eos_token_id] * 0.9   # abstain ≈ "output nothing"
    model.to(DEVICE)

    orig_fwd = FlashNextMoE.forward

    def make_patched(orig):
        def patched(self, x):
            B, S, D = x.shape
            out, idx = orig(self, x)
            d = self.dom(x)                               # frozen domain module (gate 0 = off)
            out = out + d
            h = self.abst.norm(self.abst.expert(x.reshape(-1, D)).view(B, S, D))
            h = h / h.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()   # unit RMS
            # v6b: head must SEE module activity — abstention = "my expert for this
            # domain is not online". d is exactly that signal.
            self.last_abs_logit = self.abst_head((h + d).float())      # [B,S,1]
            return out + h * self.abst.gate, idx
        return patched

    # v6 FIX: the abstain head must enter the vocabulary softmax itself.
    # Wrapper LM head: logit[<abs>] += ABS_HEAD_K * mean of per-layer abstain logits.
    class AbsLMHead(nn.Module):
        def __init__(self, lin):
            super().__init__()
            self.lin = lin
            self.boost = nn.Parameter(torch.tensor(4.0, device=DEVICE))
        def forward(self, hs):
            logits = self.lin(hs)
            head_sum = torch.cat([moe.last_abs_logit for moe in model.moe_layers],
                                 dim=-1).mean(-1)          # [B,S,1]
            return logits + head_sum.squeeze(-1).unsqueeze(-1) * ABS_MASK * self.boost
    ABS_MASK = torch.zeros(1, 1, mcfg.vocab_size, device=DEVICE)
    ABS_MASK[0, 0, ABS_ID] = 1.0
    model.lm_head = AbsLMHead(model.lm_head)

    for moe in model.moe_layers:
        moe.dom = v3mod.SideExpertModule(mcfg).to(DEVICE)
        with torch.no_grad():
            moe.dom.gate_raw.fill_(0.0)             # module absent initially
        moe.abst = v3mod.SideExpertModule(mcfg).to(DEVICE)
        src = moe.experts[0]
        with torch.no_grad():
            for sp, tp in zip(moe.abst.expert.parameters(), src.parameters()):
                sp.copy_(tp)
                sp.add_(torch.randn_like(sp) * 0.02)
            moe.abst.gate_raw.fill_(math.atanh(GATE_INIT / GATE_MAX))
        moe.abst_head = nn.Linear(mcfg.hidden_dim, 1).to(DEVICE)
        with torch.no_grad():
            moe.abst_head.weight.mul_(0.01)
            moe.abst_head.bias.fill_(-1.0)          # neutral start: reachable in <600 steps
        moe.last_abs_logit = None
        moe.forward = make_patched(orig_fwd).__get__(moe, type(moe))

    print(f"Model ready: vocab {mcfg.vocab_size}, <abs> id {ABS_ID}")
    return model, mcfg, tok


def freeze_for_abstention(model):
    for p in model.parameters():
        p.requires_grad_(False)
    n = 0
    for moe in model.moe_layers:
        for p in moe.abst.parameters():
            p.requires_grad_(True); n += p.numel()
        for p in moe.abst_head.parameters():
            p.requires_grad_(True); n += p.numel()
    # <abs> embedding row only via grad hook masking other rows
    model.embed.weight.requires_grad_(True)
    model.lm_head.boost.requires_grad_(True)
    keep = torch.zeros(model.embed.weight.shape[0], device=DEVICE)
    keep[ABS_ID] = 1.0

    def hook(grad):
        return grad * keep.unsqueeze(1)
    model.embed.weight.register_hook(hook)
    print(f"Trainable (abstention bundle + abs emb row): {n/1e6:.2f}M + emb row")


def set_domain_module(model, snap):
    for i, moe in enumerate(model.moe_layers):
        if snap is None:
            with torch.no_grad():
                moe.dom.gate_raw.fill_(0.0)
        else:
            moe.dom.load_state_dict(snap[i])   # includes trained gate_raw


# ==========================================
# Packer with supervision weights
# ==========================================

class MixPacker:
    """Builds (inputs, targets, weights, abstain_mask) windows over three modes:
       'plain'   : normal continuation, abstain target 0
       'abstain' : problem + 'Solution:' then <abs>, abstain target 1 at answer pos
       returns numpy-free torch batches on DEVICE
    """

    def __init__(self, tokenizer, gens, prefix="Solution:\n"):
        self.tok = tokenizer
        self.gens = gens          # list of (generator, mode)
        self.i = 0
        self.gi = [0] * len(gens)
        self.prefix = prefix
        self.buf_t, self.buf_y, self.buf_w, self.buf_a = [], [], [], []

    def _next_text(self):
        gen, mode = self.gens[self.i % len(self.gens)]
        self.i += 1
        return mode, next(gen)

    def _fill(self, need):
        while len(self.buf_t) < need:
            mode, text = self._next_text()
            ids = self.tok(text, truncation=True, max_length=4096)["input_ids"]
            eos = self.tok.eos_token_id
            if mode == "abstain":
                pfx = self.tok(self.prefix, add_special_tokens=False)["input_ids"]
                # T stream: problem + prefix + eos ; Y stream same but eos-><abs>
                L = len(ids) + len(pfx) + 1
                self.buf_t.extend(ids + pfx + [eos])
                self.buf_y.extend(ids + pfx + [ABS_ID])
                w = [0.0] * L
                w[len(ids) + len(pfx) - 1] = 30.0    # CE at answer decision point (anti-dilution)
                self.buf_w.extend(w)
                a = [0.0] * L
                a[len(ids) + len(pfx) - 1] = 1.0     # abstain BCE target 1 at that position
                self.buf_a.extend(a)
            else:  # plain
                L = len(ids) + 1
                self.buf_t.extend(ids + [eos])
                self.buf_y.extend(ids + [eos])
                self.buf_w.extend([1.0] * len(ids) + [0.0])
                self.buf_a.extend([0.0] * L)

    def batch(self):
        need = (SEQ_LEN + 1) * BATCH
        self._fill(need)
        T = torch.tensor(self.buf_t[:need]).view(BATCH, SEQ_LEN + 1)
        Y = torch.tensor(self.buf_y[:need]).view(BATCH, SEQ_LEN + 1)
        W = torch.tensor(self.buf_w[:need], dtype=torch.float32).view(BATCH, SEQ_LEN + 1)
        A = torch.tensor(self.buf_a[:need], dtype=torch.float32).view(BATCH, SEQ_LEN + 1)
        for lst in (self.buf_t, self.buf_y, self.buf_w, self.buf_a):
            del lst[:need]
        inp = T[:, :-1].to(DEVICE)
        return (inp, Y[:, 1:].to(DEVICE), W[:, 1:].to(DEVICE), A[:, 1:].to(DEVICE))


# ==========================================
# Loss
# ==========================================

def abst_loss(model, inp, y, w, a):
    logits, _ = model(inp)
    vloss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                            reduction="none").view(y.shape)
    vterm = (vloss * w).sum() / w.sum().clamp_min(1.0)
    abs_logit = torch.cat([moe.last_abs_logit for moe in model.moe_layers],
                          dim=-1).mean(-1).squeeze(-1)   # [B,S]
    bce = F.binary_cross_entropy_with_logits(abs_logit, a, reduction="none")
    # Both decision classes (w>=20 slots: abstain AND answer) get full BCE mass;
    # keying only on a>0 suppressed answer-decisions 80:1 (v9c math-on miss).
    pos_w = torch.where(w >= 20, 8.0, torch.where(a > 0, 8.0, 0.1))
    bterm = (bce * pos_w).sum() / pos_w.sum().clamp_min(1.0)
    return vterm + BCE_WEIGHT * bterm, vterm.item(), bterm.item()


# ==========================================
# Training with two-world alternation
# ==========================================

def train(model, mcfg, mix_off, mix_on_math, mix_on_code, bundles):
    print("\n" + "=" * 60)
    print("TRAINING ABSTENTION BUNDLE (world alternation: off / math-on / code-on)")
    print("=" * 60)
    params_mod, params_gate, params_head = [], [], []
    for moe in model.moe_layers:
        for nm, p in moe.abst.named_parameters():
            (params_gate if nm == "gate_raw" else params_mod).append(p)
        params_head += list(moe.abst_head.parameters())
    opt = torch.optim.AdamW([
        {"params": params_mod, "lr": LR_AB},
        {"params": params_gate, "lr": LR_GATE_AB},
        {"params": params_head, "lr": LR_AB},
        {"params": [model.embed.weight], "lr": LR_EMB},
        {"params": [model.lm_head.boost], "lr": 5e-3},
    ], weight_decay=0.0)

    model.train()
    t0 = time.time()
    for step in range(AB_STEPS):
        phase = (step // SWAP_EVERY) % 3     # 0: off, 1: math-on, 2: code-on
        if phase == 0:
            set_domain_module(model, None)
            packer = mix_off
        elif phase == 1:
            set_domain_module(model, bundles["math"])
            packer = mix_on_math
        else:
            set_domain_module(model, bundles["code"])
            packer = mix_on_code
        opt.zero_grad()
        av = ab = 0.0
        for _ in range(AB_GRAD_ACCUM):
            inp, y, w, a = packer.batch()
            l, vl, bl = abst_loss(model, inp, y, w, a)
            (l / AB_GRAD_ACCUM).backward()
            av += vl; ab += bl
        torch.nn.utils.clip_grad_norm_(
            params_mod + params_gate + params_head + [model.embed.weight], 1.0)
        opt.step()
        if (step + 1) % 50 == 0:
            world = ["off", "math-on", "code-on"][phase]
            print(f"  step {step+1}/{AB_STEPS} | world {world} | "
                  f"vocab {av/AB_GRAD_ACCUM:.3f} | abstain-bce {ab/AB_GRAD_ACCUM:.4f} | "
                  f"{time.time()-t0:.0f}s")


# ==========================================
# Eval
# ==========================================

@torch.no_grad()
def abs_probs(model, tok, prompts, prefix="Solution:\n"):
    model.eval()
    out = []
    for pr in prompts:
        ids = tok(pr + prefix, return_tensors="pt")["input_ids"].to(DEVICE)
        logits, _ = model(ids)
        probs = F.softmax(logits[0, -1].float(), dim=-1)
        out.append({"p_abs": round(probs[ABS_ID].item(), 4),
                    "p_top": round(probs.max().item(), 4),
                    "argmax_abs": int(probs.argmax().item() == ABS_ID)})
    return out


@torch.no_grad()
def generate(model, tok, prompt, max_new=80):
    ids = tok(prompt, return_tensors="pt")["input_ids"].to(DEVICE)
    model.eval()
    for _ in range(max_new):
        logits, _ = model(ids)
        nxt = int(logits[0, -1].argmax().item())
        if nxt == tok.eos_token_id:
            break
        if nxt == ABS_ID:
            return tok.decode(ids[0], skip_special_tokens=True) + "  ⟦ABSTAINED⟧"
        ids = torch.cat([ids, torch.tensor([[nxt]], device=DEVICE)], dim=1)
        if ids.shape[1] > SEQ_LEN:
            ids = ids[:, -SEQ_LEN:]
    return tok.decode(ids[0], skip_special_tokens=True)


MATH_PROMPTS = ["Problem: What is the derivative of x^2 + 3x?\n",
                "Problem: Solve for x: 2x + 5 = 17\n",
                "Problem: What is the sum of 127 and 358?\n"]
CODE_PROMPTS = ["def fib(n):\n    if n < 2:\n",
                "Write a python function to sort a list.\ndef sort_list(lst):\n"]
STORY_PROMPTS = ["Once upon a time there was a little girl",
                 "There was a small boy who lived near a river"]


def main():
    model, mcfg, tok = build_model()
    freeze_for_abstention(model)
    path = BUNDLES if BUNDLES.exists() else BUNDLES_FALLBACK
    bundles = torch.load(path, map_location=DEVICE, weights_only=False)
    print(f"Domain bundles from {path}")

    mix_off = MixPacker(tok, [
        (story_stream(60000), "plain"),
        (_cycle(lambda: math_stream("train")), "abstain"),
        (_cycle(lambda: code_stream("train")), "abstain"),
    ])
    mix_on_math = MixPacker(tok, [
        (story_stream(70000), "plain"),
        (_cycle(lambda: math_stream("train")), "plain"),      # capability present -> answer
        (_cycle(lambda: code_stream("train")), "abstain"),    # other domain absent -> abstain
    ])
    mix_on_code = MixPacker(tok, [
        (story_stream(80000), "plain"),
        (_cycle(lambda: code_stream("train")), "plain"),
        (_cycle(lambda: math_stream("train")), "abstain"),
    ])

    train(model, mcfg, mix_off, mix_on_math, mix_on_code, bundles)

    # ---- grid ----
    grid = {}
    for mname, snap in [("none", None), ("math", bundles["math"]), ("code", bundles["code"])]:
        set_domain_module(model, snap)
        grid[f"module_{mname}"] = {
            "math": abs_probs(model, tok, MATH_PROMPTS),
            "code": abs_probs(model, tok, CODE_PROMPTS),
            "story": abs_probs(model, tok, STORY_PROMPTS, prefix="\n"),
        }
        print(f"\nmodule={mname}:")
        for dom, rows in grid[f"module_{mname}"].items():
            print(f"  {dom}: p_abs={[r['p_abs'] for r in rows]} argmax_abs={[r['argmax_abs'] for r in rows]}")

    # ---- final-run diagnostics (self-explaining result) ----
    x, y, w, a = mix_off.batch()
    with torch.no_grad():
        lg, _ = model(x)
    lg = lg.float()
    hl = torch.cat([moe.last_abs_logit for moe in model.moe_layers], -1).mean(-1).squeeze(-1)
    pos = a > 0.5
    ar = torch.arange(lg.size(-1), device=lg.device)
    al = lg[:, :, ABS_ID]
    mx = lg.masked_fill(ar == ABS_ID, -1e9).amax(-1)
    print(f"DIAG boost={model.lm_head.boost.item():.2f} head_z_dec={hl[pos].mean().item():.2f} "
          f"head_z_plain={hl[~pos].mean().item():.2f} margin_dec={(al[pos]-mx[pos]).mean().item():.1f}")

    # ---- generations ----
    gens = {}
    set_domain_module(model, None)
    gens["math_no_module"] = generate(model, tok, "Problem: Solve for x: 3x + 6 = 21\nSolution:")
    gens["code_no_module"] = generate(model, tok, "Write a python function to sum a list.\ndef sum_list(lst):\nSolution:")
    gens["story_no_module"] = generate(model, tok, "Once upon a time there was a little girl")
    set_domain_module(model, bundles["math"])
    gens["math_with_module"] = generate(model, tok, "Problem: Solve for x: 3x + 6 = 21\nSolution:")
    set_domain_module(model, bundles["code"])
    gens["code_with_module"] = generate(model, tok, "Write a python function to sum a list.\ndef sum_list(lst):\nSolution:")
    print("\nGenerations:")
    for k, v in gens.items():
        print(f"  {k}: ...{v[-220:]}")

    out = {"abstain_grid": grid, "generations": gens}
    (WS / "abstention_results.json").write_text(json.dumps(out, indent=2))
    torch.save({"abst": [moe.abst.state_dict() for moe in model.moe_layers],
                "abs_heads": [moe.abst_head.state_dict() for moe in model.moe_layers],
                "abs_emb_row": model.embed.weight[ABS_ID].detach().cpu()},
               WS / "checkpoints/abstention_bundle_v5.pt")
    print("\nSaved: abstention_results.json, abstention_bundle_v5.pt")
    print("ABSTENTION EXPERIMENT COMPLETE")


def _cycle(factory):
    while True:
        n = 0
        for item in factory():
            n += 1
            yield item
        if n == 0:
            raise RuntimeError("empty factory")


if __name__ == "__main__":
    main()
