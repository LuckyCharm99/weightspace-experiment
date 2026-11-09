"""
Module-Expert Experiment v3: frozen Mini-Qwen base + swappable multi-layer modules.

Doctrine: tokenizer + language experts = foundation; domain experts = modules.
A module = one side-expert branch per MoE layer (8 total, ~6.3M params), the
only trainable unit. Base stays 100% frozen. Swapping module weight bundles
must swap capability with zero gradient steps on the foundation.

v3 changes vs v2:
  - side experts on ALL MoE layers (multi-layer module, LoRA-style reach)
  - gate envelope tanh(raw)*GATE_MAX with unit-RMS normalized output
  - eval on hendrycks_math TEST split + code_contests TEST split (no leakage)
  - fuse guard only for numerical blowups
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
from datasets import load_dataset
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_and_experiment import ModelConfig, MiniFlashNextLM, SwiGLUExpert, FlashNextMoE, RMSNorm

torch.manual_seed(42)
WS = Path(__file__).resolve().parent.parent
CKPT = WS / "checkpoints/ckpt_final.pt"
DEVICE = torch.device("cuda")
SEQ_LEN = 128
BATCH = 2
GRAD_ACCUM = 4
STEPS_PER_DOMAIN = 1500
LR_MOD = 3e-4
LR_GATE = 1e-2
GATE_MAX = 0.5
GATE_INIT = 0.1
SPIKE_LIMIT = 60.0

MATH_SUBJECTS = ["algebra", "prealgebra", "number_theory", "intermediate_algebra"]

# ==========================================
# Base model
# ==========================================

def load_base():
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    mcfg = ModelConfig(**{k: v for k, v in ck["config"].items()
                          if k in ModelConfig.__dataclass_fields__})
    model = MiniFlashNextLM(mcfg)
    model.load_state_dict(ck["model"], strict=False)
    model.to(DEVICE)
    print(f"Base loaded: step {ck['step']}, {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
    return model, mcfg


class SideExpertModule(nn.Module):
    """Unit-RMS normalized additive SwiGLU branch with bounded gate."""

    def __init__(self, mcfg):
        super().__init__()
        self.expert = SwiGLUExpert(mcfg)
        self.norm = RMSNorm(mcfg.hidden_dim)
        self.gate_raw = nn.Parameter(
            torch.tensor(math.atanh(GATE_INIT / GATE_MAX), device=DEVICE))

    @property
    def gate(self):
        return torch.tanh(self.gate_raw) * GATE_MAX

    def forward(self, x):
        B, S, D = x.shape
        h = self.norm(self.expert(x.reshape(-1, D)).view(B, S, D))
        h = h / h.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()
        return h * self.gate


def attach_modules(model, mcfg):
    """One side module per MoE layer; warm-init each from that layer's expert 0."""
    sides = nn.ModuleList()
    orig_fwd = FlashNextMoE.forward

    def make_patched(orig):
        def patched(self, x):
            out, idx = orig(self, x)
            return out + self.side(x), idx
        return patched

    for i, moe in enumerate(model.moe_layers):
        side = SideExpertModule(mcfg).to(DEVICE)
        src = moe.experts[0]
        with torch.no_grad():
            for sp, tp in zip(side.expert.parameters(), src.parameters()):
                sp.copy_(tp)
                sp.add_(torch.randn_like(sp) * 0.02)
        moe.side = side
        moe.forward = make_patched(orig_fwd).__get__(moe, type(moe))
        sides.append(side)
    model.sides = sides
    return sides


def reinit_modules(sides, model):
    """Reset every side module to its warm init (base expert + noise)."""
    with torch.no_grad():
        for i, side in enumerate(sides):
            src = model.moe_layers[i].experts[0]
            for sp, tp in zip(side.expert.parameters(), src.parameters()):
                sp.copy_(tp)
                sp.add_(torch.randn_like(sp) * 0.02)
            side.norm.weight.fill_(1.0)
            side.gate_raw.fill_(math.atanh(GATE_INIT / GATE_MAX))


def freeze_base(sides, model):
    for p in model.parameters():
        p.requires_grad_(False)
    trainable = 0
    for side in sides:
        for p in side.parameters():
            p.requires_grad_(True)
            trainable += p.numel()
    print(f"Frozen base. Trainable module params: {trainable/1e6:.2f}M ({len(sides)} layers)")


# ==========================================
# Data
# ==========================================

def _cycling(factory):
    """Yield factory() repeatedly -> multi-epoch stream over a finite dataset."""
    while True:
        n = 0
        for item in factory():
            n += 1
            yield item
        if n == 0:
            raise RuntimeError("factory produced no items")


def math_stream(split="train"):
    for subject in MATH_SUBJECTS:
        ds = load_dataset("EleutherAI/hendrycks_math", subject, split=split, streaming=True)
        for row in ds:
            yield row["problem"].strip() + "\nSolution:\n" + row["solution"].strip()


def code_stream(split="train"):
    ds = load_dataset("deepmind/code_contests", split=split, streaming=True)
    for row in ds:
        sols = row.get("solutions") or {}
        blobs = []
        for s in sols.get("solution") or []:
            if isinstance(s, dict) and "python" in str(s.get("language", "")).lower():
                c = s.get("content", "")
                if c:
                    blobs.append(c)
        if blobs or row.get("description"):
            yield (row.get("description") or "") + "\n" + "\n".join(blobs[:2])


def story_stream(skip=50000):
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    for i, row in enumerate(ds):
        if i >= skip:
            yield row["text"]


class Packer:
    """EOS-delimited token packing -> [BATCH, SEQ+1] GPU batches.

    `gen` may be an iterator OR a zero-arg factory returning a fresh iterator;
    factories get cycled for multi-epoch training over finite streams.
    """

    def __init__(self, tokenizer, gen, max_text_tokens=4096):
        self.tok = tokenizer
        self.gen = _cycling(gen) if callable(gen) else gen
        self.buf = []
        self.eos = tokenizer.eos_token_id
        self.max_t = max_text_tokens

    def _fill(self, need):
        while len(self.buf) < need:
            ids = self.tok(next(self.gen), truncation=True,
                           max_length=self.max_t)["input_ids"]
            self.buf.extend(ids)
            self.buf.append(self.eos)

    def batch(self):
        need = (SEQ_LEN + 1) * BATCH
        self._fill(need)
        flat = self.buf[:need]
        self.buf = self.buf[need:]
        return torch.tensor(flat).view(BATCH, SEQ_LEN + 1).to(DEVICE)


def make_batches(packer, n):
    return [packer.batch() for _ in range(n)]


# ==========================================
# Train / eval
# ==========================================

def train_modules(sides, model, mcfg, packer, name):
    print("\n" + "=" * 60)
    print(f"TRAINING MODULE BUNDLE: {name}")
    print("=" * 60)
    mod_params, gate_params = [], []
    for side in sides:
        for nm, p in side.named_parameters():
            (gate_params if nm == "gate_raw" else mod_params).append(p)
    opt = torch.optim.AdamW(
        [{"params": mod_params, "lr": LR_MOD}, {"params": gate_params, "lr": LR_GATE}],
        weight_decay=0.0)
    crit = nn.CrossEntropyLoss()
    model.train()
    step, skipped, t0 = 0, 0, time.time()
    while step < STEPS_PER_DOMAIN:
        opt.zero_grad()
        acc = 0.0
        for _ in range(GRAD_ACCUM):
            batch = packer.batch()
            inputs, targets = batch[:, :-1], batch[:, 1:].contiguous()
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, _ = model(inputs)
                loss = crit(logits.view(-1, mcfg.vocab_size), targets.view(-1)) / GRAD_ACCUM
            loss.backward()
            acc += loss.item() * GRAD_ACCUM
        if acc > SPIKE_LIMIT:
            skipped += 1
            opt.zero_grad()
        else:
            torch.nn.utils.clip_grad_norm_(mod_params + gate_params, 1.0)
            opt.step()
        step += 1
        if step % 25 == 0:
            print(f"  {name} step {step}/{STEPS_PER_DOMAIN} | loss {acc:.3f} | "
                  f"gate {sides[-1].gate.item():.3f} | fuse-skips {skipped} | {time.time()-t0:.0f}s")


@torch.no_grad()
def eval_loss_on(sides, model, mcfg, batches):
    """Eval with whatever state is currently loaded (gate kept as-is)."""
    model.eval()
    crit = nn.CrossEntropyLoss()
    tot = 0.0
    for batch in batches:
        inputs, targets = batch[:, :-1], batch[:, 1:].contiguous()
        with autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, _ = model(inputs)
            tot += crit(logits.view(-1, mcfg.vocab_size), targets.view(-1)).item()
    return round(tot / len(batches), 4)


@torch.no_grad()
def greedy(model, tokenizer, prompt, max_new=60):
    ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(DEVICE)
    model.eval()
    for _ in range(max_new):
        logits, _ = model(ids)
        nxt = logits[0, -1].argmax().view(1, 1)
        ids = torch.cat([ids, nxt], dim=1)
        if ids.shape[1] > SEQ_LEN:
            ids = ids[:, -SEQ_LEN:]
    return tokenizer.decode(ids[0], skip_special_tokens=True)


@torch.no_grad()
def nucleus(model, tokenizer, prompt, max_new=150, temp=0.8, top_p=0.9):
    ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(DEVICE)
    model.eval()
    for _ in range(max_new):
        logits, _ = model(ids)
        nxt_logits = logits[0, -1].float() / temp
        # nucleus filter
        sorted_l, sorted_i = torch.sort(nxt_logits, descending=True)
        probs = torch.softmax(sorted_l, dim=-1)
        cum = torch.cumsum(probs, dim=-1)
        keep = cum - probs <= top_p
        keep[0] = True  # always keep the most likely
        mask = torch.full_like(nxt_logits, float("-inf"))
        mask[sorted_i[keep]] = sorted_l[keep]
        nxt = torch.multinomial(torch.softmax(mask, dim=-1), 1).view(1, 1)
        if nxt.item() == tokenizer.eos_token_id:
            break
        ids = torch.cat([ids, nxt], dim=1)
        if ids.shape[1] > SEQ_LEN:
            ids = ids[:, -SEQ_LEN:]
    return tokenizer.decode(ids[0], skip_special_tokens=True)


def snapshot(sides):
    return [copy.deepcopy(s.state_dict()) for s in sides]


def restore(sides, snap):
    for side, sd in zip(sides, snap):
        side.load_state_dict(sd)


# ==========================================
# Main
# ==========================================

def main():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model, mcfg = load_base()
    sides = attach_modules(model, mcfg)
    freeze_base(sides, model)

    print("\nBuilding held-out eval batches (TEST splits / distant stories)...")
    eval_math = make_batches(Packer(tokenizer, math_stream("test")), 24)
    eval_code = make_batches(Packer(tokenizer, code_stream("test")), 24)
    eval_story = make_batches(Packer(tokenizer, story_stream(50000)), 24)
    print("Eval batches ready.")

    # NOTE: gate = tanh(raw)*GATE_MAX, so "off" means raw=0 (NOT -30, that's sigmoid idiom).
    results = {}
    with torch.no_grad():
        for side in sides:
            side.gate_raw.fill_(0.0)   # tanh(0)=0 -> module fully off
    results["module_none"] = {
        "story": eval_loss_on(sides, model, mcfg, eval_story),
        "math": eval_loss_on(sides, model, mcfg, eval_math),
        "code": eval_loss_on(sides, model, mcfg, eval_code)}
    print(f"\nBASE (no module): {results['module_none']}")

    # ---- math module bundle ----
    reinit_modules(sides, model)
    train_modules(sides, model, mcfg, Packer(tokenizer, lambda: math_stream("train")), "math")
    math_snap = snapshot(sides)
    results["module_math"] = {
        "story": eval_loss_on(sides, model, mcfg, eval_story),
        "math": eval_loss_on(sides, model, mcfg, eval_math),
        "code": eval_loss_on(sides, model, mcfg, eval_code)}
    print(f"MATH module: {results['module_math']}")

    # ---- code module bundle ----
    reinit_modules(sides, model)
    train_modules(sides, model, mcfg, Packer(tokenizer, lambda: code_stream("train")), "code")
    code_snap = snapshot(sides)
    results["module_code"] = {
        "story": eval_loss_on(sides, model, mcfg, eval_story),
        "math": eval_loss_on(sides, model, mcfg, eval_math),
        "code": eval_loss_on(sides, model, mcfg, eval_code)}
    print(f"CODE module: {results['module_code']}")

    # ---- pure weight swap, no training ----
    restore(sides, math_snap)
    results["swap_math_again"] = {
        "story": eval_loss_on(sides, model, mcfg, eval_story),
        "math": eval_loss_on(sides, model, mcfg, eval_math),
        "code": eval_loss_on(sides, model, mcfg, eval_code)}
    print(f"SWAP -> MATH bundle (zero training in swap): {results['swap_math_again']}")

    # ---- module distance ----
    dist = {"cosine": [], "relative_L2": []}
    for sd_m, sd_c in zip(math_snap, code_snap):
        for k in sd_m:
            if sd_m[k].numel() == 1 or k == "gate_raw":
                continue
            a, b = sd_m[k].float(), sd_c[k].float()
            dist["cosine"].append(round(F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item(), 4))
            dist["relative_L2"].append(round(((a - b).norm() / (a.norm() + 1e-9)).item(), 4))
    results["module_distance"] = dist
    print(f"Module distance: cos {dist['cosine'][:4]}... relL2 {dist['relative_L2'][:4]}...")

    # ---- generations ----
    prompts = {
        "story": "Once upon a time there was a little girl",
        "math": "Problem: Solve for x: 3x + 6 = 21\nSolution:",
        "code": "def solution(n):\n    total = 0",
    }
    gens = {}
    for label, snap in [("math-module", math_snap), ("code-module", code_snap)]:
        restore(sides, snap)
        gens[label] = {d: nucleus(model, tokenizer, p) for d, p in prompts.items()}
        print(f"\n[{label}] generations:")
        for d, g in gens[label].items():
            print(f"  {d}: ...{g[-250:]}")
    results["generations"] = gens

    torch.save({"math": math_snap, "code": code_snap}, WS / "checkpoints/module_bundles_v3.pt")
    (WS / "module_results.json").write_text(json.dumps(results, indent=2))
    print("\nSaved: module_bundles_v3.pt, module_results.json")
    print("MODULE EXPERIMENT COMPLETE")


if __name__ == "__main__":
    main()
