"""
Mini Qwen3.8-Flash-Next: Real Training + Weight-Space Experiments
=================================================================
Architecture: 3:1 GDN:Attention hybrid, 4-branch Gated Residual, Fine-Grained MoE
Training: bf16 AMP, cosine LR, gradient accumulation, proper batching
Then: weight-space profiling, pruning, injection, task vector surgery
"""

import copy
import math
import time
import json
from dataclasses import dataclass, field
from pathlib import Path

from datasets import load_dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer

torch.manual_seed(42)

# ==========================================
# CONFIG
# ==========================================

@dataclass
class ModelConfig:
    vocab_size: int = 151665
    hidden_dim: int = 512
    num_heads: int = 8
    head_dim: int = 64
    num_layers: int = 8          # 6 GDN + 2 Attention (3:1 ratio)
    num_experts: int = 16
    intermediate_dim: int = 512  # SwiGLU expert FFN dim
    top_k: int = 2
    bottleneck_rank: int = 64    # Gated Residual low-rank

@dataclass
class TrainConfig:
    seq_len: int = 128
    batch_size: int = 2
    grad_accum: int = 8          # effective batch = 16
    max_steps: int = 2000
    lr: float = 1e-4
    warmup_steps: int = 100
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    log_every: int = 20
    ckpt_every: int = 500
    save_dir: str = str(Path(__file__).resolve().parent.parent / "checkpoints")
    use_amp: bool = True

# ==========================================
# 1. CORE ARCHITECTURAL COMPONENTS
# ==========================================

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight


class GatedResidual(nn.Module):
    """4-branch widened residual stream with data-dependent read gates."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.num_branches = 4
        self.hidden_dim = cfg.hidden_dim
        self.read_down = nn.Linear(cfg.hidden_dim, cfg.bottleneck_rank, bias=False)
        self.read_up = nn.Linear(cfg.bottleneck_rank, 4 * cfg.hidden_dim, bias=False)
        self.write_gate = nn.Parameter(torch.ones(4))

    def forward(self, branches, block_output):
        B, S, D = block_output.shape
        gate_logits = self.read_up(F.silu(self.read_down(block_output)))
        read_scores = torch.sigmoid(gate_logits).view(B, S, 4, self.hidden_dim)
        mixed_input = (branches * read_scores).sum(dim=2)
        updated = branches.clone()
        for i in range(4):
            updated[:, :, i, :] = updated[:, :, i, :] + self.write_gate[i] * block_output
        return mixed_input, updated


class GatedDeltaNet(nn.Module):
    """Linear Attention via Recurrent Gated Delta Rule with L2-normalized QK."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.head_dim
        hdim = cfg.num_heads * cfg.head_dim

        self.q_proj = nn.Linear(cfg.hidden_dim, hdim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_dim, hdim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_dim, hdim, bias=False)
        self.alpha_gate = nn.Linear(cfg.hidden_dim, cfg.num_heads, bias=True)
        self.beta_gate = nn.Linear(cfg.hidden_dim, cfg.num_heads, bias=True)
        self.out_gate = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)
        self.o_proj = nn.Linear(hdim, cfg.hidden_dim, bias=False)

    def forward(self, x):
        B, S, D = x.shape
        H, d = self.num_heads, self.head_dim
        q = F.normalize(self.q_proj(x).view(B, S, H, d), p=2, dim=-1)
        k = F.normalize(self.k_proj(x).view(B, S, H, d), p=2, dim=-1)
        v = self.v_proj(x).view(B, S, H, d)
        alpha = torch.sigmoid(self.alpha_gate(x)).unsqueeze(-1)
        beta = torch.sigmoid(self.beta_gate(x)).unsqueeze(-1)

        state = torch.zeros(B, H, d, d, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(S):
            q_t, k_t, v_t = q[:, t], k[:, t], v[:, t]
            a_t = alpha[:, t].unsqueeze(-1)
            b_t = beta[:, t]
            state = state * a_t
            kv_pred = torch.einsum("bhde,bhe->bhd", state, k_t)
            delta = (v_t - kv_pred) * b_t
            state = state + torch.einsum("bhd,bhe->bhde", k_t, delta)
            y_t = torch.einsum("bhde,bhe->bhd", state, q_t)
            outputs.append(y_t)

        stacked = torch.stack(outputs, dim=1).view(B, S, -1)
        return self.o_proj(stacked) * torch.sigmoid(self.out_gate(x))


class GatedAttention(nn.Module):
    """Multi-Head Softmax Attention with output gating."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=cfg.hidden_dim, num_heads=cfg.num_heads, batch_first=True
        )
        self.out_gate = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)

    def forward(self, x):
        S = x.size(1)
        mask = torch.triu(torch.full((S, S), float("-inf"), device=x.device), diagonal=1)
        out, _ = self.mha(x, x, x, attn_mask=mask, is_causal=True)
        return out * torch.sigmoid(self.out_gate(x))


class SwiGLUExpert(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_dim, cfg.intermediate_dim, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_dim, cfg.intermediate_dim, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_dim, cfg.hidden_dim, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class FlashNextMoE(nn.Module):
    """Fine-grained MoE: 1 shared + N routed experts, top-k normalized routing."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.num_experts = cfg.num_experts
        self.top_k = cfg.top_k
        self.router = nn.Linear(cfg.hidden_dim, cfg.num_experts, bias=False)
        nn.init.normal_(self.router.weight, std=0.01)
        self.shared_expert = SwiGLUExpert(cfg)
        self.experts = nn.ModuleList([SwiGLUExpert(cfg) for _ in range(cfg.num_experts)])

    def forward(self, x):
        B, S, D = x.shape
        flat_x = x.view(-1, D)
        shared_out = self.shared_expert(flat_x)

        logits = self.router(flat_x)
        scores = F.softmax(logits, dim=-1)
        topk_w, topk_idx = torch.topk(scores, self.top_k, dim=-1)
        topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)

        routed_out = torch.zeros_like(flat_x)
        for idx, expert in enumerate(self.experts):
            mask = (topk_idx == idx).any(dim=-1)
            if mask.any():
                tokens = flat_x[mask]
                match = (topk_idx[mask] == idx).nonzero(as_tuple=True)
                w = topk_w[mask][match[0], match[1]].unsqueeze(-1)
                routed_out[mask] += expert(tokens) * w

        return (shared_out + routed_out).view(B, S, D), topk_idx


# ==========================================
# 2. FULL MODEL
# ==========================================

class MiniFlashNextLM(nn.Module):

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_dim)

        self.mixers = nn.ModuleList()
        self.gr_layers = nn.ModuleList()
        self.moe_layers = nn.ModuleList()

        for i in range(cfg.num_layers):
            if i % 4 == 3:
                self.mixers.append(GatedAttention(cfg))
            else:
                self.mixers.append(GatedDeltaNet(cfg))
            self.gr_layers.append(GatedResidual(cfg))
            self.moe_layers.append(FlashNextMoE(cfg))

        self.norm = RMSNorm(cfg.hidden_dim)
        self.norm_mixer = RMSNorm(cfg.hidden_dim)   # pre-norm before each mixer
        self.norm_moe = RMSNorm(cfg.hidden_dim)     # pre-norm before each MoE
        self.lm_head = nn.Linear(cfg.hidden_dim, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight  # weight tying
        nn.init.normal_(self.embed.weight, std=0.02)

    def forward(self, input_ids):
        B, S = input_ids.shape
        x = self.embed(input_ids)
        branches = x.unsqueeze(2).expand(B, S, 4, self.cfg.hidden_dim).clone()
        routing_logs = []

        for i in range(self.cfg.num_layers):
            mixed_in, branches = self.gr_layers[i](branches, x)
            mixer_out = self.mixers[i](self.norm_mixer(mixed_in))
            mixed_moe_in, branches = self.gr_layers[i](branches, mixer_out)
            moe_out, topk_idx = self.moe_layers[i](self.norm_moe(mixed_moe_in))
            routing_logs.append(topk_idx)
            x = moe_out

        logits = self.lm_head(self.norm(x))
        return logits, routing_logs

    def count_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


# ==========================================
# 3. DATA PIPELINE
# ==========================================

class TinyStoriesStream(IterableDataset):
    """Streams TinyStories, tokenizes to fixed-length sequences."""

    def __init__(self, tokenizer, seq_len, split="train"):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.ds = load_dataset("roneneldan/TinyStories", split=split, streaming=True)

    def __iter__(self):
        for row in self.ds:
            tokens = self.tokenizer(
                row["text"], truncation=True, max_length=self.seq_len + 1,
                padding="max_length", return_tensors="pt"
            )["input_ids"][0]
            yield tokens


def collate_batch(batch):
    """Stack token sequences into a batch tensor."""
    return torch.stack(batch, dim=0)


# ==========================================
# 4. TRAINING LOOP
# ==========================================

def cosine_lr(step, max_steps, warmup, base_lr):
    if step < warmup:
        return base_lr * step / max(1, warmup)
    progress = (step - warmup) / max(1, max_steps - warmup)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


def train(model, mcfg, tcfg):
    print("=" * 60)
    print("TRAINING: Mini Qwen3.8-Flash-Next")
    print("=" * 60)

    total, trainable = model.count_params()
    print(f"Parameters: {total/1e6:.1f}M total, {trainable/1e6:.1f}M trainable")
    print(f"Layers: {mcfg.num_layers} ({sum(1 for i in range(mcfg.num_layers) if i%4!=3)} GDN + {sum(1 for i in range(mcfg.num_layers) if i%4==3)} Attn)")
    print(f"Experts: {mcfg.num_experts} routed + 1 shared, top-{mcfg.top_k}")
    print(f"Hidden: {mcfg.hidden_dim}, Heads: {mcfg.num_heads}, Seq: {tcfg.seq_len}")
    print(f"Steps: {tcfg.max_steps}, Batch: {tcfg.batch_size}×{tcfg.grad_accum}={tcfg.batch_size*tcfg.grad_accum} effective")
    print(f"Device: {next(model.parameters()).device}")
    print()

    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    criterion = nn.CrossEntropyLoss()

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading TinyStories dataset...")
    dataset = TinyStoriesStream(tokenizer, tcfg.seq_len)
    loader = DataLoader(dataset, batch_size=tcfg.batch_size, collate_fn=collate_batch)
    data_iter = iter(loader)

    save_dir = Path(tcfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    step = 0
    accum_loss = 0.0
    accum_count = 0
    t0 = time.time()

    while step < tcfg.max_steps:
        for micro in range(tcfg.grad_accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)

            batch = batch.to(device)
            inputs = batch[:, :-1]
            targets = batch[:, 1:].contiguous()

            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=tcfg.use_amp):
                logits, _ = model(inputs)
                loss = criterion(logits.view(-1, mcfg.vocab_size), targets.view(-1))
                loss = loss / tcfg.grad_accum

            loss.backward()
            accum_loss += loss.item() * tcfg.grad_accum
            accum_count += 1

        # Update LR
        lr = cosine_lr(step, tcfg.max_steps, tcfg.warmup_steps, tcfg.lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Gradient clipping + step
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        optimizer.step()
        optimizer.zero_grad()
        step += 1

        if step % tcfg.log_every == 0:
            avg_loss = accum_loss / accum_count
            elapsed = time.time() - t0
            tok_per_sec = tcfg.batch_size * tcfg.grad_accum * tcfg.seq_len * tcfg.log_every / max(1, elapsed)
            print(f"Step {step:5d}/{tcfg.max_steps} | Loss: {avg_loss:.4f} | LR: {lr:.2e} | {tok_per_sec:.0f} tok/s")
            accum_loss = 0.0
            accum_count = 0
            t0 = time.time()

        if step % tcfg.ckpt_every == 0:
            path = save_dir / f"ckpt_step{step}.pt"
            torch.save({"step": step, "model": model.state_dict(), "config": mcfg.__dict__}, path)
            print(f"  → Checkpoint saved: {path}")

    # Final checkpoint
    final_path = save_dir / "ckpt_final.pt"
    torch.save({"step": step, "model": model.state_dict(), "config": mcfg.__dict__}, final_path)
    print(f"\nFinal checkpoint: {final_path}")
    return model


# ==========================================
# 5. WEIGHT-SPACE EXPERIMENTS
# ==========================================

def run_weight_experiments(model, tokenizer, device):
    """Phases 3-6: Profiling, Pruning, Injection, Task Vector Surgery."""

    print("\n" + "=" * 60)
    print("WEIGHT-SPACE EXPERIMENT 1: Expert Profiling & Pruning")
    print("=" * 60)

    target_moe = model.moe_layers[0]
    expert_counts = torch.zeros(target_moe.num_experts)

    model.eval()
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    data_iter = iter(ds)

    with torch.no_grad():
        for _ in range(50):  # More samples for stable profiling
            row = next(data_iter)
            batch = tokenizer(row["text"], max_length=128, truncation=True,
                            padding="max_length", return_tensors="pt")
            _, routing_logs = model(batch["input_ids"].to(device))
            allocs = routing_logs[0].flatten()
            for idx in allocs:
                expert_counts[idx.item()] += 1

    print(f"Expert Allocation Counts (50 samples): {expert_counts.tolist()}")
    total_tokens = expert_counts.sum().item()
    dist = expert_counts / total_tokens
    print(f"Distribution: {['{:.2f}'.format(d) for d in dist.tolist()]}")
    print(f"Entropy: {-((dist * dist.log()).sum()).item():.4f} nats (max={math.log(target_moe.num_experts):.4f})")

    prune_idx = torch.argmin(expert_counts).item()
    print(f"Pruning Expert {prune_idx} ({expert_counts[prune_idx].item():.0f} allocations, least used)")

    keep = [i for i in range(target_moe.num_experts) if i != prune_idx]
    target_moe.experts = nn.ModuleList([target_moe.experts[i] for i in keep])
    target_moe.router.weight = nn.Parameter(target_moe.router.weight[keep, :])
    target_moe.num_experts = len(keep)
    print(f"Layer 0 now has {target_moe.num_experts} routed experts")

    print("\n" + "=" * 60)
    print("WEIGHT-SPACE EXPERIMENT 2: Expert Injection")
    print("=" * 60)

    # Find the most-used expert to clone from
    clone_idx = torch.argmax(expert_counts[torch.tensor(keep)]).item()
    print(f"Cloning from Expert {keep[clone_idx]} (most active, {expert_counts[keep[clone_idx]].item():.0f} allocations)")

    new_expert = copy.deepcopy(target_moe.experts[clone_idx])
    with torch.no_grad():
        for p in new_expert.parameters():
            p.add_(torch.randn_like(p) * 0.02)

    target_moe.experts.append(new_expert)
    old_w = target_moe.router.weight.data
    new_row = torch.zeros((1, old_w.shape[1]), dtype=old_w.dtype, device=old_w.device)
    target_moe.router.weight = nn.Parameter(torch.cat([old_w, new_row], dim=0))
    target_moe.num_experts += 1
    print(f"Injected new expert → {target_moe.num_experts} routed experts in Layer 0")

    print("\n" + "=" * 60)
    print("WEIGHT-SPACE EXPERIMENT 3: Task Vector Injection")
    print("=" * 60)

    # Compute task vector from two expert weight sets
    print("Computing task vector τ = W_expert_top - W_expert_bottom")
    top_idx = torch.argmax(expert_counts[torch.tensor(keep)]).item()
    bot_idx = 0  # the injected expert (last)
    tau_down = target_moe.experts[top_idx].down_proj.weight.data - target_moe.experts[-1].down_proj.weight.data
    tau_gate = target_moe.experts[top_idx].gate_proj.weight.data - target_moe.experts[-1].gate_proj.weight.data
    tau_up = target_moe.experts[top_idx].up_proj.weight.data - target_moe.experts[-1].up_proj.weight.data

    lambda_scale = 0.3
    print(f"Applying: W_new = W + {lambda_scale} * τ (across gate/up/down projections)")
    with torch.no_grad():
        target_moe.experts[-1].down_proj.weight.add_(tau_down * lambda_scale)
        target_moe.experts[-1].gate_proj.weight.add_(tau_gate * lambda_scale)
        target_moe.experts[-1].up_proj.weight.add_(tau_up * lambda_scale)
    print("Task vector injected into all 3 SwiGLU projections of new expert.")

    print("\n" + "=" * 60)
    print("WEIGHT-SPACE EXPERIMENT 4: Verification")
    print("=" * 60)

    test_texts = [
        "Once upon a time there was a little dog",
        "The princess went to the market to buy",
        "In a small village near the mountain",
    ]
    for text in test_texts:
        tokens = tokenizer(text, return_tensors="pt")
        logits, logs = model(tokens["input_ids"].to(device))
        probs = F.softmax(logits[0, -1], dim=-1)
        top5 = torch.topk(probs, 5)
        preds = tokenizer.decode(top5.indices)
        print(f"  Input: '{text}'")
        print(f"  Top-5 next: {preds}")
        print(f"  Routing L0: {logs[0][0].tolist()}")
        print()

    print("All weight-space experiments completed successfully.")


# ==========================================
# MAIN
# ==========================================

def main():
    device = torch.device("cuda")
    mcfg = ModelConfig()
    tcfg = TrainConfig()

    print("Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    mcfg.vocab_size = len(tokenizer)

    print(f"Building model (vocab={mcfg.vocab_size})...")
    model = MiniFlashNextLM(mcfg).to(device)

    # Phase 2: Training
    model = train(model, mcfg, tcfg)

    # Phases 3-6: Weight-space experiments
    run_weight_experiments(model, tokenizer, device)


if __name__ == "__main__":
    main()
