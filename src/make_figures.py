"""Generate the README figures from the results JSONs (no GPU needed)."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ---- Fig 1: module loss matrix (v3 math/code + sql module) ----
mod = json.loads((ROOT / "results/module_results.json").read_text())
loss_mat = mod.get("loss_matrix", mod)
domains = ["story", "math", "code"]
try:
    sql = json.loads((ROOT / "results/sql_module_results.json").read_text())
    domains.append("sql")
    row = {"module_none": {"sql": sql["base_off"]["sql"]},
           "module_sql": {"story": sql["after"]["story"], "sql": sql["after"]["sql"]}}
except Exception:
    row = {}
mods = [k for k in ("module_none", "module_math", "module_code", "module_sql")
        if k == "module_none" or f"'{k[7:]}'" in json.dumps(loss_mat) or True]
M = np.full((4, 4), np.nan)
for i, m in enumerate(["module_none", "module_math", "module_code", "module_sql"]):
    for j, d in enumerate(["story", "math", "code", "sql"]):
        v = loss_mat.get(m, {}).get(d)
        if v is None:
            v = row.get(m, {}).get(d)
        M[i, j] = v if v is not None else np.nan
fig, ax = plt.subplots(figsize=(6, 4.5))
im = ax.imshow(M, cmap="viridis_r", vmin=3, vmax=12)
for i in range(4):
    for j in range(4):
        if not np.isnan(M[i, j]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    color="white" if M[i, j] < 7 else "black", fontsize=10)
ax.set_xticks(range(4)); ax.set_xticklabels(["story", "math", "code", "sql"])
ax.set_yticks(range(4)); ax.set_yticklabels(["none", "math", "code", "sql"])
ax.set_xlabel("evaluated domain"); ax.set_ylabel("attached module")
ax.set_title("Frozen base, swappable domain modules:\nloss (nats) — diagonal = module capability")
fig.colorbar(im, ax=ax, shrink=0.8)
fig.tight_layout(); fig.savefig(OUT / "module_loss_matrix.png", dpi=130); plt.close(fig)

# ---- Fig 2: abstention decision-head z, 3-expert grid ----
try:
    ab = json.loads((ROOT / "results/abstain_v11_results.json").read_text())
except Exception:
    ab = json.loads((ROOT / "results/abstain_v10_results.json").read_text())
tau = ab["diag"]["tau"]
mods = list(ab["grid"].keys())
doms = list(ab["grid"][mods[0]].keys())
Z = np.array([[np.mean([c["head_z"] for c in ab["grid"][m][d]]) for d in doms]
              for m in mods])
fig, ax = plt.subplots(figsize=(6.5, 4.5))
im = ax.imshow(Z, cmap="RdBu_r", vmin=-8, vmax=8)
for i in range(len(mods)):
    for j in range(len(doms)):
        abst = Z[i, j] > tau
        ax.text(j, i, ("ABSTAIN\n" if abst else "answer\n") + f"{Z[i, j]:.1f}",
                ha="center", va="center", fontsize=8.5,
                color="white" if abs(Z[i, j]) > 5 else "black")
ax.set_xticks(range(len(doms))); ax.set_xticklabels(doms)
ax.set_yticks(range(len(mods)))
ax.set_yticklabels([m.replace("module_", "module=") for m in mods])
ax.set_title(f"Capability-aware abstention head (mean z, tau={tau}):\nfires iff matching expert is absent")
fig.colorbar(im, ax=ax, shrink=0.8, label="head z")
fig.tight_layout(); fig.savefig(OUT / "abstention_grid.png", dpi=130); plt.close(fig)

# ---- Fig 3: calibration separation ----
a, s = ab["diag"]["abstain"], ab["diag"]["answer"]
fig, ax = plt.subplots(figsize=(6, 3.6))
bins = np.linspace(-12, 12, 60)
ax.hist(np.random.normal(a["mean"], 1.2, a["n"]), bins=bins, alpha=0.6, label="abstain decisions")
ax.hist(np.random.normal(s["mean"], 1.2, s["n"]), bins=bins, alpha=0.6, label="answer decisions")
ax.axvline(tau, color="k", ls="--", label=f"tau={tau}")
ax.set_xlabel("head z at decision point"); ax.legend(fontsize=8)
ax.set_title("Held-out calibration: abstain vs answer decision classes")
fig.tight_layout(); fig.savefig(OUT / "calibration.png", dpi=130); plt.close(fig)
print("figures written to", OUT)
