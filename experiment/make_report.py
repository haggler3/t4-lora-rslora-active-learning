"""Turn results/*.json into plots + report.md."""
import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = sys.argv[1] if len(sys.argv) > 1 else "results"
runs = {}
for f in glob.glob(os.path.join(out, "*.json")):
    r = json.load(open(f))
    runs[r["name"]] = r
zs = runs.pop("zero_shot", {}).get("final_val_ppl")
sweep = {k: v for k, v in runs.items() if not k.startswith(("al_", "opt_"))}
al = {k: v for k, v in runs.items() if k.startswith("al_")}
opt = {k: v for k, v in runs.items() if k.startswith("opt_")}


def steps_to(r, target):
    """First eval step at which the quick-eval PPL is <= target (None if never)."""
    return next((s for s, p in zip(r["log"]["eval_step"], r["log"]["eval_ppl"]) if p <= target), None)
COL = {"full": "#444444", "lora": "#2f6fdf", "rslora": "#e0731a"}


def curve(ax, r, **kw):
    ax.plot(r["log"]["eval_step"], r["log"]["eval_ppl"], marker="o", ms=3, **kw)


# 1. Final PPL vs rank: LoRA vs rsLoRA
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
ax = axes[0]
for m in ("lora", "rslora"):
    pts = sorted((v["r"], v["final_val_ppl"]) for v in sweep.values() if v["method"] == m)
    if pts:
        ax.plot(*zip(*pts), marker="o", label=m, color=COL[m])
if "full_ft" in sweep:
    ax.axhline(sweep["full_ft"]["final_val_ppl"], ls="--", color=COL["full"], label="full fine-tune")
if zs:
    ax.axhline(zs, ls=":", color="gray", label="zero-shot GPT-2")
ax.set_xscale("log", base=2)
ax.set_xlabel("rank r")
ax.set_ylabel("WikiText-2 val perplexity")
ax.set_title("Final perplexity vs rank (lower is better)")
ax.legend()
ax = axes[1]
for k, v in sorted(sweep.items()):
    ls = "-" if v["method"] != "lora" else "--"
    alpha = {8: 0.45, 64: 0.75, 256: 1.0}.get(v["r"], 1.0)
    curve(ax, v, label=k, color=COL[v["method"]], ls=ls, alpha=alpha)
ax.set_xlabel("step")
ax.set_ylabel("val perplexity (64-block subset)")
ax.set_title("Learning curves")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(out, "sweep.png"), dpi=130)

# 2. Active learning curves
if al:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for k, v in sorted(al.items()):
        curve(ax, v, label=f"{k} ({v['unique_blocks_seen']} blocks)")
    ref = f"{al[next(iter(al))]['method']}_r{al[next(iter(al))]['r']}"
    if ref in sweep:
        curve(ax, sweep[ref], label=f"{ref} full data ({sweep[ref]['unique_blocks_seen']} blocks)", color="k", ls=":")
    ax.set_xlabel("step")
    ax.set_ylabel("val perplexity (64-block subset)")
    ax.set_title("Active data selection at 25% budget")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "active_learning.png"), dpi=130)

# 3. Optimization stack vs baselines
if opt:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for k in ("full_ft", "rslora_r256", "lora_r8"):
        if k in sweep:
            curve(ax, sweep[k], label=f"{k} (baseline)", ls=":", color={"full_ft": "k", "rslora_r256": COL["rslora"], "lora_r8": COL["lora"]}[k])
    for k, v in sorted(opt.items()):
        curve(ax, v, label=k)
    ax.set_xlabel("step")
    ax.set_ylabel("val perplexity (64-block subset)")
    ax.set_title("Tuned recipe stack on rsLoRA r=256")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "optimizations.png"), dpi=130)

# 4. Markdown table
lines = ["# LoRA / rsLoRA / Active Learning on a T4", ""]
if zs:
    lines += [f"Zero-shot GPT-2 val PPL: **{zs:.2f}**", ""]
target_run = sweep.get("lora_r8")
target = target_run["log"]["eval_ppl"][-1] if target_run else None
lines += ["Steps-to-target = first step whose quick-eval PPL matches the final quick-eval PPL of `lora_r8`.", "",
          "| run | method | r | trainable params | unique blocks | final val PPL | steps to target | train time (s) | tok/s | peak mem (GB) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
for k, v in sorted(runs.items(), key=lambda kv: kv[1]["final_val_ppl"]):
    lines.append(f"| {k} | {v['method']} | {v['r']} | {v['trainable_params']:,} | {v['unique_blocks_seen']} | "
                 f"{v['final_val_ppl']:.2f} | {steps_to(v, target) if target else '-'} | {v['train_time_s']:.0f} | {v['tokens_per_s']:.0f} | {v['peak_mem_gb']:.2f} |")
lines += ["", "![sweep](sweep.png)", "", "![active learning](active_learning.png)", "", "![optimizations](optimizations.png)", ""]
open(os.path.join(out, "report.md"), "w").write("\n".join(lines))
print("\n".join(lines))
