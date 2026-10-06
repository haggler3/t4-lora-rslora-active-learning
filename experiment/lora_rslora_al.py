"""
LoRA vs rsLoRA vs full fine-tuning + active data selection, sized for one T4 (16 GB).

Task: continued pre-training (causal LM) of GPT-2 (124M) on WikiText-2, 512-token blocks.

Part A - adapter sweep (same steps, batch, seed):
    full_ft, lora_r{8,64,256}, rslora_r{8,64,256}
    rsLoRA scales the adapter by alpha/sqrt(r) instead of alpha/r, so it should keep
    improving as rank grows while plain LoRA flattens out (Kalajdzievski, 2023).

Part B - active learning (pool-based data selection) with rsLoRA r=64:
    Fixed labelled budget (25% of the training blocks), acquired over R rounds.
    Strategies: random | max_loss (pick highest current-model loss)
                | max_loss_trim (highest loss after dropping the top 5% outliers)
    Same optimizer/schedule/total steps as Part A, so the full-data rsLoRA r=64 run
    is the upper bound.

Every finished run is written to results/<run>.json and skipped on restart, so a
spot preemption only costs the run in progress.
"""
import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_blocks(tokenizer, block_size, cache_dir):
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", cache_dir=cache_dir)
    out = {}
    for split in ("train", "validation"):
        text = "".join(ds[split]["text"])
        ids = tokenizer(text, return_attention_mask=False)["input_ids"]
        n = len(ids) // block_size
        out[split] = torch.tensor(ids[: n * block_size], dtype=torch.long).view(n, block_size)
    return out["train"], out["validation"]


def make_model(cfg, args, device):
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    if cfg["method"] == "full":
        return model
    lcfg = LoraConfig(
        r=cfg["r"],
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        target_modules=["c_attn", "c_proj", "c_fc"],
        fan_in_fan_out=True,  # GPT-2 uses Conv1D
        use_rslora=cfg["method"] == "rslora",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lcfg)


@torch.no_grad()
def per_block_loss(model, blocks, device, bs=32):
    model.eval()
    losses = []
    for i in range(0, len(blocks), bs):
        x = blocks[i : i + bs].to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
            logits = model(input_ids=x).logits[:, :-1].float()
        tok = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), x[:, 1:].reshape(-1), reduction="none"
        ).view(x.size(0), -1)
        losses.append(tok.mean(1).cpu())
    model.train()
    return torch.cat(losses)


def eval_ppl(model, blocks, device):
    return math.exp(per_block_loss(model, blocks, device).mean().item())


def wsd_schedule(opt, warmup, total_steps, decay_frac=0.2):
    """Warmup-stable-decay with a (1 - sqrt) cooldown over the last decay_frac of steps (Hagele et al. 2024)."""
    decay_start = int(total_steps * (1 - decay_frac))

    def f(step):
        if step < warmup:
            return (step + 1) / warmup
        if step < decay_start:
            return 1.0
        return 1.0 - math.sqrt((step - decay_start) / max(1, total_steps - decay_start))

    return torch.optim.lr_scheduler.LambdaLR(opt, f)


class Trainer:
    def __init__(self, model, lr, total_steps, warmup, device, lora_plus=1.0, schedule="cosine"):
        self.model, self.device = model, device
        named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        self.n_trainable = sum(p.numel() for _, p in named)
        self.n_total = sum(p.numel() for p in model.parameters())
        # LoRA+ (Hayou et al. 2024): B matrices get a lora_plus times larger LR than A.
        groups = [
            {"params": [p for n, p in named if "lora_B" not in n], "lr": lr},
            {"params": [p for n, p in named if "lora_B" in n], "lr": lr * lora_plus},
        ]
        groups = [g for g in groups if g["params"]]
        self.opt = torch.optim.AdamW(groups, weight_decay=0.0, betas=(0.9, 0.98), fused=device == "cuda")
        if schedule == "wsd":
            self.sched = wsd_schedule(self.opt, warmup, total_steps)
        else:
            self.sched = get_cosine_schedule_with_warmup(self.opt, warmup, total_steps)
        self.scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
        self.step = 0
        model.train()

    def train_step(self, x):
        x = x.to(self.device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.device == "cuda"):
            loss = self.model(input_ids=x, labels=x).loss
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.opt)
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad], 1.0
        )
        self.scaler.step(self.opt)
        self.scaler.update()
        self.opt.zero_grad(set_to_none=True)
        self.sched.step()
        self.step += 1
        return loss.item()


def run_training(name, cfg, args, train_blocks, val_blocks, device, select_fn=None):
    """select_fn(model, round_idx) -> tensor of block indices to train on (active learning)."""
    set_seed(args.seed)
    model = make_model(cfg, args, device)
    lr = cfg.get("lr") or (args.full_lr if cfg["method"] == "full" else args.lora_lr)
    tr = Trainer(model, lr, args.steps, args.warmup, device,
                 lora_plus=cfg.get("lora_plus", 1.0), schedule=cfg.get("schedule", "cosine"))
    val_quick = val_blocks[: args.quick_eval_blocks]
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    log = {"step": [], "train_loss": [], "eval_step": [], "eval_ppl": []}
    log["eval_step"].append(0)
    log["eval_ppl"].append(eval_ppl(model, val_quick, device))
    rounds = args.al_rounds if select_fn else 1
    steps_per_round = args.steps // rounds
    selected = torch.arange(len(train_blocks))
    selection_log = []
    t0 = time.time()
    select_time = 0.0
    g = torch.Generator().manual_seed(args.seed)
    for rnd in range(rounds):
        if select_fn:
            ts = time.time()
            selected, info = select_fn(model, rnd)
            select_time += time.time() - ts
            selection_log.append(info)
        for _ in range(steps_per_round):
            idx = selected[torch.randint(len(selected), (args.batch_size,), generator=g)]
            loss = tr.train_step(train_blocks[idx])
            if not math.isfinite(loss):
                print(f"[{name}] non-finite loss at step {tr.step}, stopping run")
                break
            log["step"].append(tr.step)
            log["train_loss"].append(loss)
            if tr.step % args.eval_every == 0:
                ppl = eval_ppl(model, val_quick, device)
                log["eval_step"].append(tr.step)
                log["eval_ppl"].append(ppl)
                print(f"[{name}] step {tr.step:4d}  train {loss:.3f}  val_ppl {ppl:.2f}", flush=True)
    train_time = time.time() - t0
    final_ppl = eval_ppl(model, val_blocks, device)
    res = {
        "name": name,
        **cfg,
        "seed": args.seed,
        "final_val_ppl": final_ppl,
        "trainable_params": tr.n_trainable,
        "total_params": tr.n_total,
        "train_time_s": train_time,
        "select_time_s": select_time,
        "tokens_per_s": tr.step * args.batch_size * args.block_size / max(train_time - select_time, 1e-6),
        "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0,
        "unique_blocks_seen": len(selected) if select_fn else min(len(train_blocks), args.steps * args.batch_size),
        "selection": selection_log,
        "log": log,
    }
    print(f"[{name}] FINAL val_ppl {final_ppl:.2f}  ({train_time:.0f}s, {tr.n_trainable:,} trainable)", flush=True)
    del model, tr
    if device == "cuda":
        torch.cuda.empty_cache()
    return res


def irreducible_loss(train_blocks, args, device):
    """RHO-LOSS irreducible loss without a holdout (Mindermann et al. 2022): train a small
    rsLoRA r8 model on each half of the pool and score the *other* half with it."""
    path = os.path.join(args.out, "il_loss.pt")
    if os.path.exists(path):
        return torch.load(path)
    perm = torch.randperm(len(train_blocks), generator=torch.Generator().manual_seed(args.seed))
    halves = perm.chunk(2)
    il = torch.empty(len(train_blocks))
    g = torch.Generator().manual_seed(args.seed + 1)
    t0 = time.time()
    for fit, score in ((halves[0], halves[1]), (halves[1], halves[0])):
        set_seed(args.seed)
        model = make_model({"method": "rslora", "r": 8}, args, device)
        tr = Trainer(model, args.lora_lr, args.il_steps, args.warmup, device)
        for _ in range(args.il_steps):
            tr.train_step(train_blocks[fit[torch.randint(len(fit), (args.batch_size,), generator=g)]])
        il[score] = per_block_loss(model, train_blocks[score], device)
        del model, tr
    print(f"[rho] irreducible-loss models trained + pool scored in {time.time() - t0:.0f}s", flush=True)
    torch.save(il, path)
    return il


def make_selector(strategy, train_blocks, args, device):
    """Pool-based acquisition: each round adds budget/rounds new blocks to the labelled set."""
    n_pool = len(train_blocks)
    budget = int(args.al_budget_frac * n_pool)
    per_round = budget // args.al_rounds
    rng = np.random.default_rng(args.seed)
    labelled = []
    il = irreducible_loss(train_blocks, args, device).numpy() if strategy == "rho" else None

    def select(model, rnd):
        unlabelled = np.setdiff1d(np.arange(n_pool), np.array(labelled, dtype=int))
        if strategy == "random":
            pick = rng.choice(unlabelled, per_round, replace=False)
            info = {"round": rnd}
        else:
            # Score a random candidate subset of the pool to bound the cost per round.
            cand = rng.choice(unlabelled, min(args.al_candidates, len(unlabelled)), replace=False)
            scores = per_block_loss(model, train_blocks[torch.from_numpy(cand)], device).numpy()
            if strategy == "rho":
                # Reducible loss: high = still learnable, not just noisy/unlearnable.
                scores = scores - il[cand]
            order = np.argsort(-scores)
            if strategy == "max_loss_trim":
                order = order[int(args.al_trim_frac * len(order)) :]
            pick = cand[order[:per_round]]
            info = {
                "round": rnd,
                "cand_loss_mean": float(scores.mean()),
                "picked_loss_mean": float(scores[order[:per_round]].mean()),
            }
        labelled.extend(pick.tolist())
        info["labelled"] = len(labelled)
        return torch.tensor(labelled, dtype=torch.long), info

    return select


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gpt2")
    p.add_argument("--block_size", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--lora_lr", type=float, default=3e-4)
    p.add_argument("--full_lr", type=float, default=5e-5)
    p.add_argument("--lora_alpha", type=float, default=8)
    p.add_argument("--ranks", type=int, nargs="+", default=[8, 64, 256])
    p.add_argument("--eval_every", type=int, default=50)
    p.add_argument("--quick_eval_blocks", type=int, default=64)
    p.add_argument("--al_rounds", type=int, default=4)
    p.add_argument("--al_budget_frac", type=float, default=0.25)
    p.add_argument("--al_candidates", type=int, default=1500)
    p.add_argument("--al_trim_frac", type=float, default=0.05)
    p.add_argument("--al_method", default="rslora")
    p.add_argument("--al_rank", type=int, default=64)
    p.add_argument("--al_strategies", nargs="+", default=["random", "max_loss", "max_loss_trim", "rho"])
    p.add_argument("--il_steps", type=int, default=150, help="steps per irreducible-loss model (rho)")
    p.add_argument("--opt_rank", type=int, default=256)
    p.add_argument("--opt_lr", type=float, default=1e-3)
    p.add_argument("--opt_lora_plus", type=float, default=4.0)
    p.add_argument("--parts", nargs="+", default=["sweep", "al", "opt"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results")
    p.add_argument("--smoke", action="store_true", help="tiny CPU-friendly run to check the pipeline")
    args = p.parse_args()

    if args.smoke:
        args.model, args.block_size, args.steps, args.eval_every = "sshleifer/tiny-gpt2", 64, 8, 4
        args.ranks, args.al_rounds, args.al_candidates, args.quick_eval_blocks = [4], 2, 64, 8
        args.il_steps, args.opt_rank = 4, 4
        args.out = args.out + "_smoke"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        print("GPU:", torch.cuda.get_device_name(0), flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    train_blocks, val_blocks = build_blocks(tok, args.block_size, os.path.join(args.out, ".hf_cache"))
    if args.smoke:
        train_blocks, val_blocks = train_blocks[:512], val_blocks[:32]
    print(f"train blocks {len(train_blocks)}, val blocks {len(val_blocks)}", flush=True)

    jobs = []
    if "sweep" in args.parts:
        jobs.append(("full_ft", {"method": "full", "r": 0}, None))
        for m in ("lora", "rslora"):
            for r in args.ranks:
                jobs.append((f"{m}_r{r}", {"method": m, "r": r}, None))
    if "al" in args.parts:
        for s in args.al_strategies:
            jobs.append((f"al_{s}", {"method": args.al_method, "r": args.al_rank, "al_strategy": s}, s))
    if "opt" in args.parts:
        # Tuned-recipe stack on high-rank rsLoRA, one change at a time: LR -> WSD schedule -> LoRA+.
        base = {"method": "rslora", "r": args.opt_rank, "lr": args.opt_lr}
        jobs.append(("opt_lr", base, None))
        jobs.append(("opt_lr_wsd", {**base, "schedule": "wsd"}, None))
        jobs.append(("opt_lr_wsd_loraplus", {**base, "schedule": "wsd", "lora_plus": args.opt_lora_plus}, None))

    # Zero-shot reference.
    zs_path = os.path.join(args.out, "zero_shot.json")
    if not os.path.exists(zs_path):
        m = AutoModelForCausalLM.from_pretrained(args.model).to(device)
        json.dump({"name": "zero_shot", "final_val_ppl": eval_ppl(m, val_blocks, device)}, open(zs_path, "w"))
        del m

    for name, cfg, strategy in jobs:
        path = os.path.join(args.out, f"{name}.json")
        if os.path.exists(path):
            print(f"[{name}] already done, skipping", flush=True)
            continue
        sel = make_selector(strategy, train_blocks, args, device) if strategy else None
        res = run_training(name, cfg, args, train_blocks, val_blocks, device, select_fn=sel)
        res["args"] = vars(args)
        tmp = path + ".tmp"
        json.dump(res, open(tmp, "w"), indent=1)
        os.replace(tmp, path)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
