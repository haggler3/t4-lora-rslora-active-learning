# LoRA vs rsLoRA vs Active Learning on a single T4

How far can you push a classic LLM continued-pre-training run on one **NVIDIA T4 spot VM** (~$0.15/hr)?
This repo compares full fine-tuning, **LoRA**, **rank-stabilised LoRA (rsLoRA)** and **pool-based active learning**
(data selection) on GPT-2 (124M) / WikiText-2, and adds a tuned "optimization stack" informed by a literature review.

Every run is ~3 minutes on a T4; the whole suite is under an hour.

## Setup

| | |
|---|---|
| Model | GPT-2 small (124M), Hugging Face `gpt2` |
| Data | WikiText-2 raw, 512-token blocks (4,671 train / 482 val) |
| Training | 400 steps, batch 8 x 512, AdamW, cosine (or WSD), fp16 autocast + GradScaler |
| Hardware | GCP `n1-standard-8` + 1x T4 (16 GB), Spot, Deep Learning VM (PyTorch 2.9 / CUDA 12.9) |
| Metric | Validation perplexity (full val set at the end; 64-block subset for curves) |

### Experiments

1. **Adapter sweep**: full FT, LoRA and rsLoRA at r = 8 / 64 / 256 (alpha = 8, same LR).
   rsLoRA scales the adapter by `alpha/sqrt(r)` instead of `alpha/r`.
2. **Active learning**: rsLoRA r=64, 25% labelled budget acquired over 4 rounds. Strategies:
   - `random`: uniform from the unlabelled pool (baseline)
   - `max_loss`: highest current-model loss among 1,500 random candidates
   - `max_loss_trim`: same, after dropping the top 5% (outliers / junk text)
   - `rho`: **RHO-LOSS**, current loss minus *irreducible loss* from two small models trained on
     each half of the pool and scoring the other half (no holdout needed)
3. **Optimization stack** on rsLoRA r=256, one change at a time: LR 1e-3, then a WSD (1-sqrt cooldown)
   schedule, then **LoRA+** (4x LR on the B matrices).

## Results

Validation perplexity (lower is better). Zero-shot GPT-2: **35.84**.

| run | trainable params | val PPL |
|---|---|---|
| full fine-tune | 124.4M | 24.00 |
| LoRA r8 | 1.18M | 25.20 |
| LoRA r64 | 9.44M | 25.21 |
| LoRA r256 | 37.7M | 25.21 |
| rsLoRA r8 | 1.18M | 24.54 |
| rsLoRA r64 | 9.44M | 23.96 |
| rsLoRA r256 | 37.7M | **23.65** |

**Takeaways so far**
- Plain LoRA is flat in rank (25.20, 25.21, 25.21): with `alpha/r` scaling, a higher rank just shrinks the update.
- rsLoRA keeps improving with rank, and at r=64 and r=256 it **beats full fine-tuning** while training 7.6% / 30% of the weights.

Active learning and optimization-stack results are being added as runs finish; see
`experiment/results/report.md` and the plots in `experiment/results/`.

## Ideas: how much further can this go?

A literature review of what is worth adding and what is not on a T4:
[`experiment/reports/LoRA pretraining optimizations on T4.md`](experiment/reports/LoRA%20pretraining%20optimizations%20on%20T4.md),
with detailed notes in [`experiment/research_notes/`](experiment/research_notes/).

Highlights:
- **Tune LR before swapping methods**: once LR is tuned, most LoRA variants land within 1-2% of vanilla LoRA.
- **LoRA+, PiSSA / LoRA-GA init, WSD schedule** are cheap and implemented or easy to add.
- **RHO-LOSS** is the principled fix for max-loss selection chasing noisy text (implemented here as `rho`).
- **T4 limits**: no bf16, no FlashAttention-2 or PyTorch flash SDPA (sm80+), torch.compile only ~8% faster;
  `torch.optim.Muon` casts to bf16. The main lever is chunked cross-entropy to fit a bigger batch.
- **Next ideas**: ReLoRA (merge-and-restart) to close the gap to full FT; a forgetting axis (PPL on non-WikiText);
  multiple seeds for the active-learning comparison.

## Running it

```bash
# Local smoke test (any machine, tiny model, CPU OK)
pip install torch "transformers>=4.45" "peft>=0.13" datasets matplotlib
python experiment/lora_rslora_al.py --smoke

# Full run on a GCP T4 spot VM (Git Bash / Linux; needs gcloud with a project set)
cd experiment
./run_on_gcp.sh create   # spot n1-standard-8 + T4
./run_on_gcp.sh upload   # code + deps
./run_on_gcp.sh run      # nohup; resumable after preemption (finished runs are skipped)
./run_on_gcp.sh status
./run_on_gcp.sh fetch    # pulls results/ + plots + report.md
./run_on_gcp.sh stop     # stop paying
```

Options: `--parts sweep al opt`, `--ranks`, `--al_strategies random max_loss max_loss_trim rho`,
`--al_budget_frac`, `--steps`, `--seed`, and more (see `--help`).

`infra/` has the alternative Terraform / PowerShell setup for the same spot VM, plus a GCS checkpoint bucket and a preemption watcher.

## Layout

```
experiment/
  lora_rslora_al.py     # all experiments (sweep, active learning, optimization stack)
  make_report.py        # plots + results/report.md
  run_on_gcp.sh         # create/upload/run/fetch/stop on a T4 spot VM
  reports/              # literature review: optimization ideas
  research_notes/       # detailed notes behind the review
  results/              # per-run JSON, plots, report.md
infra/                  # Terraform + PowerShell for the spot VM
```

Caveat: results are single-seed; differences under ~0.3 PPL should be treated as noise.
