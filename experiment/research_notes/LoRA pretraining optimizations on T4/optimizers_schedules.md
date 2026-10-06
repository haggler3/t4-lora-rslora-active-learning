# Optimizer, LR-Schedule and Training-Recipe Improvements for Short Continued Pre-training of Small Transformers (GPT-2 124M on WikiText-2, T4, 400 steps)

Context: current recipe = AdamW (0.9/0.98, wd 0), cosine with 20 warmup steps, LR 5e-5 full FT / 3e-4 LoRA, clip 1.0, fp16 on T4. Zero-shot val PPL 35.84, full FT 24.00, LoRA r8 25.20.

## Q1. Muon: pre-training vs fine-tuning, full FT vs LoRA, fp16 caveats

### Takeaway
Muon gives a real ~1.3-1.35x step speedup for *from-scratch* pre-training at ~100M scale, but naively applying it to full fine-tuning of an Adam-pretrained checkpoint (GPT-2 was Adam-trained) tends to *hurt* ("optimizer mismatch"); under LoRA it roughly matches Adam with half the optimizer state, so it is not a speed win for this experiment. `torch.optim.Muon` exists (torch >= 2.9) but hard-casts Newton-Schulz to bfloat16, which the T4 lacks in hardware.

### Cited Findings
- Muon = SGD-momentum update post-processed by a 5-step Newton-Schulz orthogonalization; for 2D hidden-layer weights only. Embeddings, LM head, scalars/vectors should use AdamW (embedding and head empirically better with AdamW even though 2D). NS "can be stably run in bfloat16". — [Keller Jordan, Muon blog](https://kellerjordan.github.io/posts/muon/)
- Reported speedups: NanoGPT speedrun 1.35x; CIFAR-10 record 3.3 -> 2.6 A100-s; 1.5B model reached GPT-2 XL level in 10 h vs 13.3 h with AdamW. Author explicitly lists as open question whether Muon works for fine-tuning/RL. — [Keller Jordan, Muon blog](https://kellerjordan.github.io/posts/muon/)
- `torch.optim.Muon(params, lr=1e-3, weight_decay=0.1, momentum=0.95, nesterov=True, ns_coefficients=(3.4445,-4.775,2.0315), eps=1e-7, ns_steps=5, adjust_lr_fn=None)`; `adjust_lr_fn` in {"original" (scale sqrt(max(1,A/B))), "match_rms_adamw" (0.2*sqrt(max(A,B)), lets you reuse AdamW LR/wd), "spectral_unclamped"}. Only for 2D params; use AdamW for bias/embeddings. — [PyTorch 2.9 docs](https://docs.pytorch.org/docs/2.9/generated/torch.optim.Muon.html)
- The torch 2.9 implementation casts to bf16 inside NS: `ortho_grad = grad.bfloat16()`, and raises if the gradient is not 2D. — [PyTorch v2.9.0 source, torch/optim/_muon.py](https://raw.githubusercontent.com/pytorch/pytorch/v2.9.0/torch/optim/_muon.py)
- Moonshot "Muon is Scalable": two fixes needed at scale -- add weight decay (lambda=0.1) and per-matrix update-RMS matching (0.2 * O_t * sqrt(max(A,B)), matching AdamW's typical 0.2-0.4 update RMS) -- giving "~2x computational efficiency compared to AdamW with compute optimal training" (Moonlight 3B/16B MoE, 5.7T tokens; large-scale, from-scratch claim). — [Liu et al. 2025, arXiv 2502.16982](https://arxiv.org/abs/2502.16982); [HTML](https://arxiv.org/html/2502.16982)
- Moonshot SFT results: Muon-SFT on AdamW-pretrained Qwen2.5-7B was at parity or worse vs Adam (MMLU 70.8 vs 71.4; GSM8K 85.8 vs 89.8). Best combo was Muon-pretrained + Muon-finetuned (GSM8K 68.0 vs 64.6 for AdamW/AdamW) -- an optimizer-mismatch effect. — [arXiv 2502.16982 HTML](https://arxiv.org/html/2502.16982)
- "Can Muon Fine-tune Adam-Pretrained Models?" (ICML 2026): naively switching to Muon for fine-tuning an Adam-pretrained model degrades performance due to differing implicit biases; LoRA mitigates this -- LoRA+Muon matches or beats LoRA+Adam across language and vision tasks, saving ~50% optimizer-state memory; mismatch severity correlates with update strength (rank, forgetting); tune Muon LR separately. — [arXiv 2605.10468](https://arxiv.org/abs/2605.10468); [HF papers](https://huggingface.co/papers/2605.10468)
- Same paper notes the optimal Muon update is typically full-rank orthogonal, which cannot be represented within LoRA's low-rank constraint. — [search summary of arXiv 2605.10468](https://arxiv.org/pdf/2605.10468)
- MuonAll is a Muon variant proposed specifically for efficient fine-tuning of LLMs (not inspected in detail). — [arXiv 2511.06086](https://arxiv.org/html/2511.06086v1)
- "Fantastic Pretraining Optimizers": Muon ~1.35x over well-tuned AdamW at 130M / 1x Chinchilla, shrinking to ~1.1x at 1.2B; from-scratch pretraining only. — [Wen et al. 2025, arXiv 2509.02046](https://arxiv.org/html/2509.02046)

### Inferences
- For GPT-2 full FT (Adam-pretrained, 400 steps), expect Muon to be neutral-to-negative; it is a risky demo. If you try it, apply Muon only to the transformer-block 2D weights (HF GPT-2 `Conv1D` weights `c_attn`, `c_proj`, `c_fc` are 2D, stored transposed as (in, out) -- orthogonalization is transpose-invariant so this is fine), keep `wte`/`wpe`/LN/biases on AdamW, and use `adjust_lr_fn="match_rms_adamw"` so the AdamW LR (5e-5) is a sane starting point.
- `c_attn` is a fused QKV matrix (768x2304); orthogonalizing it jointly vs per Q/K/V is a design choice the sources don't address for fine-tuning.
- On LoRA: Muon on A (r x d) and B (d x r) is legal (2D) and the 2026 paper says parity with Adam; benefit is memory, not speed. Not worth demoing for speed.
- fp16/T4: torch's built-in Muon calls `.bfloat16()`. The T4 (sm_75) has no bf16 tensor cores, so this will either be slow (emulated) or fail depending on the CUDA/cuBLAS build -- unverified. Safer: copy Keller Jordan's ~10-line `zeropower_via_newtonschulz5` and run NS in fp32 (matrices are tiny at 124M, cost negligible) rather than fp16, because fp16's 10-bit mantissa is worse than bf16 for this iteration though its range is fine after the Frobenius-norm normalization (inference, not sourced).
- Also: with `GradScaler` in fp16, gradients must be unscaled (`scaler.unscale_(opt)`) before Muon's NS, otherwise the normalization makes the scale irrelevant but momentum mixing across steps with changing scales would be wrong (inference).

### Gaps
- No source found that measures Muon on GPT-2-124M *continued* pre-training (language-modeling loss on new domain text) specifically; the evidence is SFT/instruction tasks and from-scratch pretraining.
- No authoritative statement found on fp16 Newton-Schulz stability on pre-Ampere GPUs.

## Q2. Schedule-Free AdamW, WSD/trapezoidal, warmup; SOAP, Shampoo, Sophia, Adam-mini, Lion, Cautious -- which hold up in reproduced benchmarks?

### Takeaway
In the most careful 2025 benchmark (Wen et al., "Fantastic Pretraining Optimizers"), only matrix preconditioners (Muon, SOAP, Kron) give ~1.3x at 130M; Cautious/NAdamW/MARS give ~1.1x; Lion, Adam-mini and Sophia give ~nothing. The biggest "free" gains come from tuning AdamW LR/weight decay/warmup and the schedule itself. Schedule-Free AdamW won the AlgoPerf self-tuning track and is a drop-in package; WSD with a (1-sqrt) cooldown matches or beats cosine and is the speedrun's schedule.

### Cited Findings
- Fantastic Pretraining Optimizers (Wen, Hall, Ma, Liang; ICLR 2026): 10+ optimizers, 0.1B-1.2B, 1x-8x Chinchilla, each carefully tuned. Matrix-based optimizers speedup ~1.4x at 0.1B falling to ~1.1x at 1.2B; "actual speedup of many proposed optimizers over well-tuned baselines is lower than claimed". Muon best at 1x Chinchilla; SOAP overtakes at >=8x data-to-model ratio. — [arXiv 2509.02046](https://arxiv.org/abs/2509.02046); [ICLR 2026 poster](https://www.iclr.cc/virtual/2026/poster/10011758)
- Approximate speedups vs tuned AdamW at 130M/1x Chinchilla (read off Fig. 3 by the fetch tool -- treat as approximate): Muon ~1.35x, SOAP ~1.30x, Kron ~1.28x, Scion ~1.25x, MARS ~1.15x, NAdamW ~1.12x, Cautious ~1.10x, Lion ~0.98x, Adam-mini ~0.97x, Sophia minimal. Lion and Adam-mini "slowdown of at most 5%". — [arXiv 2509.02046 HTML](https://arxiv.org/html/2509.02046)
- Same paper: tuning just one hyperparameter of the standard GPT-3 recipe can give a 2x speedup; weight decay is important for all optimizers (AdamW ~0.1, Lion ~0.6); warmup is scale-sensitive and needs re-tuning; rankings can flip mid-run because of LR decay (so compare at end of schedule). — [arXiv 2509.02046 HTML](https://arxiv.org/html/2509.02046)
- Cautious optimizers: one-line mask (only apply update components whose sign agrees with the current gradient) to any momentum optimizer (C-AdamW, C-Lion); up to 1.47x on Llama and MAE pretraining in the authors' runs; code at github kyleliang919/C-Optim. — [Liang et al., arXiv 2411.16085](https://arxiv.org/abs/2411.16085); independent benchmark shows only ~1.1x ([Wen et al.](https://arxiv.org/html/2509.02046)). Cautious weight decay appears in the modded-nanogpt speedrun records. — [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt)
- Schedule-Free AdamW (Defazio et al. 2024) won the AlgoPerf self-tuning ruleset. — [Kasimbeg et al., ICLR 2025 AlgoPerf analysis](https://proceedings.iclr.cc/paper_files/paper/2025/hash/6bdde0373d53d4a501249547084bed43-Abstract-Conference.html)
- Schedule-Free usage: `pip install schedulefree`; `schedulefree.AdamWScheduleFree(params, lr=..., warmup_steps=...)`; must call `optimizer.train()` / `optimizer.eval()` alongside `model.train()` / `model.eval()` (eval uses the averaged point) and save checkpoints in eval mode; LR "1x-10x larger than with schedule-based approaches"; beta default 0.9, 0.95-0.98 for longer runs; warmup recommended; README says it still "does require tuning". — [facebookresearch/schedule_free](https://github.com/facebookresearch/schedule_free)
- WSD / constant + cooldown (Hägele et al., NeurIPS 2024): constant LR then short aggressive cooldown matches or beats cosine; the (1-sqrt) cooldown shape consistently beats linear cooldown, both beating untuned cosine for long runs; lets one run be reused for many budgets. Code: github epfml/schedules-and-scaling. — [arXiv 2405.18392](https://arxiv.org/pdf/2405.18392); [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/8b970e15a89bf5d12542810df8eae8fc-Abstract.html)
- modded-nanogpt uses a WSD/trapezoidal schedule with a cooldown fraction, decaying to a floor (0.1x rather than 0) in later records. — [modded-nanogpt README](https://github.com/KellerJordan/modded-nanogpt)

### Inferences
- For a 400-step fine-tune of an Adam-trained checkpoint, the cheapest reliable wins are recipe-level: (1) LR sweep (Q4), (2) WSD with ~20-30% (1-sqrt) cooldown in place of cosine, (3) Schedule-Free AdamW at ~2-5x the current LR, (4) C-AdamW one-line mask. Expect single-digit-to-~10% fewer steps to target loss from (2)-(4); none of these numbers are measured on fine-tuning in the sources -- all are from-scratch pretraining evidence.
- SOAP/Shampoo/Kron: small at 124M but extra code and per-step cost; on an Adam-pretrained checkpoint they share Muon's mismatch risk (inference by analogy). Skip for a 3-minute demo.
- Lion, Adam-mini, Sophia: no evidence of speedup in fair benchmarks; skip.
- Current betas (0.9, 0.98) are reasonable for short runs; wd=0 is fine for 400-step fine-tuning (the wd findings are for long pretraining).
- 20 warmup steps of 400 (5%) is in the normal range; Schedule-Free also wants warmup.

### Gaps
- No AlgoPerf-style independent benchmark of Schedule-Free specifically on short LLM fine-tuning found.
- No numbers found for WSD vs cosine at ~400 steps.

## Q3. modded-nanoGPT speedrun tricks: which transfer to a pretrained GPT-2 checkpoint?

### Takeaway
Most speedrun gains (45 min -> 0.665 min, ~68x) come from architecture changes (RoPE, QK-norm, ReLU^2, untied head, U-net/value-embedding skips, logit softcap, sliding-window attention, FP8, multi-token prediction) that change the function the pretrained weights compute, so they do NOT apply to a GPT-2 checkpoint without retraining. Transferable pieces are optimizer/schedule/data tricks: trapezoidal LR with cooldown, (cautious) weight decay, batch-size schedule, aligning batches to EoS/document boundaries, and fused/compiled kernels.

### Cited Findings
- Record history: llm.c baseline 45 min (May 2024) -> 2.992 min (Jan 2025, record 20) -> 2.128 min (Dec 2025, record 50) -> 0.665 min (Aug 2026, record 92), to reach the GPT-2-124M val-loss target (3.28) on FineWeb. — [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt)
- Tricks listed: rotary embeddings, QK-Norm, ReLU^2, zero-init projections, untied embedding/lm_head ("at 2/3 of training"), Muon/NorMuon, cautious weight decay with schedule, FlashAttention-3 sliding window, window-size warmup with YaRN, long-short attention, FP8 head/MLP/QKV, asymmetric rescale and softcap logits, embedding-to-every-block skips and U-net skips, value embeddings, batch starts aligned with EoS, max-document-length scheduling, batch-size schedule, multi-token prediction, hashed bigram/trigram embedding tables, custom Triton kernels, WSD/trapezoidal LR. — [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt)

### Inferences
- Classification for a pretrained HF GPT-2 checkpoint:
  - Requires architecture change / incompatible with checkpoint: RoPE (GPT-2 uses learned absolute `wpe`), QK-norm, ReLU^2 (GPT-2 uses GELU), U-net / value-embedding skips, multi-token prediction heads, hash n-gram embeddings, sliding-window/long-short attention (changes outputs), zero-init projections (destroys weights).
  - Possibly addable with care: logit soft-capping (tanh(x/c)*c with large c ~ 30 is near-identity on GPT-2 logits -- only a regularizer, not a speed win; inference); untying `lm_head` from `wte` (initialize as copy -- legal for full FT, doubles 38M embedding params; no evidence of benefit for 400 steps).
  - Directly transferable: trapezoidal/WSD schedule, EoS-aligned packing (WikiText-2 blocks), cautious masking, torch.compile/fused AdamW (`torch.optim.AdamW(..., fused=True)`), Muon on hidden matrices (but see Q1 mismatch).
- T4-specific: FP8 and FlashAttention-3 need Hopper; not applicable. SDPA (memory-efficient kernel) works on T4 for fp16.

### Gaps
- No source quantifies how much each speedrun trick contributes on its own in a fine-tuning context.

## Q4. LR sensitivity of LoRA vs rsLoRA vs full FT; is a tiny LR sweep or muP-style transfer worth it?

### Takeaway
Yes -- the LR sweep is likely the highest-value change. Optimal LoRA LR is ~10x the full-FT LR (~15x for runs under ~100 steps); with properly tuned LR, nine LoRA variants land within 1-2% of vanilla LoRA, so tuning LR matters more than LoRA flavor. The current 5e-5 full FT / 3e-4 LoRA (6x ratio) are plausibly both below optimum for a 400-step run. LoRA+ (higher LR on B) is a cheap knob with up to ~2x reported speedup.

### Cited Findings
- "LoRA Without Regret" (Thinking Machines, 2025): optimal LR for full FT is ~10x lower than for high-rank LoRA, consistent across SL and RL; ~15x for very short runs (<~100 steps). With the standard alpha/r parametrization, learning curves are identical at the start regardless of rank, so optimal LR is ~rank-independent. LoRA is less tolerant of large batch sizes than full FT, not fixed by more rank. Apply LoRA to all layers, especially MLPs: attention-only r=256 underperforms MLP-only r=128. — [Thinking Machines blog](https://thinkingmachines.ai/blog/lora/)
- "Learning Rate Matters: Vanilla LoRA May Suffice" (ICML 2026): re-tuned 9 LoRA variants + vanilla over LR, batch size, rank, duration; different variants prefer distinct LR ranges; once LR is tuned all are within 1-2%; differing optimal LRs explained by the largest Hessian eigenvalue. — [arXiv 2602.04998](https://arxiv.org/abs/2602.04998v1)
- LoRA+ (Hayou, Ghosh, Yu; ICML 2024): set LR of B larger than A by a fixed ratio; 1-2% better and up to ~2x faster fine-tuning at the same cost; code github nikhil-ghosh-berkeley/loraplus (also available in PEFT as `create_loraplus_optimizer`, ratio commonly 16 -- from memory, verify). — [arXiv 2402.12354](https://arxiv.org/html/2402.12354v2); [PMLR](https://proceedings.mlr.press/v235/hayou24a.html)

### Inferences
- rsLoRA uses alpha/sqrt(r) instead of alpha/r, so effective update scale grows with sqrt(r): at fixed LR, r=256 rsLoRA has 16x larger effective scale than r=1 vs standard LoRA's constant. Therefore rsLoRA r64/r256 runs need their own LR (likely lower), while standard LoRA's optimum is ~rank-independent per Thinking Machines. Comparing LoRA vs rsLoRA at one shared LR (3e-4) confounds rank with LR.
- Recommended tiny sweep (each run ~400 steps, can cut to 150-200 steps for the sweep since LR rankings at short horizon usually hold -- but note Wen et al. warn rankings can change across decay): full FT {5e-5, 1e-4, 2e-4, 4e-4}; LoRA r8 {3e-4, 1e-3, 2e-3, 4e-3}. A 3-4 point log grid with factor-2 spacing is enough.
- muP-style transfer is unnecessary here: only one model width; Thinking Machines' 10x rule plus rank-invariance of alpha/r is the practical "transfer" rule (full-FT optimum x10 -> LoRA LR).
- Make sure LoRA targets include MLP (`c_fc`, `c_proj` in GPT-2) not just `c_attn`.

### Gaps
- No source specific to GPT-2 124M on WikiText-2 LR optima.

## Q5. Model merging / averaging (EMA, LAWA) as a free eval-time improvement

### Takeaway
Weight averaging is cheap and well supported for high-LR or noisy runs (LAWA beat EMA and SWA in GPT-2-scale pretraining), but its benefit is mostly *before* the LR has decayed; with a cosine-to-zero or WSD cooldown, the final checkpoint already captures most of the averaging gain. Schedule-Free does averaging internally.

### Cited Findings
- LAWA (Sanyal et al., NeurIPS 2023 WANT workshop): FIFO queue of k latest, widely spaced checkpoints, averaged; outperforms conventional training and EMA/SWA baselines, gains largest with high LR and large spacing; tested on nanoGPT-2 125M-770M (OpenWebText) and Pythia 1B-12B. — [arXiv 2306.03241](https://arxiv.org/pdf/2306.03241)
- Schedule-Free's eval-mode parameters are an online average (hence the required `optimizer.eval()` swap). — [schedule_free README](https://github.com/facebookresearch/schedule_free)
- Hägele et al. motivate that cooldown with a constant LR plays a similar role to averaging, enabling reusable runs. — [arXiv 2405.18392](https://arxiv.org/pdf/2405.18392)

### Inferences
- For a demo: keep an EMA of trainable params (decay ~0.99 for 400 steps; `torch.optim.swa_utils.AveragedModel` with `get_ema_multi_avg_fn(0.99)`) and evaluate both. Most informative pairing: constant LR (no decay) + EMA/LAWA vs cosine -- averaging should recover most of the decay benefit. With cosine already, expect small or zero gain.
- For LoRA, EMA only over adapter params (tiny memory). Also possible: averaging LoRA adapters across seeds/ranks is not clean (different subspaces) -- avoid.

### Gaps
- No 2024-2026 source quantifying EMA/LAWA gains on short fine-tuning runs found.

## Q6. Single most impressive cheap optimization for a 3-minute T4 run, and overall ranked list

### Takeaway
The best 3-minute demo is an LR sweep (or the "LoRA LR = 10x full-FT LR" rule) plus a WSD/(1-sqrt) cooldown schedule -- reliable, zero-code, and backed by fair-comparison papers; the flashiest alternative is Schedule-Free AdamW (one import, no schedule). Muon is the trendiest name but is expected to underperform on full FT of Adam-pretrained GPT-2.

### Cited Findings
- Tuning a single hyperparameter of a standard recipe can yield ~2x speedup, more than any optimizer swap at scale (1.1-1.4x). — [Wen et al. 2025](https://arxiv.org/html/2509.02046)
- LoRA optimum LR ~10x full-FT LR (15x for very short runs); tuned LR erases most differences between LoRA variants. — [Thinking Machines](https://thinkingmachines.ai/blog/lora/); [arXiv 2602.04998](https://arxiv.org/abs/2602.04998v1)
- Muon on Adam-pretrained full FT degrades; LoRA+Muon only matches Adam. — [arXiv 2605.10468](https://arxiv.org/abs/2605.10468)

### Inferences
Ranked list (expected value for this exact setup; speedup figures are the sources' numbers, mostly from-scratch pretraining, not measured on this task):

| Rank | Change | Reported gain (source setting) | Code / package | Applies to |
|---|---|---|---|---|
| 1 | Tiny log-grid LR sweep (factor 2, 3-4 pts); LoRA LR ~10-15x full-FT LR | up to ~2x from tuning one HP (pretraining); LoRA variants within 1-2% once tuned | loop over `lr` | Both (separately for rsLoRA) |
| 2 | WSD / trapezoidal schedule, (1-sqrt) cooldown over last 20-30% | matches/beats cosine; 1-sqrt > linear | `LambdaLR` with piecewise lambda | Both |
| 3 | LoRA+ (lr_B = ratio x lr_A) | up to ~2x faster, +1-2% | `loraplus` repo / PEFT `create_loraplus_optimizer` | LoRA only |
| 4 | LoRA on all linear layers incl. MLP | attn-only r256 < MLP-only r128 | PEFT `target_modules=["c_attn","c_proj","c_fc"]` | LoRA |
| 5 | Schedule-Free AdamW (LR 1-10x, warmup) | AlgoPerf self-tuning winner | `pip install schedulefree`; `AdamWScheduleFree` + `.train()/.eval()` | Both |
| 6 | Cautious mask (C-AdamW) | up to 1.47x (authors); ~1.1x (independent) | one line, `kyleliang919/C-Optim` | Both |
| 7 | EMA / LAWA of weights at eval | beats EMA/SWA at high LR (LAWA) | `torch.optim.swa_utils.AveragedModel` | Both |
| 8 | Muon on hidden 2D weights | 1.35x from scratch; ~2x at scale (Moonshot); degrades Adam-pretrained full FT; parity under LoRA | `torch.optim.Muon` (bf16 NS -- T4 issue) or Keller's NS in fp32 | LoRA (memory only); full FT risky |
| 9 | SOAP / Kron / Shampoo | ~1.3x at 130M from scratch | external repos | Not worth it here |
| 10 | Lion, Adam-mini, Sophia | ~0.97-1.0x in fair benchmark | -- | Skip |

- Suggested 3-minute demo: two runs of LoRA r8 at 400 steps -- baseline (3e-4, cosine) vs (LR swept to best of ~4 values, WSD 1-sqrt cooldown, all-linear targets, LoRA+). Plot val loss vs step and report "steps to reach baseline's final PPL". Alternatively Schedule-Free vs cosine at same budget.
- Practical T4 notes: fp16 autocast + GradScaler; prefer `fused=True` AdamW; avoid any bf16-only code paths (including torch.optim.Muon).

### Gaps
- No source directly measures any of these on GPT-2 124M / WikiText-2 continued pretraining; all speedup numbers must be re-measured on the experiment.
- LoRA+ default ratio (16) and PEFT helper name stated from memory, not verified in this session.
