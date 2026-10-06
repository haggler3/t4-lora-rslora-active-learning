# PEFT / LoRA-family variants for continued pre-training of GPT-2 small on a T4 (peft 0.21)

Context: GPT-2 small (124M) CPT on WikiText-2, 512-tok blocks, bs 8, 400 steps, cosine LR, AdamW, fp16 autocast + GradScaler, T4 (no bf16). Current LoRA/rsLoRA r=8/64/256, alpha=8, LR 3e-4, targets c_attn/c_proj/c_fc. Zero-shot PPL 35.84, full FT 24.00, LoRA r8 25.20.

Note on GPT-2 layer type: GPT-2 in transformers uses `transformers.pytorch_utils.Conv1D` (weight stored as [in, out]); peft wraps it as a LoRA Linear with `fan_in_fan_out=True`. Whether each init handles this transpose matters (see Q1/Q3). I could not confirm whether transformers 5.x still uses Conv1D for GPT-2 (gap).

## Q1. Which LoRA-family improvements exist, what gain is reported, and what is the peft 0.21 API? (ranked for this experiment)

### Takeaway
The cheapest, highest-expected-value changes for a 3-min GPT-2 CPT run are: (1) fix the scaling/LR (rsLoRA at high rank, LR sweep, alpha choice), (2) LoRA+ (one optimizer swap), (3) PiSSA / LoRA-GA initialisations (faster early convergence, seconds of setup), then (4) DoRA (small quality gain at low rank, ~1.5-2x step cost). VeRA / LoRA-FA / AdaLoRA cut parameters or reallocate rank but are not expected to beat LoRA in CPT where capacity is the bottleneck.

### Cited Findings

**Peft 0.21 API surface (current docs; docs show v0.21.0 links)**
- All init options go through `LoraConfig(init_lora_weights=...)`: `True` (Kaiming A, zero B; default), `"gaussian"`, `"pissa"` (full SVD, "takes several minutes" on large models), `"pissa_niter_[n]"` (fast randomized SVD, "only seconds", 16 recommended for 7B), `"olora"` (QR, mutates base weights), `"loftq"`, `"eva"`, `"corda"`, `"mica"`, `"orthogonal"` (A and B orthogonal, base untouched, r must be even, linear layers only), `False` (debug only) — [PEFT LoRA guide](https://huggingface.co/docs/peft/developer_guides/lora)
- rsLoRA: `LoraConfig(use_rslora=True)` scales by `lora_alpha/sqrt(r)` instead of `lora_alpha/r` — [PEFT LoRA guide](https://huggingface.co/docs/peft/developer_guides/lora)
- DoRA: `LoraConfig(use_dora=True)`; "Introduces bigger overhead than pure LoRA" — [PEFT LoRA guide](https://huggingface.co/docs/peft/developer_guides/lora)
- LoRA+: `from peft.optimizers import create_loraplus_optimizer; optimizer = create_loraplus_optimizer(model=model, optimizer_cls=..., lr=5e-5, loraplus_lr_ratio=16)`; docs claim "up to 2x" speed and "1-2%" performance — [PEFT LoRA guide](https://huggingface.co/docs/peft/developer_guides/lora)
- LoRA-FA: `from peft.optimizers import create_lorafa_optimizer; create_lorafa_optimizer(model=model, r=128, lora_alpha=32, lr=7e-5)`; freezes A, trains B, memory "not sensitive to rank" — [PEFT LoRA guide](https://huggingface.co/docs/peft/developer_guides/lora)
- LoRA-GA: `from peft.tuners.lora import preprocess_loraga; preprocess_loraga(model, lora_config, train_step)` where `train_step` runs N forward/backward passes on a gradient dataloader; modifies base weights; "Requires full-precision" weights, no quantized models; direction strategies `"ArBr"`, `"A2rBr"`, `"ArB2r"` (default), `"random"`; scaling `"stable"` (default), `"weight_svd"`, `"gd_scale"`, `"unit"` — [PEFT LoRA guide](https://huggingface.co/docs/peft/developer_guides/lora)
- EVA: `LoraConfig(init_lora_weights="eva", eva_config=EvaConfig(rho=2.0))`, `get_peft_model(model, config, low_cpu_mem_usage=True)`, then `initialize_lora_eva_weights(peft_model, dataloader)`; data-driven SVD of input activations, adaptively redistributes rank (max r*rho) — [PEFT LoRA guide](https://huggingface.co/docs/peft/developer_guides/lora)
- CorDA: `CordaConfig(corda_method="kpm")` (knowledge-preserving) or `"ipm"`; `preprocess_corda(model, lora_config, run_model=run_model)` before `get_peft_model` — [PEFT LoRA guide](https://huggingface.co/docs/peft/developer_guides/lora)
- MiCA (new in 0.20): init B from the *smallest* singular vectors, A=0, only A trainable, ~half LoRA's params — [PEFT LoRA guide](https://huggingface.co/docs/peft/developer_guides/lora); [PEFT releases](https://github.com/huggingface/peft/releases)
- Intruder-dimension reduction (0.19, anti-forgetting post-hoc): `from peft.tuners.lora.intruders import reduce_intruder_dimension; reduce_intruder_dimension(peft_model, mitigation_lambda=0.75)` — [PEFT LoRA guide](https://huggingface.co/docs/peft/developer_guides/lora)
- Release timeline: 0.19.0 added LoRA-GA init, intruder-dimension reduction, GraLoRA and others; 0.20.0 added MiCA, HiRA, VeLoRA, MonteCLoRA, Uni-LoRA, etc.; 0.21.0 added a Riemannian-preconditioned LoRA optimizer, KaSA, Super-Tuning, ShadowPEFT; 0.21.2 fixed encoder-decoder compatibility with transformers >= 5.18 — [PEFT GitHub releases](https://github.com/huggingface/peft/releases) (dates shown in my fetched summary were inconsistent/garbled, so I do not report them)

**Reported gains (original papers)**
- LoRA+: setting different LRs for A and B gives "1-2%" improvements and "up to ~2X" speedup at the same compute — [LoRA+ arXiv 2402.12354](https://arxiv.org/abs/2402.12354). (Abstract does not state the ratio; peft's example uses `loraplus_lr_ratio=16`.)
- PiSSA: Mistral-7B GSM8K 72.86% vs LoRA 67.7% (+5.16 pts); QPiSSA on LLaMA-3-70B 86.05% vs QLoRA 81.73%; faster convergence; init "in only a few seconds" with fast SVD — [PiSSA arXiv 2404.02948](https://arxiv.org/abs/2404.02948)
- LoRA-GA: "2-4 times convergence speed improvement" vs vanilla LoRA; +5.69% avg on GLUE (T5-Base); Llama-2-7B +0.34 MT-bench, +11.52% GSM8K, +5.05% HumanEval — [LoRA-GA arXiv 2407.05000](https://arxiv.org/abs/2407.05000)
- DoRA: decomposes W into magnitude and direction, LoRA on direction; improves "learning capacity and training stability" with no inference overhead after merge; evaluated on commonsense reasoning (LLaMA), visual instruction tuning (LLaVA), VL-BART — [DoRA arXiv 2402.09353](https://arxiv.org/abs/2402.09353). Peft: helps "especially at low ranks" — [PEFT DoRA reference](https://huggingface.co/docs/peft/en/package_reference/lora_variant_dora)
- rsLoRA: standard alpha/r scaling "results in slowed learning and stunted performance" at higher ranks; alpha/sqrt(r) lets higher rank buy better performance at no inference cost — [rsLoRA arXiv 2312.03732](https://arxiv.org/abs/2312.03732)
- "LoRA Without Regret" (Thinking Machines, 2025): optimal LoRA LR is ~10x FullFT LR, rising to ~15x for short (~100-step) runs; with 1/r scaling (alpha=32) the optimal LR is roughly rank-independent for short runs; attention-only LoRA underperforms MLP-only even at matched params (attn-only r=256 < MLP-only r=128) — [Thinking Machines blog](https://thinkingmachines.ai/blog/lora/)

### Inferences
- Ranked list for this experiment (expected value / effort):
  1. **Scaling + LR hygiene (free).** With alpha=8 fixed, standard LoRA at r=256 has scale 8/256 = 0.03, so the r=256 run is effectively LR-starved; rsLoRA r=256 has scale 0.5. A small LR sweep (e.g. 3e-4, 1e-3, 2e-3 for LoRA vs the full-FT LR) is likely worth more than any variant, given the "~10-15x FullFT LR for short runs" finding. Also consider alpha=16-32 (Biderman et al. recommend alpha=2r).
  2. **LoRA+ (one line, zero runtime cost).** `opt = create_loraplus_optimizer(model=model, optimizer_cls=torch.optim.AdamW, lr=3e-4, loraplus_lr_ratio=16, weight_decay=0.0)`; works with the existing GradScaler and cosine scheduler. Try ratios 4 and 16.
  3. **PiSSA (`init_lora_weights="pissa_niter_4"`)** — seconds of SVD on 768x2304 matrices; handles GPT-2 Conv1D via fan_in_fan_out (see Q3). Expect lower loss in the first ~100 steps.
  4. **LoRA-GA** — needs a short gradient-estimation pass (a few batches) before training; the GPT-2 base weights are fp32 so the "full precision" requirement is met.
  5. **DoRA** (`use_dora=True`, set `lora_dropout=0`) — expect a small gain at r=8, ~1.5-2x step time.
  6. **EVA / CorDA-KPM** — data-driven; CorDA-KPM is the one explicitly aimed at preserving prior knowledge (relevant if forgetting is measured).
  7. **VeRA, LoRA-FA, MiCA, AdaLoRA** — fewer trainable params or rank re-allocation; low priority for CPT, where LoRA already lacks capacity (Q2).

### Gaps
- I did not fetch VeRA, AdaLoRA, OLoRA or EVA paper numbers, so there are no reported gains for them here.
- The LoRA+ paper's recommended ratio was not in the abstract; 16 comes from peft's doc example. From memory, the paper suggests about 2^4, but I did not verify this.
- I found no published results for any of these variants on GPT-2 / WikiText-2.

## Q2. Which matter specifically for continued pre-training (vs instruction tuning)? (Biderman et al., ReLoRA, GaLore, etc.)

### Takeaway
For CPT, the main limit is LoRA's low rank, not its initialisation. Biderman et al. find that LoRA, even at r=256, does not close the gap to full FT in CPT, while it does in instruction tuning. Methods that grow the effective rank (ReLoRA merge-and-restart; GaLore/APOLLO full-parameter low-rank-gradient optimizers) or that make high rank train well (rsLoRA plus a higher LR, all modules) target that limit. Init tricks such as PiSSA and LoRA-GA mainly speed up early convergence.

### Cited Findings
- Biderman et al. (TMLR 2024 / ICLR 2025 poster): Llama-2-7B CPT on StarCoder-Python (up to 20B tok) and OpenWebMath (14.7B tok), LoRA r=16/64/256 on all modules, alpha=2r. LoRA "substantially underperforms" full FT in CPT and the gap widens with more data. Code: best LoRA (r=256) HumanEval 0.224 vs full FT 0.263. Math: GSM8K 0.203 vs 0.293 — [arXiv 2405.09673](https://arxiv.org/html/2405.09673v2); [ICLR 2025 poster](https://iclr.cc/virtual/2025/poster/31465)
- Same paper: the best LoRA at 20B tokens roughly matches full FT at 4B tokens (code CPT). Full FT perturbations have rank 10-100x higher than typical LoRA configs. LoRA forgets less than weight decay or dropout regularisation — [arXiv 2405.09673](https://arxiv.org/html/2405.09673v2)
- Their recommendations: use LoRA for IFT rather than CPT; target "All" modules at r=256 if memory allows; alpha=2r; sweep LRs. LoRA "is more sensitive to learning rates", needing roughly 10x higher LR — [arXiv 2405.09673](https://arxiv.org/html/2405.09673v2). (My fetch summarised the LRs as 1e-5 for both methods in code CPT and 1e-5 (FT) vs 4e-5 (LoRA) in math CPT. This conflicts with the paper's own "order of magnitude" statement, so check against the paper's appendix tables before quoting it.)
- Thinking Machines: LoRA degrades when the "dataset exceeds LoRA capacity", especially in pretraining-like large-data settings. There is a persistent LoRA vs FullFT gap at large batch sizes that does not depend on rank — [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/)
- ReLoRA: low-rank updates with periodic merge-and-restart, jagged LR schedule, partial optimizer reset, and a required full-rank warm start. 1.3B: PPL 17.27 (250M trainable) vs 16.83 full-rank. Saves up to 5.5 GB RAM/GPU, 9-40% faster — [ReLoRA arXiv 2307.05695](https://arxiv.org/html/2307.05695v4)
- GaLore: full-parameter learning by projecting gradients to low rank. 8-bit GaLore pretraining LLaMA-7B on C4 matches full-rank with <10% of optimizer-state memory — [GaLore repo](https://github.com/jiaweizzhao/GaLore); [arXiv 2403.03507](https://arxiv.org/html/2403.03507v2)
- Transformers integration: `optim="galore_adamw" | "galore_adafactor" | "galore_adamw_8bit"` (+ `_layerwise`), `optim_target_modules=[r".*.attn.*", r".*.mlp.*"]`, `optim_args="rank=64, update_proj_gap=100, scale=0.10"`. Layerwise mode is single-GPU only, and gradient clipping may not work. APOLLO (`optim="apollo_adamw"`, `pip install apollo-torch`) uses random projections instead of SVD; APOLLO-Mini is rank-1 — [Transformers optimizers docs](https://huggingface.co/docs/transformers/main/en/optimizers)
- peft 0.19 added `reduce_intruder_dimension(peft_model, mitigation_lambda=0.75)` to reduce forgetting after LoRA training — [PEFT LoRA guide](https://huggingface.co/docs/peft/developer_guides/lora)

### Inferences
- The observed result (full FT 24.00 vs LoRA r8 25.20) fits the CPT capacity story. The most informative 3-min experiments are those that close this 1.2-PPL gap: rsLoRA r=256 at a higher LR, ReLoRA-style restarts, GaLore/APOLLO.
- ReLoRA is not in peft, but takes about 10 lines with peft. Every K steps (e.g. K=100, giving 4 cycles in 400 steps): `model.merge_adapter()`, or merge then re-init via `merge_and_unload()` + `get_peft_model()`; re-initialise A (Kaiming) and B=0; reset (or prune ~99% of) the AdamW state for the LoRA params; apply a short LR re-warmup (jagged cosine). The paper's warm start (full-rank steps first) can be skipped here because GPT-2 is already pretrained.
- GaLore on a 124M model brings no memory benefit (full FT already fits in 16 GB). Its value here is showing full-parameter learning at low optimizer memory. It needs targeting of GPT-2 module names (`.*attn.*`, `.*mlp.*`), and the SVD every `update_proj_gap` steps is cheap at 768-3072 dims. If the training loop is custom rather than Trainer: `from galore_torch import GaLoreAdamW; param_groups=[{'params': non_galore}, {'params': galore_params, 'rank': 64, 'update_proj_gap': 100, 'scale': 0.25, 'proj_type': 'std'}]`. This form follows the galore-torch README pattern; I did not verify it line-by-line.
- If forgetting is measured (e.g. PPL on a held-out non-WikiText set), LoRA's "forgets less" property makes a good second axis: report target PPL vs source PPL as a Pareto plot across full FT / LoRA / rsLoRA / PiSSA / DoRA. Inference (unverified): PiSSA and OLoRA move the principal components of W into trainable adapters and may forget more than default-init LoRA. CorDA-KPM is designed for the opposite.

### Gaps
- I found no study of PiSSA, LoRA-GA or DoRA on continued pre-training specifically. Their papers evaluate IFT/GLUE/math SFT.
- "LoRA-the-Explorer" (LTE, parallel low-rank adapters for pretraining) was not researched in this pass.
- The exact CPT learning rates in Biderman et al. are unverified (conflict noted above).

## Q3. Which are known to fail or be slow on Turing / fp16 (and on GPT-2's Conv1D)?

### Takeaway
None of the peft methods need bf16 or Ampere, and the T4 can run them all. The real hazards are: DoRA's per-step overhead (1.5-2x+); SVD/QR inits needing fp32 (fine, since GPT-2 base weights load fp32 and autocast only affects compute); and GPT-2's transposed Conv1D weights, which PiSSA, LoRA-GA and CorDA handle explicitly while OLoRA and orthogonal init appear not to.

### Cited Findings
- DoRA in peft: "only supports embedding, linear, and Conv2d layers"; bigger overhead than LoRA; faster when `lora_dropout=0` or in eval mode because the base result is reused (a 4090 example ran 359.7 s -> 279.3 s with this optimisation). For Llama-3.1-8B inference, DoRA without caching needs +139% time and +4% memory vs LoRA; with `DoraCaching` +17% time and +41% memory — [PEFT DoRA reference](https://huggingface.co/docs/peft/en/package_reference/lora_variant_dora)
- Secondary report: a DoRA training step is ~1.5x-1.8x slower than LoRA, and `get_peft_model` with DoRA is noticeably slower — [search snippet, zeroshot.it.com aggregator](https://zeroshot.it.com/?p=6196) (aggregator, lower confidence)
- The 2026 paper "Scaling DoRA: High-Rank Adaptation via Factored Norms and Fused Kernels" says peft's DoRA materialises a dense [d_out, d_in] product per module. Its factored norms plus Triton kernels are 1.5-2.0x faster at inference, 1.5-1.9x faster for gradients, and use up to 7 GB less VRAM. This is not part of peft — [arXiv 2603.22276](https://arxiv.org/abs/2603.22276)
- PiSSA init in peft transposes via `fan_in_fan_out` (`weight = transpose(weight.to(torch.float32), self.fan_in_fan_out)`) and requires the weight in float32/float16/bfloat16. LoRA-GA also transposes (`grad = transpose(grad, self.fan_in_fan_out)`). CorDA has explicit `Conv1D` handling. My source read found no `fan_in_fan_out` handling in OLoRA or orthogonal init — [peft lora/layer.py (main)](https://raw.githubusercontent.com/huggingface/peft/main/src/peft/tuners/lora/layer.py) (read via an LLM summariser, so verify)
- LoRA-GA "requires full-precision" weights — [PEFT LoRA guide](https://huggingface.co/docs/peft/developer_guides/lora)
- SVD-based adapter merging (`combination_type="svd"`) is not supported in fp16/bf16 — [PEFT LoRA guide](https://huggingface.co/docs/peft/developer_guides/lora)

### Inferences
- **OLoRA / orthogonal on GPT-2: test before trusting.** Without the Conv1D transpose, the non-square `c_attn` (768->2304) and `c_fc` (768->3072) layers will likely fail with a shape error. Square `c_proj` in attention may run silently with mis-oriented adapters. Check `peft_model(x)` against the base model output at step 0 (it should match exactly for OLoRA/PiSSA after the residual subtraction).
- **DoRA on GPT-2 Conv1D:** the docs list only Linear/Conv2d/Embedding. GPT-2's Conv1D is wrapped by peft's LoRA `Linear` (fan_in_fan_out=True), so it should work, but this is unverified. A one-step smoke test is needed. At 124M params the absolute overhead is small, so the run should take ~4-6 min instead of ~3 (estimate).
- fp16 + GradScaler: peft keeps adapter weights in fp32 by default, which is compatible with GradScaler. Keep the base model in fp32 and rely on autocast. Do not `.half()` the model, because the PiSSA/OLoRA/LoRA-GA residual subtraction in fp16 would add error.
- PiSSA/OLoRA/LoRA-GA mutate base weights. For a fair "zero-shot" comparison, compute zero-shot PPL before `get_peft_model`. To save, use peft's conversion path (`path_initial_model_for_weight_conversion` in `save_pretrained`, from memory, unverified) or just save the merged model.
- GaLore's SVD is in fp32 inside the optimizer. `galore_adamw_8bit` needs bitsandbytes, which supports T4/Turing. I did not verify GaLore with GradScaler (gap).

### Gaps
- I found no reports of DoRA or PiSSA failures specific to T4/sm_75. The `Scaling DoRA` Triton kernels' Turing support was not checked.
- I could not verify that GPT-2 still uses Conv1D in transformers 5.18, or whether OLoRA actually errors on it.

## Q4. Cheapest compelling demo in ~3-minute runs

### Takeaway
Plot validation loss (or PPL every 25-50 steps) against step for each variant at a fixed LR. Faster-converging inits show up as earlier crossings in a single 400-step run, and the "steps to reach PPL X" metric makes the convergence-speed claims (PiSSA, LoRA-GA "2-4x", LoRA+ "~2x") directly testable.

### Cited Findings
- Claims available to test: LoRA+ "up to ~2X" speedup — [arXiv 2402.12354](https://arxiv.org/abs/2402.12354); LoRA-GA "2-4 times convergence speed" — [arXiv 2407.05000](https://arxiv.org/abs/2407.05000); PiSSA "faster convergence" — [arXiv 2404.02948](https://arxiv.org/abs/2404.02948); rsLoRA higher-rank-helps — [arXiv 2312.03732](https://arxiv.org/abs/2312.03732); the optimal LoRA LR for short runs is ~15x FullFT — [Thinking Machines](https://thinkingmachines.ai/blog/lora/)

### Inferences
- Suggested demo set (each ~3 min, r=8 unless noted, same seed; ~8-10 runs, about 30 min):
  1. **Rank-scaling demo:** LoRA vs rsLoRA at r=8/64/256 with alpha=8 (already running). Expected result: LoRA PPL gets *worse or flat* as r rises (scale 0.03 at r=256), while rsLoRA improves with r. The plot needs no new code.
  2. **LoRA+:** swap the optimizer with `create_loraplus_optimizer(..., loraplus_lr_ratio=16)`. Metric: steps to reach LoRA-r8's final PPL (25.20). A claim like "LoRA+ reaches 25.2 PPL in ~N<400 steps" is the compelling form.
  3. **PiSSA:** `init_lora_weights="pissa_niter_4"`. Check that step-0 PPL equals zero-shot (35.84), then compare the loss curve over the first 100 steps. The candidate headline is "PiSSA converges in half the steps"; whether it holds for CPT is untested.
  4. **LoRA-GA:** `preprocess_loraga` with ~8-16 batches of grad estimation (seconds at 124M). Same metric.
  5. **DoRA r=8** (`lora_dropout=0`): report both PPL and wall-clock/step, since DoRA trades time for quality.
  6. **Gap-closer:** rsLoRA r=256 + LR 1e-3 (and ReLoRA K=100) vs full FT 24.00. This tests whether low-rank methods can close the CPT gap Biderman et al. reported.
- Report a table: final PPL, Δ vs LoRA-r8, trainable params, s/step, peak VRAM, steps-to-25.2.
- Use the same LR for all variants in the main table, and add one LR-swept row per variant if time allows. Many "gains" in LoRA-variant papers shrink after per-method LR tuning (inference, not sourced here).

### Gaps
- I found no published 400-step / GPT-2-scale ablations confirming any of these gains. The demos test paper claims rather than reproduce known results.
