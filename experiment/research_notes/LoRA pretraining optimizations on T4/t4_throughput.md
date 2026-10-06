# Throughput / memory optimizations for GPT-2 small (124M) training on a single NVIDIA T4 (sm_75), torch 2.9.1 + transformers 5.x + peft

Baseline being optimized: fp16 autocast + GradScaler, AdamW, batch 8 x 512, 400 steps, ~0.42 s/step (full FT ~180 s, LoRA r8 ~167 s), ~9.6 GB, GPU util 100%, loss via `model(input_ids, labels=...)` with full 50257-vocab logits.

## Attention: which SDPA kernel runs on sm_75, xformers / FlashAttention on Turing, expected gain at seq 512

### Takeaway
On a T4 in torch 2.9, PyTorch's built-in Flash backend is unavailable (it needs sm80 to sm121); SDPA falls back to the **memory-efficient (xformers-derived) kernel**, which supports sm50 to sm121 in fp16 and fp32. So `attn_implementation="sdpa"` with fp16 autocast gets mem-efficient attention. Expect a modest speedup at T=512 and a real memory saving compared with eager. FlashAttention-2 is Ampere+ only.

### Cited Findings
- PyTorch 2.9.1 `sdp_utils.cpp`: flash is gated by `check_sm_version<sm80, sm121>`, with the message "Flash attention only supports gpu architectures in the range [sm80, sm121]". So **no PyTorch flash kernel on T4**. — [pytorch v2.9.1 sdp_utils.cpp](https://raw.githubusercontent.com/pytorch/pytorch/v2.9.1/aten/src/ATen/native/transformers/cuda/sdp_utils.cpp)
- Same file: mem-efficient attention is gated to `[sm50, sm121]`. On SM<8.0 its allowed dtypes are `kHalf` and `kFloat` (no bf16). — [pytorch v2.9.1 sdp_utils.cpp](https://raw.githubusercontent.com/pytorch/pytorch/v2.9.1/aten/src/ATen/native/transformers/cuda/sdp_utils.cpp)
- The standalone `flash-attn` package pages (v1.x / 0.2.x line) list Turing (including T4, sm75) as supported. That is **FlashAttention-1** only. FA-2 needs Ampere+. — [flash-attn 1.0.9 on PyPI](https://pypi.org/project/flash-attn/1.0.9); [flash-attn 0.2](https://pypi.org/project/flash-attn/0.2)
- HF transformers docs present `attn_implementation="flash_attention_2"` and Hub kernels (`kernels-community/flash-attn2`) as optimized backends. These target FA-2, so they do not apply on T4. — [HF perf_infer_gpu_one](https://huggingface.co/docs/transformers/main/en/perf_infer_gpu_one)

### Inferences
- **Code (recommended, near-zero risk):**
  ```python
  model = AutoModelForCausalLM.from_pretrained("gpt2", attn_implementation="sdpa")
  print(model.config._attn_implementation)   # verify "sdpa" (GPT-2 SDPA is likely already the default in transformers 5.x; verify)
  # Optional: force the backend and make it fail loudly if it falls back to math:
  from torch.nn.attention import sdpa_kernel, SDPBackend
  with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
      loss = ...
  ```
- Size of the gain at T=512 (my own arithmetic): attention score FLOPs are 12*L*d*T ≈ 57 MFLOP/token against roughly 6N ≈ 744 MFLOP/token for the weights, so about 7% of FLOPs. The eager path materializes B*H*T*T = 8*12*512*512 ≈ 25M-element score/prob tensors per layer (fp16 about 50 MB each, several kept per layer for backward). Switching eager to mem-efficient SDPA plausibly saves about 1 GB+ of activations and a few percent to about 10% of step time. If the run already uses SDPA (likely, since it is the default), the gain is **zero**.
- Use a pure causal mask with no padding mask (packed, fixed-length sequences) so HF can call SDPA with `is_causal=True`. That avoids building a dense [B,1,T,T] additive mask. Don't pass `attention_mask` when every sequence is full-length packed text.
- xformers is not worth adding: PyTorch's mem-efficient kernel is derived from it and is already used. FA-1 has no maintained HF integration in transformers 5.x, and the expected gain over mem-efficient at T=512 is small.

### Gaps
- No published T4 benchmark found that compares eager vs SDPA (mem-efficient) for GPT-2 at T=512. Measure it with a 20-step A/B.
- I didn't confirm from the transformers 5.18 source whether GPT-2 already defaults to SDPA, or whether it passes `is_causal=True` when there is no attention mask.

## torch.compile and CUDA graphs on T4 (modes, compile time vs a 400-step run)

### Takeaway
torch.compile relies on Triton, and **official Triton dropped Turing (sm75) support from Triton 3.3 onward**. torch 2.9 ships Triton 3.5, so compile on T4 is "works in practice but unsupported" and may hit codegen bugs. Even when it works, the one measured GPT-2-on-T4 report found only about an 8% steady-state gain (against about 90% on A100). Compile time is tens of seconds to minutes, which eats most of that gain in a 180 s run. Low priority. If you try it, use the default mode on the loss function only, and lean on the inductor cache.

### Cited Findings
- Triton README, "Supported Hardware": "NVIDIA GPUs (Compute Capability 8.0+)". — [triton-lang/triton](https://github.com/triton-lang/triton)
- triton-windows README: Turing (sm75) "is officially supported by Triton <= 3.2. Support for Turing has been dropped since Triton 3.3". Also, bf16 and fp8 on Turing are not supported by official Triton. — [woct0rdho/triton-windows](https://github.com/woct0rdho/triton-windows)
- A GPT-2 (124M) reproduction on Colab T4 (B=4, T=1024 was the best fit) found that after the slow first compile, steps were "roughly an 8% drop on T4 vs 90% drop on A100" in time. — [TDS: Line-by-line reproduce GPT-2, hardware optimization](https://towardsdatascience.com/line-by-line-lets-reproduce-gpt-2-section-2-hardware-optimization-86e71c91d9bb/)
- nanoGPT docs claim about a 2x gain from compile (250 ms to 135 ms per iteration), but that is on A100. — [nanoGPT benchmarking (mintlify mirror)](https://www.mintlify.com/karpathy/nanoGPT/advanced/benchmarking)
- HF warns against calling `torch.compile(model)` outside `generate()` to avoid per-step recompiles (inference context). — [HF perf_infer_gpu_one](https://huggingface.co/docs/transformers/main/en/perf_infer_gpu_one)

### Inferences
- **Code if you try it:**
  ```python
  model = torch.compile(model)                 # mode="default"; avoid "max-autotune" (long autotune, CUDA graphs)
  # or compile only the training-step loss fn to keep peft/optimizer outside the graph
  ```
  "reduce-overhead" and "max-autotune" enable CUDA graphs. The GPU is already at 100% util with about 0.42 s steps, so launch overhead isn't the bottleneck and CUDA graphs buy little. They also interact badly with GradScaler's dynamic skip and peft hooks.
- Break-even estimate: an 8% gain on a 180 s run saves about 14 s. A GPT-2 compile of fwd+bwd typically takes more than that cold, so expect a net loss unless the inductor/FX cache is warm across runs (same shapes, same code). For a sweep of many 400-step runs on the same VM, the cache makes compile cheap after the first run, and then it's a small net win.
- Risk: since Triton >= 3.3 no longer officially supports sm75, any Triton-based tool (torch.compile, Liger, Unsloth kernels, CCE) can fail or miscompile on T4 with torch 2.9 / Triton 3.5. Pin and verify the loss curve against eager.

### Gaps
- No T4 + torch 2.9 + Triton 3.5 compile-time or speedup measurement found. No direct confirmation found of whether Triton 3.5 still compiles fp16 `tl.dot` correctly on sm75. Colab T4 users run recent torch, which suggests it works, but I found no source for that.

## Fused AdamW, 8-bit optimizers, Liger Kernel, chunked/fused cross-entropy, Unsloth

### Takeaway
`fused=True` AdamW is a free, safe, small win: CUDA-only, supports fp16/fp32, not the default. 8-bit optimizers save memory, not time, at 124M. The biggest memory lever is the 50257-vocab loss. HF's `labels=` path materializes logits [8,512,50257], about 206M elements (fp16 about 0.4 GB, fp32 upcast about 0.8 GB, plus grad copies). Chunked or fused linear-CE removes most of that. Liger has no GPT-2 model patch, but its standalone `LigerFusedLinearCrossEntropyLoss` can be called by hand. It is Triton-based, so the Turing caveat applies. Apple's CCE says it requires Ampere. Unsloth supports T4 for Llama-family models; GPT-2 support is unverified.

### Cited Findings
- AdamW docs: if neither `foreach` nor `fused` is set, PyTorch defaults to foreach on CUDA. Fused is not yet the default ("sufficient bake-in time"). Fused is CUDA-only and supports float64/32/16/bfloat16. Foreach uses about sizeof(params) extra peak memory. — [torch.optim.AdamW docs](https://docs.pytorch.org/docs/2.3/generated/torch.optim.AdamW.html)
- Liger-Kernel: model patches cover LLaMA, Mistral, Gemma, Qwen, Phi3, DeepSeek and others. **GPT-2 is not listed.** Low-level kernels include LayerNorm, CrossEntropyLoss and Fused Linear CrossEntropyLoss. Requirements are torch >= 2.1.2 and triton >= 2.3.1. Claims are +20% multi-GPU throughput and -60% memory. — [linkedin/Liger-Kernel](https://github.com/linkedin/Liger-Kernel)
- Apple Cut Cross-Entropy (CCE): requires an "Ampere (or newer) GPU". It has a `torch_compile` implementation that "works on CPU and older GPUs". Model patches cover Llama/Phi3/Mistral/Gemma2 only. On Gemma 2 2B it cut loss memory from 24 GB to 1 MB. — [apple/ml-cross-entropy](https://github.com/apple/ml-cross-entropy)
- Unsloth: supports Tesla T4 (float16 recommended). Reported Colab T4 speedups are 1.95x for Llama-2 7B and 3.87x for TinyLlama 1.1B, with VRAM -43% and -74%. For pre-Ampere it is installed with xformers. — [HF blog: Unsloth + TRL](https://huggingface.co/blog/unsloth-trl)

### Inferences
- **Fused AdamW (do it):**
  ```python
  opt = torch.optim.AdamW(params, lr=..., weight_decay=..., fused=True)
  ```
  It works with GradScaler: the fused path handles unscale and inf-check internally. Expected gain about 1 to 3% of step time for full FT (124M params). For LoRA (about 0.3M params) it is negligible.
- **Chunked loss without extra deps (recommended; pure PyTorch, Turing-safe):**
  ```python
  hidden = model.transformer(input_ids).last_hidden_state          # [B,T,768]  (for peft: model.base_model.model.transformer)
  h = hidden[:, :-1].reshape(-1, 768); y = input_ids[:, 1:].reshape(-1)
  W = model.lm_head.weight
  def ce_chunk(h_c, y_c):
      return torch.nn.functional.cross_entropy((h_c @ W.t()).float(), y_c, reduction="sum")
  loss = sum(torch.utils.checkpoint.checkpoint(ce_chunk, h[i:i+1024], y[i:i+1024], use_reentrant=False)
             for i in range(0, h.size(0), 1024)) / y.numel()
  ```
  This avoids holding full [4096,50257] fp32 logits, cutting roughly 1.5 to 2.5 GB of peak memory. It costs one extra lm_head matmul (about 2*768*50257 FLOP/token, about +10% FLOPs) because of recompute. Use the freed memory for batch 16 rather than expecting a speedup. **Liger alternative:** `from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss; loss = LigerFusedLinearCrossEntropyLoss()(W, h, y)`. It needs a Triton smoke test on sm75.
- Swapping in Liger LayerNorm for GPT-2's LayerNorm is possible by hand, but the gain at d=768 is small. Not worth the Triton risk.
- **8-bit AdamW** (`bnb.optim.AdamW8bit`) saves about 0.75 GB of optimizer state at 124M (2 states x 124M x 3 bytes saved). It doesn't speed anything up, it changes the optimizer numerics (a confound for experiments), and it's irrelevant for LoRA.
- **Unsloth for this experiment:** no evidence it accelerates GPT-2 (learned absolute positions, LayerNorm, GELU, Conv1D-style weights). Its fused kernels target Llama-style RoPE/RMSNorm/SwiGLU. Not recommended.

### Gaps
- No source found confirming Liger's FLCE runs correctly on sm75 with Triton 3.5 (Liger says only "inherits Triton's hardware support", and Triton's official floor is now sm80).
- I couldn't fetch the exact HF transformers 5.x `ForCausalLMLoss` code to confirm the logits upcast to fp32 (it is standard in 4.4x+). The memory arithmetic above assumes it.
- Vocab padding to 50304 (nanoGPT trick for GEMM alignment) can't easily be applied to pretrained GPT-2 without `resize_token_embeddings(..., pad_to_multiple_of=64)`, and its T4 speedup is unmeasured. No source fetched.

## Pure fp16 weights vs fp32 master + autocast; TF32; dtype issues on Turing

### Takeaway
Keep fp32 master weights + fp16 autocast + GradScaler. T4 has fp16 tensor cores only, with no bf16 (Triton/SDPA don't support bf16 on sm<80) and no TF32 (TF32 is an Ampere feature). So `torch.backends.cuda.matmul.allow_tf32` is a no-op on T4. Pure fp16 weights would save memory but would break GradScaler-based training and add numeric risk, for little speed gain at 124M.

### Cited Findings
- On SM < 8.0, PyTorch SDPA flash/mem-efficient dtype lists exclude bf16 (`kHalf` / `kHalf, kFloat` only). — [pytorch v2.9.1 sdp_utils.cpp](https://raw.githubusercontent.com/pytorch/pytorch/v2.9.1/aten/src/ATen/native/transformers/cuda/sdp_utils.cpp)
- Official Triton does not support bf16 on Turing. — [triton-windows README](https://github.com/woct0rdho/triton-windows)
- T4 peak is 65 FP16 TFLOPS mixed precision at a 70 W board power. Sustained GEMMs on 70 W cards are power- and clock-limited well below peak. — [ComputerBase T4 article](https://www.computerbase.de/2018-09/nvidia-tesla-t4-turing-grafikkarte/); [NVIDIA dev forum on FLOPS benchmarking](https://forums.developer.nvidia.com/t/is-there-any-official-benchmark-tool-to-test-a-gpus-flops/270261/4)

### Inferences
- `torch.backends.cuda.matmul.allow_tf32 = True` and `torch.set_float32_matmul_precision("high")` do nothing on T4. Any fp32 matmul outside autocast runs at about 8 TFLOPS of fp32 CUDA-core throughput, so make sure all matmuls, including lm_head and the loss chunks, are under autocast.
- Pure-fp16 params: `GradScaler.unscale_` refuses fp16 gradients (from my PyTorch knowledge; not re-verified here). You would need a manual loss scale or an fp32 master copy, which is what autocast already does. Not recommended.
- For LoRA, the frozen base can be stored in fp16 (`torch_dtype=torch.float16`) while the LoRA params stay fp32. That saves about 250 MB and removes the per-step fp32-to-fp16 autocast weight casts, a small speed gain of a few %. It changes the base-weight rounding slightly, so treat it as a numerics change. peft keeps adapter weights in fp32 by default with `autocast_adapter_dtype=True` (unverified for peft 0.21).

### Gaps
- No measured T4 GPT-2 comparison of fp16 frozen base vs fp32 frozen base under autocast was found.

## Batch size, sequence packing, gradient checkpointing; achievable MFU on T4 for 124M

### Takeaway
The current run achieves about **7.8 TFLOPS ≈ 12% of the 65 TFLOPS fp16 peak** (my own calculation). For a 70 W T4, realistic sustained GEMM throughput is well below peak, so ~12% MFU is low but not wildly off. The headroom comes from bigger GEMMs (batch 16 to 32 once the logits memory is freed), all-fp16 matmuls, and removing fp32 elementwise traffic. Gradient checkpointing costs about 20 to 33% more compute and is only worth it to enable a much larger batch, which for throughput rarely pays off on T4.

### Cited Findings
- On Colab T4, the best fitting GPT-2 config without extra memory tricks was B=4, T=1024 (same tokens per step as 8x512). — [TDS GPT-2 hardware optimization](https://towardsdatascience.com/line-by-line-lets-reproduce-gpt-2-section-2-hardware-optimization-86e71c91d9bb/)
- llm.c's author tested on Colab T4 and A100. Most llm.c optimizations target A100+, "but there are still some gains" on T4. No T4 ms/step was reported in the snippet. — [llm.c search result / Akshatsoni005 fork](https://github.com/Akshatsoni005/llm.c); [llm.c discussion #677](https://github.com/karpathy/llm.c/discussions/677)
- nanoGPT computes MFU against A100 bf16 peak (312 TFLOPS), and treats more than 40 to 50% as good there. That doesn't transfer to T4. — [nanoGPT benchmarking (mintlify mirror)](https://www.mintlify.com/karpathy/nanoGPT/advanced/benchmarking)

### Inferences
- MFU arithmetic: FLOPs/token ≈ 6N + 12*L*d*T = 6*124M + 12*12*768*512 ≈ 0.80 GFLOP. Times 4096 tokens/step that is 3.28 TFLOP/step, and over 0.42 s that is about 7.8 TFLOPS, or about 12% of 65 TFLOPS. For LoRA the useful FLOPs are about 4N/token (no weight-grad GEMMs), yet the step time is 0.93x full FT. So LoRA is **overhead/memory-bound, not GEMM-bound**.
- Packing: if inputs are fixed 512-token windows from a concatenated stream, they are already "packed". Make sure there is no padding and no attention_mask (see Attention).
- Batch: after chunked CE, try batch 16x512 (or 8x1024 only if the experiment permits). Expect a few % to about 15% better tokens/s from larger GEMMs. Measure, and keep LR/token-count semantics in mind for a controlled experiment.
- Gradient checkpointing (`model.gradient_checkpointing_enable()`) recomputes the forward and slows the step by roughly 25 to 35%. Skip it at 9.6 GB / 16 GB.
- Cheap host-side wins: `pin_memory=True` + `non_blocking=True` H2D, pre-tokenized data on the GPU (400 x 4096 tokens is tiny, so keep the whole dataset on the GPU), and avoid `loss.item()` every step (log every N steps, since it forces a sync).

### Gaps
- No published measured tokens/s or MFU for GPT-2 124M training on T4 in PyTorch or llm.c was found. The 8% compile number is the only T4-specific measurement located. Recommend measuring with `torch.cuda.Event` timing and `nvidia-smi -q -d CLOCK,POWER` during the run to check for 70 W power throttling.

## LoRA specifically: is backward through frozen weights the bottleneck, and what tricks help?

### Takeaway
Yes. LoRA removes only the weight-gradient GEMMs (about one third of the FLOPs). The forward (2N) and the activation-gradient GEMMs through frozen W (2N) remain. The extra small adapter GEMMs add kernel-launch and memory overhead, which on small models can make LoRA barely faster than, or slower than, full FT. The 167 s vs 180 s result matches this. The real levers are: don't backprop where you don't need to (apply LoRA only to upper layers, so autograd stops below the lowest adapter), shrink the frozen path's precision/memory traffic, and apply the loss/lm_head tricks.

### Cited Findings
- CE-LoRA: the forward is unchanged (2bmn). Full-FT backward is 4bmn, LoRA backward is about 2bmn, "half of the computation". The activation-gradient dense GEMM through frozen W₀ is the bottleneck. CE-LoRA's approximate matmul gives 3.39x backward and up to 36.3% less total time. — [CE-LoRA (arXiv 2502.01378)](https://arxiv.org/html/2502.01378v1)
- "LoRA Is Slower Than You Think" (A100, batch 4, seq 512): GPT2-medium LoRA fwd/bwd was 48.68/44.66 ms vs full FT 27.50/36.12 ms. GPT2-XL LoRA was 97.73/124.34 vs 61.89/114.92 ms. The authors attribute this to adapters being processed sequentially (extra kernels), and propose fine-tuning only upper layers (PaCA on upper layers), which saved about 2 h on LLaMA2-7B. — [arXiv 2507.08833](https://arxiv.org/html/2507.08833v1)
- Unsloth reports large T4 speedups for Llama-family LoRA via fused kernels (1.95x on Llama-2 7B). This is Llama-architecture specific. — [HF blog: Unsloth + TRL](https://huggingface.co/blog/unsloth-trl)

### Inferences
- **Freezing embeddings / lm_head** is already the case under peft unless `modules_to_save` includes them. With GPT-2's tied `wte`/`lm_head`, make sure no grad is computed for wte: `model.get_input_embeddings().weight.requires_grad_(False)`. Because nothing below the first adapter needs grad, autograd skips dX for the embedding (cheap anyway). The lm_head dX GEMM (about 2*768*50257 ≈ 77 MFLOP/token in backward, ≈ 10% of total compute) is unavoidable, since gradients must flow into the transformer.
- **Restrict LoRA to the top k blocks** (peft `layers_to_transform=list(range(6,12))`). Autograd then never computes activation grads below block 6, cutting about half the backward FLOPs through frozen weights. This is the biggest LoRA-specific speed lever (maybe 20 to 30% per step), but it changes the experiment, so treat it as a design choice, not a free optimization.
- **Fewer, larger adapters:** target only `c_attn` (fused QKV) and `c_fc`/`c_proj` rather than many modules. Each adapted module adds 4+ small kernels in fwd and bwd. In peft, LoRA dropout of 0 avoids an extra elementwise kernel and activation copy (`lora_dropout=0.0`).
- **fp16 frozen base** (see the dtype section) cuts autocast cast-kernels for every frozen weight every step.
- **Merge-free fused LoRA kernels (Unsloth-style)** don't exist for GPT-2's Conv1D layout as far as found. torch.compile could fuse the adapter path, but carries the Turing/Triton risk noted above.
- Fused AdamW and 8-bit optimizers are irrelevant for LoRA (about 0.3M trainable params at r=8).

### Gaps
- No T4-specific LoRA vs full-FT timing breakdown for GPT-2 small found. No evidence found for or against Unsloth supporting the GPT-2 architecture in 2025-2026 releases. I didn't verify the peft 0.21 `autocast_adapter_dtype` default behaviour.

### Ranked summary (expected effect on this T4 setup, inferred unless cited above)
1. Chunked / fused linear cross-entropy: -1.5 to 2.5 GB peak; then batch 16 for about 5 to 15% tokens/s. Pure PyTorch version is Turing-safe.
2. Ensure SDPA (mem-efficient) + `is_causal`, no padding mask: 0 to 10% speed, about 1 GB memory if currently eager.
3. LoRA only: adapters on upper layers only (20 to 30%, changes the experiment); fp16 frozen base (a few %); `lora_dropout=0`, fewer target modules.
4. `AdamW(fused=True)`: 1 to 3% for full FT; safe.
5. Host hygiene (data on GPU, no per-step `.item()`, pinned memory): 0 to 5%.
6. torch.compile (default mode): about 8% steady-state on T4 (measured, B=4, T=1024), minus compile time. Net negative for a single 180 s run unless cached. Triton officially dropped sm75 at 3.3.
7. Don't bother: TF32 (no-op), bf16 (unsupported), FlashAttention-2 (Ampere+), Apple CCE Triton path (Ampere+), Unsloth (Llama-family), 8-bit Adam (memory only), gradient checkpointing (slower), CUDA graphs / max-autotune (GPU already saturated; GradScaler interplay).
