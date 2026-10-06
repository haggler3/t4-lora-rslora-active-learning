# Data selection / active learning for LM (continued) pre-training: methods that beat random and are cheap to demo

Setup being optimized: GPT-2 small (124M), continued pre-training on WikiText-2 (4,671 x 512-token train blocks), batch 8, 400 steps, T4, rsLoRA r=64. AL arm: 25% budget (~1,168 blocks) in 4 rounds; strategies random / max_loss / max_loss_trim (top-5% dropped), scored on 1,500 random candidates. Note: 400 x 8 = 3,200 block-visits, so the full-data arm sees ~0.68 epoch while the 25% arm repeats its subset ~2.7 times.

## Q1. RHO-LOSS / reducible holdout loss, learnability scores, irreducible-loss models: what gains for LMs?

### Takeaway
RHO-LOSS scores each example by training loss minus "irreducible loss" (the loss of a small model trained on holdout data). It skips three kinds of points: noisy ones (high irreducible loss), redundant ones (low training loss) and less relevant ones. It gives large speedups on noisy vision data and on GLUE fine-tuning (CoLA, SST-2), and a cheap, weak irreducible-loss model is enough. LM-specific versions (Irreducible Curriculum at 1B; Rho-1 token-level excess loss at 1B/7B) report gains. However, one controlled BERT/T5 pre-training study found the gains disappear against a properly decayed-LR baseline.

### Cited Findings
- Selection rule: argmax over a candidate batch B_t of L[y|x; D_t] - L[y|x; D_ho], i.e. current-model training loss minus irreducible holdout loss (IL). Only the top n_b of a randomly pre-sampled large batch of size n_B get a gradient step. IL is computed once before training, using a model trained only on the holdout set ("Approximation 2"). The default ratio is n_b/n_B = 0.1 — [Mindermann et al. 2022, RHO-LOSS](https://arxiv.org/abs/2206.07137)
- RHO-LOSS gives low priority to redundant points (low train loss), noisy points (high IL) and less relevant or outlier points (high IL because the holdout has few nearby points) — [RHO-LOSS](https://arxiv.org/abs/2206.07137)
- The IL model can be small and inaccurate. A small CNN with 21x fewer parameters and 29x fewer FLOPs than the target ResNet-18 sped training up as much as or more than a full-size IL model, even though its accuracy was 7-11.5 points lower. An IL model can also be trained with no holdout: split the training set in two, train an IL model on each half, and score each half with the model trained on the other half. One IL model can be reused across architectures and hyperparameter sweeps — [RHO-LOSS](https://arxiv.org/abs/2206.07137)
- Text results (ALBERT-v2 fine-tuning, half of the data used for training). On CoLA, RHO-LOSS reached 75% accuracy in 3 epochs versus 34 for uniform sampling, and reached 80% in 39 epochs, which uniform never reached (76% final). On SST-2 it reached 90% in 3 epochs versus 6 for uniform. Selecting by plain training loss reached 75% on CoLA in 8 epochs but never reached 80% (78% final) — [RHO-LOSS Table 2](https://arxiv.org/abs/2206.07137)
- On Clothing-1M (noisy web data), RHO-LOSS needed 18x fewer steps and reached 2% higher final accuracy (72% vs 70%), using an IL model with 62% accuracy trained on ~10% of the data. It used 2.7x fewer FLOPs to reach peak uniform accuracy — [RHO-LOSS](https://arxiv.org/abs/2206.07137)
- The cost of the selection forward passes per step is roughly n_B/(3 n_b) times the cost of a training step. The paper reports speedups in epochs, not wall-clock time — [RHO-LOSS](https://arxiv.org/abs/2206.07137)
- Irreducible Curriculum (an LM variant) prioritizes high-learnability samples. It uses a small proxy model to simulate per-sample loss along the main model's trajectory, avoiding extra forward passes at train time. On RedPajama-1B it reports consistently lower validation perplexity across all 7 domains than uniform and anti-curriculum, lower sharpness, and better 5-shot MMLU — [Fan & Jaggi 2023](https://arxiv.org/abs/2310.15389)
- Rho-1 (token-level RHO). The score is L_Δ(x_i) = L_θ(x_i) - L_RM(x_i), where the reference model is trained on high-quality data (0.5B math tokens or 1.9B general tokens, 3 epochs). Loss is applied only to the top 60% of tokens (1B model) or top 70% (7B). Reported gains: up to 30% few-shot accuracy gains on 9 math tasks with 15B OpenWebMath tokens, +16.5% average on math for 1B, and +6.8% average across 15 tasks when continually pre-training on 80B general tokens. Rho-1-7B matched DeepSeekMath-7B with 15B vs 500B tokens — [Lin et al. 2024, Rho-1](https://arxiv.org/html/2404.07965)
- Counter-evidence: when BERT and T5 are pre-trained under a fixed compute budget, the training, validation and downstream gains of selective backprop and RHO-LOSS "vanish" against a baseline with a fully decayed learning rate — [Kaddour et al. 2023, No Train No Gain](https://arxiv.org/abs/2307.06440)

### Inferences
- Recipe A (RHO-LOSS acquisition for the 25% AL arm; about 30 lines):
  1. Split the 4,671 train blocks into halves H1 and H2. Fine-tune two IL models (frozen GPT-2 + LoRA, or distilgpt2) for ~100-150 steps, one on each half, so the IL for blocks in H1 comes from the model trained on H2 and vice versa. This follows RHO-LOSS's "no holdout" variant. Using WikiText-2 validation as the holdout would leak into evaluation, so avoid it.
  2. Cache IL[i] = mean token NLL for every block. This costs one forward pass over 4,671 blocks, about 1 minute on a T4 in fp16.
  3. In each AL round, score the 1,500 candidates with score = loss_current(i) - IL[i] and acquire the top-k.
  - A cheaper variant uses a single IL model trained on a random 10-20% of train that is excluded from the acquisition pool.
  - The cheapest variant uses the frozen base GPT-2 as the "reference". This is not a true IL model: it measures how much the adapter has already learned. It is useful only as an ablation.
- Recipe B (Rho-1 token masking; about 10 lines): compute per-token CE for the current model and the IL/reference model, keep the top 60% of tokens by excess loss, and average loss only over the kept tokens. One caveat: WikiText-2 perplexity is measured on all tokens, so skipping tokens could hurt the reported metric even if it helps downstream. This has not been tested at this scale.
- Kaddour et al. imply the random and full-data baselines must use the same well-tuned, fully decayed LR schedule. Otherwise an apparent win may just come from a schedule artifact.

### Gaps
- I found no published RHO-LOSS result for causal-LM continued pre-training at about 100M parameters on a clean corpus like WikiText. The NLP evidence is classification fine-tuning (CoLA, SST-2).
- I did not find exact perplexity deltas for Irreducible Curriculum, or a breakdown by model size for Rho-1 below 1B.

## Q2. Online batch selection, JEST, prioritized training, loss-based curricula: do max-loss strategies fail on noisy text, and what are the fixes?

### Takeaway
Pure max-loss selection is the textbook failure case. It over-selects noisy, ambiguous and outlier points, and on noisy web data it does worse than uniform sampling. Proven fixes: subtract a reference/irreducible loss (RHO-LOSS, Rho-1, JEST "learnability"); trim both tails (keep the middle of the loss distribution); or use risk-averse thresholding (ESLM/CVaR). JEST's headline gains come from multimodal contrastive training with a pretrained reference model, not LM pre-training.

### Cited Findings
- Train-loss and gradient-norm selection pick far more label-corrupted points than uniform sampling (~25% vs the 10% base rate on CIFAR-10 with 10% label noise). On CIFAR-100, loss-based selection picks low-relevance classes about 47% of the time vs ~20% for uniform. On Clothing-1M, loss and gradient-norm selection failed to match uniform. RHO-LOSS keeps corrupted-point selection near ~5% — [RHO-LOSS Fig. 3 and Sec. 4.4](https://arxiv.org/abs/2206.07137)
- With 10% label noise on CIFAR-100, train-loss selection never reached 40% accuracy (4% final), uniform took 79 epochs, and RHO-LOSS took 49 — [RHO-LOSS Table 2](https://arxiv.org/abs/2206.07137)
- JEST selects whole batches jointly (exploiting dependencies within a contrastive batch) using a learnability score defined against a pretrained reference model trained on curated data. It reports up to 13x fewer iterations and 10x less compute than the prior state of the art for multimodal contrastive learning — [Evans et al. 2024, JEST](https://arxiv.org/abs/2406.17711)
- A JEST-style joint batch selection with learner and reference models applied to machine-translation fine-tuning reports up to 5x better data efficiency than an iid baseline — [Dynamic Jointly Batch Selection, EMNLP 2025](https://arxiv.org/html/2511.04406v1)
- ESLM runs online token-level batch selection on per-token loss or entropy with value-at-risk thresholding. In the loss-based setting it recovers CVaR loss minimization. On GPT-2 pre-training it reports fewer training FLOPs with equal or better perplexity and downstream results. Exact numbers are not in the abstract — [ESLM, TMLR](https://arxiv.org/abs/2505.19893)
- GREATS selects online batches greedily using a Taylor approximation of how much each batch reduces validation loss, without a reference model. It reports faster convergence and better generalization for LLM fine-tuning — [GREATS, ICML/NeurIPS 2024 listing](https://icml.cc/virtual/2024/36426)
- In LM pre-training under a fixed compute budget, online selection (selective backprop, RHO-LOSS) showed no gain over a decayed-LR baseline — [No Train No Gain](https://arxiv.org/abs/2307.06440)

### Inferences
- The existing max_loss_trim, which drops only the top 5%, is a weak version of the tail-trim fix. On WikiText-2 the highest-loss 512-token blocks are likely tables, lists, foreign names and "@-@" / "<unk>"-heavy markup. These are learnable in-distribution patterns, not label noise, so max-loss may hurt less here than on web data. It still risks lowering diversity.
- Recipe C (online super-batch selection; about 15 lines, best suited to the full-data arm): at each step, draw 32-64 random blocks, run torch.no_grad forward passes in fp16 (chunked to fit T4 memory), compute RHO = loss - IL (IL from Recipe A), and backprop on the top 8. The cost is about n_B/(3 n_b) of a training step: with 32 candidates, roughly 1.3-2.3x the step time. That may not fit a strict 3-minute budget, so compare at equal steps and also report wall-clock time.
- JEST requires a contrastive (batch-coupled) loss to get its joint-selection benefit. For causal LM, per-example losses are independent, so JEST reduces to RHO-LOSS-style learnability top-k.

### Gaps
- I did not find ESLM's exact numbers or model sizes, or any head-to-head comparison of max-loss, RHO and random selection on a clean Wikipedia-style LM corpus.

## Q3. Static filtering: perplexity pruning, DSIR, SemDeDup/D4, influence (LESS, DsDm), Ask-LLM. Which show gains at ~100M scale?

### Takeaway
Perplexity pruning with a small reference model is the cheapest method with solid evidence. At 124M, keeping the middle 50% of the perplexity distribution beat no pruning (Marion et al.). A 125M reference model improved 1B-3B models (Ankner et al.), but which band is best depends on the dataset, and pruning hurt test-set perplexity even while downstream scores improved. D4 shows gains from 125M up, especially when data is repeated, which matches the multi-epoch 25% arm. DSIR, LESS and DsDm target distribution matching or a downstream task and add little when the training pool is already in-domain.

### Cited Findings
- Marion et al. trained 124M and 1.5B models (124M: 8,000 steps). Keeping the middle 50% of the perplexity distribution gave a 0.97% test-perplexity improvement over no pruning at 124M, and middle 30% gave 0.80%. The middle band beat the top and bottom bands at all pruning ratios. At 1.5B the improvement was 1.5%. EL2N and memorization scores did no better than random despite their cost. The reference was a 52B model trained on CommonCrawl — [Marion et al. 2023, When Less is More](https://arxiv.org/html/2309.04564)
- Ankner et al. used a 125M reference model trained on a random subset of the pretraining data (26B tokens). Pruning with it improved downstream averages by up to 2.04 points for a 3B model, and reached baseline downstream accuracy in 1.31x (Pile) and 1.45x (Dolma) fewer steps at 1B. On the Pile, high-perplexity selection worked best (+1.89); on Dolma, medium-perplexity selection worked best (+1.51). Medium selection on the Pile hurt (-0.23). Pruned 1B models had worse test-set perplexity (8.51 vs 7.83) while downstream scores improved (13.73 to 15.62) — [Ankner et al. 2024, Perplexed by Perplexity](https://arxiv.org/html/2405.20541)
- DSIR estimates importance weights between a target and the raw pool using hashed n-gram features, then resamples. It selected 100M Pile documents in 4.5 hours. It matched expert curation for continued pre-training across 8 targets and gave +2-2.5% GLUE vs random and heuristic filtering when targeting Wikipedia and books. Its KL-reduction metric correlates r=0.82 with downstream accuracy — [Xie et al. 2023, DSIR](https://arxiv.org/abs/2302.03169)
- D4 deduplicates with SemDeDup and then selects diverse, non-prototypical documents using clusters of pretrained-model embeddings. It reports 20% efficiency gains and up to 2% average accuracy gains on 16 NLP tasks at 125M-6.7B. Repeating intelligently selected data beat the baseline, while repeating randomly selected data did worse than the baseline — [Tirumala et al. 2023, D4](https://arxiv.org/abs/2308.12284)
- LESS takes LoRA gradients, applies random projection and Adam preconditioning, and selects examples by similarity to a few target examples. A LESS-selected 5% of the data often beats full-data instruction tuning, and small models can select data for larger ones — [Xia et al. 2024, LESS](https://arxiv.org/abs/2402.04333)
- DsDm picks data to optimize target-task performance via datamodels and reports about a 2x compute multiplier. It also finds that selecting by similarity to "high-quality" sources can match or underperform random selection — [Engstrom et al. 2024, DsDm](https://arxiv.org/abs/2401.12926)
- Ask-LLM, which scores quality by prompting an instruction-tuned LLM, beat full-data training even after rejecting 90% of the data and converged up to 70% faster. Density sampling, a coverage method, recovered full-data performance without any quality model — [Sachdeva et al. 2024, How to Train Data-Efficient LLMs](https://arxiv.org/abs/2402.09668)

### Inferences
- Recipe D (middle-band perplexity; about 5 lines, fixes max_loss_trim): use the base GPT-2 loss, or the current loss, on all blocks or the 1,500 candidates. Keep the 25th-75th percentile (or drop the bottom 20% and top 10%), then take a random sample from that band. This has the best small-scale evidence of any selection rule (Marion, 124M).
- Because the experiment measures WikiText-2 perplexity, Ankner's finding that pruning can raise test-set perplexity is a warning. Quality-style filtering may look worse on PPL even if it is "better" data. Selection that preserves the distribution (middle band, RHO, diversity) is safer for a PPL metric.
- DSIR with target = WikiText-2 validation would just estimate the train distribution, which is already in-domain. Expect roughly random-level results unless the pool is mixed (for example WikiText-103 plus OpenWebText).
- LESS and DsDm (per-example gradients or datamodels) and Ask-LLM (an LLM pass over the data) are too costly for a 3-minute T4 demo. Skip them.

### Gaps
- I did not fetch exact SemDeDup numbers for small OPT models, or the Albalak et al. 2024 survey ([arXiv 2402.16827](https://arxiv.org/abs/2402.16827)). No direct evidence exists on static perplexity pruning for LoRA continued pre-training of GPT-2 on WikiText.

## Q4. Diversity-aware acquisition (k-center, coreset, BADGE) vs pure uncertainty for LMs

### Takeaway
For LM pre-training data, the evidence for diversity and coverage selection comes mostly from embedding-based dedup and prototypicality (D4) and density sampling (Sachdeva). These match or beat random selection, and diversity matters more when data is repeated. I found no BADGE-style result for causal-LM pre-training. Score-only "important example" coresets are reported to collapse below random at high pruning rates.

### Cited Findings
- D4: embedding-cluster-based diversification gives a 20% efficiency gain at 125M-6.7B, and intelligent repetition beats random repetition — [D4](https://arxiv.org/abs/2308.12284)
- Density (coverage) sampling recovers full-data performance without quality scoring — [Sachdeva et al. 2024](https://arxiv.org/abs/2402.09668)
- State-of-the-art coreset methods that pick the highest-importance examples work at low pruning rates but drop catastrophically, below random, at high pruning rates — [Coreset selection via LLM concept bottlenecks (search summary)](https://arxiv.org/html/2502.16733v2)
- In French LM pre-training, diversity sampling and random sampling both cut data by up to 94% while maintaining performance, with some tasks up to 10 points better under diversity sampling (secondary search snippet, not verified in full text) — [ACL Findings 2026](https://aclanthology.org/2026.findings-acl.1707.pdf)

### Inferences
- A 25% budget is a moderate pruning rate. Pure uncertainty (max_loss) concentrates acquisitions on one high-loss "style" of block, so adding diversity is cheap insurance.
- Recipe E (loss plus k-center hybrid; about 20 lines): during the candidate scoring pass, keep mean-pooled last-layer hidden states (768-d), L2-normalized. Pre-filter to the top 3k candidates by RHO score (or middle-band loss), then run greedy k-center (farthest-first) on the embeddings to pick k. Alternatively, run k-means into k clusters and take the highest-RHO block per cluster. This needs no extra forward passes and costs about a second for 1,500 x 768.
- BADGE-style gradient embeddings for LMs (last-layer gradient, which is 50k-vocab x 768) are too large. Hidden-state embeddings are the practical proxy.

### Gaps
- I found no controlled study of k-center, BADGE or coreset acquisition against random for GPT-2-scale continued pre-training on WikiText.

## Q5. Which method most likely gives a visible win over random in a ~3-minute T4 WikiText-2 run, and why? (Ranked list)

### Takeaway
The best bet is RHO-LOSS-style reducible-loss acquisition (current loss minus the loss of a cheap irreducible-loss model trained on the other half of train). It is the only method with direct evidence for both failure modes of max_loss (noise and outliers on one side, redundancy on the other), it works with small or weak IL models, and its NLP evidence is strongest at small scale. The cheapest fallback is middle-band perplexity selection (Marion et al. at 124M). Expect any win over random to be small (around 1% PPL at best, judging from Marion's 0.8-1%) on a clean, in-domain corpus. Kaddour et al. warn that gains can vanish against a well-tuned baseline.

### Cited Findings
- Reducible-loss selection: CoLA speedup over uniform above 10x; cheap and small IL models work; noise-robust — [RHO-LOSS](https://arxiv.org/abs/2206.07137)
- Middle-band perplexity at 124M: +0.8-0.97% test perplexity improvement over no pruning — [Marion et al.](https://arxiv.org/html/2309.04564)
- The best perplexity band varies by dataset, and pruning can worsen test-set perplexity — [Ankner et al.](https://arxiv.org/html/2405.20541)
- Gains vanish against a fully decayed-LR baseline under equal compute — [Kaddour et al.](https://arxiv.org/abs/2307.06440)

### Inferences
Ranked for this setup (expected win, then cost, then evidence):
1. **RHO-LOSS acquisition (Recipe A)**. Score = loss_cur - IL; two half-split IL LoRA models, ~2x150 steps, about 1-1.5 min extra; ~30 LOC. It directly fixes the max_loss pathologies.
2. **RHO plus k-center diversity (Recipe E)**. Free on top of #1; protects against mode collapse at a 25% budget.
3. **Middle-band perplexity (Recipe D)**. Near-zero cost; best small-scale LM evidence; replaces max_loss_trim. Try keeping the 20th-70th percentiles.
4. **Online super-batch RHO selection (Recipe C)** for the full-data arm. Possibly the largest effect per step, but it costs 1.3-2.3x step time with 32 candidates. Report both equal-steps and equal-wall-clock results.
5. **Rho-1 token-level excess loss (Recipe B)**. ~10 LOC; strong at 1B/7B, untested at 124M, and may hurt full-token PPL.
6. **D4/SemDeDup**. Little duplication exists in WikiText-2, so expect a small gain; the diversity component is covered by #2.
7. **DSIR**. Useful only with a mixed or out-of-domain pool.
8. **LESS / DsDm / Ask-LLM / JEST**. Too costly, or mismatched to this objective (contrastive loss, downstream-task targets).

Practical notes for the experiment:
- Run 3 or more seeds. Random-vs-selection gaps of ~1% PPL are within seed noise at 400 steps.
- Use the same LR schedule (fully decayed) for all arms.
- With the half-split IL setup, no data needs to be held out: each half is scored by the model trained on the other half, so the whole pool stays available for acquisition.
- Report validation and test PPL, plus steps or seconds to reach the random arm's final PPL.

### Gaps
- No source directly measures any of these methods in a ~3-minute, 400-step LoRA run. The size of the expected win is extrapolated from the Marion (124M) and RHO-LOSS (CoLA/SST-2) results.
