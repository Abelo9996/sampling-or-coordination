# Preregistration: Sampling vs Coordination (confirmatory expansion)

Frozen analysis plan for the confirmatory (Phase 1B) experiments. Written before
running the additional models and larger samples. The Qwen2.5-3B anchor and the
existing analyses are already reported; this document governs the new
confirmatory runs and their statistics.

## 1. Primary questions and hypotheses (at supported strength)

- **H-primary.** On verifiable closed-form tasks at matched token cost,
  independent sampling (self-consistency) is not improved upon by interactive
  coordination (debate, generator-evaluator, hierarchical). Directional: the
  independent-sampling frontier weakly dominates.
- **H-task.** The independent-sampling advantage is larger on reasoning-style
  tasks than on knowledge-recall, and (exploratory) coordination may narrow or
  reverse this on a coordination-friendly task (code).
- **H-scale.** The coordination gap does not depend on model scale in the tested
  range (the matched 3B-vs-7B interaction is small; we test whether a larger
  matched sample detects any interaction).

No claim of an intrinsic topology defect, no scaling law, no crossover estimate.

## 2. Models (anchor-plus-breadth)

- **Anchor:** Qwen2.5-3B-Instruct, full topology and budget sweep (already run).
- **Matched scale:** Qwen2.5-7B-Instruct, headline configs, matched items.
- **Breadth (>=1 second dense family, target 2):** Llama-3.x (3B, 8B) and Gemma-2
  (2B, 9B), headline configs only, frozen. Justified as stable, widely used,
  downloadable dense families chosen for matched-scale controls, not as newest
  representatives (Gemma 4 and Qwen3 exist by 2026; Llama 4 is MoE and less
  comparable to small dense Qwen). Selection criteria: fixed downloadable weights,
  instruct variants, >=2 useful sizes, reliable seed behavior, manageable cost.
- Expansion to full sweeps on a breadth model happens only if a coordination
  workflow reaches within 2 points of that model's sampling frontier.

## 3. Tasks and frozen baselines

- Existing: GSM8K, MATH (1024 cap), MMLU (subject-labeled).
- **New coordination-friendly task: code (HumanEval/MBPP), with a frozen baseline
  protocol fixed before running:** independent generation plus a frozen
  model-based selector, versus generator-critic revision; every workflow gets
  identical public-test access; selection never uses hidden tests. pass@k is
  reported separately from single-answer accuracy.

## 4. Sample sizes (power-derived)

From the existing logs (paired per-item accuracy differences):

| Comparison family | SD of paired diff | MDE at current n | n for 5-pt effect (80% power) |
|---|---|---|---|
| GSM8K SC vs coordination | ~22-31 | ~7-10 pt (n=80) | ~150-300 |
| MATH SC vs coordination | ~25-33 | ~9-12 pt (n=60) | ~200-340 |
| MMLU SC vs coordination | ~35-44 | ~11-14 pt (n=80) | ~375-605 |
| 3B-vs-7B interaction | ~28 | ~11 pt (n=50) | ~255 (matched) |

Frozen plan:
- **Screening sample: n=300 per task** (stratified; MMLU stratified by subject).
  Detects ~4-5 pt effects on math and ~5-6 pt on MMLU at 80% power.
- **MMLU extended to n=600** because its variance is high and the observed gaps
  (2.5-8 pt) sit below the n=300 MDE for the smaller ones.
- **Matched scaling sample: n=300 matched items** across sizes (detects a ~5-pt
  interaction).
- **Two-stage:** run the screening n, then expand a comparison to double n only
  if its 95% CI intersects the practical-equivalence region (+/- 3 points).

## 5. Configurations and seeds

- Frozen headline set on every model: SA, SC(k=4), SC(k=8), debate(N=3,R=1),
  GE(R=2), HIER(M=3). Anchor keeps the full sweep.
- **>=3 seed sets** for all headline configs (5 if variance warrants). A "seed
  set" = one independent group of generations for the whole config; for SC(k=8)
  that is 8 fresh sampling seeds; seed identifiers aligned across workflows;
  aggregators deterministic (majority with a fixed tie-break: prefer the greedy
  answer). This is stated explicitly to avoid the current 2-seed ambiguity.

## 6. Statistical analysis plan (frozen)

- Primary inference: **seed-aware hierarchical bootstrap** (resample items, then
  seeds; MMLU adds a subject level), 10,000 replicates, within-task Holm
  correction across the three coordination comparisons. Report effect size, 95%
  paired-difference CI, and adjusted p together.
- Sensitivity: exact McNemar (item-only) and a paired permutation/sign-flip test.
- Scale: a model-size-by-workflow interaction test on matched items with a
  bootstrap CI (the current matched result is -2.0, p=0.76).
- Frontier: bootstrap Pareto-membership probability per config.
- **Cross-model cost:** build a within-model cost-accuracy frontier per model;
  never compare raw token counts across families as equal compute; compare
  coordination premiums and normalized cost ratios across models.
- Report total token work and idealized parallel critical-path latency
  separately.

## 7. Confirmatory vs exploratory

Confirmatory: the frozen headline comparisons on the frozen samples/seeds above.
Exploratory (labeled as such): any post-hoc implementation repair, the router,
plan-validity judgments (rubric-based, exploratory), and any full-sweep expansion
triggered by a near-frontier breadth model.

## 8. Backend and logging

API (OpenRouter/direct) for screening only; pinned-weight vLLM for all
confirmatory/headline/artifact runs. For any API call, pin and log concrete slug,
provider, provider order, fallback policy, sampling params, model revision,
returned metadata, and timestamps; never use "latest" aliases.

## 9. Artifact

Repository plus archival DOI, commit hash, prompts, model and quantization
digests, all seeds, raw generations for the confirmatory matrix, parsed outputs,
parser plus unit tests and a manual audit, per-figure scripts, environment
lockfile, and license.
