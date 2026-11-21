# Predictive validation of the correction/corruption criterion (pre-specified protocol)

Status: written and committed BEFORE any model call of this experiment
(2026-10-12). Nothing below may be changed after the first trace record is
written, except to record a deviation in the "Deviations" section at the end,
with its reason and date. The analysis script (scripts/criterion_predictive.py)
implements exactly this document.

## 1. Why this experiment

Eq. (1) of the paper, Delta = (1-a) p_fix - a p_cor, is an exact accounting
identity: with empirical plug-in estimates on the same items it reproduces the
observed net change by construction, so the fact that it "fits" the traces is
not evidence for anything. The paper's practical use of it is a decision rule:
estimate a, p_fix, p_cor on a small labeled pilot sample of traces from a
candidate workflow, and adopt the coordination step iff
p_fix/p_cor > a/(1-a). This experiment tests that use out of sample: estimates
from a pilot subset, predictions scored on held-out items of the same cell, and
(secondary) a transfer rule that predicts a cell from its pilot base accuracy
alone plus step rates measured on other tasks.

Note, stated before any data: with plug-in pilot estimates,
(1-a_P) p_fix_P - a_P p_cor_P = (n_fix_P - n_cor_P)/m exactly, so rule R1 below
is algebraically the same as "adopt iff the pilot shows a net gain". R1 is
therefore a test of whether a small labeled pilot predicts the held-out effect
of the step, which is the claim the deployment guideline makes; the structure
of the criterion is tested non-trivially only by the transfer rule R2.

## 2. Units (cells)

A cell is a (task, model, workflow) combination. The coordination step and its
"initial" and "final" answers are defined per item:

* DEBATE (N=3, R=1): initial = majority vote of the three round-0 answers;
  final = majority vote of the three round-1 answers (harness majority rule,
  ties broken by first occurrence, unparsed answers ignored).
* GE (R=2): initial = the first generation's answer; final = the answer after
  the second critique-and-revise round (unconditional revision, as in the
  paper's matrix).

An item is "correct" under the task's own deterministic scorer.

Cells (9):

| id | task | model | workflow | items | cap | source |
|:--|:--|:--|:--|--:|--:|:--|
| gsm8k_qwen_debate | GSM8K | Qwen2.5-3B | DEBATE | 80 | 640 | existing results/mechanistic.jsonl |
| gsm8k_qwen_ge | GSM8K | Qwen2.5-3B | GE | 80 | 640 | existing results/mechanistic.jsonl |
| gsm8k_llama70b_debate | GSM8K | Llama-3.3-70B | DEBATE | 80 | 640 | existing results/gsm8k_llama70b.jsonl (per-call texts) |
| math_qwen_debate | MATH L1-3 | Qwen2.5-3B | DEBATE | 80 | 1024 | new, results/math500_traces_mac.jsonl |
| math_qwen_ge | MATH L1-3 | Qwen2.5-3B | GE | 80 | 1024 | new, same file |
| mmlu_qwen_debate | MMLU | Qwen2.5-3B | DEBATE | 80 | 640 | new, results/mmlu_traces_mac.jsonl |
| mmlu_qwen_ge | MMLU | Qwen2.5-3B | GE | 80 | 640 | new, same file |
| eng_qwen_debate | MMLU-Pro eng. | Qwen2.5-3B | DEBATE | 150 | 1024 | new, results/mmlu_pro_eng_traces_mac.jsonl |
| eng_qwen_ge | MMLU-Pro eng. | Qwen2.5-3B | GE | 150 | 1024 | new, same file |

Items: the paper's own loaders with their fixed shuffle (seed 0), first n
items. For MMLU (80) and MMLU-Pro engineering (150) these are exactly the
matrix items; for MATH the first 60 of the 80 are exactly the matrix items and
20 more come from the same levels 1 to 3 pool (238 items). Caps are the
paper's: 640 tokens for GSM8K and MMLU, 1024 (the confound-free cap) for MATH
and MMLU-Pro engineering. Model: qwen2.5:3b-instruct (digest 357c53fb659c) on
the Mac's own Ollama (0.30.10, Apple M4); one generation seed (seed 1), with
the matrix's seed derivation, prompts, temperatures (0.7), and topology code,
so each trace is the seed-1 run of the matrix configuration. No new API calls
are planned for the 70B cell (its existing per-call texts suffice).

Item counts are fixed as above. The only permitted change is a reduction of
the MMLU-Pro engineering count if the 5-item throughput smoke projects more
than about 10 hours of total Mac compute: then n_eng is the largest multiple
of 10 that fits, never below 80, decided from the smoke timing alone and
recorded under Deviations before the main run. MATH and MMLU stay at 80. The
smoke's records are the first records of the main run (same code, same items)
and are not inspected for correctness before the main run starts.

Sanity check (not an endpoint): for MMLU, MATH (first 60), and engineering,
the fraction of items whose trace final answer equals the matrix seed-1 answer
of the same configuration is reported; the engineering matrix ran on a
different machine, so agreement there is expected to be lower.

## 3. Rules (decision procedures)

For each cell and each split s = 0..199, a pilot P of m = 30 items is drawn
uniformly without replacement (Python `random.Random(f"{cell_id}:{s}:20261012")`,
`.sample(range(n), m)` over the cell's items in file order); the holdout H is
the remaining n - m items.

* R1 (plug-in pilot rule, primary). a_P = pilot initial accuracy;
  p_fix_P = fixed/initially-wrong in the pilot; p_cor_P = corrupted/
  initially-correct in the pilot (a term whose denominator is zero contributes
  zero). Delta_hat = (1-a_P) p_fix_P - a_P p_cor_P. Adopt iff Delta_hat > 0
  (identical to p_fix_P/p_cor_P > a_P/(1-a_P) whenever both sides are defined).
* R2 (transfer rule, secondary). Delta_hat = (1-a_P) pfix_bar - a_P pcor_bar,
  where pfix_bar and pcor_bar are pooled over all items of the other cells of
  the same workflow (DEBATE or GE) whose task differs from the target cell's
  task (leave-one-task-out; e.g. for gsm8k_llama70b_debate both GSM8K cells are
  excluded). Only a is measured on the target task, from the pilot. Adopt iff
  Delta_hat > 0.
* R3 (base accuracy alone). Adopt iff a_P < 1/2 (the criterion's verdict if
  the step's error signal were at chance, p_fix = p_cor). Decision only.
* Baselines: always adopt; never adopt. For Delta prediction: the zero
  predictor (Delta_hat = 0), and the leave-one-task-out pooled observed Delta
  of the same workflow (item-weighted), which uses the same outside information
  as R2 but no criterion structure.

## 4. Endpoints

Observed effect on a holdout: Delta_H = final accuracy - initial accuracy on H.
A decision is correct iff (adopt and Delta_H > 0) or (not adopt and
Delta_H <= 0): a step that yields no gain is not worth its tokens.

Primary:
1. Decision accuracy of R1, averaged over the 9 x 200 (cell, split) pairs with
   cells equally weighted, against the better of the two constant baselines;
   also per cell.
2. Mean absolute error |Delta_hat - Delta_H| of R1 over the same pairs, against
   the zero predictor.

Secondary:
3. The same two metrics for R2 (and decision accuracy for R3), with R2's MAE
   compared against the pooled-Delta transfer baseline.
4. Pearson correlation between Delta_hat and Delta_H across the 9 cells within
   each split (median and 2.5 to 97.5 percentile range over splits), for R1
   and R2; and the correlation between the split-averaged Delta_hat and the
   full-cell Delta across cells.
5. Decision accuracy against the full-cell sign (Delta on all n items) as well
   as against the holdout sign.
6. Sensitivity of 1 and 2 to the pilot size, m in {10, 20, 30, 40}, same split
   construction.

Interpretation clause fixed in advance: if every cell's full-cell Delta has the
same sign, decision accuracy cannot separate any rule from the matching
constant baseline (a constant rule is then optimal by construction), and this
will be stated; the informative endpoints are then the MAE and correlation
results, and the pooled estimates below.

## 5. Pooled estimates

Per workflow (DEBATE cells, GE cells) and over all 9 cells, from summed counts:
a, p_fix, p_cor with Wilson 95% intervals; p_fix/p_cor with the Wilson-based
MOVER interval (the Fig. 2 method); a/(1-a) with the interval transformed from
the Wilson interval of a; and the pooled net change
Delta_pool = sum(n_fix - n_cor)/sum(n), with a 95% percentile interval from a
cell-stratified item bootstrap (10,000 resamples, seed 20261012). Per-cell
intervals by the same methods are reported next to them. Whether the pooled
intervals are tighter than the per-cell ones, and whether they exclude the
break-even line, is reported as observed.

Two caveats are fixed now: the GSM8K Qwen and 70B cells were analyzed before
this protocol (their full-cell Delta is already reported in the paper), so only
the six new Qwen cells are blind; results are reported for all 9 cells and for
the 6 new cells separately. Pooling across tasks mixes different a, so the
pooled ratio is compared to the pooled odds only descriptively; the pooled
Delta interval is the inferential pooled statement.

## 6. Reporting rule

Every endpoint above is reported as computed, in the paper and in
results/analysis/criterion_predictive.json, whether it favors the criterion,
is null, or is negative. If R1 does not beat the better constant baseline, or
its MAE is not below the zero predictor's, the paper says so plainly and
narrows the guideline row accordingly; no endpoint is dropped, re-defined, or
added after the data are seen, except as an explicitly labeled post-hoc
analysis.

## 7. Artifacts

* harness/traces.py: trace logging (per-agent initial and final answers for
  DEBATE; draft, critique verdict, and revision per round for GE; per-call
  tokens and latency), tests in tests/test_traces.py.
* scripts/run_traces.py: resumable Mac runner (records keyed by item and
  workflow; skips logged records).
* scripts/criterion_predictive.py: this analysis; writes
  results/analysis/criterion_predictive.json,
  paper/eaai/criterion_predictive_macros.tex, the table fragment, and the
  figure.

## Run log (appended after the smoke; not a change to the protocol)

* 2026-10-12 10:17 to 10:24 UTC: 5-item smoke (scripts/run_traces.py
  --limit 5), three task streams concurrently against the Mac's Ollama
  0.30.10 (server OLLAMA_NUM_PARALLEL=4, context 4096 per slot), model
  qwen2.5:3b-instruct digest 357c53fb659c. Smoke records: 10 MMLU, 2 MATH,
  2 engineering (the MATH and engineering smoke streams were stopped after
  their first item, enough for timing; the records are kept as the first
  records of the main run). Measured: about 12 output tokens/s per stream,
  about 36 tokens/s aggregate; output tokens per item about 1,000 (MMLU),
  4,550 (MATH), 4,300 (engineering). Projection for the protocol's counts
  (80/80/150): about 1.09M output tokens, about 8.5 to 9.5 h at the measured
  aggregate rate, i.e. under the 10 h rule, so n_eng stays at 150 (no
  reduction).
* Concurrency (three streams) is not specified by the protocol and does not
  change prompts, seeds, or caps; batched decoding can change sampled tokens
  relative to a single-stream run, which only affects the non-endpoint
  matrix-agreement sanity check.
* 2026-10-12 10:24 UTC: main run launched via scripts/_run_traces_mac.sh
  under caffeinate -dimsu (resumable; relaunch with the same command).

* 2026-10-12 10:31 UTC: the per-stream rate made MMLU-Pro engineering the
  critical path (about 340 s per item), so a second, reverse-order worker
  was added for it (scripts/_run_traces_mac.sh --second-workers; it writes
  results/mmlu_pro_eng_traces_mac_b.jsonl and both workers skip each other's
  logged items), and one for MATH once MMLU finishes. The main MATH and
  engineering workers were restarted once to load the shard-aware code (each
  lost at most its in-progress record, which was re-run from scratch with the
  same seed). The analysis merges main and shard files, keeps one record per
  item (main file first), and orders items by the task loader's order, which
  is the file order a single worker would have produced, so the split
  construction of section 3 is unchanged.
* 2026-10-12 10:49 UTC: observed aggregate rate over the first 32 minutes of
  the main run was about 27 output tokens/s (four streams; the Mac was under
  memory pressure), below the smoke's 36, which projects about 11 h in total
  rather than 9. The n_eng decision was taken from the smoke timing as the
  protocol requires and is not revisited.
* 2026-10-12 about 12:10 to 17:15 UTC: the Mac's battery ran down while on a
  weak charge and the machine stalled; the workers were not killed and
  resumed on their own. Records whose generation spanned the stall have
  inflated wall-clock and latency fields (latency is not an endpoint here);
  their text, answers, and token counts are unaffected.
* 2026-10-12 19:58 UTC: MMLU finished (80 items per workflow); the reverse
  MATH worker started automatically. It runs under the system Python 3.9.6
  instead of Homebrew Python 3.14 (PATH of the detached launcher). Checked:
  identical record schema, cap, and model; zero duplicate (item, workflow)
  keys across the two files; re-extracting its debate answers from the logged
  texts under Python 3.14 reproduces every logged answer, so the interpreter
  does not affect the traces.

## Deviations

(none yet)

## Post hoc analyses (added 2026-10-15, after all data were seen)

Added in answer to a minor-revision review that asked for per-cell results
instead of the 200 overlapping pilot splits. Nothing above is changed; these
are labeled post hoc in the paper (Table tab:predictive, guideline row).
Implemented in scripts/criterion_predictive.py, `posthoc_screen`
(numpy seed 20261015, 10,000 resamples):

* Per cell: the net change Delta of the step on all items with a 95%
  item-bootstrap percentile interval, the exact two-sided McNemar p on the
  discordant items, the decision of the base-accuracy screen applied to the
  whole cell (adopt iff a < 1/2, rule R3 with the full cell as the pilot), and
  whether it matched the sign of Delta (not adopting is right when Delta <= 0,
  as in section 4).
* Groups: cells with a > 1/2, cells with a < 1/2, and the blind cells with
  a > 1/2; Delta = sum(fix - cor)/sum(n), with an item-cluster bootstrap that
  draws each task's items once and applies the draw to every cell of that task
  (the debate and GE cells of a task, and the two GSM8K debate cells, share
  their items).
* Run-to-run reference: the spread of the final accuracy of the same
  configuration over the trace run and the two matrix seeds (from the sanity
  check of section 2).

Results as computed: the screen's decision matched the sign of Delta in 8 of
9 cells (miss: MATH GE, a = 0.79, Delta = +1.2 [-7.5, +8.8]); no cell's
interval excludes zero; pooled a > 1/2 (7 cells, 560 traces) Delta = -3.6
[-6.6, -0.5]; pooled a < 1/2 (2 engineering cells, 300) +2.0 [-3.7, +7.3]
(McNemar p = 0.86 and 0.58 per cell); blind a > 1/2 cells (4, 320) -4.1
[-8.4, +0.3]. Engineering debate final accuracy across runs: 33.3 (M4 trace)
vs 22.0 and 29.3 (RTX 4060 seeds), a spread of 11.3 points; GE 3.3 points.

Added 2026-10-15 (third review round, descriptive only): no cell has base
accuracy between 0.320 (engineering debate) and 0.625 (MMLU debate), so the
data place the screen's threshold only somewhere in that gap; 1/2 is the value
the identity gives at p_fix = p_cor. Emitted as \cpGapLo, \cpGapHi.
