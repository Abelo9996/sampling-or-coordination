# Protocol: rev3 reruns (repaired HIER on every task, HumanEval extractor, MATH L1-3 at cap 2,048)

Written 19 Nov 2026 and committed before any rev3 record exists. This commit is the
pre-registration: no record under `results/rev3/` existed when it was made. The only
outputs of the new code so far are offline tests (`tests/test_rev3.py`) and a driver
test against a fake local server that returns canned text (no model was called).

Scope is exactly three review points (review rev2, Part 1 issues 1 to 3, Tier 2):
(a) the repaired HIER (HIERFIX) on every headline task, as the headline HIER;
(b) HumanEval rerun with raw text logged and a fixed extractor;
(c) MATH levels 1 to 3 at the 2,048-token cap, so that the levels 1 to 3 versus 4 to 5
comparison has no cap confound. The headline matrix also moves to the larger runs.
Anything else (new models, human audit, figures) is out of scope.

## 1. Code (all committed with this protocol)
- `scripts/rev3_workflows.py`: HIERFIX for every task (Section 2).
- `harness/tasks.py`: new functions `code_blocks_all`, `code_block_entry`,
  `code_canonical_v2`, `humaneval_extract_item_v2`, task entry `humaneval_entryblock`
  and `HUMANEVAL_EXTRACTORS`. Additive only: `code_block`, `code_canonical` and the
  `humaneval` task are unchanged, so earlier records reproduce byte for byte.
- `scripts/run_rev3_campaign.py` (driver, one seed per process) and
  `scripts/run_rev3_campaign.sh` (launcher: queue, tunnel, shards, resume, STOP file,
  lock, hand-over from the power campaign). Log `results/logs/rev3_campaign.log`.
- `scripts/rev3_estimate.py`: the GPU-hour estimates of Section 5.

## 2. HIERFIX on every task
HIERFIX_M3 is `harness.topologies.hier` (M=3) called unchanged, with one change: the
reminder at the end of the aggregator prompt. Rule, fixed before any rev3 output:
if the task reminder asks for the template line `The answer is P.` with a placeholder
letter P, the reminder becomes
`End with exactly one line: 'The answer is P.' with P replaced by <D>.`
where <D> is the definition of P in the task's own first-call prompt ("where P is <D>.").
Otherwise the reminder is unchanged and HIERFIX is byte-identical to HIER.

| Task | HIER reminder | HIERFIX reminder |
|-|-|-|
| GSM8K | `...'The answer is N.'` | `...'The answer is N.' with N replaced by the final number.` (identical to the HIERFIX already run under EXTRA_WORKFLOWS_PROTOCOL; asserted in code) |
| MMLU | `...'The answer is X.' (a letter A-D).` | `...'The answer is X.' with X replaced by the letter (A, B, C, or D) of the correct choice.` |
| MMLU-Pro eng. | `...'The answer is X.' (the letter of one of the listed options, A-J).` | `...with X replaced by the letter (<the item's own letters, as in its prompt>) of the correct choice.` |
| MATH-500 | `Give your final answer in \boxed{}.` | unchanged (no placeholder): HIERFIX = HIER |
| HumanEval | ``Give the complete final function in one ```python code block.`` | unchanged (no placeholder): HIERFIX = HIER |

Known before writing this (no HIERFIX accuracy on any new task was computed):
GSM8K HIER aggregator outputs contain the literal `The answer is N.` in 783 of 1,400
phase A records and have no parsed answer in 311; MMLU HIER (July matrix, text not
logged) has no parsed answer in 16 of 160 records; MATH HIER returns an empty
`\boxed{}` in 2 of 714 (levels 1 to 3) and 3 of 786 (levels 4 to 5) records.

Consequence for MATH: HIERFIX prompts equal HIER prompts, so the existing HIER_M3
records of phase B (levels 1 to 3, cap 1,024) and phase H (levels 4 to 5, cap 2,048)
are HIERFIX records and are not rerun. The level 1 to 3 run at cap 2,048 (R5) runs
it under the label HIERFIX_M3.

## 3. HumanEval extractor (v2, "entry block")
Old (`code_canonical`, unchanged): the last fenced block (an unterminated final fence,
as at the cap, counts as a block); no fence: the whole text. If that code defines the
entry function at top level, program = prompt preamble + code; else it is read as a
body appended to the prompt. Docstrings removed, `ast.unparse`.
New (`code_canonical_v2`): candidates are every fenced block in order of appearance,
with the unterminated final fence (if any) last. Choose the LAST candidate that parses
(`ast.parse`) and defines the entry function at top level. If none does, use the old
choice. Program assembly, canonical form and scoring are the old ones. So v2 differs
from v1 only when the last block does not define the entry function and an earlier
block does (typical case: solution block followed by a test or usage block). Top-level
statements inside the chosen block (example calls, asserts) are kept, as before.

Generation does not depend on the extractor (the extracted program only feeds the
votes), so one run serves both: every call logs its raw text and `extracted` (v1) and
`extracted_v2`; each record has `pred`/`correct` (v1, as the topology returns) and
`pred_v2`/`correct_v2` (v2, same final calls, same `topologies._majority` vote); SC
records also carry the k=2 and k=4 prefix votes under both. Scoring: the harness
sandbox on the Mac (macOS backend; the earlier docker scores replicated 926 of 926
on this Mac). Both extractors are always reported side by side; v2 is primary.

## 4. Runs (RTX 4060, user-level Ollama with 6 parallel slots, host label rtx4060-pc-par6)
Items in harness order (`task["load"](n, split="all")`, seed-0 shuffle); SA greedy,
written once under seed 1; workflows and parameters are the harness ones (SC
temperature 0.7, DEBATE N=3 R=1, GE R=2, HIER/HIERFIX M=3). Ollama's repeat-limit
abort is recorded, not fatal (`harness/repeat_abort.py`, as in the MMLU-Pro matrix);
aborts go to `<prefix>_aborts.jsonl`. Calls with input plus output at or above the
model's context (4,096 or 8,192) are counted and reported.

| Run | Task, items | Seeds | Cap | Model alias | Configurations | Output prefix |
|-|-|-|-|-|-|-|
| R1 | GSM8K, first 500 | 1-3 | 640 | qwen2.5:3b-instruct | HIERFIX | results/rev3/gsm8k_qwen3b |
| R2 | MMLU, the 80 of the matrix | 1-2 | 640 | qwen2.5:3b-instruct | HIERFIX | results/rev3/mmlu_qwen3b |
| R3 | HumanEval, all 164 | 1-2 (SA: 1) | 1,024 | qwen2.5:3b-instruct | SA, SC k=8, DEBATE, GE, HIERFIX | results/rev3/humaneval_qwen3b |
| R4 | MMLU-Pro engineering, all 150 | 1-2 | 1,024 | qwen2.5:3b-instruct | HIERFIX | results/rev3/mmlu_pro_eng_qwen3b |
| R5 | MATH-500 levels 1-3, all 238 | 1-3 (SA: 1) | 2,048 | qwen2.5:3b-instruct-ctx8k | SA, SC k=12, DEBATE, GE, HIERFIX | results/rev3/math500_cap2048_qwen3b |

Seeds, caps and aliases follow the existing runs of each task, so that the new
configurations sit beside their comparators. R5 mirrors phase H (levels 4 to 5) in cap,
alias, SC k and seeds. Comparator data used from earlier runs (not rerun):
GSM8K phase A (`results/power/gsm8k_qwen3b_s*`, 500 x 3, SC k=8, DEBATE, GE, HIER, SA;
first 100 items made on a one-slot server); MATH levels 1 to 3 phase B
(`results/power/math500_qwen3b_s*`, 238 x 3, cap 1,024, SC k=12); MATH levels 4 to 5
phase H (`results/power/math500hard_qwen3b_s*`, 262 x 3, cap 2,048); MMLU matrix
(`results/mmlu_matrix.jsonl`, 80 items, seeds 1-2, cap 640, M4 laptop, July);
MMLU-Pro engineering matrix (`results/mmlu_pro_eng_matrix.jsonl`, 150 x 2, cap 1,024,
RTX 4060 one-slot). Stated differences: R2 and R4 run on the six-slot server, their
comparators on the M4 (MMLU) or the one-slot service (MMLU-Pro).

Hand-over: the launcher waits for the phase A completion line in the power log, then stops the
power launcher before phase C (GSM8K items 501 to 1,319 are not run), then runs R1 to R5
in this order. Remaining phase A work at the time of writing: 362 of 6,500 records.

## 5. GPU-hour estimates and queue order
Measured on the six-slot server (`scripts/rev3_estimate.py`, per-call latency fitted on
par6 records: 4k alias 0.656 + 0.0182 out + 0.00079 in seconds; 8k alias 2.075 + 0.0201
out + 0.00084 in; 6 concurrent streams; x1.15 for imbalance and retries). Measured
throughput at 6 streams, GSM8K cap 640: SA 3,986, SC k=8 499, DEBATE 535, GE 859, HIER
1,374 records per hour; MATH L1-3 cap 1,024: SC k=12 218, DEBATE 351, GE 608, HIER 755.
The fitted model over-predicts MATH cap 1,024 latencies by 15 to 28% (fitted on GSM8K
only, checked out of sample), so the estimates are conservative.

| Order | Run | Records | GPU hours | Why this position |
|-|-|-|-|-|
| 1 | R1 GSM8K HIERFIX | 1,500 | 1.3 | headline table, figure 1, abstract |
| 2 | R2 MMLU HIERFIX | 160 | 0.2 | headline table |
| 3 | R3 HumanEval | 1,476 | 2.8 | abstract code claim, extraction question |
| 4 | R4 MMLU-Pro eng. HIERFIX | 300 | 0.7 | engineering section |
| 5 | R5 MATH L1-3 cap 2,048 | 3,094 | 11.9 | difficulty section (SC k=12 5.3, DEBATE 3.1, GE 1.8, HIERFIX 1.6, SA 0.2) |
| | Total | 6,530 | 16.8 | |

The total is well under four days, so no run is cut.

## 6. Analysis plan
Test: `harness.stats.hier_paired_test` (items resampled, seeds within items, 10,000
replicates, seed 0), workflow minus comparator, two-sided 95% interval. Accuracy is
the per-item mean over seeds; SA (one greedy run) enters with its single value.
Comparator, fixed by cost alone as in the paper: for each coordination workflow, the
SC configuration with the largest k whose mean total tokens per item does not exceed
the workflow's (k in {2, 4, 8} on GSM8K, MMLU, MMLU-Pro and HumanEval; {2, 4, 8, 12} on
MATH; smaller k are the first k samples by the harness seed rule, or the separately
logged SC configurations of the MMLU and MMLU-Pro matrices). Costs are computed on the
same records; accuracy plays no part. SC k=8 is also reported for every workflow.

Confirmatory family, Holm over these five and nothing else:
- C1 GSM8K (500 x 3): HIERFIX minus its cost-fixed SC comparator.
- C2 to C4 HumanEval (164 x 2, v2 extractor): DEBATE, GE, HIERFIX minus their
  cost-fixed SC comparators (all from R3).
- C5 equal-cap difficulty interaction: D = (DEBATE minus SC k=8 on levels 4 to 5,
  phase H) minus (DEBATE minus SC k=8 on levels 1 to 3, R5), items resampled within each
  stratum independently, seeds within items, 10,000 replicates (as in
  `scripts/analyze_math_interaction.py`). Earlier work predicts D > 0; the test is
  two-sided.
Disclosure for C1: HIERFIX was already run on the first 100 of these items (extra
protocol, 4k vs 8k alias differs) and that result is known; C1 on items 101 to 500
alone is reported as a sensitivity check.

Exploratory (unadjusted 95% intervals, no claims of significance), reported in full:
- every other workflow minus comparator cell of the new headline matrix (Section 7),
  including MMLU and MMLU-Pro HIERFIX, and each workflow minus SA;
- HIERFIX minus HIER per task (HIER from the comparator runs);
- HumanEval with the v1 extractor on R3, the v1 minus v2 accuracy per configuration,
  and the earlier run (`results/humaneval_matrix_pc.jsonl`, v1) versus R3 (v1) per configuration;
- D for GE and HIERFIX; per workflow, R5 minus phase B (cap 2,048 minus cap 1,024 on
  the same items and seeds);
- the gain bound of each confirmatory test (90% interval against a 5-point margin);
- the frontier-share statistic (share of resamples on the Pareto frontier).
Verdict labels (R, C, N) in tables follow the paper's existing rule unchanged.

Descriptive checks reported with the results: HIERFIX aggregator outputs that contain
the template line with the placeholder letter, and no-answer rates, per task. HumanEval
per extractor: share of final-round outputs with no parseable program (SC samples,
debate round-1 agents, GE and HIERFIX last calls; the earlier figures were 0.4% and
25.6%), number of outputs v2 parses that v1 does not, number whose program changes,
and share of final calls at the cap.

Interim looks: record counts and failure rates only. No accuracy of any rev3
configuration is computed before its run is complete; the analysis script is
committed before that. If a run cannot finish, it is analysed on the largest complete
chunk of items and labelled incomplete.

## 7. What replaces what in the paper
- Headline matrix (Table 1, frontier figure): GSM8K 500 x 3 (phase A + R1), MATH
  levels 1 to 3 238 x 3 at cap 1,024 (phase B), MMLU 80 x 2 (matrix + R2), MMLU-Pro
  engineering 150 x 2 (matrix + R4), HumanEval 164 x 2 (R3, v2). MATH levels 4 to 5
  (262 x 3, phase H) stays in its own section. The July matrix (GSM8K n=80, MATH n=60,
  MMLU n=80) moves to the supplement as the original exploratory run.
- HIER: HIERFIX becomes the HIER of every main-text table and figure. The original HIER
  stays reported in one supplement table (all tasks, with its template-echo and
  no-answer rates); the HIER disclaimer is removed from the main-text captions. On MATH
  and HumanEval the two are the same prompt, and the text says so.
- HumanEval: R3 with v2 replaces `results/humaneval_matrix_pc.jsonl` in the main text;
  v1 numbers on R3 sit beside them; the old run, its energy figures and its audit move
  to the supplement. R3 measures no energy (six streams share the GPU).
  Abstract rule: coordination on code is called dominated only for workflows whose
  confirmatory test (C2 to C4, Holm) excludes zero; if none does, "robustly dominated
  on code" is removed; if v1 and v2 give different verdicts, both are reported.
- Difficulty: C5 (equal caps) becomes the primary interaction; the cap-confounded
  interaction of HARD_MATH_PROTOCOL moves to the supplement with its confound stated.
- One primary family: the five tests of Section 6. All other intervals are labelled
  exploratory and unadjusted.

## 8. Not covered
One model (Qwen2.5-3B-Instruct, 4-bit), the paper's workflow implementations, no new
seeds beyond those of each task's existing runs. Any wording change to HIERFIX or to
the extractor after a rev3 result is seen is a new workflow with a new protocol.
