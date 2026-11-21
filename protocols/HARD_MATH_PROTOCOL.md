# Analysis protocol: MATH levels 4 to 5 (hard-regime run), Qwen2.5-3B

Written 16 Nov 2026, about 12:50 UTC, and committed before any accuracy of any
self-consistency or coordination configuration in this run was computed or viewed.

## What was known when this was written
- A pilot on 30 level 4 to 5 items: single greedy call 12 of 30 correct, and the
  output-length distribution (30% of sampled calls exceed 1,024 tokens), which set the
  2,048-token cap.
- Record counts of the running campaign (no accuracies): SA 50, SC k=12 152,
  debate 150, GE 135, HIER 83 records at the time of writing.
- On GSM8K (100 items, 3 seeds, same machine) SC k=8 was more accurate than debate,
  GE and HIER. Earlier work (Yang et al. 2025) reports that debate helps smaller
  models more on harder problems; that predicts debate minus SC is larger here than
  on levels 1 to 3.

## Data
- `results/power/math500hard_qwen3b_s{1,2,3}*.jsonl` (all shard files), MATH-500
  levels 4 and 5, all 262 items, seeds 1 to 3, cap 2,048, model alias with an
  8,192-token context (same weights), RTX 4060, Ollama 0.40.1. Records made after
  10:26 UTC on 16 Nov used a server with six parallel slots (host label
  `rtx4060-pc-par6`); earlier ones a one-slot server.
- Configurations: SA (greedy, one run), SC k=12 (SC k=2, 4, 8 are the first k
  samples, by the harness seed rule), debate N=3 R=1, GE R=2, HIER M=3.
- Levels 1 to 3 comparison data: campaign phase B (238 items, 3 seeds, cap 1,024,
  SC k=12), same machine, run after this phase.

## Primary analysis (at 262 items x 3 seeds, harness scoring)
1. Comparator, fixed by cost only: for each coordination workflow, the SC
   configuration with the largest k in {2, 4, 8, 12} whose mean total tokens per item
   does not exceed the workflow's mean total tokens. Accuracy plays no part in the
   choice. SC k=8 and SC k=12 are also reported for every workflow.
2. Test: `harness.stats.hier_paired_test` (seed-aware hierarchical paired bootstrap,
   10,000 replicates, seed 0), workflow minus comparator, two-sided 95% interval,
   Holm correction over the three workflows.
3. Gain bound and equivalence: 90% interval of the same difference against a 5-point
   margin, Holm over the three workflows. The margin is the one used in the paper's
   Table 13 (itself chosen after seeing the easier-task data).
4. Each workflow minus the single call, same test, Holm over three.
5. The same comparisons with output tokens as the cost that picks the comparator.

## Interaction with difficulty
- Primary: D = (debate minus SC k=8 on levels 4 to 5) minus (debate minus SC k=8 on
  levels 1 to 3, phase B). Items are resampled within each stratum independently,
  seeds within items; 10,000 replicates; two-sided 95% interval. The prediction from
  earlier work is D > 0. The two strata differ in token cap (2,048 vs 1,024); this is
  a stated confound of the primary interaction.
- Secondary, same cap: (debate minus SC k=8 on level 5) minus (the same on level 4)
  within this run.
- The same two quantities for GE and HIER are reported without a prediction.

## Extraction
- Primary numbers use the unchanged harness extractor and scorer.
- Audit: the flag rule of the GSM8K audit, adapted to `\boxed{}` answers (no boxed
  answer, call reached the cap, or the last boxed answer differs from a stated final
  answer). Flagged answer-bearing calls are read one by one by an AI assistant (not a
  human) that is shown only the call text: the workflow label, the harness reading
  and the gold answer are hidden from it. Its reading is then scored against gold by
  the unchanged scorer. Differences rescored with that reading are reported next to the primary
  ones. No-answer rates per workflow are reported, and accuracy among answered
  records for HIER.

## Interim looks
- A descriptive look (accuracies and costs, no tests, no claims) is allowed when the
  first 50 items are complete for all configurations. Tests are run once, on the full
  262 items. If the run cannot finish, the analysis is run on the largest complete
  chunk of items and labelled as incomplete.

## Not covered
- One model size, one model family, the paper's own workflow implementations. Any
  further runs (other sizes, a published debate implementation, a repaired HIER
  aggregator prompt) get their own protocol before their results are viewed.

## Amendment 1 (16 Nov 2026, about 13:00 UTC, still before any accuracy was viewed)
The extraction paragraph first said flagged calls are "read by hand". The reader is
an AI assistant, as in the GSM8K audit, so the wording was corrected and the blinding
made explicit. Nothing else changed.
