# Analysis protocol: Du et al. debate and repaired HIER aggregator, Qwen2.5-3B

Written 16 Nov 2026 and committed before any of the runs below exists. The only
outputs of the new code so far are a smoke test on the Mac (3 GSM8K items, 2 MATH
level 4 to 5 items, seed 1, `results/smoke_extra/`), run to check that the code
executes; it is not part of any analysis. No HIERFIX wording was changed after it.

## Workflows (code: `scripts/extra_workflows.py`, driver `scripts/run_extra_workflows.py`)
- `DU_DEBATE_N3_R2`: the debate of Du et al. (2023) with their released GSM8K
  prompts, per-agent chat histories, 3 agents, rounds = 2 (6 calls), their parser
  and vote (`third_party/du2023_debate/`). Temperature 0.7 and max tokens = cap
  (their script sets neither). On MATH the prompt and parser are a marked adaptation.
- `HIERFIX_M3`: harness HIER M=3 with only the aggregator reminder changed, from
  `End with exactly one line: 'The answer is N.'` to `End with exactly one line:
  'The answer is N.' with N replaced by the final number.` The wording was written
  after seeing the aggregator echo the template line and was not tuned on accuracy.

## Runs (RTX 4060, Ollama, `qwen2.5:3b-instruct-ctx8k`, same weights as the 4k alias)
1. GSM8K, first 100 items in harness order, seeds 1 to 3, cap 640: DU_DEBATE and
   HIERFIX. Existing comparators: `results/power/gsm8k_qwen3b_s{1,2,3}.jsonl`
   (SC_k8, DEBATE_N3_R1, HIER_M3, seeds 1 to 3; SA greedy, seed 1), restricted to
   the same 100 items. Those were made with the 4k-context alias on a one-slot
   server; this is a stated difference.
2. MATH-500 levels 4 to 5, all 262 items, seeds 1 to 3, cap 2,048: DU_DEBATE only.
   Comparators: SC k=8 (first 8 samples of SC k=12) and DEBATE_N3_R1 from
   `results/power/math500hard_qwen3b_s*.jsonl`, under that run's own protocol.

## Scoring
- Primary for DU_DEBATE: their parse and their vote over the final-round agents,
  scored by the harness scorer (a `None` vote is wrong, unlike their eval script).
- Primary for HIERFIX: the unchanged harness extractor and scorer.
- Secondary for DU_DEBATE: harness extractor on each final-round agent, harness
  majority. Both numbers are always reported side by side.

## Tests
- `harness.stats.hier_paired_test` (items resampled, seeds within items, 10,000
  replicates, seed 0), workflow minus comparator, two-sided 95% interval.
- Comparator fixed in advance: SC k=8, chosen without looking at any result.
- Primary family, Holm over three: GSM8K DU_DEBATE minus SC_k8, GSM8K HIERFIX
  minus SC_k8, MATH 4 to 5 DU_DEBATE minus SC k=8.
- Gain bound: 90% interval of the same bootstrap difference against a 5-point
  margin (a gain of 5 points or more is excluded if the upper end is below +5),
  Holm over the same three.
- Secondary, Holm over five, reported with the same test: GSM8K DU_DEBATE minus
  DEBATE_N3_R1, HIERFIX minus HIER_M3, DU_DEBATE minus SA, HIERFIX minus SA; MATH
  DU_DEBATE minus DEBATE_N3_R1.
- Costs: mean input, output and total tokens per item next to every accuracy.

## Checks reported with the results
- HIERFIX and HIER_M3: share of aggregator calls with no number on the answer line
  (the literal `The answer is N.`) and accuracy among answered records.
- DU_DEBATE: share of `None` votes; share of calls at the cap; calls with
  `in_tokens + out_tokens >= 8192` (possible context truncation).

## Not covered
- One model, one size. GSM8K at 100 items. Any wording change after a result is
  seen is a new workflow with a new protocol.
