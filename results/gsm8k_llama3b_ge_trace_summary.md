# GE_R2 raw-text trace, llama3.2:3b, GSM8K (50 items, seed 1, cap 640)

## Commands

```
cd <repository root>
for m in qwen2.5:3b-instruct llama3.2:3b gemma2:2b; do ollama stop "$m"; done; sleep 3
env -u AGENTS_BACKEND -u OLLAMA_URL -u AGENTS_NUM_PREDICT caffeinate -i \
  /usr/bin/python3 -u scripts/aaibd_ge_trace.py --n 50 --seed 1 --cap 640 \
  2>&1 | tee -a results/logs/aaibd_ge_trace.log
```

Ollama 0.30.10 at localhost:11434 (already running, not restarted). The script
calls the unchanged `harness.topologies.ge` (with its optional `trace` dict)
and `harness.topologies.sa`, checks the item ids against the SA records of
`results/gsm8k_llama3b.jsonl`, and asserts `client.DEFAULT_NUM_PREDICT == 640`.
GE_R2 ran over all 50 items first, then SA over all 50 items. The run was not
interrupted; no resume was needed.

- Start: 2026-11-15T04:59:37Z. GE phase: 68.4 min. SA phase: 5.0 min.
- End: 2026-11-15T06:13:02Z (total 73.4 min), exit 0, 50/50 GE and 50/50 SA records.
- Outputs: `results/gsm8k_llama3b_ge_trace.jsonl`, `results/gsm8k_llama3b_sa_trace.jsonl`,
  log `results/logs/aaibd_ge_trace.log`.

## Sanity check against the logged runs

| comparison | pred match | correct match | out_tokens match | accuracy (trace / logged) |
|---|---|---|---|---|
| GE_R2 vs `gsm8k_llama3b_seed1_rerun.jsonl` (GE_R2, seed 1) | 50/50 | 50/50 | 50/50 | 19/50 / 19/50 |
| SA vs `gsm8k_llama3b_driftcheck.jsonl` (SA) | 50/50 | 50/50 | 50/50 | 42/50 / 42/50 |

The trace run reproduces both logged runs exactly on these fields.
GE_R2 accuracy in this run: 19/50 = 38%. SA: 42/50 = 84%.

## Answer format

The prompt (`harness/tasks.py`, `gsm8k_prompt`) asks for a final line written
exactly as `'The answer is N.' where N is the final number`; the GE revision
prompt appends `End with exactly one line: 'The answer is N.'`. The extractor
takes the first `answer is <number>` match and otherwise falls back to the last
number anywhere in the text.

| output | "The answer is <number>." | "The answer is $<number>." | literal "The answer is N." | no answer line |
|---|---|---|---|---|
| SA (greedy) | 4 | 0 | 43 | 3 |
| GE draft | 15 | 1 | 25 | 9 |
| GE revision r1 | 0 | 0 | 43 | 7 |
| GE revision r2 (final) | 1 | 0 | 45 | 4 |

So the final GE text follows the requested format with an actual number in
1/50 items (gsm8k-950). SA also writes the literal "N" in 43/50 items and still
scores 42/50, so in both configurations pred almost always comes from the
last-number fallback.

## Counts (from reading the raw texts of all 50 items)

| quantity | count |
|---|---|
| drafts correct | 35/50 |
| finals correct (GE_R2 pred) | 19/50 |
| revision r1 answer correct | 22/50 |
| right-to-wrong flips, draft to final | 17 |
| wrong-to-right flips, draft to final | 1 (gsm8k-554) |
| final text follows requested format with a number | 1/50 |
| final pred equals the value the final text concludes | 42/50 |
| final pred differs from a clearly concluded value | 1/50 (gsm8k-23) |
| final text concludes no number (category e) | 7/50 |
| final output hit the 640-token cap | 2 (gsm8k-175, gsm8k-465) |
| other cap hits: draft / critique r1 / revision r1 / critique r2 | 1 / 3 / 4 / 1 |
| SA correct and GE_R2 wrong / SA wrong and GE_R2 correct | 24 / 1 |

Categories for the 31 wrong finals:

| category | count | item_ids |
|---|---|---|
| (a) extraction error, text concludes the right answer | 1 | gsm8k-23 |
| (b) extraction error, text concludes a wrong answer different from pred | 0 | none |
| (c) correct draft changed to wrong after a critique that flagged an error | 6 | gsm8k-474, gsm8k-1020, gsm8k-265, gsm8k-934, gsm8k-1226, gsm8k-1236 |
| (d) draft wrong, final wrong | 11 | gsm8k-991, gsm8k-1070, gsm8k-590, gsm8k-446, gsm8k-1161, gsm8k-1110, gsm8k-789, gsm8k-1137, gsm8k-952, gsm8k-399, gsm8k-1048 |
| (e) final text has no clear final answer | 7 | gsm8k-1062, gsm8k-799, gsm8k-175, gsm8k-204, gsm8k-197, gsm8k-259, gsm8k-465 |
| (f) other: correct draft changed to wrong after a critique whose verdict was "no errors" / "correct" | 6 | gsm8k-1101, gsm8k-1257, gsm8k-625, gsm8k-620, gsm8k-680, gsm8k-743 |

Notes on the categories:

- c and f are split by the critique that preceded the round in which the
  answer first became wrong. The harness verdict heuristic
  (`critic_found_errors`, first 200 characters) disagrees with my reading for
  gsm8k-1101, gsm8k-1257 and gsm8k-620 (it reports "found errors" for texts
  such as "Yes, the proposed solution appears to be correct"); by the heuristic,
  c would be 9 and f 3.
- Of the 7 category-e items, the draft was correct in 4 (gsm8k-799,
  gsm8k-204, gsm8k-197, gsm8k-465) and wrong in 3 (gsm8k-1062, gsm8k-175,
  gsm8k-259). In all 7 the extractor returned an incidental number (a step
  numeral, "5:00", a quantity from the question). Two end truncated at the cap
  (gsm8k-175, gsm8k-465); gsm8k-259 ends one token short of the cap with an
  unevaluated sum; gsm8k-197 ends with the unevaluated expression
  "23786 - 17507" (its value, 6279, is also wrong); gsm8k-204 computes 98 and
  then retracts it ("However ... we also need to add these back in") and ends
  with an unevaluated sum.
- The 17 right-to-wrong flips are a (1) + c (6) + f (6) + e with correct draft (4).
  The 15 wrong drafts are d (11) + e with wrong draft (3) + the one
  wrong-to-right flip.
- Category assignments are one reader's judgement of the raw texts; per-item
  notes are not stored in the repo.

## Verbatim excerpts

Dominant category (d), draft already wrong and stayed wrong:

- gsm8k-1161 (gold 170, final 2390), final revision: "He sold 40 square
  meters for $50. So, he earned 40 x $50 = $2000." (The same step appears in
  the draft and in revision r1.)
- gsm8k-1048 (gold 7400, final 9239), final revision: "Step 2: Calculate the
  depreciation for the second year (2009). Depreciation = 21% of $15,800"
  (compounding depreciation in draft and both revisions).

Second-largest groups (c and f, a correct draft revised to wrong), for
comparison:

- gsm8k-265 (gold 9500, draft 9500, final 11750), critique r2: "the
  transfer fee is actually a percentage of the selling price plus brokerage
  fee." The following revision computes "Transfer fee = 12% of ($350,000 +
  $17,500)".

## What the data supports and does not support

Supported by the counts:

- The GE_R2 deficit on this item set is not an extraction artifact: 1 of the
  31 wrong finals is an extraction error on a clearly stated correct answer
  (gsm8k-23); 0 are extraction errors on a stated wrong answer.
- 17 items have a correct draft and a wrong final; 1 item has the reverse.
  Draft accuracy (35/50) and final accuracy (19/50) differ by 16 items.
- 11 of the 31 wrong finals come from drafts that were already wrong.
- The literal "The answer is N." output occurs in both SA (43/50) and GE finals
  (45/50), so it is not by itself specific to GE_R2.

Not supported or not determined:

- Whether a stricter extractor (format-only, no fallback) would change the
  ranking: it would score almost every SA and GE output as unparseable, and
  this was not computed as a separate score.
- Why the draft (35/50, temperature 0.7) is below greedy SA (42/50); this run
  has one seed and does not separate sampling temperature from other causes.
- Anything about other models, tasks, seeds or caps.
