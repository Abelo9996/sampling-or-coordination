Hard-MATH answer-extraction audit (protocols/HARD_MATH_PROTOCOL.md, "Extraction" and Amendment 1). Script: hardmath_extraction_audit.py (build, freeze, candidates, analyze).

Input: results/power/math500hard_qwen3b_s*.jsonl, all 9 files including shard files, sorted path order, deduplicated by (config, item_id, seed), first occurrence wins. MATH-500 levels 4-5, 262 items, seeds 1-3, cap 2,048, Qwen2.5-3B. 3,406 records: SA 262 (seed 1), SC_k12, DEBATE_N3_R1, GE_R2, HIER_M3 786 each. All records present.
Replication: harness math_extract reproduces the logged extraction on every call (0 mismatches) and the logged pred on every record (0 mismatches).

Deciding calls (from harness/topologies.py): SA the one call; SC all 12 samples (k = 2, 4, 8, 12 use the first k, majority vote); DEBATE final-round calls idx 3-5 (majority vote); GE last revision idx 4; HIER aggregator idx 4. 13,624 deciding calls.
Flag rule, on call text and hit_cap only: (a) no complete \boxed{}, (b) call reached the cap, (c) last \boxed differs (math_score, either direction) from an earlier \boxed or from an "answer is / answer: / answer = / Final Answer" statement.
Flagged: SA 10, SC (12 samples) 207, DEBATE 200, GE 28, HIER 27; total 472. All 472 were read (no sampling).

Blinding: flagged calls written to hardmath_extraction_cases.json with shuffled opaque ids and call text only; record, workflow and flag reasons kept in hardmath_extraction_key.json.
Reader: an AI assistant (Claude), run as eight parallel reader instances, each given one contiguous slice of the cases file (case id and text only) and told to open no other file. Blind to workflow, item, gold and harness reading at reading time. No human check.
Pre-freeze checks from text only: 12 text-consistency cases re-read, no change; 25 random readings re-read (random.Random(7)), 25/25 agree. No clerical fixes.
Readings frozen before the key or any gold was opened: hardmath_extraction_readings.json, sha256 0d7f588f6da966ba024d988c6a7cecf3cf77c46f3e5fdf4b0b0e6bdcd8913fce (05:39 UTC 17 Nov 2026). Status counts: 262 answer, 204 truncated, 6 none; 6 answers cut off after an explicit final-answer statement.

Rescoring: flagged calls take the reading (none or truncated -> None), unflagged calls keep the harness reading; topologies._majority for SC and DEBATE; math_score against gold. Accuracy = mean over 262 items of the per-item mean over seeds (as in scripts/analyze_hard_math.py).

Per workflow (accuracy %, logged -> rescored; flips wrong->right / right->wrong; deciding calls; flagged; reading differs from harness; share at cap; share with no final answer stated):
SA      45.42 -> 45.42; 0/0;   262;  10; 0; 3.82%; 3.82%
SC k=2  46.06 -> 46.06; 0/0;  1572;  38; 0; 1.65%; 1.65%
SC k=4  50.38 -> 50.38; 0/0;  3144;  77; 2; 1.81%; 1.78%
SC k=8  52.80 -> 52.80; 0/0;  6288; 138; 5; 1.53%; 1.54%
SC k=12 54.33 -> 54.33; 0/0;  9432; 207; 8; 1.53%; 1.55%
DEBATE  44.40 -> 44.40; 0/0;  2358; 200; 4; 1.02%; 1.02%
GE      45.42 -> 45.42; 0/0;   786;  28; 1; 1.53%; 1.65%
HIER    42.11 -> 42.11; 0/0;   786;  27; 2; 1.78%; 2.16%
Reading-vs-harness differences (SC 12 samples, DEBATE, GE, HIER): extractor read a non-final or empty box 5, extractor None with an answer stated 2, no single answer stated but extractor read one 5, truncated but extractor read a box fragment or a quoted answer 3. Formatting class 0. None changes a record's correctness.

Differences in points, logged / rescored:
DEBATE - SC k=8  -8.40 / -8.40   DEBATE - SC k=12  -9.92 / -9.92   DEBATE - SA -1.02 / -1.02
GE     - SC k=8  -7.38 / -7.38   GE     - SC k=12  -8.91 / -8.91   GE     - SA  0.00 /  0.00
HIER   - SC k=8 -10.69 / -10.69  HIER   - SC k=12 -12.21 / -12.21  HIER   - SA -3.31 / -3.31
Unchanged when the 6 cut-off answers are counted as truncated.

No-answer rate (record pred None), logged / rescored: SA 3.82% / 3.82%, SC k=2 0.25% / 0.25%, SC k=4, 8, 12 0% / 0%, DEBATE 0.38% / 0.38%, GE 1.53% / 1.65%, HIER 2.04% / 2.16%.
HIER accuracy among answered records: logged 42.99% (770 records), rescored 43.04% (769 records).

Scorer problems (judged after the freeze, against gold; listed in SCORER_EQUAL_MARKED_WRONG in the script):
Stated answer equal to gold but marked wrong, 17 (item, answer) forms over 13 items: 'x=5' vs '5'; set order '-2, 1'; unreduced or decimal forms of a fraction (2187/5625, 2240/78125, 0.028672, 5.5, \frac{-35}{9}); -\mathbf{I} vs the 2x2 matrix; pmatrix with \frac or decimals vs '1/5'; a unit in \text{} (dollars, inches per second, meters, inches/second, gallons).
Calls marked wrong although equal: SA 3, SC k=8 54 (k=12 79), DEBATE 24, GE 10, HIER 11. Records (logged pred): SA 3, SC k=2 9, k=4 9, k=8 6, k=12 7, DEBATE 8, GE 10, HIER 11.
Marked right although unequal: 0 (no tolerance case found). Not counted: plane equations not in the required form, open/closed interval endpoints, answers with extra values.
With those cases scored as correct: SA 46.56, SC k=8 53.56, SC k=12 55.22, DEBATE 45.42, GE 46.69, HIER 43.51. Differences: DEBATE - SC k=8 -8.14, GE - SC k=8 -6.87, HIER - SC k=8 -10.05; minus SC k=12 -9.80 / -8.52 / -11.70; minus SA -1.15 / 0.13 / -3.05. The vote itself is not re-run under the corrected scorer (equal forms with different strings still split SC and DEBATE votes).

Files: hardmath_extraction_cases.json, hardmath_extraction_key.json, hardmath_extraction_readings.json, hardmath_extraction_summary.json (all numbers, disagreement, flip and scorer case lists).
