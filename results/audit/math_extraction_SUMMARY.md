MATH answer-extraction audit, results/math500_qwen3b_coord_trace.jsonl (60 MATH L1-3 items x SC_k8, DEBATE_N3_R1, GE_R2, HIER_M3; seed 1; cap 1,024; Qwen2.5-3B). Script: math_extraction_audit.py (build, auto, analyze).
Deciding calls per harness/topologies.py: SC all 8 samples (majority), DEBATE final-round calls idx 3-5 (majority), GE last revision idx 4, HIER aggregator idx 4. Total 780 calls; harness extractor replicates the logged extraction on all calls and the logged pred on all 240 records.
Reader: an AI assistant (Claude), reading math_extraction_cases.json (shuffled opaque ids, call text only), blind to workflow, item, gold and harness reading at reading time. No human check.
690 calls auto-resolved from text alone (exactly one \boxed, under 300 chars after it with no 'answer', '=' or further \boxed); 90 read in full by the reader; 30 random auto cases re-read, 30/30 agree.
Readings frozen to math_extraction_readings.json before the join (no clerical fixes). Status counts: 770 answer, 9 truncated, 1 none; 4 answers flagged cut off after an explicit final-answer statement.
Equivalence of reading and harness extraction: harness.tasks.math_score in either direction; both None counts as agreement. Rescoring: _majority over readings (SC, DEBATE), last-call reading (GE, HIER), math_score against gold.
Misread calls: SC 2/480 (0.42%), DEBATE 0/180, GE 0/60, HIER 1/60 (1.67%). SC samples 2/480 (0.42%) vs coordination final calls 1/300 (0.33%).
Misread classes: extractor None though answer stated 1 (SC, cap hit); extractor non-final token 1 (HIER, read ':' after the \boxed was cut off at the cap); no single answer stated but extractor read one 1 (SC, '\boxed{36} or \boxed{49}'); formatting 0.
Cap hits among deciding calls: SC 6/480, DEBATE 4/180, GE 2/60, HIER 1/60. No final answer (truncated or none): SC 6/480, DEBATE 2/180, GE 2/60, HIER 0/60.
Accuracy logged -> rescored: SC 53/60 -> 53/60 (88.3%), DEBATE 49/60 -> 49/60 (81.7%), GE 46/60 -> 46/60 (76.7%), HIER 47/60 -> 47/60 (78.3%). Flips 0 in each direction for every workflow.
Gaps SC minus DEBATE/GE/HIER: 6.7/11.7/10.0 points logged and rescored; unchanged when the 4 cut-off answers are treated as truncated.
Scorer problems (judged after freeze, against gold): gold '\text{(C)}' (math-227) and '\text{(E)}' (math-255) vs stated 'C'/'E'/'\text{C}'/'\text{E}' marked wrong although equal: calls SC 16, DEBATE 6, GE 2, HIER 2; records 2 per workflow.
Scorer marks unequal as right: 1 SC call (math-176, '0.0000671875' vs gold '.0000672', inside the 1e-4 absolute tolerance); 0 records.
With those scorer cases corrected: SC 55/60 (91.7%), DEBATE 51/60 (85.0%), GE 48/60 (80.0%), HIER 49/60 (81.7%); gaps unchanged (6.7/11.7/10.0).
Files: math_extraction_cases.json, math_extraction_key.json, math_extraction_readings.json, math_extraction_summary.json (all numbers, disagreement and scorer case lists).
