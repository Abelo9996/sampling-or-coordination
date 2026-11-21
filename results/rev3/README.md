# results/rev3: rev3 reruns R1 to R5

Records of the reruns fixed in `protocols/REV3_RERUN_PROTOCOL.md` (R1 GSM8K HIERFIX, R2 MMLU
HIERFIX, R3 HumanEval with raw text and both extractors, R4 MMLU-Pro engineering HIERFIX,
R5 MATH levels 1 to 3 at the 2,048 cap). File names: `<run>_qwen3b_s<seed>_sh<i>of<n>.jsonl`
(one file per seed and shard; concatenate the shards of a seed). Analysis:
`scripts/analyze_rev3_reruns.py --r5`. See the README, section "rev3 reruns".
