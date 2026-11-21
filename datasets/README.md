---
pretty_name: "Sampling or Coordination? Per-record logs"
license: cc-by-4.0
language:
  - en
task_categories:
  - question-answering
  - text-generation
tags:
  - self-consistency
  - multi-agent-debate
  - test-time-compute
  - llm-evaluation
  - energy
size_categories:
  - 10K<n<100K
source_datasets:
  - openai/gsm8k
  - hendrycks/competition_math
  - HuggingFaceH4/MATH-500
  - cais/mmlu
  - TIGER-Lab/MMLU-Pro
  - openai/openai_humaneval
configs:
  - config_name: gsm8k_500_headline
    data_files: "results/power/gsm8k_qwen3b_s?_sh?of2.jsonl"
  - config_name: gsm8k_rerun_100_raw
    data_files: "results/power/gsm8k_qwen3b_s?.jsonl"
  - config_name: math_l1_3_raw
    data_files: "results/power/math500_qwen3b_s*_sh*of2.jsonl"
  - config_name: math_l4_5_raw
    data_files: "results/power/math500hard_qwen3b_s*.jsonl"
  - config_name: extra_workflows_raw
    data_files: "results/power/extra_*_qwen3b_s*.jsonl"
  - config_name: rev3_r1_gsm8k_hierfix
    data_files: "results/rev3/gsm8k_qwen3b_s*_sh*of2.jsonl"
  - config_name: rev3_r2_mmlu_hierfix
    data_files: "results/rev3/mmlu_qwen3b_s*_sh*of3.jsonl"
  - config_name: rev3_r3_humaneval_raw
    data_files: "results/rev3/humaneval_qwen3b_s*_sh*of3.jsonl"
  - config_name: rev3_r4_mmlu_pro_eng_hierfix
    data_files: "results/rev3/mmlu_pro_eng_qwen3b_s*_sh*of3.jsonl"
  - config_name: rev3_r5_math_l1_3_cap2048
    data_files: "results/rev3/math500_cap2048_qwen3b_s*_sh*of2.jsonl"
  - config_name: humaneval_energy
    data_files: "results/humaneval_matrix_pc.jsonl"
  - config_name: july_matrices
    data_files:
      - "results/gsm8k_matrix.jsonl"
      - "results/mmlu_matrix.jsonl"
      - "results/math500_trunc.jsonl"
      - "results/math500_matrix.jsonl"
---

# Sampling or Coordination? Per-record logs

<!-- DRAFT dataset card. Not uploaded. File paths are relative to the code repository root;
     when uploading, copy results/ next to this card (or adjust the data_files globs).
     The schemas of the records differ between runs, so each config is loaded separately. -->

Per-record logs behind every table and figure of the paper *Sampling or Coordination?
Self-Consistency vs. Multi-Agent Debate at Matched Token, Price and Energy Cost in a Small
LLM* (Abel Yagubyan, Independent Researcher, 2026). Code: https://github.com/Abelo9996/sampling-or-coordination

## What a record is

One line = one query answered by one workflow configuration with one seed:
`item_id` (benchmark item, e.g. `gsm8k-12`, `math-371`, `mmlu-40`, `mmlupro-<question_id>`,
`humaneval-17`), `task`, `config` (`SA`, `SC_k8`, `DEBATE_N3_R1`, `GE_R2`, `HIER_M3`,
`HIERFIX_M3`, `DU_DEBATE_N3_R2`, ...), `seed`, `model`, `pred`, `gold`, `correct`,
`in_tokens`, `out_tokens`, `total_tokens`, `n_calls`, `latency_s`, and, in the November 2026
runs (including the rev3 reruns in `results/rev3/`), `cap`, `host` (a hardware label: `mac-m4`, `rtx4060-pc`, `rtx4060-pc-par6`) and
`calls` (one entry per model call with the raw generated `text`, token counts,
temperature, seed, `hit_cap` and the harness `extracted` answer). The HumanEval run adds
per-call GPU energy (`energy_j`) and `energy_j_net` (idle-subtracted). The July 2026
matrices log per-record totals only, without text.

## Models

Qwen2.5-3B-Instruct (4-bit, Ollama `qwen2.5:3b-instruct`, digest `357c53fb659c`) for
all main results; probes with Qwen2.5-7B-Instruct, Llama-3.2-3B, Gemma-2-2B (Ollama) and
Llama-3.3-70B-Instruct (hosted API).

## Licenses

- The logs: CC BY 4.0.
- Model outputs also fall under the terms of the model that produced them (Qwen
  Research License for Qwen2.5-3B, Apache-2.0 for Qwen2.5-7B, Llama 3.2 and Llama 3.3
  Community Licenses, Gemma Terms of Use).
- Benchmark content: some records repeat the benchmark question (`question` fields of
  the trace files, debate prompts in `du_contexts`) and every record carries the gold
  answer. GSM8K, MATH, MMLU and HumanEval are MIT licensed; MMLU-Pro is Apache-2.0. All
  permit redistribution with attribution.
- Contamination notice: these files contain benchmark items and answers. Do not train
  on them if you evaluate on these benchmarks.

## Citation

See `CITATION.cff` in the code repository.
