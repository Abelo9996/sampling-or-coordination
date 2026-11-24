# Sampling or Coordination?

Code, prompts and per-record logs for the paper

> **Sampling or Coordination? Cost-Accuracy Frontiers for Homogeneous Small-Model LLM
> Agent Workflows.** Abel Yagubyan, Independent Researcher, 2026.
> arXiv identifier: to be added.

The paper compares, on one small open model (Qwen2.5-3B-Instruct, 4-bit, served locally by
Ollama), a single call (SA), self-consistency (SC, majority vote over k samples),
multi-agent debate, a generator-evaluator (GE) and a planner-executor (HIER), on GSM8K,
MATH-500 (levels 1 to 3 and 4 to 5), MMLU, MMLU-Pro engineering and HumanEval. Every
configuration is costed in total, output and price-weighted tokens, serial time and (on
HumanEval) measured GPU energy, and compared with a noise-aware Pareto verdict.

Every table, figure and macro of the paper regenerates from the logs in `results/`
without a model call. Re-running the experiments needs Ollama and the models below.

## Contents

| Path | What |
|---|---|
| `harness/` | Evaluation harness: Ollama client with exact token accounting (`client.py`), task loaders, prompts and scorers (`tasks.py`), the workflows (`topologies.py`: SA, SC, DEBATE, GE, HIER and variants), statistics (`stats.py`: seed-aware hierarchical bootstrap), HumanEval sandbox (`sandbox.py`), GPU energy attribution (`energy.py`), hosted-API client for the 70B probe (`openai_client.py`), OpenAI-compatible vLLM backend for the cross-model replication (`vllm_client.py`, `AGENTS_BACKEND=vllm`) |
| `configs/matrix.py` | The configuration matrix (which workflows and parameters are run) |
| `scripts/run_*.py`, `scripts/run_*.sh` | Run drivers (model calls); see "Re-running experiments" |
| `scripts/cloud/` | Cross-model replication on rented GPUs: vLLM driver, budget plan, Modal and Kaggle launchers, and `cross_model_analysis.py` (the paper's tests on every model under `results/cloud/`); see "Cross-model replication" |
| `scripts/within_run_check.py` | Within-run check of every comparison whose arms come from different runs, and the run-to-run drift of identical configurations |
| `scripts/extra_workflows.py`, `scripts/rev3_workflows.py` | Du et al. debate reimplementation and the repaired planner-executor (HIERFIX) |
| `scripts/analyze_*.py`, `scripts/emit_*.py`, `scripts/criterion_predictive*.py`, `scripts/aaibd_*.py`, `scripts/make_*.py` | Analysis and emitters (logs to JSON, LaTeX macros and table fragments, figures) |
| `paper/arxiv_long/scripts/` | Emitters for the audit, equivalence, hard-MATH, extra-workflow, context and review tables |
| `paper/arxiv_rev3/scripts/` | rev3 emitters and figure scripts (each figure script asserts its plotted numbers against the macros) |
| `paper/arxiv_rev3/*_macros.tex`, `paper/arxiv_rev3/tables/`, `paper/arxiv_rev3/figures/`, `paper/arxiv_rev3/data/` | The exact generated numbers, table fragments and figures used in the paper (reference outputs; the manuscript source itself is not included) |
| `paper/arxiv_long/*_macros.tex`, `paper/arxiv_long/tables/`, `paper/aaibd/revision_stats.json` | Intermediate generated outputs that the rev3 scripts read |
| `protocols/` | Analysis protocols fixed before the runs they govern (hard MATH, extra workflows, rev3 reruns) and the original preregistration |
| `soc/dominance.py` | The cost-matched dominance test and the break-even price ratio as two functions (numpy only); see "Reusable tools" |
| `soc/plotstyle.py` | The shared matplotlib style of every paper figure (fonts, sizes, workflow colors, error bars); a copy of `paper/arxiv_rev3/scripts/plotstyle.py` |
| `tests/` | Offline unit tests (no model calls) |
| `tasks_data/` | Benchmark items used (see `tasks_data/LICENSES.md`) |
| `results/` | Per-record logs (`*.jsonl`) and analysis outputs; see "Logs" |
| `results/audit/` | Answer-extraction audits: cases, keys, AI readings and summaries |
| `datasets/README.md` | Draft Hugging Face dataset card for the logs |
| `reproduce/` | `regenerate_all.sh` and `check_rev3.py` (regenerate everything and compare with the paper's numbers) |
| `third_party/du2023_debate/` | Pointer to the Du et al. (2023) debate code (not redistributed; `fetch_upstream.sh` fetches it at the pinned commit) |

## Setup

Analysis needs Python 3.9 or later with numpy and matplotlib only:

```bash
python3 -m venv .venv && .venv/bin/pip install "numpy>=1.26,<2.1" "matplotlib>=3.8"
```

The paper's outputs were produced with Python 3.9.6, numpy 2.0.2 and matplotlib 3.9.4 on
macOS. The scripts call `/usr/bin/python3` in a few shell wrappers; set `PY` (see
`reproduce/regenerate_all.sh`) to use another interpreter. The harness itself uses only
the standard library. Offline tests: `python3 -m unittest discover -s tests`.

## Reproducing the tables and figures

Everything (about 3 hours on an Apple M4, almost all of it bootstrap resampling in
`analyze_hard_math.py`, `analyze_rev3_reruns.py`, `emit_math_easy.py`, `emit_humaneval.py` and `emit_evidence.py`;
each stage can also be run on its own, since every intermediate output is included):

```bash
sh reproduce/regenerate_all.sh          # writes every macro, table fragment and figure
python3 reproduce/check_rev3.py         # compares them with the reference copies used in the paper
```

`regenerate_all.sh` runs four stages, each from the logs only:

1. Pre-specified analyses to JSON: `scripts/analyze_hard_math.py`,
   `scripts/analyze_math_interaction.py`, `scripts/analyze_extra_gsm8k.py`, and the
   extraction audits `results/audit/math_extraction_audit.py analyze`,
   `results/audit/hardmath_extraction_audit.py analyze` (results in `results/extra/` and
   `results/audit/`), and `scripts/analyze_rev3_reruns.py --r5` (every rev3 rerun test, the
   family of five tests C1 to C5 fixed before the reruns and its Holm adjustment; `results/extra/rev3_reruns_analysis.json`),
   and `scripts/analyze_rev3_costs.py` (modeled energy per query and the price-ratio crossover
   intervals, exploratory; `results/extra/rev3_costs_analysis.json`).
2. First-generation emitters: `scripts/emit_crosstask.py`, `emit_macros.py`,
   `emit_evidence.py`, `emit_humaneval.py`, `emit_mmlu_pro_eng.py`, `emit_scale70b.py`,
   `emit_throughput.py`, `criterion_predictive.py`, `aaibd_revision_stats.py`,
   `aaibd_emit_tables.py`, `aaibd_emit_rerun_macros.py`, `make_eaai_figures.py`.
3. `paper/arxiv_long/scripts/emit_*.py` (audit, context, equivalence, noise-aware
   verdicts, extra workflows, hard MATH, HumanEval audit, MATH audit, MATH interaction,
   review revision).
4. rev3: `paper/arxiv_rev3/scripts/emit_math_easy.py`, `emit_rev3_macros.py`,
   `check_gsm8k_rerun_subset.py`, `emit_hard_ci.py` (accuracy intervals of `tab:hard`),
   `emit_rev3_tables.py`, `emit_rev3_reruns.py` (rerun macros `rev3_reruns_macros.tex` and the
   `tables/rev3/rr_*.tex` fragments), `emit_token_multiple.py` (token multiples,
   `token_multiple_macros.tex`), `emit_table_bodies.py` (the tabular of each whole-table
   fragment, `tables/rev3/tabular_*.tex`, for the merged tables), `scripts/within_run_check.py`
   (within-run check, `within_run_macros.tex`, `tables/within_run_*.tex`),
   `scripts/cloud/cross_model_analysis.py` (cross-model replication, `cross_model_macros.tex`,
   `tables/cross_model_*.tex`), then the figures and the flow table (`make_fig_eaai_adapted.py`).

Stage outputs of steps 2 and 3 are written under `paper/` (to `paper/*.tex`,
`paper/eaai/`, `paper/aaibd/`, `paper/arxiv_long/`); `check_rev3.py` maps each reference
file in `paper/arxiv_rev3/` to the regenerated file it was copied from and reports any
difference (`EXACT`, `SAME-NUMBERS` when only caption or header text was edited for rev3,
`DIFF-UNUSED` for six stale router and p-value macros of an earlier draft that the
emitters still write but the paper does not use, `DIFF`). Files that stage 4 rewrites in
place are checked against `reproduce/rev3_reference.sha256`. Figure PDFs differ in bytes
(timestamps) but every figure script asserts its plotted numbers against the macros.
Expected `git status` after a full run: the figure PDFs, and
`paper/aaibd/revision_stats.json`, whose only changes are line-number pointers into the
manuscript (not included) and into `harness/tasks.py`.


### Where each table and figure comes from

| Paper item | File used by the paper | Produced by | Main logs |
|---|---|---|---|
| Table 1 `tab:crosstask` | `tables/rev3/rr_main_table.tex` | `emit_rev3_reruns.py` from `rev3_reruns_analysis.json` and `tables/math_easy.json` | `power/gsm8k_qwen3b_s*_sh*of2` (GSM8K 500 x 3), `rev3/*` (HIERFIX), `power/math500_qwen3b_s*`, `mmlu_matrix`, `mmlu_pro_eng_matrix`, `rev3/humaneval_*` |
| Table 2 `tab:domination` | `rr_domination_gsm.tex`, `rr_gsm200_domination.tex`, `rr_math_domination.tex` | `emit_rev3_reruns.py` | as Table 1, `gsm8k_scale200` |
| Table 3 `tab:robust` | `poster_robustness_table.tex` | `aaibd_revision_stats.py` then `aaibd_emit_tables.py` | July matrices |
| Table 4 `tab:flow` (initial against final answer; replaces the earlier flow figure) | `tables/flow_table.tex` (copied to `tables/rev3/`) | `make_fig_eaai_adapted.py` (`flow_table`), `emit_rev3_tables.py` | `mechanistic` |
| Table 5 `tab:costverdict` | `rr_cost_verdict.tex` | `analyze_rev3_reruns.py`, `emit_rev3_reruns.py` | as Table 1 |
| Table 6 `tab:outverdict` (a: output tokens; b: modeled energy) | `rr_output_verdict.tex`, `rr_energy_verdict.tex` | `emit_rev3_reruns.py`; modeled J from `scripts/analyze_rev3_costs.py` (HumanEval per-call energy fit; item-bootstrap intervals) | GSM8K 500 x 3, MMLU matrix + R2, MATH 238 x 3, MMLU-Pro eng. matrix + R4, R3 HumanEval, `gsm8k_scale200`, `humaneval_matrix_pc` (fit) |
| Price-ratio crossovers with 95% intervals (`\RrCross*`, `\RrCrossCI*`; `fig:crossover`) | `rev3_reruns_macros.tex`, `fig_price_crossover.pdf` | `scripts/analyze_rev3_costs.py` (seed-aware bootstrap of r*, 10,000 replicates), `emit_rev3_reruns.py`, `fig_price_crossover.py` | `power/gsm8k_qwen3b_s*_sh*of2`, `power/math500_qwen3b_s*` |
| `tab:confirm` (C1 to C5, Holm over five) | `rr_confirm_table.tex` | `analyze_rev3_reruns.py --r5`, `emit_rev3_reruns.py` | `power/gsm8k_qwen3b_s*_sh*of2`, `rev3/gsm8k_*`, `rev3/humaneval_*`, `power/math500hard_*`, `rev3/math500_cap2048_*` |
| `tab:hard` | `hard_math_table.tex` | `scripts/analyze_hard_math.py`, `emit_hard_math.py`; accuracy intervals `emit_hard_ci.py` (`tables/hard_math_ci.json`), inserted by `emit_rev3_tables.py` | `power/math500hard_qwen3b_s*`, `power/extra_math45_qwen3b_s*` |
| `tab:extra`, `tab:equiv` (upper block), `tab:extra-acc` (b), `tab:hierorig`, `fig:frontier-extra` | `rr_extra_full.tex`, `rr_equivalence.tex`, `rr_humaneval_table.tex`, `rr_hier_table.tex`, `fig_frontier_extra.pdf` | `emit_rev3_reruns.py`; `make_fig_frontier.py --extra` | rerun records and the original matrices |
| `tab:orig`, `tab:orig-dom` (a: total tokens; b: each cost model), `tab:equiv` (lower block) (original exploratory run, unrepaired HIER) | `poster_main_table.tex`, `poster_domination_table.tex` with `extra_domination_compact.tex`, `cost_verdict_table.tex`, `equivalence_orig_rows.tex` | `aaibd_emit_tables.py`, `emit_review_revision.py`, `emit_extra_noise_aware.py`, `emit_equivalence.py`; `emit_table_bodies.py` | July matrices, `mmlu_pro_eng_matrix`, `humaneval_matrix_pc` |
| `tab:overview` | inline text | | |
| `tab:cost` (a: cost models; b: cost tails and latency), `tab:throughput` | `poster_cost_table.tex`, `tabular_latency_table.tex`, `tabular_rr_throughput_table.tex` | `aaibd_emit_tables.py`, `emit_throughput.py`; `emit_table_bodies.py` | July matrices, RTX 4060 runs |
| `tab:predictive` (a: cells; b: other units), `tab:S-splits` (a: pilot splits; b: pooled), `fig:criterion` | `tabular_criterion_*_table.tex`, `criterion_units_table.tex`, `fig_criterion.pdf` | `scripts/criterion_predictive.py`; `make_fig_eaai_adapted.py`; `emit_table_bodies.py` | `*_traces_mac*`, `mechanistic`, `gsm8k_llama70b` |
| `tab:extra-acc` (a), `fig:proeng` | `tabular_mmlu_pro_eng_table.tex`, `fig_proeng_pareto.pdf` | `scripts/emit_mmlu_pro_eng.py`; `make_fig_eaai_adapted.py` | `mmlu_pro_eng_matrix` |
| `tab:code-orig` (a: pass@1 and energy; b: vote inputs; c: unparsed records), `fig:code`, `fig:energy` (first HumanEval run) | `tabular_rr_orig_humaneval_*table.tex`, `humaneval_audit_table.tex`, `fig_code_energy.pdf`, `fig_energy.pdf` | `scripts/emit_humaneval.py`, `emit_humaneval_audit.py`; `make_fig_eaai_adapted.py`; `fig_energy.py` | `humaneval_matrix_pc`, `humaneval_matrix_pc_idle`, `humaneval_secondary_vote` |
| `tab:rerun` (a: accuracy and cost; b: run-to-run change; c: extraction audit), `fig:drift` | `audit_*_table.tex`, `fig_drift.pdf` | `emit_audit.py`; `fig_drift.py` | `power/gsm8k_qwen3b_s{1,2,3}`, `power/gsm8k_cap2048_*`, `gsm8k_qwen3b_coord_trace`, `results/audit/gsm8k_*` |
| `tab:extrawf` | `extra_workflows_table.tex` | `analyze_extra_gsm8k.py`, `emit_extra_workflows.py` | `power/extra_gsm8k_qwen3b_s*` |
| `tab:probes` (a: model size; b: Llama-3.3-70B; c: model family; d: diversity) | `tabular_scale_table.tex`, macros in `scale70b_macros.tex` (whole-point intervals `\RthSevCI*` in `rev3_macros.tex`), `rerun_macros.tex`, `review_revision_macros.tex`, `tabular_diversity_table.tex` | `revision_stats.json`, `emit_scale70b.py`, `emit_rev3_macros.py`, `aaibd_emit_rerun_macros.py`, `emit_review_revision.py` | `gsm8k_7b`, `gsm8k_llama70b`, `gsm8k_llama3b*`, `gsm8k_gemma2b*`, `gsm8k_debate_*` |
| `tab:trunc` (a: 640 against 1,024; b: 1,024 against 2,048), `tab:positioning` | `tabular_trunc_table.tex`, macros in `rev3_reruns_macros.tex`, `tabular_positioning_table.tex` | `emit_truncation.py`, `emit_rev3_reruns.py`; positioning is a hand-written comparison table | `math500_matrix`, `math500_trunc`, `rev3/math500_cap2048_*` |
| Fig. 1 `fig:overview` | `fig_forest.pdf` | `make_fig_forest.py` | via macros and `tables/math_easy.json` |
| Fig. 2 `fig:workflows`, Fig. 3 `fig:frontier`, Fig. 4 `fig:gsm8k` | `fig_workflows.pdf`, `fig_frontier.pdf`, `fig_gsm8k_pareto.pdf` | `make_fig_workflows.py`, `make_fig_frontier.py`, `make_fig_gsm8k_pareto.py` | matrices, `power/math500_qwen3b_s*` |
| `fig:crossover`, `fig:hard` | `fig_price_crossover.pdf`, `fig_hard_frontier.pdf` | `fig_price_crossover.py`, `fig_hard_frontier.py` | via `rev3_macros.tex`, `power/math500hard_*` |
| `tab:withinrun`, `tab:wrsupp`, `tab:wrdrift` (within-run check of the cross-run comparisons; run-to-run drift) | `tables/within_run_main.tex`, `within_run_supp.tex`, `within_run_drift.tex`, macros `within_run_macros.tex` | `scripts/within_run_check.py` (`results/extra/within_run_check.json`) | `rev3/*`, `power/extra_*`, `power/gsm8k_qwen3b_*`, `gsm8k_scale200`, `rev3/humaneval_*`, `tables/audit.json` |
| `tab:xmodel`, `tab:xmodel-supp`, `tab:xmodel-supp-math` (cross-model replication) | `tables/cross_model_table.tex`, `cross_model_supp_gsm8k.tex`, `cross_model_supp_math500.tex`, macros `cross_model_macros.tex` | `scripts/cloud/cross_model_analysis.py` (`results/extra/cross_model_analysis.json`) | `cloud/qwen3-4b/*`, `cloud/qwen3-8b/*` (in `cloud/qwen3_raw_logs.tar.xz`), `cloud/phi4-mini/*`, `cloud/qwen25-3b/*` (in `cloud/nonthinking_raw_logs.tar.xz`) |
| All in-text numbers | `paper/arxiv_rev3/*_macros.tex` | the emitter named in each file's first line | |

Table and figure files are under `paper/arxiv_rev3/`; scripts named without a path are in
`scripts/`, `paper/arxiv_long/scripts/` or `paper/arxiv_rev3/scripts/`.

## Reusable tools

`soc/dominance.py` (numpy only) provides the paper's two procedures for any workflow and cost
unit. `cost_matched_dominance` returns R (robustly dominated: some cheaper reference
configuration has a 95% seed-aware hierarchical interval of its accuracy advantage above
zero), C (cheaper, accuracy not distinguishable) or N (not dominated). `break_even_price_ratio`
returns the output:input price ratio below which one configuration is cheaper than another in
price-weighted tokens (input + r x output), with a 95% interval. Inputs are arrays of shape
(items, seeds); with the released logs they return the paper's values
(`python3 -m unittest tests.test_dominance`, about 20 seconds).

```python
from soc.dominance import cost_matched_dominance, break_even_price_ratio
v = cost_matched_dominance(acc_debate, tok_debate, {"SA": acc_sa, "SC_k4": acc_sc4}, {"SA": tok_sa, "SC_k4": tok_sc4})
r = break_even_price_ratio(in_sc8, out_sc8, in_debate, out_debate)   # r["point"], r["ci95"]
```

## Logs

One JSONL line per (item, configuration, seed): `item_id`, `config`, `seed`, `pred`,
`gold`, `correct`, `in_tokens`, `out_tokens`, `total_tokens`, `n_calls`, `latency_s`;
the November 2026 runs (`results/power/`, `*_coord_trace`) add `calls` with the raw text
of every model call, the cap, per-call tokens and the harness-extracted answer. The
HumanEval run adds per-call GPU energy. The `host` field is a hardware label
(`mac-m4`, `rtx4060-pc`, `rtx4060-pc-par6` = the RTX 4060 server with six parallel
slots), not a network name. Appendix A of the paper (Table `tab:overview`) lists which log
backs which experiment. No log is larger than 10 MB, so none is compressed, except the
cross-model replication logs (`results/cloud/qwen3_raw_logs.tar.xz`, 35 MB, 358 MB unpacked;
`results/cloud/nonthinking_raw_logs.tar.xz`, 10 MB, 139 MB unpacked; see "Cross-model
replication").

### rev3 reruns (R1 to R5)

`results/rev3/` holds the reruns fixed in `protocols/REV3_RERUN_PROTOCOL.md` (written and
committed before any rerun record existed), all on the RTX 4060 server (`host`
`rtx4060-pc-par6`, six parallel slots):

| Run | Task, items | Seeds | Cap | Configurations | Files |
|---|---|---|---|---|---|
| R1 | GSM8K, first 500 | 1-3 | 640 | HIERFIX | `rev3/gsm8k_qwen3b_s*_sh*of2.jsonl` |
| R2 | MMLU, the 80 of the matrix | 1-2 | 640 | HIERFIX | `rev3/mmlu_qwen3b_*` |
| R3 | HumanEval, all 164 (raw text, both extractors) | 1-2 (SA: 1) | 1,024 | SA, SC k=8, debate, GE, HIERFIX | `rev3/humaneval_qwen3b_*` |
| R4 | MMLU-Pro engineering, 150 | 1-2 | 1,024 | HIERFIX | `rev3/mmlu_pro_eng_qwen3b_*` |
| R5 | MATH-500 levels 1-3, 238 | 1-3 (SA: 1) | 2,048 | SA, SC k=12, debate, GE, HIERFIX | `rev3/math500_cap2048_qwen3b_*` |

The GSM8K headline run (500 items x 3 seeds, SA, SC, debate, GE, HIER) is
`results/power/gsm8k_qwen3b_s{1,2,3}_sh{0,1}of2.jsonl`. HumanEval records keep both the
original last-block extractor (`pred`, `correct`) and the entry-block extractor
(`pred_v2`, `correct_v2`) on the same calls.

Analysis only (no model call; about 50 minutes with 8 processes on an Apple M4):

```bash
python3 scripts/analyze_rev3_reruns.py --r5 --jobs 8     # results/extra/rev3_reruns_analysis.json
python3 scripts/analyze_rev3_costs.py                     # results/extra/rev3_costs_analysis.json
python3 paper/arxiv_rev3/scripts/emit_rev3_reruns.py      # rev3_reruns_macros.tex, tables/rev3/rr_*.tex
python3 reproduce/check_rev3.py
```

`--r5` adds the equal-cap difficulty interaction (C5) and the Holm adjustment over C1 to C5;
the script refuses to compute anything from R5 if any R5 record is missing.
`paper/arxiv_rev3/scripts/fold_in_r5.sh` is the author's one-command fold-in; it also
compiles the manuscript, which is not part of this repository, so use the commands above.

To regenerate the records (needs the GPU server; one process per seed and shard, resumable):

```bash
OLLAMA_URL=http://localhost:11435 AGENTS_HOST_LABEL=rtx4060-pc-par6 \
  python3 -u scripts/run_rev3_campaign.py --task humaneval --cap 1024 --n 164 --seed 1 \
    --configs SA,SC,DEBATE,GE,HIERFIX --sc-k 8 --shard 0/3 --out-prefix results/rev3/humaneval_qwen3b
```

`scripts/run_rev3_campaign.sh` runs the whole queue R1 to R5 in protocol order with shards,
resume and a STOP file; set `GPU_SSH` (SSH alias of the GPU host) and `GPU_START_CMD`
(command that starts a six-slot Ollama server there) for your machine. The R5 runs use the
8,192-context alias `qwen2.5:3b-instruct-ctx8k`.

## Cross-model replication (Qwen3-4B, Qwen3-8B, Phi-4-mini, Qwen2.5-3B)

The GSM8K (first 500 items) and MATH levels 1 to 3 (238 items) runs were repeated with four
further models served by vLLM 0.9.2 at 16-bit precision, with the same harness, prompts,
extraction, seeds and record schema (SA greedy seed 1; SC k=8, whose first 2 and 4 calls give
SC k=2 and 4; debate N=3 R=1; GE R=2; the repaired HIER):

- Thinking models: Qwen/Qwen3-4B and Qwen/Qwen3-8B with thinking on, bf16, one rented H100 each,
  prefix caching on. The think block counts as output tokens and is not passed to other agents.
  Caps: 4,096 on GSM8K, 8,192 on MATH. Qwen3-8B on MATH stopped at its budget cap: seed 1 on all
  238 items and seed 2 on the 135 items with all four workflow records.
- Non-thinking models: microsoft/Phi-4-mini-instruct and Qwen/Qwen2.5-3B-Instruct (unquantized,
  not the 4-bit Ollama build of the main runs), fp16 on a Kaggle 2x T4 session (one T4 per
  model; the T4 has no bf16), vLLM's V0 engine with prefix caching off (its prefix-prefill kernel
  fails in fp16 on this GPU). The main runs' caps (640 on GSM8K, 1,024 on MATH); seeds 1 to 3
  complete, 9,594 units per model, no failed request. Notebook:
  `scripts/cloud/kaggle_nonthinking.ipynb`, built by `scripts/cloud/make_kaggle_notebook.py`.

A seed of an item is used only when every workflow record exists, so every comparison is paired.
Records are in `results/cloud/qwen3_raw_logs.tar.xz` (Qwen3) and
`results/cloud/nonthinking_raw_logs.tar.xz` (Phi-4-mini, Qwen2.5-3B): per-record logs with the
raw text of every call, and the vLLM server logs; `results/cloud/<model>/` holds the run
summaries and progress files. Analysis (no model call):

```bash
tar -xJf results/cloud/qwen3_raw_logs.tar.xz -C results/cloud        # optional, see below
tar -xJf results/cloud/nonthinking_raw_logs.tar.xz -C results/cloud  # optional, see below
python3 scripts/cloud/cross_model_analysis.py --outdir paper/arxiv_rev3
python3 reproduce/check_rev3.py      # the cross-model files are in its sha256 list
```

The script applies the paper's tests (hierarchical paired bootstrap, 10,000 replicates, seed 0),
the R/C/N verdict under total and output tokens, the token multiple, the break-even price ratio
and truncation shares to every model directory under `results/cloud/`, and writes
`results/extra/cross_model_analysis.json`, `cross_model_macros.tex` and `tables/cross_model_*.tex`.
A (model, task) whose logs are unchanged, or not unpacked, is taken from that JSON;
`--fresh` recomputes everything (logs unpacked). Under total tokens no coordination workflow
was detectably more accurate than every cheaper sampling configuration in any of the 24
workflow, model and task combinations (verdicts: 12 R, 12 C); every non-thinking combination
was R. Under output tokens (6 R, 17 C, 1 N) HIER on Qwen3-8B, MATH levels 1 to 3, was
(against SA and SC k=2). The runs themselves need a GPU and an OpenAI-compatible server:
`scripts/cloud/README.md` describes the driver (`run_cloud_campaign.py`), the budget plan and
the Modal and Kaggle launchers.

`scripts/within_run_check.py` repeats each comparison whose arms come from different runs (the
repaired HIER on GSM8K, MMLU and MMLU-Pro engineering, the Du et al. debate, and C5) within one
run and compares its margin with the run-to-run drift of identical configurations
(`results/extra/within_run_check.json`; `--reuse-drift` takes the drift tests from that JSON).

## Hardware, model and software versions

- Apple M4 laptop (16 GB, macOS), Ollama 0.30.10: the July 2026 GSM8K, MMLU and 60-item
  MATH matrices, probes, traces, cross-family runs, `*_coord_trace` reruns.
- Desktop NVIDIA RTX 4060 (8 GB) under WSL2 Ubuntu 24.04, Ollama 0.40.1 with flash attention
  and an 8-bit key-value cache: the 200-item GSM8K run, MMLU-Pro engineering, HumanEval
  (with `nvidia-smi` energy sampling at 200 ms), and all `results/power/` runs (six
  parallel slots, reached from the laptop over an SSH tunnel; the launchers take the SSH
  alias in `GPU_SSH` and the server start command in `GPU_START_CMD`).
- Models (Ollama tags and manifest digests): `qwen2.5:3b-instruct` (Q4_K_M,
  `357c53fb659c`; an alias with an 8,192-token context, `qwen2.5:3b-instruct-ctx8k`, is
  used for the 2,048-cap runs), `qwen2.5:7b-instruct` (`845dbda0ea48`), `llama3.2:3b`
  (`a80c4f17acd5`), `gemma2:2b` (`8ccf136fdd52`). Llama-3.3-70B-Instruct ran through a
  hosted API (OpenRouter, cheapest provider; per-call provider and cost are logged).
- Decoding: temperature 0 for SA and the HIER aggregator, 0.7 for SC, debate, GE and HIER
  planner and executors; caps 640 (GSM8K, MMLU), 1,024 (MATH levels 1 to 3, MMLU-Pro
  engineering, HumanEval), 2,048 (MATH levels 4 to 5); seeds derived per call as documented
  in `harness/topologies.py`.
- HumanEval programs run in `docker run --network none` (`python:3.11-slim`) on the PC.

## Re-running experiments

```bash
ollama pull qwen2.5:3b-instruct
python3 scripts/run_matrix.py --task gsm8k --n 80 --set lean      # resumable; skips logged (item, config, seed)
OLLAMA_URL=http://localhost:11434 python3 -u scripts/run_power_campaign.py --help
```

Each driver documents its arguments in its docstring. Llama and Gemma generations under
Ollama are not bit-reproducible from run to run; Qwen2.5-3B greedy and seeded calls
reproduced exactly on the same server version (see Appendix of the paper on drift).
The Du et al. debate needs `sh third_party/du2023_debate/fetch_upstream.sh` first.

## Answer-extraction audits

`results/audit/` holds the cases, keys and readings of the GSM8K, MATH levels 1 to 3 and
MATH levels 4 to 5 extraction audits. The readings were made by an AI assistant (Claude);
no human checked them (as the paper states). A blind human spot check of 100 sampled calls
is in preparation.

## Licenses

- Code: MIT. Logs and generated data: CC BY 4.0. See `LICENSE`.
- Benchmarks (`tasks_data/`, and question text and gold answers that appear in some logs):
  GSM8K, MATH, MATH-500, MMLU and HumanEval are MIT; MMLU-Pro is Apache-2.0. All allow
  redistribution with attribution; see `tasks_data/LICENSES.md`.
- Model outputs in the logs were produced by models under their own terms: Qwen2.5-3B-Instruct
  (Qwen Research License), Qwen2.5-7B-Instruct (Apache-2.0), Llama 3.2 3B and Llama 3.3 70B
  (Llama Community Licenses), Gemma 2 2B (Gemma Terms of Use). Check those terms before
  using the outputs to train models.
- Contamination notice: the logs contain benchmark items with gold answers. Do not train on
  them if you evaluate on these benchmarks.
- The Du et al. (2023) debate code is not included (no upstream license).

## Citation

See `CITATION.cff`.
