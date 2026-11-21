# Sampling or Coordination?

Code, prompts and per-record logs for the paper

> **Sampling or Coordination? Self-Consistency vs. Multi-Agent Debate at Matched Token,
> Price and Energy Cost in a Small LLM.** Abel Yagubyan, Independent Researcher, 2026.
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
| `harness/` | Evaluation harness: Ollama client with exact token accounting (`client.py`), task loaders, prompts and scorers (`tasks.py`), the workflows (`topologies.py`: SA, SC, DEBATE, GE, HIER and variants), statistics (`stats.py`: seed-aware hierarchical bootstrap), HumanEval sandbox (`sandbox.py`), GPU energy attribution (`energy.py`), hosted-API client for the 70B probe (`openai_client.py`) |
| `configs/matrix.py` | The configuration matrix (which workflows and parameters are run) |
| `scripts/run_*.py`, `scripts/run_*.sh` | Run drivers (model calls); see "Re-running experiments" |
| `scripts/extra_workflows.py`, `scripts/rev3_workflows.py` | Du et al. debate reimplementation and the repaired planner-executor (HIERFIX) |
| `scripts/analyze_*.py`, `scripts/emit_*.py`, `scripts/criterion_predictive*.py`, `scripts/aaibd_*.py`, `scripts/make_*.py` | Analysis and emitters (logs to JSON, LaTeX macros and table fragments, figures) |
| `paper/arxiv_long/scripts/` | Emitters for the audit, equivalence, hard-MATH, extra-workflow, context and review tables |
| `paper/arxiv_rev3/scripts/` | rev3 emitters and figure scripts (each figure script asserts its plotted numbers against the macros) |
| `paper/arxiv_rev3/*_macros.tex`, `paper/arxiv_rev3/tables/`, `paper/arxiv_rev3/figures/`, `paper/arxiv_rev3/data/` | The exact generated numbers, table fragments and figures used in the paper (reference outputs; the manuscript source itself is not included) |
| `paper/arxiv_long/*_macros.tex`, `paper/arxiv_long/tables/`, `paper/aaibd/revision_stats.json` | Intermediate generated outputs that the rev3 scripts read |
| `protocols/` | Analysis protocols fixed before the runs they govern (hard MATH, extra workflows, rev3 reruns) and the original preregistration |
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
   pre-registered family C1 to C5 and its Holm adjustment; `results/extra/rev3_reruns_analysis.json`),
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
   `check_gsm8k_rerun_subset.py`, `emit_rev3_tables.py`, `emit_rev3_reruns.py` (rerun macros
   `rev3_reruns_macros.tex` and the `tables/rev3/rr_*.tex` fragments), then the figures.

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
| Table 3 `tab:costverdict` | `rr_cost_verdict.tex` | `analyze_rev3_reruns.py`, `emit_rev3_reruns.py` | as Table 1 |
| `tab:energyverdict` (modeled energy verdicts) | `rr_energy_verdict.tex` | `scripts/analyze_rev3_costs.py` (modeled J per query from the HumanEval per-call energy fit; item-bootstrap intervals), then `emit_rev3_reruns.py` | GSM8K 500 x 3, MMLU matrix + R2, MATH 238 x 3, MMLU-Pro eng. matrix + R4, R3 HumanEval, `humaneval_matrix_pc` (fit) |
| Price-ratio crossovers with 95% intervals (`\RrCross*`, `\RrCrossCI*`; `fig:crossover`) | `rev3_reruns_macros.tex`, `fig_price_crossover.pdf` | `scripts/analyze_rev3_costs.py` (seed-aware bootstrap of r*, 10,000 replicates), `emit_rev3_reruns.py`, `fig_price_crossover.py` | `power/gsm8k_qwen3b_s*_sh*of2`, `power/math500_qwen3b_s*` |
| `tab:confirm` (C1 to C5, Holm over five) | `rr_confirm_table.tex` | `analyze_rev3_reruns.py --r5`, `emit_rev3_reruns.py` | `power/gsm8k_qwen3b_s*_sh*of2`, `rev3/gsm8k_*`, `rev3/humaneval_*`, `power/math500hard_*`, `rev3/math500_cap2048_*` |
| `tab:hard` | `hard_math_table.tex` | `scripts/analyze_hard_math.py`, then `emit_hard_math.py` | `power/math500hard_qwen3b_s*`, `power/extra_math45_qwen3b_s*` |
| `tab:outverdict`, `tab:extra`, `tab:extra-full`, `tab:equiv`, `tab:code`, `tab:hierorig`, `fig:frontier-extra` | `rr_output_verdict.tex`, `rr_extra_compact.tex`, `rr_extra_full.tex`, `rr_equivalence.tex`, `rr_humaneval_table.tex`, `rr_hier_table.tex`, `fig_frontier_extra.pdf` | `emit_rev3_reruns.py`; `make_fig_frontier.py --extra` | rerun records and the original matrices |
| `tab:orig`, `tab:orig-dom`, `tab:costverdict-orig`, `tab:extra-orig`, `tab:equiv-orig` (original exploratory run, unrepaired HIER) | `poster_main_table.tex`, `poster_domination_table.tex`, `cost_verdict_table.tex`, `extra_domination_compact.tex`, `equivalence_table.tex` | `aaibd_emit_tables.py`, `emit_review_revision.py`, `emit_extra_noise_aware.py`, `emit_equivalence.py` | July matrices, `mmlu_pro_eng_matrix`, `humaneval_matrix_pc` |
| Table 5 `tab:overview` | inline text | | |
| `tab:round0`, `tab:robust`, `tab:cost` | `poster_*_table.tex` | `aaibd_revision_stats.py` then `aaibd_emit_tables.py` | July matrices, `mechanistic`, traces |
| `tab:predictive`, `tab:S-splits`, `tab:S-pooled`, `fig:criterion` | `criterion_*_table.tex`, `fig_criterion.pdf` | `scripts/criterion_predictive.py`; figure `make_fig_eaai_adapted.py` | `*_traces_mac*`, `mechanistic`, `gsm8k_llama70b` |
| `tab:proeng`, `fig:proeng` | `mmlu_pro_eng_table.tex`, `fig_proeng_pareto.pdf` | `scripts/emit_mmlu_pro_eng.py`; `make_fig_eaai_adapted.py` | `mmlu_pro_eng_matrix` |
| `fig:code`, `fig:energy` (original HumanEval run, energy) | `fig_code_energy.pdf`, `fig_energy.pdf` | `scripts/emit_humaneval.py`; `make_fig_eaai_adapted.py`; `fig_energy.py` | `humaneval_matrix_pc`, `humaneval_matrix_pc_idle`, `humaneval_secondary_vote` |
| `tab:heaudit` | `humaneval_audit_table.tex` | `emit_humaneval_audit.py` | `humaneval_matrix_pc` |
| `tab:rerun`, `tab:drift`, `tab:audit`, `fig:drift` | `audit_*_table.tex`, `fig_drift.pdf` | `emit_audit.py`; `fig_drift.py` | `power/gsm8k_qwen3b_s{1,2,3}`, `power/gsm8k_cap2048_*`, `gsm8k_qwen3b_coord_trace`, `results/audit/gsm8k_*` |
| `tab:extrawf` | `extra_workflows_table.tex` | `analyze_extra_gsm8k.py`, `emit_extra_workflows.py` | `power/extra_gsm8k_qwen3b_s*` |
| `tab:crossfamily` | macros in `rerun_macros.tex`, `review_revision_macros.tex` | `aaibd_emit_rerun_macros.py`, `emit_review_revision.py` | `gsm8k_llama3b*`, `gsm8k_gemma2b*`, `mmlu_*` |
| `tab:trunc`, `tab:scale`, `tab:scale70b`, `tab:diversity`, `tab:latency`, `tab:throughput`, `tab:units`, `tab:positioning` | `trunc_`, `scale_`, `scale70b_`, `diversity_`, `latency_`, `throughput_`, `criterion_units_`, `positioning_table.tex` | `emit_truncation.py`, `revision_stats.json`, `emit_scale70b.py`, `emit_throughput.py`, `make_fig_eaai_adapted.py`; positioning is a hand-written comparison table | `math500_matrix`, `math500_trunc`, `gsm8k_7b`, `gsm8k_llama70b`, `gsm8k_debate_*` |
| Fig. 1 `fig:overview` | `fig_forest.pdf` | `make_fig_forest.py` | via macros and `tables/math_easy.json` |
| Fig. 2 `fig:workflows`, Fig. 3 `fig:frontier`, Fig. 4 `fig:gsm8k`, Fig. 5 `fig:flow` | `fig_workflows.pdf`, `fig_frontier.pdf`, `fig_gsm8k_pareto.pdf`, `fig_flow.pdf` | `make_fig_workflows.py`, `make_fig_frontier.py`, `make_fig_gsm8k_pareto.py`, `make_fig_eaai_adapted.py` | matrices, `power/math500_qwen3b_s*`, `mechanistic` |
| `fig:crossover`, `fig:hard` | `fig_price_crossover.pdf`, `fig_hard_frontier.pdf` | `fig_price_crossover.py`, `fig_hard_frontier.py` | via `rev3_macros.tex`, `power/math500hard_*` |
| All in-text numbers | `paper/arxiv_rev3/*_macros.tex` | the emitter named in each file's first line | |

Table and figure files are under `paper/arxiv_rev3/`; scripts named without a path are in
`scripts/`, `paper/arxiv_long/scripts/` or `paper/arxiv_rev3/scripts/`.

## Logs

One JSONL line per (item, configuration, seed): `item_id`, `config`, `seed`, `pred`,
`gold`, `correct`, `in_tokens`, `out_tokens`, `total_tokens`, `n_calls`, `latency_s`;
the November 2026 runs (`results/power/`, `*_coord_trace`) add `calls` with the raw text
of every model call, the cap, per-call tokens and the harness-extracted answer. The
HumanEval run adds per-call GPU energy. The `host` field is a hardware label
(`mac-m4`, `rtx4060-pc`, `rtx4060-pc-par6` = the RTX 4060 server with six parallel
slots), not a network name. Appendix A of the paper (Table `tab:overview`) lists which log
backs which experiment. No log is larger than 10 MB, so none is compressed.

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
