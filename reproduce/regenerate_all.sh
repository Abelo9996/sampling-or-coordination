#!/bin/sh
# Regenerate every analysis JSON, macro file, table fragment and figure of the paper from
# the logs in results/ (no model call). Run from anywhere; PY selects the interpreter.
# Afterwards: python3 reproduce/check_rev3.py   and   git status --short
set -e
cd "$(dirname "$0")/.."
PY=${PY:-python3}
export MPLBACKEND=Agg
mkdir -p paper/eaai/figures paper/eaai/upload paper/aaibd paper/arxiv_long/tables
run() { echo "== $*"; t0=$(date +%s); "$PY" "$@" >/dev/null; echo "   $(( $(date +%s) - t0 ))s"; }

# 1. pre-specified analyses and extraction-audit summaries (results/extra, results/audit)
run scripts/analyze_hard_math.py
run scripts/analyze_math_interaction.py
run scripts/analyze_extra_gsm8k.py
run results/audit/math_extraction_audit.py analyze
run results/audit/hardmath_extraction_audit.py analyze
# rev3 reruns R1-R5 (protocols/REV3_RERUN_PROTOCOL.md): every rerun test, C1-C5, Holm over five
run scripts/analyze_rev3_reruns.py --r5 --jobs ${JOBS:-8}

# 2. first-generation emitters (paper/*.tex, paper/eaai/, paper/aaibd/, results/analysis/)
for s in emit_crosstask emit_macros emit_evidence emit_humaneval emit_mmlu_pro_eng \
         emit_scale70b emit_throughput emit_truncation criterion_predictive \
         aaibd_revision_stats aaibd_emit_tables aaibd_emit_rerun_macros make_eaai_figures; do
  run scripts/$s.py
done
run reproduce/pareto_membership.py

# 3. long-version emitters (paper/arxiv_long/)
for s in emit_audit emit_context emit_equivalence emit_extra_noise_aware emit_extra_workflows \
         emit_hard_math emit_humaneval_audit emit_math_audit emit_math_interaction \
         emit_review_revision; do
  run paper/arxiv_long/scripts/$s.py
done

# 4. rev3 (paper/arxiv_rev3/): numbers, tables, figures
run paper/arxiv_rev3/scripts/emit_math_easy.py
run paper/arxiv_rev3/scripts/emit_rev3_macros.py
run paper/arxiv_rev3/scripts/check_gsm8k_rerun_subset.py
run paper/arxiv_rev3/scripts/emit_rev3_tables.py
run paper/arxiv_rev3/scripts/emit_rev3_reruns.py
for s in make_fig_forest make_fig_workflows make_fig_frontier make_fig_gsm8k_pareto \
         make_fig_eaai_adapted fig_hard_frontier fig_price_crossover fig_drift fig_energy; do
  run paper/arxiv_rev3/scripts/$s.py
done
run paper/arxiv_rev3/scripts/make_fig_frontier.py --extra
echo "done. Compare: $PY reproduce/check_rev3.py"
