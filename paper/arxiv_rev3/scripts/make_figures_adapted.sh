#!/bin/sh
# Regenerate the adapted rev3 figures and the criterion-units table (run from anywhere).
# Needs tables/math_easy.json (scripts/emit_math_easy.py) for fig_frontier and fig_forest.
set -e
cd "$(dirname "$0")"
PY=${PY:-python3}
$PY make_fig_forest.py
$PY make_fig_workflows.py
$PY make_fig_frontier.py
$PY make_fig_frontier.py --extra
$PY make_fig_gsm8k_pareto.py
$PY make_fig_eaai_adapted.py
cd ../figures
for f in fig_forest fig_workflows fig_frontier fig_frontier_extra fig_gsm8k_pareto fig_flow fig_criterion \
         fig_code_energy fig_proeng_pareto; do
  if pdffonts "$f.pdf" | grep -q "Type 3"; then echo "Type 3 font in $f.pdf"; exit 1; fi
done
echo "all adapted figures written; no Type 3 fonts"
