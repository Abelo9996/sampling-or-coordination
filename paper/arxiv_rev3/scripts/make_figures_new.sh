#!/bin/sh
# rev3: build the four new figures (hard-MATH frontier, price crossover, run-to-run drift,
# HumanEval energy). Each script asserts its plotted numbers against the paper's sources.
set -e
cd "$(dirname "$0")/../../.."
PY=${PY:-python3}
for s in fig_hard_frontier fig_price_crossover fig_drift fig_energy; do
  echo "== $s"
  $PY paper/arxiv_rev3/scripts/$s.py
done
for f in fig_hard_frontier fig_price_crossover fig_drift fig_energy; do
  if pdffonts paper/arxiv_rev3/figures/$f.pdf | grep -q "Type 3"; then
    echo "Type 3 font in $f.pdf" >&2; exit 1
  fi
done
echo "ok: 4 figures, no Type 3 fonts"
