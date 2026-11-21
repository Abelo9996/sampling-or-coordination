#!/bin/sh
# rev3 R5 fold-in in one command (protocols/REV3_RERUN_PROTOCOL.md, run R5: MATH levels 1 to 3 at
# the 2,048 cap). Run only after the R5 campaign has written its last record.
#  1. scripts/analyze_rev3_reruns.py --r5 recomputes every rerun test and adds C5 (equal-cap
#     difficulty interaction) and the Holm adjustment over C1 to C5. It counts the R5 records first
#     and exits with an error, computing nothing from R5, if any record is missing.
#  2. build.sh regenerates rev3_reruns_macros.tex (\RrRfivetrue) and the confirmatory table, and
#     the \ifRrRfive branches of main.tex switch the C5 and Holm sentences on.
#  3. figures, a second build, and the tarball (compiled from an empty directory).
# The manual checks that remain after this command are listed in REV3_TODO.md, section "R5".
# Usage: sh paper/arxiv_rev3/scripts/fold_in_r5.sh
set -e
cd "$(dirname "$0")/../../.."
/usr/bin/python3 scripts/analyze_rev3_reruns.py --r5 --jobs 9
cd paper/arxiv_rev3
sh scripts/build.sh
grep -q '\\RrRfivetrue' rev3_reruns_macros.tex || { echo "R5 not folded in"; exit 1; }
sh scripts/make_figures.sh
sh scripts/build.sh
sh scripts/make_tarball.sh
echo "R5 folded in. Remaining manual checks: REV3_TODO.md, section R5."
