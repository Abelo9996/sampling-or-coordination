#!/bin/sh
# Regenerate every rev3 figure (each script asserts its plotted numbers against its source).
set -e
cd "$(dirname "$0")"
sh make_figures_adapted.sh
sh make_figures_new.sh
