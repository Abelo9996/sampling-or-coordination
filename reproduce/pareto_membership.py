"""GSM8K frontier membership over seed-aware resamples (paper Section 4, macros \\RthMem*).

Writes paper/arxiv_long/scripts/pareto_membership.out (the file that
paper/arxiv_rev3/data/pareto_membership.out copies). Logs only; no model calls.
Same call as the private consistency script that produced the file for the paper.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from harness import stats as S  # noqa: E402

CFGS = ["SA", "SC_k2", "SC_k4", "SC_k8", "SC_k12", "DEBATE_N2_R1", "DEBATE_N3_R1",
        "GE_R1", "GE_R2", "HIER_M2", "HIER_M3"]
out = S.pareto_membership("gsm8k", configs=CFGS)
path = os.path.join(ROOT, "paper", "arxiv_long", "scripts", "pareto_membership.out")
with open(path, "w") as f:
    f.write(json.dumps(out) + "\n")
print(json.dumps(out))
