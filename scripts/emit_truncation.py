"""Emit the MATH truncation-sensitivity table (640-cap vs 1024-cap) as LaTeX,
straight from results/math500_matrix.jsonl (640) and results/math500_trunc.jsonl
(1024). Addresses the reviewer ask: how much of each gap is real vs. a
generation-truncation artifact?
"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def acc(path, cfg):
    rows = [json.loads(l) for l in open(path)
            if json.loads(l)["config"] == cfg]
    return 100 * np.mean([r["correct"] for r in rows]) if rows else float("nan")


PRETTY = {"SA": "SA", "SC_k8": "SC $k{=}8$", "DEBATE_N3_R1": "Debate",
          "GE_R2": "GE", "HIER_M3": "HIER"}
ORDER = ["SA", "SC_k8", "DEBATE_N3_R1", "GE_R2", "HIER_M3"]

v1 = os.path.join(ROOT, "results", "math500_matrix.jsonl")
v2 = os.path.join(ROOT, "results", "math500_trunc.jsonl")

lines = [
    r"\begin{table}[t]\centering\small",
    r"\caption{MATH truncation sensitivity. Accuracy (\%) at the original "
    r"$640$-token generation cap (where $10$ to $17\%$ of single-agent generations "
    r"were truncated before emitting a parseable answer) versus a $1024$-token "
    r"cap ($<$$2\%$ truncation). Every configuration improves, but the ranking is "
    r"unchanged: at the confound-free cap SC$_{k=8}$ still beats every multi-agent "
    r"topology.}",
    r"\label{tab:trunc}",
    r"\begin{tabular}{lrrr}",
    r"\toprule",
    r"Config & 640-cap & 1024-cap & $\Delta$ \\",
    r"\midrule",
]
for c in ORDER:
    a1, a2 = acc(v1, c), acc(v2, c)
    lines.append(f"{PRETTY[c]} & {a1:.1f} & {a2:.1f} & {a2 - a1:+.1f} \\\\")
sc8 = acc(v2, "SC_k8")
lines += [r"\midrule",
          r"\multicolumn{4}{l}{\footnotesize SC$_{k=8}-$MAS at 1024-cap: "
          f"Debate ${sc8 - acc(v2, 'DEBATE_N3_R1'):+.1f}$, "
          f"GE ${sc8 - acc(v2, 'GE_R2'):+.1f}$, "
          f"HIER ${sc8 - acc(v2, 'HIER_M3'):+.1f}$ (all still favor SC).}} \\\\",
          r"\bottomrule", r"\end{tabular}", r"\end{table}"]

out = os.path.join(ROOT, "paper", "trunc_table.tex")
with open(out, "w") as f:
    f.write("\n".join(lines) + "\n")
print("wrote", out)
