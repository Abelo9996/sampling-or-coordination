"""Redraw the workflow schematic for the arXiv version.

Calls scripts/make_eaai_figures.py:fig_workflows() unchanged, except that the
SC label is corrected: the journal figure said "k = 12 on GSM8K only", but the
1024-cap MATH log (results/math500_trunc.jsonl) also has SC k=12 (one seed),
which the poster and this version report. Output goes to
paper/arxiv_long/figures/fig_workflows.pdf; nothing under paper/eaai/ is written.

Run: .venv/bin/python paper/arxiv_long/scripts/make_workflows_figure.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, ROOT)
import make_eaai_figures as M  # noqa: E402

OUT = os.path.join(HERE, "..", "figures")
k12_math = sorted({int(r["params"]["k"]) for r in M.load("math500_trunc.jsonl")
                   if r["config"].startswith("SC_")})
assert 12 in k12_math, k12_math


def save(fig, name):
    n = 0
    for t in fig.findobj(M.matplotlib.text.Text):
        s = t.get_text()
        if "on GSM8K only" in s:
            t.set_text(s.replace("on GSM8K only", "on GSM8K and MATH"))
            n += 1
    assert n == 1, n
    fig.savefig(os.path.join(OUT, f"{name}.pdf"))


M.save = save
print(json.dumps(M.fig_workflows(), default=str))
