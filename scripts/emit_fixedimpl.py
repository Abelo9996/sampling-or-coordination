"""Emit the fixed-implementation robustness table (GES vs GE, HIERC vs HIER,
DEBATE R=2 vs R=1) from gsm8k_matrix.jsonl, with paired tests vs SC_k8."""
import json, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from harness import stats as S
from collections import defaultdict

rows = [json.loads(l) for l in open(os.path.join(ROOT,'results','gsm8k_matrix.jsonl'))]
cor = S.per_item_correct(rows)
tok = S.per_item_tokens(rows)

def agg(c):
    if c not in cor: return None
    a = 100*np.mean(list(cor[c].values())); t = np.mean(list(tok[c].values()))
    return a, t

PAIRS = [("GE_R2","GES_R2","GE $\\to$ gated GE"),
         ("HIER_M3","HIERC_M3","HIER $\\to$ chained HIER"),
         ("DEBATE_N3_R1","DEBATE_N3_R2","Debate $R{=}1 \\to R{=}2$")]
lines = [r"\begin{table}[t]\centering\small",
 r"\caption{Repaired-implementation robustness on GSM8K: the gated "
 r"generator-evaluator (stops when its critic accepts), the sequentially "
 r"chained hierarchical pipeline (executors see prior sub-step results), and "
 r"debate at its canonical two-round depth. $p$ is the paired bootstrap test of "
 r"SC$_{k=8}$ minus the repaired variant.}",
 r"\label{tab:fixedimpl}", r"\begin{tabular}{lrrrr}", r"\toprule",
 r"Variant & Acc.\ (\%) & Tok/query & SC$_{k=8}$ $\Delta$ & $p$ \\", r"\midrule"]
sck8 = agg("SC_k8")
for old, new, name in PAIRS:
    o, n = agg(old), agg(new)
    if n is None: continue
    d, p = S.paired_test(cor["SC_k8"], cor[new])
    o_str = f"{o[0]:.1f} @ {o[1]:.0f}" if o else "--"
    pd = f"{max(p,1e-4):.3f}" if p >= 0.001 else "$<$0.001"
    lines.append(f"{name} & {n[0]:.1f} (was {o[0]:.1f}) & {n[1]:.0f} & {d:+.1f} & {pd} \\\\")
lines += [r"\midrule",
 rf"\multicolumn{{5}}{{l}}{{\footnotesize Reference: SC$_{{k=8}}$ = {sck8[0]:.1f}\% @ {sck8[1]:.0f} tokens/query.}} \\",
 r"\bottomrule", r"\end{tabular}", r"\end{table}"]
out = os.path.join(ROOT,'paper','fixedimpl_table.tex')
open(out,'w').write("\n".join(lines)+"\n")
print("wrote", out)
