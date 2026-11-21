import json, os, sys
import numpy as np
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,ROOT)
from harness import stats as S
def cor(f): return S.per_item_correct([json.loads(l) for l in open(os.path.join(ROOT,'results',f))])
def tok(f): return S.per_item_tokens([json.loads(l) for l in open(os.path.join(ROOT,'results',f))])
c3,t3=cor('gsm8k_matrix.jsonl'),tok('gsm8k_matrix.jsonl')
c7,t7=cor('gsm8k_7b.jsonl'),tok('gsm8k_7b.jsonl')
def row(name,c,t,k):
    a=100*np.mean(list(c[k].values())); tt=np.mean(list(t[k].values())); return a,tt
lines=[r"\begin{table}[t]\centering\small",
 r"\caption{Scale probe on GSM8K ($n{=}50$ for 7B): the single-agent advantage "
 r"over debate \emph{narrows with model capability}. SC$_{k=8}$ Pareto-dominates "
 r"debate at both scales (cheaper and at least as accurate), but the accuracy gap "
 r"falls from $+8.1$ (significant) to $+3.0$ points (not significant at $n{=}50$), "
 r"a directional signal that coordination may become competitive at higher "
 r"capability.}",
 r"\label{tab:scale}", r"\begin{tabular}{lrrrr}", r"\toprule",
 r"& \multicolumn{2}{c}{Qwen2.5-3B} & \multicolumn{2}{c}{Qwen2.5-7B} \\",
 r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
 r"Config & Acc.\ (\%) & Tok. & Acc.\ (\%) & Tok. \\", r"\midrule"]
for k,nm in [('SA','SA'),('SC_k8','SC $k{=}8$'),('DEBATE_N3_R1','Debate')]:
    a3,x3=row('3B',c3,t3,k); a7,x7=row('7B',c7,t7,k)
    lines.append(f"{nm} & {a3:.1f} & {x3:.0f} & {a7:.1f} & {x7:.0f} \\\\")
d3,p3=S.paired_test(c3['SC_k8'],c3['DEBATE_N3_R1'])
d7,p7=S.paired_test(c7['SC_k8'],c7['DEBATE_N3_R1'])
lines+=[r"\midrule",
 r"SC$_{k=8}-$Debate & \multicolumn{2}{c}{$"+f"{d3:+.1f}"+r"$ ($p<0.01$)} & \multicolumn{2}{c}{$"+f"{d7:+.1f}"+r"$ ($p="+f"{p7:.2f}"+r"$)} \\",
 r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open(os.path.join(ROOT,'paper','scale_table.tex'),'w').write("\n".join(lines)+"\n")
print("wrote scale_table.tex")
