"""Tier-C analyses (round-2 reviewer asks):
(1) paired significance at the confound-free 1024 cap (math500_trunc.jsonl);
(2) parse-failure (pred=None) rates per config/task.
Run: .venv/bin/python scripts/tierc_analysis.py
"""
import json, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from harness import stats as S
from collections import defaultdict

def load(path):
    return [json.loads(l) for l in open(os.path.join(ROOT,'results',path)) if l.strip()]

print("== (1) paired tests at 1024 cap (math500_trunc.jsonl) ==")
rows = load('math500_trunc.jsonl')
cor = S.per_item_correct(rows)
if 'SC_k8' in cor:
    for c in ['SA','DEBATE_N3_R1','GE_R2','HIER_M3']:
        if c in cor:
            d,p = S.paired_test(cor['SC_k8'], cor[c])
            n_seeds = len({r['seed'] for r in rows if r['config']=='SC_k8'})
            print(f"  SC_k8 vs {c:14} D={d:+5.1f}  p={p:.3f}  (seeds so far: {n_seeds})")

print("\n== (2) parse-failure (pred=None) rates per config ==")
for f,task in [('gsm8k_matrix.jsonl','gsm8k'),('mmlu_matrix.jsonl','mmlu'),
               ('math500_matrix.jsonl','math-640'),('math500_trunc.jsonl','math-1024')]:
    rows = load(f)
    byc = defaultdict(lambda: [0,0])
    for r in rows:
        byc[r['config']][1]+=1
        if r['pred'] is None or r['pred']=='None': byc[r['config']][0]+=1
    parts=[]
    for c in sorted(byc):
        n_null,n = byc[c]
        if n_null>0: parts.append(f"{c}:{100*n_null/n:.0f}%")
    print(f"  {task:10} " + (", ".join(parts) if parts else "all configs 0%"))
