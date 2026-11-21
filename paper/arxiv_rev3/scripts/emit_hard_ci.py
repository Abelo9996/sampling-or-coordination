#!/usr/bin/env python3
"""rev3: 95% accuracy intervals for the MATH levels 4 to 5 table (tab:hard).

No model call and no new test: the records of the levels 4 to 5 run are read with the loader of
scripts/analyze_hard_math.py (imported read-only), with the same item set (all 262 items of
levels 4 and 5; SA seed 1, every other configuration seeds 1 to 3; SC k=2,4,8 are the first k
samples of each SC k=12 record; Du et al. debate scored by its own parser and vote). Each
interval is the 95% item-bootstrap interval of the seed-averaged item correctness
(harness.analyze._bootstrap_ci, B=10,000, seed 0), the convention of every accuracy interval
of Table 1 (scripts/emit_math_easy.py). The script asserts that every accuracy equals
results/extra/hard_math_analysis.json, so the intervals belong to the published accuracies.

Writes tables/hard_math_ci.json (unrounded); scripts/emit_rev3_tables.py prints the intervals
in tab:hard, rounded to whole points like every accuracy interval of the paper.
Run: /usr/bin/python3 paper/arxiv_rev3/scripts/emit_hard_ci.py
"""
import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REV3 = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(REV3))
sys.path.insert(0, ROOT)
from harness import analyze, tasks, topologies  # noqa: E402

spec = importlib.util.spec_from_file_location("hm", os.path.join(ROOT, "scripts/analyze_hard_math.py"))
hm = importlib.util.module_from_spec(spec); spec.loader.exec_module(hm)

SEEDS, COORD, KS = hm.SEEDS, hm.COORD, hm.KS


def main():
    task = tasks.get_task("math500")
    items = hm.camp.load_math_levels(tasks, 4, 5, 262)
    ids = [it["id"] for it in items]
    recs = hm.load("results/power/math500hard_qwen3b_s*.jsonl")
    du = hm.load("results/power/extra_math45_qwen3b_s*.jsonl")
    M = {}
    for c in ["SA"] + COORD:
        seeds = [1] if c == "SA" else SEEDS
        M[c] = {i: np.array([float(bool(recs[(c, i, s)]["correct"])) for s in seeds]) for i in ids}
    for k in KS:
        m = {}
        for i in ids:
            v = []
            for s in SEEDS:
                r = recs[("SC_k12", i, s)]
                ans = topologies._majority([c["extracted"] for c in r["calls"][:k]])
                v.append(float(bool(task["score"](ans, r["gold"]))))
            m[i] = np.array(v)
        M[f"SC_k{k}"] = m
    M["DU"] = {i: np.array([float(bool(du[("DU_DEBATE_N3_R2", i, s)]["correct"])) for s in SEEDS]) for i in ids}

    P = json.load(open(os.path.join(ROOT, "results/extra/hard_math_analysis.json")))
    pub = dict(P["accuracy"], DU=P["du"]["accuracy"])
    J = {"source": "results/power/math500hard_qwen3b_s*.jsonl, results/power/extra_math45_qwen3b_s*.jsonl",
         "n_items": len(ids), "method": "harness.analyze._bootstrap_ci(seed-averaged item correctness, B=10000, seed=0)",
         "configs": {}}
    for c, m in M.items():
        corr = np.array([m[i].mean() for i in ids])
        acc = 100 * float(corr.mean())
        assert abs(round(acc, 2) - pub[c]) < 1e-9, (c, acc, pub[c])
        lo, hi = analyze._bootstrap_ci(corr, B=10000)
        J["configs"][c] = {"acc": acc, "lo": 100 * lo, "hi": 100 * hi}
    json.dump(J, open(os.path.join(REV3, "tables", "hard_math_ci.json"), "w"), indent=1)
    for c, d in J["configs"].items():
        print(f"{c:14s} {d['acc']:6.2f} [{d['lo']:.1f},{d['hi']:.1f}]")


if __name__ == "__main__":
    main()
