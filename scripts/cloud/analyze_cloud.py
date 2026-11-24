"""Analyse a cloud run with the rev3 analysis functions, unchanged
(scripts/analyze_rev3_reruns.py: load, cell_from, sc_prefix_cells, plan_task,
finish_task with the paired item bootstrap of the paper, harness.stats via
scripts/analyze_hard_math.test), plus truncation and think-block statistics.

Uses the seeds that are complete for every config (all items), so a run stopped
by the budget guard is analysed on its finished seeds.

  /usr/bin/python3 scripts/cloud/analyze_cloud.py results/cloud/qwen3-4b [--out file.json]
"""
import argparse
import glob
import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from harness import tasks  # noqa: E402

spec = importlib.util.spec_from_file_location("rr", os.path.join(ROOT, "scripts/analyze_rev3_reruns.py"))
rr = importlib.util.module_from_spec(spec)
sys.modules["rr"] = rr  # the forked test workers unpickle rr._job by module name
spec.loader.exec_module(rr)

SAMP = ["SA", "SC_k2", "SC_k4", "SC_k8"]
COORD = ["DEBATE_N3_R1", "GE_R2", "HIERFIX_M3"]


def analyse(d, n_jobs=4, min_items=None):
    out = {"dir": d, "tasks": {}}
    for tn in ("gsm8k", "math500"):
        files = sorted(glob.glob(os.path.join(d, f"{tn}_s*.jsonl")))
        if not files:
            continue
        recs = rr.load([os.path.relpath(p, ROOT) for p in files])
        ids = sorted({k[1] for k in recs if k[0] == "SA"})
        if min_items:
            assert len(ids) >= min_items, (tn, len(ids))
        seeds = sorted({k[2] for k in recs})
        full = [s for s in seeds if all((c, i, s) in recs for c in ["SC_k8"] + COORD for i in ids)]
        if not ids or not full:
            out["tasks"][tn] = {"n_items": len(ids), "complete_seeds": full, "note": "nothing complete"}
            continue
        task = tasks.get_task(tn)
        cells = {"SA": rr.cell_from(recs, "SA", ids, full)}
        cells.update({c: rr.cell_from(recs, c, ids, full) for c in ["SC_k8"] + COORD})
        cells.update(rr.sc_prefix_cells(recs, "SC_k8", ids, full, task, [2, 4]))
        T = rr.run_tests(rr.plan_task(tn, cells, ids, SAMP, COORD), n_jobs)
        res = rr.finish_task(tn, cells, ids, SAMP, COORD, T, [2, 4, 8])
        calls = [c for k, r in recs.items() if k[2] in full for c in r["calls"]]
        res["complete_seeds"] = full
        res["calls"] = len(calls)
        res["hit_cap_pct"] = round(100 * float(np.mean([c["hit_cap"] for c in calls])), 2)
        res["think_unclosed_pct"] = round(100 * float(np.mean([c.get("think_closed") is False for c in calls])), 2)
        res["mean_out_tokens_per_call"] = round(float(np.mean([c["out_tokens"] for c in calls])), 1)
        res["by_config_hit_cap_pct"] = {
            cfg: round(100 * float(np.mean([c["hit_cap"] for k, r in recs.items()
                                             if k[0] == cfg and k[2] in full for c in r["calls"]])), 2)
            for cfg in ["SA", "SC_k8"] + COORD}
        out["tasks"][tn] = res
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--out", default="")
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    r = analyse(os.path.abspath(a.dir), a.jobs)
    s = json.dumps(r, indent=1, default=float)
    if a.out:
        open(a.out, "w").write(s)
    for tn, t in r["tasks"].items():
        print(tn, "items", t.get("n_items"), "seeds", t.get("complete_seeds"), "cap%", t.get("hit_cap_pct"),
              "unclosed%", t.get("think_unclosed_pct"))
        for c, v in t.get("configs", {}).items():
            print(f"  {c:14s} acc {v['acc']:.1f} [{v['lo']:.1f},{v['hi']:.1f}] tok {v['total_tokens']:.0f}")
        for w, v in t.get("verdict_total", {}).items():
            print(f"  verdict {w}: {v['verdict']} comparator {t['comparator'][w]}")
