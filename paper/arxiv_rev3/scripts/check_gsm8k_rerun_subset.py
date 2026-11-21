#!/usr/bin/env python3
"""Pin the GSM8K raw-output rerun of Appendix L to its 100 items (audit L13).

results/power/gsm8k_qwen3b_s{1,2,3}.jsonl now also hold records of a later, larger campaign
(about 500 items per seed). The paper's rerun numbers (Tables 21 to 24, tables/audit.json,
results/extra/extra_gsm8k_analysis.json) are for the first 100 items of the harness's GSM8K
loader order only. This script writes those 100 item ids to data/gsm8k_rerun_items.json and
checks that the records restricted to them reproduce tables/audit.json (record counts,
accuracies and token means of every 640-cap configuration). It reads the logs only.
Run: /usr/bin/python3 paper/arxiv_rev3/scripts/check_gsm8k_rerun_subset.py
"""
import json, os, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REV3 = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(REV3))
sys.path.insert(0, ROOT)
from harness import tasks  # noqa: E402

ids = [it["id"] for it in tasks.get_task("gsm8k")["load"](n=100, split="all")]
assert len(ids) == len(set(ids)) == 100
json.dump({"source": "harness.tasks gsm8k loader, n=100, split='all' (first 100 items, loader order)",
           "item_ids": ids}, open(os.path.join(REV3, "data", "gsm8k_rerun_items.json"), "w"), indent=1)
keep = set(ids)
recs = defaultdict(list)
n_other = 0
for s in (1, 2, 3):
    for l in open(os.path.join(ROOT, "results", "power", f"gsm8k_qwen3b_s{s}.jsonl")):
        if not l.strip():
            continue
        r = json.loads(l)
        if r["item_id"] in keep:
            recs[r["config"]].append(r)
        else:
            n_other += 1
A = json.load(open(os.path.join(REV3, "tables", "audit.json")))["configs_640"]
for cfg, ref in A.items():
    R = recs[cfg]
    keys = {(r["item_id"], r["seed"]) for r in R}
    assert len(R) == len(keys) == ref["records"], (cfg, len(R), len(keys), ref["records"])
    acc = 100 * sum(bool(r["correct"]) for r in R) / len(R)
    tot = sum(r["total_tokens"] for r in R) / len(R)
    assert abs(acc - ref["acc"]) < 1e-9 and abs(tot - ref["total_tokens"]) < 1e-6, (cfg, acc, tot)
print(f"ok: 100 items pinned; {sum(len(v) for v in recs.values())} records reproduce tables/audit.json; "
      f"{n_other} records of other items ignored")
