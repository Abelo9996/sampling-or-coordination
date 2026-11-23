"""Shared 95% item-bootstrap accuracy intervals of the rev3 paper (Section 3, Statistics: seed
means within item, items resampled, 10,000 resamples, numpy seed 0, percentile interval).

The percentile bootstrap with a fixed seed depends on the number of resamples and on the order of
the per-item vector (the same random indices pick different items). The canonical values for every
run of Table 1 (GSM8K 500 x 3, MMLU 80 x 2, MMLU-Pro engineering 150 x 2, HumanEval 164 x 2) are the
ones in results/extra/rev3_reruns_analysis.json (scripts/analyze_rev3_reruns.py: describe ->
item_ci = harness.analyze._bootstrap_ci(B=10000, seed 0), items in the sorted order of that
script). Every script that prints one of these intervals reads it here, so a table, a macro and a
figure cannot disagree. Formatting: whole points, ci_cell().

  rr_config(task, cfg)  -> {"acc", "lo", "hi", ...} in percent (task: gsm8k, mmlu, proeng, code_v2, code_v1)
  ci_cell(lo, hi)       -> "[lo,hi]" rounded to whole points (the paper's table format)
  item_ci(corr)         -> (lo, hi) in [0, 1] with the stated method, for runs not in the JSON
  item_ci_by_id(d)      -> the same over {item_id: value} in sorted id order
                           (harness.analyze.item_bootstrap_ci; scripts/emit_humaneval.py uses it)
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
RR_JSON = os.path.join(ROOT, "results", "extra", "rev3_reruns_analysis.json")
B = 10000
SEED = 0
_RR = {}


def rr_tasks(path=None):
    path = path or os.environ.get("REV3_RERUNS_JSON", RR_JSON)
    if path not in _RR:
        _RR[path] = json.load(open(path))["tasks"]
    return _RR[path]


def rr_config(task, cfg, path=None):
    return rr_tasks(path)[task]["configs"][cfg]


def ci_cell(lo, hi):
    return f"[{lo:.0f},{hi:.0f}]"


def item_ci(corr):
    """95% item-bootstrap interval of the mean of per-item (seed-averaged) correctness."""
    import sys
    sys.path.insert(0, ROOT)
    from harness import analyze
    return analyze._bootstrap_ci(corr, B=B, seed=SEED)


def item_ci_by_id(per_item):
    """95% item-bootstrap interval over {item_id: value}, items in sorted id order."""
    import sys
    sys.path.insert(0, ROOT)
    from harness import analyze
    assert analyze.ITEM_CI_B == B
    return analyze.item_bootstrap_ci(per_item, B=B, seed=SEED)
