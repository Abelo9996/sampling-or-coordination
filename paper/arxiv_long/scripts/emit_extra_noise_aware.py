"""Noise-aware Pareto verdicts for the two tasks that are not in the poster
(MMLU-Pro engineering, HumanEval), with the poster's rule and statistics.

Rule (as in paper/aaibd/main_aaibd_rev.tex, Sec. II): for a coordination config C
and each sampling config S (SA, SC k) with lower mean cost, compute the
seed-aware hierarchical paired bootstrap of accuracy S - C
(harness.stats.hier_paired_test, 10,000 replicates, seed 0) and an item-level
paired bootstrap interval of the cost difference. R (robustly dominated) if some
cheaper S has a 95% accuracy interval above zero; C (cheaper, accuracy not
distinguishable) if no R and some cheaper S has an interval containing zero;
N otherwise. Holm over the three SC k=8 vs coordination comparisons and, on
HumanEval, over the three SA vs coordination comparisons.

Reads only results/*.jsonl; no model calls. Writes
paper/arxiv_long/tables/extra_domination_table.tex and
paper/arxiv_long/tables/extra_noise_aware.json.
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
from harness import stats as S  # noqa: E402

B = 10000
SEED = 0
SAMPLING = ["SA", "SC_k2", "SC_k4", "SC_k8"]
COORD = ["DEBATE_N3_R1", "GE_R2", "HIER_M3"]
TASKS = {
    "proeng": ("mmlu_pro_eng_matrix.jsonl", ["total_tokens"]),
    "code": ("humaneval_matrix_pc.jsonl", ["total_tokens", "energy_j"]),
}
LBL = {"SA": "SA", "SC_k2": "SC$_2$", "SC_k4": "SC$_4$", "SC_k8": "SC$_8$",
       "DEBATE_N3_R1": "Debate", "GE_R2": "GE", "HIER_M3": "HIER"}

_rows = {}


def rows(f):
    if f not in _rows:
        _rows[f] = [json.loads(l) for l in open(os.path.join(ROOT, "results", f)) if l.strip()]
    return _rows[f]


def job(args):
    f, a, b = args
    cor = S.per_item_seed_correct(rows(f))
    d, p, lo, hi = S.hier_paired_test(cor[a], cor[b], B=B, seed=SEED)
    return (f, a, b), {"diff": d, "p": p, "lo": lo, "hi": hi}


def cost_diff(f, a, b, field):
    m = S.per_item_seed_tokens(rows(f), field)
    items = sorted(set(m[a]) & set(m[b]))
    d = np.array([m[a][i].mean() - m[b][i].mean() for i in items])
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(d), size=(B, len(d)))
    boot = d[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def mean_cost(f, c, field):
    m = S.per_item_seed_tokens(rows(f), field)[c]
    return float(np.mean([v.mean() for v in m.values()]))


def mean_acc(f, c):
    m = S.per_item_seed_correct(rows(f))[c]
    return 100 * float(np.mean([v.mean() for v in m.values()]))


def main():
    jobs = []
    for t, (f, _) in TASKS.items():
        for s in SAMPLING:
            for c in COORD:
                jobs.append((f, s, c))
    with Pool(min(8, os.cpu_count() or 2)) as pool:
        res = dict(pool.map(job, jobs))
    out = {}
    for t, (f, fields) in TASKS.items():
        T = {"configs": {}, "pairs": [], "verdicts": {}, "holm": {}}
        for c in SAMPLING + COORD:
            T["configs"][c] = {"acc": mean_acc(f, c),
                               **{fld: mean_cost(f, c, fld) for fld in fields}}
        for fld in fields:
            T["verdicts"][fld] = {}
            for c in COORD:
                best = "N"
                for s in SAMPLING:
                    if T["configs"][s][fld] >= T["configs"][c][fld]:
                        continue
                    r = res[(f, s, c)]
                    v = "R" if r["lo"] > 0 else ("C" if r["hi"] >= 0 else "N")
                    dc = cost_diff(f, s, c, fld)
                    T["pairs"].append({"cost": fld, "sampling": s, "coord": c,
                                       **r, "cost_diff": dc[0], "cost_lo": dc[1],
                                       "cost_hi": dc[2], "verdict": v})
                    if v == "R" or (v == "C" and best == "N"):
                        best = v
                T["verdicts"][fld][c] = best
        for base in (["SC_k8", "SA"] if t == "code" else ["SC_k8"]):
            ps = [res[(f, base, c)]["p"] for c in COORD]
            T["holm"][base] = dict(zip(COORD, S.holm(ps)))
        out[t] = T
    os.makedirs(os.path.join(HERE, "..", "tables"), exist_ok=True)
    with open(os.path.join(HERE, "..", "tables", "extra_noise_aware.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    render(out)


def fmt_row(t, c, p, show_t, show_c):
    name = {"proeng": "MMLU-Pro eng.", "code": "HumanEval"}
    return (f"{name[t] if show_t else ''} & {LBL[c] if show_c else ''} & {LBL[p['sampling']]} & "
            f"${p['diff']:+.1f}$ {{\\tiny$[{p['lo']:+.1f},{p['hi']:+.1f}]$}} & "
            f"${p['cost_diff']:+.0f}$ {{\\tiny$[{p['cost_lo']:+.0f},{p['cost_hi']:+.0f}]$}} & {p['verdict']} \\\\").replace("+-", "-")


def render(out):
    """Full table (every cheaper comparator) and a compact one (per coordination
    workflow: the cheapest sampling config, plus the cheapest robust dominator if
    that is a different config), both on total tokens; plus text macros."""
    hdr = "% AUTO-GENERATED by paper/arxiv_long/scripts/emit_extra_noise_aware.py. Do not edit."
    full, compact = [hdr], [hdr]
    for t in TASKS:
        ft = ct = True
        for c in COORD:
            ps = [p for p in out[t]["pairs"] if p["cost"] == "total_tokens" and p["coord"] == c]
            ps.sort(key=lambda p: out[t]["configs"][p["sampling"]]["total_tokens"])
            fc = True
            for p in ps:
                full.append(fmt_row(t, c, p, ft, fc)); ft = fc = False
            keep = [ps[0]]
            rob = [p for p in ps if p["verdict"] == "R"]
            if rob and rob[0] is not ps[0]:
                keep.append(rob[0])
            cc = True
            for p in keep:
                compact.append(fmt_row(t, c, p, ct, cc)); ct = cc = False
        if t == "proeng":
            full.append("\\midrule"); compact.append("\\midrule")
    T = os.path.join(HERE, "..", "tables")
    open(os.path.join(T, "extra_domination_table.tex"), "w").write("\n".join(full) + "\n")
    open(os.path.join(T, "extra_domination_compact.tex"), "w").write("\n".join(compact) + "\n")
    # macros
    word = {"R": "robustly dominated", "C": "cheaper sampling, accuracy not distinguishable", "N": "not dominated"}
    key = {"proeng": "Eng", "code": "Code"}
    ck = {"DEBATE_N3_R1": "Deb", "GE_R2": "Ge", "HIER_M3": "Hier"}
    m = [hdr]
    for t in TASKS:
        for fld, fk in (("total_tokens", "Tok"), ("energy_j", "En")):
            if fld not in out[t]["verdicts"]:
                continue
            for c in COORD:
                m.append(f"\\newcommand{{\\Xtr{key[t]}{ck[c]}Verdict{fk}}}{{{out[t]['verdicts'][fld][c]}}}")
        for base, bk in (("SC_k8", "Sc"), ("SA", "Sa")):
            if base in out[t]["holm"]:
                for c in COORD:
                    p = out[t]["holm"][base][c]
                    s = "{<}0.001" if p < 0.001 else f"{p:.3f}"
                    m.append(f"\\newcommand{{\\Xtr{key[t]}{bk}{ck[c]}Holm}}{{{s}}}")
    open(os.path.join(HERE, "..", "extra_noise_aware_macros.tex"), "w").write("\n".join(m) + "\n")


if __name__ == "__main__":
    if sys.argv[1:] == ["--render"]:
        render(json.load(open(os.path.join(HERE, "..", "tables", "extra_noise_aware.json"))))
    else:
        main()
