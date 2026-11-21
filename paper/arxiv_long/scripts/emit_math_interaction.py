"""Primary difficulty interaction of protocols/HARD_MATH_PROTOCOL.md: numbers for the paper.

Inputs (read only; no model calls):
  results/extra/math_interaction_analysis.json  output of scripts/analyze_math_interaction.py
  results/power/math500hard_qwen3b_s*.jsonl     MATH levels 4 to 5, 262 items, cap 2,048
  results/power/math500_qwen3b_s*.jsonl         MATH levels 1 to 3, 238 items, cap 1,024
                                                (campaign phase B)

D = (W minus SC k=8 on levels 4 to 5) minus (W minus SC k=8 on levels 1 to 3), W in
debate (pre-specified prediction D > 0), GE and HIER (no prediction).

Recomputed here from the records and asserted equal to the JSON: item sets, caps, every
per-stratum accuracy (SC k=8 = majority by harness.topologies._majority of the first 8
extracted answers of SC k=12, scored by the math500 scorer), the observed D values, and
(same bootstrap draws as the analysis script: hard stratum seed 0, easy stratum seed 1,
10,000 replicates, items then seeds within items) the 95% interval and two-sided p of
each D, plus each workflow minus SC k=8 on levels 1 to 3 (harness.stats.hier_paired_test,
B=10,000, seed 0). Asserts the statements the text makes.
Writes math_interaction_macros.tex (prefix \\Mi).
Usage: python3 scripts/emit_math_interaction.py
"""
import glob
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
from harness import stats, tasks, topologies  # noqa: E402

AJ = os.path.join(ROOT, "results", "extra", "math_interaction_analysis.json")
SEEDS = [1, 2, 3]
COORD = ["DEBATE_N3_R1", "GE_R2", "HIER_M3"]
PAT = {"hard": "results/power/math500hard_qwen3b_s*.jsonl", "easy": "results/power/math500_qwen3b_s*.jsonl"}


def load(pat):  # as scripts/analyze_hard_math.py: sorted paths, first occurrence wins
    out = {}
    for p in sorted(glob.glob(os.path.join(ROOT, pat))):
        for line in open(p):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            out.setdefault((r["config"], r["item_id"], r["seed"]), r)
    return out


def boot(args):
    """Bootstrap distribution of a - b (percentage points), the draws of
    harness.stats.hier_paired_test with the given seed."""
    a, b, seed = args
    items = sorted(set(a) & set(b))
    A_, B_ = [a[i] for i in items], [b[i] for i in items]
    rng, n, out = np.random.default_rng(seed), len(items), np.empty(10000)
    for t in range(10000):
        s = 0.0
        for ii in rng.integers(0, n, size=n):
            av, bv = A_[ii], B_[ii]
            s += av[rng.integers(0, len(av), size=len(av))].mean() - bv[rng.integers(0, len(bv), size=len(bv))].mean()
        out[t] = s / n
    return out * 100


def ptest(args):
    a, b = args
    return stats.hier_paired_test(a, b, B=10000, seed=0)


def stratum(name, A):
    R = load(PAT[name])
    score = tasks.get_task("math500")["score"]
    ids = sorted({k[1] for k in R if k[0] == "SA"})
    assert len(ids) == A[name]["n_items"], name
    for i in ids:  # complete: SA one greedy seed, the rest three seeds
        assert ("SA", i, 1) in R and all((c, i, s) in R for c in ["SC_k12"] + COORD for s in SEEDS), (name, i)
    assert sorted({R[("SC_k12", i, 1)]["cap"] for i in ids}) == A[name]["cap"], name
    M = {"SA": {i: np.array([float(bool(R[("SA", i, 1)]["correct"]))] * 3) for i in ids}}
    for c in COORD:
        M[c] = {i: np.array([float(bool(R[(c, i, s)]["correct"])) for s in SEEDS]) for i in ids}
    m = {}
    for i in ids:
        v = []
        for s in SEEDS:
            r = R[("SC_k12", i, s)]
            assert bool(score(topologies._majority([c["extracted"] for c in r["calls"]]), r["gold"])) == bool(r["correct"])
            v.append(float(bool(score(topologies._majority([c["extracted"] for c in r["calls"][:8]]), r["gold"]))))
        m[i] = np.array(v)
    M["SC_k8"] = m
    acc = {k: 100 * float(np.mean([v[i].mean() for i in ids])) for k, v in M.items()}
    for k, v in A[name]["accuracy"].items():
        assert abs(acc[k] - v) < 0.006, (name, k, acc[k], v)
    return ids, M, acc


def main():
    A = json.load(open(AJ))
    hid, Mh, acch = stratum("hard", A)
    eid, Me, acce = stratum("easy", A)
    # strata: all of MATH-500, levels 4 to 5 and levels 1 to 3, disjoint
    assert not set(hid) & set(eid) and len(hid) + len(eid) == 500
    assert sorted(eid) == sorted(it["id"] for it in tasks.get_task("math500")["load"](n=238, split="all"))
    assert A["hard"]["cap"] == [2048] and A["easy"]["cap"] == [1024]
    # hard-stratum accuracies equal those in the paper's Table 13 (hard_math_analysis.json)
    H = json.load(open(os.path.join(ROOT, "results", "extra", "hard_math_analysis.json")))
    for k in ["SA", "SC_k8"] + COORD:
        assert abs(acch[k] - H["accuracy"][k]) < 0.006, k

    IX, WS = A["interaction_hard_minus_easy"], A["within_stratum"]
    obs = {w: (acch[w] - acch["SC_k8"]) - (acce[w] - acce["SC_k8"]) for w in COORD}
    for w in COORD:  # as the analysis script: from the 2-decimal accuracies, and unrounded
        a2h, a2e = A["hard"]["accuracy"], A["easy"]["accuracy"]
        assert round((a2h[w] - a2h["SC_k8"]) - (a2e[w] - a2e["SC_k8"]), 2) == IX[w]["obs"], w
        assert abs(obs[w] - IX[w]["obs"]) < 0.011, w

    jobs_b = [(Mh[w], Mh["SC_k8"], 0) for w in COORD] + [(Me[w], Me["SC_k8"], 1) for w in COORD]
    jobs_t = [(Me[w], Me["SC_k8"]) for w in COORD] + [(Me["SC_k8"], Me["SA"])]
    with Pool(min(len(jobs_b) + len(jobs_t), max(1, (os.cpu_count() or 2) - 1))) as pool:
        rb = pool.map_async(boot, jobs_b)
        rt = pool.map_async(ptest, jobs_t)
        rb, rt = rb.get(), rt.get()
    ix, es = {}, {}
    for j, w in enumerate(COORD):
        d = rb[j] - rb[j + 3]
        lo, hi = np.percentile(d, [2.5, 97.5])
        p = float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))
        assert [round(float(lo), 2), round(float(hi), 2)] == IX[w]["ci95"] and p == IX[w]["p"], w
        assert float((d <= 0).mean()) == IX[w]["p_one_sided_D_le_0"], w
        ix[w] = {"obs": obs[w], "ci95": [float(lo), float(hi)], "p": p}
        # the hard-stratum draws are those of the Table 13 test of W minus SC k=8
        assert [round(float(x), 2) for x in np.percentile(rb[j], [2.5, 97.5])] == H["vs_SC_k8"][w]["ci95"], w
    for (dd, p, lo, hi), key in zip(rt, [("w", w) for w in COORD] + [("sc", None)]):
        t = WS[key[1]]["easy_minus_SC_k8"] if key[0] == "w" else A["easy_SC_k8_minus_SA"]
        assert round(dd, 2) == t["diff"] and p == t["p"] and [round(lo, 2), round(hi, 2)] == t["ci95"], key
        es[key[1] or "SC_k8_minus_SA"] = {"diff": dd, "p": p, "ci95": [lo, hi]}

    # statements the text makes
    assert ix["DEBATE_N3_R1"]["obs"] < 0  # opposite in sign to the prediction D > 0
    assert all(ix[w]["ci95"][0] < 0 < ix[w]["ci95"][1] for w in COORD)  # every interval includes zero
    assert all(ix[w]["obs"] < 0 for w in COORD)
    assert all(es[w]["ci95"][1] < 0 for w in COORD)  # each below SC k=8 on levels 1 to 3

    def f1(x, sign=True):
        x = round(x, 1) + 0.0
        return f"{'+' if sign and x > 0 else ''}{x:.1f}"

    def pt(p):  # for running text, as emit_hard_math.py
        return "p{<}0.001" if p < 0.001 else f"p={p:.3f}" if p < 0.1 else f"p={p:.2f}"

    MM = ["% AUTO-GENERATED by paper/arxiv_long/scripts/emit_math_interaction.py. Do not edit."]

    def add(k, v):
        MM.append(f"\\newcommand{{\\Mi{k}}}{{{v}}}")
    add("NEasy", len(eid))
    add("NHard", len(hid))
    add("CapEasy", "1{,}024")
    add("CapHard", "2{,}048")
    KEY = {"SA": "Sa", "SC_k8": "Sceight", "DEBATE_N3_R1": "Deb", "GE_R2": "Ge", "HIER_M3": "Hier"}
    for k, key in KEY.items():
        add(f"AccEasy{key}", f1(acce[k], False))
    for w in COORD:
        key = KEY[w]
        t = ix[w]
        add(f"D{key}", f"${f1(t['obs'])}$")
        add(f"I{key}", f"$[{f1(t['ci95'][0])},{f1(t['ci95'][1])}]$")
        add(f"P{key}", pt(t["p"]))
        t = es[w]
        add(f"DEasy{key}", f"${f1(t['diff'])}$")
        add(f"IEasy{key}", f"$[{f1(t['ci95'][0])},{f1(t['ci95'][1])}]$")
        add(f"PEasy{key}", pt(t["p"]))
    out = os.path.join(HERE, "..", "math_interaction_macros.tex")
    open(out, "w").write("\n".join(MM) + "\n")
    print("\n".join(MM))


if __name__ == "__main__":
    main()
