"""Numbers added in the review-driven revision of the long arXiv version.

Three analyses, all from results/*.jsonl with harness/stats.py (no model calls):

1. Cost-model verdicts. For debate, GE and HIER on GSM8K and MATH (1024 cap)
   under total tokens, output tokens, price-weighted tokens in + r * out
   (r = 2..5; r = 1 is total tokens), and serial wall-clock time (GSM8K only;
   MATH serial times are not used, as in the poster), and on HumanEval under
   total tokens and GPU energy. The rule is the poster's (paper/aaibd,
   Sec. II): a sampling configuration S is "cheaper" if its mean cost is lower;
   R if some cheaper S has a 95% hierarchical accuracy interval (S minus C)
   above zero, C if no R and some cheaper S has an interval containing zero,
   N otherwise. Accuracy intervals: stats.hier_paired_test, 10,000 replicates,
   seed 0 (the same call as the poster, so the total-token column reproduces
   the poster's Table II). MATH sampling includes SC k=2,4 from the November
   1024-cap rerun (math500_trunc_sc24.jsonl), as in the poster's
   \\MathRerunCompareSentence.
2. Holm. (a) The declared family: SC k=8 vs the three coordination workflows
   per task (on HumanEval also SA vs the three). (b) A wider family: every
   cheaper-sampling comparison that the total-token verdicts of a task scan.
   For each R verdict on total tokens, whether it stands under each.
3. Detectable difference. Half-widths of the hierarchical intervals of every
   sampling vs coordination comparison per task; minimum detectable difference
   at 95% (= half-width) and at 80% power (half-width x (1.96 + 0.84) / 1.96).

Writes tables/review_revision.json, tables/cost_verdict_table.tex and
review_revision_macros.tex. Usage: python3 scripts/emit_review_revision.py
[--render] [--workers N]
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
COORD = ["DEBATE_N3_R1", "GE_R2", "HIER_M3"]
RATIOS = [2, 3, 4, 5]
TASKS = {
    "gsm8k": {"files": ["gsm8k_matrix.jsonl"],
              "sampling": ["SA", "SC_k2", "SC_k4", "SC_k8", "SC_k12"],
              "costs": ["total_tokens", "out_tokens"] + [f"pw{r}" for r in RATIOS] + ["latency_s"]},
    "math": {"files": ["math500_trunc.jsonl", "math500_trunc_sc24.jsonl"],
             "sampling": ["SA", "SC_k2", "SC_k4", "SC_k8", "SC_k12"],
             "costs": ["total_tokens", "out_tokens"] + [f"pw{r}" for r in RATIOS]},
    "mmlu": {"files": ["mmlu_matrix.jsonl"],
             "sampling": ["SA", "SC_k2", "SC_k4", "SC_k8"], "costs": ["total_tokens"]},
    "gsm8k200": {"files": ["gsm8k_scale200.jsonl"], "sampling": ["SA", "SC_k8"],
                 "coord": ["DEBATE_N3_R1"], "costs": ["total_tokens"]},
    "proeng": {"files": ["mmlu_pro_eng_matrix.jsonl"],
               "sampling": ["SA", "SC_k2", "SC_k4", "SC_k8"], "costs": ["total_tokens"]},
    "code": {"files": ["humaneval_matrix_pc.jsonl"],
             "sampling": ["SA", "SC_k2", "SC_k4", "SC_k8"],
             "costs": ["total_tokens", "energy_j"]},
}
Z80 = (1.959964 + 0.841621) / 1.959964

_rows = {}


def rows(task):
    if task not in _rows:
        rs = []
        for f in TASKS[task]["files"]:
            rs += [json.loads(l) for l in open(os.path.join(ROOT, "results", f)) if l.strip()]
        for r in rs:
            for k in RATIOS:
                r[f"pw{k}"] = r["in_tokens"] + k * r["out_tokens"]
        _rows[task] = rs
    return _rows[task]


def coords(task):
    return TASKS[task].get("coord", COORD)


def job(args):
    t, a, b = args
    cor = S.per_item_seed_correct(rows(t))
    d, p, lo, hi = S.hier_paired_test(cor[a], cor[b], B=B, seed=SEED)
    return f"{t}|{a}|{b}", {"diff": d, "p": p, "lo": lo, "hi": hi}


def mean_cost(t, c, field):
    m = S.per_item_seed_tokens(rows(t), field)[c]
    return float(np.mean([v.mean() for v in m.values()]))


def mean_acc(t, c):
    m = S.per_item_seed_correct(rows(t))[c]
    return 100 * float(np.mean([v.mean() for v in m.values()]))


def cost_ci(t, a, b, field):
    m = S.per_item_seed_tokens(rows(t), field)
    items = sorted(set(m[a]) & set(m[b]))
    d = np.array([m[a][i].mean() - m[b][i].mean() for i in items])
    rng = np.random.default_rng(SEED)
    boot = d[rng.integers(0, len(d), size=(B, len(d)))].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def compute(workers):
    jobs = [(t, s, c) for t in TASKS for s in TASKS[t]["sampling"] for c in coords(t)]
    jobs.sort(key=lambda j: -len(rows(j[0])))
    with Pool(workers) as pool:
        tests = dict(pool.map(job, jobs, chunksize=1))
    out = {"meta": {"B": B, "seed": SEED, "rule": __doc__.split("\n\n")[1]}, "tasks": {}}
    for t, spec in TASKS.items():
        T = {"configs": {}, "tests": {}, "verdicts": {}}
        for c in spec["sampling"] + coords(t):
            T["configs"][c] = {"acc": mean_acc(t, c),
                               **{f: mean_cost(t, c, f) for f in spec["costs"]}}
        for s in spec["sampling"]:
            for c in coords(t):
                T["tests"][f"{s}|{c}"] = tests[f"{t}|{s}|{c}"]
        for f in spec["costs"]:
            T["verdicts"][f] = {}
            for c in coords(t):
                cheaper = [s for s in spec["sampling"]
                           if T["configs"][s][f] < T["configs"][c][f]]
                cheaper.sort(key=lambda s: T["configs"][s][f])
                rob = [s for s in cheaper if T["tests"][f"{s}|{c}"]["lo"] > 0]
                ind = [s for s in cheaper if T["tests"][f"{s}|{c}"]["lo"] <= 0
                       <= T["tests"][f"{s}|{c}"]["hi"]]
                v = "R" if rob else ("C" if ind else "N")
                e = {"verdict": v, "cheaper": cheaper, "robust_by": rob}
                if rob:
                    e["cost_ci_first_robust"] = cost_ci(t, rob[0], c, f)
                T["verdicts"][f][c] = e
        # Holm (a): declared family
        fam = {"SC_k8": [T["tests"][f"SC_k8|{c}"]["p"] for c in coords(t)]}
        if t == "code":
            fam["SA"] = [T["tests"][f"SA|{c}"]["p"] for c in coords(t)]
        T["holm_declared"] = {b: dict(zip(coords(t), S.holm(ps))) for b, ps in fam.items()}
        # Holm (b): every cheaper comparison scanned by the total-token verdicts
        scan = [(s, c) for c in coords(t) for s in T["verdicts"]["total_tokens"][c]["cheaper"]]
        adj = S.holm([T["tests"][f"{s}|{c}"]["p"] for s, c in scan])
        T["holm_wide"] = {f"{s}|{c}": a for (s, c), a in zip(scan, adj)}
        T["holm_wide_m"] = len(scan)
        T["holm_check"] = {}
        for c in coords(t):
            e = T["verdicts"]["total_tokens"][c]
            if e["verdict"] != "R":
                continue
            in_fam = [b for b in T["holm_declared"] if b in e["robust_by"]]
            outside = [s for s in e["robust_by"] if s not in T["holm_declared"]]
            stands_a = bool(outside) or any(T["holm_declared"][b][c] < 0.05 for b in in_fam)
            stands_b = any(T["holm_wide"][f"{s}|{c}"] < 0.05 for s in e["robust_by"])
            T["holm_check"][c] = {"robust_by": e["robust_by"], "stands_declared": stands_a,
                                  "decided_outside_family": outside, "stands_wide": stands_b,
                                  "wide_p": {s: T["holm_wide"][f"{s}|{c}"] for s in e["robust_by"]}}
        hw = [(v["hi"] - v["lo"]) / 2 for v in T["tests"].values()]
        T["halfwidth"] = {"median": float(np.median(hw)), "min": float(min(hw)),
                          "max": float(max(hw)), "n_tests": len(hw),
                          "mde80_median": float(np.median(hw)) * Z80}
        out["tasks"][t] = T
    json.dump(out, open(os.path.join(HERE, "..", "tables", "review_revision.json"), "w"), indent=1)
    return out


LBL = {"SA": "SA", "SC_k2": "SC$_2$", "SC_k4": "SC$_4$", "SC_k8": "SC$_8$",
       "SC_k12": "SC$_{12}$"}
CN = {"DEBATE_N3_R1": "Debate", "GE_R2": "GE", "HIER_M3": "HIER"}


def cell(T, f, c):
    if f not in T["verdicts"]:
        return "--"
    e = T["verdicts"][f][c]
    if e["verdict"] != "R":
        return e["verdict"]
    s = e["robust_by"][0]
    dag = "$^\\dagger$" if e["cost_ci_first_robust"][2] >= 0 else ""
    return f"R{dag} {{\\scriptsize {LBL[s]}}}"


def price_cell(T, c):
    vs = {r: T["verdicts"][f"pw{r}"][c] for r in RATIOS}
    if len({(v["verdict"], tuple(v["robust_by"][:1])) for v in vs.values()}) == 1:
        return cell(T, f"pw{RATIOS[0]}", c)
    parts, prev = [], None
    for r in RATIOS:
        x = cell(T, f"pw{r}", c)
        if x != prev:
            parts.append(f"{x} ($r{{=}}{r}$)" if prev is None else f"{x} ($r{{\\ge}}{r}$)")
            prev = x
    return ", ".join(parts)


def render(out):
    hdr = "% AUTO-GENERATED by paper/arxiv_long/scripts/emit_review_revision.py. Do not edit."
    L = [hdr]
    names = {"gsm8k": "GSM8K", "math": "MATH", "code": "HumanEval"}
    for t in ("gsm8k", "math", "code"):
        T = out["tasks"][t]
        for j, c in enumerate(COORD):
            row = [names[t] if j == 0 else "", CN[c], cell(T, "total_tokens", c),
                   cell(T, "out_tokens", c),
                   price_cell(T, c) if "pw2" in T["verdicts"] else "--",
                   cell(T, "latency_s", c), cell(T, "energy_j", c)]
            L.append(" & ".join(row) + " \\\\")
        if t != "code":
            L.append("\\midrule")
    open(os.path.join(HERE, "..", "tables", "cost_verdict_table.tex"), "w").write("\n".join(L) + "\n")

    M = [hdr]
    key = {"gsm8k": "Gsm", "math": "Math", "mmlu": "Mmlu", "gsm8k200": "Gsmtwo",
           "proeng": "Eng", "code": "Code"}
    n_r, n_a, n_b, chg_a, chg_b = 0, 0, 0, [], []
    for t, T in out["tasks"].items():
        h = T["halfwidth"]
        M.append(f"\\newcommand{{\\RvHw{key[t]}}}{{{h['median']:.1f}}}")
        M.append(f"\\newcommand{{\\RvHwLo{key[t]}}}{{{h['min']:.1f}}}")
        M.append(f"\\newcommand{{\\RvHwHi{key[t]}}}{{{h['max']:.1f}}}")
        M.append(f"\\newcommand{{\\RvMde{key[t]}}}{{{h['mde80_median']:.0f}}}")
        M.append(f"\\newcommand{{\\RvHolmM{key[t]}}}{{{T['holm_wide_m']}}}")
        for c, e in T["holm_check"].items():
            n_r += 1
            n_a += e["stands_declared"]
            n_b += e["stands_wide"]
            if not e["stands_declared"]:
                chg_a.append(f"{t}:{c}")
            if not e["stands_wide"]:
                chg_b.append(f"{t}:{c}")
    M.append(f"\\newcommand{{\\RvNR}}{{{n_r}}}")
    M.append(f"\\newcommand{{\\RvNRstandDeclared}}{{{n_a}}}")
    M.append(f"\\newcommand{{\\RvNRstandWide}}{{{n_b}}}")
    M.append(f"% changed under declared family: {chg_a}")
    M.append(f"% changed under wide family: {chg_b}")
    num = {0: "none", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
           7: "seven", 8: "eight", 9: "nine", 10: "ten"}
    M.append(f"\\newcommand{{\\RvNRchangeDeclaredWord}}{{{num[n_r - n_a]}}}")
    M.append(f"\\newcommand{{\\RvNRchangeWideWord}}{{{num[n_r - n_b]}}}")
    # R verdicts on GSM8K and MATH under total tokens, and how they move under
    # output tokens; N verdicts under any cost model of any task
    tn = {"gsm8k": "GSM8K", "math": "MATH", "gsm8k200": "the 200-item run",
          "proeng": "MMLU-Pro engineering", "code": "HumanEval"}
    cw = {"DEBATE_N3_R1": "debate", "GE_R2": "GE", "HIER_M3": "HIER"}
    r_tot, r_out, n_any = [], [], []
    for t in ("gsm8k", "math"):
        V = out["tasks"][t]["verdicts"]
        for c in COORD:
            if V["total_tokens"][c]["verdict"] == "R":
                r_tot.append((t, c))
                if V["out_tokens"][c]["verdict"] == "R":
                    r_out.append((t, c))
    for t, T in out["tasks"].items():
        for f, V in T["verdicts"].items():
            for c, e in V.items():
                if e["verdict"] == "N":
                    n_any.append((t, f, c))
    M.append(f"\\newcommand{{\\RvMathRtot}}{{{num[len(r_tot)]}}}")
    M.append(f"\\newcommand{{\\RvMathRout}}{{{num[len(r_out)]}}}")
    M.append(f"\\newcommand{{\\RvMathRweaken}}{{{num[len(r_tot) - len(r_out)]}}}")
    M.append(f"% R under total: {r_tot}; still R under output: {r_out}; N anywhere: {n_any}")
    M.append(f"\\newcommand{{\\RvNanyWord}}{{{num[len(n_any)]}}}")
    fails = [f"{cw[k.split(':')[1]]} on {tn[k.split(':')[0]]}" for k in chg_b]
    txt = ", ".join(fails[:-1]) + (" and " if len(fails) > 1 else "") + fails[-1] if fails else "none"
    M.append(f"\\newcommand{{\\RvHolmWideFails}}{{{txt}}}")
    # specific comparisons quoted in the design paragraph
    g = out["tasks"]["gsm8k"]["tests"]
    for k, nm in (("SC_k12|DEBATE_N3_R1", "GsmSctwelveDeb"), ("SC_k8|GE_R2", "GsmSceightGe"),
                  ("SC_k4|HIER_M3", "GsmScfourHier")):
        M.append(f"\\newcommand{{\\RvHw{nm}}}{{{(g[k]['hi'] - g[k]['lo']) / 2:.1f}}}")
    open(os.path.join(HERE, "..", "review_revision_macros.tex"), "w").write("\n".join(M) + "\n")


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--render" in args:
        render(json.load(open(os.path.join(HERE, "..", "tables", "review_revision.json"))))
    else:
        w = int(args[args.index("--workers") + 1]) if "--workers" in args else 6
        render(compute(w))
