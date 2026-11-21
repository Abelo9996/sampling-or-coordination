"""Bounds on the accuracy gain of coordination over sampling (re-review revision,
comparator and multiplicity fixed in the third-review revision).

From results/*.jsonl only (no model calls). For each task and each coordination
workflow (debate, GE, HIER; the 200-item GSM8K run has debate only), the paired
difference coordination minus a sampling comparator, with the seed-aware
hierarchical bootstrap of harness/stats.hier_paired_test (10,000 replicates,
seed 0). Two comparators:
  (a) SC k=8 on every task, fixed without reference to accuracy. SC k=8 was run
      on every task, so every row has it (HumanEval included; the earlier
      version used the single call there, the most accurate configuration,
      which was a choice made after seeing accuracy);
  (b) picked after seeing accuracy: the most accurate sampling configuration
      (SA or SC k) whose mean total tokens are no more than the workflow's (ties
      in accuracy: the cheaper one). Reported for reference only.
For each comparison: point estimate, 95% interval (equal to the paper's), 90%
two-sided interval, and its upper limit (the one-sided 95% upper bound on the
gain). One-sided bootstrap p values from the same distribution: p0 for the null
"gain >= 0" (share of replicates with coordination minus comparator >= 0) and
p5 for "gain >= 5 points"; the upper limit is below 0 (below 5) exactly when
p0 (p5) is below 0.05, up to ties at the percentile. Holm-adjusted p0 and p5
over the 16 comparator-(a) rows (one family each). Margin: 5 points, chosen
after seeing the data. Verdict at the margin, unadjusted: "equiv" if the whole
90% interval lies inside (-5, +5) (two one-sided tests at the 5% level);
"upper" if only the upper limit is below +5; "neither" otherwise.

hier_paired_test returns only the 95% interval, so this script repeats its
resampling loop call for call (same generator, same draw order), keeps the
bootstrap distribution, and checks that its 95% interval and p equal the values
emit_review_revision.py stored from hier_paired_test. The resampled statistic is
comparator minus coordination, exactly as in the paper's tests; the coordination
minus comparator interval is its mirror image, so every 95% interval here is the
paper's interval with the sign flipped.

Also: the verdict rule under output tokens on every task, and a table of those
verdicts and costs for MMLU, MMLU-Pro engineering, HumanEval and the 200-item
GSM8K run (GSM8K and MATH are in Table costverdict).

Writes tables/equivalence.json, tables/equivalence_table.tex,
tables/output_verdict_table.tex and equivalence_macros.tex.
Usage: python3 scripts/emit_equivalence.py [--render] [--workers N]
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import emit_review_revision as RR  # noqa: E402
from emit_review_revision import S, B, SEED  # noqa: E402

MARGIN = 5.0
ALPHA = 0.05
ORDER = ["gsm8k", "math", "mmlu", "gsm8k200", "proeng", "code"]
MAIN = {t: "SC_k8" for t in ORDER}
ONE_SEED = {("math", "SC_k12")}  # sampling runs with a single generation seed


def hier_boot(a, b, B=B, seed=SEED):
    """The loop of stats.hier_paired_test, unchanged in its random draws, also
    returning the bootstrap distribution (fractions, a minus b)."""
    items = sorted(set(a) & set(b))
    A = [a[i] for i in items]
    Bv = [b[i] for i in items]
    obs = float(np.mean([A[j].mean() - Bv[j].mean() for j in range(len(items))]))
    rng = np.random.default_rng(seed)
    n = len(items)
    boot = np.empty(B)
    for t in range(B):
        idx = rng.integers(0, n, size=n)
        s = 0.0
        for ii in idx:
            av, bv = A[ii], Bv[ii]
            ai = av[rng.integers(0, len(av), size=len(av))].mean()
            bi = bv[rng.integers(0, len(bv), size=len(bv))].mean()
            s += ai - bi
        boot[t] = s / n
    p = 2.0 * min((boot <= 0).mean(), (boot >= 0).mean())
    return obs, min(p, 1.0), boot


def job(args):
    t, s, c = args
    cor = S.per_item_seed_correct(RR.rows(t))
    obs, p, boot = hier_boot(cor[s], cor[c])
    lo95, hi95, lo90, hi90 = np.percentile(boot, [2.5, 97.5, 5, 95]) * 100
    gain = np.round(-boot * 100, 9)  # coordination minus comparator, points
    return f"{t}|{s}|{c}", {"s_minus_c": obs * 100, "p": p,
                            "s_minus_c_95": [float(lo95), float(hi95)],
                            "s_minus_c_90": [float(lo90), float(hi90)],
                            "p_gain0": float((gain >= 0).mean()),
                            "p_gain5": float((gain >= MARGIN).mean())}


def verdict(lo, hi):
    if -MARGIN < lo and hi < MARGIN:
        return "equiv"
    if hi < MARGIN:
        return "upper"
    return "neither"


def compute(workers):
    prev = json.load(open(os.path.join(HERE, "..", "tables", "review_revision.json")))["tasks"]
    plan = []
    for t in ORDER:
        cfg = prev[t]["configs"]
        for c in RR.coords(t):
            assert MAIN[t] in cfg, (t, "SC k=8 missing")
            cand = [s for s in RR.TASKS[t]["sampling"]
                    if cfg[s]["total_tokens"] <= cfg[c]["total_tokens"]]
            best = sorted(cand, key=lambda s: (-cfg[s]["acc"], cfg[s]["total_tokens"]))[0]
            plan.append((t, c, MAIN[t], best))
    jobs = sorted({(t, s, c) for t, c, a, b in plan for s in (a, b)},
                  key=lambda j: -len(RR.rows(j[0])))
    with Pool(workers) as pool:
        res = dict(pool.map(job, jobs, chunksize=1))
    checks = []
    for k, v in res.items():
        t, s, c = k.split("|")
        ref = prev[t]["tests"][f"{s}|{c}"]
        ok = (abs(ref["lo"] - v["s_minus_c_95"][0]) < 1e-9 and abs(ref["hi"] - v["s_minus_c_95"][1]) < 1e-9
              and abs(ref["diff"] - v["s_minus_c"]) < 1e-9 and abs(ref["p"] - v["p"]) < 1e-12)
        checks.append({"test": k, "matches_hier_paired_test": bool(ok)})
    assert all(x["matches_hier_paired_test"] for x in checks), checks
    out = {"meta": {"B": B, "seed": SEED, "margin": MARGIN, "margin_chosen": "after seeing the data",
                    "alpha_one_sided": ALPHA, "holm_family": "the 16 SC k=8 rows, separately for p0 and p5",
                    "doc": __doc__.split("\n\n")[1], "checks": checks}, "rows": []}
    for t, c, a, b in plan:
        cfg = prev[t]["configs"]
        row = {"task": t, "coord": c, "coord_acc": cfg[c]["acc"],
               "coord_tokens": cfg[c]["total_tokens"], "comparators": {}}
        for role, s in (("main", a), ("best_cheaper", b)):
            v = res[f"{t}|{s}|{c}"]
            lo, hi = -v["s_minus_c_90"][1], -v["s_minus_c_90"][0]
            row["comparators"][role] = {
                "config": s, "acc": cfg[s]["acc"], "tokens": cfg[s]["total_tokens"],
                "one_seed": bool((t, s) in ONE_SEED),
                "est": -v["s_minus_c"], "ci95": [-v["s_minus_c_95"][1], -v["s_minus_c_95"][0]],
                "ci90": [lo, hi], "upper": hi, "verdict": verdict(lo, hi),
                "p_gain0": v["p_gain0"], "p_gain5": v["p_gain5"]}
        out["rows"].append(row)
    for role in ("main", "best_cheaper"):
        for pk in ("p_gain0", "p_gain5"):
            adj = S.holm([r["comparators"][role][pk] for r in out["rows"]])
            for r, a_ in zip(out["rows"], adj):
                r["comparators"][role][pk + "_holm"] = a_
    for r in out["rows"]:  # percentile bound and p agree (no row sits on a tie)
        for e in r["comparators"].values():
            assert (e["upper"] < 0) == (e["p_gain0"] < ALPHA), (r["task"], r["coord"], e)
            assert (e["upper"] < MARGIN) == (e["p_gain5"] < ALPHA), (r["task"], r["coord"], e)
    json.dump(out, open(os.path.join(HERE, "..", "tables", "equivalence.json"), "w"), indent=1)
    return out


TN = {"gsm8k": "GSM8K", "math": "MATH", "mmlu": "MMLU", "gsm8k200": "GSM8K-200",
      "proeng": "MMLU-Pro eng.", "code": "HumanEval"}
KEY = {"gsm8k": "Gsm", "math": "Math", "mmlu": "Mmlu", "gsm8k200": "Gsmtwo",
       "proeng": "Eng", "code": "Code"}
CK = {"DEBATE_N3_R1": "Deb", "GE_R2": "Ge", "HIER_M3": "Hier"}
VW = {"equiv": "equiv.", "upper": "upper", "neither": "neither"}
NUMW = {0: "none", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
        7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve",
        13: "thirteen", 14: "fourteen", 15: "fifteen", 16: "sixteen"}


def f1(x):
    x = round(x, 1) + 0.0
    return f"${'+' if x > 0 else ''}{x:.1f}$"


def iv(lo, hi):
    return f"$[{f1(lo).strip('$')},{f1(hi).strip('$')}]$"


def pv(p):
    return "${<}0.001$" if p < 0.001 else f"${p:.3f}$" if p < 0.1 else f"${p:.2f}$"


def comma(x):
    return f"{x:,.0f}".replace(",", "{,}")


def names(lst):
    """['GSM8K HIER', ...] -> 'A, B and C'."""
    return lst[0] if len(lst) == 1 else ", ".join(lst[:-1]) + " and " + lst[-1]


def render(out):
    hdr = "% AUTO-GENERATED by paper/arxiv_long/scripts/emit_equivalence.py. Do not edit."
    L = [hdr]
    prev_t = None
    for r in out["rows"]:
        t, c = r["task"], r["coord"]
        if prev_t is not None and t != prev_t:
            L.append("\\midrule")
        m, b = r["comparators"]["main"], r["comparators"]["best_cheaper"]
        assert m["config"] == "SC_k8"
        cells = [TN[t] if t != prev_t else "", RR.CN[c],
                 f1(m["est"]), iv(*m["ci90"]), pv(m["p_gain0"]), pv(m["p_gain0_holm"]),
                 pv(m["p_gain5_holm"])]
        if b["config"] == m["config"]:
            cells += ["same", "", "", ""]
        else:
            dag = "$^\\ddagger$" if b["one_seed"] else ""
            cells += [RR.LBL[b["config"]] + dag, f1(b["est"]), iv(*b["ci90"]), VW[b["verdict"]]]
        L.append(" & ".join(cells) + " \\\\")
        prev_t = t
    open(os.path.join(HERE, "..", "tables", "equivalence_table.tex"), "w").write("\n".join(L) + "\n")

    M = [hdr, f"\\newcommand{{\\EqMargin}}{{{MARGIN:.0f}}}"]
    cnt = {"main": {"equiv": 0, "upper": 0, "neither": 0},
           "best_cheaper": {"equiv": 0, "upper": 0, "neither": 0}}
    per_task_max = {}
    for r in out["rows"]:
        t, c = r["task"], r["coord"]
        for role, tag in (("main", "Main"), ("best_cheaper", "Best")):
            e = r["comparators"][role]
            cnt[role][e["verdict"]] += 1
            p = f"\\Eq{tag}{KEY[t]}{CK[c]}"
            M.append(f"\\newcommand{{{p}Up}}{{${e['upper']:.1f}$}}")
            M.append(f"\\newcommand{{{p}Est}}{{{f1(e['est']).strip('$')}}}")
            M.append(f"\\newcommand{{{p}Ci}}{{{iv(*e['ci90'])}}}")
            M.append(f"\\newcommand{{{p}Cfg}}{{{RR.LBL[e['config']]}}}")
            if role == "main":
                M.append(f"\\newcommand{{{p}Pz}}{{{pv(e['p_gain0'])}}}")
                M.append(f"\\newcommand{{{p}PzHolm}}{{{pv(e['p_gain0_holm'])}}}")
        for role in ("main", "best_cheaper"):
            per_task_max.setdefault((role, t), []).append(r["comparators"][role]["upper"])
    for (role, t), ups in per_task_max.items():
        tag = "Main" if role == "main" else "Best"
        M.append(f"\\newcommand{{\\Eq{tag}MaxUp{KEY[t]}}}{{${max(ups):.1f}$}}")
    # which SC k=8 rows exclude a gain (upper limit < 0), unadjusted and after Holm
    main = [(r["task"], r["coord"], r["comparators"]["main"]) for r in out["rows"]]
    lab = lambda t, c: f"{TN[t]} {RR.CN[c] if c != 'DEBATE_N3_R1' else 'debate'}"  # noqa: E731
    raw0 = [lab(t, c) for t, c, e in main if e["p_gain0"] < ALPHA]
    holm0 = [lab(t, c) for t, c, e in main if e["p_gain0_holm"] < ALPHA]
    raw5 = [lab(t, c) for t, c, e in main if e["p_gain5"] < ALPHA]
    holm5 = [lab(t, c) for t, c, e in main if e["p_gain5_holm"] < ALPHA]
    out["summary"] = {"exclude_gain_raw": raw0, "exclude_gain_holm": holm0,
                      "exclude_gain5_raw": raw5, "exclude_gain5_holm": holm5,
                      "not_exclude_gain5_holm": [lab(t, c) for t, c, e in main
                                                 if e["p_gain5_holm"] >= ALPHA]}
    M.append(f"% exclude a gain, unadjusted: {raw0}")
    M.append(f"% exclude a gain, Holm over 16: {holm0}")
    M.append(f"% exclude a gain of 5 points, Holm over 16: {holm5}")
    M.append(f"\\newcommand{{\\EqNMainExclRaw}}{{{NUMW[len(raw0)]}}}")
    M.append(f"\\newcommand{{\\EqNMainExclHolm}}{{{NUMW[len(holm0)]}}}")
    M.append(f"\\newcommand{{\\EqMainExclHolmList}}{{{names(holm0)}}}")
    M.append(f"\\newcommand{{\\EqNMainUpperRaw}}{{{NUMW[len(raw5)]}}}")
    M.append(f"\\newcommand{{\\EqNMainUpperHolm}}{{{NUMW[len(holm5)]}}}")
    # the text names the unadjusted rows in words
    assert sorted(raw0) == sorted([lab(t, c) for t, c, _ in main if t in ("gsm8k", "math", "code")]
                                  + ["GSM8K-200 debate", "MMLU-Pro eng. debate"]), raw0
    below = [lab(t, c) for t, c, e in main if e["ci90"][1] < -MARGIN]
    M.append(f"\\newcommand{{\\EqMainBelowMarginList}}{{{names(below)}}}")
    nh5 = out["summary"]["not_exclude_gain5_holm"]
    M.append(f"\\newcommand{{\\EqMainNotUpperHolmList}}{{{names(nh5) if nh5 else 'none'}}}")
    # side facts quoted with the bounds (no bootstrap: stored tests and record means)
    prev = json.load(open(os.path.join(HERE, "..", "tables", "review_revision.json")))["tasks"]
    for t in ("gsm8k", "math", "proeng"):
        M.append(f"\\newcommand{{\\EqAccSa{KEY[t]}}}{{{prev[t]['configs']['SA']['acc']:.1f}}}")
    allsa = [prev[t]["configs"]["SA"]["acc"] for t in ("gsm8k", "math", "mmlu", "proeng", "code")]
    M.append(f"\\newcommand{{\\EqAccSaMinAll}}{{{min(allsa):.0f}}}")
    M.append(f"\\newcommand{{\\EqAccSaMaxAll}}{{{max(allsa):.0f}}}")
    for c in RR.COORD:  # MMLU-Pro engineering: coordination minus the single call
        M.append(f"\\newcommand{{\\EqEng{CK[c]}MinusSa}}{{{f1(-prev['proeng']['tests'][f'SA|{c}']['diff']).strip('$')}}}")
    # number of cheaper sampling comparators per workflow under total tokens (the
    # verdict rule needs one of them for R and all of them for N)
    ncmp = {t: [len(prev[t]["verdicts"]["total_tokens"][c]["cheaper"]) for c in RR.coords(t)]
            for t in ORDER}
    seq = lambda v: v[0] if len(v) == 1 else ", ".join(map(str, v[:-1])) + " and " + str(v[-1])  # noqa: E731
    assert ncmp["gsm8k"] == ncmp["math"]
    M.append("\\newcommand{\\EqNCmpSentence}{" +
             f"{seq(ncmp['gsm8k'])} for debate, GE and HIER on GSM8K and on MATH, "
             f"{seq(ncmp['mmlu'])} on MMLU, {seq(ncmp['proeng'])} on MMLU-Pro engineering, "
             f"{seq(ncmp['code'])} on HumanEval, and {ncmp['gsm8k200'][0]} for debate in the "
             "200-item run}")
    M.append(f"\\newcommand{{\\EqNCmpMin}}{{{min(min(v) for v in ncmp.values())}}}")
    M.append(f"\\newcommand{{\\EqNCmpMax}}{{{max(max(v) for v in ncmp.values())}}}")
    # held-out identity test: pooled net change on the cells blind to the protocol
    # (stored by scripts/criterion_predictive.py, cell-stratified item bootstrap)
    cp = json.load(open(os.path.join(RR.ROOT, "results", "analysis", "criterion_predictive.json")))
    nb = cp["pooled"]["new_cells"]
    assert len(nb["cells"]) == 6
    M.append(f"\\newcommand{{\\EqCpBlindAllCells}}{{{NUMW[len(nb['cells'])]}}}")
    M.append(f"\\newcommand{{\\EqCpBlindAllDelta}}{{{f1(100 * nb['delta'])}, "
             f"{iv(100 * nb['delta_ci'][0], 100 * nb['delta_ci'][1])}}}")
    m8 = prev["math"]["tests"]["SC_k8|DEBATE_N3_R1"]
    M.append(f"\\newcommand{{\\EqMathScEightMinusDeb}}{{{f1(m8['diff'])}, {iv(m8['lo'], m8['hi'])}}}")
    # Verdict rule of the paper under output tokens on every task (Table costverdict
    # covers GSM8K, MATH and HumanEval in total tokens and GSM8K and MATH in output
    # tokens), and whether each workflow is detectably more accurate than the most
    # accurate sampling configuration that generates no more output tokens.
    ov = {}
    for t in ORDER:
        sp = RR.TASKS[t]["sampling"]
        oc = {c: RR.mean_cost(t, c, "out_tokens") for c in sp + RR.coords(t)}
        for c in RR.coords(t):
            T = prev[t]["tests"]
            ch = sorted([s for s in sp if oc[s] < oc[c]], key=lambda s: oc[s])
            rob = [s for s in ch if T[f"{s}|{c}"]["lo"] > 0]
            ind = [s for s in ch if T[f"{s}|{c}"]["lo"] <= 0 <= T[f"{s}|{c}"]["hi"]]
            le = [s for s in sp if oc[s] <= oc[c]]
            best = sorted(le, key=lambda s: (-prev[t]["configs"][s]["acc"], oc[s]))[0] if le else None
            ov[f"{t}|{c}"] = {"verdict": "R" if rob else ("C" if ind else "N"),
                              "cheaper": ch, "robust_by": rob, "out_tokens": oc,
                              "nearest_cheaper": ch[-1] if ch else None,
                              "best_le": best,
                              "coord_detectably_above_best_le": bool(best and T[f"{best}|{c}"]["hi"] < 0)}
    out["output_token_verdicts"] = ov
    json.dump(out, open(os.path.join(HERE, "..", "tables", "equivalence.json"), "w"), indent=1)
    # output-token verdict table for the tasks not in Table costverdict
    L = [hdr]
    prev_t = None
    for t in ("mmlu", "proeng", "code", "gsm8k200"):
        if prev_t is not None:
            L.append("\\midrule")
        for c in RR.coords(t):
            v = ov[f"{t}|{c}"]
            oc = v["out_tokens"]
            T = prev[t]["tests"]
            near = v["nearest_cheaper"]
            tn = T[f"{near}|{c}"]
            chs = ", ".join(f"{RR.LBL[s]}~{comma(oc[s])}" for s in v["cheaper"])
            vv = v["verdict"] + (f" {{\\scriptsize {RR.LBL[v['robust_by'][0]]}}}" if v["robust_by"] else "")
            L.append(" & ".join([TN[t] if t != prev_t else "", RR.CN[c], comma(oc[c]), chs,
                                 RR.LBL[near], f"{f1(tn['diff'])} {{\\scriptsize {iv(tn['lo'], tn['hi'])}}}",
                                 vv]) + " \\\\")
            prev_t = t
    open(os.path.join(HERE, "..", "tables", "output_verdict_table.tex"), "w").write("\n".join(L) + "\n")
    # the text states these verdicts in words
    assert all(ov[f"{t}|{c}"]["verdict"] == "C" for t in ("mmlu", "proeng") for c in RR.coords(t))
    assert all(ov[f"code|{c}"]["verdict"] == "R" for c in RR.coords("code"))
    assert ov["gsm8k200|DEBATE_N3_R1"]["verdict"] == "N"
    # coordination detectably more accurate than a sampling configuration that
    # generates fewer output tokens (the text says: only debate over SA, 200-item run)
    beats = [f"{k}>{s}" for k, v in ov.items() for s in v["cheaper"]
             if prev[k.split("|")[0]]["tests"][f"{s}|{k.split('|')[1]}"]["hi"] < 0]
    assert beats == ["gsm8k200|DEBATE_N3_R1>SA"], beats
    M.append(f"% coordination detectably above a cheaper-in-output sampling config: {beats}")
    nN = [k for k, v in ov.items() if v["verdict"] == "N"]
    M.append(f"% output-token N verdicts: {nN}")
    M.append(f"\\newcommand{{\\EqOutNWord}}{{{NUMW[len(nN)]}}}")
    g2 = ov["gsm8k200|DEBATE_N3_R1"]["out_tokens"]
    M.append(f"\\newcommand{{\\EqTwoOutDeb}}{{{g2['DEBATE_N3_R1']:,.0f}}}")
    M.append(f"\\newcommand{{\\EqTwoOutSa}}{{{g2['SA']:,.0f}}}")
    M.append(f"\\newcommand{{\\EqTwoOutSc}}{{{g2['SC_k8']:,.0f}}}")
    t2 = prev["gsm8k200"]["tests"]["SA|DEBATE_N3_R1"]
    M.append(f"\\newcommand{{\\EqTwoDebMinusSa}}{{{f1(-t2['diff'])}, {iv(-t2['hi'], -t2['lo'])}}}")
    # abstract lead: no workflow detectably above the best sampling at no higher total tokens
    assert all(r["comparators"]["best_cheaper"]["ci95"][0] <= 0 for r in out["rows"])
    # and no workflow detectably above SC k=8 (fixed comparator)
    assert all(r["comparators"]["main"]["ci95"][0] <= 0 for r in out["rows"])
    lows = [e["ci90"][0] for r in out["rows"] for e in r["comparators"].values()]
    assert max(lows) < -MARGIN, "text says every lower limit is below -margin"
    M.append(f"% largest 90% lower limit over all rows: {max(lows):.2f}")
    n = len(out["rows"])
    M.append(f"\\newcommand{{\\EqNrows}}{{{n}}}")
    M.append(f"\\newcommand{{\\EqNrowsWord}}{{{NUMW[n]}}}")
    for role, tag in (("main", "Main"), ("best_cheaper", "Best")):
        for v, w in (("equiv", "Equiv"), ("upper", "Upper"), ("neither", "Neither")):
            M.append(f"\\newcommand{{\\EqN{tag}{w}}}{{{NUMW[cnt[role][v]]}}}")
    open(os.path.join(HERE, "..", "equivalence_macros.tex"), "w").write("\n".join(M) + "\n")
    json.dump(out, open(os.path.join(HERE, "..", "tables", "equivalence.json"), "w"), indent=1)


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--render" in args:
        render(json.load(open(os.path.join(HERE, "..", "tables", "equivalence.json"))))
    else:
        w = int(args[args.index("--workers") + 1]) if "--workers" in args else 6
        render(compute(w))
