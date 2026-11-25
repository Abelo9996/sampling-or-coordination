"""Within-run check of every cross-run comparison in the EAAI rev3 manuscript (reviewer point 3).

A comparison is cross-run when the workflow's records sit in a log that holds no sampling record
(the rule of paper/arxiv_rev3/scripts/make_fig_forest.py, which marks these rows with a dagger):
  HIER (repaired, HIERFIX) on GSM8K, MMLU and MMLU-Pro engineering (results/rev3/*), and the
  Du et al. debate on MATH levels 4 to 5 (results/power/extra_math45_*).
The confirmatory difficulty test C5 also crosses runs (MATH L4-5 run against the MATH L1-3 rerun
at the 2,048 cap), although it is not daggered; it is checked here too.

For each such claim the script reports
  cross       the published cross-run statistic (read from results/extra/*.json, not recomputed);
  within      the same contrast computed only from records of one run, where one exists:
                HIER: the original HIER (HIER_M3), which ran in the same run as the sampling
                      configurations (same items, seeds, machine), against the same comparators;
                C5:   the within-run difficulty contrast of the MATH L4-5 run (level 5 minus level 4,
                      debate minus SC k=8), from hard_math_analysis.json;
                Du et al. debate on MATH L4-5: no run holds it together with sampling (uncheckable);
  drift       the run-to-run change of identical configurations on the same GPU (new paired tests:
              GSM8K repaired HIER, results/rev3 vs results/power/extra_gsm8k, 100 items x 3 seeds;
              GSM8K SA, SC k=8, debate, 200-item run vs 500-item run on the 200 shared items;
              HumanEval first run vs rerun, read from rev3_reruns_analysis.json) and across machines
              (GSM8K, Apple M4 July matrix vs RTX 4060 rerun, tables/audit.json drift80);
  margin      the distance from zero of the interval bound the claim rests on, compared with the
              largest same-GPU drift (or, for MMLU, whose arms ran on different machines, the
              largest cross-machine drift).

Outcome per claim: "survives" (within-run counterpart gives the same verdict, and where the claim is
cross-run the margin exceeds the relevant drift), "changes" (within-run counterpart, or the drift
bound, does not support the claim as worded), "uncheckable" (no within-run counterpart).

Tests: harness.stats.hier_paired_test through scripts/analyze_hard_math.test (10,000 replicates,
seed 0), as every other test of the paper.

Writes results/extra/within_run_check.json and, in each --outdir DIR (default paper/eaai_rev3/ if present,
else paper/arxiv_rev3/):
  within_run_macros.tex, tables/within_run_main.tex, tables/within_run_supp.tex,
  tables/within_run_drift.tex.
Usage: /usr/bin/python3 scripts/within_run_check.py [--reuse-drift] [--outdir DIR ...]
  (the arXiv build runs it with --reuse-drift --outdir paper/arxiv_rev3)
"""
import glob, importlib.util, json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from harness import tasks  # noqa: E402

spec = importlib.util.spec_from_file_location("hm", os.path.join(ROOT, "scripts/analyze_hard_math.py"))
hm = importlib.util.module_from_spec(spec); spec.loader.exec_module(hm)

EXTRA = os.path.join(ROOT, "results/extra")
# inputs tables/audit.json and tables/extra_workflows.json (identical copies in paper/eaai_rev3)
PAPER = os.path.join(ROOT, "paper/arxiv_rev3")
# paper directories that receive the macros and tables (--outdir DIR, repeatable); default
# paper/eaai_rev3 where it exists (the manuscript repository), else paper/arxiv_rev3 (release)
OUTDIRS = [os.path.join(ROOT, "paper/eaai_rev3") if os.path.isdir(os.path.join(ROOT, "paper/eaai_rev3")) else PAPER]
SAMPLING = {"SA", "SC_k2", "SC_k4", "SC_k8", "SC_k12"}


def load(pats):
    recs = {}
    for pat in pats:
        for p in sorted(glob.glob(os.path.join(ROOT, pat))):
            for l in open(p):
                try:
                    r = json.loads(l)
                except json.JSONDecodeError:
                    continue
                recs.setdefault((r["config"], r["item_id"], r["seed"]), r)
    return recs


def mat(recs, cfg, ids, seeds):
    return {i: np.array([float(bool(recs[(cfg, i, s)]["correct"])) for s in seeds]) for i in ids}


def cross_run_logs():
    """Logs of coordination records that hold no sampling record (the dagger rule)."""
    out = {}
    for name, pat in {"gsm8k HIER": "results/rev3/gsm8k_qwen3b_*.jsonl",
                      "mmlu HIER": "results/rev3/mmlu_qwen3b_*.jsonl",
                      "proeng HIER": "results/rev3/mmlu_pro_eng_qwen3b_*.jsonl",
                      "math45 Du": "results/power/extra_math45_qwen3b_s*.jsonl",
                      "gsm8k Du/HIER extra": "results/power/extra_gsm8k_qwen3b_s*.jsonl"}.items():
        cs = {k[0] for k in load([pat])}
        assert cs and not cs & SAMPLING, (name, cs)
        out[name] = sorted(cs)
    # and the reruns that do hold sampling records (within-run): HumanEval, MATH L1-3 at 2,048
    for pat in ("results/rev3/humaneval_qwen3b_*.jsonl", "results/rev3/math500_cap2048_qwen3b_*.jsonl"):
        assert {k[0] for k in load([pat])} & SAMPLING, pat
    return out


def drift_tests():
    """Run-to-run change of identical configurations (later run minus earlier run)."""
    D = {}
    # GSM8K repaired HIER: R1 (results/rev3) vs the published-debate run (results/power/extra_gsm8k)
    R1 = load(["results/rev3/gsm8k_qwen3b_s*.jsonl"])
    X = load(["results/power/extra_gsm8k_qwen3b_s*.jsonl"])
    xid = sorted({k[1] for k in X if k[0] == "HIERFIX_M3"})
    assert len(xid) == 100
    D["gsm8k_hierfix"] = {"task": "GSM8K", "config": "HIER", "runs": "R1 vs published-debate run",
                          "hw": "same GPU", "n": len(xid), "seeds": 3,
                          **hm.test(mat(X, "HIERFIX_M3", xid, [1, 2, 3]), mat(R1, "HIERFIX_M3", xid, [1, 2, 3]))}
    # GSM8K SA, SC k=8, debate: 500-item run (results/power) vs 200-item run, on the shared items
    PA = load(["results/power/gsm8k_qwen3b_s[123].jsonl", "results/power/gsm8k_qwen3b_s[123]_sh*of2.jsonl"])
    S2 = load(["results/gsm8k_scale200.jsonl"])
    sid = sorted({k[1] for k in S2})
    assert len(sid) == 200 and all(("SC_k8", i, 1) in PA for i in sid)
    for cfg, nm in (("SA", "SA"), ("SC_k8", "SC $k{=}8$"), ("DEBATE_N3_R1", "debate")):
        sa, sb = ([1], [1]) if cfg == "SA" else ([1, 2, 3], [1, 2])
        D[f"gsm8k_{cfg}"] = {"task": "GSM8K", "config": nm, "runs": "500-item run vs 200-item run",
                             "hw": "same GPU", "n": len(sid), "seeds": f"{len(sa)}/{len(sb)}",
                             **hm.test(mat(PA, cfg, sid, sa), mat(S2, cfg, sid, sb))}
    # HumanEval first run vs rerun (v1 extractor on both), already computed
    ra = json.load(open(os.path.join(EXTRA, "rev3_reruns_analysis.json")))
    for k, v in ra["code_old_v1_minus_R3_v1"].items():
        a, b = k.split("|")
        if a != b:  # HIER changed between the runs (original vs repaired); not an identical configuration
            continue
        nm = {"SA": "SA", "SC_k2": "SC $k{=}2$", "SC_k4": "SC $k{=}4$", "SC_k8": "SC $k{=}8$",
              "DEBATE_N3_R1": "debate", "GE_R2": "GE"}[a]
        # stored as first run minus rerun; report rerun minus first run, as the other rows
        D[f"code_{a}"] = {"task": "HumanEval", "config": nm, "runs": "rerun vs first run", "hw": "same GPU",
                          "n": 164, "seeds": 2, "diff": -v["diff"], "ci95": [-v["ci95"][1], -v["ci95"][0]], "p": v["p"]}
    # GSM8K across machines: Apple M4 July matrix vs RTX 4060 rerun, 80 items (tables/audit.json)
    A = json.load(open(os.path.join(PAPER, "tables/audit.json")))["drift80"]
    for cfg, nm in (("SA", "SA"), ("SC_k8", "SC $k{=}8$"), ("DEBATE_N3_R1", "debate"), ("GE_R2", "GE"),
                    ("HIER_M3", "HIER (original)")):
        D[f"m4_{cfg}"] = {"task": "GSM8K", "config": nm, "runs": "4060 rerun vs M4 matrix", "hw": "across machines",
                          "n": 80, "seeds": "", "diff": round(A[cfg]["rerun80"] - A[cfg]["matrix80"], 2),
                          "ci95": None, "p": None}
    return D


def main():
    if "--outdir" in sys.argv:
        OUTDIRS[:] = [os.path.abspath(sys.argv[i + 1]) for i, x in enumerate(sys.argv) if x == "--outdir"]
    J = {"cross_run_logs": cross_run_logs()}
    ra = json.load(open(os.path.join(EXTRA, "rev3_reruns_analysis.json")))
    ha = json.load(open(os.path.join(EXTRA, "hard_math_analysis.json")))
    T = ra["tasks"]
    old = os.path.join(EXTRA, "within_run_check.json")
    if "--reuse-drift" in sys.argv and os.path.exists(old):  # the drift tests are the slow part
        D = json.load(open(old))["drift"]
    else:
        D = drift_tests()
    J["drift"] = D
    same = [abs(v["diff"]) for v in D.values() if v["hw"] == "same GPU"]
    same_sampling = [abs(v["diff"]) for v in D.values() if v["hw"] == "same GPU" and v["config"].startswith(("SA", "SC"))]
    cross_m = [abs(v["diff"]) for v in D.values() if v["hw"] == "across machines"]
    dmax_same, dmax_cross = max(same), max(cross_m)
    J["drift_max"] = {"same_gpu_any": dmax_same, "same_gpu_sampling": max(same_sampling),
                      "across_machines_any": dmax_cross,
                      "same_gpu_max_cfg": [k for k, v in D.items() if v["hw"] == "same GPU" and abs(v["diff"]) == dmax_same],
                      "across_max_cfg": [k for k, v in D.items() if v["hw"] == "across machines" and abs(v["diff"]) == dmax_cross]}

    def adv(task, w, a):  # sampling minus workflow: (diff, [lo, hi]) from the stored workflow-minus-sampling test
        t = T[task]["tests"][f"{w}|{a}"]
        return -t["diff"], [-t["ci95"][1], -t["ci95"][0]], t

    rows = []

    fit = json.load(open(os.path.join(EXTRA, "rev3_costs_analysis.json")))["energy_fit"]
    UNITS = [("total", "total tokens"), ("output", "output tokens")] + \
            [(f"price{r}", f"price $r{{=}}{r}$") for r in (2, 3, 4, 5)] + [("energy", "modeled energy")]

    def cost(task, cfg, unit):
        c = T[task]["configs"][cfg]
        out = c["out_tokens"]; inp = c["total_tokens"] - out
        if unit == "total":
            return inp + out
        if unit == "output":
            return out
        if unit.startswith("price"):
            return inp + int(unit[5:]) * out
        return fit["J_per_output_token"] * out + fit["J_per_input_token"] * inp

    def verdict(task, w, unit):
        cheaper = [a for a in ("SA", "SC_k2", "SC_k4", "SC_k8") if cost(task, a, unit) < cost(task, w, unit)]
        robust = [a for a in cheaper if adv(task, w, a)[1][0] > 0]
        v = "R" if robust else ("N" if (not cheaper or all(adv(task, w, a)[1][1] < 0 for a in cheaper)) else "C")
        return v, robust, max([adv(task, w, a)[1][0] for a in robust], default=None)

    for task, hw in (("gsm8k", "same GPU"), ("mmlu", "across machines"), ("proeng", "same GPU")):
        dm = dmax_same if hw == "same GPU" else dmax_cross
        for unit, ulbl in UNITS:
            vx, byx, marg = verdict(task, "HIERFIX_M3", unit)
            vw, byw, _ = verdict(task, "HIER_M3", unit)
            if unit in ("total", "output"):  # agrees with the published analysis
                assert vx == T[task]["verdict_" + unit]["HIERFIX_M3"]["verdict"], (task, unit)
                assert vw == T[task]["verdict_" + unit]["HIER_M3"]["verdict"], (task, unit)
            robust_beyond = [a for a in byx if adv(task, "HIERFIX_M3", a)[1][0] > dm]
            drift_ok = (marg is not None and marg > dm) if vx == "R" else None
            outcome = "survives" if vx == vw and (vx != "R" or drift_ok) else "changes"
            rows.append({"id": f"{task}_hier_{unit}", "task": task, "claim": f"HIER verdict, {ulbl}",
                         "unit": unit, "hw": hw,
                         "cross": {"verdict": vx, "robust_by": byx, "margin": marg},
                         "within": {"workflow": "HIER_M3 (original HIER, same run as sampling)",
                                    "verdict": vw, "robust_by": byw},
                         "drift_bound": dm, "robust_beyond_drift": robust_beyond, "drift_ok": drift_ok,
                         "outcome": outcome})

    # C1: repaired HIER minus its cost-fixed comparator (SC k=4) on GSM8K
    c1 = ra["confirmatory"]["C1"]
    w1 = T["gsm8k"]["tests"]["HIER_M3|" + c1["comparator"]]
    rows.append({"id": "C1", "task": "gsm8k", "claim": "C1: HIER minus SC $k{=}4$", "hw": "same GPU",
                 "cross": {"diff": c1["diff"], "ci95": c1["ci95"], "margin": -c1["ci95"][1]},
                 "within": {"workflow": "HIER_M3", "diff": w1["diff"], "ci95": w1["ci95"]},
                 "drift_bound": dmax_same, "drift_ok": -c1["ci95"][1] > dmax_same,
                 "outcome": "survives" if (w1["ci95"][1] < 0 and -c1["ci95"][1] > dmax_same) else "changes"})

    # gain bounds against SC k=8 (one-sided 95%, upper limit of the 90% interval)
    for task, hw in (("gsm8k", "same GPU"), ("mmlu", "across machines"), ("proeng", "same GPU")):
        tx, tw = T[task]["tests"]["HIERFIX_M3|SC_k8"], T[task]["tests"]["HIER_M3|SC_k8"]
        dm = dmax_same if hw == "same GPU" else dmax_cross
        ex_any, ex_any_w = tx["ci90"][1] < 0, tw["ci90"][1] < 0
        ex5, ex5_w = tx["ci90"][1] < 5, tw["ci90"][1] < 5
        claim = ("any gain over SC $k{=}8$ excluded" if ex_any else
                 f"gain over SC $k{{=}}8$ at most {tx['ci90'][1]:+.1f}")
        ok = tw["ci90"][1] <= tx["ci90"][1] and (not ex_any or -tx["ci90"][1] > dm)
        rows.append({"id": f"{task}_hier_bound", "task": task, "claim": "HIER: " + claim, "hw": hw,
                     "cross": {"diff": tx["diff"], "upper": tx["ci90"][1], "any_excluded": ex_any, "five_excluded": ex5},
                     "within": {"workflow": "HIER_M3", "diff": tw["diff"], "upper": tw["ci90"][1],
                                "any_excluded": ex_any_w, "five_excluded": ex5_w},
                     "drift_bound": dm, "outcome": "survives" if ok else "changes"})

    # Du et al. debate on MATH L4-5 against SC k=8: no run holds it with sampling
    du = ha["du"]["minus_SC_k8"]
    rows.append({"id": "du45", "task": "math45", "claim": "Du et al.\\ debate minus SC $k{=}8$", "hw": "same GPU",
                 "cross": {"diff": du["diff"], "ci95": du["ci95"], "margin": -du["ci95"][1]},
                 "within": None, "drift_bound": dmax_same, "drift_ok": -du["ci95"][1] > dmax_same,
                 "outcome": "uncheckable"})

    # published-debate run on GSM8K (100 items x 3 seeds; comparators: the 500-item run's first 100 items)
    xw = json.load(open(os.path.join(PAPER, "tables/extra_workflows.json")))["tests"]
    t, tw = xw["HIERFIX - SC_k8"], xw["HIER - SC_k8"]
    rows.append({"id": "gsm8k_xw_hierfix", "task": "gsm8k", "claim": "HIER (published-debate run) minus SC $k{=}8$",
                 "hw": "same GPU", "cross": {"diff": t["diff"], "ci95": t["ci95"], "margin": -t["ci95"][1]},
                 "within": {"workflow": "HIER_M3", "diff": tw["diff"], "ci95": tw["ci95"]},
                 "drift_bound": dmax_same,
                 "outcome": "survives" if (tw["ci95"][1] < 0 and -t["ci95"][1] > dmax_same) else "changes"})
    t = xw["DU - SC_k8"]
    rows.append({"id": "gsm8k_xw_du", "task": "gsm8k", "claim": "Du et al.\\ debate minus SC $k{=}8$",
                 "hw": "same GPU", "cross": {"diff": t["diff"], "ci95": t["ci95"], "margin": None},
                 "within": None, "drift_bound": dmax_same, "outcome": "uncheckable"})

    # C5: equal-cap difficulty interaction (MATH L4-5 run vs MATH L1-3 rerun)
    c5 = ra["r5"]["interaction_equal_cap"]["DEBATE_N3_R1"]
    ix = ha["interaction_level5_minus_level4"]["DEBATE_N3_R1"]
    rows.append({"id": "C5", "task": "math", "claim": "C5: debate's difficulty interaction", "hw": "same GPU",
                 "cross": {"diff": c5["obs"], "ci95": c5["ci95"], "p": c5["p"]},
                 "within": {"contrast": "level 5 minus level 4, MATH L4-5 run", "diff": ix["obs"],
                            "ci95": ix["ci95"], "p": ix["p"]},
                 "no_detectable_gain_cross": c5["ci95"][1] >= 0 and c5["ci95"][0] <= 0,
                 "no_detectable_gain_within": ix["ci95"][0] <= 0,
                 "sign_cross": int(np.sign(c5["obs"])), "sign_within": int(np.sign(ix["obs"])),
                 "outcome": "changes" if np.sign(c5["obs"]) != np.sign(ix["obs"]) else "survives"})
    J["claims"] = rows

    # within-run already: every other headline verdict (debate and GE on every task; HIER on MATH and code)
    J["within_run_already"] = {
        t: {w: T[t]["verdict_total"][w]["verdict"] for w in ("DEBATE_N3_R1", "GE_R2")}
        for t in ("gsm8k", "mmlu", "proeng", "code_v2")}
    J["within_run_already"]["code_v2"]["HIERFIX_M3"] = T["code_v2"]["verdict_total"]["HIERFIX_M3"]["verdict"]
    os.makedirs(EXTRA, exist_ok=True)
    json.dump(J, open(os.path.join(EXTRA, "within_run_check.json"), "w"), indent=1)
    emit(J, T)
    for r in rows:
        print(r["id"], r["outcome"], r["cross"], r.get("within"))
    print("drift max", J["drift_max"])


# ===================================================================== LaTeX
def f1(x, sign=True):
    s = f"{x:+.1f}" if sign else f"{x:.1f}"
    return "$" + s.replace("-", "-") + "$"


def ci(c):
    return f"$[{c[0]:+.1f},{c[1]:+.1f}]$"


VNAME = {"SA": "SA", "SC_k2": "SC $k{=}2$", "SC_k4": "SC $k{=}4$", "SC_k8": "SC $k{=}8$"}
TNAME = {"gsm8k": "GSM8K", "mmlu": "MMLU", "proeng": "MMLU-Pro eng.", "math45": "MATH L4-5", "math": "MATH"}


def emit(J, T):
    R = {r["id"]: r for r in J["claims"]}
    D, M = J["drift"], J["drift_max"]
    mac = {}
    def m(name, val):
        mac[name] = val
    # drift summary
    m("WrDriftSame", f"{M['same_gpu_any']:.1f}")
    m("WrDriftSameSampling", f"{M['same_gpu_sampling']:.1f}")
    m("WrDriftCross", f"{M['across_machines_any']:.1f}")
    sm = [k for k in M["same_gpu_max_cfg"]][0]
    m("WrDriftSameWhere", f"{D[sm]['config']} on {D[sm]['task']}")
    cm = M["across_max_cfg"][0]
    m("WrDriftCrossWhere", f"{D[cm]['config']} on {D[cm]['task']}")
    m("WrDriftSampCross", f"{max(abs(v['diff']) for v in D.values() if v['hw'] == 'across machines' and v['config'].startswith(('SA', 'SC'))):.1f}")
    g = D["gsm8k_hierfix"]
    m("WrHierfixDrift", f1(g["diff"])); m("WrHierfixDriftI", ci(g["ci95"]))
    gs = [D[f"gsm8k_{c}"] for c in ("SA", "SC_k8", "DEBATE_N3_R1")]
    m("WrGsmDriftMax", f"{max(abs(x['diff']) for x in gs + [g]):.1f}")
    # GSM8K HIER total verdict
    r = R["gsm8k_hier_total"]
    m("WrGsmMargin", f"{r['cross']['margin']:.1f}")
    m("WrGsmRobustBeyond", ", ".join(VNAME[a] for a in r["robust_beyond_drift"]) or "none")
    r = R["gsm8k_hier_output"]
    m("WrGsmOutMargin", f"{r['cross']['margin']:.1f}")
    m("WrGsmOutOutcome", r["outcome"])
    weak = [x["unit"] for x in J["claims"] if x["task"] == "gsm8k" and "unit" in x and x["outcome"] == "changes"]
    pr = [u[5:] for u in weak if u.startswith("price")]
    parts = (["output tokens"] if "output" in weak else []) + \
            ([("price ratios " + pr[0] + " to " + pr[-1]) if len(pr) > 1 else ("price ratio " + pr[0])] if pr else []) + \
            (["modeled energy"] if "energy" in weak else [])
    m("WrGsmWeakUnits", ", ".join(parts[:-1]) + (" and " if len(parts) > 1 else "") + parts[-1])
    keep = [x["unit"] for x in J["claims"] if x["task"] == "gsm8k" and "unit" in x and x["outcome"] == "survives"]
    pk = [u[5:] for u in keep if u.startswith("price")]
    m("WrGsmKeepUnits", ", ".join((["total tokens"] if "total" in keep else []) +
                                  ([("price ratio " + pk[0]) if len(pk) == 1 else ("price ratios " + pk[0] + " to " + pk[-1])] if pk else [])).replace(", ", " and "))
    # C1
    r = R["C1"]
    m("WrCOneWithinD", f1(r["within"]["diff"])); m("WrCOneWithinI", ci(r["within"]["ci95"]))
    m("WrCOneMargin", f"{r['cross']['margin']:.1f}")
    # MMLU bound
    r = R["mmlu_hier_bound"]
    m("WrMmluUp", f1(r["cross"]["upper"])); m("WrMmluUpWithin", f1(r["within"]["upper"]))
    r = R["proeng_hier_bound"]
    m("WrEngUp", f1(r["cross"]["upper"])); m("WrEngUpWithin", f1(r["within"]["upper"]))
    # Du
    r = R["du45"]
    m("WrDuMargin", f"{r['cross']['margin']:.1f}")
    # C5
    r = R["C5"]
    m("WrCFiveWithinD", f1(r["within"]["diff"])); m("WrCFiveWithinI", ci(r["within"]["ci95"]))
    m("WrCFiveWithinP", f"p={r['within']['p']:.2f}")
    for _d in OUTDIRS:
        with open(os.path.join(_d, "within_run_macros.tex"), "w") as fh:
            fh.write("% AUTO-GENERATED by scripts/within_run_check.py (results/extra/within_run_check.json).\n")
            for k, v in mac.items():
                fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")

    # main-text table: one row per cross-run claim
    def vcell(v, by):
        return v + (f" {{\\scriptsize {VNAME[by[0]]}}}" if v == "R" and by else "")
    OUT = {"survives": "survives", "changes": "weakens", "uncheckable": "not checkable"}
    L = []
    UL = {"total": "total tok.", "output": "output tok.", "energy": "energy"}
    def ulist(us):
        if len(us) == 7:
            return "every cost unit"
        pr = [u[5:] for u in us if u.startswith("price")]
        parts = [UL[u] for u in us if u in ("total", "output")]
        if pr:
            parts.append("$r{=}" + (pr[0] if len(pr) == 1 else f"{pr[0]}$ to ${pr[-1]}") + "$")  # r: price ratio (caption)
        parts += [UL[u] for u in us if u == "energy"]
        return ", ".join(parts)
    # one group heading per task (no task column), so the claim fits on one line
    def head(name):
        if L:
            L.append(r"\addlinespace[2pt]")
        L.append(f"\\multicolumn{{6}}{{@{{}}l}}{{\\textit{{{name}}}}} \\\\")
    for task in ("gsm8k", "mmlu", "proeng"):
        head(TNAME[task])
        rs = [x for x in J["claims"] if x["task"] == task and "unit" in x]
        groups = {}
        for x in rs:
            k = (x["cross"]["verdict"], x["within"]["verdict"], x["outcome"],
                 (x["cross"]["margin"] if x["cross"]["margin"] is None else round(x["cross"]["margin"], 1)))
            groups.setdefault(k, []).append(x)
        for (vx, vw, oc, mg), xs in groups.items():
            r = xs[0]
            L.append(f"HIER verdict, {ulist([x['unit'] for x in xs])} & "
                     f"{vcell(vx, r['cross']['robust_by'])} & {vcell(vw, r['within']['robust_by'])} & "
                     f"{'--' if mg is None else f'{mg:.1f}'} & {r['drift_bound']:.1f} & {OUT[oc]} \\\\")
        if task == "gsm8k":
            r = R["C1"]
            L.append(f"C1, HIER $-$ SC $k{{=}}4$ & {f1(r['cross']['diff'])} {{\\scriptsize{ci(r['cross']['ci95'])}}} & "
                     f"{f1(r['within']['diff'])} {{\\scriptsize{ci(r['within']['ci95'])}}} & {r['cross']['margin']:.1f} & "
                     f"{r['drift_bound']:.1f} & {OUT[r['outcome']]} \\\\")
        r = R[f"{task}_hier_bound"]
        lbl = "HIER, any gain over SC $k{=}8$" if r["cross"]["any_excluded"] else "HIER, gain cap vs SC $k{=}8$"
        mg = f"{-r['cross']['upper']:.1f}" if r["cross"]["any_excluded"] else "--"
        L.append(f"{lbl} & $\\le${f1(r['cross']['upper'])} & $\\le${f1(r['within']['upper'])} & "
                 f"{mg} & {r['drift_bound']:.1f} & {OUT[r['outcome']]} \\\\")
    head("MATH")
    r = R["du45"]
    L.append(f"Du et al.\\ $-$ SC $k{{=}}8$, levels 4 to 5 & {f1(r['cross']['diff'])} {{\\scriptsize{ci(r['cross']['ci95'])}}} & "
             f"none & {r['cross']['margin']:.1f} & {r['drift_bound']:.1f} & {OUT[r['outcome']]} \\\\")
    r = R["C5"]
    L.append(f"C5, difficulty interaction & {f1(r['cross']['diff'])} {{\\scriptsize{ci(r['cross']['ci95'])}}} & "
             f"{f1(r['within']['diff'])} {{\\scriptsize{ci(r['within']['ci95'])}}} & -- & -- & {OUT[r['outcome']]} \\\\")
    for _d in OUTDIRS:
        with open(os.path.join(_d, "tables/within_run_main.tex"), "w") as fh:
            fh.write("% AUTO-GENERATED by scripts/within_run_check.py: cross-run claims and their within-run check.\n")
            fh.write("\n".join(L) + "\n")

    # supplement: all claim rows incl. output-token and MMLU-Pro bound rows
    L = []
    for r in J["claims"]:
        t = TNAME[r["task"]]
        if "verdict" in r["cross"]:
            cross = vcell(r["cross"]["verdict"], r["cross"]["robust_by"])
            within = vcell(r["within"]["verdict"], r["within"]["robust_by"])
            extra = ("robust beyond drift: " + (", ".join(VNAME[a] for a in r["robust_beyond_drift"]) or "none")
                     if r["cross"]["verdict"] == "R" else "")
        elif "upper" in r["cross"]:
            cross = f"$\\le${f1(r['cross']['upper'])}"; within = f"$\\le${f1(r['within']['upper'])}"; extra = ""
        else:
            cross = f"{f1(r['cross']['diff'])} {ci(r['cross']['ci95'])}"
            within = (f"{f1(r['within']['diff'])} {ci(r['within']['ci95'])}" if r["within"] else "none")
            extra = ""
        L.append(f"{t} & {r['claim']} & {r['hw']} & {cross} & {within} & {extra} & {OUT[r['outcome']]} \\\\")
    for _d in OUTDIRS:
        with open(os.path.join(_d, "tables/within_run_supp.tex"), "w") as fh:
            fh.write("% AUTO-GENERATED by scripts/within_run_check.py: every cross-run claim.\n")
            fh.write("\n".join(L) + "\n")

    # supplement: drift of identical configurations
    L = []
    for k, v in D.items():
        c = ci(v["ci95"]) if v.get("ci95") else "--"
        L.append(f"{v['task']} & {v['config']} & {v['runs']} & {v['hw']} & {v['n']} & {f1(v['diff'])} & {c} \\\\")
    for _d in OUTDIRS:
        with open(os.path.join(_d, "tables/within_run_drift.tex"), "w") as fh:
            fh.write("% AUTO-GENERATED by scripts/within_run_check.py: later run minus earlier run, identical configuration.\n")
            fh.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
