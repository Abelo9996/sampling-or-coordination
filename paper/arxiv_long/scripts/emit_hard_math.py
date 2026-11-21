"""Hard regime, MATH levels 4 to 5: numbers and table for the paper.

Inputs (read only; no model calls):
  results/extra/hard_math_analysis.json      output of scripts/analyze_hard_math.py
      (protocols: HARD_MATH_PROTOCOL.md, EXTRA_WORKFLOWS_PROTOCOL.md)
  results/audit/hardmath_extraction_summary.json   blind extraction audit
  results/power/math500hard_qwen3b_s*.jsonl  SA, SC k=12, debate, GE, HIER records
  results/power/extra_math45_qwen3b_s*.jsonl Du et al. debate records

Taken from the analysis JSON: accuracies, costs, every pre-specified test, the
interaction and the Du checks. Recomputed here from the records and asserted
equal to the JSON: every accuracy (SC k = first k samples of SC k=12, majority by
harness.topologies._majority, scored by the MATH scorer), every mean total and
output token cost, the per-level accuracies, the cost-only comparator choice, and
the three primary tests plus Du minus SC k=8 (harness.stats.hier_paired_test,
B=10,000, seed 0). Asserts the statements the text makes.
Writes tables/hard_math.json, tables/hard_math_table.tex, hard_math_macros.tex.
Usage: python3 scripts/emit_hard_math.py
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

AJ = os.path.join(ROOT, "results", "extra", "hard_math_analysis.json")
SJ = os.path.join(ROOT, "results", "audit", "hardmath_extraction_summary.json")
SEEDS = [1, 2, 3]
COORD = ["DEBATE_N3_R1", "GE_R2", "HIER_M3"]
KS = [2, 4, 8, 12]
MARGIN = 5.0


def _test(ab):
    """harness.stats.hier_paired_test plus the 90% interval from the same draws
    (items resampled, then seeds within items, independently for each arm)."""
    a, b = ab
    d, p, lo, hi = stats.hier_paired_test(a, b, B=10000, seed=0)
    items = sorted(set(a) & set(b))
    A_, B_ = [a[i] for i in items], [b[i] for i in items]
    rng, n, boot = np.random.default_rng(0), len(items), np.empty(10000)
    for t in range(10000):
        s = 0.0
        for ii in rng.integers(0, n, size=n):
            av, bv = A_[ii], B_[ii]
            s += av[rng.integers(0, len(av), size=len(av))].mean() - bv[rng.integers(0, len(bv), size=len(bv))].mean()
        boot[t] = 100 * s / n
    l2, h2, l90, h90 = np.percentile(boot, [2.5, 97.5, 5, 95])
    assert abs(l2 - lo) < 1e-9 and abs(h2 - hi) < 1e-9, "draws differ from hier_paired_test"
    return {"diff": float(d), "p": float(p), "ci95": [float(lo), float(hi)], "ci90": [float(l90), float(h90)]}


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


def main():
    A = json.load(open(AJ))
    S = json.load(open(SJ))
    assert A["complete"] and A["n_items_complete"] == 262 and A["du_complete"]
    R = load("results/power/math500hard_qwen3b_s*.jsonl")
    D = load("results/power/extra_math45_qwen3b_s*.jsonl")
    score = tasks.get_task("math500")["score"]
    ids = sorted({k[1] for k in R if k[0] == "SA"})
    assert len(ids) == 262
    level = {i: R[("SA", i, 1)]["level"] for i in ids}
    assert {str(L): sum(v == L for v in level.values()) for L in (4, 5)} == A["n_by_level"]

    M, tot, outt = {}, {}, {}
    for cfg in ["SA"] + COORD:
        ss = [1] if cfg == "SA" else SEEDS
        recs = [R[(cfg, i, s)] for i in ids for s in ss]
        M[cfg] = {i: np.array([float(bool(R[(cfg, i, s)]["correct"])) for s in ss] * (3 if cfg == "SA" else 1))
                  for i in ids}
        tot[cfg] = float(np.mean([sum(c["in_tokens"] + c["out_tokens"] for c in r["calls"]) for r in recs]))
        outt[cfg] = float(np.mean([sum(c["out_tokens"] for c in r["calls"]) for r in recs]))
    for k in KS:
        m, t, o = {}, [], []
        for i in ids:
            v = []
            for s in SEEDS:
                r = R[("SC_k12", i, s)]
                cs = r["calls"][:k]
                ok = bool(score(topologies._majority([c["extracted"] for c in cs]), r["gold"]))
                if k == 12:
                    assert ok == bool(r["correct"])
                v.append(float(ok))
                t.append(sum(c["in_tokens"] + c["out_tokens"] for c in cs))
                o.append(sum(c["out_tokens"] for c in cs))
            m[i] = np.array(v)
        M[f"SC_k{k}"], tot[f"SC_k{k}"], outt[f"SC_k{k}"] = m, float(np.mean(t)), float(np.mean(o))
    durecs = [D[("DU_DEBATE_N3_R2", i, s)] for i in ids for s in SEEDS]
    M["DU"] = {i: np.array([float(bool(D[("DU_DEBATE_N3_R2", i, s)]["correct"])) for s in SEEDS]) for i in ids}
    tot["DU"] = float(np.mean([r["total_tokens"] for r in durecs]))
    outt["DU"] = float(np.mean([r["out_tokens"] for r in durecs]))

    def acc(m, sub=None):
        return 100 * float(np.mean([m[i].mean() for i in (sub or ids)]))
    ACC = {k: acc(v) for k, v in M.items()}
    for k, v in A["accuracy"].items():
        assert abs(ACC[k] - v) < 0.006, k
        assert abs(tot[k] - A["cost"][k]["total_tokens"]) < 0.06, k
        assert abs(outt[k] - A["cost"][k]["out_tokens"]) < 0.06, k
        for L in ("4", "5"):
            sub = [i for i in ids if level[i] == int(L)]
            assert abs(acc(M[k], sub) - A["accuracy_by_level"][L][k]) < 0.006, (k, L)
    du = A["du"]
    assert abs(ACC["DU"] - du["accuracy"]) < 0.006 and abs(tot["DU"] - du["total_tokens"]) < 0.06

    # comparator by cost only (protocol step 1), total and output tokens
    def comp(w, cost):
        return f"SC_k{max(k for k in KS if cost[f'SC_k{k}'] <= cost[w])}"
    for w in COORD:
        assert comp(w, tot) == A["primary_total_tokens"][w]["comparator"], w
        assert comp(w, outt) == A["output_tokens"][w]["comparator"], w

    # every test quoted in the paper, recomputed from the records (in parallel),
    # asserted equal to the JSON (2 decimals), then the unrounded values are used
    # for the 1-decimal display (the JSON's 2-decimal values tie at x.x5 in places)
    P = A["primary_total_tokens"]
    O = A["output_tokens"]
    jobs = [(P[w], w, P[w]["comparator"]) for w in COORD] + [(O[w], w, O[w]["comparator"]) for w in COORD] \
        + [(A["vs_SA"][w], w, "SA") for w in COORD] + [(A["SC_vs_SA"][f"SC_k{k}"], f"SC_k{k}", "SA") for k in KS] \
        + [(du["minus_SC_k8"], "DU", "SC_k8"), (du["minus_DEBATE_N3_R1"], "DU", "DEBATE_N3_R1"),
           (du["minus_SA"], "DU", "SA"), (du["minus_SC_k12"], "DU", "SC_k12")]
    with Pool(min(len(jobs), max(1, (os.cpu_count() or 2) - 1))) as pool:
        res = pool.map(_test, [(M[a], M[b]) for _, a, b in jobs])
    for (v, a, b), r in zip(jobs, res):
        assert round(r["diff"], 2) == v["diff"] and r["p"] == v["p"], (a, b)
        assert [round(x, 2) for x in r["ci95"]] == v["ci95"], (a, b)
        assert [round(x, 2) for x in r["ci90"]] == v["ci90"], (a, b)
        v.update({"diff": r["diff"], "ci95": r["ci95"], "ci90": r["ci90"]})

    # statements the text makes ------------------------------------------------
    O, SA_, SCSA, IX = A["output_tokens"], A["vs_SA"], A["SC_vs_SA"], A["interaction_level5_minus_level4"]
    assert all(P[w]["ci95"][1] < 0 and P[w]["p_holm"] < 0.05 for w in COORD)
    assert O["DEBATE_N3_R1"]["ci95"][1] < 0 and O["DEBATE_N3_R1"]["p_holm"] < 0.05
    assert all(O[w]["ci95"][0] < 0 < O[w]["ci95"][1] for w in ("GE_R2", "HIER_M3"))
    assert all(SA_[w]["ci95"][0] < 0 < SA_[w]["ci95"][1] for w in COORD)
    assert all(SA_[w]["p_gain_margin_holm"] < 0.05 and SA_[w]["ci90"][1] < MARGIN for w in COORD)
    assert all(SCSA[f"SC_k{k}"]["ci95"][0] > 0 for k in (4, 8, 12))
    assert SCSA["SC_k2"]["ci95"][0] < 0 < SCSA["SC_k2"]["ci95"][1]
    assert all(IX[w]["ci95"][0] < 0 < IX[w]["ci95"][1] and IX[w]["obs"] > 0 for w in COORD)
    assert du["minus_SC_k8"]["ci95"][1] < 0
    assert du["minus_DEBATE_N3_R1"]["ci95"][0] < 0 < du["minus_DEBATE_N3_R1"]["ci95"][1]
    assert du["minus_SA"]["ci95"][0] < 0 < du["minus_SA"]["ci95"][1]
    assert du["minus_SA"]["p_gain_margin"] >= 0.05  # a 5-point gain over one call is NOT excluded for Du
    assert tot["SC_k8"] < tot["DU"] and du["accuracy"] == du["accuracy_harness_scoring"]
    assert du["none_vote_pct"] == 0 and du["calls_ge_8192"] == 1
    # every named workflow is below its comparator by Lo to Hi points, rounded to whole points (abstract)
    gaps = [-P[w]["diff"] for w in COORD] + [-du["minus_SC_k8"]["diff"]]
    rlo, rhi = int(np.round(min(gaps))), int(np.round(max(gaps)))
    assert all(rlo - 0.5 <= g < rhi + 0.5 for g in gaps) and (rlo, rhi) == (5, 8), gaps
    # extraction audit
    pw = S["per_workflow"]
    assert all(pw[k]["flip_wrong_to_right"] == pw[k]["flip_right_to_wrong"] == 0 for k in pw) and S["flips"] == []
    assert all(pw[k]["logged_acc_pct"] == pw[k]["rescored_acc_pct"] for k in pw)
    for k in pw:
        assert abs(pw[k]["logged_acc_pct"] - A["accuracy"][k]) < 0.006, k
    dec = pw["SA"]["deciding_calls"] + pw["SC_k12"]["deciding_calls"] + sum(pw[w]["deciding_calls"] for w in COORD)
    flg = pw["SA"]["flagged_calls"] + pw["SC_k12"]["flagged_calls"] + sum(pw[w]["flagged_calls"] for w in COORD)
    dif = pw["SA"]["reading_differs_from_harness_calls"] + pw["SC_k12"]["reading_differs_from_harness_calls"] \
        + sum(pw[w]["reading_differs_from_harness_calls"] for w in COORD)
    assert flg == S["read_plan"]["n"] and S["read_plan"]["mode"] == "all" and dif == len(S["disagreements"])
    forms = S["scorer_problem_rules"]["equal_but_marked_wrong"]
    nforms, nitems = len(forms), len({f[0] for f in forms})
    SF = S["differences_points"]
    for k in pw:  # no-answer shares in the JSON and the audit agree
        if k in A["cost"]:
            assert abs(100 * pw[k]["record_no_answer_logged_share"] - A["cost"][k]["no_answer_pct"]) < 0.01, k

    # outputs -------------------------------------------------------------------
    json.dump({"acc": ACC, "total_tokens": tot, "out_tokens": outt, "level": {str(L): {k: acc(M[k], [i for i in ids if level[i] == L]) for k in M} for L in (4, 5)},
               "gaps_vs_comparator": gaps,
               "tests_unrounded": [{"a": a, "b": b, **r} for (_, a, b), r in zip(jobs, res)]}, open(os.path.join(HERE, "..", "tables", "hard_math.json"), "w"), indent=1)

    def f1(x, sign=True):
        x = round(x, 1) + 0.0
        return f"{'+' if sign and x > 0 else ''}{x:.1f}"

    def iv(t, key="diff"):
        return f"${f1(t[key])}$ $[{f1(t['ci95'][0])},{f1(t['ci95'][1])}]$"

    def pv(p):
        return "${<}0.001$" if p < 0.001 else f"${p:.3f}$" if p < 0.1 else f"${p:.2f}$"

    def pt(p):  # for running text
        return "p{<}0.001" if p < 0.001 else f"p={p:.3f}" if p < 0.1 else f"p={p:.2f}"

    def tk(x):
        return f"{round(x):,}".replace(",", "{,}")

    SCN = {"SC_k2": "SC $k{=}2$", "SC_k4": "SC $k{=}4$", "SC_k8": "SC $k{=}8$", "SC_k12": "SC $k{=}12$"}
    L = ["% AUTO-GENERATED by paper/arxiv_long/scripts/emit_hard_math.py. Do not edit."]
    L.append(" & ".join(["Single call (greedy, 1 seed)", f1(ACC["SA"], False), tk(tot["SA"]), "--", "--", "--", "--"]) + " \\\\")
    for k in KS:
        c = f"SC_k{k}"
        L.append(" & ".join([SCN[c], f1(ACC[c], False), tk(tot[c]), "--", "--", "--", iv(SCSA[c])]) + " \\\\")
    L.append("\\midrule")
    for w, name in (("DEBATE_N3_R1", "Debate ($N{=}3$, $R{=}1$)"), ("GE_R2", "GE ($R{=}2$)"), ("HIER_M3", "HIER ($M{=}3$)")):
        L.append(" & ".join([name, f1(ACC[w], False), tk(tot[w]), SCN[P[w]["comparator"]], iv(P[w]), pv(P[w]["p_holm"]),
                             iv(SA_[w])]) + " \\\\")
    hx = A["extra_primary_family_holm"]["MATH45 DU - SC_k8"]
    L.append(" & ".join(["Du et al.\\ debate", f1(ACC["DU"], False), tk(tot["DU"]), SCN["SC_k8"] + "$^\\dagger$",
                         iv(du["minus_SC_k8"]), pv(hx), iv(du["minus_SA"])]) + " \\\\")
    open(os.path.join(HERE, "..", "tables", "hard_math_table.tex"), "w").write("\n".join(L) + "\n")

    MM = [L[0]]

    def add(k, v):
        MM.append(f"\\newcommand{{\\Hm{k}}}{{{v}}}")
    KEY = {"SA": "Sa", "SC_k2": "Sctwo", "SC_k4": "Scfour", "SC_k8": "Sceight", "SC_k12": "Sctwelve",
           "DEBATE_N3_R1": "Deb", "GE_R2": "Ge", "HIER_M3": "Hier", "DU": "Du"}
    for k, key in KEY.items():
        add(f"Acc{key}", f1(ACC[k], False))
        add(f"Tok{key}", tk(tot[k]))
        for Lv, Ln in (("4", "Lfour"), ("5", "Lfive")):
            v = du["accuracy_by_level"][Lv] if k == "DU" else A["accuracy_by_level"][Lv][k]
            add(f"Acc{key}{Ln}", f1(v, False))
        if k in A["cost"]:
            add(f"NoAns{key}", f1(A["cost"][k]["no_answer_pct"], False))
    CK = {"DEBATE_N3_R1": "Deb", "GE_R2": "Ge", "HIER_M3": "Hier"}
    for w, key in CK.items():
        for fam, tag in ((P, "Pr"), (O, "Out")):
            t = fam[w]
            add(f"D{tag}{key}", f"${f1(t['diff'])}$")
            add(f"I{tag}{key}", f"$[{f1(t['ci95'][0])},{f1(t['ci95'][1])}]$")
            add(f"P{tag}{key}", pt(t["p_holm"]))
            add(f"C{tag}{key}", SCN[t["comparator"]])
        t = SA_[w]
        add(f"DSa{key}", f"${f1(t['diff'])}$")
        add(f"ISa{key}", f"$[{f1(t['ci95'][0])},{f1(t['ci95'][1])}]$")
        add(f"UpSa{key}", f"${f1(t['ci90'][1])}$")
        t = IX[w]
        add(f"DIx{key}", f"${f1(t['obs'])}$")
        add(f"IIx{key}", f"$[{f1(t['ci95'][0])},{f1(t['ci95'][1])}]$")
        add(f"PIx{key}", pt(t["p"]))
        add(f"DSf{key}", f"${f1(SF[f'{w}_minus_SC_k8']['logged_scorer_fixed'])}$")
        add(f"DEight{key}", f"${f1(A['vs_SC_k8'][w]['diff'])}$")
    for k in (4, 8, 12):
        t = SCSA[f"SC_k{k}"]
        add(f"DScSa{['', '', '', '', 'four', '', '', '', 'eight', '', '', '', 'twelve'][k]}", f"${f1(t['diff'])}$")
        add(f"PScSa{['', '', '', '', 'four', '', '', '', 'eight', '', '', '', 'twelve'][k]}", pt(t["p"]))
    for k, key in (("minus_SC_k8", "DuSc"), ("minus_DEBATE_N3_R1", "DuDeb"), ("minus_SA", "DuSa"), ("minus_SC_k12", "DuSctwelve")):
        t = du[k]
        add(f"D{key}", f"${f1(t['diff'])}$")
        add(f"I{key}", f"$[{f1(t['ci95'][0])},{f1(t['ci95'][1])}]$")
        add(f"P{key}", pt(t["p"]))
    add("HolmDuSc", pt(hx))
    PF, SFm = A["extra_primary_family_holm"], A["extra_secondary_family_holm"]
    add("HolmGsmDuSc", pt(PF["GSM8K DU - SC_k8"]))
    add("HolmGsmFixSc", pt(PF["GSM8K HIERFIX - SC_k8"]))
    add("HolmGsmDuDeb", pt(SFm["GSM8K DU - DEBATE"]))
    add("HolmGsmFixHier", pt(SFm["GSM8K HIERFIX - HIER"]))
    add("HolmGsmDuSa", pt(SFm["GSM8K DU - SA"]))
    add("HolmGsmFixSa", pt(SFm["GSM8K HIERFIX - SA"]))
    add("HolmDuDeb", pt(SFm["MATH45 DU - DEBATE"]))
    add("DuCalls", f"{sum(r['n_calls'] for r in durecs):,}".replace(",", "{,}"))
    add("DuGeCtx", du["calls_ge_8192"])
    add("DuMaxCtx", f"{du['max_in_plus_out']:,}".replace(",", "{,}"))
    add("AccSaRound", f"{ACC['SA']:.0f}")
    add("RangeLo", rlo)
    add("RangeHi", rhi)
    add("NLfour", A["n_by_level"]["4"])
    add("NLfive", A["n_by_level"]["5"])
    add("Deciding", f"{dec:,}".replace(",", "{,}"))
    add("Flagged", flg)
    add("Differ", dif)
    add("Flips", len(S["flips"]))
    add("ScorerForms", nforms)
    add("ScorerItems", nitems)
    add("CapPctSa", f1(A["cost"]["SA"]["records_with_capped_call_pct"], False))
    open(os.path.join(HERE, "..", "hard_math_macros.tex"), "w").write("\n".join(MM) + "\n")
    print("\n".join(L + MM))


if __name__ == "__main__":
    main()
