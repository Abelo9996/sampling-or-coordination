"""Published debate protocol and repaired planner-executor (GSM8K): numbers for the paper.

Inputs (read only; no model calls):
  results/extra/extra_gsm8k_analysis.json   output of scripts/analyze_extra_gsm8k.py
      (protocol: protocols/EXTRA_WORKFLOWS_PROTOCOL.md, committed before the runs)
  results/power/extra_gsm8k_qwen3b_s{1,2,3}.jsonl   DU_DEBATE_N3_R2 and HIERFIX_M3
  results/power/gsm8k_qwen3b_s{1,2,3}.jsonl         comparators (RTX 4060 rerun of App. D)

Taken from the analysis JSON: accuracies, costs, the pre-specified tests
(DU and HIERFIX minus SC k=8; secondary DU minus debate, HIERFIX minus HIER,
DU and HIERFIX minus SA), the HIER aggregator checks and the DU checks.
Recomputed here from the records and asserted equal to the JSON: every accuracy
and mean total-token cost, the HIER and HIERFIX no-answer rates, and every
pre-specified test (same function, B=10,000, seed 0).
Added here (not in the JSON, for the table and one post hoc sentence):
  * SA, debate and HIER minus SC k=8 with the same test (table rows);
  * no-answer share of every row;
  * cost-matched sampling: the largest k such that the first k samples of each
    SC k=8 record (majority by harness.topologies._majority over the logged
    per-call `extracted` values, scored by the GSM8K scorer) cost no more mean
    total tokens than HIERFIX, its accuracy, and HIERFIX minus it (post hoc).
Writes tables/extra_workflows.json, tables/extra_workflows_table.tex and
extra_workflows_macros.tex. Usage: python3 scripts/emit_extra_workflows.py
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
from harness.stats import hier_paired_test  # noqa: E402
from harness.tasks import gsm8k_score  # noqa: E402
from harness.topologies import _majority  # noqa: E402

AJ = os.path.join(ROOT, "results", "extra", "extra_gsm8k_analysis.json")
SEEDS = [1, 2, 3]


def load(pat):
    out = {}
    for p in sorted(glob.glob(os.path.join(ROOT, pat))):
        for line in open(p):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[(r["config"], r["item_id"], r["seed"])] = r
    return out


def main():
    A = json.load(open(AJ))
    X = load("results/power/extra_gsm8k_qwen3b_s[123].jsonl")
    B = load("results/power/gsm8k_qwen3b_s[123].jsonl")
    ids = sorted({k[1] for k in X})
    assert len(ids) == A["n_items"] == 100 and A["seeds"] == SEEDS

    CFG = {"SA": (B, "SA", "correct"), "SC_k8": (B, "SC_k8", "correct"),
           "DEBATE": (B, "DEBATE_N3_R1", "correct"), "HIER": (B, "HIER_M3", "correct"),
           "DU": (X, "DU_DEBATE_N3_R2", "correct"), "DU_harness": (X, "DU_DEBATE_N3_R2", "correct_harness"),
           "HIERFIX": (X, "HIERFIX_M3", "correct")}

    def recs(name):
        D, cfg, _ = CFG[name]
        return [D[(cfg, i, s)] for i in ids for s in ([1] if cfg == "SA" else SEEDS)]

    def mat(name):
        D, cfg, key = CFG[name]
        if cfg == "SA":
            return {i: np.array([float(bool(D[(cfg, i, 1)][key]))] * 3) for i in ids}
        return {i: np.array([float(bool(D[(cfg, i, s)][key])) for s in SEEDS]) for i in ids}

    M = {k: mat(k) for k in CFG}
    acc = {k: 100 * float(np.mean([v.mean() for v in M[k].values()])) for k in CFG}
    for k in CFG:
        assert abs(acc[k] - A["accuracy"][k]) < 0.006, k
    tok = {k: float(np.mean([r["total_tokens"] for r in recs(k)])) for k in CFG}
    for k in CFG:
        if k != "DU_harness":
            assert abs(tok[k] - A["cost"][k]["total_tokens"]) < 0.06, k
    calls = {k: sorted({r["n_calls"] for r in recs(k)}) for k in CFG}
    assert all(len(v) == 1 for v in calls.values()), calls
    calls = {k: v[0] for k, v in calls.items()}

    # no-answer share (harness extractor returned nothing; DU: their vote returned None)
    def pct(x):
        return 100 * float(np.mean(x))
    noans = {k: pct([r["pred"] is None for r in recs(k)]) for k in ("SA", "SC_k8", "DEBATE", "HIER", "HIERFIX")}
    noans["DU"] = pct([r["du_vote_none"] for r in recs("DU")])
    noans["DU_harness"] = pct([r["pred_harness"] is None for r in recs("DU_harness")])
    assert abs(noans["HIERFIX"] - A["hier_checks"]["HIERFIX_M3"]["no_answer_pct"]) < 0.006
    assert abs(noans["HIER"] - A["hier_checks"]["HIER_M3"]["no_answer_pct"]) < 0.006
    assert noans["DU"] == A["du_checks"]["none_vote_pct"]

    # tests: workflow minus comparator, as in the analysis script
    def test(a, b):
        d, p, lo, hi = hier_paired_test(M[a], M[b], B=10000, seed=0)
        return {"diff": d, "p": float(p), "ci95": [lo, hi]}
    T = {}
    for a in ("SA", "DEBATE", "HIER", "DU", "DU_harness", "HIERFIX"):
        T[f"{a} - SC_k8"] = test(a, "SC_k8")
    for a, b in (("DU", "DEBATE"), ("HIERFIX", "HIER"), ("DU", "SA"), ("HIERFIX", "SA")):
        T[f"{a} - {b}"] = test(a, b)
    pre = {**A["primary_gsm8k"], **A["primary_secondary_scoring"], **A["secondary_gsm8k"]}
    for k, v in pre.items():
        t = T[k]
        assert round(t["diff"], 2) == v["diff"] and t["p"] == v["p"], k
        assert [round(x, 2) for x in t["ci95"]] == v["ci95"], k

    # cost-matched sampling (post hoc): prefixes of the SC k=8 records
    sc = recs("SC_k8")
    pref = {}
    for k in range(1, 9):
        ok = [bool(gsm8k_score(_majority([c["extracted"] for c in r["calls"][:k]]), r["gold"])) for r in sc]
        t = [sum(c["in_tokens"] + c["out_tokens"] for c in r["calls"][:k]) for r in sc]
        pref[k] = {"acc": pct(ok), "total_tokens": float(np.mean(t)), "ok": ok}
    assert pref[8]["ok"] == [bool(r["correct"]) for r in sc]
    assert abs(pref[8]["total_tokens"] - tok["SC_k8"]) < 1e-6
    kmatch = max(k for k in pref if pref[k]["total_tokens"] <= tok["HIERFIX"])
    okm = dict(zip([(r["item_id"], r["seed"]) for r in sc], pref[kmatch]["ok"]))
    M["SC_pref"] = {i: np.array([float(okm[(i, s)]) for s in SEEDS]) for i in ids}
    T["HIERFIX - SC_pref"] = test("HIERFIX", "SC_pref")

    ratio = A["cost"]["DU"]["total_tokens"] / A["cost"]["SC_k8"]["total_tokens"]
    HC, DC = A["hier_checks"], A["du_checks"]
    out = {"acc": acc, "total_tokens": tok, "calls": calls, "no_answer_pct": noans, "tests": T,
           "sc_prefix": {k: {"acc": v["acc"], "total_tokens": v["total_tokens"]} for k, v in pref.items()},
           "k_cost_matched": kmatch, "du_over_sc8_tokens": ratio}
    json.dump(out, open(os.path.join(HERE, "..", "tables", "extra_workflows.json"), "w"), indent=1)

    # formatting
    def f1(x, sign=True):
        x = round(x, 1) + 0.0
        return f"{'+' if sign and x > 0 else ''}{x:.1f}"

    def iv(t):
        return f"${f1(t['diff'])}$ $[{f1(t['ci95'][0])},{f1(t['ci95'][1])}]$"

    def pv(p):
        return "${<}0.001$" if p < 0.001 else f"${p:.3f}$" if p < 0.1 else f"${p:.2f}$"

    def tk(x):
        return f"{round(x):,}".replace(",", "{,}")

    ROWS = [("SA", "Single call (greedy, 1 seed)"), ("SC_k8", "SC $k{=}8$"),
            ("DEBATE", "Debate, ours ($N{=}3$, $R{=}1$)"),
            ("DU", "Du et al.\\ debate, their scoring"), ("DU_harness", "\\quad same runs, our scoring"),
            ("HIER", "HIER, original"), ("HIERFIX", "HIERFIX (aggregator line repaired)")]
    L = ["% AUTO-GENERATED by paper/arxiv_long/scripts/emit_extra_workflows.py. Do not edit."]
    for k, name in ROWS:
        if k == "SC_k8":
            d, p = "--", "--"
        else:
            d, p = iv(T[f"{k} - SC_k8"]), pv(T[f"{k} - SC_k8"]["p"])
        L.append(" & ".join([name, str(calls[k]), f1(acc[k], False), tk(tok[k]), d, p,
                             f1(noans[k], False)]) + " \\\\")
    open(os.path.join(HERE, "..", "tables", "extra_workflows_table.tex"), "w").write("\n".join(L) + "\n")

    MM = [L[0]]

    def add(k, v):
        MM.append(f"\\newcommand{{\\Xw{k}}}{{{v}}}")
    KEY = {"SA": "Sa", "SC_k8": "Sc", "DEBATE": "Deb", "HIER": "Hier", "DU": "Du",
           "DU_harness": "DuH", "HIERFIX": "Fix"}
    for k, key in KEY.items():
        add(f"Acc{key}", f1(acc[k], False))
        add(f"Tok{key}", tk(tok[k]))
        add(f"NoAns{key}", f1(noans[k], False))
    for k, key in (("DU - SC_k8", "DuSc"), ("DU_harness - SC_k8", "DuHSc"), ("HIERFIX - SC_k8", "FixSc"),
                   ("DU - DEBATE", "DuDeb"), ("HIERFIX - HIER", "FixHier"), ("DU - SA", "DuSa"),
                   ("HIERFIX - SA", "FixSa"), ("HIERFIX - SC_pref", "FixPref")):
        t = T[k]
        add(f"D{key}", f"${f1(t['diff'])}$")
        add(f"I{key}", f"$[{f1(t['ci95'][0])},{f1(t['ci95'][1])}]$")
        add(f"P{key}", "p{<}0.001" if t["p"] < 0.001 else f"p={t['p']:.3f}" if t["p"] < 0.1 else f"p={t['p']:.2f}")
    d = A["primary_gsm8k"]["DU - SC_k8"]
    add("UpDuSc", f"${f1(d['ci90'][1])}$")
    assert d["gain5_excluded_unadj"] and not d["any_gain_excluded_unadj"]
    add("Ratio", f"{ratio:.1f}")
    add("Kmatch", kmatch)
    add("AccPref", f1(pref[kmatch]["acc"], False))
    add("TokPref", tk(pref[kmatch]["total_tokens"]))
    add("LitFix", f"{HC['HIERFIX_M3']['literal_template_pct']:.0f}")
    add("LitHier", f"{HC['HIER_M3']['literal_template_pct']:.0f}")
    add("DuCapPct", f"{DC['calls_at_cap_pct']:.1f}")
    add("DuMaxCtx", f"{DC['max_in_plus_out']:,}".replace(",", "{,}"))
    add("DuDisagree", DC["scoring_disagree_records"])
    add("DuCalls", f"{DC['calls']:,}".replace(",", "{,}"))
    add("FixHierAbs", f"{abs(round(T['HIERFIX - HIER']['diff'], 1)):.1f}")
    add("FixMinusScAbs", f"{abs(round(T['HIERFIX - SC_k8']['diff'])):.0f}")
    # statements made in the text
    assert DC["none_vote_pct"] == 0 and DC["calls_ge_8192"] == 0
    assert T["HIERFIX - SA"]["ci95"][0] < 0 < T["HIERFIX - SA"]["ci95"][1]
    assert T["HIERFIX - SC_k8"]["ci95"][1] < 0 and T["HIERFIX - HIER"]["ci95"][0] > 0
    assert T["DU - SC_k8"]["ci95"][0] < 0 < T["DU - SC_k8"]["ci95"][1]
    assert tok["HIERFIX"] < tok["SC_k8"]
    open(os.path.join(HERE, "..", "extra_workflows_macros.tex"), "w").write("\n".join(MM) + "\n")
    print("\n".join(L + MM))


if __name__ == "__main__":
    main()
