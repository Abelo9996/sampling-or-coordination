#!/usr/bin/env python3
"""rev3: Table 1 MATH column from the existing MATH levels 1 to 3 run (238 items, seeds 1 to 3,
1,024 cap, RTX 4060; results/power/math500_qwen3b_s*.jsonl, campaign phase B).

No model call and no new test design: the records are read with the loaders of
scripts/analyze_hard_math.py and scripts/analyze_math_interaction.py (imported read-only, not
edited), SC k=2,4,8,12 are the first k samples of each SC k=12 record (as in those scripts), and
the paired comparisons use harness.stats.hier_paired_test (B=10,000, seed 0), exactly as the
published analysis does. The script asserts that it reproduces every number of
results/extra/math_interaction_analysis.json that it shares (accuracies, SC k=8 minus workflow and
workflow minus SA tests), so nothing here can drift from the analysis already reported.

Per-config accuracy intervals are 95% item-bootstrap intervals of the seed-averaged item
correctness (harness.analyze._bootstrap_ci, B=10,000, seed 0), the convention of every other
Table 1 cell (scripts/aaibd_emit_rerun_macros.py, cell()).

Writes (inside paper/arxiv_rev3 only):
  tables/math_easy.json        every number below, unrounded and rounded
  math_easy_macros.tex         \\MeN, per-config accuracy/CI/tokens, comparisons
Run: /usr/bin/python3 paper/arxiv_rev3/scripts/emit_math_easy.py
"""
import importlib.util, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REV3 = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(REV3))
sys.path.insert(0, ROOT)
from harness import analyze, stats, tasks, topologies  # noqa: E402

spec = importlib.util.spec_from_file_location("hm", os.path.join(ROOT, "scripts/analyze_hard_math.py"))
hm = importlib.util.module_from_spec(spec); spec.loader.exec_module(hm)

SEEDS = [1, 2, 3]
COORD = ["DEBATE_N3_R1", "GE_R2", "HIER_M3"]
KS = [2, 4, 8, 12]
NAME = {"SA": "Sa", "SC_k2": "Sctwo", "SC_k4": "Scfour", "SC_k8": "Sceight", "SC_k12": "Sctwelve",
        "DEBATE_N3_R1": "Deb", "GE_R2": "Ge", "HIER_M3": "Hier"}


def fmt_int(x):
    s = f"{int(round(x)):,}"
    return s.replace(",", "{,}")


def sgn(x, d=1):
    v = round(x, d)
    if v == 0:
        v = 0.0
    return f"{'+' if v > 0 else ('' if v == 0 else '-')}{abs(v):.{d}f}"


def main():
    task = tasks.get_task("math500")
    items = task["load"](n=238, split="all")
    ids = [it["id"] for it in items]
    recs = hm.load("results/power/math500_qwen3b_s*.jsonl")
    M, tot, out = {}, {}, {}
    for c in ["SA"] + COORD:
        seeds = [1] if c == "SA" else SEEDS
        R = {i: [recs[(c, i, s)] for s in seeds] for i in ids}
        M[c] = {i: np.array([float(bool(r["correct"])) for r in R[i]] * (3 if c == "SA" else 1)) for i in ids}
        tot[c] = {i: np.mean([sum(x["in_tokens"] + x["out_tokens"] for x in r["calls"]) for r in R[i]]) for i in ids}
        out[c] = {i: np.mean([sum(x["out_tokens"] for x in r["calls"]) for r in R[i]]) for i in ids}
    for k in KS:
        m, t, o = {}, {}, {}
        for i in ids:
            v, tt, oo = [], [], []
            for s in SEEDS:
                r = recs[("SC_k12", i, s)]; cs = r["calls"][:k]
                ans = topologies._majority([c["extracted"] for c in cs])
                ok = bool(task["score"](ans, r["gold"]))
                if k == 12:
                    assert ok == bool(r["correct"]), (i, s)
                v.append(float(ok)); tt.append(sum(c["in_tokens"] + c["out_tokens"] for c in cs))
                oo.append(sum(c["out_tokens"] for c in cs))
            m[i] = np.array(v); t[i] = np.mean(tt); o[i] = np.mean(oo)
        M[f"SC_k{k}"] = m; tot[f"SC_k{k}"] = t; out[f"SC_k{k}"] = o

    cfgs = ["SA"] + [f"SC_k{k}" for k in KS] + COORD
    J = {"source": "results/power/math500_qwen3b_s*.jsonl", "n_items": len(ids), "seeds": SEEDS,
         "cap": sorted({recs[("SC_k12", i, 1)]["cap"] for i in ids}), "configs": {}, "tests": {}}
    for c in cfgs:
        corr = np.array([M[c][i].mean() for i in ids])
        lo, hi = analyze._bootstrap_ci(corr, B=10000)
        J["configs"][c] = {"acc": 100 * float(corr.mean()), "lo": 100 * lo, "hi": 100 * hi,
                           "total_tokens": float(np.mean([tot[c][i] for i in ids])),
                           "out_tokens": float(np.mean([out[c][i] for i in ids]))}
    # every SC k versus every coordination workflow, and SA versus each, as in Table 2's scan
    for a in ["SA"] + [f"SC_k{k}" for k in KS]:
        for w in COORD:
            # One direction per comparison (audit L6): the test is run as workflow minus
            # sampling, the direction of results/extra/math_interaction_analysis.json and
            # Section 7, and negated for display as sampling minus workflow.
            raw = hm.test(M[w], M[a])
            J["tests"][f"{a}|{w}"] = {"diff": -raw["diff"], "p": raw["p"],
                                      "ci95": [-raw["ci95"][1], -raw["ci95"][0]],
                                      "ci90": [-raw["ci90"][1], -raw["ci90"][0]],
                                      "raw_workflow_minus_sampling": raw}
            dtok = np.array([tot[a][i] - tot[w][i] for i in ids])
            lo, hi = analyze._bootstrap_ci(dtok, B=10000)
            J["tests"][f"{a}|{w}"]["dtok"] = float(dtok.mean())
            J["tests"][f"{a}|{w}"]["dtok_ci95"] = [lo, hi]
    J["tests"]["SC_k8|SA"] = hm.test(M["SC_k8"], M["SA"])

    # Reproduce the published analysis exactly.
    P = json.load(open(os.path.join(ROOT, "results/extra/math_interaction_analysis.json")))
    assert P["easy"]["n_items"] == len(ids)
    for c, v in P["easy"]["accuracy"].items():
        assert abs(round(J["configs"][c]["acc"], 2) - v) < 1e-9, (c, J["configs"][c]["acc"], v)
    for w in COORD:
        pub = P["within_stratum"][w]["easy_minus_SC_k8"]
        mine = J["tests"][f"SC_k8|{w}"]["raw_workflow_minus_sampling"]
        assert mine["diff"] == pub["diff"] and mine["ci95"] == pub["ci95"] and mine["p"] == pub["p"], (w, mine, pub)
        pub = P["within_stratum"][w]["easy_minus_SA"]
        mine = J["tests"][f"SA|{w}"]["raw_workflow_minus_sampling"]
        assert mine["diff"] == pub["diff"] and mine["ci95"] == pub["ci95"] and mine["p"] == pub["p"], (w, mine, pub)
    pub = P["easy_SC_k8_minus_SA"]; mine = J["tests"]["SC_k8|SA"]
    assert mine["diff"] == pub["diff"] and mine["ci95"] == pub["ci95"], (mine, pub)
    J["reproduces"] = "results/extra/math_interaction_analysis.json (accuracies, easy_minus_SC_k8, easy_minus_SA, easy_SC_k8_minus_SA)"

    # Noise-aware verdict per workflow under total tokens (rule of Table 2): every cheaper
    # sampling configuration, R if one has a 95% interval above zero, C if none is above zero and
    # not all are below zero, N if every cheaper one is detectably less accurate.
    J["verdicts"] = {}
    for w in COORD:
        cheaper = [a for a in ["SA"] + [f"SC_k{k}" for k in KS]
                   if J["configs"][a]["total_tokens"] < J["configs"][w]["total_tokens"]]
        rows = {a: J["tests"][f"{a}|{w}"] for a in cheaper}
        robust = [a for a in cheaper if rows[a]["ci95"][0] > 0]
        if robust:
            v = "R"
        elif cheaper and all(rows[a]["ci95"][1] < 0 for a in cheaper):
            v = "N"
        else:
            v = "C"
        J["verdicts"][w] = {"cheaper": cheaper, "robust_by": robust, "verdict": v,
                            "cheapest_robust": robust[0] if robust else None}
    json.dump(J, open(os.path.join(REV3, "tables", "math_easy.json"), "w"), indent=1)

    L = ["% AUTO-GENERATED by paper/arxiv_rev3/scripts/emit_math_easy.py from",
         "% results/power/math500_qwen3b_s*.jsonl (reproduces results/extra/math_interaction_analysis.json). Do not edit."]
    mac = lambda n, v: L.append(f"\\newcommand{{\\{n}}}{{{v}}}")
    mac("MeN", str(len(ids)))
    for c in cfgs:
        d = J["configs"][c]; n = NAME[c]
        mac(f"Me{n}Acc", f"{d['acc']:.1f}")
        mac(f"Me{n}CI", f"[{d['lo']:.0f},{d['hi']:.0f}]")
        mac(f"Me{n}Tok", fmt_int(d["total_tokens"]))
        mac(f"Me{n}Out", fmt_int(d["out_tokens"]))
    for key, t in J["tests"].items():
        a, w = key.split("|")
        n = NAME[a] + "Minus" + NAME[w]
        mac(f"MeD{n}", f"${sgn(t['diff'])}$")
        mac(f"MeI{n}", f"$[{sgn(t['ci95'][0])},{sgn(t['ci95'][1])}]$")
        pv = "{<}0.001" if t["p"] < 0.001 else f"{t['p']:.3f}" if t["p"] < 0.1 else f"{t['p']:.2f}"
        mac(f"MeP{n}", "p" + (pv if pv.startswith("{") else "=" + pv))
        mac(f"MePv{n}", pv)
    for w, v in J["verdicts"].items():
        mac(f"MeV{NAME[w]}", v["verdict"])
    open(os.path.join(REV3, "math_easy_macros.tex"), "w").write("\n".join(L) + "\n")
    for c in cfgs:
        d = J["configs"][c]
        print(f"{c:14s} {d['acc']:6.2f} [{d['lo']:.1f},{d['hi']:.1f}] tot {d['total_tokens']:8.1f} out {d['out_tokens']:8.1f}")
    for w, v in J["verdicts"].items():
        print(w, v)


if __name__ == "__main__":
    main()
