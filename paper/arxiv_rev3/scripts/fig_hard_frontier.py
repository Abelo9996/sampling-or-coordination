#!/usr/bin/env python3
"""rev3 figure: accuracy against cost on MATH-500 levels 4 to 5 (262 items, seeds 1 to 3, cap 2,048).

Output: paper/arxiv_rev3/figures/fig_hard_frontier.pdf (full text width, printed=1.0).
Left panel: accuracy vs mean total tokens per query (log x). Right panel: accuracy vs mean output
tokens per query (log x). Sampling curve SA, SC k=2,4,8,12 (SC k = first k samples of each SC k=12
record, majority vote with harness.topologies._majority, as in scripts/analyze_hard_math.py);
coordination points debate, GE, HIER and Du et al. debate.

Sources (read only):
  results/power/math500hard_qwen3b_s*.jsonl   per-record logs of the hard-MATH run
  results/power/extra_math45_qwen3b_s*.jsonl  per-record logs of the Du et al. debate, same items
  scripts/analyze_hard_math.py                loaders and SC-k logic (imported with importlib,
                                              exactly as paper/arxiv_rev3/scripts/emit_math_easy.py)
  results/extra/hard_math_analysis.json       accuracy, cost, du (published analysis, 2 and 1 dp)
  paper/arxiv_rev3/tables/hard_math.json      acc, total_tokens, out_tokens (unrounded)
  paper/arxiv_rev3/hard_math_macros.tex       \\HmAcc<cfg> (1 dp) and \\HmTok<cfg> (integer)

Assertions: for every plotted configuration the accuracy recomputed from the records equals
hard_math.json acc (to 1e-9), hard_math_analysis.json accuracy (2 dp) and \\HmAcc<cfg> (1 dp); the
mean total and output tokens equal hard_math.json (to 1e-6), hard_math_analysis.json cost / du (1 dp)
and \\HmTok<cfg> (integer, total tokens). The item set is the 262 items of the analysis.

Error bars: 95% item-bootstrap interval of the per-item seed-averaged correctness,
harness.analyze._bootstrap_ci(corr, B=10000) (seed 0), the convention of Table 1. The paper prints no
interval for these cells, so the bars are computed here and not asserted against a printed value.
Cost-matched comparators (from hard_math_analysis.json, total tokens: debate -> SC k=8, GE and
HIER -> SC k=4; output tokens: debate -> SC k=4, GE and HIER -> SC k=2) are named in the caption
and asserted here, not drawn.

Run: /usr/bin/python3 paper/arxiv_rev3/scripts/fig_hard_frontier.py
"""
import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import figstyle as fs  # noqa: E402

ROOT = fs.ROOT
sys.path.insert(0, ROOT)
from harness import analyze, tasks, topologies  # noqa: E402

spec = importlib.util.spec_from_file_location("hm", os.path.join(ROOT, "scripts/analyze_hard_math.py"))
hm = importlib.util.module_from_spec(spec); spec.loader.exec_module(hm)

SEEDS = [1, 2, 3]
COORD = ["DEBATE_N3_R1", "GE_R2", "HIER_M3"]
KS = [2, 4, 8, 12]
SAMP = ["SA"] + [f"SC_k{k}" for k in KS]
MAC = {"SA": "Sa", "SC_k2": "Sctwo", "SC_k4": "Scfour", "SC_k8": "Sceight", "SC_k12": "Sctwelve",
       "DEBATE_N3_R1": "Deb", "GE_R2": "Ge", "HIER_M3": "Hier", "DU": "Du"}
KEYC = {"DEBATE_N3_R1": "DEBATE", "GE_R2": "GE", "HIER_M3": "HIER", "DU": "DU"}


def recompute():
    task = tasks.get_task("math500")
    items = hm.camp.load_math_levels(tasks, 4, 5, 262)
    recs = hm.load("results/power/math500hard_qwen3b_s*.jsonl")
    du = hm.load("results/power/extra_math45_qwen3b_s*.jsonl")
    ids = [it["id"] for it in items]
    assert len(ids) == 262
    corr, tot, out = {}, {}, {}
    for c in ["SA"] + COORD:
        R = {i: [recs[(c, i, s)] for s in ([1] if c == "SA" else SEEDS)] for i in ids}
        corr[c] = np.array([np.mean([float(bool(r["correct"])) for r in R[i]]) for i in ids])
        allr = [r for i in ids for r in R[i]]
        tot[c] = float(np.mean([sum(x["in_tokens"] + x["out_tokens"] for x in r["calls"]) for r in allr]))
        out[c] = float(np.mean([sum(x["out_tokens"] for x in r["calls"]) for r in allr]))
    for k in KS:
        v, tt, oo = [], [], []
        for i in ids:
            vi = []
            for s in SEEDS:
                r = recs[("SC_k12", i, s)]; cs = r["calls"][:k]
                ans = topologies._majority([x["extracted"] for x in cs])
                ok = bool(task["score"](ans, r["gold"]))
                if k == 12:
                    assert ok == bool(r["correct"]), (i, s)
                vi.append(float(ok))
                tt.append(sum(x["in_tokens"] + x["out_tokens"] for x in cs))
                oo.append(sum(x["out_tokens"] for x in cs))
            v.append(np.mean(vi))
        c = f"SC_k{k}"
        corr[c] = np.array(v); tot[c] = float(np.mean(tt)); out[c] = float(np.mean(oo))
    R = [du[("DU_DEBATE_N3_R2", i, s)] for i in ids for s in SEEDS]
    corr["DU"] = np.array([np.mean([float(bool(du[("DU_DEBATE_N3_R2", i, s)]["correct"])) for s in SEEDS])
                           for i in ids])
    tot["DU"] = float(np.mean([r["total_tokens"] for r in R]))
    out["DU"] = float(np.mean([r["out_tokens"] for r in R]))
    P = {}
    for c in corr:
        lo, hi = analyze._bootstrap_ci(corr[c], B=10000)
        P[c] = {"acc": 100 * float(corr[c].mean()), "lo": 100 * lo, "hi": 100 * hi,
                "tot": tot[c], "out": out[c]}
    return P


def check(P):
    A = json.load(open(os.path.join(ROOT, "results/extra/hard_math_analysis.json")))
    T = json.load(open(os.path.join(fs.REV3, "tables/hard_math.json")))
    mac = fs.macros("hard_math_macros.tex")
    assert A["n_items_complete"] == 262 and A["complete"]
    for c, p in P.items():
        assert abs(p["acc"] - T["acc"][c]) < 1e-9, (c, p["acc"], T["acc"][c])
        assert abs(p["tot"] - T["total_tokens"][c]) < 1e-6, (c, p["tot"])
        assert abs(p["out"] - T["out_tokens"][c]) < 1e-6, (c, p["out"])
        if c == "DU":
            a, t, o = A["du"]["accuracy"], A["du"]["total_tokens"], A["du"]["out_tokens"]
            assert round(p["acc"], 2) == a, (c, p["acc"], a)
        else:
            a, t, o = A["accuracy"][c], A["cost"][c]["total_tokens"], A["cost"][c]["out_tokens"]
            assert round(p["acc"], 2) == a, (c, p["acc"], a)
        assert round(p["tot"], 1) == t and round(p["out"], 1) == o, (c, p["tot"], t, p["out"], o)
        assert f"{p['acc']:.1f}" == mac[f"HmAcc{MAC[c]}"], (c, p["acc"], mac[f"HmAcc{MAC[c]}"])
        assert fs.fmt_int(p["tot"]).replace(",", "{,}") == mac[f"HmTok{MAC[c]}"], (c, p["tot"])
    comp = {"total": {w: A["primary_total_tokens"][w]["comparator"] for w in COORD},
            "out": {w: A["output_tokens"][w]["comparator"] for w in COORD}}
    assert comp["total"] == {"DEBATE_N3_R1": "SC_k8", "GE_R2": "SC_k4", "HIER_M3": "SC_k4"}, comp
    assert comp["out"] == {"DEBATE_N3_R1": "SC_k4", "GE_R2": "SC_k2", "HIER_M3": "SC_k2"}, comp
    return comp


# label offsets (points) per panel: (dx, dy, ha, va)
OFF = {
    "tot": {"SA": (5, -4, "left", "top"), "SC_k2": (4, -5, "left", "top"),
            "SC_k4": (-5, 5, "right", "bottom"), "SC_k8": (-5, 5, "right", "bottom"),
            "SC_k12": (5, 3, "left", "bottom"),
            "DEBATE_N3_R1": (5, -2, "left", "top"), "GE_R2": (5, 1, "left", "center"),
            "HIER_M3": (-6, -1, "right", "center"), "DU": (6, 2, "left", "center")},
    "out": {"SA": (5, -4, "left", "top"), "SC_k2": (-4, 5, "right", "bottom"),
            "SC_k4": (-5, 5, "right", "bottom"), "SC_k8": (-5, 5, "right", "bottom"),
            "SC_k12": (5, 3, "left", "bottom"),
            "DEBATE_N3_R1": (6, -5, "left", "top"), "GE_R2": (-6, 1, "right", "center"),
            "HIER_M3": (-6, -1, "right", "center"), "DU": (6, 0, "left", "top")},
}


def panel(ax, P, key, xlabel):
    xs = [P[c][key] for c in SAMP]; ys = [P[c]["acc"] for c in SAMP]
    ax.plot(xs, ys, "-", color=fs.COL["SC"], lw=1.1, zorder=2)
    for c in SAMP + COORD + ["DU"]:
        p = P[c]
        kc = "SA" if c == "SA" else ("SC" if c.startswith("SC") else KEYC[c])
        col = fs.COL[kc]
        ax.errorbar(p[key], p["acc"], yerr=[[p["acc"] - p["lo"]], [p["hi"] - p["acc"]]], fmt="none",
                    ecolor=fs.tint(col, 0.55), elinewidth=0.6, capsize=0, zorder=1)
        hollow = c == "DU"
        ax.plot(p[key], p["acc"], fs.MARK[kc], color=col, ms=4.3 if kc in ("SA", "SC") else 4.8,
                mfc="white" if hollow else col, mew=0.9, zorder=3)
        lab = fs.NAME[c] if c in ("SA",) + tuple(COORD) + ("DU",) else rf"$k$={c.split('k')[1]}"
        dx, dy, ha, va = OFF[key][c]
        ax.annotate(lab, (p[key], p["acc"]), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                    fontsize=7.5, color=col if kc not in ("SA", "SC") else "#333333", zorder=4)
    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.grid(True, which="major", ls=":", lw=0.5, alpha=0.6)
    ax.set_ylim(34, 62)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    P = recompute()
    check(P)
    fs.setup()
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    fig, axs = plt.subplots(1, 2, figsize=(fs.TEXTW, 2.3), sharey=True)
    panel(axs[0], P, "tot", "mean total tokens per query (log scale)")
    panel(axs[1], P, "out", "mean output tokens per query (log scale)")
    axs[0].set_ylabel("accuracy (%)")
    axs[0].set_xlim(600, 32000); axs[1].set_xlim(500, 14000)
    for ax, ticks in ((axs[0], [1000, 2000, 5000, 10000]), (axs[1], [1000, 2000, 5000, 10000])):
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: fs.fmt_int(x)))
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.175, top=0.97, wspace=0.08)
    fs.save(fig, "fig_hard_frontier", printed=1.0)
    for c in SAMP + COORD + ["DU"]:
        p = P[c]
        print(f"{c:14s} acc {p['acc']:6.2f} [{p['lo']:.1f},{p['hi']:.1f}] tot {p['tot']:8.1f} out {p['out']:8.1f}")


if __name__ == "__main__":
    main()
