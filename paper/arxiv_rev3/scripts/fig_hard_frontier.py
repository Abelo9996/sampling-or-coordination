#!/usr/bin/env python3
"""rev3 figure: accuracy against cost on MATH-500 levels 4 to 5 (262 items, seeds 1 to 3, cap 2,048).

Output: paper/arxiv_rev3/figures/fig_hard_frontier.pdf (FULL_W wide, drawn at printed size).
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
import plotstyle as fs  # noqa: E402

ROOT = fs.ROOT
sys.path.insert(0, ROOT)
from harness import analyze, tasks, topologies  # noqa: E402

import eaai_lib as E  # noqa: E402

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


def panel(ax, P, key, xlabel):
    """Same marks as the frontier figure (Fig. 3): SA black, SC blue line, error bars in each
    point's color; sampling points labeled SA, k=2, ...; workflows in the legend."""
    import matplotlib
    xs = [P[c][key] for c in SAMP]; ys = [P[c]["acc"] for c in SAMP]
    ax.plot(xs, ys, "-", color=fs.COL["SC"], lw=fs.LW, zorder=2)
    for c in SAMP + COORD + ["DU"]:
        p = P[c]
        kc = "SA" if c == "SA" else ("SC" if c.startswith("SC") else KEYC[c])
        fs.errorbar(ax, p[key], p["acc"], p["lo"], p["hi"], color=fs.COL[kc], marker=fs.MARK[kc],
                    mfc="white" if c == "DU" else None)
    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylim(34, 64)


def label(ax, P, key):
    items = [(fs.klabel(c), P[c][key], P[c]["acc"], P[c]["lo"], P[c]["hi"]) for c in SAMP]
    texts = E.place_in_column(ax, items, fontsize=fs.FONT_PT, sides=True, dirs=(1, -1), pad=1.6)
    probs = E.check_labels(ax, texts, pad=1.0)
    assert not probs, probs


def main():
    P = recompute()
    check(P)
    fs.setup()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    fig, axs = plt.subplots(1, 2, figsize=(fs.FULL_W, 2.6), sharey=True)
    panel(axs[0], P, "tot", "mean total tokens per query (log scale)")
    panel(axs[1], P, "out", "mean output tokens per query (log scale)")
    axs[0].set_ylabel("accuracy (%)")
    axs[0].set_xlim(600, 20000); axs[1].set_xlim(450, 12000)
    for ax, ticks in ((axs[0], [1000, 2000, 5000, 10000]), (axs[1], [500, 1000, 2000, 5000, 10000])):
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: fs.fmt_int(x)))
    hs = E.legend_handles(with_front=False)
    hs.append(Line2D([], [], color=fs.COL["DU"], marker=fs.MARK["DU"], mfc="white", ls="none",
                     ms=fs.MS, label="Du et al. debate"))
    fig.legend(handles=hs, loc="outside upper center", ncol=6)
    for ax, key in ((axs[0], "tot"), (axs[1], "out")):
        label(ax, P, key)
    fs.save(fig, "fig_hard_frontier")
    for c in SAMP + COORD + ["DU"]:
        p = P[c]
        print(f"{c:14s} acc {p['acc']:6.2f} [{p['lo']:.1f},{p['hi']:.1f}] tot {p['tot']:8.1f} out {p['out']:8.1f}")

if __name__ == "__main__":
    main()
