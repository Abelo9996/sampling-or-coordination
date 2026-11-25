#!/usr/bin/env python3
"""rev3 figure: price-weighted cost of SC k=8 minus that of debate as a function of the
output:input price ratio r.

Output: paper/arxiv_rev3/figures/fig_price_crossover.pdf (HALF_W wide, drawn at printed size).
Grey bands: the 95% bootstrap intervals of the break-even ratios (items, then seeds within item,
10,000 replicates), results/extra/rev3_costs_analysis.json crossover.<task>.SC_k8.ci95, asserted
against \\RrCrossCIGsm and \\RrCrossCIMath; the break-even ratios are x-axis ticks.
cost(config) = mean input tokens + r x mean output tokens per query, input = total - output.
y(r) = cost(SC k=8) - cost(debate); one line for GSM8K (rev3 rerun fold-in: the 500-item,
three-seed run, results/extra/rev3_reruns_analysis.json, asserted against \\RrCrossGsm; 640 cap) and one for MATH
levels 1 to 3 (the 238-item, three-seed run of Table 1, 1,024 cap; audit M1); break-even
r* = -(dIn / dOut) marked.

Sources (read only):
  paper/arxiv_rev3/tables/review_revision.json  tasks.gsm8k / tasks.math, configs SC_k8 and
                                                 DEBATE_N3_R1, fields total_tokens and out_tokens
  paper/arxiv_rev3/tables/math_easy.json         configs SC_k8 and DEBATE_N3_R1 (238-item MATH run)
  paper/arxiv_rev3/rev3_macros.tex               \\RthCrossGsm, \\RthCrossMath

Assertions: r* rounds (2 dp) to 7.97 (GSM8K) and 5.27 (MATH, recomputed independently by the
audit), equal to \\RthCrossGsm and \\RthCrossMath; for GSM8K the pw2..pw5 fields of
review_revision.json equal input + r x output (so the cost definition matches the one the paper
used); SC k=8 total tokens equal Table 1 (3,224 GSM8K; MATH \\MeSceightTok).

Run: /usr/bin/python3 paper/arxiv_rev3/scripts/fig_price_crossover.py
"""
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import plotstyle as fs  # noqa: E402

TASKS = [("gsm8k", "GSM8K", "Gsm", "#333333", "-"), ("math", "MATH L1-3", "Math", "#333333", "--")]


def load():
    J = json.load(open(os.path.join(fs.REV3, "tables/review_revision.json")))["tasks"]
    ME = json.load(open(os.path.join(fs.REV3, "tables/math_easy.json")))["configs"]
    mac = fs.macros("rev3_macros.tex", "math_easy_macros.tex")
    rmac = fs.macros("rev3_reruns_macros.tex")
    RA = json.load(open(os.path.join(fs.ROOT, "results/extra/rev3_reruns_analysis.json")))["tasks"]
    t1 = open(os.path.join(fs.REV3, "tables/poster_main_table.tex")).read()
    row = [l for l in t1.splitlines() if l.startswith("SC $k{=}8$")][0]
    out = {}
    for key, *_ in TASKS:
        d = {}
        if key == "gsm8k":
            # rev3 rerun fold-in: GSM8K 500 x 3 (phase A), results/extra/rev3_reruns_analysis.json
            C = RA["gsm8k"]["configs"]
            for c in ("SC_k8", "DEBATE_N3_R1"):
                d[c] = (C[c]["total_tokens"] - C[c]["out_tokens"], C[c]["out_tokens"])
            assert f"{C['SC_k8']['total_tokens']:,.0f}".replace(",", "{,}") == rmac["RrGsmSceightTok"]
        else:
            for c in ("SC_k8", "DEBATE_N3_R1"):
                d[c] = (ME[c]["total_tokens"] - ME[c]["out_tokens"], ME[c]["out_tokens"])
            assert f"{ME['SC_k8']['total_tokens']:,.0f}".replace(",", "{,}") == mac["MeSceightTok"]
        din = d["SC_k8"][0] - d["DEBATE_N3_R1"][0]
        dout = d["SC_k8"][1] - d["DEBATE_N3_R1"][1]
        rstar = -din / dout
        if key == "math":
            assert f"{rstar:.2f}" == "5.27", (key, rstar)
            assert f"{rstar:.2f}" == mac["RthCrossMath"], (key, rstar)
        else:
            assert f"{rstar:.2f}" == rmac["RrCrossGsm"], (key, rstar)
        out[key] = (din, dout, rstar)
    return out


def intervals():
    CA = json.load(open(os.path.join(fs.ROOT, "results/extra/rev3_costs_analysis.json")))["crossover"]
    rmac = fs.macros("rev3_reruns_macros.tex")
    out = {}
    for key, tn in (("gsm8k", "Gsm"), ("math", "Math")):
        lo, hi = CA[key]["SC_k8"]["ci95"]
        assert f"$[{lo:.1f},{hi:.1f}]$" == rmac[f"RrCrossCI{tn}"], (key, lo, hi)
        out[key] = (lo, hi)
    return out


def main():
    V = load()
    CI = intervals()
    fs.setup()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    fig, ax = plt.subplots(figsize=(fs.HALF_W, 2.2))
    XHI = 20
    assert max(hi for _, hi in CI.values()) < XHI
    r = np.linspace(1, XHI, 300)
    # y range holds both lines over the whole r range (no line leaves the axes)
    ends = [V[k][0] + rr * V[k][1] for k in V for rr in (1, XHI)]
    YLO = -500 * np.ceil((-min(ends) + 300) / 500)
    YHI = 500 * np.ceil((max(ends) + 300) / 500)
    for key in CI:
        ax.axvspan(*CI[key], color=fs.BAND, lw=0, zorder=0)
    ax.axhline(0, color="#777777", lw=0.6, zorder=1)
    hs = []
    for key, lab, _, col, ls in TASKS:
        din, dout, rs = V[key]
        hs += ax.plot(r, din + r * dout, ls, color=col, lw=fs.LW, zorder=3, label=lab)
        ax.axvline(rs, color="#555555", lw=0.6, ls=":", zorder=2)
    hs.append(Patch(color=fs.BAND, lw=0, label="95% interval"))
    fig.legend(handles=hs, loc="outside upper center", ncol=3, handlelength=1.5, columnspacing=0.8)
    ax.text(9.0, YLO + 0.03 * (YHI - YLO), r"SC $k$=8" "\n" "cheaper", ha="center", va="bottom",
            color=fs.MUTED,
            style="italic", linespacing=1.1)
    ax.set_xlim(1, XHI); ax.set_ylim(YLO, YHI)
    br = sorted(V[k][2] for k in V)
    ax.set_xticks([1, br[0], 10, br[1], XHI])
    ax.set_xticklabels(["1", f"{br[0]:.2f}", "10", f"{br[1]:.2f}", str(XHI)])
    ax.grid(False, axis="x")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: fs.minus(fs.fmt_int(y))))
    ax.set_xlabel("output:input price ratio $r$")
    ax.set_ylabel(r"SC $k$=8 minus debate" "\n" "(price-weighted tokens / query)")
    fs.save(fig, "fig_price_crossover")
    for key in V:
        print(key, "dIn %.2f dOut %.2f r* %.4f" % V[key], "CI", CI[key])

if __name__ == "__main__":
    main()
