#!/usr/bin/env python3
"""rev3 figure: price-weighted cost of SC k=8 minus that of debate as a function of the
output:input price ratio r.

Output: paper/arxiv_rev3/figures/fig_price_crossover.pdf (design width 2.7 in, printed=0.49).
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
import figstyle as fs  # noqa: E402

TASKS = [("gsm8k", "GSM8K (500 items)", "Gsm", "#333333", "-"), ("math", "MATH L1-3 (238 items)", "Math", "#333333", "--")]


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


def main():
    V = load()
    fs.setup()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(2.7, 2.05))
    r = np.linspace(1, 16, 300)
    ax.axhspan(-3000, 0, color=fs.SHADE, zorder=0, lw=0)
    ax.axhline(0, color="#777777", lw=0.6, zorder=1)
    YLO, YHI = -2700, 2500
    for key, lab, _, col, ls in TASKS:
        din, dout, rs = V[key]
        ax.plot(r, din + r * dout, ls, color=col, lw=1.2, zorder=3, label=lab)
        ax.axvline(rs, color="#555555", lw=0.7, ls=":", zorder=2)
        # break-even label inside the shaded region, left of its dotted line (both lines lie
        # above it there)
        ax.annotate(f"$r$ = {rs:.2f}", (rs, YLO), xytext=(-2, 4), textcoords="offset points",
                    ha="right", va="bottom", fontsize=7.5, color="#333333", rotation=90)
    ax.legend(loc="upper left", frameon=False, handlelength=2.2, borderaxespad=0.3)
    ax.annotate(r"SC $k$=8 cheaper", (6.0, 0), xytext=(4, -3), textcoords="offset points",
                fontsize=7.5, color="#444444", ha="left", va="top")
    ax.annotate("debate cheaper", (6.0, 0), xytext=(4, 3), textcoords="offset points",
                fontsize=7.5, color="#444444", ha="left", va="bottom")
    ax.set_xlim(1, 16); ax.set_ylim(YLO, YHI)
    ax.set_xticks([1, 2, 4, 6, 8, 10, 12, 14, 16])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: fs.minus(fs.fmt_int(y))))
    ax.set_xlabel("output:input price ratio $r$")
    ax.set_ylabel(r"SC $k$=8 minus debate" "\n" "(price-weighted tokens / query)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=0.265, right=0.975, bottom=0.185, top=0.975)
    fs.save(fig, "fig_price_crossover", printed=0.49)
    for key in V:
        print(key, "dIn %.2f dOut %.2f r* %.4f" % V[key])


if __name__ == "__main__":
    main()
