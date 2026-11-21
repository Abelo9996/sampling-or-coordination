"""rev3 workflow schematic (calls per query), re-rendered with plotstyle.

Copy of scripts/make_eaai_figures.py:fig_workflows() as used by
paper/arxiv_long/scripts/make_workflows_figure.py (which produced
paper/arxiv_rev2/figures/fig_workflows.pdf), with: plotstyle fonts and names ("SC k", "debate"),
8 pt text at the printed size, and the
debate round-0 and round-1 columns moved apart so that their header labels do not run together
(asserted from the rendered text boxes).
Sources: results/gsm8k_matrix.jsonl (calls per configuration and parameters N, R, M, k; the
call counts are asserted to equal N(R+1), 2R+1 and M+2), the SC k values present in every
matrix log (gsm8k, mmlu, math500, mmlu_pro_eng) and the k=12 note, asserted present in both
MATH logs (results/math500_trunc.jsonl and the 238-item run results/power/math500_qwen3b_s*.jsonl).
Output: paper/arxiv_rev3/figures/fig_workflows.pdf (FULL_W wide, drawn at printed size).
"""
import glob
import json
import os

import plotstyle as FS

FS.setup()
import matplotlib.pyplot as plt  # noqa: E402

import eaai_lib as E  # noqa: E402
from eaai_lib import box, arrow, R, L  # noqa: E402

COL = FS.COL
F = FS.FONT_PT   # text size (printed at 1:1)


def main():
    rows = E.load("gsm8k_matrix.jsonl")
    cpc = E.calls_per_config(rows)
    params = {r["config"]: r["params"] for r in rows}
    for c, s in cpc.items():
        assert len(s) == 1 or c.startswith("GES"), (c, s)
    ks = sorted(int(params[c]["k"]) for c in cpc if c.startswith("SC_"))
    k_all = set(ks)
    for t in ("mmlu", "math500", "mmlu_pro_eng"):
        k_all &= {int(r["params"]["k"]) for r in E.load(f"{t}_matrix.jsonl")
                  if r["config"].startswith("SC_")}
    k_gsm = sorted(set(ks) - k_all)
    k_all = sorted(k_all)
    assert k_gsm == [12], k_gsm
    k_math = {int(r["params"]["k"]) for r in E.load("math500_trunc.jsonl")
              if r["config"].startswith("SC_")}
    k_pow = set()
    for f in glob.glob(os.path.join(E.RES, "power", "math500_qwen3b_s*.jsonl")):
        k_pow |= {int(json.loads(l)["params"]["k"]) for l in open(f)
                  if l.strip() and json.loads(l)["config"].startswith("SC_")}
    assert 12 in k_math and 12 in k_pow, (k_math, k_pow)
    for c in cpc:
        if c.startswith("SC_"):
            assert cpc[c] == {int(params[c]["k"])}
    deb, ge, hier = params["DEBATE_N3_R1"], params["GE_R2"], params["HIER_M3"]
    N, Rr = int(deb["N"]), int(deb["R"])
    Rg, M = int(ge["R"]), int(hier["M"])
    nsa, ndeb = cpc["SA"].pop(), cpc["DEBATE_N3_R1"].pop()
    nge, nhier = cpc["GE_R2"].pop(), cpc["HIER_M3"].pop()
    assert ndeb == N * (Rr + 1) and nge == 2 * Rg + 1 and nhier == M + 2
    assert N == 3 and M == 3 and Rr == 1 and Rg == 2

    bh, gap, pad = 0.215, 0.075, 0.11
    stack = 3 * bh + 2 * gap
    heights = {"SA": 0.56, "SC": stack + 2 * pad + 0.08, "DEBATE": stack + 2 * pad + 0.16,
               "GE": 0.74, "HIER": stack + 2 * pad}
    order = ["SA", "SC", "DEBATE", "GE", "HIER"]
    leg_h = 0.32
    H = sum(heights.values()) + leg_h
    W = FS.FULL_W
    fig = plt.figure(figsize=(W, H), layout="none")   # one full-figure axes in inches
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    kk = ", ".join(map(str, k_all))
    label = {
        "SA": ("SA", "one greedy call", f"{nsa} call per query"),
        "SC": (r"SC $k$", "$k$ independent samples,\nmajority vote",
               f"$k$ calls, $k$ = {kk}\n($k$ = {', '.join(map(str, k_gsm))} on GSM8K and MATH)"),
        "DEBATE": ("debate", f"$N$ = {N} agents, $R$ = {Rr} round", f"{ndeb} calls = $N(R+1)$"),
        "GE": ("GE", f"generator-evaluator,\n$R$ = {Rg}", f"{nge} calls = $2R+1$"),
        "HIER": ("HIER", f"planner, $M$ = {M} executors,\naggregator", f"{nhier} calls = $M+2$"),
    }
    xq, wq = 1.40, 0.44
    xa, wa = W - 0.52, 0.47
    y = H
    heads = []
    for k in order:
        h = heights[k]
        y0 = y - h
        if k != "SA":
            ax.plot([0.05, W - 0.05], [y, y], color="#DDDDDD", lw=0.6, zorder=1)
        cy = y0 + h / 2 - (0.07 if k in ("DEBATE", "GE") else 0.0)
        c = COL[k]
        nm, desc, calls = label[k]
        ax.text(0.06, y - 0.05, nm, fontsize=9.5, weight="bold", color=c, va="top", ha="left")
        ax.text(0.06, y - 0.245, desc, fontsize=F, va="top", ha="left", color="#333333",
                linespacing=1.0)
        ax.text(0.06, y0 + 0.04, calls, fontsize=F, va="bottom", ha="left", weight="bold",
                color="#333333", linespacing=1.0)
        q = box(ax, xq, cy - bh / 2, wq, bh, "query", "#F2F2F2", "#7F7F7F", fs=F)
        ans = box(ax, xa, cy - bh / 2, wa, bh, "answer", "#FFFFFF", "#333333", weight="bold",
                  fs=F)
        fc, ec = FS.tint(c, 0.78), c
        if k == "SA":
            b = box(ax, 2.75, cy - bh / 2, 0.95, bh, "model call", fc, ec, fs=F)
            arrow(ax, R(q), L(b))
            arrow(ax, R(b), L(ans))
        elif k == "SC":
            v = box(ax, 3.95, cy - 0.17, 0.62, 0.34, "majority\nvote", "#FFFFFF", "#555555",
                    ls="--", fs=F)
            for i, t in enumerate(["sample 1", "sample 2", "sample $k$"]):
                yy = cy + (1 - i) * (bh + gap) - bh / 2
                b = box(ax, 2.45, yy, 0.8, bh, t, fc, ec, fs=F)
                arrow(ax, R(q), L(b))
                arrow(ax, R(b), L(v))
            arrow(ax, R(v), L(ans))
        elif k == "DEBATE":
            # round-0 and round-1 columns 0.62 in apart (0.36 in the rev2 figure), so the
            # two header labels have a clear gap between them
            x0, x1, w = 1.98, 3.34, 0.74
            r0, r1 = [], []
            for i in range(3):
                yy = cy + (1 - i) * (bh + gap) - bh / 2
                r0.append(box(ax, x0, yy, w, bh, f"agent {i + 1}", fc, ec, fs=F))
                r1.append(box(ax, x1, yy, w, bh, f"agent {i + 1}", fc, ec, fs=F))
            heads.append(ax.text(x0 + w / 2, cy + stack / 2 + 0.04, "round 0: answer",
                                 ha="center", va="bottom", fontsize=F, color="#333333"))
            heads.append(ax.text(x1 + w / 2, cy + stack / 2 + 0.04, "round 1: critique, update",
                                 ha="center", va="bottom", fontsize=F, color="#333333"))
            v = box(ax, 4.27, cy - 0.17, 0.56, 0.34, "majority\nvote", "#FFFFFF", "#555555",
                    ls="--", fs=F)
            for i in range(3):
                arrow(ax, R(q), L(r0[i]))
                for j in range(3):
                    if j != i:
                        arrow(ax, R(r0[j]), L(r1[i]), col="#888888", lw=0.6)
                arrow(ax, R(r1[i]), L(v))
            arrow(ax, R(v), (xa, ans[1] + ans[3] / 2))
        elif k == "GE":
            names = ["generate", "critique", "revise", "critique", "revise"]
            xs0, xs1 = 1.98, xa - 0.12
            w = 0.52
            g = (xs1 - xs0 - 5 * w) / 4
            prev = q
            bs = []
            for i, t in enumerate(names):
                b = box(ax, xs0 + i * (w + g), cy - bh / 2, w, bh, t, fc, ec, fs=F)
                arrow(ax, R(prev), L(b))
                bs.append(b)
                prev = b
            for rr in range(2):
                a0, a1 = bs[1 + 2 * rr], bs[2 + 2 * rr]
                ax.text((a0[0] + a1[0] + a1[2]) / 2, cy + bh / 2 + 0.04, f"round {rr + 1}",
                        ha="center", va="bottom", fontsize=F, color="#333333")
            arrow(ax, R(prev), (xa, ans[1] + ans[3] / 2))
        elif k == "HIER":
            p = box(ax, 2.0, cy - bh / 2, 0.66, bh, "planner", fc, ec, fs=F)
            ag = box(ax, 4.0, cy - bh / 2, 0.76, bh, "aggregator", fc, ec, fs=F)
            arrow(ax, R(q), L(p))
            for i in range(3):
                yy = cy + (1 - i) * (bh + gap) - bh / 2
                e = box(ax, 2.95, yy, 0.8, bh, f"executor {i + 1}", fc, ec, fs=F)
                arrow(ax, R(p), L(e))
                arrow(ax, R(e), L(ag))
            arrow(ax, R(ag), L(ans))
        y = y0
    ax.plot([0.05, W - 0.05], [leg_h, leg_h], color="#DDDDDD", lw=0.6)
    ly = leg_h / 2 - 0.09
    box(ax, 0.10, ly, 0.42, 0.18, "", FS.tint(COL["SC"], 0.78), COL["SC"])
    ax.text(0.58, ly + 0.09, "one model call (color = workflow)", fontsize=F, va="center")
    box(ax, 2.55, ly, 0.42, 0.18, "", "#FFFFFF", "#555555", ls="--")
    ax.text(3.03, ly + 0.09, "vote, no model call", fontsize=F, va="center")
    arrow(ax, (4.15, ly + 0.09), (4.5, ly + 0.09))
    ax.text(4.56, ly + 0.09, "text passed on", fontsize=F, va="center")
    # the two debate headers must be clearly apart (at least 0.12 in)
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    b0, b1 = (t.get_window_extent(rend) for t in heads)
    assert (b1.x0 - b0.x1) / fig.dpi >= 0.12, (b0, b1)
    FS.save(fig, "fig_workflows")
    print("fig_workflows: calls", {"SA": nsa, "SC_k": ks, "debate": ndeb, "GE": nge, "HIER": nhier})


if __name__ == "__main__":
    main()
