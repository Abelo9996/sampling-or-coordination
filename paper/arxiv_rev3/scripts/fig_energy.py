#!/usr/bin/env python3
"""rev3 figure: per-call GPU energy on HumanEval (first run, RTX 4060) against output tokens per call,
with the paper's energy model: least squares J = a*out + b*in + c with the intercept constrained
to c >= 0 (the unconstrained fit has c = -7.5 J, not physical; the constrained fit has c = 0).

Output: paper/arxiv_rev3/figures/fig_energy.pdf (HALF_W wide, drawn at printed size).
Points: every call of every record of the HumanEval matrix (seven configurations, 2,132 records,
10,004 calls), shaded by the call's input tokens (log color scale). Lines: the fitted model
evaluated at the median and at the 95th percentile of input tokens per call.

Source (read only): results/humaneval_matrix_pc.jsonl (scripts/emit_humaneval.py LOG). The fit is
fit_energy() of scripts/analyze_rev3_costs.py (imported read-only), which also asserts the
unconstrained fit against paper/arxiv_rev3/humaneval_macros.tex (\\regOutCode, \\regInCode,
\\regIcptCode, \\regRsqCode).

Assertions (against paper/arxiv_rev3/rev3_reruns_macros.tex, to the printed precision; the same
values as results/extra/rev3_costs_analysis.json energy_fit):
  a  = per output token  f"{a:.2f}"  == \\RrEnOut
  b  = per input token   f"{b:.3f}"  == \\RrEnIn
  c  = intercept         == 0        (\\RrEnIcpt = 0.0)
  R^2                    f"{r2:.3f}" == \\RrEnRsq
  a/b                    f"{a/b:.0f}" == \\RrEnRatioRound
plus the completeness checks of emit_humaneval.py (164 items, 2,132 records, energy on every record).
The annotation text is formatted from the computed values.

Run: /usr/bin/python3 paper/arxiv_rev3/scripts/fig_energy.py
"""
import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import plotstyle as fs  # noqa: E402

ROOT = fs.ROOT
sys.path.insert(0, ROOT)
from harness import analyze  # noqa: E402

spec = importlib.util.spec_from_file_location("eh", os.path.join(ROOT, "scripts/emit_humaneval.py"))
eh = importlib.util.module_from_spec(spec); spec.loader.exec_module(eh)
spec = importlib.util.spec_from_file_location("costs", os.path.join(ROOT, "scripts/analyze_rev3_costs.py"))
costs = importlib.util.module_from_spec(spec); spec.loader.exec_module(costs)


def fit():
    rows = analyze.load(eh.LOG)
    items = sorted({r["item_id"] for r in rows})
    expect = len(items) * sum(eh.N_SEEDS.get(c, 2) for c in eh.ORDER)
    assert len(rows) == expect == 2132 and len(items) == 164, (len(rows), expect, len(items))
    assert not any(r.get("energy_j") is None or r.get("energy_j_net") is None for r in rows)
    calls = [c for r in rows for c in r["calls"]]
    X = np.array([[1.0, c["in_tokens"], c["out_tokens"]] for c in calls])
    y = np.array([c["energy_j"] for c in calls])
    E = costs.fit_energy()
    c0, b_in, a_out, r2 = E["intercept_J_per_call"], E["J_per_input_token"], E["J_per_output_token"], E["r2"]
    assert E["n_calls"] == len(calls)
    with np.errstate(all="ignore"):
        res = y - X @ np.array([c0, b_in, a_out])
    assert abs(1 - (res ** 2).sum() / ((y - y.mean()) ** 2).sum() - r2) < 1e-12
    m = fs.macros("rev3_reruns_macros.tex")
    assert f"{a_out:.2f}" == m["RrEnOut"], (a_out, m["RrEnOut"])
    assert f"{b_in:.3f}" == m["RrEnIn"], (b_in, m["RrEnIn"])
    assert c0 == 0.0 and m["RrEnIcpt"] == "0.0", (c0, m["RrEnIcpt"])
    assert f"{r2:.3f}" == m["RrEnRsq"], (r2, m["RrEnRsq"])
    assert f"{a_out / b_in:.0f}" == m["RrEnRatioRound"], (a_out / b_in, m["RrEnRatioRound"])
    return X, y, (c0, b_in, a_out), r2


def main():
    X, y, beta, r2 = fit()
    c0, b_in, a_out = beta
    fs.setup()
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    fig, ax = plt.subplots(figsize=(fs.HALF_W, 2.2))
    order = np.argsort(X[:, 1])  # draw prompt-heavy calls last so they stay visible
    norm = LogNorm(vmin=X[:, 1].min(), vmax=X[:, 1].max())
    sc = ax.scatter(X[order, 2], y[order], c=X[order, 1], cmap="viridis_r", norm=norm, s=2.5,
                    alpha=0.45, lw=0, rasterized=True, zorder=2)
    xo = np.array([0.0, 1024.0])
    for q, ls in ((50, "-"), (95, "--")):
        xin = float(np.percentile(X[:, 1], q))
        ax.plot(xo, a_out * xo + b_in * xin + c0, ls, color=fs.TEXT, lw=0.9, zorder=3,
                label=f"{fs.fmt_int(xin)} ({'median' if q == 50 else 'p95'})")
    # fit summary alone in the empty upper left, legend in the empty lower right (below the lines)
    ax.legend(loc="lower right", title="fit at input tokens", alignment="left", handlelength=1.5,
              borderaxespad=0.3)
    txt = (f"{a_out:.2f} J per output token\n{b_in:.3f} J per input token\n"
           f"intercept \u2265 0 (fitted 0)\n$R^2$ = {r2:.3f}")
    tb = ax.text(0.03, 0.98, txt, transform=ax.transAxes, ha="left", va="top", color=fs.TEXT,
                 linespacing=1.2)
    ax.set_xlim(0, 1060); ax.set_ylim(0, 1400)
    ax.set_xticks([0, 256, 512, 768, 1024])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: fs.fmt_int(v)))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: fs.fmt_int(v)))
    ax.set_xlabel("output tokens per call")
    ax.set_ylabel("GPU energy per call (J)")
    cb = fig.colorbar(sc, ax=ax, fraction=0.08, pad=0.03, aspect=25)
    cb.set_ticks([100, 300, 1000, 3000])
    cb.set_ticklabels([fs.fmt_int(t) for t in (100, 300, 1000, 3000)])
    cb.ax.minorticks_off()
    cb.set_label("input tokens per call (log scale)", labelpad=2)
    cb.solids.set(alpha=1, rasterized=True)
    # the fit summary and the legend touch no point and no fit line (3 pt clearance)
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    k = fig.dpi / 72.0
    P = ax.transData.transform(np.c_[X[:, 2], y])
    L = np.concatenate([ax.transData.transform(np.c_[np.linspace(0, 1024, 400), l.get_ydata()[0]
                        + (l.get_ydata()[1] - l.get_ydata()[0]) * np.linspace(0, 1, 400)])
                        for l in ax.get_lines()])
    for art in (tb, ax.get_legend()):
        bb = art.get_window_extent(rend).expanded(1, 1)
        x0, y0, x1, y1 = bb.x0 - 3 * k, bb.y0 - 3 * k, bb.x1 + 3 * k, bb.y1 + 3 * k
        for Q in (P, L):
            hit = (Q[:, 0] > x0) & (Q[:, 0] < x1) & (Q[:, 1] > y0) & (Q[:, 1] < y1)
            assert not hit.any(), (art, int(hit.sum()), ax.transData.inverted().transform(Q[hit]))
    fs.save(fig, "fig_energy")
    print("a %.4f b %.5f c %.3f R2 %.5f ratio %.2f n %d" % (a_out, b_in, c0, r2, a_out / b_in, len(y)))

if __name__ == "__main__":
    main()
