#!/usr/bin/env python3
"""rev3 figure: per-call GPU energy on HumanEval (RTX 4060) against output tokens per call, with
the least-squares fit J = a*out + b*in + c.

Output: paper/arxiv_rev3/figures/fig_energy.pdf (design width 2.7 in, printed=0.49).
Points: every call of every record of the HumanEval matrix (seven configurations, 2,132 records,
10,004 calls), shaded by the call's input tokens (log colour scale). Lines: the fitted plane
evaluated at the median and at the 95th percentile of input tokens per call.

Source (read only): results/humaneval_matrix_pc.jsonl, read with harness.analyze.load. The fit is
the exact code of scripts/emit_humaneval.py (main(), "per-call energy against input and output
tokens"): X = [1, in_tokens, out_tokens] per call over all calls of all records, y = the call's
gross GPU board energy "energy_j" (not the idle-subtracted energy_j_net, which exists per record
only), numpy.linalg.lstsq, R^2 = 1 - SSE/SST. The script is imported read-only for its LOG path.

Assertions (against paper/arxiv_rev3/humaneval_macros.tex, to the printed precision):
  a  = per output token  f"{a:.2f}"  == \\regOutCode   (1.11)
  b  = per input token   f"{b:.3f}"  == \\regInCode    (0.040)
  c  = intercept         f"{c:.1f}"  == \\regIcptCode  (-7.5)
  R^2                    f"{r2:.3f}" == \\regRsqCode   (0.996)
  a/b                    f"{a/b:.0f}" == \\regRatioCode (28)
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
import figstyle as fs  # noqa: E402

ROOT = fs.ROOT
sys.path.insert(0, ROOT)
from harness import analyze  # noqa: E402

spec = importlib.util.spec_from_file_location("eh", os.path.join(ROOT, "scripts/emit_humaneval.py"))
eh = importlib.util.module_from_spec(spec); spec.loader.exec_module(eh)


def fit():
    rows = analyze.load(eh.LOG)
    items = sorted({r["item_id"] for r in rows})
    expect = len(items) * sum(eh.N_SEEDS.get(c, 2) for c in eh.ORDER)
    assert len(rows) == expect == 2132 and len(items) == 164, (len(rows), expect, len(items))
    assert not any(r.get("energy_j") is None or r.get("energy_j_net") is None for r in rows)
    calls = [c for r in rows for c in r["calls"]]
    # begin: exact fit code of scripts/emit_humaneval.py
    X = np.array([[1.0, c["in_tokens"], c["out_tokens"]] for c in calls])
    y = np.array([c["energy_j"] for c in calls])
    with np.errstate(all="ignore"):  # spurious BLAS warnings on macOS numpy
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        r2 = float(1 - ((y - X @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    # end: exact fit code
    m = fs.macros("humaneval_macros.tex")
    assert f"{beta[2]:.2f}" == m["regOutCode"] == "1.11", (beta[2], m["regOutCode"])
    assert f"{beta[1]:.3f}" == m["regInCode"] == "0.040", (beta[1], m["regInCode"])
    assert f"{beta[0]:.1f}" == m["regIcptCode"] == "-7.5", (beta[0], m["regIcptCode"])
    assert f"{r2:.3f}" == m["regRsqCode"] == "0.996", (r2, m["regRsqCode"])
    assert f"{beta[2] / beta[1]:.0f}" == m["regRatioCode"] == "28", (beta[2] / beta[1], m["regRatioCode"])
    return X, y, beta, r2


def main():
    X, y, beta, r2 = fit()
    c0, b_in, a_out = beta
    fs.setup()
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    fig, ax = plt.subplots(figsize=(2.7, 2.2))
    order = np.argsort(X[:, 1])  # draw prompt-heavy calls last so they stay visible
    norm = LogNorm(vmin=X[:, 1].min(), vmax=X[:, 1].max())
    sc = ax.scatter(X[order, 2], y[order], c=X[order, 1], cmap="viridis_r", norm=norm, s=2.5,
                    alpha=0.45, lw=0, rasterized=True, zorder=2)
    xo = np.array([0.0, 1024.0])
    for q, ls in ((50, "-"), (95, "--")):
        xin = float(np.percentile(X[:, 1], q))
        ax.plot(xo, a_out * xo + b_in * xin + c0, ls, color="#222222", lw=0.9, zorder=3,
                label=f"{fs.fmt_int(xin)} ({'median' if q == 50 else 'p95'})")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.76), title="fit at input tokens", title_fontsize=7.5, alignment="left", frameon=False, handlelength=1.5, borderaxespad=0.1, labelspacing=0.3, handletextpad=0.5)
    txt = (f"{a_out:.2f} J per output token\n{b_in:.3f} J per input token\n"
           f"$R^2$ = {r2:.3f}")
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, ha="left", va="top", fontsize=7.5,
            color="#222222", linespacing=1.25)
    ax.set_xlim(0, 1060); ax.set_ylim(0, 1320)
    ax.set_xticks([0, 256, 512, 768, 1024])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: fs.fmt_int(v)))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: fs.fmt_int(v)))
    ax.set_xlabel("output tokens per call")
    ax.set_ylabel("GPU energy per call (J)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=0.19, right=0.75, bottom=0.175, top=0.975)
    pos = ax.get_position()
    cax = fig.add_axes([pos.x1 + 0.025, pos.y0, 0.03, pos.height])
    cb = fig.colorbar(sc, cax=cax)
    cb.set_ticks([100, 300, 1000, 3000])
    cb.set_ticklabels([fs.fmt_int(t) for t in (100, 300, 1000, 3000)])
    cb.ax.minorticks_off()
    cb.set_label("input tokens per call (log scale)", fontsize=7.5, labelpad=2)
    cb.ax.tick_params(labelsize=7.5)
    cb.solids.set(alpha=1, rasterized=True)
    fs.save(fig, "fig_energy", printed=0.49)
    print("a %.4f b %.5f c %.3f R2 %.5f ratio %.2f n %d" % (a_out, b_in, c0, r2, a_out / b_in, len(y)))


if __name__ == "__main__":
    main()
