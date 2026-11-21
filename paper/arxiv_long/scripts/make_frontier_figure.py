"""Frontier figure for the arXiv version: the poster's Fig. 1
(scripts/make_aaibd_figure.py) with the same data, aggregation and styles,
redrawn at the single-column text width (6.5 in) with larger fonts. Only the
layout and font sizes differ. Output: paper/arxiv_long/figures/frontier_large.pdf.

Run: python3 paper/arxiv_long/scripts/make_frontier_figure.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import make_aaibd_figure as M  # noqa: E402

plt = M.plt
OUT = os.path.join(HERE, "..", "figures", "frontier_large.pdf")
FS = 8.5


def panel(ax, a, title):
    samp = sorted([c for c in a if a[c]["topo"] in ("SA", "SC")], key=lambda c: a[c]["tok"])
    ax.plot([a[c]["tok"] for c in samp], [a[c]["acc"] for c in samp], "-",
            color=M.STYLE["SC"][0], lw=1.3, zorder=2)
    for c, v in a.items():
        key = "SC" if v["topo"] in ("SA", "SC") else v["topo"]
        col, mk, _ = M.STYLE[key]
        hollow = v["nseed"] == 1 and c != "SA"
        ax.errorbar(v["tok"], v["acc"], yerr=[[v["acc"] - v["lo"]], [v["hi"] - v["acc"]]],
                    fmt=mk, color=col, ms=5, capsize=1.8, elinewidth=0.8, zorder=3,
                    mfc="white" if hollow else col, mew=1.0)
        lab = "SA" if c == "SA" else (f"$k{{=}}{c.split('k')[1]}$" if v["topo"] == "SC" else None)
        if lab:
            off, ha = {"SC_k8": ((-5, 4), "right"), "SC_k12": ((0, -16), "center")}.get(c, ((3, -13), "left"))
            ax.annotate(lab, (v["tok"], v["acc"]), xytext=off, ha=ha,
                        textcoords="offset points", fontsize=FS - 1, color="#333333")
    ax.set_xscale("log")
    ax.set_title(title, fontsize=FS)
    ax.set_xlabel("total tokens / query (log)", fontsize=FS)
    ax.tick_params(labelsize=FS - 1)
    ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.5)


def main():
    mrows, sc24 = M.math_rows()
    assert sc24
    panels = [(M.agg(M.read("gsm8k_matrix.jsonl")), "GSM8K ($n{=}80$, 640 cap)"),
              (M.agg(M.read("mmlu_matrix.jsonl")), "MMLU ($n{=}80$, 640 cap)"),
              (M.agg(mrows), "MATH L1-3 ($n{=}60$, 1024 cap)")]
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.75))
    for ax, (a, t) in zip(axes, panels):
        panel(ax, a, t)
    axes[0].set_ylabel("accuracy (%)", fontsize=FS)
    handles = [plt.Line2D([], [], color=c, marker=m, ls="-" if k == "SC" else "",
                          ms=5, lw=1.3, label=l) for k, (c, m, l) in M.STYLE.items()]
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=FS - 0.5,
               frameon=False, bbox_to_anchor=(0.5, 1.09))
    fig.tight_layout(pad=0.3, w_pad=0.8)
    fig.savefig(OUT, bbox_inches="tight")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
