"""Combined cross-task figure: one panel per task, each showing the independent-sampling
test-time curve (SA+SC) as a line and the multi-agent topologies as markers.
The headline generalization visual: the independent-sampling frontier on
every task. Skips tasks whose results file is absent, so it works incrementally.

  .venv/bin/python scripts/make_combined_figure.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from harness import analyze  # noqa: E402

TASKS = [("gsm8k", "GSM8K (math reasoning)"),
         ("mmlu", "MMLU (knowledge MCQ)"),
         ("math500", "MATH (levels 1-3)")]
STYLE = {"SA": ("#000000", "o", "single agent"),
         "SC": ("#1f77b4", "o", None),
         "DEBATE": ("#d62728", "s", "debate"),
         "GE": ("#2ca02c", "^", "gen-eval"),
         "HIER": ("#9467bd", "D", "hierarchical")}


def panel(ax, agg, title):
    sc = analyze.sc_frontier(agg)
    if sc:
        ax.plot([t[1] for t in sc], [t[2] * 100 for t in sc], "-",
                color=STYLE["SC"][0], lw=1.6, zorder=2)
    shown = set()
    for c, a in agg.items():
        col, mk, lab = STYLE.get(a["topology"], ("#777", "x", None))
        leglab = None
        if a["topology"] in ("SA", "DEBATE", "GE", "HIER") and a["topology"] not in shown:
            leglab = {"SA": "single agent (SA+SC)", "DEBATE": "debate",
                      "GE": "gen-eval", "HIER": "hierarchical"}[a["topology"]]
            shown.add(a["topology"])
        ax.errorbar(a["tokens"], a["accuracy"] * 100,
                    yerr=[[(a["accuracy"] - a["acc_lo"]) * 100],
                          [(a["acc_hi"] - a["accuracy"]) * 100]],
                    fmt=mk, color=col, ms=6, capsize=2, zorder=3, label=leglab)
    ax.set_xscale("log")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("tokens / query (log)")
    ax.grid(True, which="both", ls=":", alpha=0.4)


def main():
    avail = [(t, lbl) for t, lbl in TASKS
             if os.path.exists(os.path.join(ROOT, "results", f"{t}_matrix.jsonl"))]
    if not avail:
        print("no results yet"); return
    fig, axes = plt.subplots(1, len(avail), figsize=(4.6 * len(avail), 4.0),
                             squeeze=False)
    for ax, (t, lbl) in zip(axes[0], avail):
        agg = analyze.aggregate(analyze.load(
            os.path.join(ROOT, "results", f"{t}_matrix.jsonl")))
        panel(ax, agg, lbl)
    axes[0][0].set_ylabel("accuracy (%)")
    axes[0][0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Measured cost-accuracy results across the three evaluated task samples",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    out = os.path.join(ROOT, "figures", "combined_frontier.pdf")
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out, f"({len(avail)} task panels)")


if __name__ == "__main__":
    main()
