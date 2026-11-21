"""Generate the paper figures from a results JSONL.

Figure 1 (pareto): accuracy vs. token cost, with the single-agent test-time
curve (SA+SC) drawn as a line and the multi-agent topologies as distinct
markers, CIs as error bars, Pareto frontier highlighted. This is the headline.
Figure 2 (crossover): per-MAS accuracy gap vs. a single agent at equal token
budget.

Run with the analysis venv:  .venv/bin/python scripts/make_figures.py <jsonl>
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from harness import analyze  # noqa: E402

FIGDIR = os.path.join(ROOT, "figures")
TOPO_STYLE = {
    "SA":     ("#000000", "o"),
    "SC":     ("#1f77b4", "o"),
    "DEBATE": ("#d62728", "s"),
    "GE":     ("#2ca02c", "^"),
    "HIER":   ("#9467bd", "D"),
}


def fig_pareto(agg, path):
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    # single-agent test-time curve (SA + SC), connected
    sc = analyze.sc_frontier(agg)
    if sc:
        xs = [t[1] for t in sc]
        ys = [t[2] * 100 for t in sc]
        ax.plot(xs, ys, "-", color=TOPO_STYLE["SC"][0], lw=1.6, zorder=2,
                label="single agent (SA + self-consistency)")
    # connect each multi-agent topology's operating points into its own curve
    for topo in ("DEBATE", "GE", "HIER"):
        pts = sorted(((a["tokens"], a["accuracy"] * 100)
                      for a in agg.values() if a["topology"] == topo),
                     key=lambda p: p[0])
        if len(pts) >= 2:
            col = TOPO_STYLE[topo][0]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "--",
                    color=col, lw=1.2, alpha=0.7, zorder=2)
    # all configs with CIs
    seen = set()
    for c, a in agg.items():
        col, mk = TOPO_STYLE.get(a["topology"], ("#777", "x"))
        lbl = a["topology"] if a["topology"] not in seen else None
        if a["topology"] in ("DEBATE", "GE", "HIER"):
            seen.add(a["topology"])
        else:
            lbl = None
        ax.errorbar(a["tokens"], a["accuracy"] * 100,
                    yerr=[[(a["accuracy"] - a["acc_lo"]) * 100],
                          [(a["acc_hi"] - a["accuracy"]) * 100]],
                    fmt=mk, color=col, ms=7, capsize=3, zorder=3, label=lbl)
        ax.annotate(c, (a["tokens"], a["accuracy"] * 100),
                    textcoords="offset points", xytext=(5, 4), fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("inference cost  (tokens per query, log scale)")
    ax.set_ylabel("accuracy (%)")
    ax.set_title("Accuracy vs. inference compute across agent topologies")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    print("wrote", path)


def fig_crossover(agg, path, task=None):
    """Honest crossover: solid bars where the equal-budget SC comparison is
    MEASURED (MAS cost <= max measured SC cost); hatched bars where it requires
    extrapolating the SC curve (MAS costs more than the most expensive SC point),
    which we flag rather than treat as a measurement."""
    from harness import stats as _stats
    if task is None:
        print("crossover: task required, skipping"); return
    rep, sc_max = _stats.crossover_report(task)
    rep.sort(key=lambda r: r["gap"])
    names = [r["config"] for r in rep]
    gaps = [r["gap"] for r in rep]
    extrap = [r["extrapolated"] for r in rep]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for i, (n, g, e) in enumerate(zip(names, gaps, extrap)):
        ax.barh(i, g, color="#2ca02c" if g >= 0 else "#d62728",
                hatch="//" if e else None, alpha=0.55 if e else 1.0,
                edgecolor="k", linewidth=0.6)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("accuracy gap: single agent − MAS at equal token budget  (pts)")
    ax.set_title("Single agent vs. multi-agent at equal budget\n"
                 "(solid = measured; hatched = SC extrapolated beyond measured range)")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    print("wrote", path)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        ROOT, "results", "gsm8k_matrix.jsonl")
    os.makedirs(FIGDIR, exist_ok=True)
    rows = analyze.load(path)
    agg = analyze.aggregate(rows)
    tag = os.path.splitext(os.path.basename(path))[0]
    task = tag.replace("_matrix", "")
    fig_pareto(agg, os.path.join(FIGDIR, f"{tag}_pareto.pdf"))
    fig_crossover(agg, os.path.join(FIGDIR, f"{tag}_crossover.pdf"), task=task)


if __name__ == "__main__":
    main()
