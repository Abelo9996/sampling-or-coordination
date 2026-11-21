"""AAIBD revision figure: accuracy vs total tokens per query, three panels in the
column order of the main table (GSM8K, MMLU, MATH). No model calls.

  GSM8K  results/gsm8k_matrix.jsonl   (n=80, 640-token cap)
  MMLU   results/mmlu_matrix.jsonl    (n=80, 640-token cap)
  MATH   results/math500_trunc.jsonl  (n=60, levels 1-3, 1024-token cap)
         + results/math500_trunc_sc24.jsonl (SC k=2,4 at 1024) when complete

Only the headline configurations are drawn (SA, SC k, debate N=3 R=1, GE R=2,
HIER M=3); every MATH point is at the 1024 cap. Aggregation as in the paper:
seed-average within item, item mean; error bars are 95% item-bootstrap
percentile intervals (B=10,000, numpy seed 0). Style follows
scripts/make_combined_figure.py; colors are the Okabe-Ito set with a distinct
marker per workflow. Rerun after the reruns land:

  python3 scripts/make_aaibd_figure.py
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from harness import analyze  # noqa: E402

RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "paper", "aaibd", "figures", "frontier_rev.pdf")
HEAD = {"SA", "SC_k2", "SC_k4", "SC_k8", "SC_k12", "DEBATE_N3_R1", "GE_R2", "HIER_M3"}
STYLE = {"SC": ("#0072B2", "o", "SA and SC $k$ (sampling)"),
         "DEBATE": ("#D55E00", "s", "debate ($N{=}3,R{=}1$)"),
         "GE": ("#009E73", "^", "GE ($R{=}2$)"),
         "HIER": ("#CC79A7", "D", "HIER ($M{=}3$)")}


def read(fn):
    p = os.path.join(RES, fn)
    if not os.path.exists(p):
        return []
    return [json.loads(l) for l in open(p) if l.strip()]


def math_rows():
    ref = read("math500_trunc.jsonl")
    items = {r["item_id"] for r in ref if r["config"] == "SA"}
    extra = read("math500_trunc_sc24.jsonl")
    keep = []
    for cfg in ("SC_k2", "SC_k4"):
        have = {(r["item_id"], r["seed"]) for r in extra if r["config"] == cfg}
        if all((i, s) in have for i in items for s in (1, 2)):
            keep += [r for r in extra if r["config"] == cfg]
    return ref + keep, bool(keep)


def agg(rows):
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["config"] in HEAD:
            by[r["config"]][r["item_id"]].append(r)
    out = {}
    for c, items in by.items():
        corr = np.array([np.mean([int(r["correct"]) for r in v]) for v in items.values()])
        tok = np.array([np.mean([r["total_tokens"] for r in v]) for v in items.values()])
        lo, hi = analyze._bootstrap_ci(corr, B=10000)
        nseed = len({r["seed"] for v in items.values() for r in v})
        out[c] = dict(acc=100 * corr.mean(), lo=100 * lo, hi=100 * hi,
                      tok=tok.mean(), topo=c.split("_")[0], nseed=nseed)
    return out


def panel(ax, a, title):
    samp = sorted([c for c in a if a[c]["topo"] in ("SA", "SC")],
                  key=lambda c: a[c]["tok"])
    ax.plot([a[c]["tok"] for c in samp], [a[c]["acc"] for c in samp], "-",
            color=STYLE["SC"][0], lw=1.2, zorder=2)
    for c, v in a.items():
        key = "SC" if v["topo"] in ("SA", "SC") else v["topo"]
        col, mk, _ = STYLE[key]
        hollow = v["nseed"] == 1 and c != "SA"
        ax.errorbar(v["tok"], v["acc"], yerr=[[v["acc"] - v["lo"]], [v["hi"] - v["acc"]]],
                    fmt=mk, color=col, ms=4.5, capsize=1.5, elinewidth=0.8, zorder=3,
                    mfc="white" if hollow else col, mew=1.0)
        lab = "SA" if c == "SA" else (f"$k{{=}}{c.split('k')[1]}$" if v["topo"] == "SC" else None)
        if lab:
            ax.annotate(lab, (v["tok"], v["acc"]), xytext=(3, -9),
                        textcoords="offset points", fontsize=6, color="#333333")
    ax.set_xscale("log")
    ax.set_title(title, fontsize=7.5)
    ax.set_xlabel("total tokens / query (log)", fontsize=7)
    ax.tick_params(labelsize=6.5)
    ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.5)


def main():
    mrows, sc24 = math_rows()
    panels = [(agg(read("gsm8k_matrix.jsonl")), "GSM8K ($n{=}80$, 640-token cap)"),
              (agg(read("mmlu_matrix.jsonl")), "MMLU ($n{=}80$, 640-token cap)"),
              (agg(mrows), "MATH L1-3 ($n{=}60$, 1024-token cap)")]
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.25))
    for ax, (a, t) in zip(axes, panels):
        panel(ax, a, t)
    axes[0].set_ylabel("accuracy (%)", fontsize=7)
    handles = [plt.Line2D([], [], color=c, marker=m, ls="-" if k == "SC" else "",
                          ms=4.5, lw=1.2, label=l) for k, (c, m, l) in STYLE.items()]
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=6.5,
               frameon=False, bbox_to_anchor=(0.5, 1.11))
    fig.tight_layout(pad=0.3, w_pad=0.6)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print("wrote", OUT, "| MATH SC k=2,4 at 1024:", "included" if sc24 else "pending")


if __name__ == "__main__":
    main()
