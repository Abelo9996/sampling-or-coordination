"""rev3 frontier figure: accuracy against total tokens (top row) and against output tokens
(bottom row) per query, log x; columns GSM8K, MMLU, MATH levels 1 to 3, MMLU-Pro engineering and HumanEval
(rev3 rerun fold-in: the headline matrix of protocols/REV3_RERUN_PROTOCOL.md, Section 7).

Adapted from paper/arxiv_rev2/scripts/make_frontier_figure.py (which drew frontier_rev2.pdf with
scripts/make_aaibd_figure.py). Sources (nothing typed by hand):
  * GSM8K (results/gsm8k_matrix.jsonl) and MMLU (results/mmlu_matrix.jsonl): per configuration
    (SA, SC k, debate, GE, HIER) accuracy, 95% item-bootstrap interval (harness.analyze.
    _bootstrap_ci, B=10,000, seed 0) and mean total tokens with the aggregation of
    make_aaibd_figure.agg (copied below: seed-average within item, then item mean); mean output
    tokens with harness.stats.per_item_seed_tokens(rows, "out_tokens"), the function behind
    tables/review_revision.json. Asserted: accuracy and total tokens equal
    review_revision.json tasks.<task>.configs.<cfg> (acc, total_tokens) for every config, and
    GSM8K output tokens equal its out_tokens. review_revision.json has no MMLU out_tokens, so
    the MMLU output tokens are recomputed from the records with that same function (not
    checkable against a JSON).
  * MATH L1-3 (238 items, 3 seeds, 1,024 cap): tables/math_easy.json configs.<cfg> (acc, lo,
    hi, total_tokens, out_tokens), written by scripts/emit_math_easy.py. Asserted: n_items,
    the cap, and accuracies equal results/extra/math_interaction_analysis.json easy.accuracy
    (2 decimals) for the five configurations it lists.
  * The 640-token cap in the GSM8K and MMLU titles: the default of harness/client.py
    (DEFAULT_NUM_PREDICT), read from that file.
Output: paper/arxiv_rev3/figures/fig_frontier.pdf (FULL_W wide; --extra: fig_frontier_extra.pdf,
two thirds of FULL_W, same panel size), drawn at printed size with plotstyle.
"""
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

import plotstyle as FS

FS.setup()
import matplotlib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import eaai_lib as E  # noqa: E402

sys.path.insert(0, FS.ROOT)
from harness import analyze, stats as S  # noqa: E402

HEAD = ["SA", "SC_k2", "SC_k4", "SC_k8", "SC_k12", "DEBATE_N3_R1", "GE_R2", "HIER_M3"]
TOPO = {"SA": "SA", "SC_k2": "SC", "SC_k4": "SC", "SC_k8": "SC", "SC_k12": "SC",
        "DEBATE_N3_R1": "DEBATE", "GE_R2": "GE", "HIER_M3": "HIER"}
TAB = os.path.join(FS.REV3, "tables")


def read(fn):
    return [json.loads(l) for l in open(os.path.join(FS.ROOT, "results", fn)) if l.strip()]


def agg_records(rows):
    """make_aaibd_figure.agg (copied), plus mean output tokens."""
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["config"] in HEAD:
            by[r["config"]][r["item_id"]].append(r)
    outt = S.per_item_seed_tokens(rows, "out_tokens")
    out = {}
    for c, items in by.items():
        corr = np.array([np.mean([int(r["correct"]) for r in v]) for v in items.values()])
        tok = np.array([np.mean([r["total_tokens"] for r in v]) for v in items.values()])
        lo, hi = analyze._bootstrap_ci(corr, B=10000)
        nseed = len({r["seed"] for v in items.values() for r in v})
        out[c] = dict(acc=100 * corr.mean(), lo=100 * lo, hi=100 * hi, tok=tok.mean(),
                      out=float(np.mean([v.mean() for v in outt[c].values()])), nseed=nseed,
                      n=len(items))
    return out


def load_panels():
    """rev3 rerun fold-in: GSM8K (500 x 3, phase A + R1), MMLU (matrix + R2), MMLU-Pro
    engineering (matrix + R4) and HumanEval (R3, v2 extractor) from
    results/extra/rev3_reruns_analysis.json, HIER = HIERFIX; MATH L1-3 from tables/math_easy.json
    (HIER = HIERFIX there by the protocol). Accuracies asserted against rev3_reruns_macros.tex."""
    ra = json.load(open(os.path.join(FS.ROOT, "results", "extra", "rev3_reruns_analysis.json")))["tasks"]
    rmac = FS.macros("rev3_reruns_macros.tex")
    seeds = {"gsm8k": 3, "mmlu": 2, "proeng": 2, "code_v2": 2}
    caps = {"gsm8k": 640, "mmlu": 640, "proeng": 1024, "code_v2": 1024}
    def from_rr(key, nm, tn):
        T = ra[key]
        a = {}
        for c, v in T["configs"].items():
            k = "HIER_M3" if c == "HIERFIX_M3" else c
            if c == "HIER_M3" or k not in HEAD:
                continue
            a[k] = dict(acc=v["acc"], lo=v["lo"], hi=v["hi"], tok=v["total_tokens"], out=v["out_tokens"],
                        nseed=1 if c == "SA" else seeds[key], n=T["n_items"])
            mn = {"SA": "Sa", "SC_k2": "Sctwo", "SC_k4": "Scfour", "SC_k8": "Sceight", "DEBATE_N3_R1": "Deb",
                  "GE_R2": "Ge", "HIER_M3": "Hier"}[k]
            assert f"{v['acc']:.1f}" == rmac[f"Rr{tn}{mn}Acc"], (key, c)
        return (f"{nm}\n($n$={T['n_items']}, {caps[key]:,} cap)", a)
    panels = [from_rr("gsm8k", "GSM8K", "Gsm"), from_rr("mmlu", "MMLU", "Mmlu")]
    me = json.load(open(os.path.join(TAB, "math_easy.json")))
    mi = json.load(open(os.path.join(FS.ROOT, "results", "extra", "math_interaction_analysis.json")))
    assert me["n_items"] == mi["easy"]["n_items"] and me["cap"] == mi["easy"]["cap"]
    for c, v in mi["easy"]["accuracy"].items():
        assert round(me["configs"][c]["acc"], 2) == v, (c, me["configs"][c]["acc"], v)
    a = {c: dict(acc=v["acc"], lo=v["lo"], hi=v["hi"], tok=v["total_tokens"],
                 out=v["out_tokens"], nseed=len(me["seeds"]), n=me["n_items"])
         for c, v in me["configs"].items() if c in HEAD}
    assert set(a) == set(HEAD), set(a) ^ set(HEAD)
    panels.append((f"MATH L1-3\n($n$={me['n_items']}, {me['cap'][0]:,} cap)", a))
    panels.append(from_rr("proeng", "MMLU-Pro eng.", "Eng"))
    panels.append(from_rr("code_v2", "HumanEval", "Code"))
    return panels


def ticks_for(ax):
    lo, hi = ax.get_xlim()
    t = [v for v in (100, 200, 500, 1000, 2000, 5000, 10000) if lo <= v <= hi]
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(t))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def thin_ticks(ax, gap=3.0):
    """Drop the largest x tick while two adjacent tick labels are closer than gap points."""
    fig = ax.figure
    while True:
        fig.canvas.draw()
        rend = fig.canvas.get_renderer()
        labs = sorted([(t.get_loc(), t.label1.get_window_extent(rend)) for t in ax.xaxis.get_major_ticks()
                       if t.label1.get_visible() and t.label1.get_text()
                       and ax.get_xlim()[0] <= t.get_loc() <= ax.get_xlim()[1]], key=lambda u: u[0])
        k = fig.dpi / 72.0
        bad = [b for (a, ba), (b, bb) in zip(labs[:-1], labs[1:]) if bb.x0 - ba.x1 < gap * k]
        if not bad:
            return
        ticks = [t for t in ax.get_xticks() if t != max(bad)]
        ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))


def draw(ax, c, v, x):
    t = TOPO[c]
    col, mk = FS.COL[t], FS.MARK[t]
    hollow = v["nseed"] == 1 and c != "SA"
    E.errbar(ax, x, v["acc"], v["lo"], v["hi"], col, mk, FS.MS, "white" if hollow else col,
             zorder=3)


def draw_workflows(ax, a, key):
    """Coordination workflows, drawn after the layout is final: a marker that would sit on
    another point's interval is nudged sideways by E.dodge_x (presentation only; the plotted
    accuracy and interval are unchanged, the exact token counts are in the tables)."""
    samp = [c for c in a if TOPO[c] in ("SA", "SC")]
    coord = sorted([c for c in a if TOPO[c] not in ("SA", "SC")], key=lambda c: a[c][key])
    pos = E.dodge_x(ax, [(a[c][key], a[c]["acc"], a[c]["lo"], a[c]["hi"]) for c in samp],
                    [(a[c][key], a[c]["acc"], a[c]["lo"], a[c]["hi"]) for c in coord])
    for c, (x, d) in zip(coord, pos):
        draw(ax, c, a[c], x)
    return max([abs(d) for _, d in pos] + [0.0])


def panel(ax, a, key):
    samp = sorted([c for c in a if TOPO[c] in ("SA", "SC")], key=lambda c: a[c][key])
    ax.plot([a[c][key] for c in samp], [a[c]["acc"] for c in samp], "-", color=FS.COL["SC"],
            lw=FS.LW, zorder=2)
    for c in samp:
        draw(ax, c, a[c], a[c][key])
    xs = [v[key] for v in a.values()]
    ax.set_xscale("log")
    ax.set_xlim(min(xs) / 1.5, max(xs) * 2.0)
    ticks_for(ax)


def main():
    # main figure: GSM8K, MMLU, MATH L1-3; --extra: MMLU-Pro engineering and HumanEval
    # (fig_frontier_extra.pdf, appendix), same layout and styling at two thirds of the width.
    extra = "--extra" in sys.argv
    allp = load_panels()
    panels = allp[3:] if extra else allp[:3]
    fig, axes = plt.subplots(2, len(panels), figsize=(FS.FULL_W * len(panels) / 3, 4.7))
    lims = []
    for title, a in panels:
        ys = [v["lo"] for v in a.values()] + [v["hi"] for v in a.values()]
        lims.append((min(ys) - 3, max(ys) + 10))   # room for labels beyond the bar ends
    if extra:   # the two appendix panels share one y scale (same span, same tick step)
        span = max(hi - lo for lo, hi in lims)
        lims = [(lo, lo + span) for lo, hi in lims]
    for j, (title, a) in enumerate(panels):
        ylim = lims[j]
        for i, key in enumerate(("tok", "out")):
            ax = axes[i, j]
            panel(ax, a, key)
            ax.set_ylim(*ylim)
            ax.set_yticks([t for t in ax.get_yticks() if ylim[0] <= t <= min(ylim[1], 100)])
            if i == 0:
                ax.set_title(title, linespacing=1.05)
            ax.set_xlabel("total tokens / query (log)" if key == "tok"
                          else "output tokens / query (log)")
    for ax in axes[:, 0]:
        ax.set_ylabel("accuracy (%)")
    # one legend row; the appendix figure is narrower and names the workflows without their
    # parameters (its caption refers to Figure 3, whose legend gives them)
    names = {} if extra else {"DEBATE": "debate ($N$=3, $R$=1)", "GE": "GE ($R$=2)",
                              "HIER": "HIER ($M$=3)"}
    handles = E.legend_handles(with_front=False, names=names)
    leg = fig.legend(handles=handles, loc="outside upper center", ncol=len(handles))
    fig.canvas.draw()
    assert leg.get_window_extent().width <= fig.bbox.width, "legend wider than the figure"
    shift = max(draw_workflows(axes[i, j], a, key) for j, (_, a) in enumerate(panels)
                for i, key in enumerate(("tok", "out")))
    print(f"largest sideways nudge of a workflow marker: {shift:.2f} pt")
    assert shift <= 7.0 + 1e-6, shift   # the caption of fig:frontier says "at most 7 pt"
    for ax in axes.flat:
        thin_ticks(ax)
    problems = []
    for j, (_, a) in enumerate(panels):
        for i, key in enumerate(("tok", "out")):
            ax = axes[i, j]
            sc = sorted([c for c in HEAD if TOPO[c] in ("SA", "SC") and c in a], key=lambda c: a[c][key])
            items = [(FS.klabel(c), a[c][key], a[c]["acc"], a[c]["lo"],
                      a[c]["hi"]) for c in sc]
            texts = E.place_in_column(ax, items, fontsize=FS.FONT_PT, sides=True, dirs=(1, -1),
                                      pad=2.4)
            problems += E.check_labels(ax, texts, pad=1.0)
    assert not problems, problems
    FS.save(fig, "fig_frontier_extra" if extra else "fig_frontier")
    for (t, a) in panels:
        print("fig_frontier:", t.split("\n")[0], {c: (round(v["acc"], 2), round(v["tok"]),
                                                     round(v["out"])) for c, v in a.items()})


if __name__ == "__main__":
    main()
