"""Conceptual and data figures for the EAAI manuscript (paper/eaai/).

Every number drawn in a figure is computed here from the logged JSONL files in
results/ or read from the generated macro files in paper/; nothing is typed by
hand. One colorblind-safe palette (Okabe and Ito) with a fixed color per
workflow is used in every figure.

  .venv/bin/python scripts/make_eaai_figures.py

Writes (vector PDF):
  paper/eaai/figures/fig_workflows.pdf      workflow schematic (calls per query)
  paper/eaai/figures/fig_criterion.pdf      criterion map with measured points
  paper/eaai/figures/fig_flow.pdf           correction/corruption flows, GSM8K
  paper/eaai/figures/fig_frontier4.pdf      four-task accuracy-cost frontier
  paper/eaai/figures/fig_gsm8k_pareto.pdf   GSM8K frontier with swept workflows
  paper/eaai/figures/fig_proeng_pareto.pdf  engineering-task frontier (Fig. S1)
  paper/eaai/figures/fig_code_energy.pdf    code task: frontier in tokens and in joules
  paper/eaai/figures/fig_decision.pdf       decision flowchart (guidelines)
  paper/eaai/upload/graphical_abstract.pdf and .png (13 x 5 cm, 300 dpi)

Data sources:
  results/{gsm8k,mmlu,math500,mmlu_pro_eng}_matrix.jsonl  accuracy, tokens, calls
  results/math500_trunc.jsonl   MATH at the 1024-token cap (as in Table 5)
  results/humaneval_matrix_pc.jsonl  HumanEval on the RTX 4060, with GPU energy
  results/mechanistic.jsonl     debate and generator-evaluator traces (GSM8K)
  paper/scale70b_macros.tex     Llama-3.3-70B debate counts (scripts/emit_scale70b.py)
"""
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.text  # noqa: E402,F401
from matplotlib.patches import FancyBboxPatch, Polygon  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from harness import analyze  # noqa: E402

RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "paper", "eaai", "figures")
UPLOAD = os.path.join(ROOT, "paper", "eaai", "upload")

# ==== style ====
# Okabe-Ito palette. Fixed color per workflow in every figure.
COL = {"SA": "#000000", "SC": "#0072B2", "DEBATE": "#D55E00",
       "GE": "#009E73", "HIER": "#CC79A7", "OTHER": "#8C8C8C"}
MARK = {"SA": "o", "SC": "o", "DEBATE": "s", "GE": "^", "HIER": "D", "OTHER": "x"}
NAME = {"SA": "SA (single call)", "SC": r"SC$_k$ (self-consistency)",
        "DEBATE": "DEBATE", "GE": "GE", "HIER": "HIER"}
# semantic (non-workflow) colors: repair / helps vs corruption / hurts
FIX = "#56B4E9"      # sky blue
CORR = "#E69F00"     # orange
KEEP = "#D9D9D9"
STILL = "#9E9E9E"
FRONT = "#BDBDBD"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 8.5,
    "axes.labelsize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "mathtext.fontset": "dejavusans",
    "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.7,
    "savefig.dpi": 300, "savefig.bbox": "standard",
})
TEXTW = 390.0 / 72.27  # elsarticle preprint 12pt \textwidth in inches


PRINTED = {"fig_workflows": 1.0, "fig_criterion": 0.92, "fig_criterion_units": 0.85,
           "fig_flow": 1.0, "fig_frontier4": 1.0, "fig_gsm8k_pareto": 1.0,
           "fig_proeng_pareto": 0.85, "fig_decision": 1.0, "fig_code_energy": 1.0}
MIN_PT = 7.0


def save(fig, name):
    """Save at the printed size and check that no text is below MIN_PT points
    at the \\includegraphics width used in the manuscript (PRINTED x textwidth)."""
    scale = PRINTED[name] * TEXTW / fig.get_figwidth()
    small = [(t.get_text(), t.get_fontsize() * scale) for t in fig.findobj(matplotlib.text.Text)
             if t.get_visible() and t.get_text().strip()
             and t.get_fontsize() * scale < MIN_PT - 1e-6]
    assert not small, (name, small[:5])
    fig.savefig(os.path.join(OUT, f"{name}.pdf"))


def tint(hexcol, f):
    """Mix a hex color with white (f = share of white)."""
    h = hexcol.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return "#%02x%02x%02x" % tuple(int(c + (255 - c) * f) for c in (r, g, b))


def fmt_int(x):
    return f"{int(round(x)):,}"


def minus(s):
    return s.replace("-", "\u2212")


# ==== data ====
def load(name):
    return analyze.load(os.path.join(RES, name))


def macros(path):
    out = {}
    for m in re.finditer(r"\\newcommand\{\\(\w+)\}\{(.*)\}", open(path).read()):
        out[m.group(1)] = m.group(2).strip("{}")
    return out


def calls_per_config(rows):
    d = defaultdict(set)
    for r in rows:
        d[r["config"]].add(r["n_calls"])
    return d


def wilson(x, n, z=1.959964):
    p = x / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def mover_ratio(x1, n1, x2, n2):
    """95% MOVER-Wilson interval for p1/p2 of two independent binomials
    (Donner and Zou 2012). Defined for x1 = 0 (lower bound 0, finite upper)."""
    p1, p2 = x1 / n1, x2 / n2
    l1, u1 = wilson(x1, n1)
    l2, u2 = wilson(x2, n2)
    lo_num = p1 * p2 - math.sqrt(max(0.0, (p1 * p2) ** 2
                                     - l1 * u2 * (2 * p1 - l1) * (2 * p2 - u2)))
    lo = lo_num / (u2 * (2 * p2 - u2))
    hi_num = p1 * p2 + math.sqrt(max(0.0, (p1 * p2) ** 2
                                     - u1 * l2 * (2 * p1 - u1) * (2 * p2 - l2)))
    hi = hi_num / (l2 * (2 * p2 - l2))
    return max(0.0, lo), hi


def trace_counts():
    """Debate (per item, round-0 vs final majority) and GE (per item and per
    revision opportunity) transitions from results/mechanistic.jsonl, with the
    definitions of harness/mechanistic_analyze.py."""
    rs = load("mechanistic.jsonl")
    out = {}
    for wf in ("debate", "ge"):
        d = [r for r in rs if r["wf"] == wf]
        c = Counter((r["init_correct"], r["final_correct"]) for r in d)
        out[wf] = {"n": len(d), "kept": c[(True, True)], "cor": c[(True, False)],
                   "fix": c[(False, True)], "still": c[(False, False)]}
    rounds = [rd for r in rs if r["wf"] == "ge" for rd in r["rounds"]]
    bw = [rd for rd in rounds if not rd["before_correct"]]
    br = [rd for rd in rounds if rd["before_correct"]]
    out["ge_opp"] = {"wrong": len(bw), "right": len(br),
                     "fix": sum(rd["after_correct"] for rd in bw),
                     "cor": sum(not rd["after_correct"] for rd in br)}
    return out


def agg_task(task):
    """Seven main configurations per task, as in Tables 5 and 6. MATH: the five
    configurations rerun at the 1024-token cap come from math500_trunc.jsonl;
    SC k=2, 4 exist only at the 640-token cap and are returned separately as
    not comparable (they enter neither the sampling line nor the frontier)."""
    main = ["SA", "SC_k2", "SC_k4", "SC_k8", "DEBATE_N3_R1", "GE_R2", "HIER_M3"]
    agg = analyze.aggregate(load(f"{task}_matrix.jsonl"))
    capped = set()
    if task == "math500":
        tr = analyze.aggregate(load("math500_trunc.jsonl"))
        for c in main:
            if c in tr:
                agg[c] = tr[c]
            else:
                capped.add(c)
    return {c: agg[c] for c in main}, capped


def mixture_frontier(agg, keys):
    """Mixture-achievable frontier: upper concave hull of the measured
    (tokens, accuracy) points in LINEAR cost, from the cheapest point up to the
    most accurate one (what a randomized mix of configurations can reach)."""
    pts = sorted((agg[c]["tokens"], agg[c]["accuracy"], c) for c in keys)
    hull = []
    for p in pts:
        while len(hull) >= 2:
            (x0, y0, _), (x1, y1, _) = hull[-2], hull[-1]
            if (x1 - x0) * (p[1] - y0) - (y1 - y0) * (p[0] - x0) >= 0:
                hull.pop()
            else:
                break
        hull.append(p)
    imax = max(range(len(hull)), key=lambda i: hull[i][1])
    return [h[2] for h in hull[:imax + 1]]


def hull_value(agg, hull, x):
    """Accuracy of the mixture frontier at cost x (linear interpolation)."""
    xs = [agg[c]["tokens"] for c in hull]
    ys = [agg[c]["accuracy"] for c in hull]
    return float(np.interp(x, xs, ys))


def pretty(c):
    return (c.replace("_k", " k=").replace("_N", " N=").replace("_R", " R=")
            .replace("_M", " M=").replace("_", " "))


def _obstacles(ax):
    """Display-space obstacles drawn so far in ax: sampled points along every
    line (incl. error bars) and a box per marker."""
    from matplotlib.collections import LineCollection
    pts, boxes = [], []
    for ch in ax.get_children():
        if isinstance(ch, Line2D) and ch.get_visible():
            xy = ax.transData.transform(np.asarray(ch.get_xydata(), float))
            if ch.get_linestyle() not in ("None", "none", "") and len(xy) > 1:
                for (x0, y0), (x1, y1) in zip(xy[:-1], xy[1:]):
                    n = int(max(abs(x1 - x0), abs(y1 - y0)) / 2) + 2
                    pts.extend(zip(np.linspace(x0, x1, n), np.linspace(y0, y1, n)))
            if ch.get_marker() not in ("None", "none", "", None):
                r = ch.get_markersize() * fig_dpi(ax) / 72 / 2 + 1
                for x, y in xy:
                    boxes.append((x - r, y - r, x + r, y + r))
        elif isinstance(ch, LineCollection):
            for seg in ch.get_segments():
                xy = ax.transData.transform(np.asarray(seg, float))
                for (x0, y0), (x1, y1) in zip(xy[:-1], xy[1:]):
                    n = int(max(abs(x1 - x0), abs(y1 - y0)) / 2) + 2
                    pts.extend(zip(np.linspace(x0, x1, n), np.linspace(y0, y1, n)))
    return np.array(pts) if pts else np.zeros((0, 2)), boxes


def fig_dpi(ax):
    return ax.figure.dpi


def place_labels(ax, items, fontsize=7.2, pad=1.5):
    """Greedy collision-free label placement. items: list of (text, x, y, color)
    in data coordinates. Tries offsets of growing distance and keeps the first
    whose text box hits no line, marker, axis edge, or earlier label; a leader
    line is drawn when the label sits away from its point."""
    fig = ax.figure
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    pts, boxes = _obstacles(ax)
    axbb = ax.get_window_extent(rend)
    k = fig.dpi / 72.0
    dxs = [0] + [s * v for v in (5, 10, 16, 24, 34, 46, 60, 76) for s in (1, -1)]
    dys = [0] + [s * v for v in (6, 11, 17, 24, 32, 42, 54, 66, 80) for s in (1, -1)]
    cands = sorted({(dx, dy) for dx in dxs for dy in dys if (dx, dy) != (0, 0)},
                   key=lambda d: (d[0] ** 2 + d[1] ** 2, abs(d[1])))
    placed = []
    for text, x, y, col in items:
        px, py = ax.transData.transform((x, y))
        best, best_cost = None, None
        for dx, dy in cands:
            dist = math.hypot(dx, dy)
            if best_cost is not None and dist > best_cost:
                break   # candidates are sorted by distance; cost >= distance
            ha = "left" if dx > 0 else ("right" if dx < 0 else "center")
            t = ax.text(px + dx * k, py + dy * k, text, transform=None, fontsize=fontsize,
                        ha=ha, va="center", color="#222222")
            bb = t.get_window_extent(rend)
            t.remove()
            x0, y0, x1, y1 = bb.x0 - pad, bb.y0 - pad, bb.x1 + pad, bb.y1 + pad
            ok = (x0 >= axbb.x0 + 2 and x1 <= axbb.x1 - 2 and y0 >= axbb.y0 + 2
                  and y1 <= axbb.y1 - 2)
            if ok and len(pts):
                ins = ((pts[:, 0] > x0) & (pts[:, 0] < x1) & (pts[:, 1] > y0)
                       & (pts[:, 1] < y1))
                ok = not ins.any()
            if ok:
                ok = not any(b[0] < x1 and b[2] > x0 and b[1] < y1 and b[3] > y0
                             for b in boxes + placed)
            if not ok:
                continue
            # cost: distance plus a penalty for every line the leader crosses
            cost = dist
            if dist > 13 and len(pts):
                ex, ey = min(max(px, x0), x1), min(max(py, y0), y1)
                L_ = math.hypot(ex - px, ey - py)
                if L_ > 6:
                    ux, uy = (ex - px) / L_, (ey - py) / L_
                    rx, ry = pts[:, 0] - px, pts[:, 1] - py
                    along = rx * ux + ry * uy
                    perp = np.abs(rx * uy - ry * ux)
                    hit = (along > 6 * k) & (along < L_) & (perp < 1.5)
                    cost += 25 * len(np.unique(np.round(pts[hit] / 6), axis=0))
            if best_cost is None or cost < best_cost:
                best, best_cost = (ha, dx, dy, (x0, y0, x1, y1)), cost
        if best is None:
            raise RuntimeError(f"no free position for label {text!r}")
        ha, dx, dy, bb = best
        placed.append(bb)
        far = dx * dx + dy * dy > 13 ** 2
        ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points",
                    ha=ha, va="center", fontsize=fontsize, color="#222222", zorder=6,
                    arrowprops=(dict(arrowstyle="-", color="#777777", lw=0.5,
                                     shrinkA=1, shrinkB=3) if far else None))


PARENT = {"GES": "GE", "HIERC": "HIER"}
VARIANTS = {"GES_R2", "HIERC_M3", "DEBATE_N3_R2"}   # open markers (Table 3 variants)
SWEEPS = {"DEBATE": ["DEBATE_N2_R1", "DEBATE_N3_R1"], "GE": ["GE_R1", "GE_R2"],
          "HIER": ["HIER_M2", "HIER_M3"]}


def frontier_panel(ax, agg, title, excluded=(), labels=False, sweep_lines=False,
                   rings=True, jitter=None, xpad=(0.8, 1.3)):
    jitter = jitter or {}
    keys = [c for c in agg if c not in excluded]
    hull = mixture_frontier(agg, keys)
    hx, hy = [], []
    for c0, c1 in zip(hull[:-1], hull[1:]):
        xs = np.linspace(agg[c0]["tokens"], agg[c1]["tokens"], 60)
        hx.extend(xs)
        hy.extend(np.interp(xs, [agg[c0]["tokens"], agg[c1]["tokens"]],
                            [agg[c0]["accuracy"] * 100, agg[c1]["accuracy"] * 100]))
    if len(hull) == 1:
        hx, hy = [agg[hull[0]]["tokens"]], [agg[hull[0]]["accuracy"] * 100]
    ax.plot(hx, hy, "-", color=FRONT, lw=6, alpha=0.55, zorder=1,
            solid_capstyle="round")
    if rings:
        ax.plot([agg[c]["tokens"] for c in hull], [agg[c]["accuracy"] * 100 for c in hull],
                "o", ms=10.5, mfc="none", mec="#7A7A7A", mew=1.1, zorder=5,
                clip_on=False)
    X = {c: agg[c]["tokens"] * jitter.get(c, 1.0) for c in agg}
    sc = sorted((X[c], agg[c]["accuracy"] * 100) for c in keys
                if agg[c]["topology"] in ("SA", "SC"))
    ax.plot(*zip(*sc), "-", color=COL["SC"], lw=1.4, zorder=2)
    if sweep_lines:
        for topo, cs in SWEEPS.items():
            if all(c in agg for c in cs):
                ax.plot([X[c] for c in cs], [agg[c]["accuracy"] * 100 for c in cs],
                        "--", color=COL[topo], lw=1.0, zorder=2)
    for c, a in agg.items():
        t = PARENT.get(a["topology"], a["topology"])
        t = t if t in COL else "OTHER"
        col, mfc, ls = COL[t], COL[t], "-"
        if c in VARIANTS:
            mfc = "white"
        if c in excluded:
            col, mfc, ls = "#7FA9C9", "white", ":"
        eb = ax.errorbar(X[c], a["accuracy"] * 100,
                         yerr=[[(a["accuracy"] - a["acc_lo"]) * 100],
                               [(a["acc_hi"] - a["accuracy"]) * 100]],
                         fmt=MARK[t], color=col, ms=5.0 if t != "SA" else 5.5,
                         mfc=mfc, mew=1.0, capsize=1.8, elinewidth=0.8, zorder=3)
        eb[2][0].set_linestyle(ls)
    ax.set_xscale("log")
    xs = list(X.values())
    ax.set_xlim(min(xs) * xpad[0], max(xs) * xpad[1])
    lo, hi = ax.get_xlim()
    ticks = [t for t in (200, 500, 1000, 2000, 5000, 10000) if lo <= t <= hi]
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: f"{int(v):,}"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_title(title)
    ax.grid(True, which="major", ls=":", lw=0.5, alpha=0.6)
    ax.tick_params(length=2.5, pad=1.5)
    if labels:
        # labels: cfg -> (x_text, y_text, ha[, relpos]) in data coordinates,
        # chosen by eye in free space; a grey leader joins label and marker,
        # leaving the label at relpos (fraction of the text box; default centre)
        for c, spec in labels.items():
            xt, yt, ha = spec[:3]
            relpos = spec[3] if len(spec) > 3 else (0.5, 0.5)
            y = agg[c]["accuracy"] * 100
            ax.annotate(pretty(c), (X[c], y), xytext=(xt, yt), textcoords="data",
                        ha=ha, va="center", fontsize=7.2, color="#222222", zorder=6,
                        arrowprops=dict(arrowstyle="-", color="#888888", lw=0.5,
                                        shrinkA=1.5, shrinkB=3.5, relpos=relpos))
    return hull


def finish_labels(fig):
    for ax in fig.axes:
        items = getattr(ax, "_pending_labels", None)
        if items:
            place_labels(ax, items)


def legend_handles(with_front=True, extra=()):
    h = []
    if with_front:
        h.append(Line2D([], [], color=FRONT, lw=6, alpha=0.55, marker="o", ms=9,
                        mfc="none", mec="#7A7A7A", mew=1.1, solid_capstyle="butt",
                        label="mixture frontier"))
    h.append(Line2D([], [], color=COL["SC"], marker="o", lw=1.4, ms=4.5,
                    label=r"SA + SC$_k$ (sampling)"))
    for t, lab in (("DEBATE", "DEBATE"), ("GE", "GE"), ("HIER", "HIER")):
        h.append(Line2D([], [], color=COL[t], marker=MARK[t], ls="none", ms=4.5,
                        label=lab))
    h.extend(extra)
    return h


JIT = 1.06   # display-only horizontal offset for overlapping markers


def fig_frontier4():
    tasks = [("gsm8k", "GSM8K (grade-school mathematics)"),
             ("mmlu", "MMLU (knowledge multiple choice)"),
             ("math500", "MATH, levels 1-3 (1024-token cap)"),
             ("mmlu_pro_eng", "MMLU-Pro engineering (1024-token cap)")]
    fig, axes = plt.subplots(2, 2, figsize=(TEXTW, 4.6))
    info = {}
    for ax, (t, title) in zip(axes.flat, tasks):
        agg, capped = agg_task(t)
        jit = {"GE_R2": JIT} if t == "gsm8k" else {}
        hull = frontier_panel(ax, agg, title, excluded=capped, jitter=jit,
                              xpad=(0.8, 1.35))
        if capped:
            ax.set_ylim(top=98)
            ax.text(agg["SA"]["tokens"] * 1.12, 94.6,
                    "open: SC k = 2, 4 at the 640-token\ncap (not comparable)",
                    fontsize=7.2, color="#4F7FA5", va="center", ha="left",
                    linespacing=1.0)
        info[t] = {"agg": agg, "hull": hull, "capped": capped}
    for ax in axes[:, 0]:
        ax.set_ylabel("accuracy (%)")
    for ax in axes[1, :]:
        ax.set_xlabel("tokens per query (log scale)")
    hs = legend_handles()
    fig.legend(handles=hs, loc="upper center", ncol=5,
               frameon=False, bbox_to_anchor=(0.5, 1.0), columnspacing=0.7,
               handletextpad=0.35, handlelength=1.6)
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=0.9, w_pad=0.8)
    save(fig, "fig_frontier4")
    plt.close(fig)
    return info


def fig_gsm8k_pareto():
    """Every configuration of Table 3."""
    agg = analyze.aggregate(load("gsm8k_matrix.jsonl"))
    fig, ax = plt.subplots(figsize=(TEXTW, 4.3))
    # crowded labels go to free bands above and below the data. GE R=1, GES
    # R=2 and both debate N=3 labels sit straight above (GE R=1, GES R=2,
    # DEBATE N=3 R=1) or below (DEBATE N=3 R=2) their own error bar, so each
    # leader runs along that bar and crosses no other mark or label; DEBATE N=2
    # R=1 sits below the lower end of the DEBATE N=3 R=1 bar.
    jit = {"GE_R2": JIT, "DEBATE_N3_R1": JIT}
    xd3 = agg["DEBATE_N3_R1"]["tokens"] * JIT
    offs = {"SA": (430, 76.4, "left"), "SC_k2": (860, 75.4, "left"),
            "SC_k4": (1530, 85.0, "right"), "SC_k8": (3420, 89.6, "left"),
            "SC_k12": (5050, 93.4, "left"),
            "GE_R1": (agg["GE_R1"]["tokens"], 92.0, "right", (1.0, 0.0)),
            "GES_R2": (agg["GES_R2"]["tokens"], 89.0, "left", (0.0, 0.0)),
            "GE_R2": (3850, 74.2, "left"),
            "DEBATE_N2_R1": (2700, 68.6, "left"),
            "DEBATE_N3_R1": (xd3, 87.6, "left", (0.0, 0.0)),
            "DEBATE_N3_R2": (agg["DEBATE_N3_R2"]["tokens"], 64.6, "center", (0.5, 1.0)),
            "HIER_M2": (1360, 65.0, "right"),
            "HIER_M3": (2050, 51.5, "center"), "HIERC_M3": (3950, 51.5, "center")}
    hull = frontier_panel(ax, agg, "", labels=offs, sweep_lines=True, jitter=jit,
                          xpad=(0.82, 1.3))
    ax.set_ylim(48, 99.5)
    ax.set_xlabel("tokens per query (log scale)")
    ax.set_ylabel("accuracy (%)")
    extra = [Line2D([], [], color="#555555", marker="s", mfc="white", ls="none",
                    ms=4.5, label="open: GES, HIERC, DEBATE R = 2")]
    ax.legend(handles=legend_handles(extra=extra), loc="lower center", ncol=4,
              frameon=False, bbox_to_anchor=(0.5, 1.0), columnspacing=0.9,
              handletextpad=0.4)
    fig.tight_layout()
    save(fig, "fig_gsm8k_pareto")
    plt.close(fig)
    return {"agg": agg, "hull": hull}


def fig_proeng_pareto():
    agg, _ = agg_task("mmlu_pro_eng")
    fig, ax = plt.subplots(figsize=(0.85 * TEXTW, 3.2))
    offs = {"SA": (880, 25.0, "left"), "SC_k2": (1770, 27.0, "left"),
            "SC_k4": (3120, 32.8, "right"), "SC_k8": (6950, 36.3, "left"),
            "GE_R2": (5150, 21.8, "right"), "HIER_M3": (6300, 17.8, "center"),
            "DEBATE_N3_R1": (8300, 25.7, "left")}
    hull = frontier_panel(ax, agg, "", labels=offs, xpad=(0.85, 1.9))
    ax.set_ylim(15.5, 42.5)
    ax.set_xlabel("tokens per query (log scale)")
    ax.set_ylabel("accuracy (%)")
    ax.legend(handles=legend_handles(), loc="lower center", ncol=3, frameon=False,
              bbox_to_anchor=(0.5, 1.0), columnspacing=1.0, handletextpad=0.4)
    fig.tight_layout()
    save(fig, "fig_proeng_pareto")
    plt.close(fig)
    return {"agg": agg, "hull": hull}


# ==== workflows ====
def box(ax, x, y, w, h, text, fc, ec, ls="-", fs=7.5, lw=0.9, weight="normal",
        style="round,pad=0,rounding_size=0.05"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, fc=fc, ec=ec,
                                lw=lw, ls=ls, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            zorder=4, weight=weight, linespacing=1.0)
    return (x, y, w, h)


def arrow(ax, p, q, col="#444444", lw=0.8, rad=0.0, ms=6):
    ax.annotate("", xy=q, xytext=p, zorder=2,
                arrowprops=dict(arrowstyle="-|>", color=col, lw=lw,
                                mutation_scale=ms, shrinkA=0, shrinkB=0,
                                connectionstyle=f"arc3,rad={rad}"))


def R(b):  # right-middle
    return (b[0] + b[2], b[1] + b[3] / 2)


def L(b):  # left-middle
    return (b[0], b[1] + b[3] / 2)


def fig_workflows():
    rows = load("gsm8k_matrix.jsonl")
    cpc = calls_per_config(rows)
    params = {r["config"]: r["params"] for r in rows}
    for c, s in cpc.items():
        assert len(s) == 1 or c.startswith("GES"), (c, s)
    ks = sorted(int(params[c]["k"]) for c in cpc if c.startswith("SC_"))
    # k values present in every task's matrix log vs GSM8K only
    k_all = set(ks)
    for t in ("mmlu", "math500", "mmlu_pro_eng"):
        k_all &= {int(r["params"]["k"]) for r in load(f"{t}_matrix.jsonl")
                  if r["config"].startswith("SC_")}
    k_gsm = sorted(set(ks) - k_all)
    k_all = sorted(k_all)
    for c in cpc:
        if c.startswith("SC_"):
            assert cpc[c] == {int(params[c]["k"])}
    deb, ge, hier = params["DEBATE_N3_R1"], params["GE_R2"], params["HIER_M3"]
    N, Rr = int(deb["N"]), int(deb["R"])
    Rg, M = int(ge["R"]), int(hier["M"])
    nsa, ndeb = cpc["SA"].pop(), cpc["DEBATE_N3_R1"].pop()
    nge, nhier = cpc["GE_R2"].pop(), cpc["HIER_M3"].pop()
    assert ndeb == N * (Rr + 1) and nge == 2 * Rg + 1 and nhier == M + 2
    assert N == 3 and M == 3 and Rr == 1 and Rg == 2  # drawn explicitly below

    bh, gap, pad = 0.215, 0.075, 0.11
    stack = 3 * bh + 2 * gap
    heights = {"SA": 0.56, "SC": stack + 2 * pad + 0.08,
               "DEBATE": stack + 2 * pad + 0.14, "GE": 0.74,
               "HIER": stack + 2 * pad}
    order = ["SA", "SC", "DEBATE", "GE", "HIER"]
    leg_h = 0.32
    H = sum(heights.values()) + leg_h
    W = TEXTW
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    label = {
        "SA": ("SA", "one greedy call", f"{nsa} call per query"),
        "SC": (r"SC$_k$", "k independent samples,\nmajority vote",
               "k calls, k = " + ", ".join(map(str, k_all)) + "\n(k = "
               + ", ".join(map(str, k_gsm)) + " on GSM8K only)"),
        "DEBATE": ("DEBATE", f"N = {N} agents, R = {Rr} round",
                   f"{ndeb} calls = N(R+1)"),
        "GE": ("GE", f"generator-evaluator,\nR = {Rg}", f"{nge} calls = 2R+1"),
        "HIER": ("HIER", f"planner, M = {M} executors,\naggregator",
                 f"{nhier} calls = M+2"),
    }
    xq, wq = 1.38, 0.44
    xa, wa = W - 0.52, 0.47
    y = H
    for k in order:
        h = heights[k]
        y0 = y - h
        if k != "SA":
            ax.plot([0.05, W - 0.05], [y, y], color="#DDDDDD", lw=0.6, zorder=1)
        cy = y0 + h / 2 - (0.06 if k in ("DEBATE", "GE") else 0.0)
        c = COL[k]
        nm, desc, calls = label[k]
        ax.text(0.06, y - 0.05, nm, fontsize=9, weight="bold", color=c,
                va="top", ha="left")
        ax.text(0.06, y - 0.235, desc, fontsize=7.2, va="top", ha="left",
                color="#333333", linespacing=1.0)
        ax.text(0.06, y0 + 0.04, calls, fontsize=7.2, va="bottom", ha="left",
                weight="bold", color="#333333", linespacing=1.0)
        q = box(ax, xq, cy - bh / 2, wq, bh, "query", "#F2F2F2", "#7F7F7F")
        ans = box(ax, xa, cy - bh / 2, wa, bh, "answer", "#FFFFFF", "#333333",
                  weight="bold")
        fc, ec = tint(c, 0.78), c
        if k == "SA":
            b = box(ax, 2.75, cy - bh / 2, 0.95, bh, "model call", fc, ec)
            arrow(ax, R(q), L(b))
            arrow(ax, R(b), L(ans))
        elif k == "SC":
            v = box(ax, 3.95, cy - 0.17, 0.62, 0.34, "majority\nvote", "#FFFFFF",
                    "#555555", ls="--")
            for i, t in enumerate(["sample 1", "sample 2", "sample k"]):
                yy = cy + (1 - i) * (bh + gap) - bh / 2
                b = box(ax, 2.45, yy, 0.8, bh, t, fc, ec)
                arrow(ax, R(q), L(b))
                arrow(ax, R(b), L(v))
            arrow(ax, R(v), L(ans))
        elif k == "DEBATE":
            cyb = cy
            x0, x1, w = 1.98, 3.08, 0.74
            r0, r1 = [], []
            for i in range(3):
                yy = cyb + (1 - i) * (bh + gap) - bh / 2
                r0.append(box(ax, x0, yy, w, bh, f"agent {i + 1}", fc, ec))
                r1.append(box(ax, x1, yy, w, bh, f"agent {i + 1}", fc, ec))
            ax.text(x0 + w / 2, cyb + stack / 2 + 0.04, "round 0: answer",
                    ha="center", va="bottom", fontsize=7.2, color="#333333")
            ax.text(x1 + w / 2, cyb + stack / 2 + 0.04, "round 1: critique, update",
                    ha="center", va="bottom", fontsize=7.2, color="#333333")
            v = box(ax, 4.12, cyb - 0.17, 0.6, 0.34, "majority\nvote", "#FFFFFF",
                    "#555555", ls="--")
            for i in range(3):
                arrow(ax, R(q), L(r0[i]))
                for j in range(3):
                    if j != i:  # agent i sees the other agents' round-0 solutions
                        arrow(ax, R(r0[j]), L(r1[i]), col="#888888", lw=0.6)
                arrow(ax, R(r1[i]), L(v))
            ans_y = ans[1] + ans[3] / 2
            arrow(ax, R(v), (xa, ans_y))
        elif k == "GE":
            cyb = cy
            names = ["generate", "critique", "revise", "critique", "revise"]
            xs0, xs1 = 1.98, xa - 0.12
            w = 0.5
            g = (xs1 - xs0 - 5 * w) / 4
            prev = q
            bs = []
            for i, t in enumerate(names):
                b = box(ax, xs0 + i * (w + g), cyb - bh / 2, w, bh, t, fc, ec, fs=7.2)
                arrow(ax, R(prev), L(b))
                bs.append(b)
                prev = b
            for rr in range(2):
                a0, a1 = bs[1 + 2 * rr], bs[2 + 2 * rr]
                ax.text((a0[0] + a1[0] + a1[2]) / 2, cyb + bh / 2 + 0.04,
                        f"round {rr + 1}", ha="center", va="bottom", fontsize=7.2,
                        color="#333333")
            arrow(ax, R(prev), (xa, ans[1] + ans[3] / 2))
        elif k == "HIER":
            p = box(ax, 2.0, cy - bh / 2, 0.66, bh, "planner", fc, ec)
            ag = box(ax, 4.0, cy - bh / 2, 0.74, bh, "aggregator", fc, ec)
            arrow(ax, R(q), L(p))
            for i in range(3):
                yy = cy + (1 - i) * (bh + gap) - bh / 2
                e = box(ax, 2.95, yy, 0.8, bh, f"executor {i + 1}", fc, ec, fs=7.2)
                arrow(ax, R(p), L(e))
                arrow(ax, R(e), L(ag))
            arrow(ax, R(ag), L(ans))
        y = y0
    # legend row
    ax.plot([0.05, W - 0.05], [leg_h, leg_h], color="#DDDDDD", lw=0.6)
    ly = leg_h / 2 - 0.09
    box(ax, 0.10, ly, 0.42, 0.18, "", tint(COL["SC"], 0.78), COL["SC"])
    ax.text(0.58, ly + 0.09, "one model call (color = workflow)", fontsize=7.2,
            va="center")
    box(ax, 2.55, ly, 0.42, 0.18, "", "#FFFFFF", "#555555", ls="--")
    ax.text(3.03, ly + 0.09, "vote, no model call", fontsize=7.2, va="center")
    arrow(ax, (4.05, ly + 0.09), (4.4, ly + 0.09))
    ax.text(4.46, ly + 0.09, "text passed on", fontsize=7.2, va="center")
    save(fig, "fig_workflows")
    plt.close(fig)
    return {"SA": nsa, "SC_k": ks, "DEBATE": ndeb, "GE": nge, "HIER": nhier}


# ==== criterion map ====
def _majority(answers):
    """Same rule as harness.topologies._majority (copied so that this script
    does not depend on the working-tree harness)."""
    vals = [v for v in answers if v is not None]
    if not vals:
        return None
    top = Counter(str(v) for v in vals).most_common(1)[0][0]
    for v in vals:
        if str(v) == top:
            return v
    return vals[0]


def units_3b():
    """Per-item clusters from results/mechanistic.jsonl. Each cluster is
    (right, wrong, fix, cor) counted in one unit:
      debate_item: round-0 majority vs final majority (one unit per item)
      ge_item:     initial draft vs final answer (one unit per item)
      ge_rev:      each revision round (two units per item)."""
    rs = load("mechanistic.jsonl")
    out = {"debate_item": [], "ge_item": [], "ge_rev": []}
    for r in rs:
        i, f = r["init_correct"], r["final_correct"]
        cl = (int(i), int(not i), int((not i) and f), int(i and not f))
        out["debate_item" if r["wf"] == "debate" else "ge_item"].append(cl)
        if r["wf"] == "ge":
            R_ = W_ = F_ = C_ = 0
            for rd in r["rounds"]:
                if rd["before_correct"]:
                    R_ += 1
                    C_ += int(not rd["after_correct"])
                else:
                    W_ += 1
                    F_ += int(rd["after_correct"])
            out["ge_rev"].append((R_, W_, F_, C_))
    return out


def units_70b():
    """Per-item clusters from results/gsm8k_llama70b.jsonl (debate N=3, R=1,
    items with all four configurations, as in scripts/emit_scale70b.py):
      item:  round-0 majority vs final majority; agent: each of the 3 agents."""
    from harness import tasks
    task = tasks.get_task("gsm8k")
    rows = load("gsm8k_llama70b.jsonl")
    cfgs = {"SA", "SC_k4", "SC_k8", "DEBATE_N3_R1"}
    have = defaultdict(set)
    for r in rows:
        have[r["item_id"]].add(r["config"])
    ok = {i for i, c in have.items() if cfgs <= c}
    out = {"item": [], "agent": []}
    for r in rows:
        if r["config"] != "DEBATE_N3_R1" or r["item_id"] not in ok:
            continue
        texts = [c["text"] for c in r["calls"]]
        init = [task["extract"](t) for t in texts[:3]]
        fin = [task["extract"](t) for t in texts[3:]]
        ic = bool(task["score"](_majority(init), r["gold"]))
        fc = bool(task["score"](_majority(fin), r["gold"]))
        assert fc == bool(r["correct"])
        out["item"].append((int(ic), int(not ic), int((not ic) and fc), int(ic and not fc)))
        R_ = W_ = F_ = C_ = 0
        for a0, a1 in zip(init, fin):
            c0, c1 = task["score"](a0, r["gold"]), task["score"](a1, r["gold"])
            if c0:
                R_ += 1
                C_ += int(not c1)
            else:
                W_ += 1
                F_ += int(c1)
        out["agent"].append((R_, W_, F_, C_))
    return out


def summarize(clusters, B=10000, seed=0):
    """Point estimates and 95% item-cluster bootstrap percentile intervals for
    a = right / (right + wrong) and ratio = (fix / wrong) / (cor / right)."""
    X = np.array(clusters, dtype=float)
    Rt, Wt, Ft, Ct = X.sum(0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(X), size=(B, len(X)))
    S = X[idx].sum(1)
    Rb, Wb, Fb, Cb = S[:, 0], S[:, 1], S[:, 2], S[:, 3]
    ab = Rb / (Rb + Wb)
    with np.errstate(divide="ignore", invalid="ignore"):
        rb = (Fb / Wb) / (Cb / Rb)
    rb = np.sort(rb[~np.isnan(rb)])   # inf (no corruption in a resample) kept
    lo_i, hi_i = int(np.floor(0.025 * (len(rb) - 1))), int(np.ceil(0.975 * (len(rb) - 1)))
    ratio = (Ft / Wt) / (Ct / Rt) if Ct else float("inf")
    return dict(right=int(Rt), wrong=int(Wt), fix=int(Ft), cor=int(Ct),
                a=Rt / (Rt + Wt), a_ci=tuple(np.percentile(ab, [2.5, 97.5])),
                ratio=ratio, ratio_ci=(float(rb[lo_i]), float(rb[hi_i])),
                p_zero=float(np.mean(rb == 0)), p_inf=float(np.mean(np.isinf(rb))),
                odds=Rt / Wt)


def criterion_points(units=("debate_item", "ge_item", "item70")):
    u3, u70 = units_3b(), units_70b()
    m = macros(os.path.join(ROOT, "paper", "scale70b_macros.tex"))
    spec = {
        "debate_item": ("3B DEBATE, per item", "DEBATE", u3["debate_item"], COL["DEBATE"], 1 / 1.006),
        "ge_item": ("3B GE, per item", "GE", u3["ge_item"], COL["GE"], 1.006),
        "item70": ("70B DEBATE, per item", "DEBATE", u70["item"], "white", 1.0),
        "ge_rev": ("3B GE, per revision", "GE", u3["ge_rev"], "white", 1.0),
        "agent70": ("70B DEBATE, per agent", "DEBATE", u70["agent"], "white", 1.0),
    }
    pts = []
    for k in units:
        lab, topo, cl, mfc, jit = spec[k]
        p = summarize(cl)
        p.update(key=k, lab=lab, topo=topo, mfc=mfc, jit=jit)
        pts.append(p)
    # consistency with the reported counts
    chk = {q["key"]: q for q in pts}
    if "item70" in chk:
        q = chk["item70"]
        assert (q["fix"], q["wrong"], q["cor"], q["right"]) == (
            int(m["sevCritFix"]), int(m["sevCritInitWrong"]), int(m["sevCritCor"]),
            int(m["sevCritInitRight"]))
    if "agent70" in chk:
        q = chk["agent70"]
        assert (q["fix"], q["wrong"], q["cor"], q["right"]) == (
            int(m["sevAgFix"]), int(m["sevAgWrong"]), int(m["sevAgCor"]),
            int(m["sevAgRight"]))
    if "ge_rev" in chk:
        q, go = chk["ge_rev"], trace_counts()["ge_opp"]
        assert (q["fix"], q["wrong"], q["cor"], q["right"]) == (
            go["fix"], go["wrong"], go["cor"], go["right"])
    return pts


def point_marker(p):
    return "v" if p["ratio"] == 0 else MARK[p["topo"]]


def draw_criterion(ax, pts, ylim=(0.03, 300), compact=False, zero_label=None):
    a = np.linspace(0.5, 0.99, 400)
    be = a / (1 - a)
    ax.fill_between(a, be, ylim[1], color=tint(FIX, 0.72), lw=0, zorder=0)
    ax.fill_between(a, ylim[0], be, color=tint(CORR, 0.75), lw=0, zorder=0)
    ax.plot(a, be, color="black", lw=1.3, zorder=2)
    ax.set_yscale("log")
    ax.set_xlim(0.5, 0.99)
    ax.set_ylim(*ylim)
    if not compact:
        ax.axhline(1.0, color="#555555", lw=0.8, ls=":", zorder=1)
        ax.text(0.515, 5.5, "coordination helps\n" r"($\Delta > 0$)", fontsize=8.5,
                weight="bold", color="#1F5F86", va="center")
        ax.text(0.515, 0.12, "coordination hurts\n" r"($\Delta < 0$)", fontsize=8.5,
                weight="bold", color="#8A5A00", va="center")
    floor = ylim[0] * 1.18
    for p in pts:
        c = COL[p["topo"]]
        mk = point_marker(p)
        ms = 6 if not compact else 4.5
        x = p["a"] * p["jit"]
        xerr = [[p["a"] - p["a_ci"][0]], [p["a_ci"][1] - p["a"]]]
        y = p["ratio"] if p["ratio"] > 0 else floor
        lo, hi = p["ratio_ci"]
        # vertical interval drawn by hand so that a lower bound of 0 runs to
        # the axis edge and a degenerate interval (no repairs) draws nothing
        if hi > 0:
            ax.plot([x, x], [max(lo, ylim[0]), min(hi, ylim[1])], color=c, lw=1.0,
                    zorder=3.5, clip_on=True, solid_capstyle="butt")
            for yy in (lo, hi):
                if ylim[0] < yy < ylim[1]:
                    ax.plot([x - 0.004, x + 0.004], [yy, yy], color=c, lw=0.9, zorder=3)
        ax.errorbar(x, y, xerr=xerr, fmt=mk, color=c, mfc=p["mfc"], mew=1.1, ms=ms,
                    capsize=2, elinewidth=0.9, zorder=4, clip_on=False)
        if p["ratio"] == 0 and not compact:
            txt = f"observed {p['fix']}/{p['wrong']} fixed\n(ratio 0)"
            if zero_label is None:   # centered above the marker
                ax.annotate(txt, (x, y), xytext=(0, 12), textcoords="offset points",
                            ha="center", va="bottom", fontsize=7.2, color=c,
                            linespacing=1.0)
            else:                    # in free space, with a leader line
                ax.annotate(txt, (x, y), xytext=zero_label, textcoords="data",
                            ha="center", va="center", fontsize=7.2, color=c,
                            linespacing=1.0,
                            arrowprops=dict(arrowstyle="-", color=c, lw=0.6,
                                            shrinkA=2, shrinkB=4))
    ax.set_xlabel(r"base accuracy $a$")
    ax.set_ylabel(r"$p_{\mathrm{fix}}\,/\,p_{\mathrm{cor}}$ (log scale)")
    ax.grid(True, which="major", ls=":", lw=0.5, alpha=0.5)
    ax.tick_params(length=2.5, pad=1.5)


def _criterion_legend(ax, pts):
    hs = [Line2D([], [], color="black", lw=1.3, label=r"break-even $a/(1-a)$"),
          Line2D([], [], color="#555555", lw=0.8, ls=":",
                 label=r"$p_{\mathrm{fix}} = p_{\mathrm{cor}}$ (no discrimination)")]
    for p in pts:
        lab = f"{p['lab']}: {p['fix']}/{p['wrong']} vs {p['cor']}/{p['right']}"
        hs.append(Line2D([], [], color=COL[p["topo"]], marker=point_marker(p),
                         mfc=p["mfc"], mew=1.1, ms=5.5, ls="none", label=lab))
    ax.legend(handles=hs, loc="upper left", fontsize=7.2, frameon=True,
              framealpha=0.92, edgecolor="#BBBBBB", borderpad=0.4,
              labelspacing=0.3, handletextpad=0.5)


def fig_criterion():
    pts = criterion_points()
    fig, ax = plt.subplots(figsize=(0.92 * TEXTW, 3.45))
    draw_criterion(ax, pts)
    _criterion_legend(ax, pts)
    fig.tight_layout()
    save(fig, "fig_criterion")
    plt.close(fig)
    return pts


def fig_criterion_units():
    """Supplementary: the same traces counted in other units."""
    pts = criterion_points(("ge_item", "ge_rev", "item70", "agent70"))
    for p in pts:   # display-only horizontal offsets, as in the main figure
        p["jit"] = {"ge_item": 1 / 1.006, "ge_rev": 1.006}.get(p["key"], 1.0)
    fig, ax = plt.subplots(figsize=(0.85 * TEXTW, 3.2))
    draw_criterion(ax, pts, zero_label=(0.70, 0.06))
    _criterion_legend(ax, pts)
    fig.tight_layout()
    save(fig, "fig_criterion_units")
    plt.close(fig)
    return pts


# ==== flows ====
def band(ax, x0, x1, y0a, y0b, y1a, y1b, col, alpha=1.0):
    t = np.linspace(0, 1, 60)
    s = 3 * t ** 2 - 2 * t ** 3
    xs = x0 + (x1 - x0) * t
    top = y0b + (y1b - y0b) * s
    bot = y0a + (y1a - y0a) * s
    poly = np.vstack([np.c_[xs, top], np.c_[xs[::-1], bot[::-1]]])
    ax.add_patch(Polygon(poly, closed=True, fc=col, ec="none", alpha=alpha,
                         zorder=2))


def draw_flow(ax, d, title, tcol):
    n = d["n"]
    right0, wrong0 = d["kept"] + d["cor"], d["fix"] + d["still"]
    right1, wrong1 = d["kept"] + d["fix"], d["cor"] + d["still"]
    g = 4.0
    bw = 0.075
    xl, xr = 0.0, 1.0
    # left: correct on top [g + wrong0, g + wrong0 + right0], wrong at [0, wrong0]
    Lc = (wrong0 + g, wrong0 + g + right0)
    Lw = (0, wrong0)
    Rc = (wrong1 + g, wrong1 + g + right1)
    Rw = (0, wrong1)
    for (x, seg, col) in ((xl, Lc, "#4D4D4D"), (xl, Lw, "#4D4D4D"),
                          (xr, Rc, "#4D4D4D"), (xr, Rw, "#4D4D4D")):
        ax.add_patch(plt.Rectangle((x - bw / 2, seg[0]), bw, seg[1] - seg[0],
                                   fc=col, ec="none", zorder=3))
    # left splits: correct -> kept (top), corrupted (bottom);
    #              wrong -> fixed (top), still wrong (bottom)
    kept_L = (Lc[1] - d["kept"], Lc[1])
    cor_L = (Lc[0], Lc[0] + d["cor"])
    fix_L = (Lw[1] - d["fix"], Lw[1])
    still_L = (Lw[0], Lw[0] + d["still"])
    # right: correct = kept (top) + fixed (bottom); wrong = corrupted (top) + still
    kept_R = (Rc[1] - d["kept"], Rc[1])
    fix_R = (Rc[0], Rc[0] + d["fix"])
    cor_R = (Rw[1] - d["cor"], Rw[1])
    still_R = (Rw[0], Rw[0] + d["still"])
    x0, x1 = xl + bw / 2, xr - bw / 2
    band(ax, x0, x1, *kept_L, *kept_R, KEEP)
    band(ax, x0, x1, *still_L, *still_R, STILL, 0.75)
    band(ax, x0, x1, *cor_L, *cor_R, CORR, 0.95)
    band(ax, x0, x1, *fix_L, *fix_R, FIX, 0.95)
    fs = 7.4
    ax.text(xl - bw, sum(Lc) / 2, f"initially\ncorrect\n{right0}", ha="right",
            va="center", fontsize=fs, linespacing=1.0)
    ax.text(xl - bw, sum(Lw) / 2, f"initially\nwrong\n{wrong0}", ha="right",
            va="center", fontsize=fs, linespacing=1.0)
    ax.text(xr + bw, sum(Rc) / 2, f"finally\ncorrect\n{right1}", ha="left",
            va="center", fontsize=fs, linespacing=1.0)
    ax.text(xr + bw, sum(Rw) / 2, f"finally\nwrong\n{wrong1}", ha="left",
            va="center", fontsize=fs, linespacing=1.0)
    ax.text(0.5, (sum(kept_L) + sum(kept_R)) / 4 + 4, f"kept {d['kept']}",
            ha="center", va="center", fontsize=fs)
    ax.text(0.5, d["still"] / 2, f"still wrong {d['still']}", ha="center",
            va="center", fontsize=fs, color="white")
    # the two crossing bands: labels inside the light kept band, leader lines
    # to the band ends (corrupted at its source, fixed at its destination)
    ax.annotate(f"corrupted {d['cor']}", (x0 + 0.02, sum(cor_L) / 2),
                xytext=(x0 + 0.05, kept_L[0] + 5), fontsize=fs, ha="left",
                va="center", color="#8A5A00", weight="bold",
                arrowprops=dict(arrowstyle="-", color="#8A5A00", lw=0.7,
                                shrinkA=1, shrinkB=0))
    ax.annotate(f"fixed {d['fix']}", (x1 - 0.02, sum(fix_R) / 2),
                xytext=(x1 - 0.05, kept_R[0] + 13), fontsize=fs, ha="right",
                va="center", color="#1F5F86", weight="bold",
                arrowprops=dict(arrowstyle="-", color="#1F5F86", lw=0.7,
                                shrinkA=1, shrinkB=0))
    net = d["fix"] - d["cor"]
    ax.set_title(title + minus(f": net {net:+d} of {n} items"),
                 color=tcol, fontsize=8.2, weight="bold")
    ax.set_xlim(-0.42, 1.42)
    ax.set_ylim(-2, n + g + 2)
    ax.axis("off")


def fig_flow():
    tc = trace_counts()
    rows = load("gsm8k_matrix.jsonl")
    params = {r["config"]: r["params"] for r in rows}
    deb, ge = params["DEBATE_N3_R1"], params["GE_R2"]
    fig, axes = plt.subplots(1, 2, figsize=(TEXTW, 2.45))
    draw_flow(axes[0], tc["debate"],
              f"DEBATE (N = {deb['N']}, R = {deb['R']})",
              COL["DEBATE"])
    draw_flow(axes[1], tc["ge"], f"GE (R = {ge['R']})",
              COL["GE"])
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig_flow")
    plt.close(fig)
    return tc


# ==== decision chart ====
def fig_decision():
    """Flowchart of the rows of the deployment-guideline table (tab:guidelines) and
    nothing else. k_top is the largest SC k on the GSM8K mixture frontier; the
    engineering numbers are the macros of paper/eaai/mmlu_pro_eng_macros.tex."""
    agg = analyze.aggregate(load("gsm8k_matrix.jsonl"))
    hull = mixture_frontier(agg, list(agg))
    ktop = max(int(c.split("_k")[1]) for c in hull if c.startswith("SC_k"))
    me = macros(os.path.join(ROOT, "paper", "eaai", "mmlu_pro_eng_macros.tex"))
    W, H = TEXTW, 5.75
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    fs = 7.7
    sty = "round,pad=0,rounding_size=0.06"

    def q(x, y, w, h, t):
        return box(ax, x, y, w, h, t, "#F2F2F2", "#666666", fs=fs, style=sty)

    def rec(x, y, w, h, t, c, fsz=fs, ls="-"):
        return box(ax, x, y, w, h, t, tint(c, 0.8), c, fs=fsz, lw=1.1, style=sty,
                   ls=ls)

    def lab(x, y, t, ha="center", va="center"):
        ax.text(x, y, t, fontsize=7.2, ha=ha, va=va, style="italic", color="#333333")

    # top: task type
    s0 = q(1.35, 5.36, 2.7, 0.34, "what kind of task?")
    # four task branches (Table 7 rows on task type, code row included)
    gap = 0.08
    bw = (W - 2 * 0.06 - 3 * gap) / 4
    xs4 = [0.06 + i * (bw + gap) for i in range(4)]
    xR = xs4[3]
    yb, hb0 = 4.30, 0.78
    out = rec(xs4[0], yb, bw, hb0, "open-ended or not\nautomatically\nverifiable: "
              "outside\nthe measured scope;\nmeasure first", COL["OTHER"], fsz=7.5)
    kn = rec(xs4[1], yb, bw, hb0, "knowledge-recall\nmultiple choice:\nsingle call (SA)",
             COL["SA"], fsz=7.5)
    cd = rec(xs4[2], yb, bw, hb0, "code with hidden\ntests, exact-program\nvote: single "
             "call (SA);\nsampling and\ncoordination lost", COL["SA"], fsz=7.5)
    rs = q(xR, yb, bw, hb0, "numeric or symbolic\nanswer, or\ncalculation-based\n"
           "multiple choice")
    for bx_ in (out, kn, cd, rs):
        arrow(ax, (s0[0] + s0[2] / 2, s0[1]), (bx_[0] + bx_[2] / 2, bx_[1] + bx_[3]))
    # reasoning branch continues
    bq = q(xR, 3.50, bw, 0.42, "budget above one\ncall per query?")
    arrow(ax, (xR + bw / 2, rs[1]), (xR + bw / 2, bq[1] + bq[3]))
    one = rec(xs4[2], 3.50, bw - 0.3, 0.42, "single call (SA)", COL["SA"])
    arrow(ax, L(bq), R(one))
    lab((one[0] + one[2] + bq[0]) / 2, bq[1] + bq[3] / 2 + 0.08, "no", va="bottom")
    scw = 2.28
    sc = rec(W - 0.06 - scw, 2.16, scw, 1.08,
             r"SC$_k$: self-consistency, majority vote," "\n"
             f"k up to about {ktop}; returns diminish.\n"
             f"Engineering task: {me['scDeltaProEng']} points over\n"
             f"a single call (p = {me['pScSaProEng']}; one run,\nnot replicated).\n"
             "Latency-bound: run the samples in\nparallel, not sequential rounds",
             COL["SC"], fsz=7.5)
    arrow(ax, (xR + bw / 2, bq[1]), (xR + bw / 2, sc[1] + sc[3]))
    lab(xR + bw / 2 + 0.06, bq[1] - 0.1, "yes", ha="left")
    # side note: agreement gate (directional)
    gate = rec(0.06, 2.32, 2.3, 0.86,
               "side note, directional evidence only:\nbudget-bound service with "
               "mixed easy\nand hard queries: draw two samples,\nescalate only on "
               "disagreement", COL["SC"], fsz=7.5, ls="--")
    arrow(ax, R(gate), L(sc), col="#888888", lw=0.7)

    ax.text(0.06, 2.00, "Before adding a coordination step or variant:",
            fontsize=7.8, weight="bold", va="center")
    gx = 0.1
    wq = (W - 2 * 0.06 - 2 * gx) / 3
    yb2, hb = 0.48, 1.38
    odds = ", ".join(f"{v / (1 - v):g}" for v in (0.5, 0.8, 0.9))
    # Criterion row of tab:guidelines after its out-of-sample test
    # (scripts/criterion_predictive.py -> results/analysis/criterion_predictive.json)
    cp = json.load(open(os.path.join(ROOT, "results", "analysis",
                                     "criterion_predictive.json")))
    hi = [c for c in cp["cells"].values() if c["a"] > 0.5]
    nneg = sum(1 for c in hi if c["delta"] < 0)
    a_hi = min(c["a"] for c in hi)
    a_lo = max(c["a"] for c in cp["cells"].values() if c["a"] < 0.5)
    items = [
        ("critique, debate, or\nrevision step: screen on\n"
         f"base accuracy; at a \u2265 {a_hi:.2f},\nexpect a loss ({nneg} of {len(hi)} cells);\n"
         f"at a \u2264 {a_lo:.2f}, a gain is not\nestablished (2 cells, within\n"
         f"run-to-run noise); no cell\nin between, so the cut-off\nlies in "
         f"[{a_lo:.2f}, {a_hi:.2f}] (1/2 if\n"
         r"$p_\mathrm{fix} = p_\mathrm{cor}$)", "#4D4D4D"),
        ("planner-executor: do not\nuse on mathematics, code,\nor knowledge tasks "
         "(below\nthe single call); on the\nengineering task not\nsignificantly below "
         "SC k = 8\nbut below the mixture\nfrontier", COL["HIER"]),
        ("diversity by personas,\ntemperatures, or weaker\nmodels in a debate:\n"
         "do not expect a gain", COL["OTHER"]),
    ]
    for i, (t, c) in enumerate(items):
        rec(0.06 + i * (wq + gx), yb2, wq, hb, t, c, fsz=7.5)
    ax.text(0.06, 0.29, "Output-token billing or prefix caching: the same choices. "
            "Another open model family of similar size:", fontsize=7.2,
            va="center", color="#333333")
    ax.text(0.06, 0.10, "the same ordering held in three families; re-measure "
            "before relying on effect sizes.", fontsize=7.2, va="center",
            color="#333333")
    save(fig, "fig_decision")
    plt.close(fig)
    return ktop


def fig_code_energy():
    """HumanEval: pass@1 against tokens per query (left) and against gross GPU
    board energy per query (right), seven configurations, as in Table tab:code."""
    rows = load("humaneval_matrix_pc.jsonl")
    agg = analyze.aggregate(rows)
    main = ["SA", "SC_k2", "SC_k4", "SC_k8", "DEBATE_N3_R1", "GE_R2", "HIER_M3"]
    agg = {c: agg[c] for c in main}
    en = defaultdict(lambda: defaultdict(list))
    for r in rows:
        en[r["config"]][r["item_id"]].append(r["energy_j"])
    agg_e = {c: dict(agg[c], tokens=float(np.mean([np.mean(v) for v in en[c].values()])))
             for c in main}
    fig, axes = plt.subplots(1, 2, figsize=(TEXTW, 2.75), sharey=True)
    hulls = {}
    for ax, A, xl, key in ((axes[0], agg, "tokens per query (log scale)", "tokens"),
                           (axes[1], agg_e, "gross GPU energy per query, J (log scale)", "joules")):
        hulls[key] = frontier_panel(ax, A, "", xpad=(0.75, 1.45))
        ax.set_xlabel(xl)
        ax.set_ylim(40, 80)
        ax._pending_labels = [(pretty(c).replace("DEBATE N=3 R=1", "DEBATE")
                               .replace("GE R=2", "GE").replace("HIER M=3", "HIER"),
                               A[c]["tokens"], A[c]["accuracy"] * 100, None)
                              for c in main]
    axes[0].set_ylabel("pass@1 (%)")
    fig.legend(handles=legend_handles(), loc="upper center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, 1.0), columnspacing=0.7, handletextpad=0.35,
               handlelength=1.6)
    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=0.8)
    finish_labels(fig)
    save(fig, "fig_code_energy")
    plt.close(fig)
    return hulls


# ==== graphical abstract ====
def graphical_abstract():
    """Elsevier graphical abstract: 13 x 5 cm (w x h); at 300 dpi the PNG is
    1535 x 591 px, above the 1328 x 531 px minimum."""
    W, H = 13 / 2.54, 5 / 2.54
    fig = plt.figure(figsize=(W, H))
    tfs, fs = 7.6, 7.2
    # panel 1: the question
    ax0 = fig.add_axes([0.0, 0.0, 0.34, 1.0])
    ax0.set_xlim(0, 0.34 * W)
    ax0.set_ylim(0, 1.97)
    ax0.axis("off")
    ax0.text(0.05, 1.90, "Same small model and\ntoken budget per query:\n"
             "sample or coordinate?", fontsize=tfs, weight="bold", va="top",
             linespacing=1.1)
    sfs = 7.2
    qb = box(ax0, 0.04, 0.72 - 0.09, 0.40, 0.18, "query", "#F2F2F2", "#7F7F7F",
             fs=sfs)
    xv, wv, xa, wa = 0.88, 0.32, 1.25, 0.47
    # sampling branch: k independent samples -> vote -> answer
    ax0.text(0.52, 1.30, "sample and vote", fontsize=fs, weight="bold",
             color=COL["SC"], va="center")
    yc = 1.08
    vb = box(ax0, xv, yc - 0.08, wv, 0.16, "vote", "#FFFFFF", "#555555", ls="--",
             fs=sfs)
    ab = box(ax0, xa, yc - 0.08, wa, 0.16, "answer", "#FFFFFF", "#333333",
             fs=sfs, weight="bold")
    for i in range(3):
        yy = yc + (1 - i) * 0.11
        b = box(ax0, 0.53, yy - 0.04, 0.24, 0.08, "", tint(COL["SC"], 0.7),
                COL["SC"], lw=0.7)
        arrow(ax0, R(qb), L(b), lw=0.5, ms=4)
        arrow(ax0, R(b), L(vb), lw=0.5, ms=4)
    arrow(ax0, R(vb), L(ab), lw=0.5, ms=4)
    # coordination branch: e.g. debate, two rounds -> vote -> answer
    ax0.text(0.52, 0.60, "or debate", fontsize=fs, weight="bold",
             color=COL["DEBATE"], va="center")
    yc = 0.36
    vd = box(ax0, xv, yc - 0.08, wv, 0.16, "vote", "#FFFFFF", "#555555", ls="--",
             fs=sfs)
    ad = box(ax0, xa, yc - 0.08, wa, 0.16, "answer", "#FFFFFF", "#333333",
             fs=sfs, weight="bold")
    r0, r1 = [], []
    for i in range(3):
        yy = yc + (1 - i) * 0.11 - 0.04
        r0.append(box(ax0, 0.53, yy, 0.1, 0.08, "", tint(COL["DEBATE"], 0.7),
                      COL["DEBATE"], lw=0.7))
        r1.append(box(ax0, 0.70, yy, 0.1, 0.08, "", tint(COL["DEBATE"], 0.7),
                      COL["DEBATE"], lw=0.7))
    for i in range(3):
        arrow(ax0, R(qb), L(r0[i]), lw=0.5, ms=4)
        for j in range(3):
            if i != j:
                arrow(ax0, R(r0[j]), L(r1[i]), col="#999999", lw=0.4, ms=3)
        arrow(ax0, R(r1[i]), L(vd), lw=0.5, ms=4)
    arrow(ax0, R(vd), L(ad), lw=0.5, ms=4)
    # panel 2: criterion
    ax1 = fig.add_axes([0.435, 0.21, 0.175, 0.52])
    pts = criterion_points(("debate_item", "ge_item"))
    draw_criterion(ax1, pts, ylim=(0.1, 100), compact=True)
    ax1.set_xlabel(r"base accuracy $a$", fontsize=fs, labelpad=1)
    ax1.set_ylabel(r"$p_{\mathrm{fix}}/p_{\mathrm{cor}}$", fontsize=fs, labelpad=0)
    ax1.tick_params(labelsize=7.2)
    ax1.text(0.52, 25, "helps", fontsize=fs, weight="bold", color="#1F5F86")
    ax1.text(0.52, 0.18, "hurts", fontsize=fs, weight="bold", color="#8A5A00")
    d = pts[0]
    ax1.annotate("debate", (d["a"], d["ratio"]), xytext=(-12, 9),
                 textcoords="offset points", fontsize=7.2, color=COL["DEBATE"],
                 ha="right")
    g = pts[1]
    ax1.annotate("GE", (g["a"], g["ratio"]), xytext=(-12, -9),
                 textcoords="offset points", fontsize=7.2, color=COL["GE"],
                 ha="right")
    fig.text(0.525, 0.965, "Coordination pays\nonly if "
             r"$p_{\mathrm{fix}}/p_{\mathrm{cor}} > a/(1-a)$",
             fontsize=tfs, weight="bold", ha="center", va="top", linespacing=1.15)
    # panel 3: GSM8K frontier
    ax2 = fig.add_axes([0.715, 0.21, 0.27, 0.50])
    agg, _ = agg_task("gsm8k")
    frontier_panel(ax2, agg, "", rings=False, jitter={"GE_R2": JIT}, xpad=(0.8, 1.9))
    ax2.set_xlabel("tokens per query", fontsize=fs, labelpad=1)
    ax2.set_ylabel("accuracy (%)", fontsize=fs, labelpad=1)
    ax2.tick_params(labelsize=7.2)
    ax2.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([500, 2000]))
    sc8, deb = agg["SC_k8"], agg["DEBATE_N3_R1"]
    ax2.text(0.04, 0.05, r"SA + SC$_k$", transform=ax2.transAxes, fontsize=7.2,
             color=COL["SC"], weight="bold")
    # label positions in data coordinates, in free space (leader lines)
    for t, (xt, yt, ha) in (("DEBATE_N3_R1", (4873, 65.5, "center")),
                            ("GE_R2", (4250, 91.0, "center")),
                            ("HIER_M3", (1600, 68.8, "right"))):
        a = agg[t]
        col = COL[t.split("_")[0]]
        lead = None if t == "HIER_M3" else dict(arrowstyle="-", color=col, lw=0.5,
                                                 shrinkA=0.5, shrinkB=3)
        ax2.annotate(t.split("_")[0],
                     (a["tokens"] * (JIT if t == "GE_R2" else 1.0), a["accuracy"] * 100),
                     xytext=(xt, yt), textcoords="data", fontsize=7.2,
                     color=col, ha=ha, va="center", arrowprops=lead,
                     bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none"))
    fig.text(0.85, 0.965, "Grade-school math (3B)", fontsize=tfs, weight="bold",
             ha="center", va="top")
    fig.text(0.85, 0.875,
             f"SC$_{{k=8}}$ {sc8['accuracy'] * 100:.1f}%, {fmt_int(sc8['tokens'])} tokens\n"
             f"debate {deb['accuracy'] * 100:.1f}%, {fmt_int(deb['tokens'])} tokens",
             fontsize=7.2, ha="center", va="top", linespacing=1.15)
    small = [(t.get_text(), t.get_fontsize()) for t in fig.findobj(matplotlib.text.Text)
             if t.get_visible() and t.get_text().strip() and t.get_fontsize() < MIN_PT]
    assert not small, ("graphical_abstract", small[:5])
    with plt.rc_context({"savefig.bbox": "standard"}):
        fig.savefig(os.path.join(UPLOAD, "graphical_abstract.pdf"))
        png = os.path.join(UPLOAD, "graphical_abstract.png")
        fig.savefig(png, dpi=300, facecolor="white")
    plt.close(fig)
    from PIL import Image
    Image.open(png).convert("RGB").save(png, dpi=(300, 300))


def emit_macros(f4, flows, units):
    """paper/eaai/figure_macros.tex: the numbers that figure captions and the
    text quote about the figures."""
    eng = f4["mmlu_pro_eng"]
    agg, hull = eng["agg"], eng["hull"]
    gap = hull_value(agg, hull, agg["HIER_M3"]["tokens"]) - agg["HIER_M3"]["accuracy"]
    me = macros(os.path.join(ROOT, "paper", "eaai", "mmlu_pro_eng_macros.tex"))
    # the equal-budget gap of the emitter interpolates the SC curve linearly;
    # SC k=4 and k=8 bracket HIER and are hull vertices, so the two agree
    assert f"{gap * 100:+.1f}" == me["gapHierProEng"], (gap, me["gapHierProEng"])
    names = {"SA": "SA", "SC_k2": "SC $k{=}2$", "SC_k4": "SC $k{=}4$",
             "SC_k8": "SC $k{=}8$", "HIER_M3": "HIER", "GE_R2": "GE",
             "DEBATE_N3_R1": "debate"}
    u = {p["key"]: p for p in units}
    d = {
        "mixFrontierProEng": ", ".join(names[c] for c in hull),
        "hierMixGapProEng": f"{gap * 100:.1f}",
        "flowN": str(flows["debate"]["n"]),
        "flowDebFinal": str(flows["debate"]["kept"] + flows["debate"]["fix"]),
        "flowGeFinal": str(flows["ge"]["kept"] + flows["ge"]["fix"]),
        "flowGeFinalPct": f"{100 * (flows['ge']['kept'] + flows['ge']['fix']) / flows['ge']['n']:.1f}",
        "flowDebFinalPct": f"{100 * (flows['debate']['kept'] + flows['debate']['fix']) / flows['debate']['n']:.1f}",
        "unitRatioAgentSev": f"{u['agent70']['ratio']:.1f}",
        "unitRatioItemSev": f"{u['item70']['ratio']:g}",
        "figJitterPct": f"{(JIT - 1) * 100:.0f}",
    }
    lines = ["% Generated by scripts/make_eaai_figures.py. Do not edit by hand."]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in d.items()]
    with open(os.path.join(ROOT, "paper", "eaai", "figure_macros.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    return d


def main():
    os.makedirs(OUT, exist_ok=True)
    report = {}
    report["workflow_calls"] = fig_workflows()
    keys = ("lab", "fix", "wrong", "cor", "right", "a", "a_ci", "ratio", "ratio_ci",
            "odds", "p_zero", "p_inf")
    report["criterion"] = [{k: p[k] for k in keys} for p in fig_criterion()]
    units = fig_criterion_units()
    report["criterion_units"] = [{k: p[k] for k in keys} for p in units]
    flows = fig_flow()
    f4 = fig_frontier4()
    report["hulls"] = {t: v["hull"] for t, v in f4.items()}
    report["gsm8k_hull"] = fig_gsm8k_pareto()["hull"]
    report["proeng_hull"] = fig_proeng_pareto()["hull"]
    report["code_hulls"] = fig_code_energy()
    report["decision_ktop"] = fig_decision()
    graphical_abstract()
    report["macros"] = emit_macros(f4, flows, units)
    print(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
