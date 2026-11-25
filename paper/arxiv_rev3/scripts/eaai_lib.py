"""Helpers for the adapted rev3 figures, copied from scripts/make_eaai_figures.py and
scripts/make_aaibd_figure.py (those originals are not imported and not modified).

Only presentation differs from the originals (plotstyle names, colors, error bars); the
data functions (aggregation, mixture frontier, trace counts, unit clusters, bootstrap summaries)
are verbatim copies so that every plotted number is the number the original computed.
"""
import math
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker  # noqa: F401
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Polygon

import plotstyle as FS

ROOT = FS.ROOT
RES = os.path.join(ROOT, "results")
sys.path.insert(0, ROOT)
from harness import analyze  # noqa: E402

COL, MARK = FS.COL, FS.MARK


# ==== data (verbatim copies) ====
def load(name):
    return analyze.load(os.path.join(RES, name))


def calls_per_config(rows):
    d = defaultdict(set)
    for r in rows:
        d[r["config"]].add(r["n_calls"])
    return d


def trace_counts():
    """Copy of make_eaai_figures.trace_counts (results/mechanistic.jsonl)."""
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
    """Copy of make_eaai_figures.agg_task (seven main configurations)."""
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
    """Copy of make_eaai_figures.mixture_frontier (upper concave hull, linear cost)."""
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


def _majority(answers):
    vals = [v for v in answers if v is not None]
    if not vals:
        return None
    top = Counter(str(v) for v in vals).most_common(1)[0][0]
    for v in vals:
        if str(v) == top:
            return v
    return vals[0]


def units_3b():
    """Copy of make_eaai_figures.units_3b: (right, wrong, fix, cor) clusters per item."""
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
    """Copy of make_eaai_figures.units_70b (results/gsm8k_llama70b.jsonl, debate N=3, R=1)."""
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
    """Copy of make_eaai_figures.summarize (item-cluster bootstrap, B=10,000, seed 0)."""
    X = np.array(clusters, dtype=float)
    Rt, Wt, Ft, Ct = X.sum(0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(X), size=(B, len(X)))
    S = X[idx].sum(1)
    Rb, Wb, Fb, Cb = S[:, 0], S[:, 1], S[:, 2], S[:, 3]
    ab = Rb / (Rb + Wb)
    with np.errstate(divide="ignore", invalid="ignore"):
        rb = (Fb / Wb) / (Cb / Rb)
    rb = np.sort(rb[~np.isnan(rb)])
    lo_i, hi_i = int(np.floor(0.025 * (len(rb) - 1))), int(np.ceil(0.975 * (len(rb) - 1)))
    ratio = (Ft / Wt) / (Ct / Rt) if Ct else float("inf")
    return dict(right=int(Rt), wrong=int(Wt), fix=int(Ft), cor=int(Ct),
                a=Rt / (Rt + Wt), a_ci=tuple(np.percentile(ab, [2.5, 97.5])),
                ratio=ratio, ratio_ci=(float(rb[lo_i]), float(rb[hi_i])),
                p_zero=float(np.mean(rb == 0)), p_inf=float(np.mean(np.isinf(rb))),
                odds=Rt / Wt)


def criterion_points(units=("debate_item", "ge_item", "item70")):
    """Copy of make_eaai_figures.criterion_points, with plotstyle display labels; the count
    checks against paper/arxiv_rev3/scale70b_macros.tex are kept."""
    u3, u70 = units_3b(), units_70b()
    m = FS.macros("scale70b_macros.tex")
    spec = {
        "debate_item": ("3B debate, per item", "DEBATE", u3["debate_item"], COL["DEBATE"], 1 / 1.006),
        "ge_item": ("3B GE, per item", "GE", u3["ge_item"], COL["GE"], 1.006),
        "item70": ("70B debate, per item", "DEBATE", u70["item"], "white", 1.0),
        "ge_rev": ("3B GE, per revision", "GE", u3["ge_rev"], "white", 1.0),
        "agent70": ("70B debate, per agent", "DEBATE", u70["agent"], "white", 1.0),
    }
    pts = []
    for k in units:
        lab, topo, cl, mfc, jit = spec[k]
        p = summarize(cl)
        p.update(key=k, lab=lab, topo=topo, mfc=mfc, jit=jit)
        pts.append(p)
    chk = {q["key"]: q for q in pts}
    if "item70" in chk:
        q = chk["item70"]
        assert (q["fix"], q["wrong"], q["cor"], q["right"]) == (
            int(m["sevCritFix"]), int(m["sevCritInitWrong"]), int(m["sevCritCor"]),
            int(m["sevCritInitRight"]))
    if "agent70" in chk:
        q = chk["agent70"]
        assert (q["fix"], q["wrong"], q["cor"], q["right"]) == (
            int(m["sevAgFix"]), int(m["sevAgWrong"]), int(m["sevAgCor"]), int(m["sevAgRight"]))
    if "ge_rev" in chk:
        q, go = chk["ge_rev"], trace_counts()["ge_opp"]
        assert (q["fix"], q["wrong"], q["cor"], q["right"]) == (
            go["fix"], go["wrong"], go["cor"], go["right"])
    return pts


# ==== drawing ====
PARENT = {"GES": "GE", "HIERC": "HIER"}
VARIANTS = {"GES_R2", "HIERC_M3", "DEBATE_N3_R2"}
SWEEPS = {"DEBATE": ["DEBATE_N2_R1", "DEBATE_N3_R1"], "GE": ["GE_R1", "GE_R2"],
          "HIER": ["HIER_M2", "HIER_M3"]}


def topo_of(a):
    t = PARENT.get(a["topology"], a["topology"])
    return t if t in COL else "OTHER"


def errbar(ax, x, y, lo, hi, col, mk, ms=FS.MS, mfc=None, zorder=3):
    """Marker with a thin error bar in its own color (plotstyle.errorbar)."""
    return FS.errorbar(ax, x, y, lo, hi, color=col, marker=mk, mfc=mfc, ms=ms, zorder=zorder)


def frontier_panel(ax, agg, keys_sc=None, rings=True, sweep_lines=False, xpad=(0.8, 1.3),
                   ms=FS.MS, marker_alpha=1.0, skip=()):
    """Adapted from make_eaai_figures.frontier_panel: same mixture frontier band, rings,
    sampling line and markers; no jitter; thin error bars in each point's color. Configurations
    in skip are not drawn (they still enter the frontier)."""
    keys = list(agg)
    hull = mixture_frontier(agg, keys)
    hx, hy = [], []
    for c0, c1 in zip(hull[:-1], hull[1:]):
        xs = np.linspace(agg[c0]["tokens"], agg[c1]["tokens"], 60)
        hx.extend(xs)
        hy.extend(np.interp(xs, [agg[c0]["tokens"], agg[c1]["tokens"]],
                            [agg[c0]["accuracy"] * 100, agg[c1]["accuracy"] * 100]))
    if len(hull) == 1:
        hx, hy = [agg[hull[0]]["tokens"]], [agg[hull[0]]["accuracy"] * 100]
    ax.plot(hx, hy, "-", color=FS.BAND, lw=6, zorder=1, solid_capstyle="round")
    if rings:
        ax.plot([agg[c]["tokens"] for c in hull], [agg[c]["accuracy"] * 100 for c in hull],
                "o", ms=9.5, mfc="none", mec="#8C8C8C", mew=0.9, zorder=5, clip_on=False)
    sc = sorted((agg[c]["tokens"], agg[c]["accuracy"] * 100) for c in keys
                if agg[c]["topology"] in ("SA", "SC"))
    ax.plot(*zip(*sc), "-", color=COL["SC"], lw=FS.LW, zorder=2)
    if sweep_lines:
        for topo, cs in SWEEPS.items():
            if all(c in agg for c in cs):
                ax.plot([agg[c]["tokens"] for c in cs], [agg[c]["accuracy"] * 100 for c in cs],
                        "--", color=COL[topo], lw=0.9, zorder=2)
    for c, a in agg.items():
        if c in skip:
            continue
        t = topo_of(a)
        col = COL[t]
        mfc = "white" if c in VARIANTS else col
        y = a["accuracy"] * 100
        errbar(ax, a["tokens"], y, a["acc_lo"] * 100, a["acc_hi"] * 100, col, MARK[t], ms, mfc)
    ax.set_xscale("log")
    xs = [a["tokens"] for a in agg.values()]
    ax.set_xlim(min(xs) * xpad[0], max(xs) * xpad[1])
    lo, hi = ax.get_xlim()
    ticks = [t for t in (200, 500, 1000, 2000, 5000, 10000) if lo <= t <= hi]
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    return hull


def legend_handles(with_front=True, extra=(), names=None):
    """Legend keys: mixture frontier (optional), SA, SC k (line), debate, GE, HIER."""
    names = names or {}
    h = []
    if with_front:
        h.append(Line2D([], [], color=FS.BAND, lw=6, marker="o", ms=8.5,
                        mfc="none", mec="#8C8C8C", mew=0.9, solid_capstyle="butt",
                        label="mixture frontier"))
    h.append(Line2D([], [], color=COL["SA"], marker="o", ls="none", ms=FS.MS, label="SA"))
    h.append(Line2D([], [], color=COL["SC"], marker="o", lw=FS.LW, ms=FS.MS, label=r"SC $k$"))
    for t, lab in (("DEBATE", "debate"), ("GE", "GE"), ("HIER", "HIER")):
        h.append(Line2D([], [], color=COL[t], marker=MARK[t], ls="none", ms=FS.MS,
                        label=names.get(t, lab)))
    h.extend(extra)
    return h


def obstacles(ax, include_text=True):
    """Display-space obstacles in ax: sampled points along every line (error bars included)
    and a box per marker (adapted copy of make_eaai_figures._obstacles)."""
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
                r = ch.get_markersize() * ax.figure.dpi / 72 / 2 + 1
                for x, y in xy:
                    boxes.append((x - r, y - r, x + r, y + r))
        elif isinstance(ch, LineCollection):
            for seg in ch.get_segments():
                xy = ax.transData.transform(np.asarray(seg, float))
                for (x0, y0), (x1, y1) in zip(xy[:-1], xy[1:]):
                    n = int(max(abs(x1 - x0), abs(y1 - y0)) / 2) + 2
                    pts.extend(zip(np.linspace(x0, x1, n), np.linspace(y0, y1, n)))
    return (np.array(pts) if pts else np.zeros((0, 2))), boxes


def place_labels(ax, items, fontsize=FS.FONT_PT, pad=1.5, color="#222222", avoid_lines=True,
                 max_d=60, cross_pen=25):
    """Greedy collision-free label placement (copy of make_eaai_figures.place_labels with a
    color argument): each label goes to the nearest offset whose box hits no line, error bar,
    marker, axis edge or earlier label; a short grey leader joins a label set away from its
    point. items: (text, x, y[, color])."""
    fig = ax.figure
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    pts, boxes = obstacles(ax)
    axbb = ax.get_window_extent(rend)
    k = fig.dpi / 72.0
    dxs = [0] + [s * v for v in (5, 10, 16, 24, 34, 46, 60, 76) for s in (1, -1)]
    dys = [0] + [s * v for v in (6, 11, 17, 24, 32, 42, 54, 66, 80) for s in (1, -1)]
    cands = sorted({(dx, dy) for dx in dxs for dy in dys if (dx, dy) != (0, 0)
                    and math.hypot(dx, dy) <= max_d},
                   key=lambda d: (d[0] ** 2 + d[1] ** 2, abs(d[1])))
    placed, out = [], []
    for it in items:
        text, x, y = it[:3]
        col = it[3] if len(it) > 3 and it[3] else color
        px, py = ax.transData.transform((x, y))
        best, best_cost = None, None
        for dx, dy in cands:
            dist = math.hypot(dx, dy)
            if best_cost is not None and dist > best_cost:
                break
            ha = "left" if dx > 0 else ("right" if dx < 0 else "center")
            t = ax.text(px + dx * k, py + dy * k, text, transform=None, fontsize=fontsize,
                        ha=ha, va="center")
            bb = t.get_window_extent(rend)
            t.remove()
            x0, y0, x1, y1 = bb.x0 - pad, bb.y0 - pad, bb.x1 + pad, bb.y1 + pad
            ok = (x0 >= axbb.x0 + 2 and x1 <= axbb.x1 - 2 and y0 >= axbb.y0 + 2
                  and y1 <= axbb.y1 - 2)
            if ok and avoid_lines and len(pts):
                ins = (pts[:, 0] > x0) & (pts[:, 0] < x1) & (pts[:, 1] > y0) & (pts[:, 1] < y1)
                ok = not ins.any()
            if ok:
                ok = not any(b[0] < x1 and b[2] > x0 and b[1] < y1 and b[3] > y0
                             for b in boxes + placed)
            if not ok:
                continue
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
                    cost += cross_pen * len(np.unique(np.round(pts[hit] / 6), axis=0))
                    for b in boxes + placed:   # leader through a marker or a placed label
                        cx, cy = (b[0] + b[2]) / 2 - px, (b[1] + b[3]) / 2 - py
                        al = cx * ux + cy * uy
                        if 6 * k < al < L_ and abs(cx * uy - cy * ux) < (b[2] - b[0]) / 2:
                            cost += cross_pen
            if best_cost is None or cost < best_cost:
                best, best_cost = (ha, dx, dy, (x0, y0, x1, y1)), cost
        if best is None:
            raise RuntimeError(f"no free position for label {text!r}")
        ha, dx, dy, bb = best
        placed.append(bb)
        far = dx * dx + dy * dy > 13 ** 2
        out.append(ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points",
                               ha=ha, va="center", fontsize=fontsize, color=col, zorder=6,
                               arrowprops=(dict(arrowstyle="-", color="#777777", lw=0.5,
                                                shrinkA=1, shrinkB=3) if far else None)))
    return out


def check_labels(ax, texts, pad=0.5, ignore_soft=False):
    """Assert that no label box touches a line, error bar, marker, axis edge or another label."""
    fig = ax.figure
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    pts, boxes = obstacles(ax)
    axbb = ax.get_window_extent(rend)
    k = fig.dpi / 72.0
    done, problems = [], []
    for t in texts:
        bb = matplotlib.text.Text.get_window_extent(t, rend)   # text only, not the leader
        x0, y0, x1, y1 = bb.x0 - pad * k, bb.y0 - pad * k, bb.x1 + pad * k, bb.y1 + pad * k
        name = t.get_text()
        if not (x0 >= axbb.x0 and x1 <= axbb.x1 and y0 >= axbb.y0 and y1 <= axbb.y1):
            problems.append((name, "axis edge"))
        if len(pts) and ((pts[:, 0] > x0) & (pts[:, 0] < x1) & (pts[:, 1] > y0)
                         & (pts[:, 1] < y1)).any():
            problems.append((name, "line"))
        for b in boxes:
            if b[0] < x1 and b[2] > x0 and b[1] < y1 and b[3] > y0:
                problems.append((name, "marker"))
                break
        for other, b in done:
            if b[0] < x1 and b[2] > x0 and b[1] < y1 and b[3] > y0:
                problems.append((name, "label " + other))
        done.append((name, (x0, y0, x1, y1)))
    return problems


def box(ax, x, y, w, h, text, fc, ec, ls="-", fs=FS.FONT_PT, lw=0.9, weight="normal",
        style="round,pad=0,rounding_size=0.05"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, fc=fc, ec=ec, lw=lw, ls=ls,
                                zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, zorder=4,
            weight=weight, linespacing=1.0)
    return (x, y, w, h)


def arrow(ax, p, q, col="#444444", lw=0.8, rad=0.0, ms=6):
    ax.annotate("", xy=q, xytext=p, zorder=2,
                arrowprops=dict(arrowstyle="-|>", color=col, lw=lw, mutation_scale=ms,
                                shrinkA=0, shrinkB=0, connectionstyle=f"arc3,rad={rad}"))


def R(b):
    return (b[0] + b[2], b[1] + b[3] / 2)


def L(b):
    return (b[0], b[1] + b[3] / 2)


def band(ax, x0, x1, y0a, y0b, y1a, y1b, col, alpha=1.0):
    t = np.linspace(0, 1, 60)
    s = 3 * t ** 2 - 2 * t ** 3
    xs = x0 + (x1 - x0) * t
    top = y0b + (y1b - y0b) * s
    bot = y0a + (y1a - y0a) * s
    poly = np.vstack([np.c_[xs, top], np.c_[xs[::-1], bot[::-1]]])
    ax.add_patch(Polygon(poly, closed=True, fc=col, ec="none", alpha=alpha, zorder=2))


def place_at_bar_ends(ax, items, fontsize=FS.FONT_PT, gap=1.6, pad=0.6, color="#333333",
                      fallback_d=40):
    """Label each point just beyond one end of its own error bar (no leader needed), or beside
    its marker; the first position, in an alternating below/above preference order, whose box
    hits no line, error bar, marker, axis edge or earlier label is kept. If none is free, fall
    back to place_labels (short grey leader, at most fallback_d points away).
    items: (text, x, y, lo, hi) in data coordinates."""
    fig = ax.figure
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    pts, boxes = obstacles(ax)
    axbb = ax.get_window_extent(rend)
    k = fig.dpi / 72.0
    placed, out, rest = [], [], []
    for n, (text, x, y, lo, hi) in enumerate(items):
        px, py = ax.transData.transform((x, y))
        _, plo = ax.transData.transform((x, lo))
        _, phi = ax.transData.transform((x, hi))
        r = 3.2 * k
        below = (px, plo - gap * k, "center", "top")
        above = (px, phi + gap * k, "center", "bottom")
        right = (px + r + gap * k, py, "left", "center")
        left = (px - r - gap * k, py, "right", "center")
        order = [below, above, right, left] if n % 2 == 0 else [above, below, right, left]
        done = False
        for (tx, ty, ha, va) in order:
            t = ax.text(tx, ty, text, transform=None, fontsize=fontsize, ha=ha, va=va)
            bb = t.get_window_extent(rend)
            t.remove()
            x0, y0, x1, y1 = bb.x0 - pad * k, bb.y0 - pad * k, bb.x1 + pad * k, bb.y1 + pad * k
            ok = x0 >= axbb.x0 + 1 and x1 <= axbb.x1 - 1 and y0 >= axbb.y0 + 1 and y1 <= axbb.y1 - 1
            if ok and len(pts):
                ok = not ((pts[:, 0] > x0) & (pts[:, 0] < x1) & (pts[:, 1] > y0)
                          & (pts[:, 1] < y1)).any()
            if ok:
                ok = not any(b[0] < x1 and b[2] > x0 and b[1] < y1 and b[3] > y0
                             for b in boxes + placed)
            if ok:
                placed.append((x0, y0, x1, y1))
                tx_d, ty_d = ax.transData.inverted().transform((tx, ty))
                out.append(ax.text(tx_d, ty_d, text, ha=ha, va=va, fontsize=fontsize,
                                   color=color, zorder=6))
                done = True
                break
        if not done:
            rest.append((text, x, y, color))
    if rest:
        out += place_labels(ax, rest, fontsize=fontsize, pad=1.0, color=color,
                            max_d=fallback_d)
    return out


def dodge_x(ax, fixed, movable, sep=7.0, max_shift=24.0):
    """Presentation only: horizontal display offsets for markers that would sit on another
    point's interval. fixed and movable are lists of (x, y, lo, hi) in data coordinates; fixed
    points (the sampling curve) are never moved. Movable points are pushed apart until every
    pair whose intervals (marker included) overlap vertically is at least sep points apart
    horizontally, without changing the left-to-right order of any two points (so a workflow
    never crosses the sampling point it is compared with). Returns (new x in data coordinates,
    shift in points) per movable point; y, lo and hi are never changed. Call after the axis
    limits are final."""
    fig = ax.figure
    fig.canvas.draw()
    k = fig.dpi / 72.0
    T = ax.transData
    r = FS.MS / 2 + FS.MEW   # marker half-height in points: intervals touching a marker clash
    pts = []
    for n, p in enumerate(list(fixed) + list(movable)):
        x, y, lo, hi = p
        px = T.transform((x, y))[0] / k
        pts.append([px, T.transform((x, lo))[1] / k - r, T.transform((x, hi))[1] / k + r,
                    n >= len(fixed), px])
    order = sorted(range(len(pts)), key=lambda i: (pts[i][4], i))
    rank = {i: n for n, i in enumerate(order)}
    for _ in range(2000):
        moved = False
        for a in range(len(pts)):
            for b in range(len(pts)):
                if rank[a] >= rank[b]:
                    continue
                A, B = pts[a], pts[b]                      # A left of B in the original order
                need = sep if (A[1] < B[2] and A[2] > B[1]) else 0.0
                deficit = need - (B[0] - A[0])
                if deficit <= 1e-6:
                    continue
                if A[3] and B[3]:
                    A[0] -= deficit / 2; B[0] += deficit / 2
                elif A[3]:
                    A[0] -= deficit
                elif B[3]:
                    B[0] += deficit
                else:
                    continue                               # two fixed points: not ours to move
                moved = True
        if not moved:
            break
    else:
        raise RuntimeError("dodge_x: did not converge")
    out = []
    for p in pts[len(fixed):]:
        d = p[0] - p[4]
        if abs(d) > max_shift:
            raise RuntimeError(f"dodge_x: shift {d:.1f} pt exceeds {max_shift} pt")
        out.append((float(T.inverted().transform((p[0] * k, 0))[0]), d))
    return out


def place_in_column(ax, items, fontsize=FS.FONT_PT, gap=1.6, pad=0.6, color="#333333",
                    lead_col="#8C8C8C", sides=True, dirs=(-1, 1), strict=True, side_r=3.4):
    """Label each point beside its marker if that is free, else in its own column: the label
    slides down from the lower end (or up from the upper end) of the point's own error bar to
    the nearest position whose box hits no line, error bar, marker, axis edge or earlier label;
    a short vertical grey leader joins bar end and label when they are apart. The shorter of
    the two slides is kept. items: (text, x, y, lo, hi) in data coordinates."""
    fig = ax.figure
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    pts, boxes = obstacles(ax)
    axbb = ax.get_window_extent(rend)
    k = fig.dpi / 72.0
    placed, out = [], []

    def free(x0, y0, x1, y1):
        if not (x0 >= axbb.x0 + 1 and x1 <= axbb.x1 - 1 and y0 >= axbb.y0 + 1
                and y1 <= axbb.y1 - 1):
            return False
        if len(pts) and ((pts[:, 0] > x0) & (pts[:, 0] < x1) & (pts[:, 1] > y0)
                         & (pts[:, 1] < y1)).any():
            return False
        if any(x0 < lx < x1 and min(la, lb) < y1 and max(la, lb) > y0 for lx, la, lb in leads):
            return False
        return not any(b[0] < x1 and b[2] > x0 and b[1] < y1 and b[3] > y0
                       for b in boxes + placed)

    def lead_ok(lx, la, lb):
        return not any(b[0] < lx < b[2] and min(la, lb) < b[3] and max(la, lb) > b[1]
                       for b in placed) and not any(
            b[0] < lx < b[2] and min(la, lb) < b[3] and max(la, lb) > b[1] for b in boxes
            if not (b[0] < lx < b[2] and b[1] < la < b[3]))

    leads = []
    for it in items:
        text, x, y, lo, hi = it[:5]
        tcol = it[5] if len(it) > 5 else color
        px, py = ax.transData.transform((x, y))
        _, plo = ax.transData.transform((x, lo))
        _, phi = ax.transData.transform((x, hi))
        t = ax.text(0, 0, text, transform=None, fontsize=fontsize)
        bb = t.get_window_extent(rend)
        t.remove()
        w, h = bb.width, bb.height
        best = None
        if sides:
            r = side_r * k
            for (ha, xa) in (("left", px + r + gap * k), ("right", px - r - gap * k - w)):
                box_ = (xa - pad * k, py - h / 2 - pad * k, xa + w + pad * k, py + h / 2 + pad * k)
                if free(*box_):
                    best = (0.0, ha, (px + r + gap * k) if ha == "left" else (px - r - gap * k),
                            py, "center", box_, None)
                    break
        if best is None:
            for sgn in dirs:
                if best is not None:
                    break                            # dirs is a preference order
                start = plo - gap * k if sgn < 0 else phi + gap * k
                d = 0.0
                while True:
                    edge = start + sgn * d          # label edge nearest the bar end
                    yb0, yb1 = (edge - h, edge) if sgn < 0 else (edge, edge + h)
                    box_ = (px - w / 2 - pad * k, yb0 - pad * k, px + w / 2 + pad * k, yb1 + pad * k)
                    if yb0 < axbb.y0 or yb1 > axbb.y1:
                        break
                    if free(*box_) and lead_ok(px, start, edge):
                        if best is None or d < best[0]:
                            best = (d, "center", px, edge, "top" if sgn < 0 else "bottom", box_,
                                    (start - sgn * gap * k, edge - sgn * 1.2 * k))
                        break
                    d += 1.0
        if best is None:
            if strict:
                raise RuntimeError(f"no free position for label {text!r}")
            print("UNPLACED", text)
            continue
        d, ha, tx, ty, va, box_, lead = best
        placed.append(box_)
        xd, yd = ax.transData.inverted().transform((tx, ty))
        out.append(ax.text(xd, yd, text, ha=ha, va=va, fontsize=fontsize, color=tcol, zorder=6))
        if lead is not None:
            leads.append((px, lead[0], lead[1]))
        if lead is not None and d > 2.5 * k:
            (_, y0d) = ax.transData.inverted().transform((px, lead[0]))
            (_, y1d) = ax.transData.inverted().transform((px, lead[1]))
            ax.plot([x, x], [y0d, y1d], color=lead_col, lw=0.5, ls=":", zorder=1.5)
    return out
