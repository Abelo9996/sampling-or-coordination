"""rev3 re-renders of four figures of scripts/make_eaai_figures.py, plus the criterion-units table
that replaces fig_criterion_units.

Data functions are verbatim copies in eaai_lib.py (the original script is not imported or
modified); only presentation changes (figstyle fonts, names and colors; legends and labels
moved off the data). Outputs (paper/arxiv_rev3/):
  figures/fig_flow.pdf (printed=1.0)  correction/corruption flows, results/mechanistic.jsonl
      (trace_counts); debate and GE parameters from results/gsm8k_matrix.jsonl. Asserted: the
      counts equal figure_macros.tex \\flowN, \\flowDebFinal, \\flowGeFinal, \\flowDebFinalPct,
      \\flowGeFinalPct (that file is read, never written). Corrupted bands in figstyle.CORR,
      fixed in figstyle.FIX.
  figures/fig_criterion.pdf (printed=0.8)  criterion map: 3B debate and GE per item
      (results/mechanistic.jsonl), 70B debate per item (results/gsm8k_llama70b.jsonl); counts
      asserted against scale70b_macros.tex (in criterion_points). Legend above the axes.
  figures/fig_code_energy.pdf (printed=1.0)  HumanEval (results/humaneval_matrix_pc.jsonl):
      pass@1 against tokens and against gross GPU energy. Asserted: accuracy, total tokens and
      energy equal tables/review_revision.json tasks.code.configs (acc, total_tokens, energy_j).
  figures/fig_proeng_pareto.pdf (printed=0.75)  MMLU-Pro engineering
      (results/mmlu_pro_eng_matrix.jsonl). Asserted: accuracy and total tokens equal
      review_revision.json tasks.proeng.configs; the mixture frontier equals
      figure_macros.tex \\mixFrontierProEng.
  tables/criterion_units_table.tex  rows of the dropped fig_criterion_units: counts, ratio
      p_fix/p_cor and its 95% item-cluster bootstrap interval (summarize(), B=10,000, seed 0)
      for 3B GE per item, 3B GE per revision opportunity, 70B debate per item, 70B debate per
      agent, and 3B debate per item. Asserted: the 70B ratios equal \\unitRatioItemSev and
      \\unitRatioAgentSev of figure_macros.tex.
Every label is checked against lines, error bars, markers, the axis edge and other labels.
"""
import json
import math
import os

import numpy as np

import figstyle as FS

FS.setup()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import eaai_lib as E  # noqa: E402

COL = FS.COL
TAB = os.path.join(FS.REV3, "tables")
FIGM = FS.macros("figure_macros.tex")
RR = json.load(open(os.path.join(TAB, "review_revision.json")))["tasks"]
FIX_TXT = "#1F5F86"     # dark sky blue for text on light backgrounds
CORR_TXT = FS.CORR


# ==== flows ====
def draw_flow(ax, d, title, tcol):
    n = d["n"]
    right0, wrong0 = d["kept"] + d["cor"], d["fix"] + d["still"]
    right1, wrong1 = d["kept"] + d["fix"], d["cor"] + d["still"]
    g, bw = 4.0, 0.075
    xl, xr = 0.0, 1.0
    Lc, Lw = (wrong0 + g, wrong0 + g + right0), (0, wrong0)
    Rc, Rw = (wrong1 + g, wrong1 + g + right1), (0, wrong1)
    for (x, seg) in ((xl, Lc), (xl, Lw), (xr, Rc), (xr, Rw)):
        ax.add_patch(plt.Rectangle((x - bw / 2, seg[0]), bw, seg[1] - seg[0], fc="#4D4D4D",
                                   ec="none", zorder=3))
    kept_L, cor_L = (Lc[1] - d["kept"], Lc[1]), (Lc[0], Lc[0] + d["cor"])
    fix_L, still_L = (Lw[1] - d["fix"], Lw[1]), (Lw[0], Lw[0] + d["still"])
    kept_R, fix_R = (Rc[1] - d["kept"], Rc[1]), (Rc[0], Rc[0] + d["fix"])
    cor_R, still_R = (Rw[1] - d["cor"], Rw[1]), (Rw[0], Rw[0] + d["still"])
    x0, x1 = xl + bw / 2, xr - bw / 2
    E.band(ax, x0, x1, *kept_L, *kept_R, FS.KEEP)
    E.band(ax, x0, x1, *still_L, *still_R, FS.STILL, 0.75)
    E.band(ax, x0, x1, *cor_L, *cor_R, FS.CORR, 0.85)
    E.band(ax, x0, x1, *fix_L, *fix_R, FS.FIX, 0.95)
    fs = 7.5
    ax.text(xl - bw, sum(Lc) / 2, f"initially\ncorrect\n{right0}", ha="right", va="center",
            fontsize=fs, linespacing=1.0)
    ax.text(xl - bw, sum(Lw) / 2, f"initially\nwrong\n{wrong0}", ha="right", va="center",
            fontsize=fs, linespacing=1.0)
    ax.text(xr + bw, sum(Rc) / 2, f"finally\ncorrect\n{right1}", ha="left", va="center",
            fontsize=fs, linespacing=1.0)
    ax.text(xr + bw, sum(Rw) / 2, f"finally\nwrong\n{wrong1}", ha="left", va="center",
            fontsize=fs, linespacing=1.0)
    ax.text(0.5, (sum(kept_L) + sum(kept_R)) / 4 + 4, f"kept {d['kept']}", ha="center",
            va="center", fontsize=fs)
    ax.text(0.5, d["still"] / 2, f"still wrong {d['still']}", ha="center", va="center",
            fontsize=fs, color="white")
    ax.annotate(f"corrupted {d['cor']}", (x0 + 0.02, sum(cor_L) / 2),
                xytext=(x0 + 0.05, kept_L[0] + 5), fontsize=fs, ha="left", va="center",
                color=CORR_TXT, weight="bold",
                arrowprops=dict(arrowstyle="-", color=CORR_TXT, lw=0.7, shrinkA=1, shrinkB=0))
    ax.annotate(f"fixed {d['fix']}", (x1 - 0.02, sum(fix_R) / 2),
                xytext=(x1 - 0.05, kept_R[0] + 13), fontsize=fs, ha="right", va="center",
                color=FIX_TXT, weight="bold",
                arrowprops=dict(arrowstyle="-", color=FIX_TXT, lw=0.7, shrinkA=1, shrinkB=0))
    net = d["fix"] - d["cor"]
    ax.set_title(title + FS.minus(f": net {net:+d} of {n} items"), color=tcol, fontsize=8.5,
                 weight="bold", pad=2)
    ax.set_xlim(-0.40, 1.40)
    ax.set_ylim(-1, n + g + 1)
    ax.axis("off")


def fig_flow():
    tc = E.trace_counts()
    rows = E.load("gsm8k_matrix.jsonl")
    params = {r["config"]: r["params"] for r in rows}
    deb, ge = params["DEBATE_N3_R1"], params["GE_R2"]
    d, g = tc["debate"], tc["ge"]
    assert str(d["n"]) == FIGM["flowN"] == str(g["n"])
    assert str(d["kept"] + d["fix"]) == FIGM["flowDebFinal"]
    assert str(g["kept"] + g["fix"]) == FIGM["flowGeFinal"]
    assert f"{100 * (d['kept'] + d['fix']) / d['n']:.1f}" == FIGM["flowDebFinalPct"]
    assert f"{100 * (g['kept'] + g['fix']) / g['n']:.1f}" == FIGM["flowGeFinalPct"]
    fig, axes = plt.subplots(1, 2, figsize=(FS.TEXTW, 2.1))
    draw_flow(axes[0], d, f"debate ($N$={deb['N']}, $R$={deb['R']})", COL["DEBATE"])
    draw_flow(axes[1], g, f"GE ($R$={ge['R']})", COL["GE"])
    fig.tight_layout(pad=0.2, w_pad=1.0)
    FS.save(fig, "fig_flow", printed=1.0)
    return tc


# ==== criterion ====
def point_marker(p):
    return "v" if p["ratio"] == 0 else FS.MARK[p["topo"]]


def draw_criterion(ax, pts, ylim=(0.03, 300), zero_label=None):
    a = np.linspace(0.5, 0.99, 400)
    be = a / (1 - a)
    ax.fill_between(a, be, ylim[1], color=FS.tint(FS.FIX, 0.72), lw=0, zorder=0)
    ax.fill_between(a, ylim[0], be, color=FS.tint(FS.CORR, 0.85), lw=0, zorder=0)
    ax.plot(a, be, color="black", lw=1.3, zorder=2)
    ax.set_yscale("log")
    ax.set_xlim(0.5, 0.99)
    ax.set_ylim(*ylim)
    ax.axhline(1.0, color="#555555", lw=0.8, ls=":", zorder=1)
    ax.text(0.515, 5.5, "coordination helps\n" r"($\Delta > 0$)", fontsize=8.5, weight="bold",
            color=FIX_TXT, va="center")
    ax.text(0.515, 0.12, "coordination hurts\n" r"($\Delta < 0$)", fontsize=8.5, weight="bold",
            color=CORR_TXT, va="center")
    floor = ylim[0] * 1.18
    for p in pts:
        c = COL[p["topo"]]
        mk = point_marker(p)
        x = p["a"] * p["jit"]
        xerr = [[p["a"] - p["a_ci"][0]], [p["a_ci"][1] - p["a"]]]
        y = p["ratio"] if p["ratio"] > 0 else floor
        lo, hi = p["ratio_ci"]
        if hi > 0:
            ax.plot([x, x], [max(lo, ylim[0]), min(hi, ylim[1])], color=c, lw=1.0, zorder=3.5,
                    clip_on=True, solid_capstyle="butt")
            for yy in (lo, hi):
                if ylim[0] < yy < ylim[1]:
                    ax.plot([x - 0.004, x + 0.004], [yy, yy], color=c, lw=0.9, zorder=3)
        ax.errorbar(x, y, xerr=xerr, fmt=mk, color=c, mfc=p["mfc"], mew=1.1, ms=6, capsize=2,
                    elinewidth=0.9, zorder=4, clip_on=False)
        if p["ratio"] == 0:
            txt = f"observed {p['fix']}/{p['wrong']} fixed\n(ratio 0)"
            if zero_label is None:
                ax.annotate(txt, (x, y), xytext=(0, 12), textcoords="offset points",
                            ha="center", va="bottom", fontsize=7.5, color=c, linespacing=1.0)
    ax.set_xlabel(r"base accuracy $a$")
    ax.set_ylabel(r"$p_{\mathrm{fix}}\,/\,p_{\mathrm{cor}}$ (log scale)")
    ax.grid(True, which="major", ls=":", lw=0.5, alpha=0.5)
    ax.tick_params(length=2.5, pad=1.5)


def fig_criterion():
    pts = E.criterion_points()
    fig, ax = plt.subplots(figsize=(0.8 * FS.TEXTW, 3.55))
    draw_criterion(ax, pts)
    hs = [Line2D([], [], color="black", lw=1.3, label=r"break-even $a/(1-a)$"),
          Line2D([], [], color="#555555", lw=0.8, ls=":",
                 label=r"$p_{\mathrm{fix}} = p_{\mathrm{cor}}$ (no discrimination)")]
    for p in pts:
        hs.append(Line2D([], [], color=COL[p["topo"]], marker=point_marker(p), mfc=p["mfc"],
                         mew=1.1, ms=5.5, ls="none",
                         label=f"{p['lab']}: {p['fix']}/{p['wrong']} vs {p['cor']}/{p['right']}"))
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.76))
    ax.legend(handles=[hs[0], hs[2], hs[3], hs[1], hs[4]], loc="lower center",
              bbox_to_anchor=(0.45, 1.01), ncol=2, frameon=False, fontsize=7.5,
              labelspacing=0.3, handletextpad=0.5, columnspacing=1.0, borderaxespad=0.2)
    FS.save(fig, "fig_criterion", printed=0.8)
    return pts


# ==== code energy and engineering frontier ====
def check_cfg(task, agg, extra=None):
    js = RR[task]["configs"]
    assert set(agg) == set(js), (task, set(agg) ^ set(js))
    for c, a in agg.items():
        assert abs(a["accuracy"] * 100 - js[c]["acc"]) < 1e-9, (task, c)
        assert abs(a["tokens"] - js[c]["total_tokens"]) < 1e-6, (task, c)
        if extra:
            assert abs(extra[c] - js[c]["energy_j"]) < 1e-6, (task, c)


def label_items(agg, keys):
    out = []
    for c in keys:
        t = E.topo_of(agg[c])
        col = {"SC": "#0B5A8C", "SA": "#222222"}.get(t, COL.get(t))
        out.append((FS.NAME[c], agg[c]["tokens"], agg[c]["accuracy"] * 100, col))
    return out


def fig_code_energy():
    from collections import defaultdict
    rows = E.load("humaneval_matrix_pc.jsonl")
    agg = E.analyze.aggregate(rows)
    main = ["SA", "SC_k2", "SC_k4", "SC_k8", "DEBATE_N3_R1", "GE_R2", "HIER_M3"]
    agg = {c: agg[c] for c in main}
    en = defaultdict(lambda: defaultdict(list))
    for r in rows:
        en[r["config"]][r["item_id"]].append(r["energy_j"])
    joules = {c: float(np.mean([np.mean(v) for v in en[c].values()])) for c in main}
    check_cfg("code", agg, joules)
    agg_e = {c: dict(agg[c], tokens=joules[c]) for c in main}
    fig, axes = plt.subplots(1, 2, figsize=(FS.TEXTW, 2.85), sharey=True)
    hulls = {}
    for ax, A, xl, key in ((axes[0], agg, "total tokens per query (log scale)", "tokens"),
                           (axes[1], agg_e, "gross GPU energy per query, J (log scale)", "joules")):
        hulls[key] = E.frontier_panel(ax, A, xpad=(0.72, 1.5), ms=4.3)
        ax.set_xlabel(xl)
        ax.set_ylim(40, 80)
    axes[0].set_ylabel("pass@1 (%)")
    fig.legend(handles=E.legend_handles(), loc="upper center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, 1.0), columnspacing=0.9, handletextpad=0.35,
               handlelength=1.6)
    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=0.8)
    probs = []
    for ax, A in ((axes[0], agg), (axes[1], agg_e)):
        texts = E.place_labels(ax, label_items(A, main), fontsize=7.5, pad=1.0, max_d=60,
                               cross_pen=200)
        probs += E.check_labels(ax, texts, pad=0.2)
    assert not probs, probs
    FS.save(fig, "fig_code_energy", printed=1.0)
    return hulls


def fig_proeng_pareto():
    agg, _ = E.agg_task("mmlu_pro_eng")
    check_cfg("proeng", agg)
    fig, ax = plt.subplots(figsize=(0.75 * FS.TEXTW, 3.0))
    hull = E.frontier_panel(ax, agg, xpad=(0.8, 1.6), ms=4.3)
    names = {"SA": "SA", "SC_k2": "SC $k{=}2$", "SC_k4": "SC $k{=}4$", "SC_k8": "SC $k{=}8$"}
    assert ", ".join(names[c] for c in hull) == FIGM["mixFrontierProEng"], hull
    ax.set_ylim(15.5, 42.5)
    ax.set_xlabel("total tokens per query (log scale)")
    ax.set_ylabel("accuracy (%)")
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.86))
    ax.legend(handles=E.legend_handles(), loc="lower center", ncol=3, frameon=False,
              bbox_to_anchor=(0.5, 1.0), columnspacing=1.0, handletextpad=0.4, fontsize=7.5)
    keys = ["SA", "SC_k2", "SC_k4", "SC_k8", "DEBATE_N3_R1", "HIER_M3", "GE_R2"]
    # beside the marker if free, else in the point's own column beyond its error bar (vertical
    # leader along nothing but empty space), so no label or leader crosses another bar
    items = [(txt, x, y, agg[c]["acc_lo"] * 100, agg[c]["acc_hi"] * 100, col)
             for (txt, x, y, col), c in zip(label_items(agg, keys), keys)]
    texts = E.place_in_column(ax, items, fontsize=7.5, dirs=(-1, 1), side_r=5.8)
    probs = E.check_labels(ax, texts, pad=0.2)
    assert not probs, probs
    FS.save(fig, "fig_proeng_pareto", printed=0.75)
    return hull


# ==== criterion-units table (replaces fig_criterion_units) ====
def tex_int(v):
    return f"{int(v):,}".replace(",", "{,}")


def tex_num(v):
    if math.isinf(v):
        return r"$\infty$"
    return f"{v:.1f}"


def criterion_units_table():
    pts = E.criterion_points(("ge_item", "ge_rev", "item70", "agent70", "debate_item"))
    u = {p["key"]: p for p in pts}
    assert f"{u['agent70']['ratio']:.1f}" == FIGM["unitRatioAgentSev"], u["agent70"]["ratio"]
    assert f"{u['item70']['ratio']:g}" == FIGM["unitRatioItemSev"], u["item70"]["ratio"]
    lab = {"debate_item": "3B debate, per item", "ge_item": "3B GE, per item",
           "ge_rev": "3B GE, per revision opportunity", "item70": "70B debate, per item",
           "agent70": "70B debate, per agent"}
    L = ["% AUTO-GENERATED by paper/arxiv_rev3/scripts/make_fig_eaai_adapted.py (criterion_units_table)",
         "% from results/mechanistic.jsonl and results/gsm8k_llama70b.jsonl with the unit clusters and",
         "% summarize() of scripts/make_eaai_figures.py (95% item-cluster percentile bootstrap,",
         "% B=10,000, seed 0); replaces fig_criterion_units. Rows only. Do not edit.",
         "% Suggested: \\begin{tabular}{lrrrl} \\toprule",
         "% Unit & Fixed / initially wrong & Corrupted / initially right & $p_{\\mathrm{fix}}/p_{\\mathrm{cor}}$ & 95\\% interval \\\\ \\midrule"]
    for k in ("debate_item", "ge_item", "ge_rev", "item70", "agent70"):
        p = u[k]
        lo, hi = p["ratio_ci"]
        L.append(f"{lab[k]} & {tex_int(p['fix'])} / {tex_int(p['wrong'])} & "
                 f"{tex_int(p['cor'])} / {tex_int(p['right'])} & {tex_num(p['ratio'])} & "
                 f"[{tex_num(lo)}, {tex_num(hi)}] \\\\")
    open(os.path.join(TAB, "criterion_units_table.tex"), "w").write("\n".join(L) + "\n")
    return {k: (u[k]["fix"], u[k]["wrong"], u[k]["cor"], u[k]["right"], u[k]["ratio"],
                u[k]["ratio_ci"]) for k in u}


def main():
    print("fig_flow:", fig_flow())
    print("fig_criterion:", [(p["lab"], p["fix"], p["wrong"], p["cor"], p["right"],
                              round(p["ratio"], 3)) for p in fig_criterion()])
    print("fig_code_energy hulls:", fig_code_energy())
    print("fig_proeng_pareto hull:", fig_proeng_pareto())
    print("criterion_units_table:", criterion_units_table())


if __name__ == "__main__":
    main()
