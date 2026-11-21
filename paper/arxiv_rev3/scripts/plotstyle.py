"""Shared figure style for paper/arxiv_rev3 (every figure script imports this module first).

One module, one look (a copy is in the release repository as soc/plotstyle.py):
* Fonts: STIX (Times-like, the body font of neurips_2026.sty is Times), 8 pt for every label,
  tick label, legend and annotation; embedded as TrueType (pdf.fonttype = ps.fonttype = 42), so
  no figure carries a Type 3 font.
* Sizes: every figure is drawn at its final printed size and included in LaTeX without width=
  scaling. FULL_W is \\textwidth of neurips_2026.sty (397.48 pt = 5.5 in, measured with
  \\the\\textwidth); HALF_W fits a 0.49\\linewidth minipage.
* Colors, defined once (Okabe-Ito): SA black, SC blue, debate orange, GE green, HIER pink; light
  grey BAND (#DDDDDD) for bands and intervals only. FIX and CORR are the two flow colors of the
  correction/corruption figure (not workflow colors).
* Marks: one marker size MS, line width LW = 1.2, error bars thin (ELW) in the point's color with
  capsize 2 (errorbar()); light grid; legend frame off; constrained layout; no bbox cropping
  (save() writes the figure at exactly its figsize).
* Names: one naming style for workflows everywhere (tables, text and figures): "SA", "SC k=8",
  "debate", "GE", "HIER"; swept and repaired variants "debate N=2", "debate R=2", "GE R=1",
  "HIER M=2", "GES" (gated GE) and "HIERC" (chained HIER). Points on a sampling curve are
  labeled "SA", "k=2", "k=4", ... (klabel()).
"""
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def _find_root(d):
    """Repository root: the nearest ancestor with harness/ and paper/arxiv_rev3/ (the same file
    works from paper/arxiv_rev3/scripts/ and from the release's soc/)."""
    while True:
        if os.path.isdir(os.path.join(d, "harness")) and os.path.isdir(os.path.join(d, "paper", "arxiv_rev3")):
            return d
        if os.path.dirname(d) == d:
            raise RuntimeError("repository root not found above " + HERE)
        d = os.path.dirname(d)


ROOT = _find_root(HERE)
REV3 = os.path.join(ROOT, "paper", "arxiv_rev3")
FIGDIR = os.path.join(REV3, "figures")

FULL_W = 5.5    # \textwidth (397.48 pt)
HALF_W = 2.69   # fits \begin{minipage}{0.49\linewidth} (2.695 in)
FONT_PT = 8.0

COL = {"SA": "#000000", "SC": "#0072B2", "DEBATE": "#D55E00", "DU": "#D55E00",
       "GE": "#009E73", "HIER": "#CC79A7", "OTHER": "#8C8C8C"}
MARK = {"SA": "o", "SC": "o", "DEBATE": "s", "DU": "P", "GE": "^", "HIER": "D", "OTHER": "x"}
BAND = "#DDDDDD"     # bands and intervals only
GRID = "#E6E6E6"
TEXT = "#222222"     # annotation text
MUTED = "#6E6E6E"    # secondary text (region notes, panel notes)
FIX = "#56B4E9"      # flow figure: fixed
CORR = "#B2182B"     # flow figure: corrupted
STILL = "#9E9E9E"    # flow figure: still wrong

MS = 4.5             # marker size (points)
LW = 1.2             # line width
ELW = 0.8            # error-bar line width
CAPSIZE = 2.0
MEW = 0.9            # marker edge width

NAME = {"SA": "SA", "SC_k2": r"SC $k$=2", "SC_k4": r"SC $k$=4", "SC_k8": r"SC $k$=8",
        "SC_k12": r"SC $k$=12", "DEBATE_N3_R1": "debate", "GE_R2": "GE", "HIER_M3": "HIER",
        "DEBATE_N2_R1": r"debate $N$=2", "DEBATE_N3_R2": r"debate $R$=2", "GE_R1": r"GE $R$=1",
        "GE_GATED": "GES", "GES_R2": "GES", "HIER_M2": r"HIER $M$=2", "HIER_CHAIN": "HIERC",
        "HIERC_M3": "HIERC", "DU": "Du et al. debate", "DU_DEBATE_N3_R2": "Du et al. debate",
        "HIERFIX": "HIERFIX"}


def klabel(cfg):
    """Label of a point on a sampling curve: 'SA' or 'k=2' style."""
    return "SA" if cfg == "SA" else "$k$=" + cfg.split("_k")[1]


def setup():
    plt.rcdefaults()
    plt.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix",
        "font.size": FONT_PT, "axes.titlesize": FONT_PT, "axes.labelsize": FONT_PT,
        "xtick.labelsize": FONT_PT, "ytick.labelsize": FONT_PT, "legend.fontsize": FONT_PT,
        "legend.title_fontsize": FONT_PT, "figure.titlesize": FONT_PT,
        "pdf.fonttype": 42, "ps.fonttype": 42, "axes.unicode_minus": True,
        "lines.linewidth": LW, "lines.markersize": MS, "lines.markeredgewidth": MEW,
        "errorbar.capsize": CAPSIZE,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5, "xtick.major.pad": 2.0, "ytick.major.pad": 2.0,
        "axes.labelpad": 2.0, "axes.titlepad": 3.0,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.5,
        "grid.linestyle": "-",
        "legend.frameon": False, "legend.handletextpad": 0.4, "legend.columnspacing": 1.0,
        "legend.borderaxespad": 0.2, "legend.labelspacing": 0.3, "legend.handlelength": 1.6,
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.02, "figure.constrained_layout.w_pad": 0.02,
        "figure.constrained_layout.hspace": 0.03, "figure.constrained_layout.wspace": 0.03,
        "savefig.dpi": 300, "savefig.bbox": None, "savefig.pad_inches": 0.0,
    })


def errorbar(ax, x, y, lo=None, hi=None, color="#000000", marker="o", mfc=None, ms=MS,
             xlo=None, xhi=None, zorder=3, label=None, clip_on=True):
    """One point with a thin error bar in the point's color (capsize CAPSIZE). lo/hi and
    xlo/xhi are interval ends in data coordinates (None: no bar on that axis)."""
    yerr = None if lo is None else [[y - lo], [hi - y]]
    xerr = None if xlo is None else [[x - xlo], [xhi - x]]
    return ax.errorbar([x], [y], yerr=yerr, xerr=xerr, fmt=marker, color=color, ecolor=color,
                       mfc=color if mfc is None else mfc, mec=color, ms=ms, mew=MEW,
                       elinewidth=ELW, capsize=CAPSIZE, capthick=ELW, zorder=zorder,
                       label=label, clip_on=clip_on)


def save(fig, name):
    """Write figures/<name>.pdf at exactly its figsize (no cropping). Refuses a figure wider
    than FULL_W or any visible text below FONT_PT."""
    w = fig.get_figwidth()
    assert w <= FULL_W + 1e-9, (name, w)
    small = [(t.get_text(), t.get_fontsize()) for t in fig.findobj(matplotlib.text.Text)
             if t.get_visible() and t.get_text().strip() and t.get_fontsize() < FONT_PT - 1e-6]
    assert not small, (name, small[:5])
    os.makedirs(FIGDIR, exist_ok=True)
    fig.savefig(os.path.join(FIGDIR, f"{name}.pdf"))
    plt.close(fig)


def tint(hexcol, f):
    h = hexcol.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return "#%02x%02x%02x" % tuple(int(c + (255 - c) * f) for c in (r, g, b))


def fmt_int(x):
    return f"{int(round(x)):,}"


def minus(s):
    return s.replace("-", "\u2212")


def macros(*paths):
    """\\newcommand values of the rev3 macro files (path relative to arxiv_rev3)."""
    out = {}
    for p in paths:
        for m in re.finditer(r"\\newcommand\{\\(\w+)\}\{(.*)\}", open(os.path.join(REV3, p)).read()):
            out[m.group(1)] = m.group(2)
    return out
