"""Shared figure style for paper/arxiv_rev3 (every rev3 figure script imports this first).

* Fonts: STIX (Times-like, matches the Times body of neurips_2026.sty), embedded as TrueType
  (pdf.fonttype = ps.fonttype = 42), so no figure carries a Type 3 font.
* One naming style for workflows everywhere (tables, text and figures): "SA", "SC k=8"
  (mathtext k), "debate", "GE", "HIER"; swept and repaired variants "debate N=2", "debate R=2",
  "GE R=1", "HIER M=2", "GES" (gated GE) and "HIERC" (chained HIER), defined in captions.
* One colour per workflow (Okabe-Ito); orange (vermillion) means debate and nothing else.
  Non-workflow semantics: FIX (sky blue) for repairs, CORR (dark red) for corruption,
  greys for kept / still wrong / frontier.
* save() writes figures/<name>.pdf at its printed size and refuses text below MIN_PT at the
  \\includegraphics width given in PRINTED (fraction of \\linewidth).
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REV3 = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(REV3))
FIGDIR = os.path.join(REV3, "figures")

TEXTW = 5.5  # neurips_2026 \textwidth in inches (397.48 pt)

COL = {"SA": "#000000", "SC": "#0072B2", "DEBATE": "#D55E00", "DU": "#D55E00",
       "GE": "#009E73", "HIER": "#CC79A7", "OTHER": "#8C8C8C"}
MARK = {"SA": "o", "SC": "o", "DEBATE": "s", "DU": "P", "GE": "^", "HIER": "D", "OTHER": "x"}
FIX = "#56B4E9"
CORR = "#B2182B"
KEEP = "#D9D9D9"
STILL = "#9E9E9E"
FRONT = "#BDBDBD"
SHADE = "#F2F2F2"

# Display names (one style everywhere).
NAME = {"SA": "SA", "SC_k2": r"SC $k$=2", "SC_k4": r"SC $k$=4", "SC_k8": r"SC $k$=8",
        "SC_k12": r"SC $k$=12", "DEBATE_N3_R1": "debate", "GE_R2": "GE", "HIER_M3": "HIER",
        "DEBATE_N2_R1": r"debate $N$=2", "DEBATE_N3_R2": r"debate $R$=2", "GE_R1": r"GE $R$=1",
        "GE_GATED": "GES", "GES_R2": "GES", "HIER_M2": r"HIER $M$=2", "HIER_CHAIN": "HIERC",
        "HIERC_M3": "HIERC", "DU": "Du et al. debate", "DU_DEBATE_N3_R2": "Du et al. debate",
        "HIERFIX": "HIERFIX"}

MIN_PT = 7.0


def setup():
    plt.rcdefaults()
    plt.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix",
        "font.size": 8.5, "axes.titlesize": 8.5, "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
        "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.7,
        "savefig.dpi": 300, "savefig.bbox": "standard", "axes.unicode_minus": True,
    })


def save(fig, name, printed=1.0):
    """Save figures/<name>.pdf; printed = \\includegraphics width as a fraction of \\linewidth."""
    scale = printed * TEXTW / fig.get_figwidth()
    small = [(t.get_text(), round(t.get_fontsize() * scale, 2)) for t in fig.findobj(matplotlib.text.Text)
             if t.get_visible() and t.get_text().strip() and t.get_fontsize() * scale < MIN_PT - 1e-6]
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
    import re
    out = {}
    for p in paths:
        for m in re.finditer(r"\\newcommand\{\\(\w+)\}\{(.*)\}", open(os.path.join(REV3, p)).read()):
            out[m.group(1)] = m.group(2)
    return out
