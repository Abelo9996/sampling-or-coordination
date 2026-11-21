#!/usr/bin/env python3
"""rev3 figure: run-to-run drift of identical configurations against the width of a within-run
95% interval.

Output: paper/arxiv_rev3/figures/fig_drift.pdf (design width 2.7 in, printed=0.49).
Encoding: x = accuracy minus the matrix accuracy of the same configuration on the same items
(points). Grey bar = the matrix accuracy's 95% item-bootstrap interval as printed in the paper,
shifted by the same matrix accuracy. Filled marker at 0 = matrix (July Apple M4 run); open marker
= rerun; a line joins them. Absolute accuracies are printed at the right of each row.

Rows and sources (all values are ones the paper prints; nothing typed by hand):
  GSM8K, 80 matrix items (SA, SC k=8, debate, GE, HIER):
    matrix accuracy   audit_macros.tex \\AuDriftMat{Sa,Sc,Deb,Ge,Hier}; tables/audit.json
                      drift80.<cfg>.matrix80 (unrounded, plotted); audit_drift_table.tex col 1
    rerun accuracy    audit_macros.tex \\AuDriftRe{...}; audit.json drift80.<cfg>.rerun80
                      (RTX 4060 rerun, three seeds, same 80 items, plotted); drift table col 2
    95% interval      tables/poster_main_table.tex (Table 1), GSM8K column, rows SA, SC k=8,
                      Debate, GE, HIER, e.g. "68.8 {\\tiny[59,78]}" (integer bounds as printed)
  MMLU-Pro engineering, 150 items, debate:
    runs              criterion_predictive_macros.tex \\cpEngDebMxSone (matrix seed 1),
                      \\cpEngDebMxStwo (matrix seed 2), \\cpEngDebTrFinal (Apple M4 trace run)
    reference         mmlu_pro_eng_macros.tex \\accDebateProEng (matrix accuracy, both seeds)
    95% interval      mmlu_pro_eng_macros.tex \\accLoDebateProEng, \\accHiDebateProEng
                      (also tables/mmlu_pro_eng_table.tex "25.7 {\\tiny[20,31]}")

Assertions: audit.json matrix80 / rerun80 round (1 dp) to the \\AuDriftMat / \\AuDriftRe macros and
equal columns 1 and 2 of audit_drift_table.tex; the Table 1 GSM8K accuracy of each row equals
\\AuDriftMat<cfg>; the mean of \\cpEngDebMxSone and \\cpEngDebMxStwo rounds to \\accDebateProEng;
\\cpEngDebTrFinal minus \\cpEngDebMxSone equals \\cpEngDebRunSpread; the engineering table fragment
shows the same accuracy and interval (rounded to integers) as the macros.

Run: /usr/bin/python3 paper/arxiv_rev3/scripts/fig_drift.py
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import figstyle as fs  # noqa: E402

ROWS = [("SA", "Sa", "SA", "SA"), ("SC_k8", "Sc", "SC $k{=}8$", "SC"),
        ("DEBATE_N3_R1", "Deb", "Debate", "DEBATE"), ("GE_R2", "Ge", "GE", "GE"),
        ("HIER_M3", "Hier", "HIER", "HIER")]


def load():
    m = fs.macros("audit_macros.tex", "criterion_predictive_macros.tex", "mmlu_pro_eng_macros.tex")
    A = json.load(open(os.path.join(fs.REV3, "tables/audit.json")))["drift80"]
    drift_tab = [l.split("&") for l in open(os.path.join(fs.REV3, "tables/audit_drift_table.tex"))
                 if "&" in l]
    t1 = [l.split("&") for l in open(os.path.join(fs.REV3, "tables/poster_main_table.tex")) if "&" in l]
    out = []
    for cfg, mac, tabname, col in ROWS:
        mat, re_ = A[cfg]["matrix80"], A[cfg]["rerun80"]
        assert f"{mat:.1f}" == m[f"AuDriftMat{mac}"], (cfg, mat)
        assert f"{re_:.1f}" == m[f"AuDriftRe{mac}"], (cfg, re_)
        drow = [r for r in drift_tab if r[0].strip() == tabname][0]
        assert drow[1].strip() == m[f"AuDriftMat{mac}"] and drow[2].strip() == m[f"AuDriftRe{mac}"], drow
        trow = [r for r in t1 if r[0].strip() == tabname][0]
        mm = re.fullmatch(r"([\d.]+) \{\\tiny\[(\d+),(\d+)\]\}", trow[1].strip())
        assert mm and mm.group(1) == m[f"AuDriftMat{mac}"], (cfg, trow[1])
        lo, hi = float(mm.group(2)), float(mm.group(3))
        assert lo < mat < hi
        out.append({"label": fs.NAME[cfg], "col": fs.COL[col], "ref": mat, "lo": lo, "hi": hi,
                    "base": [mat], "other": [re_],
                    "text": f"{m[f'AuDriftMat{mac}']} → {m[f'AuDriftRe{mac}']}"})
    s1, s2, tr = (float(m[k]) for k in ("cpEngDebMxSone", "cpEngDebMxStwo", "cpEngDebTrFinal"))
    ref = m["accDebateProEng"]
    assert f"{(s1 + s2) / 2 + 1e-9:.1f}" == ref, (s1, s2, ref)
    assert f"{tr - s1:.1f}" == m["cpEngDebRunSpread"], (tr, s1)
    lo, hi = float(m["accLoDebateProEng"]), float(m["accHiDebateProEng"])
    tab = open(os.path.join(fs.REV3, "tables/mmlu_pro_eng_table.tex")).read()
    assert f"Debate & {ref} {{\\tiny[{lo:.0f},{hi:.0f}]}}" in tab, (ref, lo, hi)
    out.append({"label": "debate", "col": fs.COL["DEBATE"], "ref": float(ref), "lo": lo, "hi": hi,
                "base": [s1, s2], "other": [tr],
                "text": f"{m['cpEngDebMxSone']}, {m['cpEngDebMxStwo']} → {m['cpEngDebTrFinal']}"})
    return out


def main():
    R = load()
    fs.setup()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(2.7, 2.35))
    ys = [6.0, 5.0, 4.0, 3.0, 2.0, 0.0]
    for y, r in zip(ys, R):
        ax.plot([r["lo"] - r["ref"], r["hi"] - r["ref"]], [y, y], "-", color=fs.KEEP, lw=5.5,
                solid_capstyle="butt", zorder=1)
        xs = [v - r["ref"] for v in r["base"] + r["other"]]
        ax.plot([min(xs + [0]), max(xs + [0])], [y, y], "-", color=r["col"], lw=1.0, zorder=2)
        if len(r["base"]) == 1:
            ax.plot(0, y, "|", color="#555555", ms=5, mew=0.8, zorder=2)
        for v in r["base"]:
            ax.plot(v - r["ref"], y, "o", color=r["col"], ms=4.2, mew=0.8, zorder=3)
        for v in r["other"]:
            ax.plot(v - r["ref"], y, "o", color=r["col"], mfc="white", ms=4.2, mew=0.9, zorder=3)
        ax.text(1.02, y, r["text"], transform=ax.get_yaxis_transform(), ha="left", va="center",
                fontsize=7.5, color="#333333")
    if len(R[-1]["base"]) > 1:  # reference tick for the pooled matrix accuracy of the eng row
        ax.plot(0, ys[-1], "|", color="#555555", ms=5, mew=0.8, zorder=2)
    ax.axvline(0, color="#999999", lw=0.5, ls=":", zorder=0)
    ax.axhline(1.0, color="#BBBBBB", lw=0.5, zorder=0)
    ax.set_yticks(ys)
    ax.set_yticklabels([r["label"] for r in R])
    ax.tick_params(axis="y", length=0)
    from matplotlib.transforms import blended_transform_factory
    tr = blended_transform_factory(fig.transFigure, ax.transData)
    ax.text(0.012, 6.75, "GSM8K, 80 items: matrix vs rerun", transform=tr, ha="left", va="center",
            fontsize=7.5, style="italic", color="#444444")
    ax.text(0.012, 0.75, "MMLU-Pro eng., 150 items: matrix seeds vs trace run", transform=tr,
            ha="left", va="center", fontsize=7.5, style="italic", color="#444444")
    from matplotlib.lines import Line2D
    hand = [Line2D([], [], marker="o", ls="none", color="#555555", ms=4.2),
            Line2D([], [], marker="o", ls="none", color="#555555", mfc="white", ms=4.2),
            Line2D([], [], color=fs.KEEP, lw=5.5, solid_capstyle="butt")]
    fig.legend(hand, ["matrix", "rerun", "matrix 95% interval"], loc="upper center", ncol=3,
               frameon=False, bbox_to_anchor=(0.5, 1.0), handlelength=1.4, columnspacing=1.0,
               handletextpad=0.4)
    ax.set_ylim(-0.6, 7.2)
    ax.set_xlim(-16, 12)
    ax.set_xticks([-15, -10, -5, 0, 5, 10])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: fs.minus(f"{x:+.0f}") if x else "0"))
    ax.set_xlabel("accuracy minus matrix accuracy (points)")
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.subplots_adjust(left=0.155, right=0.655, bottom=0.17, top=0.905)
    fs.save(fig, "fig_drift", printed=0.49)
    for r in R:
        print(r["label"], r["ref"], [round(v - r["ref"], 2) for v in r["other"]], (r["lo"], r["hi"]))


if __name__ == "__main__":
    main()
