"""rev3 Figure 1: coordination minus self-consistency, every task (forest plot, full width).

Adapted from paper/arxiv_rev2/scripts/make_overview_figure.py, panel (b) only. Sources (nothing
is typed by hand):
  * paper/arxiv_rev3/tables/equivalence.json, rows["comparators"]["main"] (SC k=8, asserted):
    est and ci95 (seed-aware hierarchical bootstrap, unadjusted) for GSM8K, GSM8K-200 (debate
    only), MMLU, MMLU-Pro engineering and HumanEval. The 60-item MATH rows are not used.
  * results/extra/math_interaction_analysis.json, within_stratum.<W>.easy_minus_SC_k8 (diff,
    ci95): MATH levels 1 to 3, 238 items, 3 seeds; n = easy.n_items.
  * paper/arxiv_rev3/tables/hard_math.json tests_unrounded: MATH levels 4 to 5 (262 items),
    debate vs SC k=8, GE vs SC k=4, HIER vs SC k=4, Du et al. debate vs SC k=8; checked against
    results/extra/hard_math_analysis.json (primary_total_tokens and du.minus_SC_k8, rounded to
    2 decimals) and n = n_items_complete.
  * Single-call accuracy in each task label: macros \\EqAccSaEng, \\EqAccSaGsm
    (equivalence_macros.tex), \\accSAMmlu (crosstask_macros.tex), \\accSACode
    (humaneval_macros.tex), \\MeSaAcc (math_easy_macros.tex, if present; else
    math_interaction_analysis.json easy.accuracy.SA to 1 decimal); GSM8K-200 from
    tables/review_revision.json tasks.gsm8k200.configs.SA.acc. Each is asserted equal (to the
    printed decimal) to the SA accuracy recomputed from the per-record logs with
    harness.stats.per_item_seed_correct; n is the number of items in those logs.
Asserted: comparator configs, JSON-vs-analysis agreement, label accuracies, and that no
interval lies wholly above zero (the caption says so).
rev3 rerun fold-in: GSM8K, MMLU, MMLU-Pro engineering and HumanEval rows now come from
results/extra/rev3_reruns_analysis.json (GSM8K 500 x 3; HIER = HIERFIX on every task; HumanEval
R3 with the entry-block extractor); see the comment in collect().
Output: paper/arxiv_rev3/figures/fig_forest.pdf (printed=1.0).
"""
import json
import os
import sys

import numpy as np

import figstyle as FS

FS.setup()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, FS.ROOT)
from harness import stats as S  # noqa: E402

TAB = os.path.join(FS.REV3, "tables")
EXTRA = os.path.join(FS.ROOT, "results", "extra")
FILES = {"gsm8k": ["gsm8k_matrix.jsonl"], "gsm8k200": ["gsm8k_scale200.jsonl"],
         "mmlu": ["mmlu_matrix.jsonl"], "proeng": ["mmlu_pro_eng_matrix.jsonl"],
         "code": ["humaneval_matrix_pc.jsonl"]}
ORDER = ["DEBATE_N3_R1", "DU", "GE_R2", "HIER_M3"]
CKEY = {"DEBATE_N3_R1": "DEBATE", "GE_R2": "GE", "HIER_M3": "HIER", "DU": "DU"}


def records(task):
    rs = []
    for f in FILES[task]:
        rs += [json.loads(l) for l in open(os.path.join(FS.ROOT, "results", f)) if l.strip()]
    return rs


def sa_from_records(task):
    cor = S.per_item_seed_correct(records(task))["SA"]
    return 100 * float(np.mean([v.mean() for v in cor.values()])), len(cor)


def collect():
    eq = json.load(open(os.path.join(TAB, "equivalence.json")))
    hm = json.load(open(os.path.join(TAB, "hard_math.json")))
    mi = json.load(open(os.path.join(EXTRA, "math_interaction_analysis.json")))
    ha = json.load(open(os.path.join(EXTRA, "hard_math_analysis.json")))
    rr = json.load(open(os.path.join(TAB, "review_revision.json")))
    mac = FS.macros("equivalence_macros.tex", "crosstask_macros.tex", "humaneval_macros.tex")
    me_path = os.path.join(FS.REV3, "math_easy_macros.tex")
    if os.path.exists(me_path):
        sa_math = FS.macros("math_easy_macros.tex")["MeSaAcc"]
    else:
        sa_math = f"{mi['easy']['accuracy']['SA']:.1f}"
    sa_label = {"proeng": mac["EqAccSaEng"], "mmlu": mac["accSAMmlu"], "code": mac["accSACode"],
                "gsm8k": mac["EqAccSaGsm"],
                "gsm8k200": f"{rr['tasks']['gsm8k200']['configs']['SA']['acc']:.1f}"}
    groups = []   # (label, sa_float, rows[(coord, est, lo, hi)])
    # rev3 rerun fold-in: GSM8K (500 x 3), MMLU, MMLU-Pro engineering and HumanEval (R3, v2
    # extractor) from results/extra/rev3_reruns_analysis.json, HIER = HIERFIX; the per-config SA
    # accuracy is asserted against rev3_reruns_macros.tex. GSM8K-200 (debate only) stays from
    # tables/equivalence.json.
    ra = json.load(open(os.path.join(EXTRA, "rev3_reruns_analysis.json")))["tasks"]
    rmac = FS.macros("rev3_reruns_macros.tex")
    names = {"gsm8k": ("GSM8K", "gsm8k", "Gsm"), "mmlu": ("MMLU", "mmlu", "Mmlu"),
             "proeng": ("MMLU-Pro eng.", "proeng", "Eng"), "code": ("HumanEval", "code_v2", "Code")}
    for key, (nm, rk, tn) in names.items():
        T = ra[rk]
        rows = []
        for w, lab in (("DEBATE_N3_R1", "DEBATE_N3_R1"), ("GE_R2", "GE_R2"), ("HIERFIX_M3", "HIER_M3")):
            t = T["tests"][f"{w}|SC_k8"]
            rows.append((lab, t["diff"], t["ci95"][0], t["ci95"][1]))
        sa = T["configs"]["SA"]["acc"]
        assert f"{sa:.1f}" == rmac[f"Rr{tn}SaAcc"], (key, sa)
        groups.append((f"{nm}\nSA {sa:.1f}%, $n$={T['n_items']}", float(f"{sa:.1f}"), rows))
    for key, nm in (("gsm8k200", "GSM8K-200"),):
        rows = []
        for r in eq["rows"]:
            if r["task"] == key:
                m = r["comparators"]["main"]
                assert m["config"] == "SC_k8", (key, m["config"])
                rows.append((r["coord"], m["est"], m["ci95"][0], m["ci95"][1]))
        assert rows, key
        acc, n = sa_from_records(key)
        assert f"{acc:.1f}" == sa_label[key], (key, acc, sa_label[key])
        groups.append((f"{nm}\nSA {sa_label[key]}%, $n$={n}", float(sa_label[key]), rows))
    # MATH levels 1 to 3: the 238-item, 3-seed run.
    rows = []
    for w in ("DEBATE_N3_R1", "GE_R2", "HIER_M3"):
        t = mi["within_stratum"][w]["easy_minus_SC_k8"]
        rows.append((w, t["diff"], t["ci95"][0], t["ci95"][1]))
    assert f"{mi['easy']['accuracy']['SA']:.1f}" == sa_math, sa_math
    groups.append((f"MATH L1-3\nSA {sa_math}%, $n$={mi['easy']['n_items']}", float(sa_math), rows))
    groups.sort(key=lambda g: g[1])          # lowest single-call accuracy at the top
    # MATH levels 4 to 5, cost-matched comparator.
    want = [("DEBATE_N3_R1", "SC_k8"), ("GE_R2", "SC_k4"), ("HIER_M3", "SC_k4"), ("DU", "SC_k8")]
    hard = []
    for a, b in want:
        t = [t for t in hm["tests_unrounded"] if (t["a"], t["b"]) == (a, b)]
        assert len(t) == 1, (a, b)
        t = t[0]
        ref = ha["du"]["minus_SC_k8"] if a == "DU" else ha["primary_total_tokens"][a]
        if a != "DU":
            assert ref["comparator"] == b, (a, ref["comparator"], b)
        assert round(t["diff"], 2) == ref["diff"], (a, t["diff"], ref["diff"])
        assert [round(v, 2) for v in t["ci95"]] == ref["ci95"], (a, t["ci95"], ref["ci95"])
        hard.append((a, b, t["diff"], t["ci95"][0], t["ci95"][1]))
    n_hard = ha["n_items_complete"]
    assert n_hard == ha["n_items_total"]
    return groups, hard, n_hard


def main():
    groups, hard, n_hard = collect()
    allrows = [r for g in groups for r in g[2]] + [(h[0], h[2], h[3], h[4]) for h in hard]
    assert not any(lo > 0 for _, _, lo, _ in allrows), "an interval lies wholly above zero"

    fig = plt.figure(figsize=(FS.TEXTW, 3.2))
    ax = fig.add_axes([0.255, 0.115, 0.735, 0.80])
    step, ggap = 1.0, 0.75
    y = 0.0
    ticks, tlabels, tcol = [], [], []
    for lab, _, rows in groups:
        rows = sorted(rows, key=lambda r: ORDER.index(r[0]))
        ys = []
        for c, est, lo, hi in rows:
            col = FS.COL[CKEY[c]]
            ax.plot([lo, hi], [y, y], color=col, lw=1.1, solid_capstyle="butt", zorder=3)
            ax.plot(est, y, FS.MARK[CKEY[c]], color=col, ms=4.0, mew=0.9, zorder=4)
            ys.append(y)
            y -= step
        ticks.append(float(np.mean(ys)))
        tlabels.append(lab)
        tcol.append("#222222")
        y -= ggap
    # divider and the MATH L4-5 block
    ydiv = y + ggap / 2 + 0.05
    y -= 0.55
    yhead = y + 0.05
    y -= 0.75
    hard_ticks = []
    for a, b, est, lo, hi in hard:
        col = FS.COL[CKEY[a]]
        ax.plot([lo, hi], [y, y], color=col, lw=1.1, solid_capstyle="butt", zorder=3)
        ax.plot(est, y, FS.MARK[CKEY[a]], color=col, ms=4.0, mew=0.9, zorder=4)
        hard_ticks.append((y, f"{FS.NAME[a]} vs {FS.NAME[b]}"))
        y -= step
    ymin = y + step - 0.7
    xs = [v for r in allrows for v in r[1:]]
    xlo, xhi = np.floor(min(xs) / 5) * 5 - 1, max(8.0, np.ceil(max(xs) / 5) * 5 + 1)
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ymin, 0.85)
    ax.axvspan(0, xhi, color=FS.tint(FS.FIX, 0.82), lw=0, zorder=0)
    ax.axvline(0, color="#222222", lw=0.8, zorder=2)
    ax.axhline(ydiv, color="#888888", lw=0.6, zorder=2)
    ax.text(xhi - 0.4, 0.55, "coordination\nbetter", ha="right", va="top", fontsize=7.5,
            color="#1F5F86", linespacing=1.0)
    ax.text(xlo + 0.4, 0.55, "minus SC $k$=8", ha="left", va="top", fontsize=7.5,
            color="#222222", style="italic")
    ax.text(xlo + 0.4, yhead, f"MATH L4-5, cost-matched SC, $n$={n_hard}", ha="left",
            va="center", fontsize=7.5, color="#222222", style="italic")
    yt = ticks + [t[0] for t in hard_ticks]
    yl = tlabels + [t[1] for t in hard_ticks]
    ax.set_yticks(yt)
    ax.set_yticklabels(yl, fontsize=7.5, linespacing=1.0)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.grid(True, axis="x", ls=":", lw=0.5, alpha=0.6, zorder=1)
    ax.set_xlabel("coordination minus SC (accuracy points, 95% interval)", fontsize=8)
    handles = [Line2D([], [], color=FS.COL[CKEY[c]], marker=FS.MARK[CKEY[c]], ls="-", lw=1.1,
                      ms=4.0, label=FS.NAME[c]) for c in ORDER]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.62, 1.0), ncol=4,
               frameon=False, handlelength=1.6, columnspacing=1.4, fontsize=7.5,
               borderaxespad=0.2)
    FS.save(fig, "fig_forest", printed=1.0)
    print("fig_forest:", len(allrows), "intervals; groups top to bottom:",
          [g[0].split("\n")[0] for g in groups], "+ MATH L4-5 x", len(hard))


if __name__ == "__main__":
    main()
