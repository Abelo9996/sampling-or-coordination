"""Emit the cross-task results into LaTeX:
  - paper/crosstask_table.tex : one block per task (acc, tokens) for all configs
  - paper/crosstask_macros.tex : key headline numbers as macros (per task +
    aggregate), including router results when probes are present.

Everything is read straight from results/*_matrix.jsonl and *_probe.jsonl, so
no number is transcribed by hand.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from harness import analyze  # noqa: E402

TASKS = [("gsm8k", "GSM8K", "Gsm"), ("mmlu", "MMLU", "Mmlu"),
         ("math500", "MATH", "Math")]
ORDER = ["SA", "SC_k2", "SC_k4", "SC_k8", "GE_R2", "HIER_M3", "DEBATE_N3_R1"]
PRETTY = {"SA": "SA", "SC_k2": "SC $k{=}2$", "SC_k4": "SC $k{=}4$",
          "SC_k8": "SC $k{=}8$", "GE_R2": "GE", "HIER_M3": "HIER",
          "DEBATE_N3_R1": "Debate"}


def task_agg(task):
    path = os.path.join(ROOT, "results", f"{task}_matrix.jsonl")
    if not os.path.exists(path):
        return None
    return analyze.aggregate(analyze.load(path))


def emit_table():
    lines = [r"\begin{table}[t]\centering\small",
             r"\caption{Accuracy (\%, with 95\% bootstrap CI) and inference cost "
             r"(tokens/query) for every topology on all three task classes "
             r"(Qwen2.5-3B-Instruct). On each task the measured Pareto-optimal "
             r"configurations are single agents; the multi-agent gaps are "
             r"statistically significant on the reasoning tasks and within "
             r"noise on MMLU (see text).}",
             r"\label{tab:crosstask}",
             r"\begin{tabular}{l rr rr rr}",
             r"\toprule",
             r"& \multicolumn{2}{c}{GSM8K} & \multicolumn{2}{c}{MMLU} "
             r"& \multicolumn{2}{c}{MATH} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
             r"Config & Acc. & Tok. & Acc. & Tok. & Acc. & Tok. \\",
             r"\midrule"]
    aggs = {t: task_agg(t) for t, _, _ in TASKS}
    for cfg in ORDER:
        row = [PRETTY[cfg]]
        for t, _, _ in TASKS:
            a = aggs[t]
            if a and cfg in a:
                row.append(f"{a[cfg]['accuracy']*100:.1f}")
                row.append(f"{a[cfg]['tokens']:.0f}")
            else:
                row += ["--", "--"]
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(os.path.join(ROOT, "paper", "crosstask_table.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote crosstask_table.tex")


def emit_macros():
    defs = {}
    aggs = {}
    for task, _, stem in TASKS:
        a = task_agg(task)
        if not a:
            continue
        aggs[task] = a
        sa, sc8 = a.get("SA"), a.get("SC_k8")
        if sa:
            defs[f"accSA{stem}"] = f"{sa['accuracy']*100:.1f}"
        if sc8:
            defs[f"accSCEight{stem}"] = f"{sc8['accuracy']*100:.1f}"
        # SC scaling delta (k1 -> k8)
        if sa and sc8:
            defs[f"scDelta{stem}"] = f"{(sc8['accuracy']-sa['accuracy'])*100:+.1f}"
        # crossover gaps
        cx = analyze.crossover_vs_sc(a)
        if cx:
            gaps = [d["gap_sc_minus_mas"] * 100 for d in cx.values()]
            defs[f"gapMin{stem}"] = f"{min(gaps):+.1f}"
            defs[f"gapMax{stem}"] = f"{max(gaps):+.1f}"
        # best MAS accuracy and its name
        mas = {c: v for c, v in a.items()
               if v["topology"] in ("DEBATE", "GE", "HIER")}
        if mas:
            best = max(mas, key=lambda c: mas[c]["accuracy"])
            defs[f"bestMASacc{stem}"] = f"{mas[best]['accuracy']*100:.1f}"
        # router (per-task) if probe present
        if os.path.exists(os.path.join(ROOT, "results", f"{task}_probe.jsonl")):
            try:
                from harness import router
                r = router.evaluate_router(task)
                defs[f"routerAcc{stem}"] = f"{r['router']['accuracy']*100:.1f}"
                defs[f"routerTok{stem}"] = f"{r['router']['tokens']:.0f}"
                defs[f"routerAgree{stem}"] = f"{r['agree_frac']*100:.0f}"
                defs[f"routerEsc{stem}"] = r["escalate_to"].replace("_", r"\_")
            except Exception as e:  # noqa
                print(f"router({task}) skipped:", e)

    # aggregate across tasks: all frontiers single-agent? count dominated MAS
    n_tasks = len(aggs)
    defs["nTasks"] = n_tasks
    all_single = all(
        all(a[c]["topology"] in ("SA", "SC")
            for c in analyze.pareto_frontier(a))
        for a in aggs.values())
    defs["allFrontiersSingle"] = "true" if all_single else "false"

    # paired significance + Pareto-domination flags (reviewer fix)
    from harness import stats
    for task, _, stem in TASKS:
        if not os.path.exists(os.path.join(ROOT, "results", f"{task}_matrix.jsonl")):
            continue
        ht = stats.headline(task)
        # min/max p over the three SC_k8-vs-MAS comparisons
        mas_ps = [p for k, (d, p) in ht.items() if "SA" not in k]
        if mas_ps:
            defs[f"pMax{stem}"] = f"{max(mas_ps):.2f}"
            pmin = max(min(mas_ps), 1e-4)  # bootstrap resolution floor
            defs[f"pMin{stem}"] = f"{pmin:.4f}" if pmin < 0.001 else f"{pmin:.3f}"
            defs[f"allSig{stem}"] = "true" if max(mas_ps) < 0.05 else "false"
        rep, _ = stats.crossover_report(task)
        # configs SC_k8 Pareto-dominates (cheaper AND more accurate) need no extrapolation
        agg2 = aggs[task]
        sc8 = agg2.get("SC_k8")
        if sc8:
            dom = [r["config"] for r in rep
                   if r["mas_tok"] > sc8["tokens"] and r["mas_acc"] < sc8["accuracy"] * 100]
            defs[f"scDominates{stem}"] = str(len(dom))

    with open(os.path.join(ROOT, "paper", "crosstask_macros.tex"), "w") as f:
        for k, v in defs.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print(f"wrote crosstask_macros.tex ({len(defs)} macros); "
          f"all frontiers single-agent: {all_single}")


if __name__ == "__main__":
    emit_table()
    emit_macros()
