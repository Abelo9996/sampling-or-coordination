"""Generate paper/results_macros.tex with every data-derived number as a LaTeX
macro, straight from the logs, so the prose never hard-codes a transcribed
value. Run after the matrix (and, for router macros, after the probe).
  .venv/bin/python scripts/emit_macros.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from harness import analyze  # noqa: E402

# config label -> camelCase macro stem
STEM = {"SA": "Sa", "SC_k2": "ScTwo", "SC_k4": "ScFour", "SC_k8": "ScEight",
        "DEBATE_N3_R1": "Debate", "GE_R2": "Ge", "HIER_M3": "Hier"}


def main():
    mpath = os.path.join(ROOT, "results", "gsm8k_matrix.jsonl")
    rows = analyze.load(mpath)
    agg = analyze.aggregate(rows)
    cx = analyze.crossover_vs_sc(agg)
    defs = {}

    n_items = max((a["n_items"] for a in agg.values()), default=0)
    defs["nItems"] = n_items

    for cfg, stem in STEM.items():
        if cfg not in agg:
            continue
        a = agg[cfg]
        defs[f"acc{stem}"] = f"{a['accuracy']*100:.1f}"
        defs[f"accLo{stem}"] = f"{a['acc_lo']*100:.1f}"
        defs[f"accHi{stem}"] = f"{a['acc_hi']*100:.1f}"
        defs[f"tok{stem}"] = f"{a['tokens']:.0f}"

    # crossover gaps (single agent minus MAS at equal budget). Only emit for
    # configs with a letter-only STEM (LaTeX macro names cannot contain digits or
    # underscores); the swept Tier-B configs are summarized in the text instead.
    for cfg, d in cx.items():
        if cfg not in STEM:
            continue
        defs[f"gap{STEM[cfg]}"] = f"{d['gap_sc_minus_mas']*100:+.1f}"
    if cx:
        defs["gapMax"] = f"{max(d['gap_sc_minus_mas'] for d in cx.values())*100:.1f}"
        defs["gapMin"] = f"{min(d['gap_sc_minus_mas'] for d in cx.values())*100:.1f}"

    front = analyze.pareto_frontier(agg)
    defs["paretoFront"] = " $>$ ".join(s.replace("_", r"\_") for s in front)
    defs["nFront"] = len(front)

    # router (if probe present)
    defs.setdefault("routerAcc", "??")
    defs.setdefault("routerTok", "??")
    defs.setdefault("routerEscalate", "??")
    defs.setdefault("routerAgreeFrac", "??")
    defs.setdefault("oracleAcc", "??")
    defs.setdefault("oracleTok", "??")
    if os.path.exists(os.path.join(ROOT, "results", "gsm8k_probe.jsonl")):
        try:
            from harness import router
            r = router.evaluate_router("gsm8k")
            defs["routerAcc"] = f"{r['router']['accuracy']*100:.1f}"
            defs["routerTok"] = f"{r['router']['tokens']:.0f}"
            defs["routerEscalate"] = r["escalate_to"].replace("_", r"\_")
            defs["routerAgreeFrac"] = f"{r['agree_frac']*100:.0f}"
            defs["oracleAcc"] = f"{r['oracle']['accuracy']*100:.1f}"
            defs["oracleTok"] = f"{r['oracle']['tokens']:.0f}"
            defs["routerNTest"] = r["n_test"]
        except Exception as e:  # noqa
            print("router macros skipped:", e)

    out = os.path.join(ROOT, "paper", "results_macros.tex")
    with open(out, "w") as f:
        for k, v in defs.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print(f"wrote {out} ({len(defs)} macros)")


if __name__ == "__main__":
    main()
