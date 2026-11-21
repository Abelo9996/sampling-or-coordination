"""Emit the EAAI engineering-domain results (MMLU-Pro engineering) into LaTeX
fragments used only by the EAAI manuscript, leaving the shared
paper/crosstask_table.tex and paper/crosstask_macros.tex untouched:

  paper/eaai/mmlu_pro_eng_table.tex  : compact Table tab:proeng (accuracy with
      95% item-bootstrap CI, tokens/query, unparsed outputs, probabilistic
      Pareto membership) for the seven headline configurations
  paper/eaai/mmlu_pro_eng_macros.tex : headline numbers as macros (accuracy and
      CIs, SC scaling and its paired test, seed-aware hierarchical bootstrap +
      Holm, frontier composition and membership, equal-budget crossover flags,
      unparsed rates, truncation counts, chance rate, generation cap)

Why a separate compact table and not a fourth block in tab:crosstask: the shared
tab:crosstask shows MATH at the 1024-token cap (math500_trunc.jsonl), while a
table rebuilt here from results/math500_matrix.jsonl would show MATH at the
640-token cap and so change published numbers. The shared table is therefore
left as is and the engineering task gets its own table.

Reads results/mmlu_pro_eng_matrix.jsonl (the main log) and, if present,
results/mmlu_pro_eng_sa640.jsonl (the SA truncation pilot at the 640 cap) and
results/mmlu_pro_eng_aborts.jsonl (calls aborted by the runtime);
refuses to write anything if the main log is missing or incomplete, so no
placeholder numbers can enter the paper.
  .venv/bin/python scripts/emit_mmlu_pro_eng.py [--outdir paper/eaai] [--cap 1024]
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from harness import analyze, stats  # noqa: E402
from scripts import emit_crosstask as ec  # noqa: E402

TASK = "mmlu_pro_eng"
STEM = "ProEng"
COORD_STEM = {"DEBATE_N3_R1": "Debate", "GE_R2": "Ge", "HIER_M3": "Hier"}
CFG_STEM = {"SA": "SA", "SC_k2": "SCTwo", "SC_k4": "SCFour", "SC_k8": "SCEight",
            "DEBATE_N3_R1": "Debate", "GE_R2": "Ge", "HIER_M3": "Hier"}
HEADLINE7 = ["SA", "SC_k2", "SC_k4", "SC_k8", "DEBATE_N3_R1", "GE_R2", "HIER_M3"]
N_SEEDS = {"SA": 1}  # every other headline config uses seeds 1 and 2
EQUIV_BAND = 3.0     # practical-equivalence band, points (PREREGISTRATION)


def _p(p):
    """p-value as text, floored at the bootstrap resolution (B = 10000)."""
    p = max(p, 1e-4)
    return f"{p:.4f}" if p < 0.001 else f"{p:.3f}"


def chance_rate():
    ks = [len(json.loads(l)["options"]) for l in
          open(os.path.join(ROOT, "tasks_data", "mmlu_pro_engineering.jsonl"))]
    return 100.0 * sum(1.0 / k for k in ks) / len(ks), ks


def unparsed(rows):
    n, u = {}, {}
    for r in rows:
        n[r["config"]] = n.get(r["config"], 0) + 1
        u[r["config"]] = u.get(r["config"], 0) + (r["pred"] is None)
    return {c: 100.0 * u[c] / n[c] for c in n}


def table(agg, unp, memb, cap):
    lines = [r"\begin{table}[t]\centering\footnotesize",
             r"\caption{Engineering-domain task: MMLU-Pro engineering ($n{=}"
             + str(agg["SA"]["n_items"]) + r"$ items, up to ten options),"
             r" Qwen2.5-3B-Instruct (4-bit), " + str(cap) + r"-token generation cap."
             r" Accuracy (\%) with 95\% bootstrap CI over items, measured"
             r" tokens per query, unparsed outputs (\% of records), and the share"
             r" of seed-aware hierarchical resamples on which the configuration is"
             r" Pareto-optimal among these seven. Chance accuracy on the sample is "
             + f"{chance_rate()[0]:.1f}" + r"\%.}",
             r"\label{tab:proeng}",
             r"\begin{tabular}{l r r r r}",
             r"\toprule",
             r"Config & Acc. [95\% CI] & Tok. & Unparsed & Pareto-optimal \\",
             r"\midrule"]
    for cfg in ec.ORDER:
        a = agg[cfg]
        lines.append(" & ".join([
            ec.PRETTY[cfg],
            f"{a['accuracy']*100:.1f} {{\\tiny[{a['acc_lo']*100:.0f},{a['acc_hi']*100:.0f}]}}",
            f"{a['tokens']:.0f}",
            f"{unp[cfg]:.1f}",
            f"{memb[cfg]:.0f}\\%"]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def macros(agg, rows, memb, cap, pilot):
    d = {}
    d[f"nItems{STEM}"] = max(a["n_items"] for a in agg.values())
    d[f"nRecords{STEM}"] = len(rows)
    d[f"nCalls{STEM}"] = sum(r["n_calls"] for r in rows)
    d[f"cap{STEM}"] = cap
    ch, ks = chance_rate()
    d[f"chance{STEM}"] = f"{ch:.1f}"
    d[f"nTenOpt{STEM}"] = sum(k == 10 for k in ks)
    for cfg, s in CFG_STEM.items():
        a = agg[cfg]
        d[f"acc{s}{STEM}"] = f"{a['accuracy']*100:.1f}"
        d[f"accLo{s}{STEM}"] = f"{a['acc_lo']*100:.1f}"
        d[f"accHi{s}{STEM}"] = f"{a['acc_hi']*100:.1f}"
        d[f"tok{s}{STEM}"] = f"{a['tokens']:.0f}"
        d[f"member{s}{STEM}"] = f"{memb[cfg]:.0f}"
    sa, sc8 = agg["SA"], agg["SC_k8"]
    # validity check (i): SA lower CI bound at or below chance -> floor-limited
    d[f"floorLimited{STEM}"] = "true" if sa["acc_lo"] * 100 <= ch else "false"
    d[f"scDelta{STEM}"] = f"{(sc8['accuracy']-sa['accuracy'])*100:+.1f}"
    # SC k8 minus SA, item-level paired bootstrap (SA has a single seed)
    ht = stats.headline(TASK)
    dd, pp = ht["SC_k8 vs SA"]
    d[f"pScSa{STEM}"] = _p(pp)
    hsa = stats.hier_paired_test(*(lambda c: (c["SC_k8"], c["SA"]))(
        stats.per_item_seed_correct(rows)))
    d[f"ciLoScSa{STEM}"] = f"{hsa[2]:+.1f}"
    d[f"ciHiScSa{STEM}"] = f"{hsa[3]:+.1f}"
    d[f"pHierScSa{STEM}"] = _p(hsa[1])
    # unparsed outputs
    unp = unparsed(rows)
    for cfg, s in CFG_STEM.items():
        d[f"unparsed{s}{STEM}"] = f"{unp[cfg]:.1f}"
    d[f"unparsedMax{STEM}"] = f"{max(unp.values()):.1f}"
    worst = max(unp, key=unp.get)
    d[f"unparsedMaxCfg{STEM}"] = ec.PRETTY[worst]
    d[f"anyUnparsedTen{STEM}"] = "true" if max(unp.values()) >= 10 else "false"
    # truncation: SA records at the cap (single call, so exact)
    sa_rows = [r for r in rows if r["config"] == "SA"]
    d[f"truncSA{STEM}"] = f"{100.0*sum(r['out_tokens'] >= cap for r in sa_rows)/len(sa_rows):.1f}"
    if pilot:
        d[f"truncPilot{STEM}"] = f"{100.0*sum(r['out_tokens'] >= 640 for r in pilot)/len(pilot):.1f}"
        d[f"truncPilotN{STEM}"] = sum(r["out_tokens"] >= 640 for r in pilot)
        d[f"accPilot{STEM}"] = f"{100.0*sum(r['correct'] for r in pilot)/len(pilot):.1f}"
        d[f"unparsedPilot{STEM}"] = f"{100.0*sum(r['pred'] is None for r in pilot)/len(pilot):.1f}"
    # generations aborted by Ollama's token-repeat limit (harness/repeat_abort.py)
    apath = os.path.join(ROOT, "results", f"{TASK}_aborts.jsonl")
    d[f"nAborts{STEM}"] = (sum(1 for l in open(apath) if l.strip())
                           if os.path.exists(apath) else 0)
    # token ratios against SA and SC k8
    for cfg, s in COORD_STEM.items():
        d[f"tokRatioSA{s}{STEM}"] = f"{agg[cfg]['tokens']/sa['tokens']:.1f}"
        d[f"tokRatioSC{s}{STEM}"] = f"{agg[cfg]['tokens']/sc8['tokens']:.2f}"
    d[f"tokRatioSCSA{STEM}"] = f"{sc8['tokens']/sa['tokens']:.1f}"
    # equal-budget crossover (analyze) and measured/extrapolated flag (stats)
    hl = {c: agg[c] for c in HEADLINE7}
    cxh = analyze.crossover_vs_sc(hl)
    gaps = [v["gap_sc_minus_mas"] * 100 for v in cxh.values()]
    d[f"gapMin{STEM}"] = f"{min(gaps):+.1f}"
    d[f"gapMax{STEM}"] = f"{max(gaps):+.1f}"
    rep, scmax = stats.crossover_report(TASK)
    for r in rep:
        if r["config"] in COORD_STEM:
            s = COORD_STEM[r["config"]]
            d[f"gap{s}{STEM}"] = f"{r['gap']:+.1f}"
            d[f"extrap{s}{STEM}"] = "true" if r["extrapolated"] else "false"
    d[f"nExtrap{STEM}"] = sum(r["extrapolated"] for r in rep if r["config"] in COORD_STEM)
    # measured frontier composition over the seven headline configs
    front = analyze.pareto_frontier(hl)
    d[f"frontierSingle{STEM}"] = ("true" if all(hl[c]["topology"] in ("SA", "SC")
                                                for c in front) else "false")
    d[f"frontier{STEM}"] = ", ".join(ec.PRETTY[c] for c in front)
    d[f"nFrontier{STEM}"] = len(front)
    # primary test: SC_k8 minus each coordination config, hierarchical + Holm
    hh = stats.headline_hier(TASK)
    nsig_pos = nsig_neg = 0
    inband = True
    for cfg, r in hh.items():
        s = COORD_STEM[cfg]
        d[f"delta{s}{STEM}"] = f"{r['delta']:+.1f}"
        d[f"ciLo{s}{STEM}"] = f"{r['ci'][0]:+.1f}"
        d[f"ciHi{s}{STEM}"] = f"{r['ci'][1]:+.1f}"
        d[f"pRaw{s}{STEM}"] = _p(r["p"])
        d[f"pHolm{s}{STEM}"] = _p(r["holm"])
        sig = r["holm"] < 0.05
        d[f"sig{s}{STEM}"] = "true" if sig else "false"
        nsig_pos += sig and r["delta"] > 0
        nsig_neg += sig and r["delta"] < 0
        inband &= (-EQUIV_BAND <= r["ci"][0]) and (r["ci"][1] <= EQUIV_BAND)
    d[f"nSigPos{STEM}"] = nsig_pos
    d[f"nSigNeg{STEM}"] = nsig_neg
    d[f"allInBand{STEM}"] = "true" if inband else "false"
    return d, unp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.join(ROOT, "paper", "eaai"))
    ap.add_argument("--cap", type=int, default=1024,
                    help="AGENTS_NUM_PREDICT used for the main log")
    args = ap.parse_args()
    path = os.path.join(ROOT, "results", f"{TASK}_matrix.jsonl")
    if not os.path.exists(path):
        sys.exit(f"missing {path}; run the experiment first (nothing written)")
    rows = analyze.load(path)
    agg = analyze.aggregate(rows)
    need = [c for c in ec.ORDER if c not in agg]
    if need:
        sys.exit(f"incomplete run, missing configs {need} (nothing written)")
    n_items = len({r["item_id"] for r in rows})
    expect = n_items * sum(N_SEEDS.get(c, 2) for c in HEADLINE7)
    if len(rows) != expect or any(agg[c]["n_items"] != n_items for c in HEADLINE7):
        sys.exit(f"incomplete run: {len(rows)} records, expected {expect} (nothing written)")
    over = max(r["out_tokens"] for r in rows if r["config"] == "SA")
    if over > args.cap:
        sys.exit(f"SA output {over} tokens exceeds --cap {args.cap} (nothing written)")
    ppath = os.path.join(ROOT, "results", f"{TASK}_sa640.jsonl")
    pilot = analyze.load(ppath) if os.path.exists(ppath) else None
    memb = stats.pareto_membership(TASK, configs=HEADLINE7)
    defs, unp = macros(agg, rows, memb, args.cap, pilot)
    os.makedirs(args.outdir, exist_ok=True)
    tpath = os.path.join(args.outdir, "mmlu_pro_eng_table.tex")
    mpath = os.path.join(args.outdir, "mmlu_pro_eng_macros.tex")
    with open(tpath, "w") as f:
        f.write(table(agg, unp, memb, args.cap))
    with open(mpath, "w") as f:
        f.write("% Generated by scripts/emit_mmlu_pro_eng.py from "
                "results/mmlu_pro_eng_matrix.jsonl; do not edit by hand.\n")
        for k, v in defs.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print(f"wrote {tpath}\nwrote {mpath} ({len(defs)} macros)")


if __name__ == "__main__":
    main()
