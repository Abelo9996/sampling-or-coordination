"""Emit the EAAI code-task and energy results (HumanEval on the RTX 4060) into
LaTeX fragments used only by the EAAI manuscript and supplement, following the
analysis and reporting rule of paper/eaai/CODE_ENERGY_EXPERIMENT.md, section 7.

  paper/eaai/humaneval_table.tex       Table tab:code (pass@1 with 95% item
      bootstrap CI, tokens, GPU energy gross and idle-subtracted, joules per
      token, unparsed outputs) for the seven headline configurations
  paper/eaai/humaneval_supp_table.tex  Table tab:S-code (supplement): per-sample
      pass rate, share of votes with no repeated program, the pre-specified
      secondary (example-execution) vote, Pareto membership on tokens and on
      joules, throughput
  paper/eaai/humaneval_macros.tex      every number quoted in the text
  results/analysis/humaneval.json      the same, machine-readable

Inputs: results/humaneval_matrix_pc.jsonl (2,132 records, energy_j and
energy_j_net per record and per call), results/humaneval_matrix_pc_idle.jsonl
(idle baselines), results/analysis/humaneval_secondary_vote.json and
results/humaneval_secondary_vote.jsonl (scripts/humaneval_secondary_vote.py).
Refuses to write anything if the run is incomplete or a record lacks energy.

Statistics, as for the other tasks: per-config pass@1 averaged over seeds within
item, 95% bootstrap CI over items; seed-aware hierarchical paired bootstrap
(B = 10,000) of SC k=8 against DEBATE, GE and HIER with Holm correction within
that family (pre-specified); SC k=8 against SA with the item-level paired
bootstrap and the hierarchical CI (as in the paper); and, not pre-specified,
SA against DEBATE, GE and HIER with Holm correction within that family, because
SA is the observed frontier. Measured Pareto frontier among the seven
configurations on tokens and, separately, on gross joules.

Every 95% interval of a mean over items (pass@1, example-execution vote, energy) is
harness.analyze.item_bootstrap_ci: 10,000 resamples, seed 0, items in sorted id order (the
method stated in the paper; it was B = 2,000 in log order before 23 Nov 2026).

  .venv/bin/python scripts/emit_humaneval.py [--outdir DIR] [--macros-dir DIR] [--json PATH]
  (the arXiv rev3 build: --outdir paper/arxiv_rev3/tables --macros-dir paper/arxiv_rev3)
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from harness import analyze, stats  # noqa: E402

TASK = "humaneval"
LOG = os.path.join(ROOT, "results", "humaneval_matrix_pc.jsonl")
IDLE = os.path.join(ROOT, "results", "humaneval_matrix_pc_idle.jsonl")
SV_SUM = os.path.join(ROOT, "results", "analysis", "humaneval_secondary_vote.json")
SV_REC = os.path.join(ROOT, "results", "humaneval_secondary_vote.jsonl")
OUT = os.path.join(ROOT, "paper", "eaai")
STEM = "Code"
ORDER = ["SA", "SC_k2", "SC_k4", "SC_k8", "GE_R2", "HIER_M3", "DEBATE_N3_R1"]
COORD = ["DEBATE_N3_R1", "GE_R2", "HIER_M3"]
CS = {"SA": "SA", "SC_k2": "SCTwo", "SC_k4": "SCFour", "SC_k8": "SCEight",
      "DEBATE_N3_R1": "Debate", "GE_R2": "Ge", "HIER_M3": "Hier"}
PRETTY = {"SA": "SA", "SC_k2": "SC $k{=}2$", "SC_k4": "SC $k{=}4$",
          "SC_k8": "SC $k{=}8$", "GE_R2": "GE", "HIER_M3": "HIER",
          "DEBATE_N3_R1": "Debate"}
VOTED = ["SC_k2", "SC_k4", "SC_k8", "DEBATE_N3_R1"]
N_SEEDS = {"SA": 1}
CAP = 1024


def _p(p):
    """p-value with its relation sign, for use as $p\\macro$: "{<}0.001" below
    the reporting floor (the bootstrap resolution is 1/B = 0.0001), else "{=}x"."""
    return "{<}0.001" if p < 0.001 else f"{{=}}{p:.3f}"


def _int(x):
    return f"{x:,.0f}".replace(",", "{,}")


def per_item_mean(rows, key):
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r["config"]][r["item_id"]].append(float(r[key]))
    return {c: {i: float(np.mean(v)) for i, v in d.items()} for c, d in by.items()}


def boot_ci(per_item):
    """per_item: {item_id: value}; the paper's item bootstrap (see the module docstring)."""
    return analyze.item_bootstrap_ci(per_item)


def frontier(points):
    """points: cfg -> (cost, acc); non-dominated set sorted by cost."""
    out = []
    for c, (x, y) in points.items():
        if not any(xx <= x and yy >= y and (xx < x or yy > y)
                   for cc, (xx, yy) in points.items() if cc != c):
            out.append(c)
    return sorted(out, key=lambda c: points[c][0])


def kendall_tau(a, b):
    keys = list(a)
    conc = disc = 0
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            s = np.sign(a[keys[i]] - a[keys[j]]) * np.sign(b[keys[i]] - b[keys[j]])
            conc += s > 0
            disc += s < 0
    n = len(keys)
    return (conc - disc) / (n * (n - 1) / 2), int(disc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=OUT, help="directory of the two table fragments")
    ap.add_argument("--macros-dir", default=None, help="directory of humaneval_macros.tex (default: --outdir)")
    ap.add_argument("--json", default=os.path.join(ROOT, "results", "analysis", "humaneval.json"))
    args = ap.parse_args()
    out_dir, mac_dir = args.outdir, args.macros_dir or args.outdir
    if not os.path.exists(LOG):
        sys.exit(f"missing {LOG} (nothing written)")
    rows = analyze.load(LOG)
    items = sorted({r["item_id"] for r in rows})
    expect = len(items) * sum(N_SEEDS.get(c, 2) for c in ORDER)
    if len(rows) != expect or len(items) != 164:
        sys.exit(f"incomplete run: {len(rows)} records, expected {expect} (nothing written)")
    if any(r.get("energy_j") is None or r.get("energy_j_net") is None for r in rows):
        sys.exit("a record lacks energy (nothing written)")
    if not os.path.exists(SV_SUM):
        sys.exit("run scripts/humaneval_secondary_vote.py first (nothing written)")
    idle = [json.loads(l) for l in open(IDLE) if l.strip()]
    sv = json.load(open(SV_SUM))
    svr = [json.loads(l) for l in open(SV_REC) if l.strip()]

    agg = analyze.aggregate(rows)
    pc = per_item_mean(rows, "correct")
    for c, a in agg.items():  # accuracy intervals: the paper's item bootstrap, not aggregate's
        assert abs(np.mean(list(pc[c].values())) - a["accuracy"]) < 1e-12, c
        a["acc_lo"], a["acc_hi"] = boot_ci(pc[c])
    en = per_item_mean(rows, "energy_j")
    enn = per_item_mean(rows, "energy_j_net")
    d, J = {}, {"configs": {}}
    d[f"nItems{STEM}"] = len(items)
    d[f"nRecords{STEM}"] = _int(len(rows))
    d[f"nCalls{STEM}"] = _int(sum(r["n_calls"] for r in rows))
    d[f"cap{STEM}"] = _int(CAP)
    calls = [c for r in rows for c in r["calls"]]
    t0, t1 = min(c["t0"] for c in calls), max(c["t1"] for c in calls)
    d[f"wallHours{STEM}"] = f"{(t1 - t0) / 3600:.1f}"
    d[f"callsAtCap{STEM}"] = sum(c["out_tokens"] >= CAP for c in calls)
    d[f"callsAtCapPct{STEM}"] = f"{100.0 * d[f'callsAtCap{STEM}'] / len(calls):.1f}"
    d[f"truncSA{STEM}"] = sum(r["out_tokens"] >= CAP for r in rows if r["config"] == "SA")
    idle_w = {i["label"]: i["mean_w"] for i in idle}
    d[f"idleStart{STEM}"] = f"{idle_w['start']:.1f}"
    d[f"idleEnd{STEM}"] = f"{idle_w['end']:.1f}"
    lat = sum(c["latency_s"] for c in calls)
    d[f"powerGen{STEM}"] = f"{sum(c['energy_j'] for c in calls) / lat:.0f}"
    # per-call energy against input and output tokens (least squares)
    X = np.array([[1.0, c["in_tokens"], c["out_tokens"]] for c in calls])
    y = np.array([c["energy_j"] for c in calls])
    with np.errstate(all="ignore"):  # spurious BLAS warnings on macOS numpy
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        r2 = float(1 - ((y - X @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    d[f"regOut{STEM}"] = f"{beta[2]:.2f}"
    d[f"regIn{STEM}"] = f"{beta[1]:.3f}"
    d[f"regIcpt{STEM}"] = f"{beta[0]:.1f}"
    d[f"regRsq{STEM}"] = f"{r2:.3f}"
    d[f"regRatio{STEM}"] = f"{beta[2] / beta[1]:.0f}"
    J["energy_regression"] = {"intercept": beta[0], "per_in_tok": beta[1],
                              "per_out_tok": beta[2], "r2": r2, "n_calls": len(calls)}
    unp, jpt, jpo, pw = {}, {}, {}, {}
    for cfg in ORDER:
        rc = [r for r in rows if r["config"] == cfg]
        unp[cfg] = 100.0 * sum(r["pred"] is None for r in rc) / len(rc)
        jpt[cfg] = sum(r["energy_j"] for r in rc) / sum(r["total_tokens"] for r in rc)
        jpo[cfg] = sum(r["energy_j"] for r in rc) / sum(r["out_tokens"] for r in rc)
        cc = [c for r in rc for c in r["calls"]]
        pw[cfg] = sum(c["energy_j"] for c in cc) / sum(c["latency_s"] for c in cc)
        a = agg[cfg]
        e = float(np.mean(list(en[cfg].values())))
        en_lo, en_hi = boot_ci(en[cfg])
        e_net = float(np.mean(list(enn[cfg].values())))
        s = CS[cfg]
        d[f"acc{s}{STEM}"] = f"{a['accuracy'] * 100:.1f}"
        d[f"accLo{s}{STEM}"] = f"{a['acc_lo'] * 100:.1f}"
        d[f"accHi{s}{STEM}"] = f"{a['acc_hi'] * 100:.1f}"
        d[f"tok{s}{STEM}"] = _int(a["tokens"])
        d[f"en{s}{STEM}"] = _int(e)
        d[f"enNet{s}{STEM}"] = _int(e_net)
        d[f"jpt{s}{STEM}"] = f"{jpt[cfg]:.2f}"
        d[f"jpo{s}{STEM}"] = f"{jpo[cfg]:.2f}"
        d[f"unparsed{s}{STEM}"] = f"{unp[cfg]:.1f}"
        d[f"spq{s}{STEM}"] = f"{np.mean([r['latency_s'] for r in rc]):.1f}"
        J["configs"][cfg] = {"pass1": a["accuracy"], "ci": [a["acc_lo"], a["acc_hi"]],
                             "tokens": a["tokens"], "energy_j": e,
                             "energy_ci": [en_lo, en_hi], "energy_j_net": e_net,
                             "j_per_tok": jpt[cfg], "j_per_out_tok": jpo[cfg],
                             "mean_power_w": pw[cfg], "unparsed_pct": unp[cfg],
                             "s_per_query": float(np.mean([r["latency_s"] for r in rc])),
                             "calls_per_query": float(np.mean([r["n_calls"] for r in rc]))}
    sasc = [jpt[c] for c in ("SA", "SC_k2", "SC_k4", "SC_k8")]
    co = [jpt[c] for c in COORD]
    d[f"jptSampLo{STEM}"], d[f"jptSampHi{STEM}"] = f"{min(sasc):.2f}", f"{max(sasc):.2f}"
    d[f"jptCoordLo{STEM}"], d[f"jptCoordHi{STEM}"] = f"{min(co):.2f}", f"{max(co):.2f}"
    d[f"jpoLo{STEM}"] = f"{min(jpo.values()):.2f}"
    d[f"jpoHi{STEM}"] = f"{max(jpo.values()):.2f}"
    d[f"powerLo{STEM}"] = f"{min(pw.values()):.0f}"
    d[f"powerHi{STEM}"] = f"{max(pw.values()):.0f}"
    d[f"enRatioDebSA{STEM}"] = f"{J['configs']['DEBATE_N3_R1']['energy_j'] / J['configs']['SA']['energy_j']:.1f}"
    d[f"enRatioScSA{STEM}"] = f"{J['configs']['SC_k8']['energy_j'] / J['configs']['SA']['energy_j']:.1f}"

    # frontiers on tokens and on joules, and rank agreement
    tokp = {c: (agg[c]["tokens"], agg[c]["accuracy"]) for c in ORDER}
    enp = {c: (J["configs"][c]["energy_j"], agg[c]["accuracy"]) for c in ORDER}
    ft, fe = frontier(tokp), frontier(enp)
    d[f"frontierTok{STEM}"] = ", ".join(PRETTY[c] for c in ft)
    d[f"frontierEn{STEM}"] = ", ".join(PRETTY[c] for c in fe)
    d[f"frontierSame{STEM}"] = "true" if ft == fe else "false"
    tau, ndisc = kendall_tau({c: tokp[c][0] for c in ORDER}, {c: enp[c][0] for c in ORDER})
    d[f"tau{STEM}"] = f"{tau:.2f}"
    d[f"nDiscordant{STEM}"] = ndisc
    d[f"tokRank{STEM}"] = ", ".join(PRETTY[c] for c in sorted(ORDER, key=lambda c: tokp[c][0]))
    d[f"enRank{STEM}"] = ", ".join(PRETTY[c] for c in sorted(ORDER, key=lambda c: enp[c][0]))
    J["frontier_tokens"], J["frontier_joules"] = ft, fe
    J["kendall_tau_tokens_vs_joules"], J["discordant_pairs"] = tau, ndisc
    # every coordination config dominated by SA on both axes?
    for ax, P in (("Tok", tokp), ("En", enp)):
        dom = all(P["SA"][0] < P[c][0] and P["SA"][1] > P[c][1] for c in COORD)
        d[f"saDominates{ax}{STEM}"] = "true" if dom else "false"
    memt = stats.pareto_membership(TASK, configs=ORDER)
    meme = stats.pareto_membership(TASK, configs=ORDER, cost="energy_j")
    for c in ORDER:
        d[f"memberTok{CS[c]}{STEM}"] = f"{memt[c]:.0f}"
        d[f"memberEn{CS[c]}{STEM}"] = f"{meme[c]:.0f}"
    J["pareto_membership_tokens"], J["pareto_membership_joules"] = memt, meme
    d[f"memberMaxOther{STEM}"] = f"{max(max(memt[c], meme[c]) for c in ORDER if c != 'SA'):.0f}"

    # significance
    cor = stats.per_item_seed_correct(rows)
    raw = [stats.hier_paired_test(cor["SC_k8"], cor[c]) for c in COORD]
    adj = stats.holm([r[1] for r in raw])
    J["sc8_vs_coord"] = {}
    for c, r, pa in zip(COORD, raw, adj):
        s = CS[c]
        d[f"delta{s}{STEM}"] = f"{r[0]:+.1f}"
        d[f"ciLo{s}{STEM}"] = f"{r[2]:+.1f}"
        d[f"ciHi{s}{STEM}"] = f"{r[3]:+.1f}"
        d[f"pRaw{s}{STEM}"] = _p(r[1])
        d[f"pHolm{s}{STEM}"] = _p(pa)
        d[f"sig{s}{STEM}"] = "true" if pa < 0.05 else "false"
        J["sc8_vs_coord"][c] = {"delta": r[0], "ci": [r[2], r[3]], "p": r[1], "holm": pa}
    raw = [stats.hier_paired_test(cor["SA"], cor[c]) for c in COORD]
    adj = stats.holm([r[1] for r in raw])
    J["sa_vs_coord"] = {}
    for c, r, pa in zip(COORD, raw, adj):
        s = CS[c]
        d[f"saDelta{s}{STEM}"] = f"{r[0]:+.1f}"
        d[f"saCiLo{s}{STEM}"] = f"{r[2]:+.1f}"
        d[f"saCiHi{s}{STEM}"] = f"{r[3]:+.1f}"
        d[f"saPHolm{s}{STEM}"] = _p(pa)
        J["sa_vs_coord"][c] = {"delta": r[0], "ci": [r[2], r[3]], "p": r[1], "holm": pa}
    pim = stats.per_item_correct(rows)
    dd, pp = stats.paired_test(pim["SC_k8"], pim["SA"])
    h = stats.hier_paired_test(cor["SC_k8"], cor["SA"])
    d[f"scDelta{STEM}"] = f"{dd:+.1f}"
    d[f"pScSa{STEM}"] = _p(pp)
    d[f"ciLoScSa{STEM}"] = f"{h[2]:+.1f}"
    d[f"ciHiScSa{STEM}"] = f"{h[3]:+.1f}"
    d[f"pHierScSa{STEM}"] = _p(h[1])
    # the same comparison signed as SA minus SC k=8 (guideline table)
    d[f"saScDelta{STEM}"] = f"{-h[0]:+.1f}"
    d[f"saScCiLo{STEM}"] = f"{-h[3]:+.1f}"
    d[f"saScCiHi{STEM}"] = f"{-h[2]:+.1f}"
    J["sc8_vs_sa"] = {"delta": dd, "p_item": pp, "ci_hier": [h[2], h[3]], "p_hier": h[1]}
    # base-accuracy screen: adopt a coordination step only if a < 1/2
    a = agg["SA"]["accuracy"]
    d[f"screenA{STEM}"] = f"{a:.2f}"
    d[f"nCoordBelowSA{STEM}"] = sum(agg[c]["accuracy"] < a for c in COORD)

    # secondary (example-execution) vote and per-sample pass rates
    d[f"nExamples{STEM}"] = sv["items_with_examples"]
    d[f"rescoreN{STEM}"] = _int(sv["local_rescore_n"])
    d[f"rescoreAgree{STEM}"] = _int(sv["local_rescore_agree"])
    by = defaultdict(lambda: defaultdict(list))
    for o in svr:
        by[o["config"]][o["item_id"]].append(int(o["correct2"]))
    J["secondary_vote"] = {}
    for c in VOTED:
        s = CS[c]
        vals = [float(np.mean(v)) for v in by[c].values()]
        lo, hi = boot_ci({i: float(np.mean(v)) for i, v in by[c].items()})
        d[f"accTwo{s}{STEM}"] = f"{100 * np.mean(vals):.1f}"
        d[f"accTwoLo{s}{STEM}"] = f"{100 * lo:.1f}"
        d[f"accTwoHi{s}{STEM}"] = f"{100 * hi:.1f}"
        d[f"changed{s}{STEM}"] = sv[c]["changed"]
        d[f"perSample{s}{STEM}"] = f"{sv[c]['per_sample_pass']:.1f}"
        d[f"anySample{s}{STEM}"] = f"{sv[c]['any_sample_pass']:.1f}"
        d[f"allDistinct{s}{STEM}"] = f"{sv[c]['all_distinct']:.0f}"
        J["secondary_vote"][c] = dict(sv[c], pass1=float(np.mean(vals)), ci=[lo, hi])
    sec = {i: np.asarray(v, float) for i, v in by["SC_k8"].items()}
    h2 = stats.hier_paired_test(sec, cor["SA"])
    d[f"twoDeltaScSa{STEM}"] = f"{h2[0]:+.1f}"
    d[f"twoCiLoScSa{STEM}"] = f"{h2[2]:+.1f}"
    d[f"twoCiHiScSa{STEM}"] = f"{h2[3]:+.1f}"
    d[f"twoPScSa{STEM}"] = _p(h2[1])
    J["secondary_sc8_vs_sa"] = {"delta": h2[0], "ci": [h2[2], h2[3]], "p": h2[1]}

    # tables
    t = [r"% Generated by scripts/emit_humaneval.py; do not edit by hand.",
         r"\begin{table}[t]\centering\footnotesize",
         r"\caption{Code generation with executable hidden tests: HumanEval ($n{=}"
         + str(len(items)) + r"$ problems), Qwen2.5-3B-Instruct (4-bit) on the RTX~4060, "
         + str(CAP) + r"-token cap. Pass@1 (\%) with 95\% bootstrap CI over problems; "
         r"measured tokens per query; GPU board energy per query (J), gross and with "
         r"the idle draw ($" + d[f"idleStart{STEM}"] + r"$\,W) subtracted; gross joules "
         r"per total token; unparsed outputs (\% of records, no parseable program). "
         r"Self-consistency and debate vote over exact programs after syntax-tree "
         r"normalization.}",
         r"\label{tab:code}",
         r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{@{}l r r r r r r@{}}",
         r"\toprule",
         r"Config & Pass@1 [95\% CI] & Tok. & Gross (J) & Net (J) & J/tok & Unparsed \\",
         r"\midrule"]
    for c in ORDER:
        s = CS[c]
        t.append(" & ".join([
            PRETTY[c],
            f"{d[f'acc{s}{STEM}']} {{\\tiny[{float(d[f'accLo{s}{STEM}']):.0f},{float(d[f'accHi{s}{STEM}']):.0f}]}}",
            d[f"tok{s}{STEM}"], d[f"en{s}{STEM}"], d[f"enNet{s}{STEM}"],
            d[f"jpt{s}{STEM}"], d[f"unparsed{s}{STEM}"]]) + r" \\")
    t += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(out_dir, "humaneval_table.tex"), "w").write("\n".join(t) + "\n")

    t = [r"% Generated by scripts/emit_humaneval.py; do not edit by hand.",
         r"\begin{table}[tp]\centering\footnotesize",
         r"\caption{Code task, further analyses (HumanEval, RTX~4060). Per-sample: pass "
         r"rate of the individual programs the vote is taken over (sampled at "
         r"$T{=}0.7$; for debate, the final-round agents); any: share of queries with at "
         r"least one passing program (hidden tests, descriptive only, never used to "
         r"select); no repeat: share of queries in which no program occurs twice, so "
         r"the exact-program vote falls to the first sample; Ex.\ vote: pass@1 under "
         r"the pre-specified secondary rule (vote by outputs on the prompt's own "
         r"examples, " + str(sv["items_with_examples"]) + r" of " + str(len(items)) +
         r" problems); Pareto: share of seed-aware hierarchical resamples on which the "
         r"configuration is non-dominated among the seven, with cost in tokens and in "
         r"gross joules; s/query: mean latency of one sequential request stream; n/a: no vote over candidates.}",
         r"\label{tab:S-code}",
         r"\setlength{\tabcolsep}{3.5pt}",
         r"\begin{tabular}{@{}l r r r r r r r@{}}",
         r"\toprule",
         r" & \multicolumn{3}{c}{Vote inputs (\%)} & & \multicolumn{2}{c}{Pareto (\%)} & \\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){6-7}",
         r"Config & Per-sample & Any & No repeat & Ex.\ vote [95\% CI] & Tokens & Joules & s/query \\",
         r"\midrule"]
    for c in ORDER:
        s = CS[c]
        if c in VOTED:
            v = [d[f"perSample{s}{STEM}"], d[f"anySample{s}{STEM}"],
                 d[f"allDistinct{s}{STEM}"],
                 f"{d[f'accTwo{s}{STEM}']} {{\\tiny[{float(d[f'accTwoLo{s}{STEM}']):.0f},{float(d[f'accTwoHi{s}{STEM}']):.0f}]}}"]
        else:
            v = ["n/a", "n/a", "n/a", "n/a"]
        t.append(" & ".join([PRETTY[c]] + v + [d[f"memberTok{s}{STEM}"],
                                               d[f"memberEn{s}{STEM}"],
                                               d[f"spq{s}{STEM}"]]) + r" \\")
    t += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(out_dir, "humaneval_supp_table.tex"), "w").write("\n".join(t) + "\n")

    with open(os.path.join(mac_dir, "humaneval_macros.tex"), "w") as f:
        f.write("% Generated by scripts/emit_humaneval.py from "
                "results/humaneval_matrix_pc.jsonl; do not edit by hand.\n")
        for k, v in d.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    json.dump(J, open(args.json, "w"),
              indent=1, default=float)
    for k, v in d.items():
        print(f"{k} = {v}")


if __name__ == "__main__":
    main()
