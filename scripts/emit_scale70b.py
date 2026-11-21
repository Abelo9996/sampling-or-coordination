"""Emit the 70B external-validity results (paper/eaai/SCALE70B_EXPERIMENT.md)
from results/gsm8k_llama70b.jsonl: a macros file and a compact table fragment
for supplement Section S5. No hand-typed numbers in the paper.

  .venv/bin/python scripts/emit_scale70b.py            # writes paper/scale70b_*.tex
  .venv/bin/python scripts/emit_scale70b.py --print    # summary to stdout only

Analysis (fixed before the run): accuracy with 95% item-bootstrap CI and mean
total tokens per query; Pareto set and bootstrap Pareto membership; paired item
bootstrap SC_k8 vs DEBATE and SC_k4 vs DEBATE (exploratory, Holm over the two);
correction/corruption from the debate traces (round-0 majority vs final
majority, the paper's definitions in harness/mechanistic_analyze.py), plus the
per-agent counts. Only items with all four configurations are analysed.
"""
import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from harness import analyze as A  # noqa: E402
from harness import stats as S  # noqa: E402
from harness import tasks, topologies  # noqa: E402

CONFIGS = ["SA", "SC_k4", "SC_k8", "DEBATE_N3_R1"]
MAC = {"SA": "Sa", "SC_k4": "Scfour", "SC_k8": "Sceight", "DEBATE_N3_R1": "Deb"}
NAME = {"SA": "SA", "SC_k4": "SC $k{=}4$", "SC_k8": "SC $k{=}8$",
        "DEBATE_N3_R1": "Debate $N{=}3,R{=}1$"}
N_DESIGN = 80


def fmt_int(x):
    return f"{int(round(x)):,}".replace(",", "{,}")


def fmt_p(p):
    return "p{<}0.001" if p < 0.001 else f"p{{=}}{p:.3f}"


def criterion(rows, task):
    """Round-0 vs final majority (paper definition) and per-agent transitions."""
    deb = [r for r in rows if r["config"] == "DEBATE_N3_R1"]
    out = {"n": 0, "init_right": 0, "init_wrong": 0, "fix": 0, "cor": 0,
           "ag_right": 0, "ag_wrong": 0, "ag_fix": 0, "ag_cor": 0,
           "collapse": 0, "minority_init": 0, "minority_won": 0}
    for r in deb:
        texts = [c["text"] for c in r["calls"]]
        assert len(texts) == 6, r["item_id"]
        init = [task["extract"](t) for t in texts[:3]]
        fin = [task["extract"](t) for t in texts[3:]]
        fmaj = topologies._majority(fin)
        assert str(fmaj) == str(r["pred"]), (r["item_id"], fmaj, r["pred"])
        imaj = topologies._majority(init)
        ic = task["score"](imaj, r["gold"])
        fc = task["score"](fmaj, r["gold"])
        assert bool(fc) == bool(r["correct"]), r["item_id"]
        out["n"] += 1
        if ic:
            out["init_right"] += 1
            out["cor"] += int(not fc)
        else:
            out["init_wrong"] += 1
            out["fix"] += int(fc)
            if any(task["score"](a, r["gold"]) for a in init):
                out["minority_init"] += 1
                out["minority_won"] += int(fc)
        if len(set(map(str, fin))) < len(set(map(str, init))):
            out["collapse"] += 1
        for a0, a1 in zip(init, fin):
            c0, c1 = task["score"](a0, r["gold"]), task["score"](a1, r["gold"])
            if c0:
                out["ag_right"] += 1
                out["ag_cor"] += int(not c1)
            else:
                out["ag_wrong"] += 1
                out["ag_fix"] += int(c1)
    n = out["n"]
    a = out["init_right"] / n
    pfix = out["fix"] / out["init_wrong"] if out["init_wrong"] else float("nan")
    pcor = out["cor"] / out["init_right"] if out["init_right"] else float("nan")
    out.update(a=a, pfix=pfix, pcor=pcor,
               ratio=(pfix / pcor) if pcor else float("inf"),
               odds=(a / (1 - a)) if a < 1 else float("inf"),
               delta=(out["fix"] - out["cor"]) / n)
    ag_a = out["ag_right"] / (out["ag_right"] + out["ag_wrong"])
    out.update(ag_a=ag_a,
               ag_pfix=out["ag_fix"] / out["ag_wrong"] if out["ag_wrong"] else float("nan"),
               ag_pcor=out["ag_cor"] / out["ag_right"] if out["ag_right"] else float("nan"))
    out["ag_delta"] = (out["ag_fix"] - out["ag_cor"]) / (out["ag_right"] + out["ag_wrong"])
    return out


# signed values are emitted in braces so that a leading minus stays unary
# after a relation such as {=} in math mode

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=os.path.join(ROOT, "results", "gsm8k_llama70b.jsonl"))
    ap.add_argument("--ledger", default=None)
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "paper"))
    ap.add_argument("--print", action="store_true", dest="print_only")
    ap.add_argument("--allow-partial", action="store_true")
    args = ap.parse_args()
    ledger = args.ledger or args.log.replace(".jsonl", "_spend.jsonl")

    rows_all = [json.loads(l) for l in open(args.log) if l.strip()]
    by_item = {}
    for r in rows_all:
        by_item.setdefault(r["item_id"], set()).add(r["config"])
    matched = sorted(i for i, cs in by_item.items() if set(CONFIGS) <= cs)
    if len(matched) < N_DESIGN and not args.allow_partial:
        raise SystemExit(f"only {len(matched)}/{N_DESIGN} matched items; "
                         f"rerun with --allow-partial to emit a partial result")
    rows = [r for r in rows_all if r["item_id"] in set(matched) and r["config"] in CONFIGS]
    assert len(rows) == 4 * len(matched), "duplicate records in log"
    task = tasks.get_task("gsm8k")
    n = len(matched)

    agg = A.aggregate(rows)
    front = A.pareto_frontier(agg)
    # bootstrap Pareto membership (stats.pareto_membership reads by task key)
    tmp = os.path.join(os.path.dirname(args.log), "_scale70b_matched.jsonl")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps({k: r[k] for k in ("item_id", "config", "correct",
                                                  "total_tokens")}) + "\n")
    S.TASK_FILE["_scale70b"] = os.path.relpath(tmp, os.path.join(ROOT, "results"))
    pm = S.pareto_membership("_scale70b", configs=CONFIGS)
    os.remove(tmp)

    cor = S.per_item_correct(rows)
    scor = S.per_item_seed_correct(rows)
    comps = {}
    for sc in ("SC_k8", "SC_k4"):
        d, p = S.paired_test(cor[sc], cor["DEBATE_N3_R1"])
        _, _, lo, hi = S.hier_paired_test(scor[sc], scor["DEBATE_N3_R1"])
        comps[sc] = {"delta": d, "p": p, "lo": lo, "hi": hi}
    adj = S.holm([comps["SC_k8"]["p"], comps["SC_k4"]["p"]])
    comps["SC_k8"]["holm"], comps["SC_k4"]["holm"] = adj

    crit = criterion(rows, task)

    calls = [c for r in rows for c in r["calls"]]
    prov = Counter(str(c.get("provider")) for c in calls)
    served = Counter(str(c.get("served_model")) for c in calls)
    trunc = {c: sum(1 for r in rows if r["config"] == c
                    for k in r["calls"] if k.get("finish_reason") == "length")
             for c in CONFIGS}
    ncalls_cfg = {c: sum(r["n_calls"] for r in rows if r["config"] == c) for c in CONFIGS}
    spend, nled = 0.0, 0
    if os.path.exists(ledger):
        for l in open(ledger):
            d = json.loads(l)
            spend += float(d.get("cost_usd") or 0.0)
            nled += 1
    in_frac = (sum(r["in_tokens"] for r in rows) / sum(r["total_tokens"] for r in rows))

    # ---------------- summary ----------------
    print(f"matched items: {n}; records {len(rows)}; model calls in log {len(calls)}; "
          f"ledger calls {nled}; spend {spend:.4f} USD")
    print("providers:", dict(prov), "served:", dict(served))
    print("truncated calls (finish_reason=length):", trunc)
    for c in CONFIGS:
        a = agg[c]
        print(f"  {c:13} acc {100*a['accuracy']:5.1f} [{100*a['acc_lo']:5.1f},"
              f"{100*a['acc_hi']:5.1f}]  tok {a['tokens']:7.1f}  pareto={c in front}"
              f"  membership {pm[c]:5.1f}%")
    for sc, r in comps.items():
        print(f"  {sc} - DEBATE: {r['delta']:+.1f} pts CI [{r['lo']:+.1f},{r['hi']:+.1f}] "
              f"p={r['p']:.4f} Holm={r['holm']:.4f}")
    print("  criterion:", {k: (round(v, 4) if isinstance(v, float) else v)
                           for k, v in crit.items()})
    print(f"  input-token share {in_frac:.3f}")
    if args.print_only:
        return

    # ---------------- macros ----------------
    M = []

    def mac(name, val):
        M.append(f"\\newcommand{{\\sev{name}}}{{{val}}}")
    mac("N", n)
    mac("Calls", fmt_int(len(calls)))
    mac("SpendUsd", f"{spend:.3f}")
    mac("SpendUsdExact", f"{spend:.4f}")
    mac("Providers", ", ".join(f"{p} ({fmt_int(k)} calls)" for p, k in prov.most_common()))
    mac("NProviders", len(prov))
    mac("TruncCalls", sum(trunc.values()))
    mac("InShare", f"{100*in_frac:.0f}")
    for c in CONFIGS:
        a, m = agg[c], MAC[c]
        mac(f"Acc{m}", f"{100*a['accuracy']:.1f}")
        mac(f"AccLo{m}", f"{100*a['acc_lo']:.1f}")
        mac(f"AccHi{m}", f"{100*a['acc_hi']:.1f}")
        mac(f"Tok{m}", fmt_int(a["tokens"]))
        mac(f"Pm{m}", f"{pm[c]:.0f}")
    mac("Pareto", ", ".join(NAME[c] for c in front))
    for sc, tag in (("SC_k8", "Eight"), ("SC_k4", "Four")):
        r = comps[sc]
        mac(f"Delta{tag}", "{"+f"{r['delta']:+.1f}"+"}")
        mac(f"CiLo{tag}", "{"+f"{r['lo']:+.1f}"+"}")
        mac(f"CiHi{tag}", "{"+f"{r['hi']:+.1f}"+"}")
        mac(f"P{tag}", fmt_p(r["p"]))
        mac(f"Holm{tag}", fmt_p(r["holm"]))
    mac("CritA", f"{100*crit['a']:.1f}")
    mac("CritAfrac", f"{crit['a']:.3f}")
    mac("CritInitRight", crit["init_right"])
    mac("CritInitWrong", crit["init_wrong"])
    mac("CritFix", crit["fix"])
    mac("CritCor", crit["cor"])
    mac("CritPfix", f"{crit['pfix']:.2f}" if crit["init_wrong"] else "n/a")
    mac("CritPcor", f"{crit['pcor']:.3f}")
    mac("CritRatio", ("\\infty" if crit["cor"] == 0 else f"{crit['ratio']:.1f}"))
    mac("CritOdds", f"{crit['odds']:.1f}")
    mac("CritDelta", "{"+f"{100*crit['delta']:+.2f}"+"}")
    mac("CritNet", "{"+f"{crit['fix'] - crit['cor']:+d}"+"}")
    mac("CritCollapse", crit["collapse"])
    mac("AgN", crit["ag_right"] + crit["ag_wrong"])
    mac("AgA", f"{100*crit['ag_a']:.1f}")
    mac("AgRight", crit["ag_right"])
    mac("AgWrong", crit["ag_wrong"])
    mac("AgFix", crit["ag_fix"])
    mac("AgCor", crit["ag_cor"])
    mac("AgDelta", "{"+f"{100*crit['ag_delta']:+.2f}"+"}")
    mac("AgPfix", f"{crit['ag_pfix']:.3f}")
    mac("AgPcor", f"{crit['ag_pcor']:.3f}")
    mac("AgRatio", ("\\infty" if crit["ag_cor"] == 0 else f"{crit['ag_pfix']/crit['ag_pcor']:.1f}"))
    mac("AgOdds", f"{crit['ag_a']/(1-crit['ag_a']):.1f}")
    hdr = ("% Generated by scripts/emit_scale70b.py from results/gsm8k_llama70b.jsonl.\n"
           "% Do not edit by hand.\n")
    with open(os.path.join(args.out_dir, "scale70b_macros.tex"), "w") as f:
        f.write(hdr + "\n".join(M) + "\n")

    # ---------------- table ----------------
    T = [r"\begin{table}[t]\centering\small",
         r"\caption{Frontier-scale probe: Llama-3.3-70B-Instruct (OpenRouter API) on the "
         r"same $\sevN$ GSM8K items as the main matrix, one seed, unchanged prompts, "
         r"parser, output cap, and topologies. Accuracy with $95\%$ item-bootstrap CI; "
         r"Tok.\ is mean measured input plus output tokens per query (Llama-3 "
         r"tokenizer, not comparable with the Qwen rows of Table~\ref{tab:scale}); "
         r"Pareto is the share of $10{,}000$ bootstrap resamples in which the "
         r"configuration is non-dominated. Paired differences are exploratory.}",
         r"\label{tab:scale70b}",
         r"\begin{tabular}{lrrrr}", r"\toprule",
         r"Config & Acc.\ (\%) & 95\% CI & Tok. & Pareto (\%) \\", r"\midrule"]
    for c in CONFIGS:
        m = MAC[c]
        T.append(f"{NAME[c]} & \\sevAcc{m} & [\\sevAccLo{m}, \\sevAccHi{m}] & "
                 f"\\sevTok{m} & \\sevPm{m} \\\\")
    T += [r"\midrule",
          r"SC$_{k=8}-$Debate & \multicolumn{4}{l}{$\sevDeltaEight$ pts, CI "
          r"$[\sevCiLoEight,\sevCiHiEight]$, $\sevPEight$} \\",
          r"SC$_{k=4}-$Debate & \multicolumn{4}{l}{$\sevDeltaFour$ pts, CI "
          r"$[\sevCiLoFour,\sevCiHiFour]$, $\sevPFour$} \\",
          r"Debate traces & \multicolumn{4}{l}{$a{=}\sevCritAfrac$; fixed "
          r"$\sevCritFix/\sevCritInitWrong$, corrupted $\sevCritCor/\sevCritInitRight$; "
          r"$\Delta{=}\sevCritDelta$ pts} \\",
          r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(os.path.join(args.out_dir, "scale70b_table.tex"), "w") as f:
        f.write("% Generated by scripts/emit_scale70b.py; do not edit by hand.\n"
                + "\n".join(T) + "\n")
    print("wrote scale70b_macros.tex and scale70b_table.tex in", args.out_dir)


if __name__ == "__main__":
    main()
