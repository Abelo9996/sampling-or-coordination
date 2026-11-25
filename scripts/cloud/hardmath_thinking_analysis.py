"""Hard-MATH check for a thinking model: the cross-model tests on MATH-500 levels 4 and 5.

Data: results/cloud/qwen3-4b-hardmath/math500_s1.jsonl (Qwen/Qwen3-4B, thinking on, vLLM bf16 on a
Modal H100, cap 8,192, seed 1 only; raw logs compressed in results/cloud/hardmath_raw_logs.tar.xz).
The run was stopped by its budget guard, so only part of the 262 level 4 and 5 items finished.

Which items are used. Only whole items: an item enters when SA, SC_k8 (k=2, 4 = its first 2, 4 calls),
DEBATE_N3_R1, GE_R2 and HIERFIX_M3 all exist for seed 1, so every comparison is paired. The runner
dispatched items in the phase-C order (scripts/run_power_campaign.load_math_levels: a seed-0 random
shuffle of the 262 level 4 and 5 items, not sorted by level) with many units in flight, and a unit is
written when it finishes, so the whole items are not a prefix of that order: an item whose SC k=8 (eight
long generations) had not finished at the stop is missing. The script reports the positions, the level
mix and a selection check (each workflow's accuracy and tokens on the whole items against the items it
finished that are not whole).

Tests: those of scripts/cloud/cross_model_analysis.py (functions called unchanged: R/C/N verdicts under
total and output tokens, cost-matched comparator, token multiple, break-even r* of SC k=8 against debate,
truncation), with one seed per item (SA, SC and the workflows each one record per item).

Reference: the paper's 4-bit Qwen2.5-3B hard-MATH run (results/power/math500hard_qwen3b_s*.jsonl, 262
items x seeds 1 to 3, cap 2,048; results/extra/hard_math_analysis.json), and the same run restricted to
the items used here (comparator rule of protocols/HARD_MATH_PROTOCOL.md: SC with the largest k in
2, 4, 8, 12 whose mean cost on these items is not above the workflow's; harness.stats.hier_paired_test).
Without the 4-bit logs (public release) the cached JSON values are kept for that block.

Writes results/extra/hardmath_thinking_analysis.json and, in each --outdir (default paper/eaai_rev3 if
present, else paper/arxiv_rev3): hardmath_thinking_macros.tex and tables/hardmath_thinking_table.tex.
Run: /usr/bin/python3 scripts/cloud/hardmath_thinking_analysis.py [--jobs 8] [--outdir DIR ...]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cross_model_analysis as xm  # noqa: E402  (loads rr, rc, harness)

ROOT, rr = xm.ROOT, xm.rr
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from harness import stats, tasks, topologies  # noqa: E402
from run_power_campaign import load_math_levels  # noqa: E402

D_HARD = os.path.join(ROOT, "results/cloud/qwen3-4b-hardmath")
OUT_JSON = os.path.join(ROOT, "results/extra/hardmath_thinking_analysis.json")
REF_JSON = os.path.join(ROOT, "results/extra/hard_math_analysis.json")
FOURBIT = "results/power/math500hard_qwen3b_s*.jsonl"
ALL5 = ["SA"] + xm.WF


def plan():
    task = tasks.get_task("math500")
    items = load_math_levels(tasks, 4, 5, 10 ** 6)
    order = [it["id"] for it in items]
    level = {it["id"]: it["level"] for it in items}
    recs = rr.load([os.path.relpath(p, ROOT) for p in sorted(glob.glob(os.path.join(D_HARD, "math500_s*.jsonl")))])
    assert {k[2] for k in recs} == {1}, "seed 1 only expected"
    assert not ({k[1] for k in recs} - set(order)), "items outside levels 4 and 5"
    ids = sorted(i for i in order if all((c, i, 1) in recs for c in ALL5))
    pos = sorted(order.index(i) for i in ids)
    touched = {k[1] for k in recs}
    info = {"n_items_levels_4_5": len(order), "n_items": len(ids), "n_items_with_any_record": len(touched),
            "records_in_file": len(recs), "records_per_config": {c: sum(k[0] == c for k in recs) for c in ALL5},
            "order": "phase-C order: seed-0 random shuffle of the 262 level 4 and 5 items (not level-sorted)",
            "positions_in_order": {"min": pos[0], "max": pos[-1],
                                   "whole_items_among_first_n": {str(n): sum(p < n for p in pos) for n in (50, 100, 133, 144)}},
            "level_mix": {str(L): sum(level[i] == L for i in ids) for L in (4, 5)},
            "level_mix_all_262": {str(L): sum(level[i] == L for i in order) for L in (4, 5)},
            "level_mix_first_100_in_order": {str(L): sum(level[i] == L for i in order[:100]) for L in (4, 5)},
            "seed_items": {"1": len(ids)}, "uniform_seeds": True, "seeds_present_in_files": [1],
            "items_excluded": [i for i in order if i in touched and i not in ids]}
    # selection check: each configuration on the whole items against the other items it finished
    sel = {}
    for c in ALL5:
        other = [i for i in order if i not in ids and (c, i, 1) in recs]
        row = {}
        for nm, S in (("whole", ids), ("not_whole", other)):
            R = [recs[(c, i, 1)] for i in S]
            row[nm] = {"n": len(R)} if not R else {
                "n": len(R), "acc": 100 * float(np.mean([bool(r["correct"]) for r in R])),
                "out_tokens": float(np.mean([rr.rec_cost(r)[1] for r in R])),
                "records_any_cap_pct": 100 * float(np.mean([any(x["hit_cap"] for x in r["calls"]) for r in R])),
                "level5": sum(level[i] == 5 for i in S)}
        sel[c] = row
    info["selection_check"] = sel
    return task, recs, ids, {i: [1] for i in ids}, info, level


def by_level(cells, ids, level):
    return {str(L): {c: 100 * float(np.mean([cells[c].M[i].mean() for i in ids if level[i] == L])) for c in xm.ALL}
            for L in (4, 5)}


def fourbit_subset(ids, level):
    """Paper's 4-bit hard run on the given items (all three seeds), protocol comparator rule."""
    recs = rr.load([FOURBIT])
    if not recs:
        return None
    task = tasks.get_task("math500")
    S = [1, 2, 3]
    co = {"DEBATE_N3_R1": "DEBATE_N3_R1", "GE_R2": "GE_R2", "HIERFIX_M3": "HIER_M3"}
    assert all(("SA", i, 1) in recs and all((c, i, s) in recs for c in ["SC_k12"] + list(co.values()) for s in S)
               for i in ids)
    cells = {"SA": rr.cell_from(recs, "SA", ids, S)}
    cells.update({w: rr.cell_from(recs, src, ids, S) for w, src in co.items()})
    cells.update(rr.sc_prefix_cells(recs, "SC_k12", ids, S, task, [2, 4, 8, 12]))
    acc = {c: 100 * float(np.mean([v.M[i].mean() for i in ids])) for c, v in cells.items()}
    cost = {c: {"total_tokens": float(np.mean([v.tot[i] for i in ids])),
                "out_tokens": float(np.mean([v.out[i] for i in ids]))} for c, v in cells.items()}
    out = {"n_items": len(ids), "seeds": S, "acc": acc, "cost": cost, "tests": {}}
    for w in co:
        for unit in ("total_tokens", "out_tokens"):
            ok = [k for k in (2, 4, 8, 12) if cost[f"SC_k{k}"][unit] <= cost[w][unit]]
            cmp_ = f"SC_k{max(ok)}" if ok else None
            if cmp_ is None:
                out["tests"][f"{w}|{unit}"] = {"comparator": None}
                continue
            d, p, lo, hi = stats.hier_paired_test(cells[w].M, cells[cmp_].M, B=10000, seed=0)
            out["tests"][f"{w}|{unit}"] = {"comparator": cmp_, "diff": float(d), "p": float(p), "ci95": [float(lo), float(hi)]}
    return out


# ========== LaTeX
def emit(J, outdirs):
    R, I, F = J["result"], J["result"], J.get("fourbit_same_items")
    D = R["configs"]
    L = ["% AUTO-GENERATED by scripts/cloud/hardmath_thinking_analysis.py from results/extra/",
         "% hardmath_thinking_analysis.json. Do not edit."]
    nm = lambda k, v: L.append("\\newcommand{\\%s}{%s}" % (k, v))
    f1, fint = xm.f1, xm.fint
    sg = lambda x, plus="": "0.0" if round(x, 1) == 0 else xm.sg(x, plus)
    nm("HtN", str(R["n_items"]))
    nm("HtNall", str(I["n_items_levels_4_5"]))
    nm("HtNany", str(I["n_items_with_any_record"]))
    nm("HtLfour", I["level_mix"]["4"])
    nm("HtLfive", I["level_mix"]["5"])
    nm("HtPosMax", str(I["positions_in_order"]["max"] + 1))
    w100 = I["positions_in_order"]["whole_items_among_first_n"]["100"]
    nm("HtFirstHundredWhole", str(w100))
    nm("HtWholeBeyond", str(I["n_items"] - w100))
    nm("HtNextN", str(I["positions_in_order"]["max"] + 1 - 100))
    nm("HtLfourFirstHundred", I["level_mix_first_100_in_order"]["4"])
    nm("HtLfourAll", I["level_mix_all_262"]["4"])
    nm("HtLfiveAll", I["level_mix_all_262"]["5"])
    sc = I["selection_check"]["SA"]
    nm("HtSelSaWhole", f1(sc["whole"]["acc"]))
    nm("HtSelSaOther", f1(sc["not_whole"]["acc"]))
    nm("HtSelSaOtherN", str(sc["not_whole"]["n"]))
    nm("HtSelSaOutWhole", fint(sc["whole"]["out_tokens"]))
    nm("HtSelSaOutOther", fint(sc["not_whole"]["out_tokens"]))
    for c, s in (("DEBATE_N3_R1", "Deb"), ("GE_R2", "Ge"), ("HIERFIX_M3", "Hier")):
        x = I["selection_check"][c]
        nm(f"HtSel{s}Whole", f1(x["whole"]["acc"]))
        nm(f"HtSel{s}Other", f1(x["not_whole"]["acc"]))
        nm(f"HtSel{s}OtherN", str(x["not_whole"]["n"]))
    nm("HtRecords", fint(I["records_in_file"]))
    nm("HtCalls", fint(J["calls_in_file"]))
    nm("HtNoAnsSa", f1(D["SA"]["no_answer_pct"]))
    nm("HtThinkOpenSa", f1(R["truncation"]["SA"]["think_unclosed_pct"]))
    for c, s in (("SA", "Sa"), ("SC_k2", "ScTwo"), ("SC_k4", "ScFour"), ("SC_k8", "Sc"), ("DEBATE_N3_R1", "Deb"),
                 ("GE_R2", "Ge"), ("HIERFIX_M3", "Hier")):
        nm(f"HtAcc{s}", f1(D[c]["acc"]))
        nm(f"HtTok{s}", fint(D[c]["total_tokens"]))
        nm(f"HtOut{s}", fint(D[c]["out_tokens"]))
        nm(f"HtIn{s}", fint(D[c]["in_tokens"]))
        nm(f"HtAccLfour{s}", f1(J["by_level"]["4"][c]))
        nm(f"HtAccLfive{s}", f1(J["by_level"]["5"][c]))
    for w in xm.COORD:
        wl = xm.WLBL[w]
        nm(f"HtV{wl}", R["verdict_total"][w]["verdict"])
        nm(f"HtVO{wl}", R["verdict_output"][w]["verdict"])
        tm = R["token_multiple"][w]["multiple"]
        nm(f"HtTm{wl}", "n/a" if tm is None else f1(tm))
        for unit, U in (("total_tokens", "Tot"), ("out_tokens", "Out")):
            m = R["matched"][f"{w}|{unit}"]
            nm(f"HtCmp{U}{wl}", xm.CL[m["comparator"]] if m["comparator"] else "none")
            nm(f"HtD{U}{wl}", sg(m["diff"], "+") if m["diff"] is not None else "n/a")
            nm(f"HtI{U}{wl}", f"[{sg(m['ci95'][0])}, {sg(m['ci95'][1])}]" if m["ci95"] else "n/a")
            if m["comparator"]:  # discordant items: only the workflow correct / only the comparator correct
                dw, dc = J["discordant"][f"{w}|{m['comparator']}"]
                nm(f"HtDisc{U}{wl}", f"{dw} against {dc}")
        for a in ("SA", "SC_k2", "SC_k4", "SC_k8"):
            t = R["tests"][f"{w}|{a}"]
            s = {"SA": "Sa", "SC_k2": "ScTwo", "SC_k4": "ScFour", "SC_k8": "Sc"}[a]
            nm(f"HtT{wl}{s}", f"{sg(t['diff'], '+')}, [{sg(t['ci95'][0])}, {sg(t['ci95'][1])}]")
    be = R["break_even"]["SC_k8"]
    nm("HtBe", xm.rstr(be))
    nm("HtBeCI", "[" + ", ".join(r"\infty" if x == "inf" else f"{x:.2f}" for x in be["ci95"]) + "]")
    be4 = R["break_even"]["SC_k4"]
    nm("HtBeFour", xm.rstr(be4))
    nm("HtDebOverSc", f"{D['DEBATE_N3_R1']['total_tokens'] / D['SC_k8']['total_tokens']:.2f}")
    tr = R["truncation"]
    nm("HtCapMax", f1(max(v["hit_cap_pct"] for v in tr.values())))
    nm("HtCapMin", f1(min(v["hit_cap_pct"] for v in tr.values())))
    nm("HtCapSa", f1(tr["SA"]["hit_cap_pct"]))
    nm("HtCapSc", f1(tr["SC_k8"]["hit_cap_pct"]))
    nm("HtRecCapMax", f1(max(v["records_any_cap_pct"] for v in tr.values())))
    nm("HtNoAnsMax", f1(max(D[c]["no_answer_pct"] for c in xm.ALL)))
    cnt = {u: {v: sum(R[vk][w]["verdict"] == v for w in xm.COORD) for v in "RCN"}
           for u, vk in (("Tot", "verdict_total"), ("Out", "verdict_output"))}
    for U in ("Tot", "Out"):
        for v in "RCN":
            nm(f"HtN{v}{U}", xm.word(cnt[U][v]))
    # 4-bit reference: full run and the same items
    H = json.load(open(REF_JSON))
    for w, src, wl in (("DEBATE_N3_R1", "DEBATE_N3_R1", "Deb"), ("GE_R2", "GE_R2", "Ge"), ("HIERFIX_M3", "HIER_M3", "Hier")):
        if F:
            t = F["tests"][f"{w}|total_tokens"]
            nm(f"HtFbD{wl}", sg(t["diff"], "+"))
            nm(f"HtFbI{wl}", f"[{sg(t['ci95'][0])}, {sg(t['ci95'][1])}]")
            nm(f"HtFbCmp{wl}", xm.CL[t["comparator"]] if t["comparator"] in xm.CL else "SC $k{=}12$")
    if F:
        nm("HtFbSa", f1(F["acc"]["SA"]))
        nm("HtFbSc", f1(F["acc"]["SC_k8"]))
        ds = [F["tests"][f"{w}|total_tokens"]["diff"] for w in xm.COORD]
        nm("HtFbRange", f"{-max(ds):.1f} to {-min(ds):.1f}" if max(ds) < 0 else "n/a")  # points below the comparator
    for d in outdirs:
        with open(os.path.join(d, "hardmath_thinking_macros.tex"), "w") as fh:
            fh.write("\n".join(L) + "\n")

    # supplementary table: thinking run (accuracy, CI, level 4 / 5, tokens, cap) and 4-bit on the same items
    T = []
    for c in xm.ALL:
        tr_ = tr.get(c)
        cap = f1(tr_["hit_cap_pct"]) if tr_ else "--"
        fb = f"{f1(F['acc'][c])}" if F and c in F["acc"] else "--"
        T.append(f"{xm.CL[c][0].upper() + xm.CL[c][1:]} & {f1(D[c]['acc'])} & [{f1(D[c]['lo'])}, {f1(D[c]['hi'])}] & "
                 f"{f1(J['by_level']['4'][c])} & {f1(J['by_level']['5'][c])} & {fint(D[c]['total_tokens'])} & "
                 f"{fint(D[c]['out_tokens'])} & {cap} & {fb} \\\\")
    for d in outdirs:
        with open(os.path.join(d, "tables/hardmath_thinking_table.tex"), "w") as fh:
            fh.write("% AUTO-GENERATED by scripts/cloud/hardmath_thinking_analysis.py. Do not edit.\n" + "\n".join(T) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--outdir", action="append", default=None)
    ap.add_argument("--fresh", action="store_true", help="recompute even if the input files are unchanged")
    A = ap.parse_args()
    outdirs = [os.path.abspath(d) for d in A.outdir] if A.outdir else list(xm.OUTDIRS)
    sig = xm.signature(D_HARD, "math500")
    old = json.load(open(OUT_JSON)) if os.path.exists(OUT_JSON) else None
    if old and not A.fresh and old.get("input_files") == sig and "discordant" in old:
        J = old  # tests are deterministic (seed 0): unchanged inputs give the same JSON
        print("unchanged input files: reusing", os.path.relpath(OUT_JSON, ROOT))
    elif sig:
        task, recs, ids, seeds_of, info, level = plan()
        res = xm.analyse("qwen3-4b-hardmath:math500", task, recs, ids, seeds_of, info, A.jobs)
        cells = xm.build_cells(recs, ids, seeds_of, task)
        J = {"source": "results/cloud/qwen3-4b-hardmath/math500_s1.jsonl",
             "script": "scripts/cloud/hardmath_thinking_analysis.py", "definitions": __doc__,
             "model_id": recs[("SA", ids[0], 1)]["model"], "cap": recs[("SA", ids[0], 1)]["cap"],
             "input_files": sig,
             "calls_in_file": int(sum(len(r["calls"]) for r in recs.values())),
             "result": res, "by_level": by_level(cells, ids, level), "ids": ids,
             "discordant": {f"{w}|{a}": [int(sum(cells[w].M[i][0] > cells[a].M[i][0] for i in ids)),
                                         int(sum(cells[w].M[i][0] < cells[a].M[i][0] for i in ids))]
                            for w in xm.COORD for a in xm.SAMP}}
        F = fourbit_subset(ids, level)
        if F is None and os.path.exists(OUT_JSON):
            F = json.load(open(OUT_JSON)).get("fourbit_same_items")
        J["fourbit_same_items"] = F
        json.dump(J, open(OUT_JSON, "w"), indent=1, default=float)
    else:  # raw logs absent (fresh clone before tar -xJf results/cloud/hardmath_raw_logs.tar.xz)
        J = json.load(open(OUT_JSON))
        print("raw logs absent: reusing", os.path.relpath(OUT_JSON, ROOT))
    emit(J, outdirs)
    R = J["result"]
    D = R["configs"]
    print("items", R["n_items"], "levels", R["level_mix"], "positions", R["positions_in_order"])
    print(" ".join(f"{c} {D[c]['acc']:.1f}/{D[c]['total_tokens']:.0f}/{D[c]['out_tokens']:.0f}" for c in xm.ALL))
    print("verdict tot", {w: R["verdict_total"][w]["verdict"] for w in xm.COORD},
          "out", {w: R["verdict_output"][w]["verdict"] for w in xm.COORD})
    print("matched", {k: (v["comparator"], v["diff"], v["ci95"]) for k, v in R["matched"].items()})
    print("Tm", {w: R["token_multiple"][w]["multiple"] for w in xm.COORD}, "r*", R["break_even"]["SC_k8"])
    print("cap", {c: round(v["hit_cap_pct"], 1) for c, v in R["truncation"].items()})
    if J.get("fourbit_same_items"):
        print("4-bit same items", {k: (v["comparator"], round(v.get("diff", 0), 2), v.get("ci95"))
                                   for k, v in J["fourbit_same_items"]["tests"].items()})


if __name__ == "__main__":
    main()
