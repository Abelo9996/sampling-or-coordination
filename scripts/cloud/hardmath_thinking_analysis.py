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

Truncation check (cap_check; JSON keys cap_check and cap_check_thinking_l13, the latter for the four
level 1 to 3 / GSM8K thinking runs of cross_model_analysis.py on its items and seeds): share of calls at
the cap per configuration and per call position (debate round 1 / round 2; GE draft, critiques, final
revision; HIER plan, executors, aggregator), share of items and records with at least one cut call, and
whether a cut answer-bearing call ever has an extracted (and a correct) answer. Three sensitivity
analyses of every comparison behind a verdict (each cheaper sampling configuration, the cost-matched
one and SC k=8), with the R/C/N rule and the same cheaper sets: 'drop' (SC votes over its uncut samples,
debate over its uncut round-2 answers, GE falls back to its latest uncut generator output; SA and the
HIER aggregator have no alternative), 'clean' (only the items and seeds on which no call of either
compared configuration hit the cap) and 'impute' (every cut answer-bearing call scored as the gold
answer: the most a higher cap could give each configuration with its other calls unchanged; cut
round-1 debate calls and HIER executors also change later prompts, which this cannot model).

Writes results/extra/hardmath_thinking_analysis.json and, in each --outdir (default paper/eaai_rev3 if
present, else paper/arxiv_rev3): hardmath_thinking_macros.tex, tables/hardmath_thinking_table.tex and
tables/hardmath_trunc_table.tex. The truncation check of the level 1 to 3 runs takes about 30 minutes on
10 cores the first time; it is cached by input-file signature.
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


# ========== truncation check (cap 8,192 on hard MATH; 4,096 / 8,192 on the L1-3 thinking runs)
POS = {"SA": [("single call", [0])],
       "SC_k8": [("samples", list(range(8)))],
       "DEBATE_N3_R1": [("round 1 (independent)", [0, 1, 2]), ("round 2 (revisions, voted)", [3, 4, 5])],
       "GE_R2": [("draft", [0]), ("critiques", [1, 3]), ("revision 1", [2]), ("revision 2 (final)", [4])],
       "HIERFIX_M3": [("plan", [0]), ("executors", [1, 2, 3]), ("aggregator (final)", [4])]}
ANSWER_CALLS = {"SA": [0], "SC_k8": list(range(8)), "DEBATE_N3_R1": list(range(6)), "GE_R2": [0, 2, 4],
                "HIERFIX_M3": [4]}


def _recs_of(recs, c, i, ss):
    src = "SC_k8" if c.startswith("SC_k") else c
    return [recs[(src, i, 1)]] if c == "SA" else [recs[(src, i, s)] for s in ss]


def _calls(c, r):
    return r["calls"][:int(c[4:])] if c.startswith("SC_k") else r["calls"]


def drop_answer(task, c, r):
    """Final answer when truncated calls are dropped where the protocol allows it: SC votes over its
    untruncated samples, debate over its untruncated round-2 answers, GE falls back to its latest
    untruncated generator output (revision 1, then the draft); SA and the HIER aggregator have no
    alternative answer and are left as they are."""
    cs = _calls(c, r)
    if c.startswith("SC_k"):
        return topologies._majority([x["extracted"] for x in cs if not x["hit_cap"]])
    if c == "DEBATE_N3_R1":
        fin = [cs[j] for j in r["final_call_idx"]]
        return topologies._majority([x["extracted"] for x in fin if not x["hit_cap"]])
    if c == "GE_R2":
        for j in (4, 2, 0):
            if not cs[j]["hit_cap"]:
                return cs[j]["extracted"]
        return None
    return r["pred"]


def cap_check(task, recs, ids, seeds_of, res, cells, n_jobs):
    """Truncation: where the cap hits, whether a truncated call is ever right, and two sensitivity
    analyses of the paired comparisons (dropping truncated calls; items and seeds with no cut call)."""
    sc_ = lambda c, a, g: bool(task["score"](a, g))
    D = res["configs"]
    out = {"positions": {}, "records_any_cap_pct": {}, "items_any_cap_pct": {}, "final_cut_pct": {},
           "truncated_calls": {}}
    for c in ["SA"] + xm.WF:
        R = [r for i in ids for r in _recs_of(recs, c, i, seeds_of[i])]
        out["positions"][c] = {nm: {"calls": len(R) * len(js),
                                    "hit_cap_pct": 100 * float(np.mean([r["calls"][j]["hit_cap"] for r in R for j in js]))}
                               for nm, js in POS[c]}
        aj = ANSWER_CALLS[c]
        cut = [(r["calls"][j], r["gold"]) for r in R for j in aj if r["calls"][j]["hit_cap"]]
        ext = [(x, g) for x, g in cut if x["extracted"] is not None]
        out["truncated_calls"][c] = {"answer_calls": len(R) * len(aj), "cut": len(cut), "cut_with_answer": len(ext),
                                     "cut_correct": int(sum(sc_(c, x["extracted"], g) for x, g in ext))}
    for c in xm.ALL:
        R = {i: _recs_of(recs, c, i, seeds_of[i]) for i in ids}
        out["records_any_cap_pct"][c] = 100 * float(np.mean([any(x["hit_cap"] for x in _calls(c, r))
                                                            for i in ids for r in R[i]]))
        out["items_any_cap_pct"][c] = 100 * float(np.mean([any(x["hit_cap"] for r in R[i] for x in _calls(c, r))
                                                          for i in ids]))
        fin = (lambda r: _calls(c, r)) if c.startswith("SC_k") else (lambda r: [r["calls"][j] for j in r["final_call_idx"]])
        out["final_cut_pct"][c] = 100 * float(np.mean([any(x["hit_cap"] for x in fin(r)) for i in ids for r in R[i]]))
    # correct final answers in records whose answer-producing call was cut (SA, GE, HIER: one final call)
    out["final_cut_correct"] = {c: [int(sum(r["calls"][r["final_call_idx"][0]]["hit_cap"] for i in ids
                                            for r in _recs_of(recs, c, i, seeds_of[i]))),
                                    int(sum(r["calls"][r["final_call_idx"][0]]["hit_cap"] and bool(r["correct"])
                                            for i in ids for r in _recs_of(recs, c, i, seeds_of[i])))]
                                for c in ("SA", "GE_R2", "HIERFIX_M3")}
    # comparisons: every cheaper sampling configuration (verdict), the cost-matched one, and SC k=8
    pairs = {}
    for w in xm.COORD:
        S = set(res["verdict_total"][w]["cheaper"]) | set(res["verdict_output"][w]["cheaper"]) | {"SC_k8"}
        S |= {m["comparator"] for k, m in res["matched"].items() if k.startswith(w + "|") and m["comparator"]}
        pairs[w] = [a for a in xm.SAMP if a in S]
    # (1) 'drop': truncated calls dropped where the protocol allows it (drop_answer); (2) 'impute': every
    # truncated answer-bearing call (SA, SC samples, debate round 2, GE revision 2, HIER aggregator) set to
    # the gold answer, the most a higher cap could give each configuration with its other calls unchanged
    def imp_answer(c, r):
        cs, g = _calls(c, r), r["gold"]
        e = lambda x: g if x["hit_cap"] else x["extracted"]
        if c.startswith("SC_k"):
            return topologies._majority([e(x) for x in cs])
        if c == "DEBATE_N3_R1":
            return topologies._majority([e(cs[j]) for j in r["final_call_idx"]])
        return e(cs[r["final_call_idx"][0]]) if cs[r["final_call_idx"][0]]["hit_cap"] else r["pred"]
    jobs, out["identical_to_original"] = [], {}
    for v, fn in (("drop", lambda c, r: drop_answer(task, c, r)), ("impute", imp_answer)):
        M = {c: {i: np.array([float(sc_(c, fn(c, r), r["gold"])) for r in _recs_of(recs, c, i, seeds_of[i])])
                 for i in ids} for c in xm.ALL}
        M["SA"] = {i: np.repeat(M["SA"][i], len(seeds_of[i])) for i in ids}
        out[v] = {"acc": {c: 100 * float(np.mean([M[c][i].mean() for i in ids])) for c in xm.ALL}, "tests": {}}
        same = all(np.array_equal(M[c][i], cells[c].M[i]) for c in xm.ALL for i in ids)
        out["identical_to_original"][v] = same
        for w in xm.COORD:
            for a in pairs[w]:
                if same:
                    t = res["tests"][f"{w}|{a}"]
                    out[v]["tests"][f"{w}|{a}"] = {"diff": t["diff"], "ci95": t["ci95"], "p": t["p"]}
                else:
                    jobs.append((f"{v}|{w}|{a}", M[w], M[a]))
    out["acc_orig"] = {c: D[c]["acc"] for c in xm.ALL}
    # wrong records with a truncated call: the most items a higher cap could turn correct
    out["wrong_with_cut_pct"] = {c: 100 * float(np.mean([(not cells[c].M[i][k]) and any(x["hit_cap"] for x in _calls(c, r))
                                                         for i in ids for k, r in enumerate(_recs_of(recs, c, i, seeds_of[i]))]))
                                 for c in xm.ALL}
    # (3) 'clean': items and seeds where no call of either configuration hit the cap
    clean = {c: {i: [not any(x["hit_cap"] for x in _calls(c, r)) for r in _recs_of(recs, c, i, seeds_of[i])]
                 for i in ids} for c in xm.ALL}
    sub = {}
    for w in xm.COORD:
        for a in pairs[w]:
            Mw, Ma = {}, {}
            for i in ids:
                ok = [k for k in range(len(seeds_of[i])) if clean[w][i][k] and clean[a][i][0 if a == "SA" else k]]
                if ok:
                    Mw[i] = cells[w].M[i][ok]
                    Ma[i] = cells[a].M[i][ok]
            sub[f"{w}|{a}"] = (len(Mw), sum(len(v) for v in Mw.values()),
                               100 * float(np.mean([v.mean() for v in Mw.values()])),
                               100 * float(np.mean([v.mean() for v in Ma.values()])))
            jobs.append((f"clean|{w}|{a}", Mw, Ma))
    T = rr.run_tests(jobs, n_jobs)
    out["clean"] = {"tests": {}}
    for k, t in T.items():
        v, w, a = k.split("|")
        e = {"diff": t["diff"], "ci95": t["ci95"], "p": t["p"]}
        if v == "clean":
            e["n_items"], e["n_item_seeds"], e["acc_w"], e["acc_a"] = sub[f"{w}|{a}"]
        out[v]["tests"][f"{w}|{a}"] = e
    for v in ("drop", "impute", "clean"):  # the paper's R/C/N rule with the same cheaper sets
        out[v]["verdict_total"], out[v]["verdict_output"], out[v]["matched"] = {}, {}, {}
        for w in xm.COORD:
            for vk in ("verdict_total", "verdict_output"):
                ch = res[vk][w]["cheaper"]
                adv = {a: [-out[v]["tests"][f"{w}|{a}"]["ci95"][1], -out[v]["tests"][f"{w}|{a}"]["ci95"][0]] for a in ch}
                rob = [a for a in ch if adv[a][0] > 0]
                out[v][vk][w] = "R" if rob else ("N" if ch and all(adv[a][1] < 0 for a in ch) else ("N" if not ch else "C"))
            for unit in ("total_tokens", "out_tokens"):
                m = res["matched"][f"{w}|{unit}"]["comparator"]
                out[v]["matched"][f"{w}|{unit}"] = dict(out[v]["tests"][f"{w}|{m}"], comparator=m) if m else None
    return out


# ========== LaTeX
def emit(J, outdirs):
    R, I, F = J["result"], J["result"], J.get("fourbit_same_items")
    XM = json.load(open(xm.OUT_JSON))["models"]
    XV = {f"{d}:{tn}": XM[d]["tasks"][tn] for d in XM for tn in XM[d]["tasks"]}
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
    # truncation check (hard MATH; then the level 1 to 3 / GSM8K thinking runs)
    K = J.get("cap_check")
    if K:
        P = K["positions"]
        nm("HtCapDeb", f1(tr["DEBATE_N3_R1"]["hit_cap_pct"]))
        nm("HtCapGe", f1(tr["GE_R2"]["hit_cap_pct"]))
        nm("HtCapHier", f1(tr["HIERFIX_M3"]["hit_cap_pct"]))
        nm("HtCapDebOne", f1(P["DEBATE_N3_R1"]["round 1 (independent)"]["hit_cap_pct"]))
        nm("HtCapDebTwo", f1(P["DEBATE_N3_R1"]["round 2 (revisions, voted)"]["hit_cap_pct"]))
        nm("HtCapGeDraft", f1(P["GE_R2"]["draft"]["hit_cap_pct"]))
        nm("HtCapGeCrit", f1(P["GE_R2"]["critiques"]["hit_cap_pct"]))
        nm("HtCapGeFinal", f1(P["GE_R2"]["revision 2 (final)"]["hit_cap_pct"]))
        nm("HtCapHierPlan", f1(P["HIERFIX_M3"]["plan"]["hit_cap_pct"]))
        nm("HtCapHierExec", f1(P["HIERFIX_M3"]["executors"]["hit_cap_pct"]))
        nm("HtCapHierAgg", f1(P["HIERFIX_M3"]["aggregator (final)"]["hit_cap_pct"]))
        for c, s_ in (("SA", "Sa"), ("SC_k2", "ScTwo"), ("SC_k4", "ScFour"), ("SC_k8", "Sc"), ("DEBATE_N3_R1", "Deb"),
                      ("GE_R2", "Ge"), ("HIERFIX_M3", "Hier")):
            nm(f"HtItemsCut{s_}", f1(K["items_any_cap_pct"][c]))
            nm(f"HtWrongCut{s_}", f1(K["wrong_with_cut_pct"][c]))
            nm(f"HtImpAcc{s_}", f1(K["impute"]["acc"][c]))
            nm(f"HtImpGain{s_}", sg(K["impute"]["acc"][c] - K["acc_orig"][c], "+"))
        T_ = K["truncated_calls"]
        nm("HtCutCalls", fint(sum(v["cut"] for v in T_.values())))
        nm("HtCutAnswer", fint(sum(v["cut_with_answer"] for v in T_.values())))
        nm("HtCutCorrect", fint(sum(v["cut_correct"] for v in T_.values())))
        nm("HtCutSc", fint(T_["SC_k8"]["cut"]))
        nm("HtCutDeb", fint(T_["DEBATE_N3_R1"]["cut"]))
        nm("HtDropSame", "yes" if K["identical_to_original"]["drop"] else "no")
        for v, V in (("clean", "Clean"), ("impute", "Imp")):
            for w in xm.COORD:
                wl = xm.WLBL[w]
                m = K[v]["matched"][f"{w}|total_tokens"]
                nm(f"Ht{V}D{wl}", sg(m["diff"], "+"))
                nm(f"Ht{V}I{wl}", f"[{sg(m['ci95'][0])}, {sg(m['ci95'][1])}]")
                t = K[v]["tests"][f"{w}|SC_k8"]
                nm(f"Ht{V}Sc{wl}", f"{sg(t['diff'], '+')}, [{sg(t['ci95'][0])}, {sg(t['ci95'][1])}]")
                if v == "clean":
                    nm(f"HtCleanN{wl}", str(m["n_items"]))
                    nm(f"HtCleanScN{wl}", str(t["n_items"]))
            vv = [K[v][vk][w] for vk in ("verdict_total", "verdict_output") for w in xm.COORD]
            nm(f"Ht{V}Verdicts", "every verdict C" if set(vv) == {"C"} else
               "verdicts " + ", ".join(vv[:3]) + " (total) and " + ", ".join(vv[3:]) + " (output)")
        nm("HtCleanMaxAbs", f1(max(abs(t["diff"]) for t in K["clean"]["tests"].values())))
        g = [K["impute"]["acc"][w] - K["acc_orig"][w] for w in xm.COORD]
        nm("HtImpGainScFourInt", f"{K['impute']['acc']['SC_k4'] - K['acc_orig']['SC_k4']:.0f}")
        nm("HtImpWfRange", f"{min(g):.0f} to {max(g):.0f}" if round(min(g)) != round(max(g)) else f"{g[0]:.0f}")
        nm("HtCleanScN", f"{min(K['clean']['tests'][w + '|SC_k8']['n_items'] for w in xm.COORD)} to "
                         f"{max(K['clean']['tests'][w + '|SC_k8']['n_items'] for w in xm.COORD)}")
        cn = [K["clean"]["matched"][f"{w}|total_tokens"]["n_items"] for w in xm.COORD]
        nm("HtCleanRange", f"{min(cn)} to {max(cn)}" if min(cn) != max(cn) else str(cn[0]))
    L13 = J.get("cap_check_thinking_l13", {})
    if L13:
        cut = sum(v["cut"] for k in L13.values() for v in k["truncated_calls"].values())
        ans = sum(v["cut_with_answer"] for k in L13.values() for v in k["truncated_calls"].values())
        cor = sum(v["cut_correct"] for k in L13.values() for v in k["truncated_calls"].values())
        nm("TcCutCalls", fint(cut))
        nm("TcCutAnswer", fint(ans))
        nm("TcCutCorrect", fint(cor))
        nm("TcRuns", xm.word(len(L13)))
        ch = {v: sum(k[v][vk][w] != XV[key][vk][w]["verdict"] for key, k in L13.items()
                     for vk in ("verdict_total", "verdict_output") for w in xm.COORD)
              for v in ("drop", "clean", "impute")}
        for v, V in (("drop", "Drop"), ("clean", "Clean"), ("impute", "Imp")):
            nm(f"Tc{V}Changed", xm.word(ch[v]))
        nv = 2 * len(xm.COORD) * len(L13)
        nm("TcNVerdicts", str(nv))
        moves = sorted({f"{XV[key][vk][w]['verdict']} to {k[v][vk][w]}" for key, k in L13.items()
                        for v in ("drop", "clean", "impute") for vk in ("verdict_total", "verdict_output")
                        for w in xm.COORD if k[v][vk][w] != XV[key][vk][w]["verdict"]})
        wd = lambda n: "none" if n == 0 else xm.word(n)
        nm("TcChangedText", f"changed none of the {nv} verdicts" if not any(ch.values()) else
           f"changed {wd(ch['drop'])}, {wd(ch['clean'])} and {wd(ch['impute'])} of the {nv} verdicts"
           + (f", each from {moves[0]}" if len(moves) == 1 else " (" + ", ".join(moves) + ")"))
        dmax = max(abs(k[v]["matched"][f"{w}|{u}"]["diff"] - XV[key]["matched"][f"{w}|{u}"]["diff"])
                   for key, k in L13.items() for v in ("drop", "clean") for w in xm.COORD
                   for u in ("total_tokens", "out_tokens") if k[v]["matched"][f"{w}|{u}"])
        nm("TcMaxShift", f1(dmax))
        imp = [k["impute"]["acc"][c] - k["acc_orig"][c] for k in L13.values() for c in xm.ALL]
        nm("TcImpGainMax", f1(max(imp)))
        sc8 = max(k["impute"]["acc"]["SC_k8"] - k["acc_orig"]["SC_k8"] for k in L13.values())
        deb = max(k["impute"]["acc"]["DEBATE_N3_R1"] - k["acc_orig"]["DEBATE_N3_R1"] for k in L13.values())
        nm("TcImpGainSc", f1(sc8))
        nm("TcImpGainDeb", f1(deb))
    for d in outdirs:
        with open(os.path.join(d, "hardmath_thinking_macros.tex"), "w") as fh:
            fh.write("\n".join(L) + "\n")

    # supplementary table: thinking run (accuracy, CI, level 4 / 5, tokens, cap) and 4-bit on the same items
    T = []
    for c in xm.ALL:
        tr_ = tr.get(c)
        cap = f1(tr_["hit_cap_pct"]) if tr_ else "--"
        fb = f"{f1(F['acc'][c])}" if F and c in F["acc"] else "--"
        # same style as the main tables: lowercase workflow names, interval inline in brackets
        T.append(f"{xm.CL[c]} & {f1(D[c]['acc'])} {{\\footnotesize[{f1(D[c]['lo'])},{f1(D[c]['hi'])}]}} & "
                 f"{f1(J['by_level']['4'][c])} & {f1(J['by_level']['5'][c])} & {fint(D[c]['total_tokens'])} & "
                 f"{fint(D[c]['out_tokens'])} & {cap} & {fb} \\\\")
    for d in outdirs:
        with open(os.path.join(d, "tables/hardmath_thinking_table.tex"), "w") as fh:
            fh.write("% AUTO-GENERATED by scripts/cloud/hardmath_thinking_analysis.py. Do not edit.\n" + "\n".join(T) + "\n")


def emit_trunc_table(J, outdirs):
    """Supplementary truncation table: per configuration, calls cut (all and by position), records with a cut
    call, wrong records with a cut call, accuracy as run and with every cut answer-bearing call scored correct."""
    K = J.get("cap_check")
    if not K:
        return
    f1 = xm.f1
    pos = {"SA": "--", "SC_k2": "--", "SC_k4": "--", "SC_k8": "--"}
    P = K["positions"]
    pos["DEBATE_N3_R1"] = "R1 %s, R2 %s" % (f1(P["DEBATE_N3_R1"]["round 1 (independent)"]["hit_cap_pct"]),
                                          f1(P["DEBATE_N3_R1"]["round 2 (revisions, voted)"]["hit_cap_pct"]))
    pos["GE_R2"] = "D %s, C %s, final %s" % (f1(P["GE_R2"]["draft"]["hit_cap_pct"]), f1(P["GE_R2"]["critiques"]["hit_cap_pct"]),
                                           f1(P["GE_R2"]["revision 2 (final)"]["hit_cap_pct"]))
    pos["HIERFIX_M3"] = "P %s, E %s, A %s" % (f1(P["HIERFIX_M3"]["plan"]["hit_cap_pct"]),
                                            f1(P["HIERFIX_M3"]["executors"]["hit_cap_pct"]),
                                            f1(P["HIERFIX_M3"]["aggregator (final)"]["hit_cap_pct"]))
    tr = J["result"]["truncation"]
    T = []
    for c in xm.ALL:
        allc = f1(tr[c]["hit_cap_pct"]) if c in tr else "--"
        T.append(f"{xm.CL[c]} & {allc} & {pos[c]} & {f1(K['items_any_cap_pct'][c])} & "
                 f"{f1(K['wrong_with_cut_pct'][c])} & {f1(K['acc_orig'][c])} & {f1(K['impute']['acc'][c])} \\\\")
    for d in outdirs:
        with open(os.path.join(d, "tables/hardmath_trunc_table.tex"), "w") as fh:
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
    # truncation check: hard MATH, then the level 1 to 3 / GSM8K thinking runs of cross_model_analysis
    # (cached by input-file signature; kept from the JSON when the raw logs are absent)
    if sig and (A.fresh or J.get("cap_check", {}).get("input_files") != sig):
        task, recs, ids, seeds_of, info, level = plan()
        cells = xm.build_cells(recs, ids, seeds_of, task)
        J["cap_check"] = cap_check(task, recs, ids, seeds_of, J["result"], cells, A.jobs)
        J["cap_check"]["input_files"] = sig
    XJ = json.load(open(xm.OUT_JSON))["models"]
    L13 = J.setdefault("cap_check_thinking_l13", {})
    for d in ("qwen3-4b", "qwen3-8b"):
        for tn, _, _, nexp in xm.TASKS:
            path, key = os.path.join(ROOT, "results/cloud", d), f"{d}:{tn}"
            s13 = xm.signature(path, tn)
            if not s13 or (not A.fresh and L13.get(key, {}).get("input_files") == s13):
                continue
            task, recs, ids, seeds_of, info = xm.plan(path, tn, nexp)
            cells = xm.build_cells(recs, ids, seeds_of, task)
            print("truncation check", key, flush=True)
            L13[key] = cap_check(task, recs, ids, seeds_of, XJ[d]["tasks"][tn], cells, A.jobs)
            L13[key]["input_files"] = s13
            L13[key]["cap"] = next(iter(recs.values()))["cap"]
    json.dump(J, open(OUT_JSON, "w"), indent=1, default=float)
    emit(J, outdirs)
    emit_trunc_table(J, outdirs)
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
