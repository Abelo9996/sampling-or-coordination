"""Cross-model replication: the paper's analysis on every model present under results/cloud/.

Models (a model is analysed when its directory holds task logs; nothing else changes when the
Kaggle runs drop in, rerun this script and rebuild paper/eaai_rev3):
  results/cloud/qwen3-4b   Qwen/Qwen3-4B, thinking on      (Modal H100, vLLM bf16)
  results/cloud/qwen3-8b   Qwen/Qwen3-8B, thinking on      (Modal H100, vLLM bf16)
  results/cloud/phi4-mini  microsoft/Phi-4-mini-instruct    (Kaggle T4, vLLM fp16)
  results/cloud/qwen25-3b  Qwen/Qwen2.5-3B-Instruct         (Kaggle T4, vLLM fp16)
Tasks: GSM8K first 500 test items; MATH-500 levels 1 to 3 (238 items). Workflows SA (greedy,
seed 1), SC k=8 (k=2, 4 = its first 2, 4 calls, voted with the harness rule), DEBATE_N3_R1, GE_R2,
HIERFIX_M3 (the HIER of the paper's main text).

Which records are used. A seed s of item i is used only when every workflow record (SC_k8,
DEBATE_N3_R1, GE_R2, HIERFIX_M3) exists for (i, s), and SA (i, 1) exists; so every comparison is
paired on the same (item, seed) sets. Runs stopped by the budget guard therefore contribute the
seeds that finished for every workflow of an item. When the seed sets differ between items a
seed-1-only sensitivity analysis is added (none since 24 Nov 2026: Qwen3-8B has seeds 1 and 2 complete
on both tasks after its top-up run, the other models seeds 1 to 3).

Statistics (as in the paper, scripts/analyze_rev3_reruns.py functions called unchanged where the
seed sets are uniform):
  accuracy    mean over items of the per-item mean over seeds; 95% item bootstrap (10,000
              resamples, numpy seed 0, items in sorted id order = harness.analyze.item_bootstrap_ci)
  tests       harness.stats.hier_paired_test via scripts/analyze_hard_math.test (items resampled,
              seeds within items, 10,000 replicates, seed 0); SA's single greedy value repeated
              over the item's seeds, as in the paper
  verdict     the paper's R/C/N rule (rr.finish_task), in total tokens and in output tokens:
              among the sampling configurations (SA, SC k=2, 4, 8) cheaper than the workflow,
              R if one is detectably more accurate (95% interval of its advantage above zero),
              N if every cheaper one is detectably less accurate (or none is cheaper), C otherwise
  token multiple  tokens of the workflow / tokens of the cheapest SC (k>=2) whose point accuracy
              is at least the workflow's (paper/arxiv_rev3/scripts/emit_token_multiple.py);
              undefined (null) when no SC configuration is that accurate
  break-even  r* of SC k=8 against debate (scripts/analyze_rev3_costs.crossover): cost = input +
              r x output tokens per query; SC k=8 is cheaper iff r < r* = (in_deb - in_SC8) /
              (out_SC8 - out_deb); r* = inf when SC k=8 generates no more output than debate.
              Bootstrap interval: the paper's function with uniform seed sets; with mixed seed
              sets the same scheme (items, then seeds within item, independently per
              configuration, 10,000 replicates, seed 0) in a loop, whose random draws differ.
  truncation  per configuration: % of calls that hit the token cap (hit_cap), % of calls whose
              think block never closed (thinking models), % of records with no extracted answer.

Deviations from the paper: none in the tests; SC k=12 was not run on MATH (comparators k <= 8).

Writes results/extra/cross_model_analysis.json and, in each --outdir (default paper/eaai_rev3 if present,
else paper/arxiv_rev3):
cross_model_macros.tex, tables/cross_model_table.tex (main) and tables/cross_model_supp_{gsm8k,math500}.tex
(CIs, tokens, truncation).
Run: /usr/bin/python3 scripts/cloud/cross_model_analysis.py [--jobs 8] [--fresh] [--outdir DIR ...]
  (from the repo root), then: sh paper/eaai_rev3/build.sh. The arXiv build (paper/arxiv_rev3/scripts/
  build.sh) runs it with --outdir paper/arxiv_rev3, so new models enter both versions.
A (model, task) whose input files (names and sizes) are unchanged is reused from the JSON (the
tests are deterministic); --fresh recomputes everything (about 30 min for the two Qwen3 models on
8 cores). Raw Qwen3 logs are committed compressed: tar -xJf results/cloud/qwen3_raw_logs.tar.xz -C
results/cloud. New models (Kaggle: unzip results_nonthinking.zip into results/cloud/) appear in
every macro and table on the next run (all four models present since 24 Nov 2026); the abstract,
highlights and arXiv takeaways are hand-written, so re-read them if the printed headline counts change.
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from harness import tasks, topologies  # noqa: E402


def _mod(name, path):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, path))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m  # forked test workers unpickle rr._job by module name
    spec.loader.exec_module(m)
    return m


rr = _mod("rr", "scripts/analyze_rev3_reruns.py")
rc = _mod("rc", "scripts/analyze_rev3_costs.py")

OUT_JSON = os.path.join(ROOT, "results/extra/cross_model_analysis.json")
PAPER_IN = os.path.join(ROOT, "paper/arxiv_rev3")  # input tables/math_easy.json (same file in paper/eaai_rev3)
EAAI = os.path.join(ROOT, "paper/eaai_rev3")
# paper directories that receive the macros and tables (--outdir, repeatable); default paper/eaai_rev3
# where it exists (the manuscript repository), else paper/arxiv_rev3 (the public release)
OUTDIRS = [EAAI if os.path.isdir(EAAI) else PAPER_IN]


def write_out(rel, text):
    for d in OUTDIRS:
        with open(os.path.join(d, rel), "w") as fh:
            fh.write(text)
SAMP = ["SA", "SC_k2", "SC_k4", "SC_k8"]
COORD = ["DEBATE_N3_R1", "GE_R2", "HIERFIX_M3"]
WF = ["SC_k8"] + COORD  # records that must all exist for a seed to be used
ALL = SAMP + COORD
MODELS = [  # dir, label, macro key, mode
    ("qwen3-4b", "Qwen3-4B", "QwFour", "thinking"),
    ("qwen3-8b", "Qwen3-8B", "QwEight", "thinking"),
    ("phi4-mini", "Phi-4-mini", "Phi", "non-thinking"),
    ("qwen25-3b", "Qwen2.5-3B fp16", "QwThree", "non-thinking"),
]
TASKS = [("gsm8k", "GSM8K", "Gsm", 500), ("math500", "MATH L1-3", "Math", 238)]
WLBL = {"DEBATE_N3_R1": "Deb", "GE_R2": "Ge", "HIERFIX_M3": "Hier"}


# ========== cells with per-item seed sets
def build_cells(recs, ids, seeds_of, task):
    cells = {c: rr.Cell() for c in ALL}
    for i in ids:
        ss = seeds_of[i]
        sa = recs[("SA", i, 1)]
        c = cells["SA"]
        c.M[i] = np.array([float(bool(sa["correct"]))] * len(ss))
        t, o = rr.rec_cost(sa)
        c.tot[i], c.out[i] = float(t), float(o)
        c.none.append(sa["pred"] is None)
        for cfg in WF:
            R = [recs[(cfg, i, s)] for s in ss]
            c = cells[cfg]
            c.M[i] = np.array([float(bool(r["correct"])) for r in R])
            cc = [rr.rec_cost(r) for r in R]
            c.tot[i] = float(np.mean([x[0] for x in cc]))
            c.out[i] = float(np.mean([x[1] for x in cc]))
            c.none += [r["pred"] is None for r in R]
        for k in (2, 4):  # rr.sc_prefix_cells, per-item seeds
            v, tt, oo = [], [], []
            c = cells[f"SC_k{k}"]
            for s in ss:
                r = recs[("SC_k8", i, s)]
                cs = r["calls"][:k]
                ans = topologies._majority([x["extracted"] for x in cs])
                v.append(float(bool(task["score"](ans, r["gold"]))))
                a, b = rr.rec_cost(r, cs)
                tt.append(a)
                oo.append(b)
                c.none.append(ans is None)
            c.M[i] = np.array(v)
            c.tot[i] = float(np.mean(tt))
            c.out[i] = float(np.mean(oo))
    return cells


def check_against_rr(cells, recs, ids, seeds, task):
    """With uniform seeds, the cells equal those of the paper's functions (rr.cell_from etc.)."""
    ref = {"SA": rr.cell_from(recs, "SA", ids, seeds)}
    ref.update({c: rr.cell_from(recs, c, ids, seeds) for c in WF})
    ref.update(rr.sc_prefix_cells(recs, "SC_k8", ids, seeds, task, [2, 4]))
    for c in ALL:
        for i in ids:
            assert np.array_equal(ref[c].M[i], cells[c].M[i]), (c, i)
            assert ref[c].tot[i] == cells[c].tot[i] and ref[c].out[i] == cells[c].out[i], (c, i)
        assert ref[c].none == cells[c].none, c


# ========== cost per (item, seed): (in, out, calls)
def cost_arrays(recs, cfg, ids, seeds_of, k=None):
    src = "SC_k8" if k else cfg
    out = {}
    for i in ids:
        ss = [1] if cfg == "SA" else seeds_of[i]
        out[i] = np.array([rc.rec_cost(recs[(src, i, s)], k) for s in ss], float)
    return out


def crossover_mixed(A, Bc, ids, seed=0, B=10000):
    """rc.crossover's scheme for item-dependent seed counts (loop; random draws differ)."""
    def rstar(din, dout):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(dout > 0, din / dout, np.inf)
    a = [A[i] for i in ids]
    b = [Bc[i] for i in ids]
    ma = np.mean([x.mean(axis=0) for x in a], axis=0)
    mb = np.mean([x.mean(axis=0) for x in b], axis=0)
    point = float(rstar(mb[0] - ma[0], ma[1] - mb[1]))
    rng = np.random.default_rng(seed)
    n = len(ids)
    rs = np.empty(B)
    for t in range(B):
        idx = rng.integers(0, n, size=n)
        sa = np.zeros(3)
        sb = np.zeros(3)
        for ii in idx:
            x, y = a[ii], b[ii]
            sa += x[rng.integers(0, len(x), size=len(x))].mean(axis=0)
            sb += y[rng.integers(0, len(y), size=len(y))].mean(axis=0)
        sa /= n
        sb /= n
        rs[t] = rstar(sb[0] - sa[0], sa[1] - sb[1])
    share_inf = float(np.mean(np.isinf(rs)))
    if share_inf > 0.025:
        ci = [float(np.percentile(rs, 2.5)) if share_inf < 0.975 else "inf", "inf"]
    else:
        lo, hi = np.percentile(rs, [2.5, 97.5])
        ci = [float(lo), float(hi)]
    return {"point": point if np.isfinite(point) else "inf", "ci95": ci, "share_infinite": share_inf,
            "d_in_debate_minus_sc": float(mb[0] - ma[0]), "d_out_sc_minus_debate": float(ma[1] - mb[1]),
            "method": "loop (mixed seed sets)"}


def token_multiple(D, w):
    ok = [k for k in ("SC_k2", "SC_k4", "SC_k8") if D[k]["acc"] >= D[w]["acc"]]
    if not ok:
        return {"multiple": None, "comparator": None, "note": "no SC configuration at least as accurate"}
    k = min(ok, key=lambda x: D[x]["total_tokens"])
    return {"multiple": D[w]["total_tokens"] / D[k]["total_tokens"], "comparator": k,
            "multiple_output": D[w]["out_tokens"] / D[k]["out_tokens"]}


def truncation(recs, ids, seeds_of):
    out = {}
    for cfg in ["SA"] + WF:
        R = [recs[(cfg, i, s)] for i in ids for s in ([1] if cfg == "SA" else seeds_of[i])]
        C = [c for r in R for c in r["calls"]]
        out[cfg] = {"calls": len(C),
                    "hit_cap_pct": 100 * float(np.mean([c["hit_cap"] for c in C])),
                    "think_unclosed_pct": 100 * float(np.mean([c.get("think_closed") is False for c in C])),
                    "records_any_cap_pct": 100 * float(np.mean([any(c["hit_cap"] for c in r["calls"]) for r in R])),
                    "no_answer_pct": 100 * float(np.mean([r["pred"] is None for r in R]))}
    return out


# ========== one (model, task)
def plan(d, tn, nexp, seed_filter=None):
    task = tasks.get_task(tn)
    ids_exp = sorted(it["id"] for it in task["load"](n=nexp, split="all"))
    recs = rr.load([os.path.relpath(p, ROOT) for p in sorted(
        __import__("glob").glob(os.path.join(d, f"{tn}_s*.jsonl")))])
    if not recs:
        return None
    extra = {k[1] for k in recs} - set(ids_exp)
    assert not extra, (d, tn, "items outside the expected set", sorted(extra)[:5])
    seeds = sorted({k[2] for k in recs})
    if seed_filter:
        seeds = [s for s in seeds if s in seed_filter]
    seeds_of = {i: [s for s in seeds if all((c, i, s) in recs for c in WF)] for i in ids_exp}
    ids = [i for i in ids_exp if seeds_of[i] and ("SA", i, 1) in recs]
    info = {"n_items_expected": nexp, "n_items": len(ids),
            "items_excluded": [i for i in ids_exp if i not in ids],
            "seed_items": {str(s): sum(s in seeds_of[i] for i in ids) for s in seeds},
            "seeds_present_in_files": sorted({k[2] for k in recs}),
            "records_unused": sum(1 for k in recs if k[0] != "SA" and (k[1] not in ids or k[2] not in seeds_of.get(k[1], [])))}
    info["uniform_seeds"] = len({tuple(seeds_of[i]) for i in ids}) == 1
    return task, recs, ids, {i: seeds_of[i] for i in ids}, info


def analyse(key, task, recs, ids, seeds_of, info, n_jobs):
    cells = build_cells(recs, ids, seeds_of, task)
    if info["uniform_seeds"]:
        check_against_rr(cells, recs, ids, seeds_of[ids[0]], task)
    T = rr.run_tests(rr.plan_task(key, cells, ids, SAMP, COORD), n_jobs)
    res = rr.finish_task(key, cells, ids, SAMP, COORD, T, [2, 4, 8])
    res.update(info)
    D = res["configs"]
    for c in ALL:
        D[c]["in_tokens"] = D[c]["total_tokens"] - D[c]["out_tokens"]
        D[c]["out_share_pct"] = 100 * D[c]["out_tokens"] / D[c]["total_tokens"]
    res["token_multiple"] = {w: token_multiple(D, w) for w in COORD}
    # cost-matched comparator in each unit: SC with the largest k whose mean cost <= the workflow's;
    # a workflow 'wins at matched cost' if its 95% interval against that comparator is above zero
    res["matched"] = {}
    for w in COORD:
        for unit in ("total_tokens", "out_tokens"):
            ok = [k for k in (2, 4, 8) if D[f"SC_k{k}"][unit] <= D[w][unit]]
            cmp_ = f"SC_k{max(ok)}" if ok else None
            t = res["tests"][f"{w}|{cmp_}"] if cmp_ else None
            res["matched"][f"{w}|{unit}"] = {"comparator": cmp_, "diff": t["diff"] if t else None,
                                             "ci95": t["ci95"] if t else None,
                                             "win": bool(t and t["ci95"][0] > 0)}
    sc8 = cost_arrays(recs, "SC_k8", ids, seeds_of)
    deb = cost_arrays(recs, "DEBATE_N3_R1", ids, seeds_of)
    sc4 = cost_arrays(recs, "SC_k4", ids, seeds_of, k=4)
    if info["uniform_seeds"]:
        res["break_even"] = {"SC_k8": rc.crossover(sc8, deb, ids), "SC_k4": rc.crossover(sc4, deb, ids)}
        for v in res["break_even"].values():
            v["method"] = "scripts/analyze_rev3_costs.crossover"
    else:
        res["break_even"] = {"SC_k8": crossover_mixed(sc8, deb, ids), "SC_k4": crossover_mixed(sc4, deb, ids)}
    res["truncation"] = truncation(recs, ids, seeds_of)
    return res


# ========== LaTeX
CL = {"SA": "SA", "SC_k2": "SC $k{=}2$", "SC_k4": "SC $k{=}4$", "SC_k8": "SC $k{=}8$",
      "DEBATE_N3_R1": "debate", "GE_R2": "GE", "HIERFIX_M3": "HIER"}
WORDS = ["no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"]


def f1(x):
    return f"{x:.1f}"


def fint(x):
    return f"{x:,.0f}".replace(",", "{,}")


def sg(x, plus=""):
    """signed number for running text, with a math minus"""
    t = f"{x:{plus}.1f}"
    return t.replace("-", "$-$")


def rstr(v):
    return r"$\infty$" if v["point"] == "inf" else f"{v['point']:.2f}"


def andlist(xs):
    xs = list(xs)
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]


def rng(xs, fmt="{:.0f}"):
    lo, hi = fmt.format(min(xs)), fmt.format(max(xs))
    return lo if lo == hi else f"{lo} to {hi}"


def word(n):
    return WORDS[n] if n < len(WORDS) else str(n)


def cells_of(J):
    for d, lbl, mk, mode in MODELS:
        if d not in J["models"]:
            continue
        for tn, tl, tk, _ in TASKS:
            if tn in J["models"][d]["tasks"]:
                yield d, lbl, mk, mode, tn, tl, tk, J["models"][d]["tasks"][tn]


def headline(J):
    """The paper's headline rule: a coordination workflow 'beats sampling at equal or lower cost' when
    some sampling configuration is cheaper and the workflow is detectably more accurate than every
    cheaper one (verdict N with a non-empty cheaper set). Also listed: the workflow is detectably more
    accurate than its cost-fixed SC comparator (largest k not costlier), a weaker condition."""
    H = {"cells": 0, "N": {"total": [], "output": []}, "R": {"total": [], "output": []},
         "matched_win": {"total": [], "output": []}}
    for d, lbl, mk, mode, tn, tl, tk, R in cells_of(J):
        for w in COORD:
            H["cells"] += 1
            for u, vk, mu in (("total", "verdict_total", "total_tokens"), ("output", "verdict_output", "out_tokens")):
                v = R[vk][w]
                tag = {"model": lbl, "task": tl, "workflow": w, "cheaper": v["cheaper"]}
                if v["verdict"] == "N" and v["cheaper"]:
                    tag["gains"] = {a: R["tests"][f"{w}|{a}"] for a in v["cheaper"]}
                    H["N"][u].append(tag)
                if v["verdict"] == "R":
                    H["R"][u].append(tag)
                m = R["matched"][f"{w}|{mu}"]
                if m["win"]:
                    H["matched_win"][u].append(dict(tag, comparator=m["comparator"], diff=m["diff"], ci95=m["ci95"]))
    return H


def emit(J):
    present = [m for m in MODELS if m[0] in J["models"]]
    L = ["% AUTO-GENERATED by scripts/cloud/cross_model_analysis.py from results/extra/",
         "% cross_model_analysis.json. Do not edit; rerun the script when more models finish."]
    def nm(name, val):
        L.append("\\newcommand{\\%s}{%s}" % (name, val))
    thinking = [m for m in present if m[3] == "thinking"]
    nm("XmNmodels", word(len(present)))
    nm("XmModels", andlist([m[1] for m in present]))
    nm("XmNthinking", word(len(thinking)))
    nm("XmThinkingModels", andlist([m[1] for m in thinking]) if thinking else "none")
    for d, lbl, mk, mode, tn, tl, tk, R in cells_of(J):
        p = f"Xm{mk}{tk}"
        D = R["configs"]
        for c, s in (("SA", "Sa"), ("SC_k4", "ScFour"), ("SC_k8", "Sc"), ("DEBATE_N3_R1", "Deb"), ("GE_R2", "Ge"),
                     ("HIERFIX_M3", "Hier")):
            nm(f"{p}Acc{s}", f1(D[c]["acc"]))
            nm(f"{p}Tok{s}", fint(D[c]["total_tokens"]))
            nm(f"{p}Out{s}", fint(D[c]["out_tokens"]))
            nm(f"{p}Share{s}", f"{D[c]['out_share_pct']:.0f}")
        nm(f"{p}Items", str(R["n_items"]))
        for w in COORD:
            wl = WLBL[w]
            nm(f"{p}V{wl}", R["verdict_total"][w]["verdict"])
            nm(f"{p}VO{wl}", R["verdict_output"][w]["verdict"])
            tm = R["token_multiple"][w]["multiple"]
            nm(f"{p}Tm{wl}", "n/a" if tm is None else f1(tm))
            for a, s in (("SA", "Sa"), ("SC_k2", "ScTwo"), ("SC_k4", "ScFour"), ("SC_k8", "Sc")):
                t = R["tests"][f"{w}|{a}"]  # workflow minus sampling configuration, points, 95% interval
                nm(f"{p}T{wl}{s}", f"{sg(t['diff'], '+')}, [{sg(t['ci95'][0])}, {sg(t['ci95'][1])}]")
        be = R["break_even"]["SC_k8"]
        nm(f"{p}Be", rstr(be))
        nm(f"{p}BeCI", "[" + ", ".join(r"\infty" if x == "inf" else f"{x:.2f}" for x in be["ci95"]) + "]")
    H = headline(J)
    J["headline"] = H
    nm("XmNcells", str(H["cells"]))
    nm("XmNcellsWord", word(H["cells"]))
    for u, U in (("total", "Tot"), ("output", "Out")):
        nm(f"XmNwin{U}", word(len(H["N"][u])))
        nm(f"XmNR{U}", word(len(H["R"][u])))
        nm(f"XmNmatched{U}", word(len(H["matched_win"][u])))
        txt = []
        for e in H["N"][u]:
            g = "; ".join(f"{CL[a]} {sg(e['gains'][a]['diff'], '+')} [{sg(e['gains'][a]['ci95'][0])}, "
                          f"{sg(e['gains'][a]['ci95'][1])}]" for a in e["cheaper"])
            txt.append(f"{CL[e['workflow']]} on {e['model']}, {e['task']} (against {g})")
        nm(f"XmWin{U}Text", "none" if not txt else andlist(txt))
        txt = [f"{CL[e['workflow']]} on {e['model']}, {e['task']} (against {CL[e['comparator']]}, "
               f"{sg(e['diff'], '+')} [{sg(e['ci95'][0])}, {sg(e['ci95'][1])}])" for e in H["matched_win"][u]]
        nm(f"XmMatched{U}Text", "none" if not txt else andlist(txt))
    # headline sentences (generated, so they stay true when models are added)
    cnt = {u: {"R": 0, "C": 0, "N": 0} for u in ("total", "output")}
    for d, lbl, mk, mode, tn, tl, tk, R in cells_of(J):
        for w in COORD:
            cnt["total"][R["verdict_total"][w]["verdict"]] += 1
            cnt["output"][R["verdict_output"][w]["verdict"]] += 1
    for u, U, unit in (("total", "Tot", "total-token"), ("output", "Out", "output-token")):
        c = cnt[u]
        vs = ", ".join(f"{c[v]} {v}" for v in ("R", "C", "N") if c[v])
        nm(f"XmVerdicts{U}", vs)
        if not H["N"][u]:
            t = (f"under {unit} cost, no coordination workflow was detectably more accurate than every sampling "
                 f"configuration of lower cost, in any of the {H['cells']} workflow, model and task combinations "
                 f"(verdicts: {vs})")
        else:
            t = (f"under {unit} cost, {word(len(H['N'][u]))} of the {H['cells']} workflow, model and task combinations "
                 f"had a coordination workflow detectably more accurate than every sampling configuration of lower "
                 f"cost (verdicts: {vs})")
        nm(f"XmHead{U}", t)
        nN = len(H["N"][u])
        nm(f"XmHead{U}Short", f"no coordination workflow detectably more accurate than every cheaper sampling "
                              f"configuration under {unit} cost" if not nN else
           f"a coordination workflow detectably more accurate than every cheaper sampling configuration in "
           f"{word(nN)} of the {H['cells']} combinations under {unit} cost")
        if H["N"][u]:
            t = (f"Under {unit} cost the exception{'s are' if len(H['N'][u]) > 1 else ' is'} "
                 + andlist(f"{CL[e['workflow']]} on {e['model']}, {e['task']}, detectably more accurate than "
                           + andlist(f"{CL[a]} ({sg(e['gains'][a]['diff'], '+')} points, "
                                     f"[{sg(e['gains'][a]['ci95'][0])}, {sg(e['gains'][a]['ci95'][1])}])"
                                     for a in e["cheaper"])
                           + ", the sampling configurations with fewer " + ("total" if u == "total" else "output")
                           + " tokens" for e in H["N"][u]) + ".")
        else:
            t = f"Under {unit} cost no coordination workflow was detectably more accurate than every cheaper sampling configuration."
        nm(f"XmWin{U}Sentence", t)
        if H["matched_win"][u]:
            t = ("Against the cost-fixed comparator alone (SC with the largest $k$ not costlier), "
                 + andlist(f"{CL[e['workflow']]} on {e['model']}, {e['task']}, was detectably more accurate than "
                           f"{CL[e['comparator']]} ({sg(e['diff'], '+')} points, [{sg(e['ci95'][0])}, {sg(e['ci95'][1])}])"
                           for e in H["matched_win"][u]) + f" under {unit} cost.")
        else:
            t = f"No coordination workflow was detectably more accurate than its cost-fixed SC comparator under {unit} cost."
        nm(f"XmMatched{U}Sentence", t)
    for u, U in (("total", "Tot"), ("output", "Out")):
        n = len(H["N"][u])
        nm(f"XmNwin{U}Phrase", f"{word(n)} exception" + ("" if n == 1 else "s"))
    # detectable gains over individual cheaper sampling configurations (total tokens)
    beats = {a: 0 for a in SAMP}
    for d, lbl, mk, mode, tn, tl, tk, R in cells_of(J):
        for w in COORD:
            for a in R["verdict_total"][w]["cheaper"]:
                if R["tests"][f"{w}|{a}"]["ci95"][0] > 0:
                    beats[a] += 1
    parts = [f"{CL[a]} in {word(beats[a])}" for a in SAMP if beats[a]]
    if parts:
        t = ("Coordination workflows were detectably more accurate than some cheaper sampling configurations "
             "(than " + andlist(parts) + f" of the {H['cells']} combinations), ")
        nT = len(H["N"]["total"])
        t += ("but never than all of them under total tokens" if not nT else
              f"and than all of them in {word(nT)} under total tokens")
    else:
        t = "No coordination workflow was detectably more accurate than any cheaper sampling configuration under total tokens"
    nm("XmBeatsText", t)
    # seed-1-only sensitivity (runs stopped mid-seed)
    sens = []
    for d, lbl, mk, mode in present:
        for tn, tl, tk, _ in TASKS:
            S1 = J["models"][d].get("sensitivity_seed1_only", {}).get(tn)
            if not S1:
                continue
            R = J["models"][d]["tasks"][tn]
            ch = []
            for w in COORD:
                for vk, un in (("verdict_total", "total-token"), ("verdict_output", "output-token")):
                    a, b = R[vk][w]["verdict"], S1[vk][w]["verdict"]
                    if a != b:
                        ch.append(f"{CL[w]}'s {un} verdict changes from {a} to {b}")
            sens.append(f"with seed 1 alone on {lbl}, {tl} ({S1['n_items']} items), " +
                        (andlist(ch) if ch else "every verdict is unchanged"))
    inc = []
    for d, lbl, mk, mode, tn, tl, tk, R in cells_of(J):
        if not R["uniform_seeds"]:
            inc.append(f"{lbl} on {tl} has seed 1 on {R['seed_items']['1']} items and "
                       + andlist(f"seed {k} on the {v} items with all four workflow records"
                                 for k, v in R["seed_items"].items() if k != "1" and v))
    nm("XmIncompleteText", ("; ".join(inc) if inc else "every run is complete"))
    t = "; ".join(sens) if sens else "every run is complete, so no seed-1-only check is needed"
    nm("XmSensText", t[0].upper() + t[1:] + ".")
    # cost structure per model group (thinking: XmThink*, non-thinking: XmNt*) against the paper's 4-bit run
    for gmode, gp in (("thinking", "XmThink"), ("non-thinking", "XmNt")):
        TH = [R for d, lbl, mk, mode, tn, tl, tk, R in cells_of(J) if mode == gmode]
        if not TH:
            continue
        nm(f"{gp}Models", andlist([m[1] for m in present if m[3] == gmode]))
        sh = lambda c: [R["configs"][c]["out_share_pct"] for R in TH]
        nm(f"{gp}ShareSc", rng(sh("SC_k8")))
        nm(f"{gp}ShareDeb", rng(sh("DEBATE_N3_R1")))
        bes = [R["break_even"]["SC_k8"]["point"] for R in TH]
        fin = [x for x in bes if x != "inf"]
        nm(f"{gp}Be", (rng(fin, "{:.2f}") if fin else "") + (r" (and $\infty$)" if fin and len(fin) < len(bes)
                                                            else r"$\infty$" if not fin else ""))
        nm(f"{gp}DebOverSc", rng([R["configs"]["DEBATE_N3_R1"]["total_tokens"] / R["configs"]["SC_k8"]["total_tokens"]
                                  for R in TH], "{:.2f}"))
        acc = [R["configs"][c]["acc"] for R in TH for c in ALL]
        nm(f"{gp}Acc", rng(acc, "{:.1f}"))
        nm(f"{gp}MaxGap", f1(max(abs(R["tests"][f"{w}|SC_k8"]["diff"]) for R in TH for w in COORD)))
        nm(f"{gp}Cap", rng([v["hit_cap_pct"] for R in TH for v in R["truncation"].values()], "{:.1f}"))
        nm(f"{gp}ScFourCheaper", word(sum(R["break_even"]["SC_k4"]["point"] == "inf" and
                                          R["break_even"]["SC_k4"]["d_in_debate_minus_sc"] > 0 for R in TH)))
        nm(f"{gp}Ncells", word(len(TH)))
        nm(f"{gp}ScFourMatch", word(sum(R["tests"]["DEBATE_N3_R1|SC_k4"]["ci95"][0] <= 0 <= R["tests"]["DEBATE_N3_R1|SC_k4"]["ci95"][1]
                                        for R in TH)))
        tms = [R["token_multiple"]["DEBATE_N3_R1"]["multiple"] for R in TH]
        nm(f"{gp}TmDebNA", word(sum(t is None for t in tms)))
        nm(f"{gp}TmDeb", rng([t for t in tms if t is not None], "{:.1f}") if any(t is not None for t in tms) else "n/a")
        # SC k=8 more accurate than debate, detectably (95% interval of SC k=8 minus debate above zero)
        nm(f"{gp}ScBeatsDeb", word(sum(R["tests"]["DEBATE_N3_R1|SC_k8"]["ci95"][1] < 0 for R in TH)))
        # SC k=4 detectably more accurate than debate
        nm(f"{gp}ScFourBeatsDeb", word(sum(R["tests"]["DEBATE_N3_R1|SC_k4"]["ci95"][1] < 0 for R in TH)))
        # verdict counts over the group's workflow, model and task combinations
        nm(f"{gp}NWcells", word(3 * len(TH)))
        for vk, U in (("verdict_total", "Tot"), ("verdict_output", "Out")):
            for v in ("R", "C", "N"):
                nm(f"{gp}N{v}{U}", word(sum(R[vk][w]["verdict"] == v for R in TH for w in COORD)))
    # reference: the paper's 4-bit Qwen2.5-3B runs (GSM8K 500 x 3; MATH L1-3 238 x 3, cap 1,024)
    G = json.load(open(os.path.join(ROOT, "results/extra/rev3_reruns_analysis.json")))["tasks"]["gsm8k"]["configs"]
    Mx = json.load(open(os.path.join(PAPER_IN, "tables/math_easy.json")))["configs"]
    shr = lambda C, c: 100 * C[c]["out_tokens"] / C[c]["total_tokens"]
    nm("XmRefShareSc", rng([shr(G, "SC_k8"), shr(Mx, "SC_k8")]))
    nm("XmRefShareDeb", rng([shr(G, "DEBATE_N3_R1"), shr(Mx, "DEBATE_N3_R1")]))
    write_out("cross_model_macros.tex", "\n".join(L) + "\n")

    # main table: one row per (model, task)
    T, prev = [], None
    for d, lbl, mk, mode, tn, tl, tk, R in cells_of(J):
        if prev is not None and d != prev:
            T.append(r"\addlinespace[3pt]")
        D = R["configs"]
        acc = " & ".join(f1(D[c]["acc"]) for c in ("SA", "SC_k8", "DEBATE_N3_R1", "GE_R2", "HIERFIX_M3"))
        ver = " & ".join(R["verdict_total"][w]["verdict"] + "/" + R["verdict_output"][w]["verdict"] for w in COORD)
        tm = R["token_multiple"]["DEBATE_N3_R1"]["multiple"]
        ns = [int(k) for k, v in R["seed_items"].items() if v]
        seeds = str(len(ns)) if R["uniform_seeds"] else f"1--{len(ns)}"
        mlab = (lbl + (r"$^{\mathrm{t}}$" if mode == "thinking" else "")) if d != prev else ""
        T.append(f"{mlab} & {tl} & {R['n_items']}$\\times${seeds} & {acc} & {ver} & "
                 f"{'n/a' if tm is None else f1(tm)} & {rstr(R['break_even']['SC_k8'])} & "
                 f"{f1(max(v['hit_cap_pct'] for v in R['truncation'].values()))} \\\\")
        prev = d
    write_out("tables/cross_model_table.tex", 
        "% AUTO-GENERATED by scripts/cloud/cross_model_analysis.py. Do not edit.\n" + "\n".join(T) + "\n")

    # supplementary tables, one per task: accuracy with 95% CI, total and output tokens, truncation
    for tn0, tl0, tk0, _ in TASKS:
        S = []
        for d, lbl, mk, mode, tn, tl, tk, R in cells_of(J):
            if tn != tn0:
                continue
            D = R["configs"]
            M = J["models"][d]
            if S:
                S.append(r"\addlinespace[2pt]")
            ns = len(R["seed_items"])
            S.append(r"\multicolumn{7}{@{}l}{\textit{" + f"{M.get('model_id', lbl)}" + (", thinking on" if mode == "thinking" else "")
                     + f", cap {fint(M.get('cap_' + tn, 0))}, {R['n_items']} items, seeds "
                     + (("1 to 3" if ns == 3 else "1 and 2" if ns == 2 else "1") if R["uniform_seeds"]
                        else "1 and 2 (see text)") + "}} \\\\")
            for c in ALL:
                tr = R["truncation"].get(c)
                cap = (f1(tr["hit_cap_pct"]) if tr else "--") + " & " + f1(D[c]["no_answer_pct"])
                S.append(f"\\quad {CL[c][0].upper() + CL[c][1:]} & {f1(D[c]['acc'])} & [{f1(D[c]['lo'])}, {f1(D[c]['hi'])}] & "
                         f"{fint(D[c]['total_tokens'])} & {fint(D[c]['out_tokens'])} & {cap} \\\\")
        write_out(f"tables/cross_model_supp_{tn0}.tex", 
            "% AUTO-GENERATED by scripts/cloud/cross_model_analysis.py. Do not edit.\n" + "\n".join(S) + "\n")


def signature(path, tn):
    import glob
    return [[os.path.basename(p), os.path.getsize(p)] for p in sorted(glob.glob(os.path.join(path, f"{tn}_s*.jsonl")))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--fresh", action="store_true", help="recompute every (model, task), ignoring the cached JSON")
    ap.add_argument("--outdir", action="append", default=None,
                    help="paper directory to write the macros and tables into (repeatable; default paper/eaai_rev3,"
                         " e.g. --outdir paper/arxiv_rev3)")
    A = ap.parse_args()
    if A.outdir:
        OUTDIRS[:] = [os.path.abspath(d) for d in A.outdir]
    old = {}
    if os.path.exists(OUT_JSON) and not A.fresh:
        old = json.load(open(OUT_JSON)).get("models", {})
    J = {"source": "results/cloud/<model>/{gsm8k,math500}_s*.jsonl", "script": "scripts/cloud/cross_model_analysis.py",
         "definitions": __doc__, "models": {}}
    for d, lbl, mk, mode in MODELS:
        path = os.path.join(ROOT, "results/cloud", d)
        if not os.path.isdir(path):
            continue
        M = {"label": lbl, "mode": mode, "tasks": {}, "sensitivity_seed1_only": {}, "input_files": {}}
        for tn, tl, tk, nexp in TASKS:
            sig = signature(path, tn)
            o = old.get(d, {})
            cached = tn in o.get("tasks", {})
            if cached and ((sig and o.get("input_files", {}).get(tn) == sig) or not sig):
                # tests are deterministic (seed 0); with no raw logs on disk (a fresh clone, before
                # tar -xJf results/cloud/qwen3_raw_logs.tar.xz) the cached analysis is kept as well
                print(d, tn, "unchanged input files: reusing" if sig else "raw logs absent: reusing cached analysis",
                      flush=True)
                sig = sig or o["input_files"][tn]
                M["tasks"][tn] = o["tasks"][tn]
                if tn in o.get("sensitivity_seed1_only", {}):
                    M["sensitivity_seed1_only"][tn] = o["sensitivity_seed1_only"][tn]
                for k in ("model_id", "cap_" + tn):
                    if k in o:
                        M[k] = o[k]
                M["input_files"][tn] = sig
                continue
            P = plan(path, tn, nexp)
            if P is None or not P[2]:
                continue
            task, recs, ids, seeds_of, info = P
            sample = next(iter(recs.values()))
            M["model_id"], M["cap_" + tn] = sample["model"], sample["cap"]
            print(d, tn, "items", info["n_items"], "seed_items", info["seed_items"], "uniform", info["uniform_seeds"], flush=True)
            M["tasks"][tn] = analyse(f"{d}:{tn}", task, recs, ids, seeds_of, info, A.jobs)
            if not info["uniform_seeds"]:
                P1 = plan(path, tn, nexp, seed_filter=[1])
                M["sensitivity_seed1_only"][tn] = analyse(f"{d}:{tn}:s1", *P1[:4], P1[4], A.jobs)
            M["input_files"][tn] = sig
        if M["tasks"]:
            J["models"][d] = M
    emit(J)
    json.dump(J, open(OUT_JSON, "w"), indent=1, default=float)
    for d, M in J["models"].items():
        for tn, R in M["tasks"].items():
            D = R["configs"]
            print(f"{d} {tn}: " + " ".join(f"{c} {D[c]['acc']:.1f}/{D[c]['total_tokens']:.0f}" for c in ALL))
            print("   verdict tot", {w: R["verdict_total"][w]["verdict"] for w in COORD},
                  "out", {w: R["verdict_output"][w]["verdict"] for w in COORD},
                  "Tm", {w: R["token_multiple"][w]["multiple"] for w in COORD},
                  "r*", R["break_even"]["SC_k8"]["point"], R["break_even"]["SC_k8"]["ci95"])
    H = J["headline"]
    print("headline: cells", H["cells"], {k: {u: len(v) for u, v in H[k].items()} for k in ("N", "R", "matched_win")})
    if H["N"]["total"]:
        print("CHECK: a coordination workflow beats every cheaper sampling configuration under total tokens;"
              " the abstract and conclusion of paper/eaai_rev3 must be revisited by hand")


if __name__ == "__main__":
    main()
