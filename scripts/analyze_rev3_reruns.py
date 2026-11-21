"""Analysis of the rev3 reruns (protocols/REV3_RERUN_PROTOCOL.md, Section 6), exactly as
pre-registered:

  test        harness.stats.hier_paired_test (items resampled, seeds within items, 10,000
              replicates, seed 0), workflow minus comparator, two-sided 95% interval; 90% interval
              and one-sided gain probabilities from the same draws (scripts/analyze_hard_math.test,
              which asserts it replicates hier_paired_test). SA (one greedy run) enters with its
              single value repeated over seeds, as in scripts/analyze_hard_math.py.
  comparator  for each coordination workflow, the SC configuration with the largest k whose mean
              total tokens per item do not exceed the workflow's (k in {2, 4, 8}; {2, 4, 8, 12} on
              MATH). Costs on the same records; accuracy plays no part. SC k=8 also reported.
  verdict     noise-aware R/C/N rule of the paper (scripts/emit_math_easy.py): every cheaper sampling
              configuration; R if one has a 95% interval of its advantage above zero; N if every
              cheaper one is detectably less accurate; C otherwise.

Confirmatory family (Holm over these five and nothing else):
  C1  GSM8K 500 x 3: HIERFIX minus its cost-fixed SC comparator (phase A + R1).
  C2-C4 HumanEval 164 x 2 (R3, v2 entry-block extractor): DEBATE, GE, HIERFIX minus their cost-fixed
      SC comparators.
  C5  equal-cap difficulty interaction: (DEBATE minus SC k=8 on MATH L4-5, phase H) minus (DEBATE
      minus SC k=8 on MATH L1-3 at cap 2,048, R5); strata resampled independently
      (scripts/analyze_math_interaction.py).

R5 (MATH L1-3 at cap 2,048) is analysed only with --r5. Without it, R5 files are not opened and the
output records C5 and the Holm family as pending. With --r5 the script refuses to compute anything
from R5 unless every record of the run is present (SA x 238 seed 1; SC_k12, DEBATE_N3_R1, GE_R2,
HIERFIX_M3 x 238 x seeds 1-3).

Writes results/extra/rev3_reruns_analysis.json.
Usage: /usr/bin/python3 scripts/analyze_rev3_reruns.py [--r5] [--jobs 8]
"""
import argparse, glob, importlib.util, json, os, re, sys
from multiprocessing import get_context
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from harness import analyze, stats, tasks, topologies  # noqa: E402

spec = importlib.util.spec_from_file_location("hm", os.path.join(ROOT, "scripts/analyze_hard_math.py"))
hm = importlib.util.module_from_spec(spec); spec.loader.exec_module(hm)

OUT = os.path.join(ROOT, "results/extra/rev3_reruns_analysis.json")
SAMP = ["SA", "SC_k2", "SC_k4", "SC_k8"]
COORD = ["DEBATE_N3_R1", "GE_R2", "HIERFIX_M3"]


# ========== loading
def load(patterns):
    """(config, item, seed) -> record; first occurrence wins (as analyze_hard_math.load)."""
    recs = {}
    for pat in patterns:
        for p in sorted(glob.glob(os.path.join(ROOT, pat))):
            for l in open(p):
                try:
                    r = json.loads(l)
                except json.JSONDecodeError:
                    continue
                recs.setdefault((r["config"], r["item_id"], r["seed"]), r)
    return recs


def rec_cost(r, calls=None):
    cs = r.get("calls") if calls is None else calls
    if cs:
        return (sum(c["in_tokens"] + c["out_tokens"] for c in cs), sum(c["out_tokens"] for c in cs))
    return (r["total_tokens"], r["out_tokens"])


class Cell:
    """Per item: correctness array over seeds, mean total and output tokens over seeds."""
    def __init__(self):
        self.M, self.tot, self.out, self.none = {}, {}, {}, []


def cell_from(recs, cfg, ids, seeds, key="correct"):
    c = Cell()
    for i in ids:
        ss = [1] if cfg == "SA" else seeds
        R = [recs[(cfg, i, s)] for s in ss]
        v = [float(bool(r[key])) for r in R]
        c.M[i] = np.array(v * (len(seeds) if cfg == "SA" else 1))
        cc = [rec_cost(r) for r in R]
        c.tot[i] = float(np.mean([x[0] for x in cc])); c.out[i] = float(np.mean([x[1] for x in cc]))
        c.none += [r["pred"] is None for r in R]
    return c


def sc_prefix_cells(recs, cfg, ids, seeds, task, ks, key="correct"):
    """SC k < K from the first k samples of each SC K record (harness seed rule)."""
    out = {}
    for k in ks:
        c = Cell()
        for i in ids:
            v, tt, oo = [], [], []
            for s in seeds:
                r = recs[(cfg, i, s)]; cs = r["calls"][:k]
                ans = topologies._majority([x["extracted"] for x in cs])
                ok = bool(task["score"](ans, r["gold"]))
                if k == int(cfg.split("_k")[1]):
                    assert ok == bool(r[key]), (cfg, i, s)
                v.append(float(ok)); a, b = rec_cost(r, cs); tt.append(a); oo.append(b)
                c.none.append(ans is None)
            c.M[i] = np.array(v); c.tot[i] = float(np.mean(tt)); c.out[i] = float(np.mean(oo))
        out[f"SC_k{k}"] = c
    return out


# ========== tests (parallel)
def _job(args):
    key, a, b = args
    return key, hm.test(a, b)


def run_tests(jobs, n_jobs):
    if not jobs:
        return {}
    with get_context("fork").Pool(n_jobs) as pool:
        return dict(pool.map(_job, jobs, chunksize=1))


def item_ci(x):
    lo, hi = analyze._bootstrap_ci(np.asarray(x, float), B=10000)
    return [lo, hi]


# ========== per-task block
def describe(cells, ids):
    out = {}
    for c, v in cells.items():
        corr = np.array([v.M[i].mean() for i in ids])
        lo, hi = item_ci(corr)
        out[c] = {"acc": 100 * float(corr.mean()), "lo": 100 * lo, "hi": 100 * hi,
                  "total_tokens": float(np.mean([v.tot[i] for i in ids])),
                  "out_tokens": float(np.mean([v.out[i] for i in ids])),
                  "no_answer_pct": 100 * float(np.mean(v.none)) if v.none else None}
    return out


def plan_task(name, cells, ids, samp, coord, extra_pairs=()):
    """Jobs: every coordination workflow minus every sampling configuration, plus extras."""
    jobs = []
    for w in coord:
        for a in samp:
            jobs.append((f"{name}|{w}|{a}", cells[w].M, cells[a].M))
    for (x, y) in extra_pairs:
        jobs.append((f"{name}|{x}|{y}", cells[x].M, cells[y].M))
    return jobs


def finish_task(name, cells, ids, samp, coord, T, ks_comp):
    D = describe(cells, ids)
    res = {"n_items": len(ids), "configs": D, "tests": {}, "verdict_total": {}, "verdict_output": {},
           "comparator": {}}
    for key, t in T.items():
        nm, x, y = key.split("|")
        if nm != name:
            continue
        e = dict(t)
        if x in cells and y in cells:
            d = np.array([cells[y].tot[i] - cells[x].tot[i] for i in ids])  # sampling minus workflow
            e["dtok"] = float(d.mean()); e["dtok_ci95"] = item_ci(d)
            d = np.array([cells[y].out[i] - cells[x].out[i] for i in ids])
            e["dout"] = float(d.mean()); e["dout_ci95"] = item_ci(d)
        res["tests"][f"{x}|{y}"] = e  # x minus y (workflow minus sampling)
    for w in coord:
        for unit, vk in (("total_tokens", "verdict_total"), ("out_tokens", "verdict_output")):
            cheaper = [a for a in samp if D[a][unit] < D[w][unit]]
            adv = {a: [-res["tests"][f"{w}|{a}"]["ci95"][1], -res["tests"][f"{w}|{a}"]["ci95"][0]] for a in cheaper}
            robust = [a for a in cheaper if adv[a][0] > 0]
            v = "R" if robust else ("N" if cheaper and all(adv[a][1] < 0 for a in cheaper) else "C")
            if not cheaper:
                v = "N"
            res[vk][w] = {"cheaper": cheaper, "robust_by": robust, "verdict": v,
                          "cheapest_robust": robust[0] if robust else None}
        ok = [k for k in ks_comp if D[f"SC_k{k}"]["total_tokens"] <= D[w]["total_tokens"]]
        res["comparator"][w] = f"SC_k{max(ok)}" if ok else None
    return res


# ========== descriptive checks
TEMPLATE = {"gsm8k": re.compile(r"The answer is N\b"), "mmlu": re.compile(r"The answer is X\b"),
            "mmlu_pro_eng": re.compile(r"The answer is X\b")}


def agg_check(recs, cfg, ids, seeds, task_name):
    R = [recs[(cfg, i, s)] for i in ids for s in seeds]
    pc = lambda x: round(100 * float(np.mean(x)), 2)
    out = {"n": len(R), "no_answer_pct": pc([r["pred"] is None for r in R])}
    if R[0].get("calls") and "text" in R[0]["calls"][-1] and task_name in TEMPLATE:
        out["template_line_pct"] = pc([bool(TEMPLATE[task_name].search(r["calls"][-1]["text"])) for r in R])
        out["agg_cap_pct"] = pc([r["calls"][-1]["hit_cap"] for r in R])
    return out


def ctx_overflow(recs, ctx):
    C = [c for r in recs.values() for c in r.get("calls", []) if "in_tokens" in c]
    return {"calls": len(C), "calls_at_or_above_ctx": int(sum(c["in_tokens"] + c["out_tokens"] >= ctx for c in C)),
            "max_in_plus_out": int(max(c["in_tokens"] + c["out_tokens"] for c in C)) if C else None, "ctx": ctx}


# ========== frontier share
def _pareto(args):
    cor, tok, seed, B = args
    configs = list(cor); items = sorted(set.intersection(*[set(cor[c]) for c in configs])); n = len(items)
    corA = {c: [cor[c][i] for i in items] for c in configs}; tokA = {c: [tok[c][i] for i in items] for c in configs}
    rng = np.random.default_rng(seed); count = {c: 0 for c in configs}
    for _ in range(B):
        idx = rng.integers(0, n, size=n); acc, cost = {}, {}
        for c in configs:
            a = corA[c]; tk = tokA[c]; sa = st = 0.0
            for ii in idx:
                av = a[ii]; tv = tk[ii]
                sa += av[rng.integers(0, len(av), size=len(av))].mean()
                st += tv[rng.integers(0, len(tv), size=len(tv))].mean()
            acc[c] = sa / n; cost[c] = st / n
        for c in configs:
            if not any((acc[d] >= acc[c] and cost[d] <= cost[c]) and (acc[d] > acc[c] or cost[d] < cost[c])
                       for d in configs if d != c):
                count[c] += 1
    return {c: 100.0 * count[c] / B for c in configs}


# ========== R5 (C5)
def r5_check():
    """Count R5 records (keys only, no outcome is read); exit if any is missing."""
    eids = [it["id"] for it in tasks.get_task("math500")["load"](n=238, split="all")]
    keys = set()
    for p in sorted(glob.glob(os.path.join(ROOT, "results/rev3/math500_cap2048_qwen3b_s*.jsonl"))):
        for l in open(p):
            try:
                r = json.loads(l)
            except json.JSONDecodeError:
                continue
            keys.add((r["config"], r["item_id"], r["seed"]))
    need = [("SA", i, 1) for i in eids] + [(c, i, s) for i in eids for s in (1, 2, 3)
                                           for c in ("SC_k12", "DEBATE_N3_R1", "GE_R2", "HIERFIX_M3")]
    missing = [k for k in need if k not in keys]
    if missing:
        raise SystemExit(f"R5 incomplete: {len(missing)} of {len(need)} records missing; nothing computed "
                         f"(no R5 accuracy was read)")


def r5_block(n_jobs):
    """Equal-cap interaction; r5_check() has already refused incomplete R5 data."""
    r5_check()
    SEEDS = [1, 2, 3]
    easy_items = tasks.get_task("math500")["load"](n=238, split="all")
    hard_items = hm.camp.load_math_levels(tasks, 4, 5, 262)
    eids = [it["id"] for it in easy_items]; hids = [it["id"] for it in hard_items]
    R5 = load(["results/rev3/math500_cap2048_qwen3b_s*.jsonl"])
    task = tasks.get_task("math500")
    H = load(["results/power/math500hard_qwen3b_s*.jsonl"])
    Bp = load(["results/power/math500_qwen3b_s*.jsonl"])
    def strat(recs, ids, wmap):
        cells = {w: cell_from(recs, src, ids, SEEDS) for w, src in wmap.items()}
        cells["SA"] = cell_from(recs, "SA", ids, SEEDS)
        cells.update(sc_prefix_cells(recs, "SC_k12", ids, SEEDS, task, [2, 4, 8, 12]))
        return cells
    WM_new = {"DEBATE_N3_R1": "DEBATE_N3_R1", "GE_R2": "GE_R2", "HIERFIX_M3": "HIERFIX_M3"}
    WM_old = {"DEBATE_N3_R1": "DEBATE_N3_R1", "GE_R2": "GE_R2", "HIERFIX_M3": "HIER_M3"}  # same prompts (protocol 2)
    E = strat(R5, eids, WM_new); Hc = strat(H, hids, WM_old); Bc = strat(Bp, eids, WM_old)
    out = {"easy_cap2048": describe(E, eids), "n_easy": len(eids), "n_hard": len(hids),
           "caps": {"R5": sorted({R5[("SC_k12", i, 1)]["cap"] for i in eids}),
                    "phaseH": sorted({H[("SC_k12", i, 1)]["cap"] for i in hids})},
           "interaction_equal_cap": {}, "within_R5": {}, "R5_minus_phaseB": {}}
    acc = lambda cells, w, ids: 100 * float(np.mean([cells[w].M[i].mean() for i in ids]))
    for w in COORD:
        bh = hm.boot_dist(Hc[w].M, Hc["SC_k8"].M, seed=0)
        be = hm.boot_dist(E[w].M, E["SC_k8"].M, seed=1)  # independent strata, as analyze_math_interaction
        d = bh - be
        obs = (acc(Hc, w, hids) - acc(Hc, "SC_k8", hids)) - (acc(E, w, eids) - acc(E, "SC_k8", eids))
        lo, hi = np.percentile(d, [2.5, 97.5])
        out["interaction_equal_cap"][w] = {"obs": round(obs, 2), "ci95": [round(float(lo), 2), round(float(hi), 2)],
                                           "p": float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))}
    jobs = [(f"r5|{w}|SC_k8", E[w].M, E["SC_k8"].M) for w in COORD]
    jobs += [(f"r5|{w}|SA", E[w].M, E["SA"].M) for w in COORD]
    jobs += [(f"r5minusB|{w}|{w}", E[w].M, Bc[w].M) for w in COORD + ["SA", "SC_k8", "SC_k12"]]
    T = run_tests(jobs, n_jobs)
    for k, v in T.items():
        a, w, b = k.split("|")
        (out["within_R5"] if a == "r5" else out["R5_minus_phaseB"])[f"{w}|{b}"] = v
    out["context_overflow"] = ctx_overflow(R5, 8192)
    return out


# ========== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r5", action="store_true", help="also run C5 and the Holm family (refuses incomplete R5)")
    ap.add_argument("--jobs", type=int, default=8)
    A = ap.parse_args()
    if A.r5:
        r5_check()  # before any other work
    J = {"protocol": "protocols/REV3_RERUN_PROTOCOL.md (commit 934b3194)", "tasks": {}}
    jobs, plans = [], {}

    # == GSM8K 500 x 3 (phase A + R1)
    g = tasks.get_task("gsm8k"); S3 = [1, 2, 3]
    gids = [it["id"] for it in g["load"](n=500, split="all")]
    PA = load(["results/power/gsm8k_qwen3b_s[123].jsonl", "results/power/gsm8k_qwen3b_s[123]_sh*of2.jsonl"])
    R1 = load(["results/rev3/gsm8k_qwen3b_s*.jsonl"])
    for k in [("SA", i, 1) for i in gids] + [(c, i, s) for i in gids for s in S3
                                             for c in ("SC_k8", "DEBATE_N3_R1", "GE_R2", "HIER_M3")]:
        assert k in PA, ("phase A incomplete", k)
    assert all(("HIERFIX_M3", i, s) in R1 for i in gids for s in S3), "R1 incomplete"
    gc = {"SA": cell_from(PA, "SA", gids, S3)}
    gc.update(sc_prefix_cells(PA, "SC_k8", gids, S3, g, [2, 4, 8]))
    for w in ("DEBATE_N3_R1", "GE_R2", "HIER_M3"):
        gc[w] = cell_from(PA, w, gids, S3)
    gc["HIERFIX_M3"] = cell_from(R1, "HIERFIX_M3", gids, S3)
    gcoord = COORD + ["HIER_M3"]
    jobs += plan_task("gsm8k", gc, gids, SAMP, gcoord, [("HIERFIX_M3", "HIER_M3"), ("SC_k8", "SA")])
    plans["gsm8k"] = (gc, gids, gcoord, [2, 4, 8])
    # C1 sensitivity: items 101 to 500 (comparator fixed below on all 500 items)
    g400 = gids[100:]

    # == MMLU 80 x 2 (July matrix + R2)
    S2 = [1, 2]
    MM = load(["results/mmlu_matrix.jsonl"]); R2 = load(["results/rev3/mmlu_qwen3b_*.jsonl"])
    mids = sorted({k[1] for k in MM}, key=lambda x: int(x.split("-")[1]))
    assert len(mids) == 80 and {k[1] for k in R2} == set(mids)
    mc = {c: cell_from(MM, c, mids, S2) for c in SAMP + ["DEBATE_N3_R1", "GE_R2", "HIER_M3"]}
    mc["HIERFIX_M3"] = cell_from(R2, "HIERFIX_M3", mids, S2)
    jobs += plan_task("mmlu", mc, mids, SAMP, gcoord, [("HIERFIX_M3", "HIER_M3"), ("SC_k8", "SA")])
    plans["mmlu"] = (mc, mids, gcoord, [2, 4, 8])

    # == MMLU-Pro engineering 150 x 2 (matrix + R4)
    PE = load(["results/mmlu_pro_eng_matrix.jsonl"]); R4 = load(["results/rev3/mmlu_pro_eng_qwen3b_*.jsonl"])
    pids = sorted({k[1] for k in PE})
    assert len(pids) == 150 and {k[1] for k in R4} == set(pids)
    pc_ = {c: cell_from(PE, c, pids, S2) for c in SAMP + ["DEBATE_N3_R1", "GE_R2", "HIER_M3"]}
    pc_["HIERFIX_M3"] = cell_from(R4, "HIERFIX_M3", pids, S2)
    jobs += plan_task("proeng", pc_, pids, SAMP, gcoord, [("HIERFIX_M3", "HIER_M3"), ("SC_k8", "SA")])
    plans["proeng"] = (pc_, pids, gcoord, [2, 4, 8])

    # == HumanEval 164 x 2 (R3), v2 primary and v1
    R3 = load(["results/rev3/humaneval_qwen3b_*.jsonl"])
    hids = [it["id"] for it in tasks.get_task("humaneval")["load"](n=None, split="all")]
    assert len(hids) == 164
    for k in [("SA", i, 1) for i in hids] + [(c, i, s) for i in hids for s in S2
                                             for c in ("SC_k8",) + tuple(COORD)]:
        assert k in R3, ("R3 incomplete", k)
    OLD = load(["results/humaneval_matrix_pc.jsonl"])
    hc = {}
    for ver, key in (("v2", "correct_v2"), ("v1", "correct")):
        c = {"SA": cell_from(R3, "SA", hids, S2, key)}
        for k in (2, 4):  # logged prefix votes (same final calls, same vote) under both extractors
            cl = Cell()
            for i in hids:
                v, tt, oo = [], [], []
                for s in S2:
                    r = R3[("SC_k8", i, s)]
                    v.append(float(bool(r["sc_prefix"][str(k)][key])))
                    a, b = rec_cost(r, r["calls"][:k]); tt.append(a); oo.append(b)
                    cl.none.append(r["sc_prefix"][str(k)]["pred" if ver == "v1" else "pred_v2"] is None)
                cl.M[i] = np.array(v); cl.tot[i] = float(np.mean(tt)); cl.out[i] = float(np.mean(oo))
            c[f"SC_k{k}"] = cl
        c["SC_k8"] = cell_from(R3, "SC_k8", hids, S2, key)
        for w in COORD:
            c[w] = cell_from(R3, w, hids, S2, key)
        if ver == "v2":  # no-answer under v2 is pred_v2 None
            for cfg in ["SA", "SC_k8"] + COORD:
                ss = [1] if cfg == "SA" else S2
                c[cfg].none = [R3[(cfg, i, s)]["pred_v2"] is None for i in hids for s in ss]
        hc[ver] = c
        jobs += plan_task(f"code_{ver}", c, hids, SAMP, COORD, [("SC_k8", "SA")])
        plans[f"code_{ver}"] = (c, hids, COORD, [2, 4, 8])
    # exploratory: v1 minus v2 per configuration; earlier run (v1) minus R3 (v1) per configuration
    oc = {}
    for cfg in ["SA", "SC_k2", "SC_k4", "SC_k8", "DEBATE_N3_R1", "GE_R2", "HIER_M3"]:
        oc[cfg] = cell_from(OLD, cfg, hids, S2)
    for cfg in SAMP + COORD:
        jobs.append((f"codev1v2|{cfg}|{cfg}", hc["v1"][cfg].M, hc["v2"][cfg].M))
        ocfg = "HIER_M3" if cfg == "HIERFIX_M3" else cfg
        jobs.append((f"codeoldnew|{ocfg}|{cfg}", oc[ocfg].M, hc["v1"][cfg].M))

    # == frontier share (GSM8K, headline configurations; and with the unrepaired HIER added)
    fr_cfg = ["SA", "SC_k2", "SC_k4", "SC_k8", "DEBATE_N3_R1", "GE_R2", "HIERFIX_M3"]
    def seedtok(cell_recs, cfg, ids, seeds, k=None):
        out = {}
        for i in ids:
            ss = [1] if cfg == "SA" else seeds
            vals = []
            for s in ss:
                r = cell_recs[(cfg if k is None else "SC_k8", i, s)]
                vals.append(rec_cost(r, r["calls"][:k] if k else None)[0])
            out[i] = np.array(vals, float)
        return out
    def seedcor(cell, cfg, ids):
        return {i: (cell.M[i][:1] if cfg == "SA" else cell.M[i]) for i in ids}
    gtok = {"SA": seedtok(PA, "SA", gids, S3), "SC_k2": seedtok(PA, None, gids, S3, 2),
            "SC_k4": seedtok(PA, None, gids, S3, 4), "SC_k8": seedtok(PA, "SC_k8", gids, S3),
            "DEBATE_N3_R1": seedtok(PA, "DEBATE_N3_R1", gids, S3), "GE_R2": seedtok(PA, "GE_R2", gids, S3),
            "HIER_M3": seedtok(PA, "HIER_M3", gids, S3), "HIERFIX_M3": seedtok(R1, "HIERFIX_M3", gids, S3)}
    fr_jobs = [({c: seedcor(gc[c], c, gids) for c in fr_cfg}, {c: gtok[c] for c in fr_cfg}, 0, 10000),
               ({c: seedcor(gc[c], c, gids) for c in fr_cfg + ["HIER_M3"]},
                {c: gtok[c] for c in fr_cfg + ["HIER_M3"]}, 0, 10000)]

    # == run all tests in parallel
    print(f"{len(jobs)} tests, 2 frontier runs, {A.jobs} processes", flush=True)
    with get_context("fork").Pool(A.jobs) as pool:
        fr_async = [pool.apply_async(_pareto, (j,)) for j in fr_jobs]
        T = dict(pool.map(_job, jobs, chunksize=1))
        FR = [f.get() for f in fr_async]

    for name, (cells, ids, coord, ks) in plans.items():
        J["tasks"][name] = finish_task(name, cells, ids, SAMP, coord, T, ks)
    J["tasks"]["gsm8k"]["frontier_share"] = FR[0]
    J["tasks"]["gsm8k"]["frontier_share_with_unrepaired_HIER"] = FR[1]
    J["code_v1_minus_v2"] = {k.split("|")[1]: v for k, v in T.items() if k.startswith("codev1v2|")}
    J["code_old_v1_minus_R3_v1"] = {k.split("|")[1] + "|" + k.split("|")[2]: v for k, v in T.items()
                                    if k.startswith("codeoldnew|")}
    J["code_old_run_accuracy"] = describe(oc, hids)

    # == confirmatory C1 to C4 (C5 below)
    conf = {}
    G = J["tasks"]["gsm8k"]; comp = G["comparator"]["HIERFIX_M3"]
    conf["C1"] = {"task": "gsm8k", "workflow": "HIERFIX_M3", "comparator": comp,
                  "prediction": "none stated (two-sided)", **G["tests"][f"HIERFIX_M3|{comp}"]}
    sens = hm.test({i: gc["HIERFIX_M3"].M[i] for i in g400}, {i: gc[comp].M[i] for i in g400})
    conf["C1"]["sensitivity_items_101_500"] = sens
    H2 = J["tasks"]["code_v2"]
    for cn, w in (("C2", "DEBATE_N3_R1"), ("C3", "GE_R2"), ("C4", "HIERFIX_M3")):
        comp = H2["comparator"][w]
        conf[cn] = {"task": "humaneval (v2)", "workflow": w, "comparator": comp, **H2["tests"][f"{w}|{comp}"]}
        H1 = J["tasks"]["code_v1"]; c1 = H1["comparator"][w]
        conf[cn]["v1_same_comparator_rule"] = {"comparator": c1, **H1["tests"][f"{w}|{c1}"]}
    for v in conf.values():
        v["ci90_vs_margin5"] = {"ci90": v["ci90"], "gain5_excluded": v["ci90"][1] < 5, "any_gain_excluded": v["ci90"][1] < 0}
    J["confirmatory"] = conf

    # == descriptive checks
    J["checks"] = {
        "gsm8k": {"HIERFIX_M3": agg_check(R1, "HIERFIX_M3", gids, S3, "gsm8k"),
                  "HIER_M3": agg_check(PA, "HIER_M3", gids, S3, "gsm8k"),
                  "phaseA_first100_host": sorted({PA[("SC_k8", i, 1)].get("host") for i in gids[:100]}),
                  "phaseA_rest_host": sorted({PA[("SC_k8", i, 1)].get("host") for i in gids[100:]}),
                  "ctx_R1": ctx_overflow(R1, 4096), "ctx_phaseA": ctx_overflow(PA, 4096)},
        "mmlu": {"HIERFIX_M3": agg_check(R2, "HIERFIX_M3", mids, S2, "mmlu"),
                 "HIER_M3": agg_check(MM, "HIER_M3", mids, S2, "mmlu"), "ctx_R2": ctx_overflow(R2, 4096)},
        "proeng": {"HIERFIX_M3": agg_check(R4, "HIERFIX_M3", pids, S2, "mmlu_pro_eng"),
                   "HIER_M3": agg_check(PE, "HIER_M3", pids, S2, "mmlu_pro_eng"), "ctx_R4": ctx_overflow(R4, 4096)},
        "humaneval": humaneval_checks(R3, hids, S2),
    }
    he_item = {it["id"]: it for it in tasks.get_task("humaneval")["load"](n=None, split="all")}
    J["checks"]["humaneval"]["ctx_R3"] = ctx_overflow(R3, 4096)
    del he_item

    # == C5 and the Holm family of five
    if A.r5:
        J["r5"] = r5_block(A.jobs)
        c5 = J["r5"]["interaction_equal_cap"]["DEBATE_N3_R1"]
        conf["C5"] = {"task": "MATH L4-5 (phase H) vs L1-3 (R5), cap 2,048 both", "workflow": "DEBATE_N3_R1",
                      "prediction": "D > 0 (two-sided test)", "diff": c5["obs"], "ci95": c5["ci95"], "p": c5["p"]}
        names = ["C1", "C2", "C3", "C4", "C5"]
        adj = stats.holm([conf[n]["p"] for n in names])
        for n, a in zip(names, adj):
            conf[n]["p_holm5"] = float(a)
        J["holm_family_of_5"] = "complete"
    else:
        conf["C5"] = {"status": "pending: R5 (MATH L1-3, cap 2,048) not analysed; run with --r5 when complete"}
        J["holm_family_of_5"] = "pending (C5 needs R5); C1-C4 p values are unadjusted"

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(J, open(OUT, "w"), indent=1)
    for n, v in conf.items():
        print(n, {k: v.get(k) for k in ("workflow", "comparator", "diff", "ci95", "p", "p_holm5", "status")})
    print("wrote", OUT)


def humaneval_checks(R3, hids, S2):
    """Per extractor: share of final-round outputs with no parseable program (SC samples, debate
    round-1 agents, GE and HIERFIX last calls), outputs v2 parses that v1 does not, outputs whose
    program changes, share of final calls at the cap."""
    out = {}
    pc = lambda x: round(100 * float(np.mean(x)), 2) if len(x) else None
    allf = []
    for cfg in ["SA", "SC_k8"] + COORD:
        ss = [1] if cfg == "SA" else S2
        F = [r["calls"][j] for i in hids for s in ss for r in [R3[(cfg, i, s)]] for j in r["final_call_idx"]]
        allf += [(cfg, c) for c in F]
        out[cfg] = {"final_outputs": len(F),
                    "no_program_v1_pct": pc([c["extracted"] is None for c in F]),
                    "no_program_v2_pct": pc([c["extracted_v2"] is None for c in F]),
                    "n_no_program_v1": int(sum(c["extracted"] is None for c in F)),
                    "n_no_program_v2": int(sum(c["extracted_v2"] is None for c in F)),
                    "n_v2_parses_v1_not": int(sum(c["extracted"] is None and c["extracted_v2"] is not None for c in F)),
                    "n_program_changes": int(sum(c["extracted"] != c["extracted_v2"] for c in F)),
                    "final_at_cap_pct": pc([c["hit_cap"] for c in F]),
                    "records_correct_v1": int(sum(bool(R3[(cfg, i, s)]["correct"]) for i in hids for s in ss)),
                    "records_correct_v2": int(sum(bool(R3[(cfg, i, s)]["correct_v2"]) for i in hids for s in ss)),
                    "records_verdict_changes": int(sum(bool(R3[(cfg, i, s)]["correct"]) != bool(R3[(cfg, i, s)]["correct_v2"])
                                                       for i in hids for s in ss))}
    coord_f = [c for cfg, c in allf if cfg in COORD]
    samp_f = [c for cfg, c in allf if cfg in ("SA", "SC_k8")]
    out["pooled"] = {"sampling_no_program_v1_pct": pc([c["extracted"] is None for c in samp_f]),
                     "sampling_no_program_v2_pct": pc([c["extracted_v2"] is None for c in samp_f]),
                     "coord_no_program_v1_pct": pc([c["extracted"] is None for c in coord_f]),
                     "coord_no_program_v2_pct": pc([c["extracted_v2"] is None for c in coord_f]),
                     "n_v2_parses_v1_not": int(sum(c["extracted"] is None and c["extracted_v2"] is not None for _, c in allf)),
                     "n_program_changes": int(sum(c["extracted"] != c["extracted_v2"] for _, c in allf)),
                     "final_outputs": len(allf)}
    return out


if __name__ == "__main__":
    main()
