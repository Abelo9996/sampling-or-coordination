"""Paired significance tests + extrapolation-aware crossover, computed from the
released matrix logs. Addresses reviewer findings: (1) report paired item-level
tests (the correct analysis, since all configs share items); (2) mark which
equal-budget crossover comparisons are *measured* vs. *extrapolated* (np.interp
clamps beyond the measured SC range).
"""
import json
import os
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(task):
    p = os.path.join(ROOT, "results", f"{task}_matrix.jsonl")
    return [json.loads(l) for l in open(p) if l.strip()]


def per_item_correct(rows):
    """config -> {item_id -> mean correctness over seeds}."""
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r["config"]][r["item_id"]].append(int(r["correct"]))
    return {c: {i: float(np.mean(v)) for i, v in items.items()}
            for c, items in by.items()}


def per_item_tokens(rows):
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r["config"]][r["item_id"]].append(r["total_tokens"])
    return {c: {i: float(np.mean(v)) for i, v in items.items()}
            for c, items in by.items()}


def paired_test(a_map, b_map, B=10000, seed=0):
    """Paired bootstrap on per-item correctness differences (a - b).
    Returns (mean_diff_pct, two_sided_p)."""
    items = sorted(set(a_map) & set(b_map))
    d = np.array([a_map[i] - b_map[i] for i in items])
    if len(d) == 0:
        return float("nan"), float("nan")
    obs = d.mean()
    rng = np.random.default_rng(seed)
    # bootstrap the sampling distribution of the mean difference
    idx = rng.integers(0, len(d), size=(B, len(d)))
    boot = d[idx].mean(axis=1)
    # two-sided p: fraction of bootstrap means on the opposite side of 0
    p = 2.0 * min((boot <= 0).mean(), (boot >= 0).mean())
    return obs * 100, min(p, 1.0)


def sc_curve(tok_map, cor_map):
    """(tokens, acc) for SA+SC configs, sorted by tokens; plus the max measured
    SC token cost (the clamp boundary)."""
    pts = []
    for c in tok_map:
        if c == "SA" or c.startswith("SC_"):
            t = np.mean(list(tok_map[c].values()))
            a = np.mean(list(cor_map[c].values()))
            pts.append((c, t, a))
    pts.sort(key=lambda x: x[1])
    return pts


def crossover_report(task):
    rows = _load(task)
    cor = per_item_correct(rows)
    tok = per_item_tokens(rows)
    sc = sc_curve(tok, cor)
    sc_tok = np.array([p[1] for p in sc])
    sc_acc = np.array([p[2] for p in sc])
    sc_max_tok = sc_tok.max()
    sc_max_acc = sc_acc[np.argmax(sc_tok)]
    out = []
    for c in cor:
        topo = c.split("_")[0]
        if topo in ("SA", "SC"):
            continue
        mas_tok = np.mean(list(tok[c].values()))
        mas_acc = np.mean(list(cor[c].values()))
        extrapolated = mas_tok > sc_max_tok
        sc_at = float(sc_max_acc if extrapolated
                      else np.interp(mas_tok, sc_tok, sc_acc))
        out.append({
            "config": c, "mas_acc": mas_acc * 100, "mas_tok": mas_tok,
            "sc_equal_acc": sc_at * 100, "gap": (sc_at - mas_acc) * 100,
            "extrapolated": extrapolated,
        })
    return out, sc_max_tok


def headline(task):
    """Paired tests for the claims that matter: SC_k8 vs each MAS, and SC_k8 vs SA."""
    rows = _load(task)
    cor = per_item_correct(rows)
    res = {}
    if "SC_k8" in cor:
        for c in cor:
            if c.split("_")[0] in ("DEBATE", "GE", "HIER") or c == "SA":
                res[f"SC_k8 vs {c}"] = paired_test(cor["SC_k8"], cor[c])
    return res


# ---------------------------------------------------------------------------
# Seed-aware hierarchical bootstrap, Holm correction, and probabilistic Pareto
# membership. These are the analyses the paper's headline significance and the
# frontier-membership figure rely on; they live here so every quoted p-value and
# membership percentage regenerates from the released logs.
# ---------------------------------------------------------------------------

# task name -> released log file (MATH uses the confound-free 1024-token cap file).
TASK_FILE = {
    "gsm8k": "gsm8k_matrix.jsonl",
    "mmlu": "mmlu_matrix.jsonl",
    "math500": "math500_trunc.jsonl",
    "math": "math500_trunc.jsonl",
    "mmlu_pro_eng": "mmlu_pro_eng_matrix.jsonl",
    "humaneval": "humaneval_matrix_pc.jsonl",
}

# headline coordination configs per task (the single operating point compared to SC_k8).
HEADLINE_COORD = ["DEBATE_N3_R1", "GE_R2", "HIER_M3"]


def _loadf(task):
    p = os.path.join(ROOT, "results", TASK_FILE[task])
    return [json.loads(l) for l in open(p) if l.strip()]


def per_item_seed_correct(rows):
    """config -> {item_id -> np.array of per-seed correctness}."""
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r["config"]][r["item_id"]].append(int(r["correct"]))
    return {c: {i: np.asarray(v, float) for i, v in items.items()}
            for c, items in by.items()}


def per_item_seed_tokens(rows, cost="total_tokens"):
    """config -> {item_id -> np.array of per-seed cost} (default: total tokens;
    `cost` names another numeric record field, e.g. energy_j)."""
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r["config"]][r["item_id"]].append(float(r[cost]))
    return {c: {i: np.asarray(v, float) for i, v in items.items()}
            for c, items in by.items()}


def hier_paired_test(a, b, B=10000, seed=0):
    """Seed-aware hierarchical paired bootstrap of (a - b) per-item correctness.

    Resamples items with replacement and, within each resampled item, resamples
    that item's generation seeds with replacement, so both item and seed variance
    are propagated. Returns (mean_diff_pct, two_sided_p, ci_lo_pct, ci_hi_pct).
    """
    items = sorted(set(a) & set(b))
    if not items:
        return float("nan"), float("nan"), float("nan"), float("nan")
    A = [a[i] for i in items]
    Bv = [b[i] for i in items]
    obs = float(np.mean([A[j].mean() - Bv[j].mean() for j in range(len(items))]))
    rng = np.random.default_rng(seed)
    n = len(items)
    boot = np.empty(B)
    for t in range(B):
        idx = rng.integers(0, n, size=n)
        s = 0.0
        for ii in idx:
            av, bv = A[ii], Bv[ii]
            ai = av[rng.integers(0, len(av), size=len(av))].mean()
            bi = bv[rng.integers(0, len(bv), size=len(bv))].mean()
            s += ai - bi
        boot[t] = s / n
    p = 2.0 * min((boot <= 0).mean(), (boot >= 0).mean())
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return obs * 100, min(p, 1.0), lo * 100, hi * 100


def holm(pvals):
    """Holm-Bonferroni step-down adjusted p-values, returned in input order."""
    m = len(pvals)
    order = sorted(range(m), key=lambda k: pvals[k])
    adj = [0.0] * m
    running = 0.0
    for rank, k in enumerate(order):
        val = (m - rank) * pvals[k]
        running = max(running, val)
        adj[k] = min(running, 1.0)
    return adj


def headline_hier(task, B=10000, seed=0):
    """SC_k8 vs each headline coordination config: hierarchical test + Holm."""
    rows = _loadf(task)
    cor = per_item_seed_correct(rows)
    coord = [c for c in HEADLINE_COORD if c in cor]
    raw = [hier_paired_test(cor["SC_k8"], cor[c], B=B, seed=seed) for c in coord]
    adj = holm([r[1] for r in raw])
    out = {}
    for c, r, pa in zip(coord, raw, adj):
        out[c] = {"delta": r[0], "p": r[1], "ci": (r[2], r[3]), "holm": pa}
    return out


def mmlu_macro():
    """Micro- and macro-averaged MMLU accuracy per config, recovering each item's
    subject by mapping its released id (mmlu-i) to line i of tasks_data/mmlu.jsonl.
    Backs the paper's subject-composition check so those numbers regenerate."""
    subs = [json.loads(l)["subject"]
            for l in open(os.path.join(ROOT, "tasks_data", "mmlu.jsonl"))]
    rows = _load("mmlu")
    cor = per_item_correct(rows)
    out = {}
    for cfg, items in cor.items():
        micro = 100.0 * np.mean(list(items.values()))
        bysub = defaultdict(list)
        for iid, a in items.items():
            bysub[subs[int(iid.split("-")[1])]].append(a)
        macro = 100.0 * np.mean([np.mean(v) for v in bysub.values()])
        out[cfg] = {"micro": micro, "macro": macro, "n_subjects": len(bysub)}
    return out


def pareto_membership(task, configs=None, B=10000, seed=0, cost="total_tokens"):
    """Fraction of hierarchical resamples on which each config is Pareto-optimal
    (max accuracy, min mean cost; cost = total tokens unless `cost` names another
    record field, e.g. energy_j)."""
    rows = _loadf(task)
    cor = per_item_seed_correct(rows)
    tok = per_item_seed_tokens(rows, cost)
    if configs is None:
        configs = [c for c in cor if c in tok]
    items = sorted(set.intersection(*[set(cor[c]) for c in configs]))
    n = len(items)
    corA = {c: [cor[c][i] for i in items] for c in configs}
    tokA = {c: [tok[c][i] for i in items] for c in configs}
    rng = np.random.default_rng(seed)
    count = {c: 0 for c in configs}
    for t in range(B):
        idx = rng.integers(0, n, size=n)
        acc, cost = {}, {}
        for c in configs:
            a = corA[c]; tk = tokA[c]
            sa = 0.0; st = 0.0
            for ii in idx:
                av = a[ii]; tv = tk[ii]
                sa += av[rng.integers(0, len(av), size=len(av))].mean()
                st += tv[rng.integers(0, len(tv), size=len(tv))].mean()
            acc[c] = sa / n; cost[c] = st / n
        for c in configs:
            dominated = any(
                (acc[d] >= acc[c] and cost[d] <= cost[c]) and
                (acc[d] > acc[c] or cost[d] < cost[c])
                for d in configs if d != c)
            if not dominated:
                count[c] += 1
    return {c: 100.0 * count[c] / B for c in configs}


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    if args and args[0] == "hier":
        # Seed-aware hierarchical bootstrap + Holm for the headline comparisons,
        # plus GSM8K probabilistic Pareto membership. Regenerates the paper's
        # quoted significance and frontier-membership numbers from the logs.
        for task in (args[1:] or ["gsm8k", "math500", "mmlu"]):
            print(f"\n===== {task} (seed-aware hierarchical bootstrap, Holm) =====")
            res = headline_hier(task)
            for c, r in res.items():
                star = "  *SIG*" if r["holm"] < 0.05 else ""
                print(f"  SC_k8 vs {c:14} Δ={r['delta']:+5.1f}  "
                      f"CI[{r['ci'][0]:+5.1f},{r['ci'][1]:+5.1f}]  "
                      f"raw p={r['p']:.4f}  Holm={r['holm']:.4f}{star}")
        print("\n===== gsm8k Pareto-membership (% of resamples non-dominated) =====")
        cfgs = ["SA", "SC_k2", "SC_k4", "SC_k8", "SC_k12",
                "DEBATE_N2_R1", "DEBATE_N3_R1", "GE_R1", "GE_R2",
                "HIER_M2", "HIER_M3"]
        pm = pareto_membership("gsm8k", configs=cfgs)
        for c in cfgs:
            if c in pm:
                print(f"  {c:14} {pm[c]:5.1f}%")
        sys.exit(0)
    if args and args[0] == "mmlu-macro":
        print("MMLU micro vs macro accuracy (subjects via tasks_data/mmlu.jsonl):")
        mm = mmlu_macro()
        for c in ("SA", "SC_k2", "SC_k4", "SC_k8", "DEBATE_N3_R1", "GE_R2", "HIER_M3"):
            if c in mm:
                r = mm[c]
                print(f"  {c:14} micro={r['micro']:5.1f}  macro={r['macro']:5.1f}  "
                      f"(subjects={r['n_subjects']})")
        sys.exit(0)
    for task in (args or ["gsm8k", "mmlu", "math500"]):
        print(f"\n===== {task} =====")
        print("-- paired tests, item-only (Δacc pts, p) --")
        for k, (d, p) in headline(task).items():
            star = "  *SIG*" if p < 0.05 else ""
            print(f"  {k:22} Δ={d:+5.1f}  p={p:.3f}{star}")
        print("-- crossover (measured vs extrapolated) --")
        rep, scmax = crossover_report(task)
        print(f"  (SC measured up to {scmax:.0f} tokens)")
        for r in rep:
            tag = "EXTRAPOLATED" if r["extrapolated"] else "measured"
            print(f"  {r['config']:14} MAS={r['mas_acc']:5.1f}% @{r['mas_tok']:5.0f}  "
                  f"gap={r['gap']:+5.1f}  [{tag}]")
