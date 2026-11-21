"""Pre-specified analysis of the hard-MATH run (protocols/HARD_MATH_PROTOCOL.md) and
of the Du et al. debate on the same items (protocols/EXTRA_WORKFLOWS_PROTOCOL.md).

Written before the run finished; refuses to run tests on an incomplete run unless
--allow-incomplete is given (the protocol allows that only if the run cannot finish).
Writes results/extra/hard_math_analysis.json.

Usage: /usr/bin/python3 scripts/analyze_hard_math.py
"""
import argparse, glob, importlib.util, json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from harness import stats, tasks, topologies  # noqa: E402

spec = importlib.util.spec_from_file_location("camp", os.path.join(ROOT, "scripts/run_power_campaign.py"))
camp = importlib.util.module_from_spec(spec); spec.loader.exec_module(camp)

SEEDS = [1, 2, 3]
COORD = ["DEBATE_N3_R1", "GE_R2", "HIER_M3"]
KS = [2, 4, 8, 12]
MARGIN = 5.0


def load(pattern):
    recs = {}
    for p in sorted(glob.glob(os.path.join(ROOT, pattern))):
        for l in open(p):
            try:
                r = json.loads(l)
            except json.JSONDecodeError:
                continue
            recs.setdefault((r["config"], r["item_id"], r["seed"]), r)
    return recs


def boot_dist(a, b, B=10000, seed=0):
    """Bootstrap distribution of harness.stats.hier_paired_test (same draws)."""
    items = sorted(set(a) & set(b))
    A = [a[i] for i in items]; Bv = [b[i] for i in items]
    rng = np.random.default_rng(seed); n = len(items); boot = np.empty(B)
    for t in range(B):
        idx = rng.integers(0, n, size=n); s = 0.0
        for ii in idx:
            av, bv = A[ii], Bv[ii]
            s += av[rng.integers(0, len(av), size=len(av))].mean() - bv[rng.integers(0, len(bv), size=len(bv))].mean()
        boot[t] = s / n
    return boot * 100


def test(a, b):
    d, p, lo, hi = stats.hier_paired_test(a, b, B=10000, seed=0)
    boot = boot_dist(a, b)
    l2, h2 = np.percentile(boot, [2.5, 97.5])
    assert abs(l2 - lo) < 1e-9 and abs(h2 - hi) < 1e-9, "boot_dist does not replicate hier_paired_test"
    l90, h90 = np.percentile(boot, [5, 95])
    return {"diff": round(d, 2), "p": float(p), "ci95": [round(lo, 2), round(hi, 2)],
            "ci90": [round(float(l90), 2), round(float(h90), 2)],
            "p_any_gain": float((boot >= 0).mean()),          # one-sided, H0: diff >= 0
            "p_gain_margin": float((boot >= MARGIN).mean()),  # one-sided, H0: diff >= +5
            "p_loss_margin": float((boot <= -MARGIN).mean())}  # one-sided, H0: diff <= -5


def holm_family(rows, key="p"):
    names = list(rows)
    adj = stats.holm([rows[n][key] for n in names])
    for n, a in zip(names, adj):
        rows[n][key + "_holm"] = float(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-incomplete", action="store_true")
    a = ap.parse_args()
    task = tasks.get_task("math500")
    items = camp.load_math_levels(tasks, 4, 5, 262)
    level = {it["id"]: it["level"] for it in items}
    recs = load("results/power/math500hard_qwen3b_s*.jsonl")
    du = load("results/power/extra_math45_qwen3b_s*.jsonl")

    def complete(i):
        return (("SA", i, 1) in recs and all((c, i, s) in recs for c in ["SC_k12"] + COORD for s in SEEDS))
    ids = [it["id"] for it in items if complete(it["id"])]
    out = {"n_items_total": len(items), "n_items_complete": len(ids), "complete": len(ids) == len(items)}
    if not out["complete"] and not a.allow_incomplete:
        raise SystemExit(f"run incomplete: {len(ids)} of {len(items)} items; tests not run")

    def mat(D, cfg, key="correct"):
        if cfg == "SA":
            return {i: np.array([float(bool(D[(cfg, i, 1)][key]))] * 3) for i in ids}
        return {i: np.array([float(bool(D[(cfg, i, s)][key])) for s in SEEDS]) for i in ids}

    M, cost = {}, {}
    def add_cost(name, tot, outt, none, cap):
        cost[name] = {"total_tokens": round(float(np.mean(tot)), 1), "out_tokens": round(float(np.mean(outt)), 1),
                      "no_answer_pct": round(100 * float(np.mean(none)), 2),
                      "records_with_capped_call_pct": round(100 * float(np.mean(cap)), 2)}
    for cfg in ["SA"] + COORD:
        M[cfg] = mat(recs, cfg)
        R = [recs[(cfg, i, s)] for i in ids for s in ([1] if cfg == "SA" else SEEDS)]
        add_cost(cfg, [sum(c["in_tokens"] + c["out_tokens"] for c in r["calls"]) for r in R],
                 [sum(c["out_tokens"] for c in r["calls"]) for r in R],
                 [r["pred"] is None for r in R], [any(c["hit_cap"] for c in r["calls"]) for r in R])
    for k in KS:
        m, tot, outt, none, cap = {}, [], [], [], []
        for i in ids:
            v = []
            for s in SEEDS:
                r = recs[("SC_k12", i, s)]; cs = r["calls"][:k]
                ans = topologies._majority([c["extracted"] for c in cs])
                ok = bool(task["score"](ans, r["gold"]))
                if k == 12:
                    assert ok == bool(r["correct"]), (i, s)
                v.append(float(ok)); tot.append(sum(c["in_tokens"] + c["out_tokens"] for c in cs))
                outt.append(sum(c["out_tokens"] for c in cs)); none.append(ans is None)
                cap.append(any(c["hit_cap"] for c in cs))
            m[i] = np.array(v)
        M[f"SC_k{k}"] = m; add_cost(f"SC_k{k}", tot, outt, none, cap)

    acc = lambda m, sub=None: round(100 * float(np.mean([m[i].mean() for i in (sub or ids)])), 2)
    out["accuracy"] = {k: acc(v) for k, v in M.items()}
    out["cost"] = cost
    out["accuracy_by_level"] = {str(L): {k: acc(v, [i for i in ids if level[i] == L]) for k, v in M.items()} for L in (4, 5)}
    out["n_by_level"] = {str(L): sum(level[i] == L for i in ids) for L in (4, 5)}

    def comparator(w, key):
        ok = [k for k in KS if cost[f"SC_k{k}"][key] <= cost[w][key]]
        return f"SC_k{max(ok)}" if ok else None

    for key, label in (("total_tokens", "primary_total_tokens"), ("out_tokens", "output_tokens")):
        fam = {}
        for w in COORD:
            c = comparator(w, key)
            if c is None:
                fam[w] = {"comparator": None, "note": "no SC k at or below the workflow's cost"}
                continue
            fam[w] = {"comparator": c, **test(M[w], M[c])}
        live = {w: r for w, r in fam.items() if r.get("comparator")}
        for kk in ("p", "p_any_gain", "p_gain_margin", "p_loss_margin"):
            holm_family(live, kk)
        out[label] = fam
    out["vs_SC_k8"] = {w: test(M[w], M["SC_k8"]) for w in COORD}
    out["vs_SC_k12"] = {w: test(M[w], M["SC_k12"]) for w in COORD}
    out["vs_SA"] = {w: test(M[w], M["SA"]) for w in COORD}
    for famname in ("vs_SC_k8", "vs_SC_k12", "vs_SA"):
        for kk in ("p", "p_any_gain", "p_gain_margin"):
            holm_family(out[famname], kk)
    out["SC_vs_SA"] = {f"SC_k{k}": test(M[f"SC_k{k}"], M["SA"]) for k in KS}

    # secondary interaction inside this run: (W - SC k8 on level 5) - (same on level 4)
    def strat_boot(w, sub, B=10000, seed=0):
        return boot_dist({i: M[w][i] for i in sub}, {i: M["SC_k8"][i] for i in sub}, B=B, seed=seed)
    L4 = [i for i in ids if level[i] == 4]; L5 = [i for i in ids if level[i] == 5]
    out["interaction_level5_minus_level4"] = {}
    for w in COORD:
        b5 = strat_boot(w, L5, seed=0); b4 = strat_boot(w, L4, seed=1)  # independent strata
        d = b5 - b4
        obs = (acc(M[w], L5) - acc(M["SC_k8"], L5)) - (acc(M[w], L4) - acc(M["SC_k8"], L4))
        lo, hi = np.percentile(d, [2.5, 97.5])
        out["interaction_level5_minus_level4"][w] = {
            "obs": round(obs, 2), "ci95": [round(float(lo), 2), round(float(hi), 2)],
            "p": float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))}

    # Du et al. debate on the same items (EXTRA_WORKFLOWS_PROTOCOL.md)
    du_ok = all(("DU_DEBATE_N3_R2", i, s) in du for i in ids for s in SEEDS)
    out["du_complete"] = bool(du_ok)
    if du_ok:
        Mdu = mat(du, "DU_DEBATE_N3_R2"); Mduh = mat(du, "DU_DEBATE_N3_R2", "correct_harness")
        R = [du[("DU_DEBATE_N3_R2", i, s)] for i in ids for s in SEEDS]
        C = [c for r in R for c in r["calls"]]
        out["du"] = {
            "accuracy": acc(Mdu), "accuracy_harness_scoring": acc(Mduh),
            "accuracy_by_level": {str(L): acc(Mdu, [i for i in ids if level[i] == L]) for L in (4, 5)},
            "total_tokens": round(float(np.mean([r["total_tokens"] for r in R])), 1),
            "out_tokens": round(float(np.mean([r["out_tokens"] for r in R])), 1),
            "none_vote_pct": round(100 * float(np.mean([r["du_vote_none"] for r in R])), 2),
            "calls_at_cap_pct": round(100 * float(np.mean([c["hit_cap"] for c in C])), 2),
            "calls_ge_8192": int(sum(c["in_tokens"] + c["out_tokens"] >= 8192 for c in C)),
            "max_in_plus_out": int(max(c["in_tokens"] + c["out_tokens"] for c in C)),
            "minus_SC_k8": test(Mdu, M["SC_k8"]), "minus_SC_k8_harness_scoring": test(Mduh, M["SC_k8"]),
            "minus_DEBATE_N3_R1": test(Mdu, M["DEBATE_N3_R1"]), "minus_SA": test(Mdu, M["SA"]),
            "minus_SC_k12": test(Mdu, M["SC_k12"]),
        }
        g = os.path.join(ROOT, "results/extra/extra_gsm8k_analysis.json")
        if os.path.exists(g):
            G = json.load(open(g))
            names = ["GSM8K DU - SC_k8", "GSM8K HIERFIX - SC_k8", "MATH45 DU - SC_k8"]
            ps = [G["primary_gsm8k"]["DU - SC_k8"]["p"], G["primary_gsm8k"]["HIERFIX - SC_k8"]["p"],
                  out["du"]["minus_SC_k8"]["p"]]
            out["extra_primary_family_holm"] = dict(zip(names, [float(x) for x in stats.holm(ps)]))
            names5 = ["GSM8K DU - DEBATE", "GSM8K HIERFIX - HIER", "GSM8K DU - SA", "GSM8K HIERFIX - SA", "MATH45 DU - DEBATE"]
            ps5 = [G["secondary_gsm8k"][k]["p"] for k in ("DU - DEBATE", "HIERFIX - HIER", "DU - SA", "HIERFIX - SA")] \
                + [out["du"]["minus_DEBATE_N3_R1"]["p"]]
            out["extra_secondary_family_holm"] = dict(zip(names5, [float(x) for x in stats.holm(ps5)]))

    os.makedirs(os.path.join(ROOT, "results/extra"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "results/extra/hard_math_analysis.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
