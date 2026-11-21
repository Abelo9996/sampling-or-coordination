"""Primary difficulty interaction of protocols/HARD_MATH_PROTOCOL.md:
D = (W minus SC k=8 on MATH levels 4 to 5) minus (W minus SC k=8 on levels 1 to 3,
campaign phase B), W = debate (prediction D > 0), and GE and HIER without a prediction.
Items are resampled within each stratum independently, seeds within items; 10,000
replicates; two-sided 95% interval. The strata differ in token cap (2,048 vs 1,024).

Written and committed before phase B finished; refuses to run on incomplete data.
Writes results/extra/math_interaction_analysis.json.
"""
import importlib.util, json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from harness import tasks, topologies  # noqa: E402

spec = importlib.util.spec_from_file_location("hm", os.path.join(ROOT, "scripts/analyze_hard_math.py"))
hm = importlib.util.module_from_spec(spec); spec.loader.exec_module(hm)
SEEDS = [1, 2, 3]
COORD = ["DEBATE_N3_R1", "GE_R2", "HIER_M3"]


def stratum(pattern, items):
    task = tasks.get_task("math500")
    recs = hm.load(pattern)
    ids = [it["id"] for it in items]
    missing = [(c, i, s) for i in ids for c in ["SC_k12"] + COORD for s in SEEDS if (c, i, s) not in recs]
    missing += [("SA", i, 1) for i in ids if ("SA", i, 1) not in recs]
    if missing:
        raise SystemExit(f"{pattern}: incomplete, {len(missing)} records missing; tests not run")
    M = {"SA": {i: np.array([float(bool(recs[("SA", i, 1)]["correct"]))] * 3) for i in ids}}
    for c in COORD:
        M[c] = {i: np.array([float(bool(recs[(c, i, s)]["correct"])) for s in SEEDS]) for i in ids}
    m = {}
    for i in ids:
        v = []
        for s in SEEDS:
            r = recs[("SC_k12", i, s)]
            ans = topologies._majority([c["extracted"] for c in r["calls"][:8]])
            v.append(float(bool(task["score"](ans, r["gold"]))))
        m[i] = np.array(v)
    M["SC_k8"] = m
    acc = {k: round(100 * float(np.mean([v[i].mean() for i in ids])), 2) for k, v in M.items()}
    caps = sorted({recs[("SC_k12", i, 1)]["cap"] for i in ids})
    return M, acc, len(ids), caps


def main():
    hard_items = hm.camp.load_math_levels(tasks, 4, 5, 262)
    easy_items = tasks.get_task("math500")["load"](n=238, split="all")
    Mh, acch, nh, caph = stratum("results/power/math500hard_qwen3b_s*.jsonl", hard_items)
    Me, acce, ne, cape = stratum("results/power/math500_qwen3b_s*.jsonl", easy_items)
    out = {"hard": {"n_items": nh, "cap": caph, "accuracy": acch}, "easy": {"n_items": ne, "cap": cape, "accuracy": acce},
           "within_stratum": {}, "interaction_hard_minus_easy": {}}
    for w in COORD:
        out["within_stratum"][w] = {"hard_minus_SC_k8": hm.test(Mh[w], Mh["SC_k8"]),
                                    "easy_minus_SC_k8": hm.test(Me[w], Me["SC_k8"]),
                                    "easy_minus_SA": hm.test(Me[w], Me["SA"])}
        bh = hm.boot_dist(Mh[w], Mh["SC_k8"], seed=0)
        be = hm.boot_dist(Me[w], Me["SC_k8"], seed=1)  # independent strata
        d = bh - be
        obs = (acch[w] - acch["SC_k8"]) - (acce[w] - acce["SC_k8"])
        lo, hi = np.percentile(d, [2.5, 97.5])
        out["interaction_hard_minus_easy"][w] = {
            "obs": round(obs, 2), "ci95": [round(float(lo), 2), round(float(hi), 2)],
            "p": float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))),
            "p_one_sided_D_le_0": float((d <= 0).mean())}
    out["easy_SC_k8_minus_SA"] = hm.test(Me["SC_k8"], Me["SA"])
    os.makedirs(os.path.join(ROOT, "results/extra"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "results/extra/math_interaction_analysis.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
