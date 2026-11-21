"""Cost analyses added for the second external review of arxiv_rev3 (exploratory, unadjusted):

1. Price-ratio crossover of SC k=8 against debate with a bootstrap interval. For output:input
   price ratio r, cost = input + r x output tokens per query; SC k=8 is cheaper than debate iff
   r < r* = (in_debate - in_SC8) / (out_SC8 - out_debate). Interval: items resampled with
   replacement, then seeds within each resampled item, independently for each configuration
   (the scheme of harness.stats.hier_paired_test), 10,000 replicates, seed 0, percentile 95%.
   A replicate in which SC k=8 has no more output tokens than debate (and fewer input tokens)
   has r* = infinity (SC k=8 cheaper at every ratio); the share of such replicates is reported.
   Runs: GSM8K 500 x 3 (phase A) and MATH levels 1 to 3, 238 x 3, cap 1,024 (phase B; SC k=8 =
   first 8 samples of SC k=12). The same for SC k=4 (in- and output-token differences).
2. Modeled energy per query, from the least-squares fit of per-call GPU energy in the first
   HumanEval run (J = c + b_in x in + a_out x out per call; refitted here exactly as
   paper/arxiv_rev3/scripts/fig_energy.py and asserted against its macros). Per query:
   a_out x output tokens + b_in x input tokens + c x number of calls. MODELED, not measured: it is
   price-weighted tokens at ratio a_out/b_in plus a per-call term. For every configuration of the
   headline runs (GSM8K 500 x 3, MMLU matrix + HIERFIX, MATH 238 x 3, MMLU-Pro engineering matrix
   + HIERFIX, HumanEval rerun): mean modeled J per query, and for each workflow and each sampling
   configuration the item-bootstrap 95% interval of (sampling minus workflow) modeled J
   (harness.analyze._bootstrap_ci, B=10,000, seed 0, as the token intervals of the paper).
   The verdicts themselves are formed in paper/arxiv_rev3/scripts/emit_rev3_reruns.py from these
   costs and the accuracy intervals already in results/extra/rev3_reruns_analysis.json and
   paper/arxiv_rev3/tables/math_easy.json (accuracy does not depend on the cost unit).
Writes results/extra/rev3_costs_analysis.json.
Usage: /usr/bin/python3 scripts/analyze_rev3_costs.py
"""
import glob, json, os, re, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from harness import analyze, tasks  # noqa: E402

OUT = os.path.join(ROOT, "results/extra/rev3_costs_analysis.json")
B = 10000


def load(patterns):
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


def rec_cost(r, k=None):
    """(input tokens, output tokens, number of calls) of a record, or of its first k calls."""
    cs = r.get("calls")
    if cs:
        cs = cs[:k] if k else cs
        return (sum(c["in_tokens"] for c in cs), sum(c["out_tokens"] for c in cs), len(cs))
    assert k is None
    return (r["in_tokens"], r["out_tokens"], r["n_calls"])


def costs(recs, cfg, ids, seeds, k=None, src=None):
    """item -> array (n_seeds, 3) of (in, out, calls); SA one seed."""
    src = src or cfg
    ss = [1] if cfg == "SA" else seeds
    return {i: np.array([rec_cost(recs[(src, i, s)], k) for s in ss], float) for i in ids}


def fit_energy():
    rows = [json.loads(l) for l in open(os.path.join(ROOT, "results/humaneval_matrix_pc.jsonl")) if l.strip()]
    calls = [c for r in rows for c in r["calls"]]
    X = np.array([[1.0, c["in_tokens"], c["out_tokens"]] for c in calls])
    y = np.array([c["energy_j"] for c in calls])
    with np.errstate(all="ignore"):
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        r2 = float(1 - ((y - X @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    mac = {m.group(1): m.group(2) for m in re.finditer(r"\\newcommand\{\\(\w+)\}\{(.*)\}",
           open(os.path.join(ROOT, "paper/arxiv_rev3/humaneval_macros.tex")).read())}
    assert f"{beta[2]:.2f}" == mac["regOutCode"] and f"{beta[1]:.3f}" == mac["regInCode"]
    assert f"{beta[0]:.1f}" == mac["regIcptCode"] and f"{r2:.3f}" == mac["regRsqCode"]
    return {"intercept_J_per_call": float(beta[0]), "J_per_input_token": float(beta[1]),
            "J_per_output_token": float(beta[2]), "r2": r2, "n_calls": len(calls),
            "ratio_out_in": float(beta[2] / beta[1])}


def crossover(A, Bc, ids, seed=0):
    """A = SC configuration, Bc = debate: item -> (n_seeds, 3). Returns point r* and bootstrap."""
    a = np.stack([A[i] for i in ids]); b = np.stack([Bc[i] for i in ids])  # (n, s, 3)
    def rstar(din, dout):  # din = in_deb - in_sc, dout = out_sc - out_deb
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(dout > 0, din / dout, np.inf)
        return r
    ma, mb = a.mean(axis=(0, 1)), b.mean(axis=(0, 1))
    point = float(rstar(mb[0] - ma[0], ma[1] - mb[1]))
    rng = np.random.default_rng(seed)
    n, sa, sb = a.shape[0], a.shape[1], b.shape[1]
    idx = rng.integers(0, n, size=(B, n))
    ia = rng.integers(0, sa, size=(B, n, sa)); ib = rng.integers(0, sb, size=(B, n, sb))
    A_ = a[idx[..., None], ia]  # (B, n, sa, 3)
    B_ = b[idx[..., None], ib]
    ma_, mb_ = A_.mean(axis=(1, 2)), B_.mean(axis=(1, 2))
    rs = rstar(mb_[:, 0] - ma_[:, 0], ma_[:, 1] - mb_[:, 1])
    share_inf = float(np.mean(np.isinf(rs)))
    if share_inf > 0.025:  # the percentile interval reaches infinity
        fin = rs[np.isfinite(rs)]
        ci = [float(np.percentile(rs, 2.5)) if share_inf < 0.975 else "inf", "inf"]
    else:
        lo, hi = np.percentile(rs, [2.5, 97.5]); ci = [float(lo), float(hi)]
    return {"point": point if np.isfinite(point) else "inf", "ci95": ci, "share_infinite": share_inf,
            "d_in_debate_minus_sc": float(mb[0] - ma[0]), "d_out_sc_minus_debate": float(ma[1] - mb[1])}


def main():
    E = fit_energy()
    a, b, c = E["J_per_output_token"], E["J_per_input_token"], E["intercept_J_per_call"]
    S3, S2 = [1, 2, 3], [1, 2]
    runs = {}
    # GSM8K 500 x 3 (phase A + R1)
    gids = [it["id"] for it in tasks.get_task("gsm8k")["load"](n=500, split="all")]
    PA = load(["results/power/gsm8k_qwen3b_s[123].jsonl", "results/power/gsm8k_qwen3b_s[123]_sh*of2.jsonl"])
    R1 = load(["results/rev3/gsm8k_qwen3b_s*.jsonl"])
    g = {"SA": costs(PA, "SA", gids, S3)}
    for k in (2, 4, 8):
        g[f"SC_k{k}"] = costs(PA, f"SC_k{k}", gids, S3, k=k, src="SC_k8")
    for w in ("DEBATE_N3_R1", "GE_R2"):
        g[w] = costs(PA, w, gids, S3)
    g["HIERFIX_M3"] = costs(R1, "HIERFIX_M3", gids, S3)
    runs["gsm8k"] = (g, gids)
    # MATH levels 1 to 3, 238 x 3, cap 1,024 (phase B); HIER = HIERFIX by the protocol
    mids = [it["id"] for it in tasks.get_task("math500")["load"](n=238, split="all")]
    PB = load(["results/power/math500_qwen3b_s*.jsonl"])
    m = {"SA": costs(PB, "SA", mids, S3)}
    for k in (2, 4, 8, 12):
        m[f"SC_k{k}"] = costs(PB, f"SC_k{k}", mids, S3, k=k, src="SC_k12")
    for w in ("DEBATE_N3_R1", "GE_R2"):
        m[w] = costs(PB, w, mids, S3)
    m["HIERFIX_M3"] = costs(PB, "HIERFIX_M3", mids, S3, src="HIER_M3")
    runs["math"] = (m, mids)
    # MMLU (July matrix + R2), MMLU-Pro engineering (matrix + R4)
    for name, mat, rr in (("mmlu", "results/mmlu_matrix.jsonl", "results/rev3/mmlu_qwen3b_*.jsonl"),
                          ("proeng", "results/mmlu_pro_eng_matrix.jsonl", "results/rev3/mmlu_pro_eng_qwen3b_*.jsonl")):
        M = load([mat]); R = load([rr])
        ids = sorted({k[1] for k in M})
        d = {cfg: costs(M, cfg, ids, S2) for cfg in ("SA", "SC_k2", "SC_k4", "SC_k8", "DEBATE_N3_R1", "GE_R2")}
        d["HIERFIX_M3"] = costs(R, "HIERFIX_M3", ids, S2)
        runs[name] = (d, ids)
    # HumanEval rerun (R3)
    R3 = load(["results/rev3/humaneval_qwen3b_*.jsonl"])
    hids = [it["id"] for it in tasks.get_task("humaneval")["load"](n=None, split="all")]
    h = {"SA": costs(R3, "SA", hids, S2)}
    for k in (2, 4, 8):
        h[f"SC_k{k}"] = costs(R3, f"SC_k{k}", hids, S2, k=k, src="SC_k8")
    for w in ("DEBATE_N3_R1", "GE_R2", "HIERFIX_M3"):
        h[w] = costs(R3, w, hids, S2)
    runs["code"] = (h, hids)

    J = {"energy_fit": E, "note": "modeled from the first HumanEval run's per-call fit; not measured", "runs": {}, "crossover": {}}
    model = lambda x: a * x[:, 1] + b * x[:, 0] + c * x[:, 2]  # per record
    for name, (d, ids) in runs.items():
        per_item = {cfg: {i: float(model(v[i]).mean()) for i in ids} for cfg, v in d.items()}
        tok = {cfg: {i: (float(v[i][:, 0].mean()), float(v[i][:, 1].mean())) for i in ids} for cfg, v in d.items()}
        R = {"n_items": len(ids), "configs": {}, "diffs": {}}
        for cfg in d:
            R["configs"][cfg] = {"modeled_J": float(np.mean(list(per_item[cfg].values()))),
                                 "in_tokens": float(np.mean([tok[cfg][i][0] for i in ids])),
                                 "out_tokens": float(np.mean([tok[cfg][i][1] for i in ids]))}
        samp = [x for x in d if x == "SA" or x.startswith("SC_")]
        for w in ("DEBATE_N3_R1", "GE_R2", "HIERFIX_M3"):
            for s in samp:
                diff = np.array([per_item[s][i] - per_item[w][i] for i in ids])
                lo, hi = analyze._bootstrap_ci(diff, B=10000)
                R["diffs"][f"{s}|{w}"] = {"mean": float(diff.mean()), "ci95": [lo, hi]}
        J["runs"][name] = R
    for name in ("gsm8k", "math"):
        d, ids = runs[name]
        J["crossover"][name] = {"SC_k8": crossover(d["SC_k8"], d["DEBATE_N3_R1"], ids),
                                "SC_k4": crossover(d["SC_k4"], d["DEBATE_N3_R1"], ids)}
    json.dump(J, open(OUT, "w"), indent=1)
    print(json.dumps(J["crossover"], indent=1))
    for n, R in J["runs"].items():
        print(n, {k: round(v["modeled_J"]) for k, v in R["configs"].items()})


if __name__ == "__main__":
    main()
