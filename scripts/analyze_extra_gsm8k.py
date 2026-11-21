"""Analysis of the GSM8K extra-workflow run per protocols/EXTRA_WORKFLOWS_PROTOCOL.md
(DU_DEBATE_N3_R2 and HIERFIX_M3 against SC_k8, DEBATE_N3_R1, HIER_M3 and SA on the
same 100 items, seeds 1 to 3). Writes results/extra/extra_gsm8k_analysis.json."""
import glob, json, os, re, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from harness.stats import hier_paired_test

def load(pat):
    out = {}
    for p in sorted(glob.glob(os.path.join(ROOT, pat))):
        for l in open(p):
            try: r = json.loads(l)
            except json.JSONDecodeError: continue
            out[(r["config"], r["item_id"], r["seed"])] = r
    return out

X = load("results/power/extra_gsm8k_qwen3b_s[123].jsonl")
B = load("results/power/gsm8k_qwen3b_s[123].jsonl")
ids = sorted({k[1] for k in X})
assert len(ids) == 100, len(ids)
SEEDS = [1, 2, 3]

def mat(D, cfg, key="correct"):
    if cfg == "SA":
        return {i: np.array([float(bool(D[(cfg, i, 1)][key]))] * 3) for i in ids}
    return {i: np.array([float(bool(D[(cfg, i, s)][key])) for s in SEEDS]) for i in ids}

def recs(D, cfg):
    return [D[(cfg, i, s)] for i in ids for s in ([1] if cfg == "SA" else SEEDS)]

def cost(D, cfg):
    R = recs(D, cfg)
    return {k: round(float(np.mean([r[k] for r in R])), 1) for k in ("in_tokens", "out_tokens", "total_tokens")}

def acc(m): return round(100 * float(np.mean([v.mean() for v in m.values()])), 2)

M = {"DU": mat(X, "DU_DEBATE_N3_R2"), "DU_harness": mat(X, "DU_DEBATE_N3_R2", "correct_harness"),
     "HIERFIX": mat(X, "HIERFIX_M3"), "SC_k8": mat(B, "SC_k8"), "DEBATE": mat(B, "DEBATE_N3_R1"),
     "HIER": mat(B, "HIER_M3"), "SA": mat(B, "SA")}
res = {"n_items": len(ids), "seeds": SEEDS, "accuracy": {k: acc(v) for k, v in M.items()},
       "cost": {"DU": cost(X, "DU_DEBATE_N3_R2"), "HIERFIX": cost(X, "HIERFIX_M3"), "SC_k8": cost(B, "SC_k8"),
                "DEBATE": cost(B, "DEBATE_N3_R1"), "HIER": cost(B, "HIER_M3"), "SA": cost(B, "SA")}}

def boot90(a, b, B_=10000, seed=0):
    """90% interval from the same bootstrap draws as harness.stats.hier_paired_test
    (items resampled, seeds resampled within items independently for each arm)."""
    k = sorted(set(a) & set(b)); A = [a[i] for i in k]; Bv = [b[i] for i in k]
    rng = np.random.default_rng(seed); n = len(k); boot = np.empty(B_)
    for t in range(B_):
        idx = rng.integers(0, n, size=n); s_ = 0.0
        for ii in idx:
            av, bv = A[ii], Bv[ii]
            s_ += av[rng.integers(0, len(av), size=len(av))].mean() - bv[rng.integers(0, len(bv), size=len(bv))].mean()
        boot[t] = s_ / n
    return [round(100 * float(np.percentile(boot, q)), 2) for q in (5, 95)]

def test(a, b):
    d, p, lo, hi = hier_paired_test(M[a], M[b], B=10000, seed=0)
    l90, h90 = boot90(M[a], M[b])
    return {"diff": round(d, 2), "p": float(p), "ci95": [round(lo, 2), round(hi, 2)], "ci90": [l90, h90],
            "gain5_excluded_unadj": h90 < 5, "any_gain_excluded_unadj": h90 < 0}

res["primary_gsm8k"] = {"DU - SC_k8": test("DU", "SC_k8"), "HIERFIX - SC_k8": test("HIERFIX", "SC_k8")}
res["primary_secondary_scoring"] = {"DU_harness - SC_k8": test("DU_harness", "SC_k8")}
res["secondary_gsm8k"] = {"DU - DEBATE": test("DU", "DEBATE"), "HIERFIX - HIER": test("HIERFIX", "HIER"),
                          "DU - SA": test("DU", "SA"), "HIERFIX - SA": test("HIERFIX", "SA")}

LIT = re.compile(r"The answer is N\b")
def hier_check(D, cfg):
    """literal = the aggregator text contains the unfilled template line; many such
    records still carry a number the harness extractor reads (e.g. 'where N is 22'),
    so no_answer is the harness extractor returning None on the aggregator output."""
    R = recs(D, cfg)
    lit = np.array([bool(LIT.search(r["calls"][-1]["text"])) for r in R])
    none = np.array([r["pred"] is None for r in R])
    ok = np.array([bool(r["correct"]) for r in R])
    pc = lambda x: round(100 * float(np.mean(x)), 2)
    return {"n": len(R), "literal_template_pct": pc(lit), "no_answer_pct": pc(none),
            "n_answered": int((~none).sum()), "acc_among_answered": pc(ok[~none]),
            "acc_literal_records": pc(ok[lit]), "acc_nonliteral_records": pc(ok[~lit]),
            "agg_cap_pct": pc([r["calls"][-1]["hit_cap"] for r in R])}
res["hier_checks"] = {"HIERFIX_M3": hier_check(X, "HIERFIX_M3"), "HIER_M3": hier_check(B, "HIER_M3")}
R = recs(X, "DU_DEBATE_N3_R2"); C = [c for r in R for c in r["calls"]]
res["du_checks"] = {"none_vote_pct": round(100 * np.mean([r["du_vote_none"] for r in R]), 2),
                    "calls": len(C), "calls_at_cap_pct": round(100 * np.mean([c["hit_cap"] for c in C]), 2),
                    "final_calls_at_cap_pct": round(100 * np.mean([c["hit_cap"] for r in R for c in r["calls"][3:]]), 2),
                    "calls_ge_8192": int(sum(c["in_tokens"] + c["out_tokens"] >= 8192 for c in C)),
                    "max_in_plus_out": int(max(c["in_tokens"] + c["out_tokens"] for c in C)),
                    "du_parse_none_final_pct": round(100 * np.mean([p is None for r in R for p in r["du_parsed_final"]]), 2),
                    "scoring_disagree_records": int(sum(bool(r["correct"]) != bool(r["correct_harness"]) for r in R))}
RB = recs(B, "DEBATE_N3_R1")
res["debate_harness_checks"] = {"final_calls_at_cap_pct": round(100 * np.mean([c["hit_cap"] for r in RB for c in r["calls"][3:]]), 2)}
json.dump(res, open(os.path.join(ROOT, "results/extra/extra_gsm8k_analysis.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
