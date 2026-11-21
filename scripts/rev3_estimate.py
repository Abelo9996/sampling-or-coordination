"""GPU-hour estimates for the rev3 rerun queue (protocols/REV3_RERUN_PROTOCOL.md).

Reads only existing logs; calls no model. Throughput model, measured on the
six-slot server (records with host rtx4060-pc-par6, campaign driver, 3 seeds x 2
shards = 6 concurrent request streams):

  per-call latency (s) = a + b * out_tokens + c * in_tokens     (least squares)

fitted separately for the 4k-context alias (qwen2.5:3b-instruct; GSM8K cap 640 and
MATH levels 1-3 cap 1,024) and the 8k alias (qwen2.5:3b-instruct-ctx8k; MATH levels
4-5 cap 2,048). Latencies were measured while six streams shared the GPU, so the
wall time of a run is (sum of predicted record latencies) / 6, times an overhead
factor for stream imbalance, stragglers at the end of a job and retries.

Token profiles of the runs to come are taken from the earlier runs of the same
configuration (MMLU, MMLU-Pro engineering: the July/September matrices; HumanEval:
results/humaneval_matrix_pc.jsonl). MATH levels 1-3 at cap 2,048: the per-call
tokens of phase B (cap 1,024) with every call that reached 1,024 tokens replaced by
the mean output of the calls of the same configuration that passed 1,024 tokens in
the levels 4-5 run (cap 2,048); input tokens of later calls are scaled by the
record's output growth.

Usage:  /usr/bin/python3 scripts/rev3_estimate.py [--json out.json]
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = lambda *a: os.path.join(ROOT, *a)  # noqa: E731
STREAMS = 6
OVERHEAD = 1.15


def jl(path):
    out = []
    for p in sorted(glob.glob(path)):
        with open(p) as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def fit(calls):
    X = np.array([[1.0, c["out_tokens"], c["in_tokens"]] for c in calls])
    y = np.array([c["latency_s"] for c in calls])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef


def pred_call(coef, out_t, in_t):
    return coef[0] + coef[1] * out_t + coef[2] * in_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    power = jl(P("results", "power", "*_qwen3b_s*_sh*.jsonl"))
    par6 = [r for r in power if r.get("host") == "rtx4060-pc-par6"]
    par4k = [r for r in par6 if r["model"] == "qwen2.5:3b-instruct"]
    par8k = [r for r in par6 if r["model"] == "qwen2.5:3b-instruct-ctx8k"
             and r["task"] == "math500" and r["cap"] == 2048
             and not r["config"].startswith("DU_")]
    c4 = fit([c for r in par4k for c in r["calls"]])
    c8 = fit([c for r in par8k for c in r["calls"]])
    rep = {"fit_4k": [float(x) for x in c4], "fit_8k": [float(x) for x in c8],
           "streams": STREAMS, "overhead": OVERHEAD}

    # measured per-config record rate on the six-slot server
    meas = defaultdict(list)
    for r in par6:
        meas[(r["task"], r["cap"], r["model"], r["config"])].append(r["latency_s"])
    rep["measured"] = {" ".join(map(str, k)): {
        "n": len(v), "lat_per_record_s": round(float(np.mean(v)), 2),
        "records_per_hour_6streams": round(STREAMS * 3600 / float(np.mean(v)), 1)}
        for k, v in sorted(meas.items(), key=str)}

    # out-of-sample check: fit on GSM8K only, predict MATH 1024 per-config means
    cg = fit([c for r in par4k if r["task"] == "gsm8k" for c in r["calls"]])
    chk = {}
    for cfg in ("SA", "SC_k12", "DEBATE_N3_R1", "GE_R2", "HIER_M3"):
        rs = [r for r in par4k if r["task"] == "math500" and r["config"] == cfg]
        if rs:
            p = np.mean([sum(pred_call(cg, c["out_tokens"], c["in_tokens"]) for c in r["calls"]) for r in rs])
            chk[cfg] = {"observed": round(float(np.mean([r["latency_s"] for r in rs])), 1),
                        "predicted_from_gsm8k_fit": round(float(p), 1)}
    rep["oos_check_math1024"] = chk

    def from_records(recs, coef, cfg_map, n_by_cfg):
        """mean predicted latency per record of each config, from per-record totals."""
        out = {}
        for new_cfg, old_cfg in cfg_map.items():
            rs = [r for r in recs if r["config"] == old_cfg]
            lat = np.mean([r["n_calls"] * coef[0] + r["out_tokens"] * coef[1]
                           + r["in_tokens"] * coef[2] for r in rs])
            out[new_cfg] = {"profile": old_cfg, "n_profile": len(rs),
                            "records": n_by_cfg[new_cfg],
                            "lat_per_record_s": round(float(lat), 2)}
        return out

    runs = {}
    # R1 GSM8K HIERFIX 500 x 3: measured HIER_M3 latency on the same server and alias
    g = [r for r in par4k if r["task"] == "gsm8k" and r["config"] == "HIER_M3"]
    runs["R1 gsm8k HIERFIX 500x3 cap640"] = {"HIERFIX_M3": {
        "profile": "HIER_M3 gsm8k par6 (measured)", "n_profile": len(g), "records": 1500,
        "lat_per_record_s": round(float(np.mean([r["latency_s"] for r in g])), 2)}}
    # R2 MMLU HIERFIX 80 x 2
    runs["R2 mmlu HIERFIX 80x2 cap640"] = from_records(
        jl(P("results", "mmlu_matrix.jsonl")), c4, {"HIERFIX_M3": "HIER_M3"}, {"HIERFIX_M3": 160})
    # R3 HumanEval 164 (SA seed 1; others seeds 1-2)
    he = jl(P("results", "humaneval_matrix_pc.jsonl"))
    runs["R3 humaneval 164 cap1024"] = from_records(
        he, c4, {"SA": "SA", "SC_k8": "SC_k8", "DEBATE_N3_R1": "DEBATE_N3_R1",
                 "GE_R2": "GE_R2", "HIERFIX_M3": "HIER_M3"},
        {"SA": 164, "SC_k8": 328, "DEBATE_N3_R1": 328, "GE_R2": 328, "HIERFIX_M3": 328})
    # R4 MMLU-Pro engineering HIERFIX 150 x 2
    runs["R4 mmlu_pro_eng HIERFIX 150x2 cap1024"] = from_records(
        jl(P("results", "mmlu_pro_eng_matrix.jsonl")), c4, {"HIERFIX_M3": "HIER_M3"},
        {"HIERFIX_M3": 300})
    # R5 MATH L1-3 at cap 2048, ctx8k alias, 238 x 3
    hard = [r for r in par8k]
    ext = {}
    for cfg in ("SA", "SC_k12", "DEBATE_N3_R1", "GE_R2", "HIER_M3"):
        long_ = [c["out_tokens"] for r in hard if r["config"] == cfg for c in r["calls"]
                 if c["out_tokens"] >= 1024]
        ext[cfg] = float(np.mean(long_)) if long_ else 1024.0
    rep["math_cap2048_extended_out_tokens"] = {k: round(v, 1) for k, v in ext.items()}
    b = [r for r in par4k if r["task"] == "math500" and r["cap"] == 1024]
    r5, capped = {}, {}
    nrec = {"SA": 238, "SC_k12": 714, "DEBATE_N3_R1": 714, "GE_R2": 714, "HIER_M3": 714}
    for cfg in nrec:
        lats, nc, nh = [], 0, 0
        for r in (x for x in b if x["config"] == cfg):
            outs = [ext[cfg] if c["out_tokens"] >= 1024 else c["out_tokens"] for c in r["calls"]]
            nh += sum(c["out_tokens"] >= 1024 for c in r["calls"])
            nc += len(r["calls"])
            grow = sum(outs) / max(1, r["out_tokens"])
            ins = [c["in_tokens"] if i == 0 or cfg in ("SA", "SC_k12") else c["in_tokens"] * grow
                   for i, c in enumerate(r["calls"])]
            lats.append(sum(pred_call(c8, o, i) for o, i in zip(outs, ins)))
        capped[cfg] = round(100 * nh / max(1, nc), 2)
        r5[cfg] = {"profile": f"{cfg} math500 L1-3 cap1024 par6, capped calls extended",
                   "n_profile": len(lats), "records": nrec[cfg],
                   "lat_per_record_s": round(float(np.mean(lats)), 2)}
    rep["math_l13_cap1024_share_calls_at_cap_pct"] = capped
    runs["R5 math500 L1-3 cap2048 ctx8k 238x3"] = r5

    total = 0.0
    for name, cfgs in runs.items():
        h = 0.0
        for cfg, d in cfgs.items():
            d["gpu_hours"] = round(d["records"] * d["lat_per_record_s"] / STREAMS / 3600 * OVERHEAD, 2)
            h += d["gpu_hours"]
        cfgs["_total_gpu_hours"] = round(h, 2)
        total += h
    rep["runs"] = runs
    rep["total_gpu_hours"] = round(total, 2)

    print(f"fit 4k alias: lat = {c4[0]:.3f} + {c4[1]:.4f}*out + {c4[2]:.5f}*in (s per call)")
    print(f"fit 8k alias: lat = {c8[0]:.3f} + {c8[1]:.4f}*out + {c8[2]:.5f}*in (s per call)")
    print("out-of-sample (GSM8K fit -> MATH 1024 per-record latency):", chk)
    print("MATH L1-3 cap1024 share of calls at cap (%):", capped)
    for name, cfgs in runs.items():
        print(f"{name}: {cfgs['_total_gpu_hours']:.2f} h")
        for cfg, d in cfgs.items():
            if cfg.startswith("_"):
                continue
            print(f"    {cfg:14s} {d['records']:5d} rec x {d['lat_per_record_s']:7.2f} s"
                  f" -> {d['gpu_hours']:6.2f} h  ({d['profile']})")
    print(f"TOTAL {total:.2f} GPU hours (6 streams, overhead x{OVERHEAD})")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rep, f, indent=1)


if __name__ == "__main__":
    main()
