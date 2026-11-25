"""Measured GPU energy on GSM8K (results/energy/gsm8k_energy_pc.jsonl, scripts/run_gsm8k_energy.py).

Run: RTX 4060 PC, Ollama 0.40.1, qwen2.5:3b-instruct (4-bit), one request at a time, board power
from NVML (about one sample per 20 ms), joules per call by trapezoid integration over the call's
interval (harness/energy.py, as in the first HumanEval run); first 130 items of the main study's
500 GSM8K items, seed 1, cap 640; SA, SC k=8 (k=2, 4 = its first calls), DEBATE_N3_R1, GE_R2,
HIERFIX_M3. Idle baselines (60 s) in results/energy/gsm8k_energy_pc_idle.jsonl. Every record lists
the other GPU compute processes and Windows games seen at its end; the script requires both empty.

1. Per-call fit, as for the HumanEval run in the paper (scripts/analyze_rev3_costs.fit_energy):
   gross joules per call J = c + b_in x in + a_out x out, least squares with the intercept
   constrained to c >= 0 (if the unconstrained c < 0, refit with c = 0); also the unconstrained fit
   and the same fits on idle-subtracted (net) joules per call (gross minus start idle power times
   the call's wall-clock time). Ratio a_out / b_in with a 95% interval from 10,000 item bootstrap
   replicates (all calls of an item resampled together; numpy default_rng(0)).
2. Measured joules per query per configuration (gross, as the HumanEval figures; net as a check),
   against the energy modeled for the same calls from the HumanEval fit of the paper
   (results/extra/rev3_costs_analysis.json "energy_fit").
3. Cost-matched dominance verdicts (R/C/N; scripts/analyze_rev3_reruns functions, hierarchical
   paired test, 10,000 replicates, seed 0, one seed per item) on these 130 items, with cost =
   measured gross joules, measured net joules, modeled joules (HumanEval fit) and total tokens.

Writes results/extra/gsm8k_energy_analysis.json and, in each --outdir (default paper/eaai_rev3 if
present, else paper/arxiv_rev3), gsm8k_energy_macros.tex and tables/gsm8k_energy_table.tex.
Run: /usr/bin/python3 scripts/analyze_gsm8k_energy.py [--jobs 8] [--outdir DIR ...]
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from harness import tasks, topologies  # noqa: E402


def _mod(name, path):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, path))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


rr = _mod("rr", "scripts/analyze_rev3_reruns.py")
DATA = os.path.join(ROOT, "results/energy/gsm8k_energy_pc.jsonl")
IDLE = os.path.join(ROOT, "results/energy/gsm8k_energy_pc_idle.jsonl")
FIT = os.path.join(ROOT, "results/extra/rev3_costs_analysis.json")
OUT_JSON = os.path.join(ROOT, "results/extra/gsm8k_energy_analysis.json")
EAAI, ARX = os.path.join(ROOT, "paper/eaai_rev3"), os.path.join(ROOT, "paper/arxiv_rev3")
SAMP = ["SA", "SC_k2", "SC_k4", "SC_k8"]
COORD = ["DEBATE_N3_R1", "GE_R2", "HIERFIX_M3"]
ALL = SAMP + COORD
UNITS = {"measured": "measured gross J", "net": "measured net J", "modeled": "modeled J (HumanEval fit)",
         "tokens": "total tokens"}


def fit(I, O, y, nonneg=True):
    X = np.column_stack([np.ones_like(I), I, O])
    with np.errstate(all="ignore"):
        b = np.linalg.lstsq(X, y, rcond=None)[0]
        if nonneg and b[0] < 0:
            b = np.r_[0.0, np.linalg.lstsq(X[:, 1:], y, rcond=None)[0]]
        r2 = float(1 - ((y - X @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    return [float(x) for x in b], r2


def fits(calls, key, B=10000):
    I = np.array([c["in_tokens"] for _, c in calls], float)
    O = np.array([c["out_tokens"] for _, c in calls], float)
    y = np.array([c[key] for _, c in calls], float)
    b_ols, r2_ols = fit(I, O, y, False)
    b, r2 = fit(I, O, y, True)
    items = sorted({i for i, _ in calls})
    rows = {i: np.array([k for k, (j, _) in enumerate(calls) if j == i]) for i in items}
    rng = np.random.default_rng(0)
    ratio = np.empty(B)
    for t in range(B):
        idx = np.concatenate([rows[items[j]] for j in rng.integers(0, len(items), len(items))])
        bb, _ = fit(I[idx], O[idx], y[idx], True)
        ratio[t] = bb[2] / bb[1]
    lo, hi = np.percentile(ratio, [2.5, 97.5])
    return {"intercept": b[0], "per_in": b[1], "per_out": b[2], "r2": r2, "ratio": b[2] / b[1],
            "ratio_ci95": [float(lo), float(hi)], "ols": {"coef": b_ols, "r2": r2_ols}, "n_calls": len(calls)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--outdir", action="append", default=None)
    ap.add_argument("--fresh", action="store_true", help="recompute even if the input files are unchanged")
    A = ap.parse_args()
    outdirs = [os.path.abspath(d) for d in A.outdir] if A.outdir else [EAAI if os.path.isdir(EAAI) else ARX]
    sig = [os.path.getsize(DATA), os.path.getsize(IDLE)]
    if os.path.exists(OUT_JSON) and not A.fresh:
        old = json.load(open(OUT_JSON))
        if old.get("input_sizes") == sig:  # deterministic (seed 0): unchanged inputs, same results
            emit(old, outdirs)
            print("unchanged input files: reused", os.path.relpath(OUT_JSON, ROOT))
            return
    R = [json.loads(l) for l in open(DATA) if l.strip()]
    idle = {d["label"]: d["mean_w"] for d in (json.loads(l) for l in open(IDLE) if l.strip())}
    task = tasks.get_task("gsm8k")
    main500 = [it["id"] for it in task["load"](n=500, split="all")]
    recs = {(r["config"], r["item_id"]): r for r in R}
    ids_run = sorted({r["item_id"] for r in R})
    assert sorted(main500[:len(ids_run)]) == ids_run, "items are not the first n of the main study"
    assert all((c, i) in recs for c in ["SA", "SC_k8"] + COORD for i in ids_run)
    clean = {"other_gpu_pids": sum(bool(r["gpu_other_pids"]) for r in R),
             "windows_games": sum(bool(r["windows_games"]) for r in R),
             "calls_without_energy": sum(c.get("energy_j") is None for r in R for c in r["calls"])}
    assert not any(clean.values()), clean
    ids = sorted(ids_run)
    w0 = idle["start"]
    for r in R:
        for c in r["calls"]:
            c["energy_net"] = c["energy_j"] - w0 * (c["t1"] - c["t0"])
        assert abs(sum(c["energy_net"] for c in r["calls"]) - r["energy_j_net"]) < 0.01 * len(r["calls"]) + 0.01
    E = json.load(open(FIT))["energy_fit"]
    he = {"per_out": E["J_per_output_token"], "per_in": E["J_per_input_token"], "intercept": E["intercept_J_per_call"],
          "ratio": E["ratio_out_in"], "n_calls": E["n_calls"]}

    def model_j(cs):
        return sum(he["intercept"] + he["per_in"] * c["in_tokens"] + he["per_out"] * c["out_tokens"] for c in cs)

    # per item and configuration: correctness, and costs in each unit
    def calls_of(cfg, i):
        if cfg in ("SC_k2", "SC_k4"):
            return recs[("SC_k8", i)]["calls"][:int(cfg[-1])]
        return recs[(cfg, i)]["calls"]

    def correct(cfg, i):
        if cfg in ("SC_k2", "SC_k4"):
            r = recs[("SC_k8", i)]
            return bool(task["score"](topologies._majority([c["extracted"] for c in calls_of(cfg, i)]), r["gold"]))
        return bool(recs[(cfg, i)]["correct"])

    cost = {u: {c: {} for c in ALL} for u in UNITS}
    for c in ALL:
        for i in ids:
            cs = calls_of(c, i)
            cost["measured"][c][i] = sum(x["energy_j"] for x in cs)
            cost["net"][c][i] = sum(x["energy_net"] for x in cs)
            cost["modeled"][c][i] = model_j(cs)
            cost["tokens"][c][i] = sum(x["in_tokens"] + x["out_tokens"] for x in cs)
    for c in ("SA", "SC_k8") + tuple(COORD):  # record-level totals agree with the per-call sums
        assert all(abs(cost["measured"][c][i] - recs[(c, i)]["energy_j"]) < 0.01 for i in ids)
    M = {c: {i: np.array([float(correct(c, i))]) for i in ids} for c in ALL}
    assert all(M["SC_k8"][i][0] == float(recs[("SC_k8", i)]["correct"]) for i in ids)
    # tests once (accuracy does not depend on the unit); verdicts per unit
    cells0 = {}
    for c in ALL:
        cl = rr.Cell()
        cl.M = M[c]
        cl.tot = {i: cost["tokens"][c][i] for i in ids}
        cl.out = {i: float(sum(x["out_tokens"] for x in calls_of(c, i))) for i in ids}
        cl.none = [False] * len(ids)
        cells0[c] = cl
    T = rr.run_tests(rr.plan_task("e", cells0, ids, SAMP, COORD, extra_pairs=[("SC_k8", "SA"), ("SC_k4", "SA")]), A.jobs)
    res = {"source": os.path.relpath(DATA, ROOT), "script": "scripts/analyze_gsm8k_energy.py",
           "definitions": __doc__, "n_items": len(ids), "n_records": len(R),
           "n_calls": sum(len(r["calls"]) for r in R), "idle_w": idle, "clean": clean,
           "wall_h": (max(c["t1"] for r in R for c in r["calls"]) - min(c["t0"] for r in R for c in r["calls"])) / 3600,
           "hit_cap_calls": sum(c["hit_cap"] for r in R for c in r["calls"]),
           "input_sizes": sig, "humaneval_fit": he, "units": UNITS, "verdicts": {}, "configs": {}}
    allcalls = [(r["item_id"], c) for r in R for c in r["calls"]]
    res["fit_gross"] = fits(allcalls, "energy_j")
    res["fit_net"] = fits(allcalls, "energy_net")
    for u in UNITS:
        cells = {}
        for c in ALL:
            cl = rr.Cell()
            cl.M, cl.out, cl.none = cells0[c].M, cells0[c].out, cells0[c].none
            cl.tot = {i: float(cost[u][c][i]) for i in ids}
            cells[c] = cl
        F = rr.finish_task("e", cells, ids, SAMP, COORD, T, [2, 4, 8])
        res["verdicts"][u] = {w: F["verdict_total"][w] for w in COORD}
        res["tests"] = F["tests"]
        for c in ALL:
            res["configs"].setdefault(c, {"acc": 100 * float(np.mean([M[c][i][0] for i in ids])),
                                          "lo": F["configs"][c]["lo"], "hi": F["configs"][c]["hi"],
                                          "in_tokens": float(np.mean([sum(x["in_tokens"] for x in calls_of(c, i)) for i in ids])),
                                          "out_tokens": float(np.mean([sum(x["out_tokens"] for x in calls_of(c, i)) for i in ids])),
                                          "calls": float(np.mean([len(calls_of(c, i)) for i in ids]))})
            res["configs"][c][u] = float(np.mean([cost[u][c][i] for i in ids]))
        res["frontier_" + u] = sorted(ALL, key=lambda c: res["configs"][c][u])
    for c in ALL:
        d = res["configs"][c]
        d["modeled_minus_measured_pct"] = 100 * (d["modeled"] - d["measured"]) / d["measured"]
        d["seconds"] = float(np.mean([sum(x["t1"] - x["t0"] for x in calls_of(c, i)) for i in ids]))
    json.dump(res, open(OUT_JSON, "w"), indent=1, default=float)
    emit(res, outdirs)
    D = res["configs"]
    for c in ALL:
        print(f"{c:13s} acc {D[c]['acc']:.1f}  J {D[c]['measured']:.0f} net {D[c]['net']:.0f} model {D[c]['modeled']:.0f}"
              f" ({D[c]['modeled_minus_measured_pct']:+.1f}%)  tok {D[c]['tokens']:.0f}")
    for k in ("fit_gross", "fit_net"):
        f = res[k]
        print(k, f"out {f['per_out']:.3f} in {f['per_in']:.4f} c {f['intercept']:.2f} ratio {f['ratio']:.1f} {f['ratio_ci95']} r2 {f['r2']:.4f} ols {f['ols']}")
    for u in UNITS:
        print(u, {w: (v["verdict"], v["robust_by"]) for w, v in res["verdicts"][u].items()})


def emit(res, outdirs):
    L = ["% AUTO-GENERATED by scripts/analyze_gsm8k_energy.py from results/extra/gsm8k_energy_analysis.json.",
         "% Do not edit."]
    nm = lambda k, v: L.append("\\newcommand{\\%s}{%s}" % (k, v))
    fint = lambda x: f"{x:,.0f}".replace(",", "{,}")
    D = res["configs"]
    nm("MgN", str(res["n_items"]))
    nm("MgNcalls", fint(res["n_calls"]))
    nm("MgHours", f"{res['wall_h']:.1f}")
    nm("MgIdleStart", f"{res['idle_w']['start']:.1f}")
    nm("MgIdleEnd", f"{res['idle_w']['end']:.1f}")
    for k, p in (("fit_gross", "Mg"), ("fit_net", "MgNet")):
        f = res[k]
        nm(p + "Out", f"{f['per_out']:.2f}")
        nm(p + "In", f"{f['per_in']:.3f}")
        nm(p + "Icpt", f"{f['intercept']:.1f}")
        nm(p + "Rsq", f"{f['r2']:.3f}")
        nm(p + "Ratio", f"{f['ratio']:.0f}")
        nm(p + "RatioCI", f"[{f['ratio_ci95'][0]:.0f}, {f['ratio_ci95'][1]:.0f}]")
        nm(p + "OlsIcpt", f"{f['ols']['coef'][0]:.1f}".replace("-", "$-$"))
    lab = {"SA": "Sa", "SC_k2": "Sctwo", "SC_k4": "Scfour", "SC_k8": "Sceight", "DEBATE_N3_R1": "Deb",
           "GE_R2": "Ge", "HIERFIX_M3": "Hier"}
    for c, s in lab.items():
        nm(f"MgJ{s}", fint(D[c]["measured"]))
        nm(f"MgNetJ{s}", fint(D[c]["net"]))
        nm(f"MgModJ{s}", fint(D[c]["modeled"]))
        nm(f"MgAcc{s}", f"{D[c]['acc']:.1f}")
        nm(f"MgErr{s}", f"{D[c]['modeled_minus_measured_pct']:+.0f}".replace("-", "$-$"))
    errs = [D[c]["modeled_minus_measured_pct"] for c in D]
    assert min(errs) > 0, "the paper's text says the HumanEval fit overestimates every configuration"
    nm("MgErrRange", f"{min(errs):.0f} to {max(errs):.0f}")  # modeled above measured, %
    V = {"R": "R", "C": "C", "N": "N"}
    for u, U in (("measured", "Meas"), ("net", "Net"), ("modeled", "Mod"), ("tokens", "Tok")):
        for w, s in (("DEBATE_N3_R1", "Deb"), ("GE_R2", "Ge"), ("HIERFIX_M3", "Hier")):
            v = res["verdicts"][u][w]
            cl = {"SA": "SA", "SC_k2": "SC $k{=}2$", "SC_k4": "SC $k{=}4$", "SC_k8": "SC $k{=}8$"}
            nm(f"MgV{U}{s}", V[v["verdict"]] + (f" {{\\scriptsize {cl[v['robust_by'][0]]}}}" if v["robust_by"] else ""))
    for d in outdirs:
        with open(os.path.join(d, "gsm8k_energy_macros.tex"), "w") as fh:
            fh.write("\n".join(L) + "\n")
        T = ["% AUTO-GENERATED by scripts/analyze_gsm8k_energy.py. Do not edit."]
        cl = {"SA": "SA", "SC_k2": "SC $k{=}2$", "SC_k4": "SC $k{=}4$", "SC_k8": "SC $k{=}8$",
              "DEBATE_N3_R1": "Debate", "GE_R2": "GE", "HIERFIX_M3": "HIER"}
        for c in ALL:
            x = D[c]
            T.append(f"{cl[c]} & {x['acc']:.1f} & {fint(x['in_tokens'])} & {fint(x['out_tokens'])} & "
                     f"{x['seconds']:.1f} & {fint(x['measured'])} & {fint(x['net'])} & {fint(x['modeled'])} & "
                     f"{x['modeled_minus_measured_pct']:+.0f} \\\\".replace("& -", "& $-$"))
        with open(os.path.join(d, "tables/gsm8k_energy_table.tex"), "w") as fh:
            fh.write("\n".join(T) + "\n")


if __name__ == "__main__":
    main()
