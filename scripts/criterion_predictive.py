"""Out-of-sample validation of the correction/corruption criterion.

Implements paper/eaai/CRITERION_PREDICTIVE.md exactly (pre-specified, committed
0e2d523 before any model call). Cells, rules R1 to R3, baselines, endpoints,
pooled estimates, and the reporting rule are defined there.

  python3 scripts/criterion_predictive.py                 # full analysis, writes outputs
  python3 scripts/criterion_predictive.py --cells gsm8k_qwen_debate,gsm8k_qwen_ge --print
                                                          # subset, stdout only (testing)
Outputs: results/analysis/criterion_predictive.json,
         paper/eaai/criterion_predictive_macros.tex,
         paper/eaai/criterion_predictive_table.tex (per-cell screen, post hoc),
         paper/eaai/criterion_splits_table.tex (pilot-split results, supplement),
         paper/eaai/criterion_pooled_table.tex, paper/eaai/guideline_criterion_cell.tex
"""
import argparse
import json
import math
import os
import random
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from harness import tasks, topologies  # noqa: E402

RES = os.path.join(ROOT, "results")
PAPER = os.path.join(ROOT, "paper", "eaai")
SPLIT_SEED = 20261012
N_SPLITS = 200
M_PILOT = 30
M_SENS = [10, 20, 30, 40]
N_BOOT = 10000

# id -> (task, model, workflow, protocol n, source)
CELLS = [
    ("gsm8k_qwen_debate", "GSM8K", "Qwen2.5-3B", "debate", 80, "mech"),
    ("gsm8k_qwen_ge", "GSM8K", "Qwen2.5-3B", "ge", 80, "mech"),
    ("gsm8k_llama70b_debate", "GSM8K", "Llama-3.3-70B", "debate", 80, "l70"),
    ("math_qwen_debate", "MATH", "Qwen2.5-3B", "debate", 80, "math500_traces_mac.jsonl"),
    ("math_qwen_ge", "MATH", "Qwen2.5-3B", "ge", 80, "math500_traces_mac.jsonl"),
    ("mmlu_qwen_debate", "MMLU", "Qwen2.5-3B", "debate", 80, "mmlu_traces_mac.jsonl"),
    ("mmlu_qwen_ge", "MMLU", "Qwen2.5-3B", "ge", 80, "mmlu_traces_mac.jsonl"),
    ("eng_qwen_debate", "MMLU-Pro eng.", "Qwen2.5-3B", "debate", 150,
     "mmlu_pro_eng_traces_mac.jsonl"),
    ("eng_qwen_ge", "MMLU-Pro eng.", "Qwen2.5-3B", "ge", 150,
     "mmlu_pro_eng_traces_mac.jsonl"),
]
NEW = {c[0] for c in CELLS if c[5].endswith(".jsonl")}


# --------------------------------------------------------------------------
# data
def _jsonl(name):
    with open(os.path.join(RES, name)) as f:
        return [json.loads(l) for l in f if l.strip()]


def load_cell(cell):
    """List of (init_correct, final_correct) per item, in file order."""
    return [(i, f) for _, i, f in load_cell_ids(cell)]


def load_cell_ids(cell):
    """List of (item_id, init_correct, final_correct) per item, in file order."""
    cid, task, model, wf, n, src = cell
    if src == "mech":
        return [(r["item_id"], bool(r["init_correct"]), bool(r["final_correct"]))
                for r in _jsonl("mechanistic.jsonl") if r["wf"] == wf]
    if src == "l70":
        t = tasks.get_task("gsm8k")
        out = []
        for r in _jsonl("gsm8k_llama70b.jsonl"):
            if r["config"] != "DEBATE_N3_R1":
                continue
            texts = [c["text"] for c in r["calls"]]
            init = topologies._majority([t["extract"](x) for x in texts[:3]])
            fin = topologies._majority([t["extract"](x) for x in texts[3:6]])
            assert str(fin) == str(r["pred"]), r["item_id"]
            out.append((r["item_id"], bool(t["score"](init, r["gold"])),
                        bool(t["score"](fin, r["gold"]))))
        return out
    # New traces: main file plus the reverse-order worker's shard (_b). One
    # record per item (main file first on a duplicate), ordered by the task
    # loader's item order, which is the file order of a single-stream run.
    task_key = {"MATH": "math500", "MMLU": "mmlu", "MMLU-Pro eng.": "mmlu_pro_eng"}[task]
    order = [it["id"] for it in tasks.get_task(task_key)["load"](n=n, split="all")]
    recs = {}
    for name in (src, src.replace(".jsonl", "_b.jsonl")):
        if os.path.exists(os.path.join(RES, name)):
            for r in _jsonl(name):
                if r["wf"] == wf and r["seed"] == 1:
                    recs.setdefault(r["item_id"], r)
    return [(i, bool(recs[i]["init_correct"]), bool(recs[i]["final_correct"]))
            for i in order if i in recs]


def counts(pairs):
    nr = sum(1 for i, _ in pairs if i)
    nw = len(pairs) - nr
    fix = sum(1 for i, f in pairs if not i and f)
    cor = sum(1 for i, f in pairs if i and not f)
    return {"n": len(pairs), "n_right": nr, "n_wrong": nw, "fix": fix, "cor": cor}


def rates(c):
    n = c["n"]
    a = c["n_right"] / n
    pf = c["fix"] / c["n_wrong"] if c["n_wrong"] else 0.0
    pc = c["cor"] / c["n_right"] if c["n_right"] else 0.0
    return a, pf, pc


def delta_hat(a, pf, pc):
    return (1 - a) * pf - a * pc


# --------------------------------------------------------------------------
# intervals (same formulas as the Fig. 2 script)
def wilson(x, n, z=1.959964):
    if n == 0:
        return (float("nan"), float("nan"))
    p = x / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def mover_ratio(x1, n1, x2, n2):
    if n1 == 0 or n2 == 0 or x2 == 0:
        return (float("nan"), float("inf"))
    p1, p2 = x1 / n1, x2 / n2
    l1, u1 = wilson(x1, n1)
    l2, u2 = wilson(x2, n2)
    lo = (p1 * p2 - math.sqrt(max(0.0, (p1 * p2) ** 2 - l1 * u2 * (2 * p1 - l1)
                                  * (2 * p2 - u2)))) / (u2 * (2 * p2 - u2))
    hi = (p1 * p2 + math.sqrt(max(0.0, (p1 * p2) ** 2 - u1 * l2 * (2 * p1 - u1)
                                  * (2 * p2 - l2)))) / (l2 * (2 * p2 - l2))
    return max(0.0, lo), hi


def interval_block(c):
    a, pf, pc = rates(c)
    al, ah = wilson(c["n_right"], c["n"])
    return {
        "a": a, "a_ci": [al, ah],
        "p_fix": pf, "p_fix_ci": list(wilson(c["fix"], c["n_wrong"])),
        "p_cor": pc, "p_cor_ci": list(wilson(c["cor"], c["n_right"])),
        "ratio": (pf / pc) if pc else float("inf"),
        "ratio_ci": list(mover_ratio(c["fix"], c["n_wrong"], c["cor"], c["n_right"])),
        "odds": a / (1 - a) if a < 1 else float("inf"),
        "odds_ci": [al / (1 - al), ah / (1 - ah) if ah < 1 else float("inf")],
        "delta": (c["fix"] - c["cor"]) / c["n"],
    }


def boot_pooled(data, ids, rng):
    """Cell-stratified item bootstrap of pooled Delta and pooled ratio/odds
    (item-cluster resampling within each cell)."""
    arrs = [np.array(data[i], dtype=bool) for i in ids]
    d, rat, marg = [], [], []
    for _ in range(N_BOOT):
        c = {"n": 0, "n_right": 0, "n_wrong": 0, "fix": 0, "cor": 0}
        for A in arrs:
            s = A[rng.integers(0, len(A), len(A))]
            I, F = s[:, 0], s[:, 1]
            c["n"] += len(s)
            c["n_right"] += int(I.sum())
            c["n_wrong"] += int((~I).sum())
            c["fix"] += int((~I & F).sum())
            c["cor"] += int((I & ~F).sum())
        a, pf, pc = rates(c)
        d.append((c["fix"] - c["cor"]) / c["n"])
        rat.append(pf / pc if pc else np.inf)
        marg.append(math.log(pf / pc) - math.log(a / (1 - a)) if pc and pf and a < 1
                    else (-np.inf if pc else np.inf))
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return {"delta_ci": q(d), "ratio_ci_boot": q(rat), "log_ratio_minus_log_odds_ci": q(marg),
            "p_delta_ge_0": float(np.mean(np.array(d) >= 0))}


# --------------------------------------------------------------------------
# rules
def split(cid, s, n, m):
    pilot = set(random.Random(f"{cid}:{s}:{SPLIT_SEED}").sample(range(n), m))
    return pilot, [i for i in range(n) if i not in pilot]


def pooled_other(data, meta, cid):
    """Leave-one-task-out pooled p_fix, p_cor (and observed Delta) of the same
    workflow, from all items of the other cells."""
    _, task, _, wf, _, _ = meta[cid]
    others = [k for k in data if meta[k][3] == wf and meta[k][1] != task]
    c = counts([p for k in others for p in data[k]])
    if c["n"] == 0:
        return None
    a, pf, pc = rates(c)
    return {"p_fix": pf, "p_cor": pc, "delta": (c["fix"] - c["cor"]) / c["n"],
            "cells": others}


def evaluate(data, meta, m):
    rows = []
    for cid, pairs in data.items():
        n = len(pairs)
        po = pooled_other(data, meta, cid)
        for s in range(N_SPLITS):
            P, H = split(cid, s, n, m)
            cp = counts([pairs[i] for i in P])
            ch = counts([pairs[i] for i in H])
            aP, pfP, pcP = rates(cp)
            dH = (ch["fix"] - ch["cor"]) / ch["n"]
            r = {"cell": cid, "s": s, "a_P": aP, "dH": dH,
                 "R1": delta_hat(aP, pfP, pcP),
                 "R3_adopt": aP < 0.5}
            if po:
                r["R2"] = delta_hat(aP, po["p_fix"], po["p_cor"])
                r["POOLD"] = po["delta"]
            rows.append(r)
    return rows


def correct(adopt, dH):
    return (adopt and dH > 0) or ((not adopt) and dH <= 0)


def summarize(rows, data, cells_subset=None):
    rr = [r for r in rows if cells_subset is None or r["cell"] in cells_subset]
    cids = sorted({r["cell"] for r in rr})
    full = {c: (counts(data[c])["fix"] - counts(data[c])["cor"]) / len(data[c]) for c in cids}

    def acc(fn, versus="holdout"):
        per = {}
        for c in cids:
            v = [correct(fn(r), r["dH"] if versus == "holdout" else full[c])
                 for r in rr if r["cell"] == c]
            v = [x for x in v if x is not None]
            per[c] = float(np.mean(v)) if v else float("nan")
        return float(np.nanmean(list(per.values()))), per

    out = {"cells": cids}
    has_r2 = all("R2" in r for r in rr)
    rules = {"R1": lambda r: r["R1"] > 0,
             "R3_a_only": lambda r: r["R3_adopt"],
             "always": lambda r: True, "never": lambda r: False}
    if has_r2:
        rules["R2"] = lambda r: r["R2"] > 0
    for k, fn in rules.items():
        m, per = acc(fn)
        mf, perf = acc(fn, "full")
        out[f"acc_{k}"] = m
        out[f"acc_{k}_per_cell"] = per
        out[f"acc_full_{k}"] = mf
    out["best_constant"] = max(out["acc_always"], out["acc_never"])

    def mae(key):
        per = {c: float(np.mean([abs((r[key] if key != "ZERO" else 0.0) - r["dH"])
                                 for r in rr if r["cell"] == c])) for c in cids}
        return float(np.mean(list(per.values()))), per
    for k in ["R1", "ZERO"] + (["R2", "POOLD"] if has_r2 else []):
        out[f"mae_{k}"], out[f"mae_{k}_per_cell"] = mae(k)

    def corr(key):
        rs = []
        for s in range(N_SPLITS):
            x = [r[key] for r in rr if r["s"] == s]
            y = [r["dH"] for r in rr if r["s"] == s]
            if len(x) > 2 and np.std(x) > 0 and np.std(y) > 0:
                rs.append(float(np.corrcoef(x, y)[0, 1]))
        if not rs:
            return None
        avg = [float(np.mean([r[key] for r in rr if r["cell"] == c])) for c in cids]
        fv = [full[c] for c in cids]
        r_avg = float(np.corrcoef(avg, fv)[0, 1]) if np.std(avg) > 0 and np.std(fv) > 0 else None
        return {"median": float(np.median(rs)), "lo": float(np.percentile(rs, 2.5)),
                "hi": float(np.percentile(rs, 97.5)), "n_splits_defined": len(rs),
                "r_splitavg_vs_full": r_avg}
    out["corr_R1"] = corr("R1")
    if has_r2:
        out["corr_R2"] = corr("R2")
    out["full_delta"] = full
    signs = {np.sign(v) if v != 0 else 0 for v in full.values()}
    out["all_same_sign"] = len({(v > 0) for v in full.values()}) == 1
    out["full_delta_signs"] = sorted(int(x) for x in signs)
    return out


# --------------------------------------------------------------------------
# POST HOC (added 2026-10-15 in answer to a review, after all data were seen;
# not part of the pre-specified protocol, and labeled post hoc in the paper):
# per-cell results of the base-accuracy screen (rule R3 applied to the whole
# cell), with uncertainty, instead of the 200 correlated pilot splits.
POSTHOC_SEED = 20261015


def mcnemar_exact(fix, cor):
    """Two-sided exact McNemar test (binomial on the discordant items)."""
    n, k = fix + cor, min(fix, cor)
    if n == 0:
        return 1.0
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def posthoc_screen(meta, cells):
    """Per cell: Delta with a 95% item-bootstrap percentile interval
    (N_BOOT resamples), exact McNemar p, the screen's decision (adopt iff
    a < 1/2 on the whole cell) and whether it matched the sign of Delta
    (not adopting is right when Delta <= 0, as in the protocol). Groups: cells
    with a > 1/2 and a < 1/2 pooled, Delta = sum(fix - cor) / sum(n), with an
    item-cluster bootstrap: items are resampled once per task and the same
    draw is applied to every cell of that task (cells of one task share their
    items), so correlated cells are not counted as independent."""
    rng = np.random.default_rng(POSTHOC_SEED)
    D = {}
    for cid in cells:
        rows = load_cell_ids(meta[cid])
        D[cid] = {i: int(f) - int(a) for i, a, f in rows}
        D[cid + ":a"] = float(np.mean([a for _, a, _ in rows]))
    out = {"seed": POSTHOC_SEED, "n_boot": N_BOOT, "cells": {}, "groups": {}}
    for cid in cells:
        d = np.array(list(D[cid].values()))
        n = len(d)
        bs = d[rng.integers(0, n, (N_BOOT, n))].mean(1)
        a = D[cid + ":a"]
        fix, cor = int((d == 1).sum()), int((d == -1).sum())
        adopt = a < 0.5
        delta = float(d.mean())
        out["cells"][cid] = {
            "n": n, "a": a, "fix": fix, "cor": cor, "delta": delta,
            "delta_ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "mcnemar_p": mcnemar_exact(fix, cor), "screen_adopt": adopt,
            "screen_right": bool(correct(adopt, delta))}
    groups = [("high_a", [c for c in cells if D[c + ":a"] > 0.5]),
              ("low_a", [c for c in cells if D[c + ":a"] < 0.5]),
              ("high_a_blind", [c for c in cells if D[c + ":a"] > 0.5 and c in NEW])]
    for name, ids in groups:
        if not ids:
            continue
        bytask = {}
        for c in ids:
            bytask.setdefault(meta[c][1], []).append(c)
        tot = sum(len(D[c]) for c in ids)
        obs = sum(sum(D[c].values()) for c in ids) / tot
        bs = np.zeros(N_BOOT)
        for t, ks in bytask.items():
            items = sorted(D[ks[0]])
            assert all(set(D[k]) == set(items) for k in ks), t
            M = np.array([[D[k][i] for i in items] for k in ks]).sum(0)
            bs += M[rng.integers(0, len(items), (N_BOOT, len(items)))].sum(1)
        bs /= tot
        out["groups"][name] = {
            "cells": ids, "n": tot, "delta": obs,
            "delta_ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "n_neg": sum(1 for c in ids if out["cells"][c]["delta"] < 0),
            "n_pos": sum(1 for c in ids if out["cells"][c]["delta"] > 0),
            "n_ci_excl_0": sum(1 for c in ids if out["cells"][c]["delta_ci"][1] < 0
                               or out["cells"][c]["delta_ci"][0] > 0)}
    out["n_right"] = sum(v["screen_right"] for v in out["cells"].values())
    out["misses"] = [c for c, v in out["cells"].items() if not v["screen_right"]]
    out["n_ci_excl_0"] = sum(1 for v in out["cells"].values()
                             if v["delta_ci"][1] < 0 or v["delta_ci"][0] > 0)
    return out


# --------------------------------------------------------------------------
def sanity(meta):
    """Pre-specified sanity check (protocol section 2, not an endpoint): per new
    cell, the share of items whose trace final answer equals the matrix seed-1
    answer of the same configuration, and the trace final accuracy next to the
    matrix accuracy per seed on the shared items."""
    mfile = {"MATH": "math500_trunc.jsonl", "MMLU": "mmlu_matrix.jsonl",
             "MMLU-Pro eng.": "mmlu_pro_eng_matrix.jsonl"}
    cfg = {"debate": "DEBATE_N3_R1", "ge": "GE_R2"}
    out = {}
    for cid, c in meta.items():
        if cid not in NEW:
            continue
        _, task, _, wf, n, src = c
        recs = {}
        for name in (src, src.replace(".jsonl", "_b.jsonl")):
            if os.path.exists(os.path.join(RES, name)):
                for r in _jsonl(name):
                    if r["wf"] == wf:
                        recs.setdefault(r["item_id"], r)
        M = {}
        for r in _jsonl(mfile[task]):
            if r["config"] == cfg[wf]:
                M[(r["item_id"], r["seed"])] = r
        ids = [i for i in recs if (i, 1) in M]
        agree = sum(str(recs[i]["final_answer"]) == str(M[(i, 1)]["pred"]) for i in ids)
        acc = lambda seed: (sum(M[(i, seed)]["correct"] for i in ids if (i, seed) in M)
                            / max(1, sum((i, seed) in M for i in ids)))
        out[cid] = {"n_shared": len(ids), "agree_final_seed1": agree / len(ids),
                    "trace_init_acc": sum(recs[i]["init_correct"] for i in ids) / len(ids),
                    "trace_final_acc": sum(recs[i]["final_correct"] for i in ids) / len(ids),
                    "matrix_acc_seed1": acc(1), "matrix_acc_seed2": acc(2)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default=None)
    ap.add_argument("--print", action="store_true")
    ap.add_argument("--allow-partial", action="store_true")
    args = ap.parse_args()
    use = CELLS if not args.cells else [c for c in CELLS if c[0] in args.cells.split(",")]
    meta = {c[0]: c for c in use}
    data = {}
    for c in use:
        pairs = load_cell(c)
        if len(pairs) != c[4]:
            msg = f"{c[0]}: {len(pairs)} items, protocol says {c[4]}"
            if not args.allow_partial:
                sys.exit("incomplete cell, " + msg)
            print("WARNING partial:", msg)
        data[c[0]] = pairs

    res = {"protocol": "paper/eaai/CRITERION_PREDICTIVE.md", "m": M_PILOT,
           "n_splits": N_SPLITS, "split_seed": SPLIT_SEED, "cells": {}}
    for cid, pairs in data.items():
        c = counts(pairs)
        res["cells"][cid] = {"task": meta[cid][1], "model": meta[cid][2],
                             "wf": meta[cid][3], **c, **interval_block(c)}
    rng = np.random.default_rng(SPLIT_SEED)
    pooled = {}
    for name, ids in [("debate", [k for k in data if meta[k][3] == "debate"]),
                      ("ge", [k for k in data if meta[k][3] == "ge"]),
                      ("all", list(data)),
                      ("new_cells", [k for k in data if k in NEW])]:
        if not ids:
            continue
        c = counts([p for k in ids for p in data[k]])
        pooled[name] = {"cells": ids, **c, **interval_block(c), **boot_pooled(data, ids, rng)}
    res["pooled"] = pooled

    rows = evaluate(data, meta, M_PILOT)
    res["main"] = summarize(rows, data)
    if any(k in NEW for k in data):
        res["main_new_cells"] = summarize(rows, data, NEW & set(data))
    res["sensitivity_m"] = {}
    for m in M_SENS:
        s = summarize(evaluate(data, meta, m), data)
        res["sensitivity_m"][str(m)] = {k: s[k] for k in s if k.startswith(("acc_", "mae_"))
                                        and not k.endswith("per_cell")}

    res["sanity_matrix_agreement"] = sanity(meta)
    res["posthoc_screen"] = posthoc_screen(meta, list(data))

    if args.print:
        print(json.dumps({k: res[k] for k in ("cells", "pooled")}, indent=1, default=str)[:6000])
        mm = res["main"]
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in mm.items()
               if not k.endswith("per_cell") and k not in ("full_delta",)})
        return
    os.makedirs(os.path.join(RES, "analysis"), exist_ok=True)
    with open(os.path.join(RES, "analysis", "criterion_predictive.json"), "w") as f:
        json.dump(res, f, indent=1, default=str)
    from criterion_predictive_emit import emit  # noqa: E402
    emit(res, PAPER)


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
