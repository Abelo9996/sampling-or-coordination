"""Aggregate JSONL run logs into per-config accuracy/cost with bootstrap CIs,
compute the accuracy-vs-cost Pareto frontier, and the single-agent (SC) vs.
multi-agent crossover. Pure numpy; no model calls.

Per-config aggregation handles multiple seeds correctly: for each item we first
average correctness and token cost over seeds (a per-item value), then take the
mean over items and bootstrap the CI over items (the items are the sampling
unit).
"""
import json
import os
from collections import defaultdict

import numpy as np

# canonical cost in $ per 1M tokens for the $-estimate column (labeled estimate,
# representative small-model API price; report alongside raw tokens).
USD_PER_MTOK = 0.20


def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def per_item_table(rows):
    """config -> {item_id -> (mean_correct_over_seeds, mean_tokens_over_seeds)}."""
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r["config"]][r["item_id"]].append((int(r["correct"]), r["total_tokens"]))
    table = {}
    for cfg, items in by.items():
        d = {}
        for iid, vals in items.items():
            cs = np.mean([v[0] for v in vals])
            ts = np.mean([v[1] for v in vals])
            d[iid] = (cs, ts)
        table[cfg] = d
    return table


def _bootstrap_ci(x, B=2000, alpha=0.05, seed=0):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(B, len(x)))
    means = x[idx].mean(axis=1)
    return (float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))))


ITEM_CI_B = 10000  # the paper's item bootstrap: 10,000 resamples, numpy seed 0


def item_bootstrap_ci(per_item, B=ITEM_CI_B, seed=0):
    """95% item-bootstrap interval of the mean of per-item values (already averaged over seeds),
    with the items in sorted id order, so the interval does not depend on log order.
    per_item: {item_id: value}."""
    return _bootstrap_ci([per_item[i] for i in sorted(per_item)], B=B, seed=seed)


def aggregate(rows):
    """config -> dict(accuracy, acc_lo, acc_hi, tokens, tok_lo, tok_hi, n_items,
    usd_per_item, topology)."""
    table = per_item_table(rows)
    topo_of = {r["config"]: r["topology"] for r in rows}
    out = {}
    for cfg, items in table.items():
        corr = np.array([v[0] for v in items.values()])
        toks = np.array([v[1] for v in items.values()])
        alo, ahi = _bootstrap_ci(corr)
        tlo, thi = _bootstrap_ci(toks)
        out[cfg] = {
            "topology": topo_of[cfg],
            "accuracy": float(corr.mean()),
            "acc_lo": alo, "acc_hi": ahi,
            "tokens": float(toks.mean()),
            "tok_lo": tlo, "tok_hi": thi,
            "usd_per_item": float(toks.mean()) / 1e6 * USD_PER_MTOK,
            "n_items": len(items),
        }
    return out


def pareto_frontier(agg):
    """Configs that are not dominated (higher accuracy AND lower tokens)."""
    pts = [(c, a["tokens"], a["accuracy"]) for c, a in agg.items()]
    front = []
    for c, x, y in pts:
        dominated = any((xx <= x and yy >= y and (xx < x or yy > y))
                        for cc, xx, yy in pts if cc != c)
        if not dominated:
            front.append(c)
    return sorted(front, key=lambda c: agg[c]["tokens"])


def sc_frontier(agg):
    """The single-agent test-time-compute curve (SA + SC configs), sorted by cost."""
    sc = [(c, a["tokens"], a["accuracy"]) for c, a in agg.items()
          if a["topology"] in ("SA", "SC")]
    return sorted(sc, key=lambda t: t[1])


def crossover_vs_sc(agg):
    """For each MAS config, interpolate the SC-frontier accuracy at the MAS's
    token cost and report the gap (SC_acc - MAS_acc). Positive gap => a single
    agent at equal budget is at least as good (H1 support)."""
    sc = sc_frontier(agg)
    if len(sc) < 2:
        return {}
    xs = np.array([t[1] for t in sc])
    ys = np.array([t[2] for t in sc])
    res = {}
    for c, a in agg.items():
        if a["topology"] in ("SA", "SC"):
            continue
        x = a["tokens"]
        sc_acc = float(np.interp(x, xs, ys))  # SC accuracy at equal token budget
        res[c] = {"mas_acc": a["accuracy"], "sc_acc_at_equal_budget": sc_acc,
                  "gap_sc_minus_mas": sc_acc - a["accuracy"], "tokens": x}
    return res


def print_report(path):
    rows = load(path)
    agg = aggregate(rows)
    order = sorted(agg, key=lambda c: agg[c]["tokens"])
    print(f"\n=== {os.path.basename(path)}  ({len(rows)} records) ===")
    print(f"{'config':14}{'topo':8}{'acc':>8}{'95% CI':>16}"
          f"{'tok/item':>10}{'$/1k items':>12}")
    for c in order:
        a = agg[c]
        print(f"{c:14}{a['topology']:8}{a['accuracy']*100:7.1f}%"
              f"  [{a['acc_lo']*100:4.1f},{a['acc_hi']*100:4.1f}]"
              f"{a['tokens']:10.0f}{a['usd_per_item']*1000:11.3f}")
    front = pareto_frontier(agg)
    print("Pareto frontier (cheap->expensive):", " > ".join(front))
    cx = crossover_vs_sc(agg)
    if cx:
        print("\nMAS vs single-agent at equal token budget "
              "(gap>0 => single agent wins):")
        for c, d in sorted(cx.items()):
            print(f"  {c:14} MAS={d['mas_acc']*100:5.1f}%  "
                  f"SC@equal={d['sc_acc_at_equal_budget']*100:5.1f}%  "
                  f"gap={d['gap_sc_minus_mas']*100:+5.1f} pts")


if __name__ == "__main__":
    import sys
    print_report(sys.argv[1] if len(sys.argv) > 1
                 else os.path.join(os.path.dirname(__file__), "..",
                                   "results", "gsm8k_matrix.jsonl"))
