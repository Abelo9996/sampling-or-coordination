"""Budget-aware router (the paper's artifact, RQ4/H4).

Realizable policy using only inference-time-observable signals:
  1. Pay a cheap 2-sample probe (2 calls).
  2. If the two samples AGREE, output that answer (stop).
  3. If they DISAGREE, escalate to a higher-budget config and output its answer.

We fit the escalation target on a train split and evaluate the resulting
(accuracy, cost) on a held-out test split, comparing against every fixed
topology and against an oracle lower bound on cost.

Inputs:
  - results/<task>_probe.jsonl   (SC k=2, raw per-sample answers, cost)
  - results/<task>_matrix.jsonl  (per-item correctness + cost for all configs)
"""
import hashlib
import json
import os
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def _split(item_id, test_frac=0.4):
    h = int(hashlib.md5(item_id.encode()).hexdigest(), 16) % 1000
    return "test" if h < test_frac * 1000 else "train"


def matrix_per_item(matrix_rows):
    """config -> item_id -> (correct in {0,1}, tokens) averaged over seeds."""
    by = defaultdict(lambda: defaultdict(list))
    for r in matrix_rows:
        by[r["config"]][r["item_id"]].append((int(r["correct"]), r["total_tokens"]))
    out = {}
    for cfg, items in by.items():
        out[cfg] = {iid: (float(np.mean([v[0] for v in vs])),
                          float(np.mean([v[1] for v in vs])))
                    for iid, vs in items.items()}
    return out


def probe_features(probe_rows):
    """item_id -> (agree: bool, probe_answer, probe_correct, probe_tokens, gold)."""
    feats = {}
    for r in probe_rows:
        raw = r.get("raw")
        ans = raw if isinstance(raw, list) else [raw]
        agree = len(ans) >= 2 and str(ans[0]) == str(ans[1]) and ans[0] is not None
        feats[r["item_id"]] = {
            "agree": bool(agree),
            "probe_correct": int(r["correct"]),   # majority of the 2 == its answer
            "probe_tokens": r["total_tokens"],
            "gold": r["gold"],
        }
    return feats


def evaluate_router(task="gsm8k", escalate_to=None):
    mat = matrix_per_item(_load(os.path.join(ROOT, "results", f"{task}_matrix.jsonl")))
    feats = probe_features(_load(os.path.join(ROOT, "results", f"{task}_probe.jsonl")))
    items = [i for i in feats if all(i in mat[c] for c in mat)]

    # candidate escalation targets = the higher-budget configs
    candidates = [c for c in mat if c not in ("SA",)]
    train = [i for i in items if _split(i) == "train"]
    test = [i for i in items if _split(i) == "test"]

    def router_outcome(item_ids, target):
        accs, costs = [], []
        for i in item_ids:
            f = feats[i]
            if f["agree"]:
                accs.append(f["probe_correct"])
                costs.append(f["probe_tokens"])
            else:
                corr, tok = mat[target][i]
                accs.append(corr)
                costs.append(f["probe_tokens"] + tok)
        return float(np.mean(accs)), float(np.mean(costs))

    # pick escalation target maximizing train accuracy per token (simple)
    if escalate_to is None:
        best, best_score = None, -1
        for t in candidates:
            a, c = router_outcome(train, t)
            score = a / (c + 1e-9)
            if score > best_score:
                best, best_score = t, score
        escalate_to = best

    r_acc, r_cost = router_outcome(test, escalate_to)

    # fixed topologies on the test split
    fixed = {}
    for c in mat:
        a = np.mean([mat[c][i][0] for i in test])
        t = np.mean([mat[c][i][1] for i in test])
        fixed[c] = (float(a), float(t))

    # oracle: cheapest config that solves each item (lower bound on cost at acc=max-reachable)
    orc_acc, orc_cost = [], []
    for i in test:
        solved = [(mat[c][i][1], mat[c][i][0]) for c in mat]
        correct_opts = [tok for tok, cr in solved if cr >= 0.5]
        if correct_opts:
            orc_acc.append(1.0)
            orc_cost.append(min(correct_opts))
        else:
            orc_acc.append(0.0)
            orc_cost.append(min(tok for tok, _ in solved))
    oracle = (float(np.mean(orc_acc)), float(np.mean(orc_cost)))

    return {
        "escalate_to": escalate_to,
        "router": {"accuracy": r_acc, "tokens": r_cost},
        "fixed": fixed,
        "oracle": {"accuracy": oracle[0], "tokens": oracle[1]},
        "n_test": len(test), "n_train": len(train),
        "agree_frac": float(np.mean([feats[i]["agree"] for i in test])),
    }


def print_router(task="gsm8k"):
    res = evaluate_router(task)
    print(f"\n=== Router ({task}), test n={res['n_test']}, "
          f"escalate->{res['escalate_to']}, "
          f"agree_frac={res['agree_frac']:.2f} ===")
    print(f"  ROUTER : acc={res['router']['accuracy']*100:5.1f}%  "
          f"tok/item={res['router']['tokens']:.0f}")
    print(f"  ORACLE : acc={res['oracle']['accuracy']*100:5.1f}%  "
          f"tok/item={res['oracle']['tokens']:.0f}  (cost lower bound)")
    print("  fixed topologies (test):")
    for c, (a, t) in sorted(res["fixed"].items(), key=lambda kv: kv[1][1]):
        print(f"    {c:14} acc={a*100:5.1f}%  tok/item={t:.0f}")


if __name__ == "__main__":
    import sys
    print_router(sys.argv[1] if len(sys.argv) > 1 else "gsm8k")
