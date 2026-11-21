"""Secondary (example-execution) vote for the HumanEval code task, computed post
hoc from the logged samples, with no model calls (paper/eaai/CODE_ENERGY_EXPERIMENT.md,
section 3, second paragraph).

For every self-consistency and debate record, each logged candidate program
(`raw`, already canonical) is executed in the sandbox on the prompt's own
`>>> f(...)` examples (never the hidden tests); the tuple of output reprs, or the
exception type, is its signature. The plurality signature wins, ties go to the
signature that occurs first, and the winner is the earliest sample with that
signature. Problems without usable examples, or records whose candidates give no
signature, fall back to the primary rule (the logged `pred`).

Also scores every logged candidate locally (descriptive per-sample pass rate,
share of records with no repeated program); hidden tests are never used to pick
an answer.

Scoring: if the secondary winner is the logged `pred`, its logged (PC, docker,
Python 3.11) result is used; otherwise the winner is scored here against the
hidden tests in the local sandbox (macOS: sandbox-exec, no network, no writes
under /Users). As a check of the local scorer, every distinct logged final
program is also re-scored locally and the agreement with the PC result is
reported.

  .venv/bin/python scripts/humaneval_secondary_vote.py [--workers 6]
writes results/humaneval_secondary_vote.jsonl and results/analysis/humaneval_secondary_vote.json
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from harness import tasks, sandbox  # noqa: E402

LOG = os.path.join(ROOT, "results", "humaneval_matrix_pc.jsonl")
VOTED = ("SC_k2", "SC_k4", "SC_k8", "DEBATE_N3_R1")


def _items():
    return {it["id"]: it for it in tasks.humaneval_load(split="all")}


def _sig_job(args):
    prog, iid = args
    return (prog, iid), tasks.example_signature(prog, ITEMS[iid])


def _score_job(args):
    prog, iid = args
    g = ITEMS[iid]["gold"]
    res = sandbox.run_program(f"{prog}\n\n{g['test']}\n\ncheck({g['entry_point']})\n")
    return (prog, iid), bool(res["ok"])


def _init():
    global ITEMS
    ITEMS = _items()


def plurality(cands, sigs):
    """Winner program by plurality signature, ties to the earliest signature."""
    order, count = [], {}
    for p in cands:
        s = sigs.get(p)
        if p is None or s is None:
            continue
        if s not in count:
            order.append((s, p))
            count[s] = 0
        count[s] += 1
    if not count:
        return None
    best = max(count.values())
    for s, p in order:
        if count[s] == best:
            return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(LOG) if l.strip()]
    items = _items()
    with_ex = {i for i, it in items.items() if tasks.humaneval_examples(it)}
    # 1. example signatures of every candidate on items with examples
    jobs = sorted({(p, r["item_id"]) for r in rows if r["config"] in VOTED
                   and r["item_id"] in with_ex for p in (r["raw"] or []) if p is not None})
    print(f"{len(with_ex)} items with examples; {len(jobs)} distinct candidates to run",
          flush=True)
    with ProcessPoolExecutor(args.workers, initializer=_init) as ex:
        sigs = dict(ex.map(_sig_job, jobs, chunksize=8))
    # 2. secondary winners
    out, to_score = [], set()
    for r in rows:
        if r["config"] not in VOTED:
            continue
        iid = r["item_id"]
        ss = {p: sigs.get((p, iid)) for p in (r["raw"] or [])}
        win = plurality(r["raw"] or [], ss) if iid in with_ex else None
        fallback = win is None
        if fallback:
            win = r["pred"]
        out.append({"item_id": iid, "config": r["config"], "seed": r["seed"],
                    "fallback": fallback, "same_as_primary": win == r["pred"],
                    "pred2": win, "correct_primary": r["correct"]})
        if win != r["pred"] and win is not None:
            to_score.add((win, iid))
    # 3. score changed winners locally, and re-score every logged final program
    check = sorted({(r["pred"], r["item_id"]) for r in rows if r["pred"] is not None})
    # every logged candidate of the voted configurations, for the per-sample
    # pass rate (descriptive: explains the vote; never used to select answers)
    samples = sorted({(p, r["item_id"]) for r in rows if r["config"] in VOTED
                      for p in (r["raw"] or []) if p is not None})
    allj = sorted(to_score | set(check) | set(samples))
    print(f"scoring {len(to_score)} changed winners, {len(check)} logged programs, "
          f"{len(samples)} distinct candidates", flush=True)
    with ProcessPoolExecutor(args.workers, initializer=_init) as ex:
        local = dict(ex.map(_score_job, allj, chunksize=8))
    pc = {(r["pred"], r["item_id"]): r["correct"] for r in rows if r["pred"] is not None}
    agree = sum(local[k] == pc[k] for k in check)
    for o in out:
        if o["same_as_primary"]:
            o["correct2"] = o["correct_primary"]
        else:
            o["correct2"] = bool(o["pred2"] is not None and local[(o["pred2"], o["item_id"])])
    with open(os.path.join(ROOT, "results", "humaneval_secondary_vote.jsonl"), "w") as f:
        for o in out:
            o2 = dict(o)
            o2.pop("pred2")
            f.write(json.dumps(o2) + "\n")
    summ = {"items_with_examples": len(with_ex),
            "distinct_candidates_run": len(jobs),
            "local_rescore_n": len(check), "local_rescore_agree": agree,
            "local_python": sys.version.split()[0],
            "backend": sandbox.default_backend()}
    for c in VOTED:
        oc = [o for o in out if o["config"] == c]
        summ[c] = {"n": len(oc),
                   "acc_primary": 100.0 * sum(o["correct_primary"] for o in oc) / len(oc),
                   "acc_secondary": 100.0 * sum(o["correct2"] for o in oc) / len(oc),
                   "changed": sum(not o["same_as_primary"] for o in oc),
                   "fallback": sum(o["fallback"] for o in oc)}
        rc = [r for r in rows if r["config"] == c]
        ps = [local[(p, r["item_id"])] if p is not None else False
              for r in rc for p in (r["raw"] or [])]
        summ[c]["per_sample_pass"] = 100.0 * sum(ps) / len(ps)
        summ[c]["any_sample_pass"] = 100.0 * sum(
            any(p is not None and local[(p, r["item_id"])] for p in (r["raw"] or []))
            for r in rc) / len(rc)
        # records whose exact-program vote has no repeated program (plurality of 1)
        summ[c]["all_distinct"] = 100.0 * sum(
            max([(r["raw"] or []).count(p) for p in (r["raw"] or []) if p is not None]
                or [0]) <= 1 for r in rc) / len(rc)
    os.makedirs(os.path.join(ROOT, "results", "analysis"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "analysis", "humaneval_secondary_vote.json"), "w") as f:
        json.dump(summ, f, indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
