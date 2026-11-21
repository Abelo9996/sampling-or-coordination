"""Heterogeneous (mixed-model) debate on GSM8K, batched by model to avoid
per-call model swapping (the slow part when 3 models do not co-reside).

Scientifically identical to harness.topologies.debate_hetero (same three models,
same debate structure, same per-agent seeds, same majority vote): this is only a
faster scheduler that runs all of one model's calls in a phase before switching
models, so each model loads a handful of times instead of once per call. Produces
the same per-(item,seed) records as the per-item topology would.

Output: results/gsm8k_debate_hetero.jsonl (overwritten fresh). Run:
  python scripts/run_debate_hetero_batched.py
"""
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from harness import client, tasks  # noqa: E402
from harness.topologies import _majority, HETERO_MODELS  # noqa: E402

TASK = "gsm8k"
SEEDS = [1, 2]
N_ITEMS = 80
MODELS = HETERO_MODELS          # [qwen2.5:3b-instruct, llama3.2:3b, gemma2:2b]
R = 1
TEMP = 0.7
OUT = os.path.join(ROOT, "results", "gsm8k_debate_hetero.jsonl")


def main():
    task = tasks.get_task(TASK)
    items = task["load"](n=N_ITEMS, split="all")
    base = {it["id"]: task["prompt"](it["question"]) for it in items}
    keys = [(it["id"], s) for it in items for s in SEEDS]  # (item, seed)
    idmap = {it["id"]: it for it in items}
    t0 = time.time()

    # round 0: batch by model so each loads once
    r0 = {}   # (item,seed,agent) -> text
    calls = {k: [] for k in keys}  # (item,seed) -> list of CallRecords
    for a, model in enumerate(MODELS):
        for (iid, s) in keys:
            r = client.generate(model, base[iid], temperature=TEMP,
                                 seed=(s * 1000 + a))
            r0[(iid, s, a)] = r["text"]
            calls[(iid, s)].append(r)
        print(f"  round0 done for {model} ({time.time()-t0:.0f}s)", flush=True)

    # round 1: batch by model; each agent sees the others' round-0 solutions
    final = {}  # (item,seed) -> list of answers (one per agent, last round)
    for k in keys:
        final[k] = [None] * len(MODELS)
    for a, model in enumerate(MODELS):
        for (iid, s) in keys:
            others = [r0[(iid, s, o)] for o in range(len(MODELS)) if o != a]
            others_blk = "\n\n".join(
                f"Solution from agent {i+1}:\n{t}" for i, t in enumerate(others))
            p = (f"{base[iid]}\nHere are other agents' solutions to the same "
                 f"problem:\n\n{others_blk}\n\nCritique them, then give your "
                 f"updated solution. {task['reminder']}")
            r = client.generate(model, p, temperature=TEMP,
                                seed=(s * 1000 + (0 + 1) * 100 + a))
            calls[(iid, s)].append(r)
            final[(iid, s)][a] = task["extract"](r["text"])
        print(f"  round1 done for {model} ({time.time()-t0:.0f}s)", flush=True)

    # vote, score, write
    with open(OUT, "w") as f:
        for (iid, s) in keys:
            ans = _majority(final[(iid, s)])
            it = idmap[iid]
            correct = task["score"](ans, it["gold"])
            cost = client.cost_of(calls[(iid, s)])
            rec = {"item_id": iid, "task": TASK, "topology": "DEBATEH",
                   "config": "DEBATEH_N3_R1", "params": {"N": 3, "R": R},
                   "seed": s, "model": "mixed-qwen-llama-gemma", "split": "all",
                   "pred": ans, "gold": it["gold"], "correct": bool(correct),
                   **cost}
            f.write(json.dumps(rec, default=str) + "\n")
    print(f"== done: {len(keys)} records in {time.time()-t0:.0f}s -> {OUT} ==",
          flush=True)


if __name__ == "__main__":
    main()
