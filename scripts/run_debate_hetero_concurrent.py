"""Heterogeneous (mixed-model) debate on GSM8K: batched by model AND concurrent
within each phase (Ollama continuous batching via OLLAMA_NUM_PARALLEL).

Scientifically identical to harness.topologies.debate_hetero (same three models,
same debate structure, same per-agent seeds, same majority vote); this only
schedules the independent calls of each phase concurrently. Deterministic: each
call carries its own fixed seed, so ordering does not affect results.

Requires the server started with OLLAMA_NUM_PARALLEL>=WORKERS. Output overwritten:
results/gsm8k_debate_hetero.jsonl. Run: python scripts/run_debate_hetero_concurrent.py
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from harness import client, tasks  # noqa: E402
from harness.topologies import _majority, HETERO_MODELS  # noqa: E402

TASK, SEEDS, N_ITEMS, MODELS, R, TEMP = "gsm8k", [1, 2], 80, HETERO_MODELS, 1, 0.7
WORKERS = 4
OUT = os.path.join(ROOT, "results", "gsm8k_debate_hetero.jsonl")


def main():
    task = tasks.get_task(TASK)
    items = task["load"](n=N_ITEMS, split="all")
    base = {it["id"]: task["prompt"](it["question"]) for it in items}
    idmap = {it["id"]: it for it in items}
    keys = [(it["id"], s) for it in items for s in SEEDS]
    calls = {k: [] for k in keys}
    t0 = time.time()

    def run_phase(prompt_of, seed_of, tag):
        # prompt_of/seed_of: (key)->(model, prompt, seed). Concurrent over keys.
        results = {}
        for a, model in enumerate(MODELS):
            def one(k, a=a, model=model):
                pr, sd = prompt_of(k, a)
                return k, client.generate(model, pr, temperature=TEMP, seed=sd)
            with ThreadPoolExecutor(max_workers=WORKERS) as ex:
                for k, r in ex.map(one, keys):
                    results[(k, a)] = r
                    calls[k].append(r)
            print(f"  {tag} done for {model} ({time.time()-t0:.0f}s)", flush=True)
        return results

    # round 0
    r0 = run_phase(lambda k, a: (base[k[0]], k[1] * 1000 + a), None, "round0")
    r0txt = {(k, a): r["text"] for (k, a), r in r0.items()}

    # round 1: each agent sees others' round-0 solutions
    def r1_prompt(k, a):
        others = [r0txt[(k, o)] for o in range(len(MODELS)) if o != a]
        blk = "\n\n".join(f"Solution from agent {i+1}:\n{t}"
                          for i, t in enumerate(others))
        p = (f"{base[k[0]]}\nHere are other agents' solutions to the same "
             f"problem:\n\n{blk}\n\nCritique them, then give your updated "
             f"solution. {task['reminder']}")
        return p, k[1] * 1000 + 100 + a
    r1 = run_phase(r1_prompt, None, "round1")

    with open(OUT, "w") as f:
        for k in keys:
            answers = [task["extract"](r1[(k, a)]["text"]) for a in range(len(MODELS))]
            ans = _majority(answers)
            it = idmap[k[0]]
            correct = task["score"](ans, it["gold"])
            cost = client.cost_of(calls[k])
            rec = {"item_id": k[0], "task": TASK, "topology": "DEBATEH",
                   "config": "DEBATEH_N3_R1", "params": {"N": 3, "R": R},
                   "seed": k[1], "model": "mixed-qwen-llama-gemma", "split": "all",
                   "pred": ans, "gold": it["gold"], "correct": bool(correct), **cost}
            f.write(json.dumps(rec, default=str) + "\n")
    print(f"== done: {len(keys)} records in {time.time()-t0:.0f}s -> {OUT} ==", flush=True)


if __name__ == "__main__":
    main()
