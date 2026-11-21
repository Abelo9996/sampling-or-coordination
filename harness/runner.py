"""Experiment runner: executes a (model, task, topology, params, seeds) config
over items, scores each, and appends one JSONL record per (item, seed) with the
parsed answer, correctness, and exact token/latency cost.

Checkpoint/resume: records already present in the output file are skipped, so a
killed job resumes for free.
"""
import json
import os

from . import client, tasks, topologies


def config_label(topo, params):
    if topo == "SA":
        return "SA"
    if topo == "SC":
        return f"SC_k{params.get('k')}"
    if topo == "DEBATE":
        return f"DEBATE_N{params.get('N')}_R{params.get('R')}"
    if topo == "DEBATED":
        return f"DEBATED_N{params.get('N')}_R{params.get('R')}"
    if topo == "DEBATEH":
        return f"DEBATEH_N{params.get('N', 3)}_R{params.get('R')}"
    if topo == "GE":
        return f"GE_R{params.get('R')}"
    if topo == "GES":
        return f"GES_R{params.get('R')}"
    if topo == "HIERC":
        return f"HIERC_M{params.get('M')}"
    if topo == "HIER":
        return f"HIER_M{params.get('M')}"
    return topo


def load_done(path):
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    done.add((d["item_id"], d["config"], d["seed"]))
                except (json.JSONDecodeError, KeyError):
                    pass
    return done


def run_config(model, task_name, topo, params, seeds, n_items, split,
               out_path, log_raw=False, progress=True):
    task = tasks.get_task(task_name)
    fn = topologies.get_topology(topo)
    items = task["load"](n=n_items, split=split)
    label = config_label(topo, params)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    done = load_done(out_path)

    n_run = 0
    with open(out_path, "a") as f:
        for it in items:
            for seed in seeds:
                if (it["id"], label, seed) in done:
                    continue
                res = fn(model, task, it, params, seed)
                correct = task["score"](res["answer"], it["gold"])
                cost = client.cost_of(res["calls"])
                rec = {
                    "item_id": it["id"], "task": task_name, "topology": topo,
                    "config": label, "params": params, "seed": seed,
                    "model": model, "split": split,
                    "pred": res["answer"], "gold": it["gold"],
                    "correct": bool(correct), **cost,
                }
                if log_raw:
                    rec["raw"] = res["raw"]
                f.write(json.dumps(rec, default=str) + "\n")
                f.flush()
                n_run += 1
                if progress:
                    print(f"  [{label} seed={seed}] {it['id']} "
                          f"pred={res['answer']} gold={it['gold']} "
                          f"{'OK' if correct else 'x'} "
                          f"tok={cost['total_tokens']} calls={cost['n_calls']}",
                          flush=True)
    return n_run, label
