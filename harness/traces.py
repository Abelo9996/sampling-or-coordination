"""Correction/corruption traces for DEBATE and GE on any task.

Runs the matrix topology functions themselves (topologies.debate / .ge) with
their optional `trace` hook, so prompts, seeds, caps, and the returned answer
are exactly those of the matrix configuration. One record per (item, workflow):

DEBATE (N=3, R=1): per-agent answers of round 0 and of the final round, the
  round majorities; initial = round-0 majority, final = final-round majority.
GE (R=2): draft answer, then per round the critic verdict and the revised
  answer; initial = draft, final = last revision.

Every record also carries per-call input/output tokens and latency, and the raw
call texts, so the traces can be audited and re-scored without re-running.
Protocol: paper/eaai/CRITERION_PREDICTIVE.md.
"""
import json
import os

from . import client, topologies

WORKFLOWS = {
    "debate": ("DEBATE", {"N": 3, "R": 1}, "DEBATE_N3_R1"),
    "ge": ("GE", {"R": 2}, "GE_R2"),
}


def _jsonable(x):
    return x if x is None or isinstance(x, (int, float, str, bool)) else str(x)


def trace_record(model, task_name, task, item, wf, seed):
    """Run one (item, workflow) trace and return the JSON-able record."""
    topo, params, label = WORKFLOWS[wf]
    fn = topologies.get_topology(topo)
    tr = {}
    res = fn(model, task, item, params, seed, trace=tr)
    score = lambda a: bool(task["score"](a, item["gold"]))
    rec = {"item_id": item["id"], "task": task_name, "wf": wf, "config": label,
           "params": params, "seed": seed, "model": model,
           "num_predict": client.DEFAULT_NUM_PREDICT,
           "gold": _jsonable(item["gold"])}
    if wf == "debate":
        rounds = [[_jsonable(a) for a in r] for r in tr["rounds"]]
        init_ans, final_ans = tr["majorities"][0], tr["majorities"][-1]
        rec.update(
            agent_answers=rounds,
            agent_correct=[[score(a) for a in r] for r in tr["rounds"]],
            init_answer=_jsonable(init_ans), final_answer=_jsonable(final_ans),
            init_unique=len(set(map(str, tr["rounds"][0]))),
            final_unique=len(set(map(str, tr["rounds"][-1]))))
    else:
        init_ans = tr["draft"]
        final_ans = tr["rounds"][-1]["answer"] if tr["rounds"] else init_ans
        prev = score(init_ans)
        rds = []
        for r in tr["rounds"]:
            now = score(r["answer"])
            rds.append({"critic_found_errors": bool(r["critic_found_errors"]),
                        "answer": _jsonable(r["answer"]),
                        "before_correct": prev, "after_correct": now})
            prev = now
        rec.update(draft_answer=_jsonable(init_ans), rounds=rds,
                   init_answer=_jsonable(init_ans),
                   final_answer=_jsonable(final_ans))
    # the topology's own returned answer must equal the traced final answer
    assert str(res["answer"]) == str(final_ans), (res["answer"], final_ans)
    rec["init_correct"] = score(init_ans)
    rec["final_correct"] = score(final_ans)
    rec.update(client.cost_of(res["calls"]))
    rec["calls"] = [{"in_tokens": c["in_tokens"], "out_tokens": c["out_tokens"],
                     "latency_s": c["latency_s"], "text": c["text"]}
                    for c in res["calls"]]
    return rec


def repair_tail(path):
    """Drop a trailing partial line (e.g. after a power loss mid-write) so the
    file stays valid JSONL and the next append starts on a fresh line."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return 0
    with open(path, "rb") as f:
        data = f.read()
    if data.endswith(b"\n"):
        return 0
    cut = data.rfind(b"\n") + 1
    with open(path, "r+b") as f:
        f.truncate(cut)
    return len(data) - cut


def load_done(path):
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    done.add((d["item_id"], d["wf"], d["seed"]))
                except (json.JSONDecodeError, KeyError):
                    pass
    return done


def append_record(path, rec):
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()
        os.fsync(f.fileno())
