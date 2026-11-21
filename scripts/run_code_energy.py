"""HumanEval matrix with per-call GPU energy (design: paper/eaai/CODE_ENERGY_EXPERIMENT.md).

Runs the 7 headline configurations on HumanEval, one request at a time,
scoring every program against the hidden tests in the sandbox
(harness/sandbox.py). With --energy nvidia, a background `nvidia-smi -lms 200`
sampler attributes joules to each model call by integrating board power over
the call's wall-clock interval; a query's energy is the sum over its calls.
Resumable: (item, config, seed) already in --out are skipped, a partial last
line is dropped first.

  PC:  AGENTS_SANDBOX=docker python3 scripts/run_code_energy.py --energy nvidia
  Mac smoke (3 problems, SA, no energy):
       python3 scripts/run_code_energy.py --configs SA --limit 3 \
           --out results/humaneval_smoke_mac.jsonl --energy none
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from harness import client, tasks, topologies, traces  # noqa: E402
from harness.runner import config_label  # noqa: E402
from configs import matrix as M  # noqa: E402

MODEL = "qwen2.5:3b-instruct"
CAP = 1024
IDLE_S = 60


class TimedGenerate:
    """Wraps client.generate: stamps each call with host start/end times and,
    if a sampler is given, its GPU energy in joules."""
    def __init__(self, inner, sampler=None):
        self.inner, self.sampler = inner, sampler

    def __call__(self, *a, **kw):
        t0 = time.time()
        r = self.inner(*a, **kw)
        t1 = time.time()
        r["t0"], r["t1"] = round(t0, 4), round(t1, 4)
        if self.sampler is not None:
            j = self.sampler.joules(t0, t1)
            r["energy_j"] = None if j is None else round(j, 3)
        return r


def idle_baseline(sampler, label, path):
    t0 = time.time()
    time.sleep(IDLE_S)
    t1 = time.time()
    w = sampler.mean_watts(t0, t1)
    rec = {"label": label, "t0": t0, "t1": t1, "mean_w": w}
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"[idle {label}] mean {w} W over {IDLE_S}s", flush=True)
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "humaneval_matrix_pc.jsonl"))
    ap.add_argument("--energy", choices=["nvidia", "none"], default="nvidia")
    ap.add_argument("--configs", default="all",
                    help="'all' or comma list of labels, e.g. SA,SC_k8")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--host", default=os.environ.get("AGENTS_HOST_LABEL", "unknown"))
    args = ap.parse_args()

    client.DEFAULT_NUM_PREDICT = CAP
    task = tasks.get_task("humaneval")
    items = task["load"](n=None, split="all")
    if args.limit:
        items = items[:args.limit]
    cfgs = M.HUMANEVAL_CONFIGS
    if args.configs != "all":
        want = set(args.configs.split(","))
        cfgs = [c for c in cfgs if config_label(c[0], c[1]) in want]

    sampler = None
    idle_path = os.path.splitext(args.out)[0] + "_idle.jsonl"
    if args.energy == "nvidia":
        from harness.energy import make_sampler
        sampler = make_sampler().start()
        print(f"[energy] sampler={type(sampler).__name__}", flush=True)
        time.sleep(2)
    inner = client.generate
    client.generate = TimedGenerate(inner, sampler)
    # warm-up call (loads the model; not logged as a record)
    client.generate(MODEL, "Say OK.", temperature=0.0, seed=1, num_predict=4)
    idle_w = idle_baseline(sampler, "start", idle_path) if sampler else None

    traces.repair_tail(args.out)
    done = set()
    if os.path.exists(args.out):
        for line in open(args.out):
            try:
                d = json.loads(line)
                done.add((d["item_id"], d["config"], d["seed"]))
            except (json.JSONDecodeError, KeyError):
                pass
    print(f"== CODE task=humaneval n={len(items)} configs={len(cfgs)} "
          f"energy={args.energy} out={args.out} already={len(done)} ==", flush=True)
    t_start = time.time()
    for topo, params, seeds in cfgs:
        label = config_label(topo, params)
        fn = topologies.get_topology(topo)
        for it in items:
            for seed in seeds:
                if (it["id"], label, seed) in done:
                    continue
                res = fn(MODEL, task, it, params, seed)
                t_s = time.time()
                ok = bool(task["score"](res["answer"], it["gold"]))
                exec_s = time.time() - t_s
                cost = client.cost_of(res["calls"])
                rec = {"item_id": it["id"], "task": "humaneval", "topology": topo,
                       "config": label, "params": params, "seed": seed,
                       "model": MODEL, "num_predict": CAP, "host": args.host,
                       "pred": res["answer"], "correct": ok, **cost,
                       "exec_s": round(exec_s, 3),
                       "raw": res["raw"] if isinstance(res["raw"], list) else None,
                       "calls": [{k: c.get(k) for k in
                                  ("in_tokens", "out_tokens", "latency_s", "t0", "t1",
                                   "energy_j")} for c in res["calls"]]}
                if sampler is not None:
                    ej = [c.get("energy_j") for c in res["calls"]]
                    rec["energy_j"] = None if None in ej else round(sum(ej), 3)
                    if rec["energy_j"] is not None and idle_w is not None:
                        busy = sum(c["t1"] - c["t0"] for c in res["calls"])
                        rec["energy_j_net"] = round(rec["energy_j"] - idle_w * busy, 3)
                traces.append_record(args.out, rec)
                print(f"  [{label} s{seed}] {it['id']} pass={ok} tok={cost['total_tokens']} "
                      f"lat={cost['latency_s']:.1f}s E={rec.get('energy_j')}", flush=True)
        print(f"[done] {label} ({(time.time()-t_start)/60:.1f} min)", flush=True)
    if sampler:
        idle_baseline(sampler, "end", idle_path)
        sampler.stop()
    print("== CODE DONE ==", flush=True)


if __name__ == "__main__":
    main()
