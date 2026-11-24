"""Cross-model replication driver: the paper's main experiment (GSM8K first 500 test
items, MATH-500 levels 1 to 3 = the 238 items of power campaign phase B) on one
model served by an OpenAI-compatible server (vLLM), with many work units in flight.

Workflows, prompts, seeds, extraction and record schema are those of
scripts/run_rev3_campaign.py (unchanged harness.topologies and
scripts/rev3_workflows.hierfix): SA (greedy, seed 1 only), SC k=8 (SC k=2, 4 are
the first 2, 4 calls by the harness seed rule), DEBATE N3 R1, GE R2, HIERFIX M3
(the HIER variant the rev3 paper reports in the main text; on MATH it is
byte-identical to HIER).

Concurrency: one work unit = (seed, task, config, item), run by the unchanged
topology function in its own thread, so calls inside an item keep their order
(debate round 1 after round 0, GE critic after the draft) while up to
--concurrency units run at once and the server batches their calls. Units are
queued seed by seed (1, 2, 3), task by task, and in chunks of 50 items with all
configs per chunk, so a run stopped by the budget guard leaves complete seeds
first and balanced configs within the last one.

Outputs in --out-dir: <task>_s<seed>.jsonl (one record per unit, written as it
completes, resumable: units already in the file are skipped), errors.jsonl,
progress.json (counters, truncation, throughput, seeds complete; rewritten every
30 s and at exit).

Deadlines: --soft-deadline (epoch s) stops starting new units; --hard-deadline
exits even with units in flight (they are rerun on resume).

Usage (server already up):
  python3 -u scripts/cloud/run_cloud_campaign.py --model Qwen/Qwen3-8B --thinking on \\
      --base-url http://127.0.0.1:8000/v1 --out-dir results/cloud/qwen3-8b
"""
import argparse
import glob
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
ROOT = os.path.dirname(SCRIPTS)

# Paper caps for non-thinking models (power campaign phases A and B), and the
# larger caps for thinking models (thinking tokens count as output).
CAPS = {False: {"gsm8k": 640, "math500": 1024}, True: {"gsm8k": 4096, "math500": 8192}}
N_ITEMS = {"gsm8k": 500, "math500": 238}
ORDER = ["SA", "SC", "DEBATE", "GE", "HIERFIX"]


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="served model id (HF id on vLLM)")
    ap.add_argument("--thinking", choices=["on", "off", ""], default="")
    ap.add_argument("--base-url", default=os.environ.get("AGENTS_VLLM_URL", "http://127.0.0.1:8000/v1"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tasks", default="gsm8k,math500")
    ap.add_argument("--n", default="", help="override item counts, e.g. gsm8k:3,math500:3")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--sa-seed", type=int, default=1)
    ap.add_argument("--configs", default=",".join(ORDER))
    ap.add_argument("--sc-k", type=int, default=8)
    ap.add_argument("--cap-gsm8k", type=int, default=0, help="0: CAPS default for the thinking mode")
    ap.add_argument("--cap-math", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=192)
    ap.add_argument("--chunk", type=int, default=50)
    ap.add_argument("--soft-deadline", type=float, default=0.0)
    ap.add_argument("--hard-deadline", type=float, default=0.0)
    ap.add_argument("--host", default=os.environ.get("AGENTS_HOST_LABEL", "cloud"))
    ap.add_argument("--max-fail-frac", type=float, default=0.5,
                    help="abort when this fraction of the last 200 finished units failed")
    return ap.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    os.environ["AGENTS_BACKEND"] = "vllm"
    os.environ["AGENTS_VLLM_URL"] = a.base_url
    os.environ["AGENTS_THINKING"] = a.thinking
    sys.path.insert(0, ROOT)
    sys.path.insert(0, SCRIPTS)
    from harness import client, runner, tasks, topologies, vllm_client  # noqa: E402
    from run_power_campaign import load_math_levels, read_jsonl  # noqa: E402
    from run_rev3_campaign import PARAMS, final_call_idx  # noqa: E402
    import rev3_workflows as rw  # noqa: E402
    assert client.generate is vllm_client.generate, "backend switch failed"

    thinking = a.thinking == "on"
    caps = dict(CAPS[thinking])
    if a.cap_gsm8k:
        caps["gsm8k"] = a.cap_gsm8k
    if a.cap_math:
        caps["math500"] = a.cap_math
    n_items = dict(N_ITEMS)
    for kv in filter(None, a.n.split(",")):
        k, v = kv.split(":")
        n_items[k] = int(v)
    task_names = a.tasks.split(",")
    seeds = [int(s) for s in a.seeds.split(",")]
    want = a.configs.split(",")
    assert set(want) <= set(ORDER), want

    T, ITEMS = {}, {}
    for tn in task_names:
        T[tn] = tasks.get_task(tn)
        if tn == "math500":
            ITEMS[tn] = load_math_levels(tasks, 1, 3, n_items[tn])
            assert [it["id"] for it in ITEMS[tn]] == \
                [it["id"] for it in T[tn]["load"](n=n_items[tn], split="all")]
        else:
            ITEMS[tn] = T[tn]["load"](n=n_items[tn], split="all")

    def configs_for(seed):
        out = []
        for topo in ORDER:
            if topo not in want or (topo == "SA" and seed != a.sa_seed):
                continue
            params = {"k": a.sc_k} if topo == "SC" else dict(PARAMS.get(topo, {}))
            label = "HIERFIX_M3" if topo == "HIERFIX" else runner.config_label(topo, params)
            out.append((topo, params, label))
        return out

    os.makedirs(a.out_dir, exist_ok=True)
    paths = {(tn, s): os.path.join(a.out_dir, f"{tn}_s{s}.jsonl") for tn in task_names for s in seeds}
    done = {(tn, s): {(r["config"], r["item_id"]) for r in read_jsonl(p)} for (tn, s), p in paths.items()}
    units = []  # (seed, task, topo, params, label, item)
    for s in seeds:
        for tn in task_names:
            its = ITEMS[tn]
            for c0 in range(0, len(its), a.chunk):
                for topo, params, label in configs_for(s):
                    for it in its[c0:c0 + a.chunk]:
                        units.append((s, tn, topo, params, label, it))
    expected = {(tn, s): sum(1 for u in units if u[0] == s and u[1] == tn) for tn in task_names for s in seeds}
    todo = [u for u in units if (u[4], u[5]["id"]) not in done[(u[1], u[0])]]

    lock = threading.Lock()
    files = {k: open(p, "a") for k, p in paths.items()}
    err_f = open(os.path.join(a.out_dir, "errors.jsonl"), "a")
    st = {"units_done": 0, "units_failed": 0, "units_skipped_deadline": 0, "calls": 0,
          "in_tokens": 0, "out_tokens": 0, "hit_cap_calls": 0, "think_unclosed_calls": 0,
          "recent": []}
    have = {k: len(v) for k, v in done.items()}
    t0 = time.time()
    stop = {"flag": False, "status": "running"}

    def run_unit(u):
        s, tn, topo, params, label, it = u
        if stop["flag"] or (a.soft_deadline and time.time() >= a.soft_deadline):
            with lock:
                st["units_skipped_deadline"] += 1
            return
        task, cap = T[tn], caps[tn]
        vllm_client.set_thread_cap(cap)
        try:
            extra = {}
            if topo == "HIERFIX":
                res = rw.hierfix(a.model, tn, task, it, params, s)
                extra = {"agg_reminder": res["agg_reminder"],
                         "hierfix_identical_to_hier": res["identical_to_hier"]}
            else:
                res = topologies.get_topology(topo)(a.model, task, it, params, s)
            calls = []
            for i, c in enumerate(res["calls"]):
                calls.append({
                    "idx": i, "text": c["text"], "in_tokens": c["in_tokens"],
                    "out_tokens": c["out_tokens"], "hit_cap": c["out_tokens"] >= cap,
                    "temperature": c["temperature"], "seed": c["seed"],
                    "latency_s": c["latency_s"],
                    "extracted": topologies._extract(task, it, c["text"]),
                    "finish_reason": c.get("finish_reason"),
                    "reasoning": c.get("reasoning"), "think_closed": c.get("think_closed")})
            fidx = final_call_idx(topo, params, len(calls))
            recomputed = topologies._majority([calls[i]["extracted"] for i in fidx])
            extra["final_call_idx"] = fidx
            extra["pred_recomputed_matches"] = recomputed == res["answer"]
            rec = {
                "item_id": it["id"], "task": tn, "topology": topo, "config": label,
                "params": params, "seed": s, "model": a.model, "split": "all",
                "pred": res["answer"], "gold": it["gold"],
                "correct": bool(task["score"](res["answer"], it["gold"])),
                "level": it.get("level"), **client.cost_of(res["calls"]),
                "cap": cap, "host": a.host, "protocol": "REV3",
                "campaign": "cloud_xmodel", "backend": "vllm", "thinking": thinking,
                "served_model": res["calls"][0].get("served_model"), **extra,
                "calls": calls,
            }
            line = json.dumps(rec, default=str) + "\n"
            with lock:
                files[(tn, s)].write(line)
                files[(tn, s)].flush()
                have[(tn, s)] += 1
                st["units_done"] += 1
                st["calls"] += len(calls)
                st["in_tokens"] += rec["in_tokens"]
                st["out_tokens"] += rec["out_tokens"]
                st["hit_cap_calls"] += sum(c["hit_cap"] for c in calls)
                st["think_unclosed_calls"] += sum(c["think_closed"] is False for c in calls)
                st["recent"] = (st["recent"] + [1])[-200:]
        except Exception as e:  # noqa: BLE001
            with lock:
                st["units_failed"] += 1
                st["recent"] = (st["recent"] + [0])[-200:]
                err_f.write(json.dumps({"ts": round(time.time(), 1), "seed": s, "task": tn,
                                        "config": label, "item_id": it["id"],
                                        "error": f"{type(e).__name__}: {str(e)[:400]}"}) + "\n")
                err_f.flush()
                r = st["recent"]
                if len(r) >= 50 and r.count(0) / len(r) > a.max_fail_frac:
                    stop["flag"], stop["status"] = True, "error_rate"

    def write_progress(status):
        el = time.time() - t0
        with lock:
            p = {k: v for k, v in st.items() if k != "recent"}
            p.update(model=a.model, thinking=thinking, caps=caps, status=status,
                     elapsed_s=round(el, 1), units_total=len(units), units_todo_at_start=len(todo),
                     units_in_files=sum(have.values()),
                     per_task_seed={f"{tn}_s{s}": [have[(tn, s)], expected[(tn, s)]] for (tn, s) in have},
                     seeds_complete=[s for s in seeds if all(have[(tn, s)] >= expected[(tn, s)] for tn in task_names)],
                     out_tok_per_s=round(st["out_tokens"] / el, 1) if el > 0 else None,
                     hit_cap_rate=round(st["hit_cap_calls"] / st["calls"], 4) if st["calls"] else None,
                     updated=time.strftime("%FT%TZ", time.gmtime()))
            done_frac = st["units_done"] / max(1, len(todo))
            p["eta_s"] = round(el * (1 - done_frac) / done_frac, 0) if done_frac > 0 else None
        tmp = os.path.join(a.out_dir, "progress.json.tmp")
        with open(tmp, "w") as f:
            json.dump(p, f, indent=1)
        os.replace(tmp, os.path.join(a.out_dir, "progress.json"))
        return p

    print(f"[{time.strftime('%F %T')}] CLOUD model={a.model} thinking={thinking} caps={caps} "
          f"url={a.base_url} units={len(units)} todo={len(todo)} conc={a.concurrency} "
          f"out={a.out_dir}", flush=True)
    ex = ThreadPoolExecutor(max_workers=a.concurrency)
    futs = [ex.submit(run_unit, u) for u in todo]
    last = 0.0
    status = "done"
    while True:
        n_left = sum(not f.done() for f in futs)
        now = time.time()
        if n_left == 0:
            break
        if stop["flag"]:
            status = stop["status"]
        if a.hard_deadline and now >= a.hard_deadline:
            status = "hard_deadline"
            break
        if now - last >= 30:
            p = write_progress("running")
            print(f"[{time.strftime('%F %T')}] done={p['units_done']} failed={p['units_failed']} "
                  f"left={n_left} out_tok={p['out_tokens']} tok/s={p['out_tok_per_s']} "
                  f"cap={p['hit_cap_rate']} seeds_complete={p['seeds_complete']} eta_s={p['eta_s']}",
                  flush=True)
            last = now
        time.sleep(2)
    if status == "done":
        if stop["flag"]:
            status = stop["status"]
        elif st["units_skipped_deadline"]:
            status = "soft_deadline"
        elif st["units_failed"]:
            status = "done_with_failures"
    p = write_progress(status)
    print(f"[{time.strftime('%F %T')}] CLOUD_END status={status} {json.dumps({k: p[k] for k in ('units_done', 'units_failed', 'seeds_complete', 'out_tokens', 'hit_cap_rate')})}", flush=True)
    for f in files.values():
        f.flush()
    if status == "hard_deadline":
        os._exit(0)  # in-flight threads are abandoned; their units rerun on resume
    ex.shutdown(wait=True)
    return 2 if status == "error_rate" else 0


if __name__ == "__main__":
    sys.exit(main())
