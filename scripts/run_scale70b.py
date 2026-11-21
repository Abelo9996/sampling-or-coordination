"""Frontier-scale external-validity run (paper/eaai/SCALE70B_EXPERIMENT.md).

Llama-3.3-70B-Instruct over OpenRouter on the main matrix's 80 GSM8K items,
configs SA, SC k=4, SC k=8, DEBATE N=3 R=1 (configs.matrix.SCALE70B_CONFIGS),
seed 1, unchanged prompts/parser/topologies, item-major order. Resumable:
completed (item, config, seed) records are skipped.

Usage (from compute-optimal-agents/):
  nohup python3 scripts/run_scale70b.py >> results/gsm8k_llama70b.log 2>&1 &

The API key is read from ../agent-consistency/.env (override with
AGENTS_ENV_FILE) into this process's environment only; it is never printed.
Progress lines report counts and spend only (no correctness), so the run can
be monitored without looking at results.
"""
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

ENV_FILE = os.environ.get(
    "AGENTS_ENV_FILE",
    os.path.join(os.path.dirname(ROOT), "agent-consistency", ".env"))
FREE_ID = "meta-llama/llama-3.3-70b-instruct:free"
PAID_ID = "meta-llama/llama-3.3-70b-instruct"
PAID_PROVIDER = {"sort": "price",
                 "max_price": {"prompt": 0.25, "completion": 0.60}}
CAP_USD = 0.50
ITEM_RESERVE_USD = 0.02
N_ITEMS = 80
WORKERS = int(os.environ.get("SCALE70B_WORKERS", "3"))

RES = os.environ.get("SCALE70B_RESULTS_DIR", os.path.join(ROOT, "results"))
OUT = os.path.join(RES, "gsm8k_llama70b.jsonl")
LEDGER = os.path.join(RES, "gsm8k_llama70b_spend.jsonl")
STATE = os.path.join(RES, "gsm8k_llama70b_state.json")


def log(msg):
    print(time.strftime("%Y-%m-%dT%H:%M:%SZ ", time.gmtime()) + msg, flush=True)


def load_key():
    if os.environ.get("OPENROUTER_API_KEY"):
        return
    with open(ENV_FILE) as f:
        for line in f:
            m = re.match(r"\s*(?:export\s+)?OPENROUTER_API_KEY\s*=\s*"
                         r"[\"']?([^\"'\s]+)", line)
            if m:
                os.environ["OPENROUTER_API_KEY"] = m.group(1)
                return
    raise SystemExit("OPENROUTER_API_KEY not found in env file")


def ledger_spend():
    usd, n = 0.0, 0
    if os.path.exists(LEDGER):
        for line in open(LEDGER):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            usd += float(d.get("cost_usd") or 0.0)
            n += 1
    return usd, n


def main():
    load_key()
    os.environ["AGENTS_BACKEND"] = "openai"
    # client-side cap a cent below the hard cap, so calls already in flight on
    # the other worker threads cannot carry the total past CAP_USD
    os.environ["AGENTS_OPENAI_SPEND_CAP_USD"] = str(CAP_USD - 0.01)
    state = json.load(open(STATE)) if os.path.exists(STATE) else {}

    from harness import openai_client as oc  # noqa: E402
    from harness import client, runner, tasks, topologies  # noqa: E402
    from configs import matrix as M  # noqa: E402
    assert client.generate is oc.generate, "backend switch not active"

    usd0, n0 = ledger_spend()
    oc.set_spend(usd0, n0)
    oc.set_ledger(LEDGER)
    log(f"start: ledger spend {usd0:.6f} USD over {n0} calls; cap {CAP_USD}")

    if not state:
        info = oc.key_info()
        state = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 "key_usage_baseline": info.get("usage"),
                 "free_daily_at_start": info.get("free_model_daily_requests")}
        # probe the free variant once (also the smoke test; not a GSM8K item)
        try:
            r = oc.generate(FREE_ID, "What is 17 + 25? Reply with only the number.",
                            temperature=0.0, seed=1, num_predict=8, retries=4)
            state["model"] = FREE_ID
            state["probe"] = {"ok": True, "provider": r.get("provider"),
                              "served_model": r.get("served_model")}
        except Exception as e:  # noqa: BLE001
            state["model"] = PAID_ID
            state["provider_prefs"] = PAID_PROVIDER
            state["probe"] = {"ok": False, "error": str(e)[:300]}
        json.dump(state, open(STATE, "w"), indent=1)
        log(f"model choice: {state['model']} (free probe ok={state['probe']['ok']})")
    model = state["model"]
    if model == PAID_ID:
        os.environ["AGENTS_OPENAI_PROVIDER"] = json.dumps(PAID_PROVIDER)
        if not state.get("paid_smoke"):
            r = oc.generate(PAID_ID, "What is 17 + 25? Reply with only the number.",
                            temperature=0.0, seed=1, num_predict=8)
            state["paid_smoke"] = {"provider": r.get("provider"),
                                   "served_model": r.get("served_model"),
                                   "cost_usd": r.get("cost_usd"),
                                   "in_tokens": r["in_tokens"],
                                   "out_tokens": r["out_tokens"]}
            json.dump(state, open(STATE, "w"), indent=1)
            log(f"paid smoke: provider={r.get('provider')} cost={r.get('cost_usd')}")
    baseline = state.get("key_usage_baseline")

    task = tasks.get_task("gsm8k")
    items = task["load"](n=N_ITEMS, split="all")
    configs = [(t, p, s, runner.config_label(t, p)) for t, p, s in M.SCALE70B_CONFIGS]
    done = runner.load_done(OUT)
    todo = [it for it in items
            if any((it["id"], lab, sd) not in done
                   for _, _, seeds, lab in configs for sd in seeds)]
    log(f"items: {len(items)} total, {len(todo)} with missing records; "
        f"workers={WORKERS}")

    wlock = threading.Lock()
    stop = threading.Event()
    counters = {"items": 0, "records": 0, "errors": 0}

    def key_guard():
        if baseline is None:
            return
        try:
            u = float(oc.key_info().get("usage") or 0.0)
        except Exception:  # noqa: BLE001
            return
        if u - float(baseline) >= CAP_USD:
            log(f"STOP: key usage delta {u - float(baseline):.4f} >= cap")
            stop.set()

    def run_item(it):
        if stop.is_set():
            return
        with oc._SPEND_LOCK:
            spent = oc.SPEND["usd"]
        if spent + ITEM_RESERVE_USD > CAP_USD:
            log(f"STOP: spend {spent:.4f} + reserve would exceed cap")
            stop.set()
            return
        for topo, params, seeds, label in configs:
            for seed in seeds:
                if (it["id"], label, seed) in done or stop.is_set():
                    continue
                fn = topologies.get_topology(topo)
                try:
                    res = fn(model, task, it, params, seed)
                except oc.SpendCapReached as e:
                    log(f"STOP: {e}")
                    stop.set()
                    return
                except Exception as e:  # noqa: BLE001
                    counters["errors"] += 1
                    log(f"ERROR {it['id']} {label}: {type(e).__name__}: "
                        f"{str(e)[:200]} (item left for a later pass)")
                    return
                correct = task["score"](res["answer"], it["gold"])
                cost = client.cost_of(res["calls"])
                calls = [{k: c.get(k) for k in
                          ("text", "in_tokens", "out_tokens", "latency_s",
                           "temperature", "seed", "provider", "served_model",
                           "finish_reason", "cost_usd", "gen_id")}
                         for c in res["calls"]]
                rec = {"item_id": it["id"], "task": "gsm8k", "topology": topo,
                       "config": label, "params": params, "seed": seed,
                       "model": model, "split": "all", "backend": "openai",
                       "pred": res["answer"], "gold": it["gold"],
                       "correct": bool(correct), **cost,
                       "cost_usd": round(sum(c["cost_usd"] or 0.0 for c in calls), 8),
                       "providers": sorted({str(c["provider"]) for c in calls}),
                       "raw": res["raw"], "calls": calls}
                with wlock:
                    with open(OUT, "a") as f:
                        f.write(json.dumps(rec, default=str) + "\n")
                    counters["records"] += 1
        with wlock:
            counters["items"] += 1
            n_it = counters["items"]
        with oc._SPEND_LOCK:
            spent, ncalls = oc.SPEND["usd"], oc.SPEND["calls"]
        log(f"item {it['id']} complete ({n_it}/{len(todo)} this pass); "
            f"records +{counters['records']}; calls {ncalls}; spend {spent:.5f} USD")
        if n_it % 10 == 0:
            key_guard()

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        list(ex.map(run_item, todo))

    done = runner.load_done(OUT)
    missing = sum(1 for it in items for _, _, seeds, lab in configs
                  for sd in seeds if (it["id"], lab, sd) not in done)
    with oc._SPEND_LOCK:
        spent = oc.SPEND["usd"]
    log(f"pass finished: missing records {missing}; errors {counters['errors']}; "
        f"spend {spent:.5f} USD; stop={stop.is_set()}")
    if missing == 0:
        log("SCALE70B COMPLETE")
        return 0
    return 3 if stop.is_set() else 2


if __name__ == "__main__":
    sys.exit(main())
