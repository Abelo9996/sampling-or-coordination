"""Modal app: the cross-model replication on H100s under a hard spend cap.

One Modal function per model (plus a smoke function), each on one H100 with a
local vLLM server inside the container (scripts/cloud/container_job.py) and the
standard driver (scripts/cloud/run_cloud_campaign.py). Records go to the Volume
`coa-xmodel-results` under runs/<model>/ and are pulled to results/cloud/<model>/
by scripts/cloud/modal_tools.py (also every 10 minutes during a run).

Spend control (scripts/cloud/budget.py has the cost model):
  * billed rate per container is bounded: GPU + cpu limit + memory limit;
  * every function has a hard timeout from the budget plan; max_containers=1,
    no min_containers, scaledown_window 2 s, retries 0, no web endpoints, no
    schedules; weights are downloaded from Hugging Face in the container;
  * before launching, the plan subtracts this month's spend (billing report and
    the run ledger on the Volume) and launches only what fits under $25 including
    every launched container's full timeout;
  * inside each container the driver stops starting units at the soft deadline
    and exits at the hard deadline; a heartbeat writes elapsed time and cost to
    the Volume ledger every minute;
  * the local entrypoint re-projects spend across all running containers every
    minute and cancels every call and stops the app before the cap.

Run (from the repo root, Modal authenticated; nothing else needed):
  python3 -m modal run scripts/cloud/modal_app.py --mode smoke
  python3 -m modal run scripts/cloud/modal_app.py --mode full [--models qwen3-4b,qwen3-8b]
Plan, status, pull and stop without starting anything: scripts/cloud/modal_tools.py.
"""
import json
import os
import subprocess
import sys
import threading
import time

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import budget as B  # noqa: E402

APP_NAME = "coa-xmodel"
VOL_NAME = "coa-xmodel-results"
VLLM_VERSION = "0.9.2"
# newer transformers register "aimv2" themselves and break vLLM 0.9.2 at import
TRANSFORMERS_VERSION = "4.53.2"
REMOTE = "/root/coa"
ROOT = os.path.dirname(os.path.dirname(HERE))
STARTUP_CHARGE_S = B.STARTUP_CHARGE_S

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOL_NAME, create_if_missing=True)

_FILES = ["harness/__init__.py", "harness/client.py", "harness/vllm_client.py",
          "harness/openai_client.py", "harness/tasks.py", "harness/topologies.py",
          "harness/runner.py", "scripts/run_power_campaign.py", "scripts/run_rev3_campaign.py",
          "scripts/rev3_workflows.py", "scripts/extra_workflows.py",
          "scripts/cloud/run_cloud_campaign.py", "scripts/cloud/container_job.py",
          "scripts/cloud/budget.py", "tasks_data/gsm8k_test.jsonl", "tasks_data/math500.jsonl"]

image = (modal.Image.debian_slim(python_version="3.12")
         .uv_pip_install(f"vllm=={VLLM_VERSION}", f"transformers=={TRANSFORMERS_VERSION}")
         .env({"TOKENIZERS_PARALLELISM": "false"}))
for _f in _FILES:
    image = image.add_local_file(os.path.join(ROOT, _f), f"{REMOTE}/{_f}")
image = image.add_local_file(os.path.join(HERE, "budget.py"), "/root/budget.py")

FN_KW = dict(image=image, gpu="H100", cpu=(B.CPU_REQ, B.CPU_LIMIT),
             memory=(B.MEM_REQ_MIB, B.MEM_LIMIT_MIB), volumes={"/vol": vol},
             max_containers=1, retries=0, scaledown_window=2)


def _job(key, run_id, max_s, campaign_args, probe=False, out_key=None):
    """Container body: heartbeat ledger, vLLM + driver, final commit."""
    t0 = time.time()
    sys.path.insert(0, f"{REMOTE}/scripts/cloud")
    import container_job as cj
    hf_id, thinking, mml = B.MODELS["qwen3-4b" if key == "smoke" else key]
    out_key = out_key or key
    out_dir = f"/vol/runs/{out_key}"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs("/vol/ledger", exist_ok=True)
    hard = t0 + max_s - 90
    soft = hard - (B.DRAIN_S if key != "smoke" else 60)
    led = {"status": "running"}

    def write_ledger():
        el = time.time() - t0
        rec = {"run_id": run_id, "key": out_key, "start": t0, "last": time.time(),
               "elapsed_s": round(el, 1), "startup_charge_s": STARTUP_CHARGE_S,
               "usd_per_s_max": B.USD_PER_S, "usd": round((el + STARTUP_CHARGE_S) * B.USD_PER_S, 4),
               "status": led["status"]}
        with open(f"/vol/ledger/{run_id}__{out_key}.json", "w") as f:
            json.dump(rec, f)

    def beat():
        n = 0
        while led["status"] == "running":
            try:
                write_ledger()
                if n % 2 == 0:
                    vol.commit()
            except Exception as e:  # noqa: BLE001
                print("heartbeat error", e, flush=True)
            n += 1
            time.sleep(60)

    threading.Thread(target=beat, daemon=True).start()
    s = cj.run(hf_id, thinking, out_dir, mml, hard, soft, campaign_args=campaign_args,
               probe=probe, host="modal-h100", concurrency=256, max_num_seqs=256)
    led["status"] = s.get("status", "unknown")
    write_ledger()
    vol.commit()
    s["container_elapsed_s"] = round(time.time() - t0, 1)
    s.pop("progress", None)
    return s


@app.function(image=image, timeout=600, cpu=1.0, memory=4096, max_containers=1, retries=0)
def check_image():
    """CPU only (no GPU): the vLLM server module and the harness import in the image."""
    import importlib
    import subprocess as sp
    out = sp.run([sys.executable, "-c", "import vllm, transformers; "
                  "import vllm.entrypoints.openai.api_server; "
                  "print('vllm', vllm.__version__, 'transformers', transformers.__version__)"],
                 capture_output=True, text=True)
    sys.path.insert(0, REMOTE)
    importlib.import_module("harness.vllm_client")
    return {"rc": out.returncode, "stdout": out.stdout[-2000:], "stderr": out.stderr[-3000:]}


@app.function(timeout=B.SMOKE_TIMEOUT_S, **FN_KW)
def smoke(run_id: str):
    return _job("smoke", run_id, B.SMOKE_TIMEOUT_S,
                ["--tasks", "gsm8k", "--n", "gsm8k:3", "--seeds", "1"], probe=True,
                out_key="smoke-qwen3-4b")


@app.function(timeout=B.STATIC_TIMEOUT_S["qwen3-4b"], **FN_KW)
def run_qwen3_4b(run_id: str, max_s: int):
    return _job("qwen3-4b", run_id, max_s, [])


@app.function(timeout=B.STATIC_TIMEOUT_S["qwen3-8b"], **FN_KW)
def run_qwen3_8b(run_id: str, max_s: int):
    return _job("qwen3-8b", run_id, max_s, [])


@app.function(timeout=B.STATIC_TIMEOUT_S["phi4-mini"], **FN_KW)
def run_phi4_mini(run_id: str, max_s: int):
    return _job("phi4-mini", run_id, max_s, [])


@app.function(timeout=B.STATIC_TIMEOUT_S["qwen25-3b"], **FN_KW)
def run_qwen25_3b(run_id: str, max_s: int):
    return _job("qwen25-3b", run_id, max_s, [])


FUNCS = {"qwen3-4b": run_qwen3_4b, "qwen3-8b": run_qwen3_8b,
         "phi4-mini": run_phi4_mini, "qwen25-3b": run_qwen25_3b}


@app.local_entrypoint()
def main(mode: str = "smoke", models: str = "", budget: float = B.BUDGET_USD):
    import modal_tools as T
    assert budget <= B.BUDGET_USD, "budget above the hard cap in budget.py"
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    spent = T.spent_estimate()
    print(f"run_id={run_id} spent_this_month_est=${spent['usd']:.2f} ({spent['source']}) "
          f"rate_max=${B.USD_PER_H:.3f}/container-h cap=${budget:.2f}", flush=True)
    launches = []  # (key, fn, args, worst_case_usd)
    if mode == "check":
        print(json.dumps(check_image.remote(), indent=1))
        return
    if mode == "smoke":
        launches.append(("smoke-qwen3-4b", smoke, (run_id,), B.usd(B.SMOKE_TIMEOUT_S + STARTUP_CHARGE_S)))
    elif mode == "full":
        want = [m for m in models.split(",") if m] or B.PRIORITY
        for key, t, u, seeds in B.plan(budget=budget, spent=spent["usd"]):
            if key not in want or t <= 0:
                print(f"skip {key}: does not fit the budget plan (or not requested)")
                continue
            max_s = int(min(t, B.STATIC_TIMEOUT_S[key]))
            # worst case uses the Modal timeout (the backstop), not the in-container stop
            launches.append((key, FUNCS[key], (run_id, max_s),
                             B.usd(B.STATIC_TIMEOUT_S[key] + STARTUP_CHARGE_S)))
            print(f"plan {key}: in-container stop {max_s / 3600:.2f} h, Modal timeout "
                  f"{B.STATIC_TIMEOUT_S[key] / 3600:.2f} h, worst case ${launches[-1][3]:.2f}, "
                  f"est seeds {seeds}")
    else:
        raise SystemExit("mode must be check, smoke or full")
    worst = spent["usd"] + sum(x[3] for x in launches)
    if worst > budget:
        raise SystemExit(f"refusing: worst case ${worst:.2f} > cap ${budget:.2f}")
    print(f"worst case total this month after launch: ${worst:.2f} <= ${budget:.2f}", flush=True)
    calls = {}
    for key, fn, args, _ in launches:
        calls[key] = {"fc": fn.spawn(*args), "t": time.time(), "done": False, "result": None}
        print(f"spawned {key}", flush=True)
    last_pull = time.time()
    while not all(c["done"] for c in calls.values()):
        time.sleep(60)
        proj = spent["usd"]
        for key, c in calls.items():
            if not c["done"]:
                try:
                    c["result"] = c["fc"].get(timeout=0)
                    c["done"] = True
                    print(f"finished {key}: {json.dumps(c['result'])[:2000]}", flush=True)
                except (TimeoutError, modal.exception.TimeoutError):
                    pass
                except Exception as e:  # noqa: BLE001
                    c["done"], c["result"] = True, {"status": "exception", "error": str(e)[:500]}
                    print(f"failed {key}: {e}", flush=True)
            el = (c.get("t_end") or time.time()) - c["t"]
            if c["done"] and "t_end" not in c:
                c["t_end"] = time.time()
            proj += B.usd(el + STARTUP_CHARGE_S)
        running = [k for k, c in calls.items() if not c["done"]]
        print(f"[guard {time.strftime('%H:%M:%S')}] projected ${proj:.2f} of ${budget:.2f}; running {running}", flush=True)
        if proj >= budget - 0.25:
            print("BUDGET GUARD: cancelling every call and stopping the app", flush=True)
            for c in calls.values():
                if not c["done"]:
                    try:
                        c["fc"].cancel(terminate_containers=True)
                    except Exception as e:  # noqa: BLE001
                        print("cancel error", e)
            subprocess.run([sys.executable, "-m", "modal", "app", "stop", app.app_id], check=False)
            break
        if time.time() - last_pull > 600:
            T.pull(quiet=True)
            last_pull = time.time()
    T.pull()
    print(json.dumps({k: c["result"] for k, c in calls.items()}, indent=1, default=str))
