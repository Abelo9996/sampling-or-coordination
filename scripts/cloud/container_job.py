"""Run one model inside a GPU container (Modal, Kaggle, any Linux box with vLLM):
start a local vLLM OpenAI server, wait for it, optionally run a throughput probe,
run scripts/cloud/run_cloud_campaign.py against it, stop the server.

Platform agnostic and standard library only. The caller passes wall-clock
deadlines (epoch seconds); this module never reads credentials.

  python3 scripts/cloud/container_job.py --hf-id Qwen/Qwen3-4B --thinking on \\
      --out-dir /kaggle/working/qwen3-4b --hard-deadline $(( $(date +%s) + 3600 ))
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


def vllm_cmd(hf_id, port, max_model_len, dtype="bfloat16", max_num_seqs=256,
             gpu_mem=0.92, extra=()):
    return [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", hf_id, "--served-model-name", hf_id,
            "--host", "127.0.0.1", "--port", str(port), "--dtype", dtype,
            "--max-model-len", str(max_model_len), "--max-num-seqs", str(max_num_seqs),
            "--gpu-memory-utilization", str(gpu_mem),
            # T4 (sm75) on vLLM V0: prefix caching breaks (Phi-4 sliding window; Triton
            # prefix-prefill kernel fails in fp16), so Kaggle sets COA_NO_PREFIX_CACHE=1.
            *([] if os.environ.get("COA_NO_PREFIX_CACHE") == "1" else ["--enable-prefix-caching"]),
            "--seed", "0", "--disable-log-requests", *extra]


def wait_ready(port, proc, timeout_s):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if proc.poll() is not None:
            raise RuntimeError(f"vLLM exited with code {proc.returncode} before ready")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5) as r:
                if r.status == 200:
                    return time.time() - t0
        except Exception:  # noqa: BLE001
            pass
        time.sleep(5)
    raise RuntimeError(f"vLLM not ready after {timeout_s} s")


def throughput_probe(hf_id, port, thinking, n_req=256, max_tokens=4096, wall_s=150):
    """Aggregate output tokens/s with n_req GSM8K first-call prompts in flight
    (seeds 9000+i; not written as records). Measures the batched throughput the
    budget plan needs."""
    sys.path.insert(0, ROOT)
    from harness import tasks
    t = tasks.get_task("gsm8k")
    items = t["load"](n=64, split="all")
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    toks, lat, fins = [], [], []
    lock = threading.Lock()
    t_end = time.time() + wall_s

    def one(i):
        body = {"model": hf_id, "messages": [{"role": "user", "content": t["prompt"](items[i % 64]["question"])}],
                "temperature": 0.7, "seed": 9000 + i, "max_tokens": max_tokens}
        if thinking:
            body["chat_template_kwargs"] = {"enable_thinking": True}
        req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        t1 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=max(5, t_end - t1)) as r:
                d = json.loads(r.read())
        except Exception:  # noqa: BLE001
            return
        with lock:
            toks.append(d["usage"]["completion_tokens"]); lat.append(time.time() - t1)
            fins.append(d["choices"][0].get("finish_reason"))

    g0 = gen_tokens_total(port)
    t0 = time.time()
    with ThreadPoolExecutor(n_req) as ex:
        list(ex.map(one, range(n_req)))
    wall = time.time() - t0
    g1 = gen_tokens_total(port)
    out = {"n_req": n_req, "n_completed": len(toks), "wall_s": round(wall, 1),
           "out_tokens_completed": sum(toks),
           "mean_out_tokens_completed": round(sum(toks) / max(1, len(toks)), 1),
           "length_stops": fins.count("length")}
    if g0 is not None and g1 is not None:
        # server counter: every generated token, finished or not
        out["agg_out_tok_per_s"] = round((g1 - g0) / wall, 1)
        out["source"] = "vllm:generation_tokens_total"
    else:
        out["agg_out_tok_per_s"] = round(sum(toks) / wall, 1)
        out["source"] = "completed requests only (lower bound)"
    return out


def gen_tokens_total(port):
    """vLLM Prometheus counter of generated tokens (None if unavailable)."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=10) as r:
            txt = r.read().decode()
    except Exception:  # noqa: BLE001
        return None
    tot = None
    for line in txt.splitlines():
        if line.startswith("vllm:generation_tokens_total"):
            tot = (tot or 0.0) + float(line.rsplit(" ", 1)[1])
    return tot


def run(hf_id, thinking, out_dir, max_model_len, hard_deadline, soft_deadline=0.0,
        port=8000, dtype="bfloat16", campaign_args=(), probe=False, gpu_env=None,
        max_num_seqs=256, concurrency=256, host="cloud", ready_timeout_s=1500, log=print):
    os.makedirs(out_dir, exist_ok=True)
    tag = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    summary = {"hf_id": hf_id, "thinking": thinking, "started": tag, "status": "starting"}
    env = dict(os.environ, **(gpu_env or {}))
    vlog = open(os.path.join(out_dir, f"vllm_{tag}.log"), "w")
    proc = subprocess.Popen(vllm_cmd(hf_id, port, max_model_len, dtype, max_num_seqs),
                            stdout=vlog, stderr=subprocess.STDOUT, env=env)
    try:
        summary["ready_s"] = round(wait_ready(port, proc, min(ready_timeout_s, max(60, hard_deadline - time.time() - 60))), 1)
        log(f"vLLM ready after {summary['ready_s']} s")
        if probe:
            summary["probe"] = throughput_probe(hf_id, port, thinking)
            log(f"probe: {summary['probe']}")
        cmd = [sys.executable, "-u", os.path.join(HERE, "run_cloud_campaign.py"),
               "--model", hf_id, "--thinking", "on" if thinking else "off",
               "--base-url", f"http://127.0.0.1:{port}/v1", "--out-dir", out_dir,
               "--concurrency", str(concurrency), "--host", host,
               "--hard-deadline", str(hard_deadline),
               "--soft-deadline", str(soft_deadline or hard_deadline), *campaign_args]
        dlog = open(os.path.join(out_dir, f"driver_{tag}.log"), "w")
        drv = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        for line in drv.stdout:
            dlog.write(line); dlog.flush(); log(line.rstrip())
        drv.wait()
        summary["driver_rc"] = drv.returncode
        try:
            summary["progress"] = json.load(open(os.path.join(out_dir, "progress.json")))
            summary["status"] = summary["progress"]["status"]
        except Exception:  # noqa: BLE001
            summary["status"] = "no_progress_file"
    except Exception as e:  # noqa: BLE001
        summary["status"] = "error"
        summary["error"] = f"{type(e).__name__}: {e}"
        log(f"ERROR {summary['error']}")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        vlog.close()
        with open(os.path.join(out_dir, f"summary_{tag}.json"), "w") as f:
            json.dump(summary, f, indent=1)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-id", required=True)
    ap.add_argument("--thinking", choices=["on", "off"], required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--hard-deadline", type=float, required=True)
    ap.add_argument("--soft-deadline", type=float, default=0.0)
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--concurrency", type=int, default=256)
    ap.add_argument("--host", default="cloud")
    ap.add_argument("campaign_args", nargs=argparse.REMAINDER,
                    help="after --: extra run_cloud_campaign.py args, e.g. -- --n gsm8k:3 --seeds 1")
    a = ap.parse_args()
    extra = [x for x in a.campaign_args if x != "--"]
    s = run(a.hf_id, a.thinking == "on", a.out_dir, a.max_model_len, a.hard_deadline,
            a.soft_deadline, a.port, a.dtype, extra, a.probe, None, a.max_num_seqs,
            a.concurrency, a.host)
    print(json.dumps({k: v for k, v in s.items() if k != "progress"}))
    return 0 if s["status"] in ("done", "soft_deadline") else 1


if __name__ == "__main__":
    sys.exit(main())
