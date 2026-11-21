"""Overnight cross-family run (zero cost, local M4). Serial, one model in the
GPU at a time, checkpointed (runner skips already-logged item/config/seed).

Priority order: the make-or-break GSM8K cross-family replication runs first, so
if we run out of wall-clock the critical evidence is already on disk. MMLU (null
replication) and the mechanistic resume follow.

Full lean headline config set (matches the Qwen tables) at 1 seed, n=50, so each
model finishes in ~4-5h. State n and seed honestly in the writeup.
"""
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from harness import runner  # noqa: E402

N = int(os.environ.get("XFAM_N", "50"))
SEEDS = [1]

# (topology, params) -- full lean headline set, single seed each.
CONFIGS = [
    ("SA",     {}),
    ("SC",     {"k": 2}),
    ("SC",     {"k": 4}),
    ("SC",     {"k": 8}),
    ("DEBATE", {"N": 3, "R": 1}),
    ("GE",     {"R": 2}),
    ("HIER",   {"M": 3}),
]

# Priority-ordered pipeline: (model, task, out_file)
PLAN = [
    ("llama3.2:3b", "gsm8k", "gsm8k_llama3b.jsonl"),
    ("gemma2:2b",   "gsm8k", "gsm8k_gemma2b.jsonl"),
    ("llama3.2:3b", "mmlu",  "mmlu_llama3b.jsonl"),
    ("gemma2:2b",   "mmlu",  "mmlu_gemma2b.jsonl"),
]


def ollama_stop(model):
    try:
        subprocess.run(["ollama", "stop", model], timeout=30,
                       capture_output=True)
    except Exception:
        pass


def stamp():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def main():
    t_start = time.time()
    prev_model = None
    for model, task, out_file in PLAN:
        if prev_model and prev_model != model:
            ollama_stop(prev_model)  # free GPU before loading the next family
            time.sleep(3)
        prev_model = model
        out = os.path.join(ROOT, "results", out_file)
        print(f"\n[{stamp()}] === {model} x {task} (n={N}) -> {out_file} ===",
              flush=True)
        for topo, params in CONFIGS:
            t0 = time.time()
            nr, label = runner.run_config(
                model, task, topo, params, SEEDS, N,
                split="all", out_path=out, progress=False)
            print(f"[{stamp()}]   {label}: +{nr} rec "
                  f"({(time.time()-t0)/60:.1f} min, "
                  f"{(time.time()-t_start)/3600:.2f} h total)", flush=True)
    ollama_stop(prev_model)
    print(f"\n[{stamp()}] === XFAM PIPELINE COMPLETE in "
          f"{(time.time()-t_start)/3600:.2f} h ===", flush=True)


if __name__ == "__main__":
    main()
