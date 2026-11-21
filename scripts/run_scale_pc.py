"""Larger-sample power confirmation on GSM8K, generation offloaded to a remote
Ollama (set OLLAMA_URL, e.g. a remote GPU machine). Runs the headline configs at
n=200 so the SC_k8-vs-debate and SC_k8-vs-SA comparisons can be reported with
tighter CIs than the primary n=80 matrix. Single base model (Qwen2.5-3B).

Writes results/gsm8k_scale200.jsonl; checkpointed/resumable. Run with:
  OLLAMA_URL=http://<gpu-host>:11434 python scripts/run_scale_pc.py
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from harness import runner, client  # noqa: E402

OUT = os.path.join(ROOT, "results", "gsm8k_scale200.jsonl")
MODEL = "qwen2.5:3b-instruct"
N = 200
CONFIGS = [
    ("SA", {}, [1]),
    ("SC", {"k": 8}, [1, 2]),
    ("DEBATE", {"N": 3, "R": 1}, [1, 2]),
]

if __name__ == "__main__":
    print(f"== scale-up on {client.OLLAMA_URL} (n={N}) ==", flush=True)
    t0 = time.time()
    for topo, params, seeds in CONFIGS:
        nr, label = runner.run_config(MODEL, "gsm8k", topo, params, seeds, N,
                                      split="all", out_path=OUT, progress=True)
        print(f"  {label}: {nr} new ({time.time()-t0:.0f}s)", flush=True)
    print(f"== done in {time.time()-t0:.0f}s -> {OUT} ==", flush=True)
