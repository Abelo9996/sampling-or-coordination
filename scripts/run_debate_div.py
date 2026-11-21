"""Diversity-injected debate probe on GSM8K.

Runs the DEBATED topology (N=3, R=1: distinct persona + temperature per agent,
one base model) on the same 80 GSM8K items and two seeds as the main matrix, so
it is paired with the matrix SC_k8 / SA / DEBATE_N3_R1 configurations. Writes a
separate log (results/gsm8k_debate_div.jsonl); checkpointed and resumable.

Usage: python scripts/run_debate_div.py
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from harness import runner  # noqa: E402

MODEL = "qwen2.5:3b-instruct"
OUT = os.path.join(ROOT, "results", "gsm8k_debate_div.jsonl")

if __name__ == "__main__":
    t0 = time.time()
    nr, label = runner.run_config(
        MODEL, "gsm8k", "DEBATED", {"N": 3, "R": 1}, [1, 2], 80,
        split="all", out_path=OUT, progress=True)
    print(f"== done {label}: {nr} new records in {time.time()-t0:.0f}s -> {OUT} ==",
          flush=True)
