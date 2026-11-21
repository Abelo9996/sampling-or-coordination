"""Run an experiment matrix in priority order, checkpointed.

Usage:
  python3 scripts/run_matrix.py --task gsm8k --n 100 --set lean
Resumable: re-running skips (item, config, seed) already logged.
"""
import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from harness import runner            # noqa: E402
from configs import matrix as M       # noqa: E402


# Tier-B matrix keys map to an underlying task loader and a specific output file.
# gsm8k_tierb extends the existing gsm8k_matrix.jsonl (incremental sweep);
# math500_trunc writes a separate sensitivity file.
TIERB_MAP = {
    "gsm8k_tierb":    ("gsm8k",   "gsm8k_matrix.jsonl"),
    "math500_trunc":  ("math500", "math500_trunc.jsonl"),
    "gsm8k_tierc":    ("gsm8k",   "gsm8k_matrix.jsonl"),
    "math500_trunc2": ("math500", "math500_trunc.jsonl"),
    "gsm8k_7b":       ("gsm8k",   "gsm8k_7b.jsonl"),
    "mmlu_pro_eng":         ("mmlu_pro_eng", "mmlu_pro_eng_matrix.jsonl"),
    "mmlu_pro_eng_llama3b": ("mmlu_pro_eng", "mmlu_pro_eng_llama3b.jsonl"),
    "mmlu_pro_eng_gemma2b": ("mmlu_pro_eng", "mmlu_pro_eng_gemma2b.jsonl"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="gsm8k")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--set", default="spec", choices=["lean", "full", "spec"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    spec = M.MATRICES[args.task]
    model = args.model or spec["model"]
    n = args.n or spec["n_items"]
    # task key -> underlying task loader + output file
    task_name, out_file = TIERB_MAP.get(args.task, (args.task, f"{args.task}_matrix.jsonl"))
    if args.set == "lean":
        configs = M.LEAN_CONFIGS
    elif args.set == "full":
        configs = M.FULL_CONFIGS
    else:  # "spec": use whatever the matrix entry defines (Tier-B sets live here)
        configs = spec["configs"]
    out = args.out or os.path.join(ROOT, "results", out_file)
    args.task = task_name  # the runner/loader needs the real task name

    print(f"== MATRIX task={args.task} model={model} n={n} "
          f"configs={len(configs)} out={out} ==", flush=True)
    t0 = time.time()
    for topo, params, seeds in configs:
        nr, label = runner.run_config(
            model, args.task, topo, params, seeds, n,
            split="all", out_path=out, progress=True)
        print(f"[done] {label}: +{nr} new records "
              f"({(time.time()-t0)/60:.1f} min elapsed)", flush=True)
    print(f"== MATRIX COMPLETE in {(time.time()-t0)/60:.1f} min ==", flush=True)


if __name__ == "__main__":
    main()
