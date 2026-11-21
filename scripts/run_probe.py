"""Run a cheap 2-sample agreement probe (SC k=2, raw logged) over the same
items as the matrix. The router uses inter-sample agreement as an
inference-time difficulty signal. Cost: 2 calls/item.

  python3 scripts/run_probe.py --task gsm8k --n 80
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from harness import runner  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="gsm8k")
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--model", default="qwen2.5:3b-instruct")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--suffix", default="")
    args = ap.parse_args()
    out = os.path.join(ROOT, "results", f"{args.task}_probe{args.suffix}.jsonl")
    runner.run_config(args.model, args.task, "SC", {"k": 2}, seeds=[args.seed],
                      n_items=args.n, split="all", out_path=out,
                      log_raw=True, progress=True)
    print("probe done:", out)


if __name__ == "__main__":
    main()
