"""Resumable correction/corruption trace runner (Mac, local Ollama only).

Usage:
  python3 scripts/run_traces.py --task mmlu [--n 80] [--limit 5]

Protocol: paper/eaai/CRITERION_PREDICTIVE.md. Item-major order (both
workflows per item), one generation seed (1). A restart skips every logged
(item, workflow, seed) and first drops a partial trailing line left by a
power loss, so it resumes cleanly. --limit runs only the first L items (the
throughput smoke; its records are the first records of the main run).
"""
import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Per-task settings fixed by the protocol: (n items, generation cap, output).
SPEC = {
    "mmlu": (80, 640, "mmlu_traces_mac.jsonl"),
    "math500": (80, 1024, "math500_traces_mac.jsonl"),
    "mmlu_pro_eng": (150, 1024, "mmlu_pro_eng_traces_mac.jsonl"),
}
MODEL = "qwen2.5:3b-instruct"
SEED = 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=sorted(SPEC))
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--reverse", action="store_true",
                    help="second worker: walk the items from the end and write "
                         "to the _b shard file; both workers skip each other's "
                         "logged records")
    args = ap.parse_args()
    n, cap, fname = SPEC[args.task]
    n = args.n or n
    # The Mac's own Ollama only (never a tunnel or remote host).
    url = os.environ.get("OLLAMA_URL", "")
    if url and not url.startswith(("http://127.0.0.1:11434", "http://localhost:11434")):
        sys.exit(f"refusing non-local OLLAMA_URL={url}")
    from harness import client, tasks, traces
    client.DEFAULT_NUM_PREDICT = cap
    task = tasks.get_task(args.task)
    items = task["load"](n=n, split="all")
    if args.limit:
        items = items[:args.limit]
    main_out = os.path.join(ROOT, "results", fname)
    shard_out = main_out.replace(".jsonl", "_b.jsonl")
    out = shard_out if args.reverse else main_out
    dropped = traces.repair_tail(out)
    if dropped:
        print(f"[resume] dropped {dropped} bytes of a partial last line", flush=True)
    if args.reverse:
        items = items[::-1]

    def all_done():
        return traces.load_done(main_out) | traces.load_done(shard_out)
    done = all_done()
    print(f"== TRACES task={args.task} n={len(items)} cap={cap} out={out} "
          f"already={len(done)} ==", flush=True)
    t0 = time.time()
    k = 0
    for i, it in enumerate(items):
        done = all_done()   # the other worker may have logged this item
        for wf in ("debate", "ge"):
            if (it["id"], wf, SEED) in done:
                continue
            t1 = time.time()
            rec = traces.trace_record(MODEL, args.task, task, it, wf, SEED)
            rec["wall_s"] = round(time.time() - t1, 3)
            rec["host"] = "mac-m4"
            traces.append_record(out, rec)
            k += 1
            print(f"  {i+1}/{len(items)} {it['id']} {wf} init={rec['init_correct']} "
                  f"final={rec['final_correct']} calls={rec['n_calls']} "
                  f"wall={rec['wall_s']:.1f}s", flush=True)
    print(f"== TRACES DONE {args.task}: +{k} records in "
          f"{(time.time()-t0)/60:.1f} min ==", flush=True)


if __name__ == "__main__":
    main()
