"""Larger-sample, more-seed rerun of the headline workflows with raw outputs kept.

Answers three review points at once: sample size, number of seeds, and an
answer-extraction audit on logged raw texts for every workflow.

Runs, through the unchanged harness.topologies functions:
  SA (greedy, written once under --sa-seed), SC_k{--sc-k}, DEBATE_N3_R1, GE_R2, HIER_M3
for ONE seed per process (one process per seed, or several with --shard i/n when
the server has parallel slots; shard files are *_s{seed}_sh{i}of{n}.jsonl),
in chunks of items so that partial results stay balanced across configurations.
Smaller SC k are not rerun: harness.topologies.sc draws sample i with seed
seed*1000+i, so the samples of SC k' are the first k' calls of the SC run; the
per-call extracted answers are logged so those votes can be recomputed.
--math-levels 4-5 loads the MATH-500 levels the harness loader leaves out.

Records use the standard result schema plus `calls` (raw text, tokens, what the
unchanged extractor reads) and `host`. Resumable. Requires OLLAMA_URL.

Usage (one process per seed):
  OLLAMA_URL=http://localhost:21434 /usr/bin/python3 -u scripts/run_power_campaign.py \
      --task gsm8k --cap 640 --n 500 --seed 1 --out-prefix results/power/gsm8k_qwen3b
"""
import argparse
import glob
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COORD = [("DEBATE", {"N": 3, "R": 1}), ("GE", {"R": 2}), ("HIER", {"M": 3})]


def load_math_levels(tasks, lo, hi, n):
    """MATH-500 items with lo <= level <= hi, in the harness order for that
    level set (same ids and same seed-0 shuffle as harness.tasks.math_load)."""
    rows = []
    with open(os.path.join(tasks.DATA_DIR, "math500.jsonl")) as f:
        for i, line in enumerate(f):
            d = json.loads(line)
            lv = int(d.get("level", 5))
            if lo <= lv <= hi:
                rows.append({"id": f"math-{i}", "question": d["problem"],
                             "gold": d["answer"], "level": lv})
    return tasks._split_take(rows, n, "all", 0)


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for l in f:
            try:
                out.append(json.loads(l))
            except json.JSONDecodeError:
                pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--model", default="qwen2.5:3b-instruct")
    ap.add_argument("--cap", type=int, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--sa-seed", type=int, default=1,
                    help="the process with this seed also runs greedy SA")
    ap.add_argument("--chunk", type=int, default=50)
    ap.add_argument("--sc-k", type=int, default=8,
                    help="SC sample count; smaller k are prefixes of these calls")
    ap.add_argument("--only", default="",
                    help="comma list of topologies to run, e.g. SC,DEBATE (default: all)")
    ap.add_argument("--math-levels", default="",
                    help="e.g. 4-5: MATH-500 levels to load (default: harness loader, 1-3)")
    ap.add_argument("--shard", default="",
                    help="i/n: run items i, i+n, ... of each chunk (several processes "
                         "per seed against a server with parallel slots)")
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--host", default=os.environ.get("AGENTS_HOST_LABEL", "rtx4060-pc"))
    a = ap.parse_args()

    if not os.environ.get("OLLAMA_URL", "").strip():
        raise SystemExit("set OLLAMA_URL (tunnel to the GPU host)")
    if os.environ.get("AGENTS_BACKEND", "").strip():
        raise SystemExit("AGENTS_BACKEND must be unset (Ollama backend only)")
    os.environ["AGENTS_NUM_PREDICT"] = str(a.cap)
    sys.path.insert(0, ROOT)
    from harness import client, runner, tasks, topologies  # noqa: E402
    assert client.DEFAULT_NUM_PREDICT == a.cap

    task = tasks.get_task(a.task)
    if a.math_levels:
        assert a.task == "math500"
        lo, hi = (int(x) for x in a.math_levels.split("-"))
        items = load_math_levels(tasks, lo, hi, a.n)
        if (lo, hi) == (1, 3):
            assert [it["id"] for it in items] == [it["id"] for it in task["load"](n=a.n, split="all")]
    else:
        items = task["load"](n=a.n, split="all")
    sh_i, sh_n = (int(x) for x in a.shard.split("/")) if a.shard else (0, 1)
    assert 0 <= sh_i < sh_n
    base = os.path.join(ROOT, f"{a.out_prefix}_s{a.seed}")
    out = base + (f"_sh{sh_i}of{sh_n}.jsonl" if a.shard else ".jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    # records of this seed in any shard file (and the unsharded file) count as done
    done = {(r["config"], r["item_id"])
            for p in glob.glob(base + ".jsonl") + glob.glob(base + "_sh*.jsonl")
            for r in read_jsonl(p)}
    configs = (([("SA", {})] if a.seed == a.sa_seed else [])
               + [("SC", {"k": a.sc_k})] + COORD)
    if a.only:
        keep = set(a.only.split(","))
        configs = [c for c in configs if c[0] in keep]
        assert configs, a.only
    print(f"[{time.strftime('%F %T')}] CAMPAIGN task={a.task} model={a.model} "
          f"cap={a.cap} n={len(items)} seed={a.seed} shard={a.shard or '-'} url={client.OLLAMA_URL} "
          f"out={out} done={len(done)}", flush=True)
    t0 = time.time()
    new = 0
    with open(out, "a") as f:
        for c0 in range(0, len(items), a.chunk):
            chunk = items[c0:c0 + a.chunk][sh_i::sh_n]
            for topo, params in configs:
                label = runner.config_label(topo, params)
                fn = topologies.get_topology(topo)
                for it in chunk:
                    if (label, it["id"]) in done:
                        continue
                    res = fn(a.model, task, it, params, a.seed)
                    correct = bool(task["score"](res["answer"], it["gold"]))
                    calls = [{
                        "idx": i, "text": c["text"], "in_tokens": c["in_tokens"],
                        "out_tokens": c["out_tokens"],
                        "hit_cap": c["out_tokens"] >= a.cap,
                        "temperature": c["temperature"], "seed": c["seed"],
                        "latency_s": c["latency_s"],
                        "extracted": topologies._extract(task, it, c["text"]),
                    } for i, c in enumerate(res["calls"])]
                    rec = {
                        "item_id": it["id"], "task": a.task, "topology": topo,
                        "config": label, "params": params, "seed": a.seed,
                        "model": a.model, "split": "all", "pred": res["answer"],
                        "gold": it["gold"], "correct": correct,
                        "level": it.get("level"),
                        **client.cost_of(res["calls"]),
                        "cap": a.cap, "host": a.host, "calls": calls,
                    }
                    f.write(json.dumps(rec, default=str) + "\n")
                    f.flush()
                    new += 1
            el = (time.time() - t0) / 3600
            print(f"[{time.strftime('%F %T')}] seed={a.seed} shard={a.shard or '-'} chunk end {min(c0 + a.chunk, len(items))}/"
                  f"{len(items)} done, new records {new}, {el:.2f} h", flush=True)
    print(f"[{time.strftime('%F %T')}] CAMPAIGN_DONE task={a.task} seed={a.seed} "
          f"new={new} total {(time.time() - t0) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
