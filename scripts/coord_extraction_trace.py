"""Raw-text trace of the headline stochastic workflows, for an answer-extraction
audit (no harness changes).

Reruns SC_k8, DEBATE_N3_R1, GE_R2 and HIER_M3 through the unchanged
harness.topologies functions on the logged item set, with the logged cap and
seed, and writes the raw text of every model call plus what the unchanged
extractor reads from it. Each record also carries the logged prediction of the
reference run for the same item, config and seed, so reproduction can be checked.

Mirrors scripts/run_aaibd_rerun.py: item set checked against the reference SA
records, cap via AGENTS_NUM_PREDICT set before the harness import, local Ollama
only. Resumable: (config, item_id) pairs already in the output are skipped.

Usage:
  /usr/bin/python3 -u scripts/coord_extraction_trace.py --task gsm8k --cap 640 \
      --n 80 --ref results/gsm8k_matrix.jsonl --out results/gsm8k_qwen3b_coord_trace.jsonl
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS = [("DEBATE", {"N": 3, "R": 1}), ("GE", {"R": 2}),
           ("HIER", {"M": 3}), ("SC", {"k": 8})]


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--model", default="qwen2.5:3b-instruct")
    ap.add_argument("--cap", type=int, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if os.environ.get("AGENTS_BACKEND", "").strip():
        raise SystemExit("AGENTS_BACKEND must be unset (Ollama backend only)")
    if os.environ.get("OLLAMA_URL", "").strip():
        raise SystemExit("OLLAMA_URL must be unset (local Ollama only)")
    os.environ["AGENTS_NUM_PREDICT"] = str(a.cap)
    sys.path.insert(0, ROOT)
    from harness import client, runner, tasks, topologies  # noqa: E402

    assert client.DEFAULT_NUM_PREDICT == a.cap, client.DEFAULT_NUM_PREDICT
    assert client.OLLAMA_URL == "http://localhost:11434/api/generate"

    task = tasks.get_task(a.task)
    items = task["load"](n=a.n, split="all")
    ids = [it["id"] for it in items]
    ref = read_jsonl(os.path.join(ROOT, a.ref))
    assert {r["model"] for r in ref} == {a.model}, {r["model"] for r in ref}
    ref_ids = [r["item_id"] for r in ref if r["config"] == "SA"]
    assert ids == ref_ids[:a.n], "loader items differ from the reference SA records"
    ref_pred = {(r["config"], r["item_id"]): (r["pred"], bool(r["correct"]))
                for r in ref if r.get("seed") == a.seed}

    out = os.path.join(ROOT, a.out)
    done = {(r["config"], r["item_id"]) for r in read_jsonl(out)}
    print(f"[{time.strftime('%F %T')}] TRACE model={a.model} task={a.task} "
          f"cap={client.DEFAULT_NUM_PREDICT} n={a.n} seed={a.seed} out={out} "
          f"done={len(done)}", flush=True)
    t0 = time.time()
    with open(out, "a") as f:
        for topo, params in CONFIGS:
            label = runner.config_label(topo, params)
            fn = topologies.get_topology(topo)
            match = n = 0
            for it in items:
                if (label, it["id"]) in done:
                    continue
                res = fn(a.model, task, it, params, a.seed)
                correct = bool(task["score"](res["answer"], it["gold"]))
                calls = [{
                    "idx": i, "text": c["text"],
                    "in_tokens": c["in_tokens"], "out_tokens": c["out_tokens"],
                    "hit_cap": c["out_tokens"] >= a.cap,
                    "temperature": c["temperature"], "seed": c["seed"],
                    "extracted": topologies._extract(task, it, c["text"]),
                } for i, c in enumerate(res["calls"])]
                rp = ref_pred.get((label, it["id"]))
                rec = {
                    "item_id": it["id"], "config": label, "seed": a.seed,
                    "model": a.model, "task": a.task, "cap": a.cap,
                    "question": it["question"], "gold": it["gold"],
                    "pred": res["answer"], "correct": correct,
                    "ref_pred": rp[0] if rp else None,
                    "ref_correct": rp[1] if rp else None,
                    "calls": calls, **client.cost_of(res["calls"]),
                }
                f.write(json.dumps(rec, default=str) + "\n")
                f.flush()
                n += 1
                match += int(rp is not None and str(rp[0]) == str(res["answer"]))
                print(f"  [{label}] {it['id']} pred={res['answer']} gold={it['gold']} "
                      f"{'OK' if correct else 'x'} ref={rp[0] if rp else None} "
                      f"tok={rec['total_tokens']}", flush=True)
            print(f"[{time.strftime('%F %T')}] {label} done: new={n} "
                  f"pred_match_ref={match} ({(time.time()-t0)/60:.1f} min)", flush=True)
    recs = read_jsonl(out)
    want = len(ids) * len(CONFIGS)
    print(f"[{time.strftime('%F %T')}] DONE records={len(recs)}/{want} "
          f"total {(time.time()-t0)/60:.1f} min", flush=True)
    if len(recs) < want:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
