"""AAIBD revision: raw-text trace of GE_R2 (and greedy SA) for llama3.2:3b on
the 50 GSM8K items of the logged reruns (no harness changes).

Mirrors scripts/run_aaibd_rerun.py: same item set (checked against
--ref results/gsm8k_llama3b.jsonl SA records), generation cap via
AGENTS_NUM_PREDICT set before the harness is imported, local Ollama only,
params {"R": 2} for GE_R2, seed 1. GE is called through the unchanged
harness.topologies.ge with its optional `trace` dict; SA through the unchanged
harness.topologies.sa (temperature 0). The returned `calls` carry each call's
text, so every call's raw output is written out.

Outputs (one JSON line per item, resumable: item_ids already present are
skipped):
  results/gsm8k_llama3b_ge_trace.jsonl
  results/gsm8k_llama3b_sa_trace.jsonl

Usage:
  /usr/bin/python3 -u scripts/aaibd_ge_trace.py [--n 50] [--seed 1] [--cap 640]
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = "llama3.2:3b"
TASK = "gsm8k"
GE_ROLES = ["generator_draft", "evaluator_critique_r1", "revision_r1",
            "evaluator_critique_r2", "revision_r2"]


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--cap", type=int, default=640)
    ap.add_argument("--ref", default="results/gsm8k_llama3b.jsonl")
    ap.add_argument("--ge-out", default="results/gsm8k_llama3b_ge_trace.jsonl")
    ap.add_argument("--sa-out", default="results/gsm8k_llama3b_sa_trace.jsonl")
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
    params = {"R": 2}
    assert runner.config_label("GE", params) == "GE_R2"

    task = tasks.get_task(TASK)
    items = task["load"](n=a.n, split="all")
    ids = [it["id"] for it in items]
    ref = read_jsonl(os.path.join(ROOT, a.ref))
    ref_ids = [r["item_id"] for r in ref if r["config"] == "SA"]
    assert {r["model"] for r in ref} == {MODEL}
    if a.n == len(ref_ids):
        assert ids == ref_ids, "loader item order differs from reference SA records"
    else:
        assert ids == ref_ids[:a.n], "loader items are not a prefix of the reference"

    ge_out = os.path.join(ROOT, a.ge_out)
    sa_out = os.path.join(ROOT, a.sa_out)
    print(f"[{time.strftime('%F %T')}] TRACE model={MODEL} task={TASK} "
          f"cap={client.DEFAULT_NUM_PREDICT} n={a.n} seed={a.seed} "
          f"ge_out={ge_out} sa_out={sa_out}", flush=True)
    t0 = time.time()

    # Phase 1: GE_R2 (all items), as in the logged run where GE_R2 ran as one config.
    done = {r["item_id"] for r in read_jsonl(ge_out)}
    with open(ge_out, "a") as f:
        for it in items:
            if it["id"] in done:
                continue
            tr = {}
            res = topologies.ge(MODEL, task, it, params, a.seed, trace=tr)
            correct = task["score"](res["answer"], it["gold"])
            assert len(res["calls"]) == len(GE_ROLES), len(res["calls"])
            calls = []
            for role, c in zip(GE_ROLES, res["calls"]):
                calls.append({
                    "role": role, "text": c["text"],
                    "in_tokens": c["in_tokens"], "out_tokens": c["out_tokens"],
                    "hit_cap": c["out_tokens"] >= a.cap,
                    "temperature": c["temperature"], "seed": c["seed"],
                    "latency_s": c["latency_s"],
                    "extracted": (topologies._extract(task, it, c["text"])
                                  if not role.startswith("evaluator") else None),
                })
            rec = {
                "item_id": it["id"], "config": "GE_R2", "seed": a.seed,
                "model": MODEL, "cap": a.cap, "question": it["question"],
                "gold": it["gold"], "pred": res["answer"], "correct": bool(correct),
                "draft_answer": tr.get("draft"),
                "draft_correct": bool(task["score"](tr.get("draft"), it["gold"])),
                "rounds": tr.get("rounds"),
                "revision_answers": [r["answer"] for r in tr.get("rounds", [])],
                "calls": calls, **client.cost_of(res["calls"]),
            }
            f.write(json.dumps(rec, default=str) + "\n")
            f.flush()
            print(f"  [GE_R2 seed={a.seed}] {it['id']} draft={rec['draft_answer']} "
                  f"revs={rec['revision_answers']} pred={rec['pred']} "
                  f"gold={it['gold']} {'OK' if correct else 'x'} "
                  f"tok={rec['total_tokens']}", flush=True)
    print(f"[{time.strftime('%F %T')}] GE phase done ({(time.time()-t0)/60:.1f} min)",
          flush=True)

    # Phase 2: greedy SA (temperature 0, as harness.topologies.sa).
    t1 = time.time()
    done = {r["item_id"] for r in read_jsonl(sa_out)}
    with open(sa_out, "a") as f:
        for it in items:
            if it["id"] in done:
                continue
            res = topologies.sa(MODEL, task, it, {}, a.seed)
            correct = task["score"](res["answer"], it["gold"])
            c = res["calls"][0]
            rec = {
                "item_id": it["id"], "config": "SA", "seed": a.seed,
                "model": MODEL, "cap": a.cap, "question": it["question"],
                "gold": it["gold"], "pred": res["answer"], "correct": bool(correct),
                "text": c["text"], "hit_cap": c["out_tokens"] >= a.cap,
                "temperature": c["temperature"], **client.cost_of(res["calls"]),
            }
            f.write(json.dumps(rec, default=str) + "\n")
            f.flush()
            print(f"  [SA] {it['id']} pred={rec['pred']} gold={it['gold']} "
                  f"{'OK' if correct else 'x'} tok={rec['total_tokens']}", flush=True)
    print(f"[{time.strftime('%F %T')}] SA phase done ({(time.time()-t1)/60:.1f} min)",
          flush=True)

    ge_n = len({r["item_id"] for r in read_jsonl(ge_out)} & set(ids))
    sa_n = len({r["item_id"] for r in read_jsonl(sa_out)} & set(ids))
    print(f"[{time.strftime('%F %T')}] DONE ge={ge_n}/{len(ids)} sa={sa_n}/{len(ids)} "
          f"total {(time.time()-t0)/60:.1f} min", flush=True)
    if ge_n < len(ids) or sa_n < len(ids):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
