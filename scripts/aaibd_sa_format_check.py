"""AAIBD revision: answer-format check on greedy single-call (SA) outputs.

The Llama GE trace (scripts/aaibd_ge_trace.py) showed that llama3.2:3b usually
ends with the literal line "The answer is N." so the extractor falls back to
the last number in the text. This script measures how often that happens for
another model by rerunning greedy SA through the unchanged harness with the
raw text saved, and compares the predictions with the logged SA records.

Usage:
  /usr/bin/python3 -u scripts/aaibd_sa_format_check.py MODEL REF_JSONL OUT_JSONL
"""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model, ref_path, out_path = sys.argv[1:4]
for v in ("AGENTS_BACKEND", "OLLAMA_URL"):
    if os.environ.get(v, "").strip():
        raise SystemExit(v + " must be unset")
os.environ["AGENTS_NUM_PREDICT"] = "640"
sys.path.insert(0, ROOT)
from harness import client, tasks, topologies  # noqa: E402

assert client.DEFAULT_NUM_PREDICT == 640


def read(p):
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


ref = {r["item_id"]: r for r in read(os.path.join(ROOT, ref_path))
       if r["config"] == "SA" and r["model"] == model}
task = tasks.get_task("gsm8k")
items = task["load"](n=len(ref), split="all")
assert [it["id"] for it in items] == list(ref), "item set differs from reference"
out = os.path.join(ROOT, out_path)
done = {r["item_id"] for r in read(out)}
with open(out, "a") as f:
    for it in items:
        if it["id"] in done:
            continue
        res = topologies.sa(model, task, it, {}, 1)
        c = res["calls"][0]
        rec = {"item_id": it["id"], "config": "SA", "model": model, "cap": 640,
               "gold": it["gold"], "pred": res["answer"],
               "correct": bool(task["score"](res["answer"], it["gold"])),
               "text": c["text"], "out_tokens": c["out_tokens"]}
        f.write(json.dumps(rec, default=str) + "\n")
        f.flush()
rows = read(out)
lit = sum(bool(re.search(r"answer is N\b", r["text"])) for r in rows)
num = sum(bool(re.search(r"answer is \$?-?[\d,\.]+", r["text"])) for r in rows)
same = sum(str(r["pred"]) == str(ref[r["item_id"]]["pred"]) for r in rows)
print(f"FORMAT_CHECK {model} n={len(rows)} literal_N={lit} numeric_line={num} "
      f"correct={sum(r['correct'] for r in rows)} "
      f"ref_correct={sum(bool(r['correct']) for r in ref.values())} pred_match_ref={same}")
