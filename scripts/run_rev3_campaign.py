"""rev3 rerun driver (protocols/REV3_RERUN_PROTOCOL.md). One seed per process.

Same conventions as scripts/run_power_campaign.py (unchanged harness topologies,
chunks of items so partial results stay balanced, --shard i/n for a server with
parallel slots, resumable: records of this seed in any shard file of the prefix
count as done, raw text of every call logged), plus:

  * --configs picks from SA, SC (k = --sc-k), DEBATE (N3 R1), GE (R2), HIER (M3)
    and HIERFIX (M3, scripts/rev3_workflows.py; label HIERFIX_M3).
  * Tasks: gsm8k, math500 (--math-levels lo-hi), mmlu, mmlu_pro_eng, humaneval.
  * Ollama's "token repeat limit" abort is recorded instead of stopping the run
    (harness/repeat_abort.py, as in the MMLU-Pro engineering matrix); aborts are
    appended to <out-prefix>_aborts.jsonl.
  * HumanEval: generation is independent of the extractor (the extracted program
    is used only for votes), so both extractors are applied to the same calls.
    Primary fields pred/correct use the harness ("humaneval", last fenced block)
    extractor exactly as the topology returns them; pred_v2/correct_v2 use the
    rev3 entry-block extractor (harness.tasks.code_canonical_v2) on the same final
    calls with the same vote (topologies._majority). Each call logs `extracted`
    (old) and `extracted_v2`. SC records also carry the votes of the first 2 and 4
    samples (SC k=2, 4 by the harness seed rule) under both extractors, scored.

Usage (one process per seed and shard):
  OLLAMA_URL=http://localhost:21435 AGENTS_HOST_LABEL=rtx4060-pc-par6 \\
    /usr/bin/python3 -u scripts/run_rev3_campaign.py --task humaneval --cap 1024 \\
      --n 164 --seed 1 --configs SA,SC,DEBATE,GE,HIERFIX --sc-k 8 \\
      --shard 0/3 --out-prefix results/rev3/humaneval_qwen3b
"""
import argparse
import glob
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_power_campaign import load_math_levels, read_jsonl  # noqa: E402

ORDER = ["SA", "SC", "DEBATE", "GE", "HIER", "HIERFIX"]
PARAMS = {"DEBATE": {"N": 3, "R": 1}, "GE": {"R": 2}, "HIER": {"M": 3}, "HIERFIX": {"M": 3}}


def final_call_idx(topo, params, n_calls):
    """Indices of the calls whose extracted answers make the returned answer."""
    if topo == "SA":
        return [0]
    if topo == "SC":
        return list(range(n_calls))
    if topo == "DEBATE":
        N = int(params.get("N", 3))
        return list(range(n_calls - N, n_calls))
    return [n_calls - 1]  # GE, HIER, HIERFIX: last call


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True,
                    choices=["gsm8k", "math500", "mmlu", "mmlu_pro_eng", "humaneval"])
    ap.add_argument("--model", default="qwen2.5:3b-instruct")
    ap.add_argument("--cap", type=int, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--sa-seed", type=int, default=1)
    ap.add_argument("--configs", required=True, help="comma list from " + ",".join(ORDER))
    ap.add_argument("--sc-k", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=50)
    ap.add_argument("--math-levels", default="", help="e.g. 1-3 or 4-5")
    ap.add_argument("--shard", default="")
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--host", default=os.environ.get("AGENTS_HOST_LABEL", "rtx4060-pc"))
    a = ap.parse_args()

    if not os.environ.get("OLLAMA_URL", "").strip():
        raise SystemExit("set OLLAMA_URL (tunnel to the GPU host)")
    if os.environ.get("AGENTS_BACKEND", "").strip():
        raise SystemExit("AGENTS_BACKEND must be unset (Ollama backend only)")
    os.environ["AGENTS_NUM_PREDICT"] = str(a.cap)
    os.environ.setdefault("AGENTS_ABORT_LOG", os.path.join(ROOT, a.out_prefix + "_aborts.jsonl"))
    sys.path.insert(0, ROOT)
    from harness import client, repeat_abort, runner, tasks, topologies  # noqa: E402
    import rev3_workflows as rw  # noqa: E402
    assert client.DEFAULT_NUM_PREDICT == a.cap
    repeat_abort.install()

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

    want = a.configs.split(",")
    assert set(want) <= set(ORDER), want
    configs = []
    for topo in ORDER:
        if topo not in want or (topo == "SA" and a.seed != a.sa_seed):
            continue
        params = {"k": a.sc_k} if topo == "SC" else dict(PARAMS.get(topo, {}))
        label = "HIERFIX_M3" if topo == "HIERFIX" else runner.config_label(topo, params)
        configs.append((topo, params, label))

    base = os.path.join(ROOT, f"{a.out_prefix}_s{a.seed}")
    out = base + (f"_sh{sh_i}of{sh_n}.jsonl" if a.shard else ".jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    done = {(r["config"], r["item_id"])
            for p in glob.glob(base + ".jsonl") + glob.glob(base + "_sh*.jsonl")
            for r in read_jsonl(p)}
    is_code = a.task == "humaneval"
    sbx = None
    if is_code:
        from harness import sandbox
        sbx = os.environ.get("AGENTS_SANDBOX") or sandbox.default_backend()
    print(f"[{time.strftime('%F %T')}] REV3 task={a.task} model={a.model} cap={a.cap} "
          f"n={len(items)} seed={a.seed} shard={a.shard or '-'} url={client.OLLAMA_URL} "
          f"configs={[c[2] for c in configs]} out={out} done={len(done)}"
          + (f" sandbox={sbx}" if is_code else ""), flush=True)

    def score(pred, it):
        return bool(task["score"](pred, it["gold"]))

    t0, new = time.time(), 0
    with open(out, "a") as f:
        for c0 in range(0, len(items), a.chunk):
            chunk = items[c0:c0 + a.chunk][sh_i::sh_n]
            for topo, params, label in configs:
                for it in chunk:
                    if (label, it["id"]) in done:
                        continue
                    extra = {}
                    if topo == "HIERFIX":
                        res = rw.hierfix(a.model, a.task, task, it, params, a.seed)
                        extra = {"agg_reminder": res["agg_reminder"],
                                 "hierfix_identical_to_hier": res["identical_to_hier"]}
                    else:
                        res = topologies.get_topology(topo)(a.model, task, it, params, a.seed)
                    correct = score(res["answer"], it)
                    calls = []
                    for i, c in enumerate(res["calls"]):
                        d = {"idx": i, "text": c["text"], "in_tokens": c["in_tokens"],
                             "out_tokens": c["out_tokens"],
                             "hit_cap": c["out_tokens"] >= a.cap,
                             "temperature": c["temperature"], "seed": c["seed"],
                             "latency_s": c["latency_s"],
                             "extracted": topologies._extract(task, it, c["text"])}
                        if c.get("aborted"):
                            d["aborted"] = True
                        if is_code:
                            d["extracted_v2"] = tasks.code_canonical_v2(c["text"], it)
                        calls.append(d)
                    fidx = final_call_idx(topo, params, len(calls))
                    recomputed = topologies._majority([calls[i]["extracted"] for i in fidx])
                    extra["final_call_idx"] = fidx
                    extra["pred_recomputed_matches"] = recomputed == res["answer"]
                    if is_code:
                        pred_v2 = topologies._majority([calls[i]["extracted_v2"] for i in fidx])
                        extra.update(pred_v2=pred_v2, correct_v2=score(pred_v2, it),
                                     extractor="last_block", extractor_v2="entry_block",
                                     sandbox=sbx)
                        if topo == "SC":
                            pre = {}
                            for k in (2, 4):
                                if k < len(calls):
                                    p1 = topologies._majority([c["extracted"] for c in calls[:k]])
                                    p2 = topologies._majority([c["extracted_v2"] for c in calls[:k]])
                                    pre[str(k)] = {"pred": p1, "correct": score(p1, it),
                                                   "pred_v2": p2, "correct_v2": score(p2, it)}
                            extra["sc_prefix"] = pre
                    rec = {
                        "item_id": it["id"], "task": a.task, "topology": topo,
                        "config": label, "params": params, "seed": a.seed,
                        "model": a.model, "split": "all", "pred": res["answer"],
                        "gold": it["gold"] if not is_code else it["gold"]["task_id"],
                        "correct": correct, "level": it.get("level"),
                        **client.cost_of(res["calls"]),
                        "cap": a.cap, "host": a.host, "protocol": "REV3", **extra,
                        "calls": calls,
                    }
                    f.write(json.dumps(rec, default=str) + "\n")
                    f.flush()
                    new += 1
            el = (time.time() - t0) / 3600
            print(f"[{time.strftime('%F %T')}] seed={a.seed} shard={a.shard or '-'} chunk end "
                  f"{min(c0 + a.chunk, len(items))}/{len(items)} done, new records {new}, "
                  f"{el:.2f} h", flush=True)
    print(f"[{time.strftime('%F %T')}] REV3_DONE task={a.task} seed={a.seed} "
          f"shard={a.shard or '-'} new={new} total {(time.time() - t0) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
