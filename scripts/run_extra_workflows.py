"""Driver for the two extra workflows (scripts/extra_workflows.py):

  DU_DEBATE  Du et al. (2023) multiagent debate, their GSM8K prompts, parser and
             vote (3 agents, rounds=2), on Ollama /api/chat. Label
             DU_DEBATE_N3_R2 (R is their `rounds`, initial answers included:
             6 calls, like harness DEBATE_N3_R1). Suffixes _T{t} / _mt{m} are
             added when temperature / max tokens differ from the defaults.
  HIERFIX    harness HIER M=3 with the repaired aggregator reminder. Label HIERFIX_M3.

Same conventions as scripts/run_power_campaign.py: one seed per process,
--shard i/n, chunks of items, resumable (records of this seed in any shard file
count as done), records in the campaign schema (`calls` with raw text, tokens,
hit_cap, harness `extracted`; `host` from AGENTS_HOST_LABEL) plus:
  DU_DEBATE: pred/correct = their parse + their vote, scored by the harness
    scorer (primary); pred_harness/correct_harness = harness extractor on each
    final-round agent + harness majority; du_parsed_final, harness_extracted_final,
    du_vote_none, du_contexts (every agent's full chat history), per call
    agent/round/n_messages_in/num_predict/du_parsed.
  HIERFIX: agg_reminder (the reminder text used).
Requires OLLAMA_URL.

Usage (one process per seed and shard):
  OLLAMA_URL=http://localhost:21435 /usr/bin/python3 -u scripts/run_extra_workflows.py \
      --task gsm8k --cap 640 --n 100 --seed 1 --model qwen2.5:3b-instruct-ctx8k \
      --out-prefix results/power/gsm8k_extra_qwen3b
"""
import argparse
import glob
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_power_campaign import load_math_levels, read_jsonl  # noqa: E402

HARNESS_DEBATE_TEMP = 0.7  # harness.topologies.debate default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["gsm8k", "math500"])
    ap.add_argument("--model", default="qwen2.5:3b-instruct")
    ap.add_argument("--cap", type=int, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--chunk", type=int, default=50)
    ap.add_argument("--only", default="", help="comma list from DU_DEBATE,HIERFIX (default: both)")
    ap.add_argument("--math-levels", default="", help="e.g. 4-5 (default: harness loader, 1-3)")
    ap.add_argument("--shard", default="", help="i/n: run items i, i+n, ... of each chunk")
    ap.add_argument("--du-agents", type=int, default=3)
    ap.add_argument("--du-rounds", type=int, default=2)
    ap.add_argument("--du-temperature", type=float, default=HARNESS_DEBATE_TEMP,
                    help="Du et al. set none (OpenAI default 1.0); default = harness debate 0.7")
    ap.add_argument("--du-max-tokens", type=int, default=None,
                    help="Du et al. set none (OpenAI default: up to the context); "
                         "default = --cap, as every harness call")
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--host", default=os.environ.get("AGENTS_HOST_LABEL", "rtx4060-pc"))
    a = ap.parse_args()

    if not os.environ.get("OLLAMA_URL", "").strip():
        raise SystemExit("set OLLAMA_URL (tunnel to the GPU host)")
    if os.environ.get("AGENTS_BACKEND", "").strip():
        raise SystemExit("AGENTS_BACKEND must be unset (Ollama backend only)")
    os.environ["AGENTS_NUM_PREDICT"] = str(a.cap)
    sys.path.insert(0, ROOT)
    from harness import client, tasks, topologies  # noqa: E402
    import extra_workflows as xw  # noqa: E402
    assert client.DEFAULT_NUM_PREDICT == a.cap

    task = tasks.get_task(a.task)
    if a.math_levels:
        assert a.task == "math500"
        lo, hi = (int(x) for x in a.math_levels.split("-"))
        items = load_math_levels(tasks, lo, hi, a.n)
    else:
        items = task["load"](n=a.n, split="all")
    sh_i, sh_n = (int(x) for x in a.shard.split("/")) if a.shard else (0, 1)
    assert 0 <= sh_i < sh_n

    du_mt = a.cap if a.du_max_tokens is None else a.du_max_tokens
    du_params = {"N": a.du_agents, "R": a.du_rounds,
                 "temperature": a.du_temperature, "max_tokens": du_mt}
    du_label = (f"DU_DEBATE_N{a.du_agents}_R{a.du_rounds}"
                + (f"_T{a.du_temperature:g}" if a.du_temperature != HARNESS_DEBATE_TEMP else "")
                + (f"_mt{du_mt}" if du_mt != a.cap else ""))
    configs = [("DU_DEBATE", du_label, du_params), ("HIERFIX", "HIERFIX_M3", {"M": 3})]
    if a.only:
        keep = set(a.only.split(","))
        assert keep <= {"DU_DEBATE", "HIERFIX"}, a.only
        configs = [c for c in configs if c[0] in keep]
    if a.task != "gsm8k":
        assert all(c[0] != "HIERFIX" for c in configs), "HIERFIX is GSM8K only; pass --only DU_DEBATE"

    base = os.path.join(ROOT, f"{a.out_prefix}_s{a.seed}")
    out = base + (f"_sh{sh_i}of{sh_n}.jsonl" if a.shard else ".jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    done = {(r["config"], r["item_id"])
            for p in glob.glob(base + ".jsonl") + glob.glob(base + "_sh*.jsonl")
            for r in read_jsonl(p)}
    print(f"[{time.strftime('%F %T')}] EXTRA task={a.task} model={a.model} cap={a.cap} "
          f"n={len(items)} seed={a.seed} shard={a.shard or '-'} chat={xw._chat_url()} "
          f"configs={[c[1] for c in configs]} out={out} done={len(done)}", flush=True)
    t0, new = time.time(), 0
    with open(out, "a") as f:
        for c0 in range(0, len(items), a.chunk):
            chunk = items[c0:c0 + a.chunk][sh_i::sh_n]
            for topo, label, params in configs:
                for it in chunk:
                    if (label, it["id"]) in done:
                        continue
                    extra = {}
                    if topo == "DU_DEBATE":
                        res = xw.du_debate(a.model, a.task, task, it, params, a.seed)
                        spec = xw.DU_TASKS[a.task]
                        extra = {
                            "pred_harness": res["answer_harness"],
                            "correct_harness": bool(task["score"](res["answer_harness"], it["gold"])),
                            "du_parsed_final": res["du_parsed_final"],
                            "harness_extracted_final": res["harness_extracted_final"],
                            "du_vote_none": res["answer"] is None,
                            "du_math_adaptation": spec["adapted"],
                            "du_contexts": res["contexts"],
                            "api": "ollama/api/chat",
                        }
                    else:
                        res = xw.hierfix(a.model, a.task, task, it, params, a.seed)
                        extra = {"agg_reminder": xw.HIERFIX_REMINDER[a.task],
                                 "api": "ollama/api/generate"}
                    correct = bool(task["score"](res["answer"], it["gold"]))
                    calls = []
                    for i, c in enumerate(res["calls"]):
                        npred = c.get("num_predict", a.cap)
                        d = {"idx": i, "text": c["text"], "in_tokens": c["in_tokens"],
                             "out_tokens": c["out_tokens"],
                             "hit_cap": npred > 0 and c["out_tokens"] >= npred,
                             "temperature": c["temperature"], "seed": c["seed"],
                             "latency_s": c["latency_s"],
                             "extracted": topologies._extract(task, it, c["text"])}
                        if topo == "DU_DEBATE":
                            d.update(agent=c["agent"], round=c["round"],
                                     n_messages_in=c["n_messages_in"], num_predict=npred,
                                     du_parsed=xw.DU_TASKS[a.task]["parse"](c["text"]))
                        calls.append(d)
                    rec = {
                        "item_id": it["id"], "task": a.task, "topology": topo,
                        "config": label, "params": params, "seed": a.seed,
                        "model": a.model, "split": "all", "pred": res["answer"],
                        "gold": it["gold"], "correct": correct,
                        "level": it.get("level"),
                        **client.cost_of(res["calls"]),
                        "cap": a.cap, "host": a.host, **extra, "calls": calls,
                    }
                    f.write(json.dumps(rec, default=str) + "\n")
                    f.flush()
                    new += 1
            el = (time.time() - t0) / 3600
            print(f"[{time.strftime('%F %T')}] seed={a.seed} shard={a.shard or '-'} chunk end "
                  f"{min(c0 + a.chunk, len(items))}/{len(items)} done, new records {new}, "
                  f"{el:.2f} h", flush=True)
    print(f"[{time.strftime('%F %T')}] EXTRA_DONE task={a.task} seed={a.seed} new={new} "
          f"total {(time.time() - t0) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
