"""GSM8K with per-call GPU energy measured on the RTX 4060 PC (reviewer request:
measured, not modeled, energy on a main-study task).

Same measurement as the HumanEval energy run (scripts/run_code_energy.py,
paper/eaai/CODE_ENERGY_EXPERIMENT.md): Ollama qwen2.5:3b-instruct (digest
357c53fb659c) on the PC, ONE request at a time, board power from
harness.energy.make_sampler() (NVML sample buffer under WSL, about one sample
per 20 ms), joules per call by trapezoid integration over the call's wall-clock
interval, a query's energy is the sum over its calls; 60 s idle baselines at the
start and the end, and idle-subtracted joules per query.

Same workflows, prompts, cap and seed rule as the main GSM8K study
(scripts/run_rev3_campaign.py): SA (greedy), SC k=8 (k=2, 4 are its first
calls), DEBATE N3 R1, GE R2, HIERFIX M3; cap 640; Ollama's repeat-limit abort
recorded as in rev3. Subset: the first --n items of the main study's 500, seed 1,
in chunks of --chunk items with every workflow per chunk (a stopped run stays
balanced). Each record also lists the GPU compute processes other than Ollama
seen at its end (`gpu_other_pids`) and any game running on the Windows side
(`windows_games`); both must be empty for a clean measurement.

Resumable: (config, item) already in --out are skipped.

  PC: OLLAMA_URL=http://127.0.0.1:11434 python3 -u scripts/run_gsm8k_energy.py --n 130
"""
import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

MODEL = "qwen2.5:3b-instruct"
DIGEST = "357c53fb659c"
CAP = 640
IDLE_S = 60
ORDER = ["SA", "SC", "DEBATE", "GE", "HIERFIX"]
PARAMS = {"DEBATE": {"N": 3, "R": 1}, "GE": {"R": 2}, "HIERFIX": {"M": 3}}


def other_gpu_pids():
    """PIDs of GPU compute processes that are not Ollama (WSL shows names as
    [Not Found], so the command line is read from /proc)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=15).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ["nvidia-smi-failed"]
    pids = []
    for tok in out.split():
        if not tok.strip().isdigit():
            continue
        try:
            cmd = open(f"/proc/{tok}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            cmd = ""
        if "ollama" not in cmd:
            pids.append(int(tok))
    return pids


TASKLIST = "/mnt/c/Windows/System32/tasklist.exe"
GAMES = ("league of legends.exe", "valorant-win64-shipping.exe", "valorant.exe")


def windows_games():
    """Game processes on the Windows side (invisible to nvidia-smi under WSL)."""
    if not os.path.exists(TASKLIST):
        return None
    try:
        out = subprocess.run([TASKLIST, "/FO", "CSV", "/NH"], capture_output=True, text=True,
                             timeout=20).stdout.lower()
    except (OSError, subprocess.TimeoutExpired):
        return ["tasklist-failed"]
    return sorted({g for g in GAMES if f'"{g}"' in out})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=130)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=10)
    ap.add_argument("--configs", default=",".join(ORDER))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "energy", "gsm8k_energy_pc.jsonl"))
    ap.add_argument("--energy", choices=["nvidia", "none"], default="nvidia")
    ap.add_argument("--host", default=os.environ.get("AGENTS_HOST_LABEL", "rtx4060-pc"))
    a = ap.parse_args()

    if os.environ.get("AGENTS_BACKEND", "").strip():
        raise SystemExit("AGENTS_BACKEND must be unset (Ollama backend only)")
    os.environ["AGENTS_NUM_PREDICT"] = str(CAP)
    os.environ.setdefault("AGENTS_ABORT_LOG", os.path.splitext(a.out)[0] + "_aborts.jsonl")
    from harness import client, repeat_abort, runner, tasks, topologies, traces  # noqa: E402
    import rev3_workflows as rw  # noqa: E402
    from run_code_energy import TimedGenerate, idle_baseline  # noqa: E402
    from run_rev3_campaign import final_call_idx  # noqa: E402
    assert client.DEFAULT_NUM_PREDICT == CAP
    repeat_abort.install()

    task = tasks.get_task("gsm8k")
    items = task["load"](n=500, split="all")[:a.n]
    want = a.configs.split(",")
    configs = []
    for topo in ORDER:
        if topo not in want or (topo == "SA" and a.seed != 1):
            continue
        params = {"k": 8} if topo == "SC" else dict(PARAMS.get(topo, {}))
        label = "HIERFIX_M3" if topo == "HIERFIX" else runner.config_label(topo, params)
        configs.append((topo, params, label))

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    sampler = None
    idle_path = os.path.splitext(a.out)[0] + "_idle.jsonl"
    if a.energy == "nvidia":
        from harness.energy import make_sampler
        sampler = make_sampler().start()
        print(f"[energy] sampler={type(sampler).__name__}", flush=True)
        time.sleep(2)
    client.generate = TimedGenerate(client.generate, sampler)
    client.generate(MODEL, "Say OK.", temperature=0.0, seed=1, num_predict=4)  # load the model
    idle_w = idle_baseline(sampler, "start", idle_path) if sampler else None

    traces.repair_tail(a.out)
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            try:
                d = json.loads(line)
                done.add((d["config"], d["item_id"]))
            except (json.JSONDecodeError, KeyError):
                pass
    print(f"[{time.strftime('%F %T')}] GSM8K_ENERGY model={MODEL} cap={CAP} n={len(items)} "
          f"seed={a.seed} configs={[c[2] for c in configs]} url={client.OLLAMA_URL} "
          f"out={a.out} done={len(done)} idle_w={idle_w}", flush=True)
    t_start, new = time.time(), 0
    for c0 in range(0, len(items), a.chunk):
        for topo, params, label in configs:
            for it in items[c0:c0 + a.chunk]:
                if (label, it["id"]) in done:
                    continue
                extra = {}
                if topo == "HIERFIX":
                    res = rw.hierfix(MODEL, "gsm8k", task, it, params, a.seed)
                    extra = {"agg_reminder": res["agg_reminder"],
                             "hierfix_identical_to_hier": res["identical_to_hier"]}
                else:
                    res = topologies.get_topology(topo)(MODEL, task, it, params, a.seed)
                calls = []
                for i, c in enumerate(res["calls"]):
                    d = {"idx": i, "text": c["text"], "in_tokens": c["in_tokens"],
                         "out_tokens": c["out_tokens"], "hit_cap": c["out_tokens"] >= CAP,
                         "temperature": c["temperature"], "seed": c["seed"],
                         "latency_s": c["latency_s"], "t0": c.get("t0"), "t1": c.get("t1"),
                         "energy_j": c.get("energy_j"),
                         "extracted": topologies._extract(task, it, c["text"])}
                    if c.get("aborted"):
                        d["aborted"] = True
                    calls.append(d)
                fidx = final_call_idx(topo, params, len(calls))
                extra["final_call_idx"] = fidx
                extra["pred_recomputed_matches"] = \
                    topologies._majority([calls[i]["extracted"] for i in fidx]) == res["answer"]
                rec = {"item_id": it["id"], "task": "gsm8k", "topology": topo, "config": label,
                       "params": params, "seed": a.seed, "model": MODEL, "split": "all",
                       "pred": res["answer"], "gold": it["gold"],
                       "correct": bool(task["score"](res["answer"], it["gold"])),
                       **client.cost_of(res["calls"]), "cap": CAP, "host": a.host,
                       "protocol": "REV3", "campaign": "gsm8k_energy", **extra}
                if sampler is not None:
                    ej = [c["energy_j"] for c in calls]
                    rec["energy_j"] = None if None in ej else round(sum(ej), 3)
                    if rec["energy_j"] is not None and idle_w is not None:
                        busy = sum(c["t1"] - c["t0"] for c in calls)
                        rec["energy_j_net"] = round(rec["energy_j"] - idle_w * busy, 3)
                    rec["gpu_other_pids"] = other_gpu_pids()
                    rec["windows_games"] = windows_games()
                rec["calls"] = calls
                traces.append_record(a.out, rec)
                new += 1
                print(f"  [{label}] {it['id']} ok={rec['correct']} tok={rec['total_tokens']} "
                      f"lat={rec['latency_s']:.1f}s E={rec.get('energy_j')} "
                      f"other={rec.get('gpu_other_pids')} games={rec.get('windows_games')}", flush=True)
        print(f"[{time.strftime('%F %T')}] chunk end {min(c0 + a.chunk, len(items))}/{len(items)} "
              f"new={new} {(time.time() - t_start) / 3600:.2f} h", flush=True)
    if sampler:
        idle_baseline(sampler, "end", idle_path)
        sampler.stop()
    print(f"[{time.strftime('%F %T')}] GSM8K_ENERGY_DONE new={new} "
          f"{(time.time() - t_start) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
