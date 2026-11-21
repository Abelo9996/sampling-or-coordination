"""Thin driver for the AAIBD revision reruns (no harness changes).

Runs selected configs x seeds at a given generation cap on the SAME items as an
existing reference JSONL, through the unchanged harness code path:
    runner.run_config -> topologies.<topo> -> client.generate
so prompts, per-sample seed derivation (seed*1000+i etc.), decoding, parsing,
scoring and the record schema are exactly those of the original runs.

The cap is applied the same way as for results/math500_trunc.jsonl
(scripts/run_tierc.sh: AGENTS_NUM_PREDICT=1024 python3 scripts/run_matrix.py
--task math500_trunc2): via the AGENTS_NUM_PREDICT environment variable, read
once by harness/client.py at import time. This driver sets it before importing
the harness and asserts the client picked it up.

Resumable: run_config skips (item_id, config, seed) already in --out.

Usage:
  run   --task math500 --model qwen2.5:3b-instruct --cap 1024 --n 60 \
        --configs SC_k2,SC_k4 --seeds 1,2 --ref results/math500_trunc.jsonl \
        --out results/math500_trunc_sc24.jsonl
  compare --new results/x_driftcheck.jsonl --ref results/x.jsonl --config SA
"""
import argparse
import json
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_label(label):
    """Config label (as written by runner.config_label) -> (topology, params)."""
    if label == "SA":
        return "SA", {}
    pats = [(r"SC_k(\d+)$", "SC", ("k",)),
            (r"DEBATE_N(\d+)_R(\d+)$", "DEBATE", ("N", "R")),
            (r"GE_R(\d+)$", "GE", ("R",)),
            (r"HIER_M(\d+)$", "HIER", ("M",))]
    for rx, topo, keys in pats:
        m = re.match(rx, label)
        if m:
            return topo, {k: int(v) for k, v in zip(keys, m.groups())}
    raise SystemExit(f"unsupported config label: {label}")


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def cmd_run(a):
    if os.environ.get("AGENTS_BACKEND", "").strip():
        raise SystemExit("AGENTS_BACKEND must be unset (Ollama backend only)")
    if os.environ.get("OLLAMA_URL", "").strip():
        raise SystemExit("OLLAMA_URL must be unset (local Ollama only)")
    os.environ["AGENTS_NUM_PREDICT"] = str(a.cap)
    sys.path.insert(0, ROOT)
    from harness import client, runner, tasks  # noqa: E402

    assert client.DEFAULT_NUM_PREDICT == a.cap, client.DEFAULT_NUM_PREDICT
    assert client.OLLAMA_URL == "http://localhost:11434/api/generate"

    configs = [parse_label(c) for c in a.configs.split(",")]
    for (t, p), lab in zip(configs, a.configs.split(",")):
        assert runner.config_label(t, p) == lab, (t, p, lab)
    seeds = [int(s) for s in a.seeds.split(",")]

    # Item identity check against the reference file (same ids, same order).
    ids = [it["id"] for it in tasks.get_task(a.task)["load"](n=a.n, split="all")]
    if a.ref:
        ref = read_jsonl(a.ref)
        ref_ids = [r["item_id"] for r in ref if r["config"] == "SA"]
        ref_models = {r["model"] for r in ref}
        assert ref_models == {a.model}, f"model mismatch: ref {ref_models} vs {a.model}"
        n_full = len(ref_ids)
        if a.n == n_full:
            assert ids == ref_ids, "loader item order differs from reference SA records"
        else:  # smoke / partial: must be a prefix of the reference order
            assert ids == ref_ids[:a.n], "loader items are not a prefix of the reference"

    out = os.path.join(ROOT, a.out) if not os.path.isabs(a.out) else a.out
    print(f"[{time.strftime('%F %T')}] RUN task={a.task} model={a.model} "
          f"cap={client.DEFAULT_NUM_PREDICT} n={a.n} configs={a.configs} "
          f"seeds={seeds} out={out}", flush=True)
    t0 = time.time()
    for topo, params in configs:
        t1 = time.time()
        nr, label = runner.run_config(a.model, a.task, topo, params, seeds, a.n,
                                      split="all", out_path=out,
                                      progress=a.progress)
        print(f"[{time.strftime('%F %T')}]   {label}: +{nr} rec "
              f"({(time.time()-t1)/60:.1f} min)", flush=True)
    # completeness check
    recs = read_jsonl(out)
    want = {(i, runner.config_label(t, p), s)
            for i in ids for t, p in configs for s in seeds}
    have = {(r["item_id"], r["config"], r["seed"]) for r in recs}
    missing = want - have
    print(f"[{time.strftime('%F %T')}] DONE {out}: {len(recs)} records, "
          f"{len(want)} expected, {len(missing)} missing, "
          f"{(time.time()-t0)/60:.1f} min", flush=True)
    if missing:
        raise SystemExit(2)


def cmd_compare(a):
    new = {r["item_id"]: r for r in read_jsonl(a.new) if r["config"] == a.config}
    ref = {r["item_id"]: r for r in read_jsonl(a.ref) if r["config"] == a.config}
    common = sorted(set(new) & set(ref))
    pm = sum(new[i]["pred"] == ref[i]["pred"] for i in common)
    cm = sum(new[i]["correct"] == ref[i]["correct"] for i in common)
    om = sum(new[i]["out_tokens"] == ref[i]["out_tokens"] for i in common)
    print(f"compare {a.config}: {a.new} vs {a.ref}")
    print(f"  items new={len(new)} ref={len(ref)} common={len(common)}")
    print(f"  pred match {pm}/{len(common)}; correct match {cm}/{len(common)}; "
          f"out_tokens match {om}/{len(common)}")
    print(f"  acc new={sum(new[i]['correct'] for i in common)}/{len(common)} "
          f"ref={sum(ref[i]['correct'] for i in common)}/{len(common)}")
    for i in common:
        if new[i]["pred"] != ref[i]["pred"] or new[i]["out_tokens"] != ref[i]["out_tokens"]:
            print(f"  diff {i}: pred {ref[i]['pred']!r} -> {new[i]['pred']!r}; "
                  f"correct {ref[i]['correct']} -> {new[i]['correct']}; "
                  f"out_tok {ref[i]['out_tokens']} -> {new[i]['out_tokens']}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--task", required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--cap", type=int, required=True)
    r.add_argument("--n", type=int, required=True)
    r.add_argument("--configs", required=True)
    r.add_argument("--seeds", required=True)
    r.add_argument("--ref", default=None)
    r.add_argument("--out", required=True)
    r.add_argument("--progress", action="store_true")
    c = sub.add_parser("compare")
    c.add_argument("--new", required=True)
    c.add_argument("--ref", required=True)
    c.add_argument("--config", default="SA")
    a = ap.parse_args()
    cmd_run(a) if a.cmd == "run" else cmd_compare(a)


if __name__ == "__main__":
    main()
