"""Cost model and static budget plan for the Modal run (standard library only,
Python 3.9 compatible; imported by modal_app.py locally and in the container).

Cost per container second is bounded from above: every function requests
cpu=(CPU_REQ, CPU_LIMIT) and memory=(MEM_REQ_MIB, MEM_LIMIT_MIB), and Modal bills
max(request, usage) with usage capped by the limit, so the billed rate can not
exceed GPU + CPU_LIMIT cores + MEM_LIMIT GiB. Prices: modal.com/pricing, read
23 Nov 2026 (H100 $0.001097/s, CPU $0.0000131/physical core/s, memory
$0.00000222/GiB/s; region selection 1.15 to 1.75x and non-preemptible 3x apply
only when requested, and neither is requested here). Volume storage is free up
to 1 TiB/month; only results (well under 1 GiB) are stored; model weights are
downloaded from Hugging Face inside each container and never stored.

The plan gives each model a hard container time (its Modal function timeout) in
priority order until the budget is used: Qwen3-4B thinking (3 seeds), Qwen3-8B
thinking (seeds 1, 2, 3 in order, stops when its time is up), then the two
non-thinking models. Token and throughput figures below are calibrated on the smoke run where
measured (see the comments); update them if the first full-run minutes differ.

  python3 scripts/cloud/budget.py            # print the plan
"""
import json
import sys

GPU_USD_S = 0.001097
CPU_USD_CORE_S = 0.0000131
MEM_USD_GIB_S = 0.00000222
CPU_REQ, CPU_LIMIT = 2.0, 4.0
MEM_REQ_MIB, MEM_LIMIT_MIB = 16384, 40960
USD_PER_S = GPU_USD_S + CPU_LIMIT * CPU_USD_CORE_S + MEM_LIMIT_MIB / 1024 * MEM_USD_GIB_S
USD_PER_H = USD_PER_S * 3600

BUDGET_USD = 25.0        # hard cap on projected spend (free credits are $30/month)
RESERVE_USD = 1.5        # smoke run, image builds, rounding
STARTUP_S = 420          # smoke: vLLM ready after 205 s (load, compile, CUDA graphs) + image start
DRAIN_S = 600            # soft deadline: no new units this long before the hard stop
SMOKE_TIMEOUT_S = 900
STARTUP_CHARGE_S = 120  # container start before the function clock runs (billed, not timed)

# model key -> (HF id, thinking, max_model_len)
MODELS = {
    "qwen3-4b": ("Qwen/Qwen3-4B", True, 16384),
    "qwen3-8b": ("Qwen/Qwen3-8B", True, 16384),
    "phi4-mini": ("microsoft/Phi-4-mini-instruct", False, 8192),
    "qwen25-3b": ("Qwen/Qwen2.5-3B-Instruct", False, 8192),
}
PRIORITY = ["qwen3-4b", "qwen3-8b", "phi4-mini", "qwen25-3b"]

# calls per seed (SA adds 1 call per item on seed 1 only): 24 per item
CALLS_PER_ITEM_SEED = 1 * 8 + 6 + 5 + 5
ITEMS = {"gsm8k": 500, "math500": 238}
# mean output tokens per call. Non-thinking: Qwen2.5-3B power campaign means
# (GSM8K 230, MATH L1-3 430). Thinking, from the smoke run of 24 Nov 2026
# (Qwen3-4B on one H100): GSM8K first calls averaged 888 tokens over 256 probe
# requests (64 items, T=0.7; 2.7% hit the 4096 cap) and 423 over the 75 calls of
# the 3-item run; 900 is used. MATH L1-3 is not measured: 2.2x GSM8K assumed.
OUT_TOK_PER_CALL = {
    True: {"gsm8k": 900, "math500": 2000},
    False: {"gsm8k": 230, "math500": 430},
}
# aggregate output tokens/s on one H100 with up to 256 requests in flight. Smoke
# run (Qwen3-4B, bf16, vLLM 0.9.2): 9,257 tok/s over a 10 s window with 95 to 256
# running, 4,757 tok/s averaged over the whole probe including its drain tail;
# 8,000 is used for the campaign (queue kept full, longer MATH contexts). The
# others are scaled guesses (8B: about 0.65x the 4B rate).
THROUGHPUT = {"qwen3-4b": 8000, "qwen3-8b": 5200, "phi4-mini": 12000, "qwen25-3b": 14000}


def usd(seconds):
    return seconds * USD_PER_S


def seed_tokens(key, seed=2):
    th = MODELS[key][1]
    tok = 0
    for t, n in ITEMS.items():
        calls = n * CALLS_PER_ITEM_SEED + (n if seed == 1 else 0)
        tok += calls * OUT_TOK_PER_CALL[th][t]
    return tok


def seed_seconds(key, seed=2):
    return seed_tokens(key, seed) / THROUGHPUT[key]


def full_seconds(key, seeds=(1, 2, 3), margin=1.15):
    return STARTUP_S + DRAIN_S + margin * sum(seed_seconds(key, s) for s in seeds)


def plan(budget=BUDGET_USD, spent=0.0, reserve=RESERVE_USD, priority=PRIORITY):
    """[(key, timeout_s, usd, est_seeds)] in priority order. A thinking model is
    dropped when less than startup + a quarter seed fits; a non-thinking model
    when its full 3 seeds do not fit."""
    left = (budget - spent - reserve) / USD_PER_S
    out = []
    for key in priority:
        need = full_seconds(key)
        alloc = min(need, left - STARTUP_CHARGE_S)
        partial_ok = MODELS[key][1]  # thinking models may stop after some seeds;
        # non-thinking ones run only in full (otherwise the free Kaggle notebook)
        if alloc < (STARTUP_S + DRAIN_S + 0.25 * seed_seconds(key) if partial_ok else need):
            out.append((key, 0, 0.0, 0.0))
            continue
        work = alloc - STARTUP_S - DRAIN_S
        est_seeds, rem = 0.0, work
        for s in (1, 2, 3):
            need_s = 1.15 * seed_seconds(key, s)
            take = min(rem, need_s)
            est_seeds += take / need_s
            rem -= take
        out.append((key, int(alloc), round(usd(alloc + STARTUP_CHARGE_S), 2), round(est_seeds, 2)))
        left -= alloc + STARTUP_CHARGE_S
    return out


# Static timeouts used at function definition (Modal fixes a function's timeout
# when it is defined). Derived from the plan with zero prior spend.
STATIC_TIMEOUT_S = {k: max(t, 600) for k, t, _, _ in plan()}


def report(spent=0.0):
    rows = []
    for key, t, u, s in plan(spent=spent):
        rows.append({"model": key, "hf_id": MODELS[key][0], "thinking": MODELS[key][1],
                     "est_tokens_per_seed_M": round(seed_tokens(key) / 1e6, 1),
                     "est_h_per_seed": round(seed_seconds(key) / 3600, 2),
                     "est_usd_per_seed": round(usd(seed_seconds(key)), 2),
                     "est_h_3_seeds_incl_startup": round(full_seconds(key) / 3600, 2),
                     "est_usd_3_seeds": round(usd(full_seconds(key)), 2),
                     "planned_timeout_h": round(t / 3600, 2), "planned_usd_max": u,
                     "planned_seeds": s})
    return {"usd_per_container_h_max": round(USD_PER_H, 4), "budget_usd": BUDGET_USD,
            "reserve_usd": RESERVE_USD, "spent_usd": spent,
            "planned_total_usd_max": round(sum(r["planned_usd_max"] for r in rows), 2),
            "rows": rows}


if __name__ == "__main__":
    sp = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
    print(json.dumps(report(sp), indent=1))
