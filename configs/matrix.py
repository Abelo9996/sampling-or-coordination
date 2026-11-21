"""Experiment matrix definitions (pure Python, no YAML dependency).

Config order == run priority: SA and SC (the core single-agent vs.
test-time-compute comparison) run first, so a partial run still yields the
paper's central claim. MAS topologies follow. Seeds: greedy SA is
deterministic (1 seed); stochastic configs use 2 seeds for error bars
(bootstrap over items provides the primary CIs).

A "lean" matrix is the default so the core task finishes in roughly one
overnight on an M4; widen seeds/sweeps via the runner flags if time allows.
"""

# CORE comparison matrix (~61 calls/item): SA + the single-agent test-time
# curve (SC k=2,4,8) + one point per MAS topology. This alone yields the
# headline crossover/H1 result; measured at ~7 s/call on the M4, n=80 ~= 9.5 h.
# (topology, params, seeds)
LEAN_CONFIGS = [
    ("SA",     {},                 [1]),
    ("SC",     {"k": 2},           [1, 2]),
    ("SC",     {"k": 4},           [1, 2]),
    ("SC",     {"k": 8},           [1, 2]),
    ("DEBATE", {"N": 3, "R": 1},   [1, 2]),
    ("GE",     {"R": 2},           [1, 2]),
    ("HIER",   {"M": 3},           [1, 2]),
]

# Enrichment configs: extra compute points per MAS topology to draw their own
# scaling curves. Run as a second stage (resumes into the same JSONL).
FULL_CONFIGS = LEAN_CONFIGS + [
    ("SC",     {"k": 12},          [1, 2]),
    ("DEBATE", {"N": 2, "R": 1},   [1, 2]),
    ("DEBATE", {"N": 3, "R": 2},   [1, 2]),
    ("GE",     {"R": 1},           [1, 2]),
    ("HIER",   {"M": 2},           [1, 2]),
]

# TIER-B (reviewer-driven), focused for tractability on the M4. Two pieces:
#
# (1) GSM8K MAS-frontier sweep: turns each single multi-agent point into a CURVE
#     (a cheap 2nd operating point per topology) and adds SC_k12 so the priciest
#     MAS configs have a *measured* equal-budget SC neighbor (no extrapolation).
#     Run INCREMENTALLY into the existing gsm8k_matrix.jsonl (same 640 cap, 2
#     seeds) -- valid because GSM8K has ~0% generation truncation; the runner
#     skips already-logged (item,config,seed), so only the new rows below run.
TIERB_GSM8K_SWEEP = LEAN_CONFIGS + [
    ("SC",     {"k": 12},          [1, 2]),
    ("DEBATE", {"N": 2, "R": 1},   [1, 2]),
    ("GE",     {"R": 1},           [1, 2]),
    ("HIER",   {"M": 2},           [1, 2]),
]

# (2) MATH truncation sensitivity: re-run the headline configs at a HIGHER cap
#     (client.num_predict raised) into a separate file, to measure how much of
#     each gap is real vs. a 640-token truncation artifact (MATH had 10-17%
#     truncation; GSM8K/MMLU ~0%). One seed suffices for a sensitivity check.
TIERB_MATH_TRUNC = [
    ("SA",     {},                 [1]),
    ("SC",     {"k": 8},           [1]),
    ("SC",     {"k": 12},          [1]),
    ("DEBATE", {"N": 3, "R": 1},   [1]),
    ("GE",     {"R": 2},           [1]),
    ("HIER",   {"M": 3},           [1]),
]

# TIER-C (round-2 reviewer-driven):
# (1) Fixed-implementation topologies on GSM8K: GE with a stop gate (GES),
#     sequentially-chained HIER (HIERC), and debate at its canonical depth
#     knob (R=2). Extends gsm8k_matrix.jsonl incrementally.
TIERC_GSM8K = TIERB_GSM8K_SWEEP + [
    ("GES",    {"R": 2},           [1, 2]),
    ("HIERC",  {"M": 3},           [1, 2]),
    ("DEBATE", {"N": 3, "R": 2},   [1, 2]),
]
# (2) Second seed at the confound-free 1024 cap on MATH (power for the paired
#     tests the abstract relies on). SA is deterministic; SC_k12 dropped for
#     budget (k8 is the headline comparator).
TIERC_MATH_TRUNC = TIERB_MATH_TRUNC + [
    ("SC",     {"k": 8},           [1, 2]),
    ("DEBATE", {"N": 3, "R": 1},   [1, 2]),
    ("GE",     {"R": 2},           [1, 2]),
    ("HIER",   {"M": 3},           [1, 2]),
]
# (3) 7B scale probe on GSM8K: does the frontier ordering hold at 7B?
TIERC_7B = [
    ("SA",     {},                 [1]),
    ("SC",     {"k": 8},           [1, 2]),
    ("DEBATE", {"N": 3, "R": 1},   [1, 2]),
]

# EAAI engineering-domain experiment (paper/eaai/ENGINEERING_EXPERIMENT.md):
# the unchanged headline set on MMLU-Pro "engineering" (ten-option MCQ, A..J),
# n=150 items sampled once by scripts/prep_mmlu_pro_eng.py (seed 0).
# Qwen2.5-3B primary at the headline seeds; optional cross-family probe at one
# seed, matching scripts/overnight_xfamily.py (SA..HIER, seed 1).
MMLU_PRO_ENG_CONFIGS = LEAN_CONFIGS
XFAM_1SEED_CONFIGS = [(t, p, [1]) for t, p, _ in LEAN_CONFIGS]

# Frontier-scale external-validity check (paper/eaai/SCALE70B_EXPERIMENT.md):
# Llama-3.3-70B-Instruct over an OpenAI-compatible API (AGENTS_BACKEND=openai),
# same 80 GSM8K items as the main matrix, one seed, headline debate config.
# Run by scripts/run_scale70b.py (item-major order) -> results/gsm8k_llama70b.jsonl
SCALE70B_CONFIGS = [
    ("SA",     {},                 [1]),
    ("SC",     {"k": 4},           [1]),
    ("SC",     {"k": 8},           [1]),
    ("DEBATE", {"N": 3, "R": 1},   [1]),
]

# Code task with executable hidden tests (paper/eaai/CODE_ENERGY_EXPERIMENT.md):
# HumanEval, all 164 problems, the unchanged 7 headline configurations, run on
# the RTX 4060 PC by scripts/run_code_energy.py with per-call GPU energy.
# SC/DEBATE vote over AST-normalized programs (execution-free).
HUMANEVAL_CONFIGS = LEAN_CONFIGS

MATRICES = {
    "gsm8k":   {"model": "qwen2.5:3b-instruct", "n_items": 80, "configs": LEAN_CONFIGS},
    # additional task classes (knowledge MCQ + harder math) to test generality
    "mmlu":    {"model": "qwen2.5:3b-instruct", "n_items": 80, "configs": LEAN_CONFIGS},
    "math500": {"model": "qwen2.5:3b-instruct", "n_items": 60, "configs": LEAN_CONFIGS},
    # Tier-B
    "gsm8k_tierb": {"model": "qwen2.5:3b-instruct", "n_items": 80,
                    "configs": TIERB_GSM8K_SWEEP},
    "math500_trunc": {"model": "qwen2.5:3b-instruct", "n_items": 60,
                      "configs": TIERB_MATH_TRUNC},
    # Tier-C
    "gsm8k_tierc": {"model": "qwen2.5:3b-instruct", "n_items": 80,
                    "configs": TIERC_GSM8K},
    "math500_trunc2": {"model": "qwen2.5:3b-instruct", "n_items": 60,
                       "configs": TIERC_MATH_TRUNC},
    "gsm8k_7b": {"model": "qwen2.5:7b-instruct", "n_items": 50,
                 "configs": TIERC_7B},
    # EAAI engineering-domain task (output files mapped in scripts/run_matrix.py)
    "mmlu_pro_eng": {"model": "qwen2.5:3b-instruct", "n_items": 150,
                     "configs": MMLU_PRO_ENG_CONFIGS},
    "mmlu_pro_eng_llama3b": {"model": "llama3.2:3b", "n_items": 150,
                             "configs": XFAM_1SEED_CONFIGS},
    "mmlu_pro_eng_gemma2b": {"model": "gemma2:2b", "n_items": 150,
                             "configs": XFAM_1SEED_CONFIGS},
    "humaneval": {"model": "qwen2.5:3b-instruct", "n_items": 164,
                  "configs": HUMANEVAL_CONFIGS},
}


def calls_per_item(configs):
    """Estimate total model calls per item for a config list (for budgeting)."""
    per = {"SA": lambda p: 1,
           "SC": lambda p: p["k"],
           "DEBATE": lambda p: p["N"] * (p["R"] + 1),
           "GE": lambda p: 1 + 2 * p["R"],
           "GES": lambda p: 1 + 2 * p["R"],   # upper bound; stop gate may cut it
           "HIER": lambda p: p["M"] + 2,
           "HIERC": lambda p: p["M"] + 2}
    return sum(per[t](p) * len(s) for t, p, s in configs)


if __name__ == "__main__":
    print("LEAN calls/item:", calls_per_item(LEAN_CONFIGS))
    print("FULL calls/item:", calls_per_item(FULL_CONFIGS))
    print("MMLU-Pro eng calls/item:", calls_per_item(MMLU_PRO_ENG_CONFIGS))
    print("x-family 1-seed calls/item:", calls_per_item(XFAM_1SEED_CONFIGS))
