# Cross-model replication on rented GPUs (Modal primary, Kaggle fallback)

Replicates the paper's main experiment on four more models with the unchanged
harness: GSM8K first 500 test items and MATH-500 levels 1 to 3 (the 238 items of
power campaign phase B), seeds 1 to 3, workflows SA (greedy, seed 1), SC k=8
(SC k=2, 4 are the first 2, 4 calls), DEBATE N3 R1, GE R2, HIERFIX M3 (the HIER
the rev3 paper reports in the main text; on MATH it is byte-identical to HIER).
Same prompts, seeds, extraction and record schema as `scripts/run_rev3_campaign.py`,
so `scripts/analyze_rev3_reruns.py` functions read the logs unchanged
(`scripts/cloud/analyze_cloud.py` runs them on a cloud directory).

| model | where | mode | dtype | caps GSM8K / MATH |
|-|-|-|-|-|
| Qwen/Qwen3-4B | Modal H100 | thinking on | bf16 | 4096 / 8192 |
| Qwen/Qwen3-8B | Modal H100 | thinking on | bf16 | 4096 / 8192 |
| microsoft/Phi-4-mini-instruct | Kaggle 2x T4 (Modal if budget allows) | non-thinking | fp16 on T4 | 640 / 1024 |
| Qwen/Qwen2.5-3B-Instruct | Kaggle 2x T4 (Modal if budget allows) | non-thinking | fp16 on T4 | 640 / 1024 |

All four are ungated on Hugging Face (checked 23 Nov 2026 via the HF API).
Qwen3-4B is the hybrid `Qwen/Qwen3-4B` with `enable_thinking=true`, not
`Qwen3-4B-Thinking-2507`: same family, chat template and thinking switch as
Qwen3-8B, so 4B versus 8B is a clean scale contrast; the 2507 thinking model
writes much longer traces, which raises truncation at these caps and the cost.

## Files

- `harness/vllm_client.py`: OpenAI-compatible backend (`AGENTS_BACKEND=vllm`),
  per-request seed and temperature, token counts from `usage`, thinking split.
- `scripts/cloud/run_cloud_campaign.py`: concurrent, resumable driver (192 to 256
  units in flight, one thread per unit, calls inside a unit in order).
- `scripts/cloud/container_job.py`: starts vLLM 0.9.2 in the container, waits,
  optional throughput probe, runs the driver, stops vLLM. Platform agnostic.
- `scripts/cloud/budget.py`: cost model and budget plan. `modal_app.py`: Modal
  functions and the guarded launcher. `modal_tools.py`: plan, status, pull, stop
  (starts nothing). `analyze_cloud.py`: rev3 analysis on a cloud directory.
- `scripts/cloud/kaggle_nonthinking.ipynb` (built by `make_kaggle_notebook.py`,
  code and data embedded): Phi-4-mini on GPU 0 and Qwen2.5-3B on GPU 1.

## Thinking models: what is logged and what is passed on

The server runs without a reasoning parser; the client splits each completion at
the first `</think>`. Per call the record keeps `text` (visible part, what the
extractor reads), `reasoning` (full think block), `think_closed` (false when the
cap was hit inside the think block: no visible answer, scored as no answer),
`finish_reason`, and `out_tokens` (server `completion_tokens`, thinking included).
**Debate peers, the GE critic and reviser, and the HIER executors and aggregator
receive only the visible text, never the think block** (the workflows pass
`text`). Truncation: `hit_cap` per call, `hit_cap_rate` in `progress.json`,
`hit_cap_pct` and `think_unclosed_pct` in `analyze_cloud.py`.

Decoding: temperatures as in the paper (SA and the HIER aggregator 0, all other
calls 0.7), seeds by the harness rule, sent per request. Other sampling
parameters are each model's own `generation_config.json` defaults (vLLM
`--generation-config auto`, e.g. Qwen3 top_p 0.95, top_k 20), not Ollama's
defaults; the vLLM log in each run directory records them. vLLM batching is not
bitwise deterministic even with fixed seeds. Greedy SA with thinking on is what
the paper's protocol implies; Qwen advises against greedy thinking (risk of
loops), which shows up as SA truncation.

## Cost model (Modal, prices read 23 Nov 2026)

H100 $0.001097/s, CPU $0.0000131/physical core/s, memory $0.00000222/GiB/s,
billed on max(request, usage). Each function requests cpu (2, limit 4) and
memory (16 GiB, limit 40 GiB), so a container costs at most
**$4.457 per hour** (GPU $3.949 + CPU $0.189 + memory $0.320). No region or
non-preemptible option is set (those carry 1.15 to 1.75x and 3x). Results only on
the Volume (free up to 1 TiB); weights downloaded from Hugging Face per container.

Smoke run, 24 Nov 2026 (Qwen3-4B thinking, one H100, vLLM 0.9.2 bf16): vLLM ready
205 s after container start (weights from HF, load, compile, CUDA graphs); the
3-item GSM8K run (75 calls, all workflows) completed with 0 failures, 0 calls at
the cap, every think block closed, 423 output tokens per call; the 256-request
probe gave 9,257 tok/s in its fullest 10 s window and 888 tokens per first call
(2.7% at the 4096 cap). Container time 301 s; ledger cost at the maximum rate
$0.52 (plus $0.17 for a first attempt that failed at import in 16 s, fixed by
pinning transformers 4.53.2).

Estimates after the smoke run (`budget.py`: thinking 900 output tokens per call
on GSM8K, 2,000 on MATH (MATH not measured yet); 8,000 tok/s for Qwen3-4B, 5,200
for Qwen3-8B (scaled), 12k to 14k for the non-thinking models; 7 min startup):

| model | output tokens per seed | H100 h per seed | $ per seed | $ for 3 seeds incl. startup |
|-|-|-|-|-|
| Qwen3-4B thinking | 22.2 M | 0.77 | 3.44 | 13.3 |
| Qwen3-8B thinking | 22.2 M | 1.19 | 5.29 | 19.8 |
| Phi-4-mini | 5.2 M | 0.12 | 0.54 | 3.2 |
| Qwen2.5-3B | 5.2 M | 0.10 | 0.46 | 2.9 |

Plan under the $25 cap ($1.50 reserve): Qwen3-4B for 3 seeds (hard time 2.98 h,
at most $13.44) and Qwen3-8B in parallel with the rest (hard time 2.22 h, at most
$10.06, about 1.4 seeds at these estimates: seed 1 complete, seed 2 partial).
The non-thinking models run on Modal only if their full 3 seeds fit, so they go
to Kaggle. If the MATH thinking traces are longer than assumed, Qwen3-4B stops
at its hard time with the completed seeds first; rerunning the launcher resumes
with whatever budget is left. Print the plan any time:
`python3 scripts/cloud/budget.py` or `$MPY scripts/cloud/modal_tools.py plan`.

## Spend guards

1. Static: every function has a hard timeout from the plan; the launcher refuses
   to launch unless spend this month + every launched container's full timeout at
   the maximum rate is at most $25. `max_containers=1`, `retries=0`,
   `scaledown_window=2`, no `min_containers`, no web endpoints, no schedules.
2. In container: the driver stops starting units 10 min before the hard stop and
   exits at it; a heartbeat writes elapsed time and cost to `ledger/` on the
   Volume every minute.
3. Local, every minute: the launcher projects spend over all containers and, at
   $24.75, cancels every call and runs `modal app stop`. Spend this month is
   max(Modal billing report, ledger); the billing report (`modal.billing`) lags
   by up to an hour, so the ledger covers the current run.
4. Closing the launcher (Ctrl-C, or the laptop disconnecting) stops the app,
   since it is run with `modal run` and not `--detach`.

## Commands (repo root, Modal already authenticated)

The workspace uses image builder 2026.10, which modal 1.2.6 rejects; use the
venv with modal 1.6.4 (`/opt/homebrew/bin/python3.12 -m venv ~/.venvs/coa-modal &&
~/.venvs/coa-modal/bin/pip install modal==1.6.4 numpy`), written `MPY` below.

```
MPY=~/.venvs/coa-modal/bin/python
python3 tests/test_cloud.py                                    # offline tests
caffeinate -i $MPY -m modal run scripts/cloud/modal_app.py --mode check   # CPU only, image import check
$MPY scripts/cloud/modal_tools.py plan                      # spend + plan, no GPU
caffeinate -i $MPY -m modal run scripts/cloud/modal_app.py --mode smoke
#   Qwen3-4B thinking, 3 GSM8K items, seed 1, all workflows, plus a 256-request
#   throughput probe; hard timeout 15 min (worst case $1.26)
caffeinate -i $MPY -m modal run scripts/cloud/modal_app.py --mode full
#   or --models qwen3-4b,qwen3-8b ; rerun the same command later to resume with
#   whatever budget is left (finished units are skipped)
$MPY scripts/cloud/modal_tools.py status                    # ledger + progress
$MPY scripts/cloud/modal_tools.py pull                      # -> results/cloud/<model>/
$MPY scripts/cloud/modal_tools.py stop                      # kill switch
/usr/bin/python3 scripts/cloud/analyze_cloud.py results/cloud/qwen3-4b
```

Kaggle: upload `scripts/cloud/kaggle_nonthinking.ipynb`, set GPU T4 x2 and
Internet on, Run All, download `results_nonthinking.zip` from the Output tab and
unzip into `results/cloud/`. Expected 3 to 5 h of the 12 h session (about
1 to 1.5k output tok/s per T4). Untested on Kaggle hardware: vLLM 0.9.2 runs the
T4 (compute 7.5) on its V0 engine (`VLLM_USE_V1=0`).

Local test without spending (Ollama's OpenAI endpoint):
```
python3 -u scripts/cloud/run_cloud_campaign.py --model qwen3:0.6b --thinking on \
  --base-url http://localhost:11434/v1 --out-dir /tmp/t --n gsm8k:3 --tasks gsm8k --seeds 1
```

## Cross-model analysis for the paper

`scripts/cloud/cross_model_analysis.py` runs the paper's statistics on every model directory
present (accuracy with item-bootstrap intervals, hierarchical paired tests, R/C/N verdicts under
total and output tokens, cost-fixed comparators, token multiples, break-even price ratio of SC k=8
against debate, truncation) and writes `results/extra/cross_model_analysis.json`,
`paper/eaai_rev3/cross_model_macros.tex` and `paper/eaai_rev3/tables/cross_model_*.tex`. A seed
of an item is used only when all four workflow records exist (the Qwen3-8B run stopped mid-seed);
a seed-1-only sensitivity is added for such runs. When the Kaggle results land in
`results/cloud/phi4-mini` and `results/cloud/qwen25-3b`, rerun it and `sh paper/eaai_rev3/build.sh`;
unchanged (model, task) inputs are reused from the JSON. The raw Qwen3 logs are committed as
`results/cloud/qwen3_raw_logs.tar.xz` (`tar -xJf results/cloud/qwen3_raw_logs.tar.xz -C results/cloud`).
