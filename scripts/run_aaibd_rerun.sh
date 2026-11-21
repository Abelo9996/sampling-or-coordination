#!/bin/zsh
# AAIBD revision reruns: Job 1 (MATH SC_k2/SC_k4 at the 1024 cap, seeds 1,2),
# then Job 2 (GSM8K seed 2, six stochastic configs) for Llama-3.2-3B, then
# Gemma-2-2B. Sequential, one model loaded at a time, resumable (the driver
# skips (item_id, config, seed) records already present in each output file).
#
# Launch:
#   cd <repository root>
#   nohup zsh scripts/run_aaibd_rerun.sh >/dev/null 2>&1 &
# (the script re-execs itself under caffeinate -i; all output goes to $LOG)
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1
mkdir -p results/logs
LOG=results/logs/aaibd_rerun.log

if [[ -z "${AAIBD_CAFFEINATED:-}" ]]; then
  export AAIBD_CAFFEINATED=1
  exec caffeinate -i zsh "$0" "$@"
fi
exec >> "$LOG" 2>&1

unset AGENTS_BACKEND OLLAMA_URL AGENTS_NUM_PREDICT
PY=/usr/bin/python3
DRV=scripts/run_aaibd_rerun.py
STOCH=SC_k2,SC_k4,SC_k8,DEBATE_N3_R1,GE_R2,HIER_M3

echo "=== AAIBD_RERUN_START $(date -u +%FT%TZ) pid=$$ commit=$(git rev-parse --short HEAD) $(ollama --version 2>&1 | tail -1) ==="

# Ollama server: start if not answering (never restart one that is running).
if ! curl -sf localhost:11434/api/version >/dev/null; then
  echo "starting ollama serve"
  nohup /opt/homebrew/bin/ollama serve >> results/logs/aaibd_ollama.log 2>&1 &
  for i in {1..30}; do curl -sf localhost:11434/api/version >/dev/null && break; sleep 2; done
fi
curl -sf localhost:11434/api/version || { echo "AAIBD_RERUN_FAIL ollama not reachable"; exit 1; }
echo

stop_all() { for m in qwen2.5:3b-instruct llama3.2:3b gemma2:2b; do ollama stop "$m" 2>/dev/null; done; sleep 3; }

run_job() {  # name, then driver args
  local name=$1; shift
  stop_all
  echo "--- $name start $(date -u +%FT%TZ)"
  $PY -u $DRV run "$@"
  local rc=$?
  echo "--- $name exit=$rc $(date -u +%FT%TZ)"
  return $rc
}

FAIL=0
# Job 1: MATH, 1024 cap (as math500_trunc.jsonl), qwen2.5:3b-instruct, 60 items x {SC_k2,SC_k4} x seeds {1,2} = 240 records
run_job JOB1_MATH_SC24 --task math500 --model qwen2.5:3b-instruct --cap 1024 --n 60 \
  --configs SC_k2,SC_k4 --seeds 1,2 --ref results/math500_trunc.jsonl \
  --out results/math500_trunc_sc24.jsonl || FAIL=1

# Job 2: GSM8K seed 2, 640 cap (harness default, as the seed-1 files), 50 items x 6 configs = 300 records each
run_job JOB2_GSM8K_LLAMA_SEED2 --task gsm8k --model llama3.2:3b --cap 640 --n 50 \
  --configs $STOCH --seeds 2 --ref results/gsm8k_llama3b.jsonl \
  --out results/gsm8k_llama3b_seed2.jsonl || FAIL=1

run_job JOB2_GSM8K_GEMMA_SEED2 --task gsm8k --model gemma2:2b --cap 640 --n 50 \
  --configs $STOCH --seeds 2 --ref results/gsm8k_gemma2b.jsonl \
  --out results/gsm8k_gemma2b_seed2.jsonl || FAIL=1

stop_all
for f in results/math500_trunc_sc24.jsonl results/gsm8k_llama3b_seed2.jsonl results/gsm8k_gemma2b_seed2.jsonl; do
  echo "$f $( [[ -f $f ]] && wc -l < $f || echo 0) records"
done
echo "AAIBD_RERUN_DONE $(date -u +%FT%TZ) fail=$FAIL"
