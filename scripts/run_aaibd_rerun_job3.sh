#!/bin/zsh
# AAIBD revision, Job 3: same-runtime seed-1 rerun of the six stochastic configs
# for Llama-3.2-3B and Gemma-2-2B on the 50 GSM8K items of the July probe.
# Reason: the drift check of 13 Nov showed greedy SA for these two models does
# not reproduce the July records bit for bit (Llama 37/50 pred match, 42/50 vs
# 35/50 correct; Gemma 34/50 pred match, 28/50 both), so July seed 1 and
# November seed 2 are not draws under one runtime state. This job gives a full
# two-seed set from one runtime state (with *_driftcheck.jsonl as the greedy SA).
# Runs AFTER scripts/run_aaibd_rerun.sh. Resumable. Output goes to $LOG.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1
mkdir -p results/logs
LOG=results/logs/aaibd_rerun_job3.log

if [[ -z "${AAIBD_CAFFEINATED3:-}" ]]; then
  export AAIBD_CAFFEINATED3=1
  exec caffeinate -i zsh "$0" "$@"
fi
exec >> "$LOG" 2>&1

unset AGENTS_BACKEND OLLAMA_URL AGENTS_NUM_PREDICT
PY=/usr/bin/python3
DRV=scripts/run_aaibd_rerun.py
STOCH=SC_k2,SC_k4,SC_k8,DEBATE_N3_R1,GE_R2,HIER_M3

echo "=== AAIBD_JOB3_START $(date -u +%FT%TZ) pid=$$ commit=$(git rev-parse --short HEAD) $(ollama --version 2>&1 | tail -1) ==="
if ! curl -sf localhost:11434/api/version >/dev/null; then
  echo "starting ollama serve"
  nohup /opt/homebrew/bin/ollama serve >> results/logs/aaibd_ollama.log 2>&1 &
  for i in {1..30}; do curl -sf localhost:11434/api/version >/dev/null && break; sleep 2; done
fi
curl -sf localhost:11434/api/version || { echo "AAIBD_JOB3_FAIL ollama not reachable"; exit 1; }
echo

stop_all() { for m in qwen2.5:3b-instruct llama3.2:3b gemma2:2b; do ollama stop "$m" 2>/dev/null; done; sleep 3; }

run_job() {
  local name=$1; shift
  stop_all
  echo "--- $name start $(date -u +%FT%TZ)"
  $PY -u $DRV run "$@"
  local rc=$?
  echo "--- $name exit=$rc $(date -u +%FT%TZ)"
  return $rc
}

FAIL=0
run_job JOB3_GSM8K_LLAMA_SEED1 --task gsm8k --model llama3.2:3b --cap 640 --n 50 \
  --configs $STOCH --seeds 1 --ref results/gsm8k_llama3b.jsonl \
  --out results/gsm8k_llama3b_seed1_rerun.jsonl || FAIL=1

run_job JOB3_GSM8K_GEMMA_SEED1 --task gsm8k --model gemma2:2b --cap 640 --n 50 \
  --configs $STOCH --seeds 1 --ref results/gsm8k_gemma2b.jsonl \
  --out results/gsm8k_gemma2b_seed1_rerun.jsonl || FAIL=1

stop_all
for f in results/gsm8k_llama3b_seed1_rerun.jsonl results/gsm8k_gemma2b_seed1_rerun.jsonl; do
  echo "$f $( [[ -f $f ]] && wc -l < $f || echo 0) records"
done
echo "AAIBD_JOB3_DONE $(date -u +%FT%TZ) fail=$FAIL"
