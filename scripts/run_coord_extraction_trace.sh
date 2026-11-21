#!/bin/zsh
# Answer-extraction audit traces for Qwen2.5-3B: GSM8K (80 items, 640 cap) then
# MATH L1-3 (60 items, 1024 cap), seed 1, SC_k8 + debate + GE + HIER with raw texts.
# Resumable. Output goes to results/logs/coord_extraction_trace.log.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1
mkdir -p results/logs
LOG=results/logs/coord_extraction_trace.log
if [[ -z "${COORD_TRACE_CAFF:-}" ]]; then
  export COORD_TRACE_CAFF=1
  exec caffeinate -i zsh "$0" "$@"
fi
exec >> "$LOG" 2>&1
unset AGENTS_BACKEND OLLAMA_URL AGENTS_NUM_PREDICT
echo "=== COORD_TRACE_START $(date -u +%FT%TZ) commit=$(git rev-parse --short HEAD) $(ollama --version 2>&1 | tail -1) ==="
curl -sf localhost:11434/api/version || { echo "COORD_TRACE_FAIL ollama not reachable"; exit 1; }
echo
FAIL=0
/usr/bin/python3 -u scripts/coord_extraction_trace.py --task gsm8k --cap 640 --n 80 \
  --ref results/gsm8k_matrix.jsonl --out results/gsm8k_qwen3b_coord_trace.jsonl || FAIL=1
ollama stop qwen2.5:3b-instruct 2>/dev/null; sleep 3
/usr/bin/python3 -u scripts/coord_extraction_trace.py --task math500 --cap 1024 --n 60 \
  --ref results/math500_trunc.jsonl --out results/math500_qwen3b_coord_trace.jsonl || FAIL=1
ollama stop qwen2.5:3b-instruct 2>/dev/null
echo "COORD_TRACE_DONE $(date -u +%FT%TZ) fail=$FAIL"
exit $FAIL
