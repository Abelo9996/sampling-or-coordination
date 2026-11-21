#!/bin/bash
# Resumable Phase A trace driver (Mac only; local Ollama at 127.0.0.1:11434).
# Relaunch after any shutdown with the same command; every logged
# (item, workflow) is skipped and a partial last line is dropped:
#   nohup caffeinate -dimsu bash scripts/_run_traces_mac.sh > results/logs/traces_main.log 2>&1 &
# and, for the reverse-order second workers on the long tasks:
#   nohup caffeinate -dimsu bash scripts/_run_traces_mac.sh --second-workers mmlu_pro_eng math500 > results/logs/traces_second.log 2>&1 &
# Protocol: paper/eaai/CRITERION_PREDICTIVE.md. Three task streams run
# concurrently against the one local server (OLLAMA_NUM_PARALLEL=4).
cd "$(dirname "$0")/.." || exit 1
unset OLLAMA_URL AGENTS_BACKEND AGENTS_NUM_PREDICT
run_task() {
  local t=$1 extra=${2:-}
  for i in $(seq 1 500); do
    until curl -s -m 5 -o /dev/null http://127.0.0.1:11434/api/version; do sleep 20; done
    python3 scripts/run_traces.py --task "$t" $extra >> "results/logs/traces_main_$t$extra.log" 2>&1 \
      && { echo "DONE $t $extra $(date -u +%FT%TZ)"; return 0; }
    echo "RETRY $t $i $(date -u +%FT%TZ)"; sleep 15
  done
}
echo "START $(date -u +%FT%TZ)"
if [ "${1:-}" = "--second-workers" ]; then
  # reverse-order second workers (shard files *_b.jsonl) for the long tasks
  shift
  for t in "$@"; do run_task "$t" --reverse & done
else
  run_task mmlu & run_task math500 & run_task mmlu_pro_eng &
fi
wait
echo "ALL_DONE $(date -u +%FT%TZ)"
