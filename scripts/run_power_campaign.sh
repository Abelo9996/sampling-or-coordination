#!/bin/zsh
# Larger-sample, 3-seed rerun on the RTX 4060 PC (generation on the PC through an
# SSH tunnel, driver and logs on the Mac). Phases in run order, each resumable:
#   G: GSM8K first 100 items, SC k=8 and debate only, cap 2048 (truncation check:
#      at cap 640 about 4% of debate final-round calls hit the cap, 0.6% for SC)
#   H: MATH-500 levels 4-5, all 262 items, cap 2048, SC k=12, model alias
#      qwen2.5:3b-instruct-ctx8k (same weights, num_ctx 8192 so that debate and
#      GE prompts carrying two 2048-token solutions fit in the context)
#   B: MATH L1-3, all 238 items (1024 cap, SC k=12)
#   A: GSM8K first 500 items (first 100 done 16 Nov)   C: GSM8K all 1,319 items
# Log: results/logs/power_campaign.log. Stop: touch results/power/STOP
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1
mkdir -p results/logs results/power
LOG=results/logs/power_campaign.log
if [[ -z "${POWER_CAFF:-}" ]]; then
  export POWER_CAFF=1
  exec caffeinate -i zsh "$0" "$@"
fi
exec >> "$LOG" 2>&1
unset AGENTS_BACKEND AGENTS_NUM_PREDICT
# From 16 Nov 10:40Z: a user-level second Ollama server on the PC (WSL, port
# 11435, OLLAMA_NUM_PARALLEL=6, same weights and settings otherwise; measured
# 333 to 366 tok/s at 6 streams vs 98 tok/s on the one-slot service). Two shard
# processes per seed. Records made there carry host rtx4060-pc-par6.
RPORT=11435
LPORT=21435
SHARDS=2
export OLLAMA_URL=http://localhost:$LPORT
export AGENTS_HOST_LABEL=rtx4060-pc-par6
PY=/usr/bin/python3

# GPU server reached over SSH (set these for your machine): ssh alias of the GPU host and
# the command that starts the parallel Ollama server there (see README, Hardware).
GPU_SSH=${GPU_SSH:-gpu-host}
GPU_START_CMD=${GPU_START_CMD:-"bash ~/ollama_par/start.sh 6"}

tunnel() {
  until curl -s -m 8 -o /dev/null "$OLLAMA_URL/api/version"; do
    [[ -f results/power/STOP ]] && { echo "STOP file found"; exit 0; }
    pkill -f "ssh -f -N .*-L $LPORT:127.0.0.1:$RPORT" 2>/dev/null
    # start the user-level parallel server if it is not listening (idempotent)
    ssh -o ConnectTimeout=15 "$GPU_SSH" "$GPU_START_CMD" 2>/dev/null
    ssh -f -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ConnectTimeout=15 \
      -L $LPORT:127.0.0.1:$RPORT "$GPU_SSH" 2>/dev/null
    sleep 20
  done
}

phase() {  # name task cap n prefix [extra driver args]
  local name=$1 task=$2 cap=$3 n=$4 prefix=$5
  shift 5
  echo "--- PHASE $name start $(date -u +%FT%TZ) task=$task cap=$cap n=$n"
  for attempt in {1..500}; do
    [[ -f results/power/STOP ]] && { echo "STOP file found"; exit 0; }
    tunnel
    local pids=()
    for s in 1 2 3; do
      for sh in {0..$((SHARDS - 1))}; do
        $PY -u scripts/run_power_campaign.py --task $task --cap $cap --n $n --seed $s \
          --shard $sh/$SHARDS --out-prefix $prefix "$@" & pids+=($!)
      done
    done
    local bad=0
    for p in $pids; do wait $p || bad=1; done
    [[ $bad -eq 0 ]] && { echo "--- PHASE $name done $(date -u +%FT%TZ)"; return 0; }
    echo "--- PHASE $name retry $attempt $(date -u +%FT%TZ)"; sleep 30
  done
  return 1
}

echo "=== POWER_CAMPAIGN_START $(date -u +%FT%TZ) commit=$(git rev-parse --short HEAD) ==="
tunnel
curl -s -m 8 "$OLLAMA_URL/api/version"; echo
phase G gsm8k 2048 100 results/power/gsm8k_cap2048_qwen3b \
  --only SC,DEBATE --sa-seed 0 --model qwen2.5:3b-instruct-ctx8k || exit 1
phase H math500 2048 262 results/power/math500hard_qwen3b \
  --math-levels 4-5 --sc-k 12 --model qwen2.5:3b-instruct-ctx8k || exit 1
phase B math500 1024 238 results/power/math500_qwen3b --sc-k 12 || exit 1
phase A gsm8k 640 500 results/power/gsm8k_qwen3b || exit 1
phase C gsm8k 640 1319 results/power/gsm8k_qwen3b || exit 1
echo "POWER_CAMPAIGN_DONE $(date -u +%FT%TZ)"
