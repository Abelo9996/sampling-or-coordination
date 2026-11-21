#!/bin/zsh
# rev3 rerun campaign (protocols/REV3_RERUN_PROTOCOL.md). Generation on the RTX 4060
# PC (user-level Ollama, 6 parallel slots, port 11435) through an SSH tunnel;
# driver, scoring and logs on the Mac. Same tunnel / shard / resume / STOP pattern
# as scripts/run_power_campaign.sh, own log and output prefix:
#   log     results/logs/rev3_campaign.log
#   output  results/rev3/<run>_s<seed>_sh<i>of<n>.jsonl
#   stop    touch results/rev3/STOP   (checked between attempts and while waiting)
#   lock    results/rev3/.launcher.lock (one launcher at a time)
#
# Hand-over from the power campaign (which must not be edited while it runs):
#   1. wait until the power log, after its last POWER_CAMPAIGN_START, shows
#      the phase A completion line ("PHASE A done");
#   2. touch results/power/STOP at once. The old launcher calls phase C right
#      after printing that line and starts its drivers within a second, before any
#      poll can touch the STOP file, so any phase C driver found running
#      (run_power_campaign.py ... --n 1319) is sent SIGTERM (background jobs of a
#      non-interactive shell ignore SIGINT); the old launcher then
#      logs a retry, sees STOP and exits. Phase C records that reached disk, if
#      any, are counted in this log (items beyond the first 500; never analysed);
#   3. wait until no run_power_campaign.sh process is left, then rm results/power/STOP;
#   4. run the rev3 queue in protocol order R1..R5.
#
# Test hooks (dry run, used by the offline test only): REV3_DRY=1 echoes the driver
# commands instead of running them and skips caffeinate and the tunnel;
# REV3_POWER_LOG, REV3_POWER_STOP, REV3_OLD_PAT, REV3_PHASEC_PAT, REV3_OUT,
# REV3_LOG, REV3_POLL override paths, process patterns and the poll interval.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1
OUT=${REV3_OUT:-results/rev3}
LOG=${REV3_LOG:-results/logs/rev3_campaign.log}
POWER_LOG=${REV3_POWER_LOG:-results/logs/power_campaign.log}
POWER_STOP=${REV3_POWER_STOP:-results/power/STOP}
OLD_PAT=${REV3_OLD_PAT:-run_power_campaign.sh}
PHASEC_PAT=${REV3_PHASEC_PAT:-run_power_campaign.py --task gsm8k --cap 640 --n 1319}
POLL=${REV3_POLL:-2}
DRY=${REV3_DRY:-0}
mkdir -p "$OUT" "${LOG:h}"
if [[ $DRY != 1 && -z "${REV3_CAFF:-}" ]]; then
  export REV3_CAFF=1
  exec caffeinate -i zsh "$0" "$@"
fi
exec >> "$LOG" 2>&1

# single-instance lock (mkdir is atomic)
LOCK=$OUT/.launcher.lock
if ! mkdir "$LOCK" 2>/dev/null; then
  op=$(cat "$LOCK/pid" 2>/dev/null)
  if [[ -n "$op" ]] && kill -0 "$op" 2>/dev/null; then
    echo "[$(date -u +%FT%TZ)] another rev3 launcher is running (pid $op); exiting"; exit 3
  fi
  echo "[$(date -u +%FT%TZ)] stale lock (pid ${op:-?}); taking it over"
  rm -rf "$LOCK"; mkdir "$LOCK" || { echo "lock race lost; exiting"; exit 3; }
fi
echo $$ > "$LOCK/pid"
trap 'rm -rf "$LOCK"' EXIT
trap 'exit 130' INT TERM

unset AGENTS_BACKEND AGENTS_NUM_PREDICT
RPORT=11435
LPORT=21435
export OLLAMA_URL=http://localhost:$LPORT
export AGENTS_HOST_LABEL=rtx4060-pc-par6
PY=/usr/bin/python3
ts() { date -u +%FT%TZ; }

# GPU server reached over SSH (set these for your machine): ssh alias of the GPU host and
# the command that starts the parallel Ollama server there (see README, Hardware).
GPU_SSH=${GPU_SSH:-gpu-host}
GPU_START_CMD=${GPU_START_CMD:-"bash ~/ollama_par/start.sh 6"}

tunnel() {
  [[ $DRY == 1 ]] && return 0
  until curl -s -m 8 -o /dev/null "$OLLAMA_URL/api/version"; do
    [[ -f $OUT/STOP ]] && { echo "STOP file found"; exit 0; }
    pkill -f "ssh -f -N .*-L $LPORT:127.0.0.1:$RPORT" 2>/dev/null
    ssh -o ConnectTimeout=15 "$GPU_SSH" "$GPU_START_CMD" 2>/dev/null
    ssh -f -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ConnectTimeout=15 \
      -L $LPORT:127.0.0.1:$RPORT "$GPU_SSH" 2>/dev/null
    sleep 20
  done
}

phase_a_done() {  # phase A completion line after the last POWER_CAMPAIGN_START
  [[ -f $POWER_LOG ]] || return 1
  awk '/=== POWER_CAMPAIGN_START/{s=0} /^--- PHASE A done/{s=1} END{exit !s}' "$POWER_LOG"
}

handover() {
  echo "=== REV3 waiting for power campaign phase A $(ts) (log $POWER_LOG) ==="
  local n=0
  until phase_a_done; do
    [[ -f $OUT/STOP ]] && { echo "STOP file found while waiting"; exit 0; }
    if (( n % (3600 / POLL) == 0 )) && ! pgrep -f "$OLD_PAT" >/dev/null; then
      echo "[$(ts)] WARNING: no $OLD_PAT process and phase A not done; still waiting"
    fi
    n=$((n + 1)); sleep $POLL
  done
  touch "$POWER_STOP"
  echo "[$(ts)] phase A done seen; touched $POWER_STOP"
  local killed=0
  while pgrep -f "$OLD_PAT" >/dev/null; do
    if pgrep -f "$PHASEC_PAT" >/dev/null; then
      echo "[$(ts)] phase C drivers running: $(pgrep -f "$PHASEC_PAT" | tr '\n' ' ')- SIGTERM"
      pkill -TERM -f "$PHASEC_PAT"; killed=1
    fi
    sleep 1
  done
  rm -f "$POWER_STOP"
  echo "[$(ts)] old launcher gone (phase C drivers interrupted: $killed); removed $POWER_STOP"
  if [[ $DRY != 1 ]]; then
    $PY - <<'EOF'
import glob, json, sys
sys.path.insert(0, ".")
from harness import tasks
first = {it["id"] for it in tasks.gsm8k_load(n=500)}
n = 0
for p in glob.glob("results/power/gsm8k_qwen3b_s*.jsonl"):
    for line in open(p):
        try:
            n += json.loads(line)["item_id"] not in first
        except Exception:
            pass
print(f"phase C records on disk (GSM8K items beyond the first 500, not analysed): {n}")
EOF
  fi
}

job() {  # name task cap n "seeds" model prefix configs [extra driver args]
  local name=$1 task=$2 cap=$3 n=$4 model=$6 prefix=$7 configs=$8
  local -a seeds
  seeds=(${=5})
  shift 8
  local shards=$(( 6 / ${#seeds} ))
  echo "--- JOB $name start $(ts) task=$task cap=$cap n=$n seeds=${seeds} shards=$shards model=$model configs=$configs"
  for attempt in {1..500}; do
    [[ -f $OUT/STOP ]] && { echo "STOP file found"; exit 0; }
    tunnel
    local pids=()
    for s in $seeds; do
      for sh in {0..$((shards - 1))}; do
        local cmd=($PY -u scripts/run_rev3_campaign.py --task $task --cap $cap --n $n
                   --seed $s --shard $sh/$shards --model $model --configs $configs
                   --out-prefix $prefix "$@")
        if [[ $DRY == 1 ]]; then echo "DRY ${cmd[*]}"; true & pids+=($!)
        else "${cmd[@]}" & pids+=($!); fi
      done
    done
    local bad=0
    for p in $pids; do wait $p || bad=1; done
    [[ $bad -eq 0 ]] && { echo "--- JOB $name done $(ts)"; return 0; }
    echo "--- JOB $name retry $attempt $(ts)"; sleep 30
  done
  return 1
}

echo "=== REV3_CAMPAIGN_START $(ts) commit=$(git rev-parse --short HEAD) pid=$$ dry=$DRY ==="
handover
tunnel
[[ $DRY == 1 ]] || { curl -s -m 8 "$OLLAMA_URL/api/version"; echo; }
M4=qwen2.5:3b-instruct
M8=qwen2.5:3b-instruct-ctx8k
job R1 gsm8k 640 500 "1 2 3" $M4 $OUT/gsm8k_qwen3b HIERFIX || exit 1
job R2 mmlu 640 80 "1 2" $M4 $OUT/mmlu_qwen3b HIERFIX || exit 1
job R3 humaneval 1024 164 "1 2" $M4 $OUT/humaneval_qwen3b SA,SC,DEBATE,GE,HIERFIX --sc-k 8 || exit 1
job R4 mmlu_pro_eng 1024 150 "1 2" $M4 $OUT/mmlu_pro_eng_qwen3b HIERFIX || exit 1
job R5 math500 2048 238 "1 2 3" $M8 $OUT/math500_cap2048_qwen3b SA,SC,DEBATE,GE,HIERFIX \
  --math-levels 1-3 --sc-k 12 || exit 1
echo "REV3_CAMPAIGN_DONE $(ts)"
