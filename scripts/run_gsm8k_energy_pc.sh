#!/bin/bash
# GSM8K x 5 workflows with per-call GPU energy on the RTX 4060 PC (WSL Ubuntu).
# Same measurement as scripts/run_code_energy_pc.sh; driver scripts/run_gsm8k_energy.py.
#
# Deploy from the Mac (the PC has no GitHub credentials; code goes by git archive):
#   git archive HEAD compute-optimal-agents/{harness,configs,scripts,tasks_data/gsm8k_test.jsonl} \
#     | gzip | base64 > /tmp/coa.b64   ... unpacked to ~/coa-energy on the PC (see the commit log)
# Launch on the PC so it survives disconnects:
#   systemd-run --user --unit=coa-gsm8k-energy --working-directory=$HOME/coa-energy \
#     bash scripts/run_gsm8k_energy_pc.sh
# Monitor: tail -f ~/coa-energy/results/logs/gsm8k_energy_pc.log
# Stop:    systemctl --user stop coa-gsm8k-energy
#
# The script first WAITS until the GPU is free: no GPU compute process other than
# Ollama and GPU utilization under 10% on three checks 5 min apart (so a job the
# owner chains after the current one is not raced). It never stops another
# experiment. A Riot client left open is closed only when no game is running.
# Gives up after WAIT_MAX_H hours (default 36). Resumable: rerun after any stop.
set -u
cd "$(dirname "$0")/.." || exit 1
mkdir -p results/logs results/energy
LOG=results/logs/gsm8k_energy_pc.log
N=${N:-130}
WAIT_MAX_H=${WAIT_MAX_H:-36}
export PATH="$PATH:/usr/lib/wsl/lib"   # nvidia-smi under WSL (not on a systemd unit's PATH)
command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found" >> "$LOG"; exit 2; }
export OLLAMA_URL=http://127.0.0.1:11434
export AGENTS_HOST_LABEL=rtx4060-pc
unset AGENTS_BACKEND AGENTS_NUM_PREDICT
exec >> "$LOG" 2>&1
say() { echo "[$(date -u +%FT%TZ)] $*"; }
fail() { say "PREFLIGHT FAIL: $*"; exit 2; }

others() {  # non-Ollama GPU compute processes
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -q ollama || echo -n "$p "
  done
}
util() { nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1 | tr -d ' '; }
TASKLIST=/mnt/c/Windows/System32/tasklist.exe
TASKKILL=/mnt/c/Windows/System32/taskkill.exe
games() { "$TASKLIST" /FO CSV /NH 2>/dev/null | grep -iE '"(League of Legends|VALORANT-Win64-Shipping|VALORANT)\.exe"' | cut -d, -f1; }
riot() { "$TASKLIST" /FO CSV /NH 2>/dev/null | grep -iE '"(Riot Client|RiotClientServices|RiotClientCrashHandler|LeagueClient|LeagueClientUx|LeagueClientUxRender)\.exe"' | cut -d, -f1 | sort -u; }

say "waiting for a free GPU (N=$N, give up after $WAIT_MAX_H h)"
deadline=$(( $(date +%s) + WAIT_MAX_H * 3600 ))
ok=0
while [ $ok -lt 3 ]; do
  [ "$(date +%s)" -lt $deadline ] || fail "GPU not free within $WAIT_MAX_H h"
  o=$(others); u=$(util); g=$(games)
  if [ -z "$o" ] && [ -z "$g" ] && [ "${u:-100}" -lt 10 ]; then ok=$((ok + 1)); else ok=0; fi
  say "check: other_pids='${o}' util=${u}% games='${g}' free_checks=$ok/3"
  [ $ok -ge 3 ] || sleep 300
done
r=$(riot)
if [ -n "$r" ] && [ -z "$(games)" ]; then
  say "closing idle Riot client: $(echo $r | tr '\n' ' ')"
  for x in $r; do "$TASKKILL" /F /IM "${x//\"/}" >/dev/null 2>&1; done
  sleep 10
fi

# preflight: Ollama, model digest, power sampler
curl -s -m 5 "$OLLAMA_URL/api/version" || fail "Ollama not reachable at $OLLAMA_URL"; echo
dig=$(curl -s "$OLLAMA_URL/api/tags" | python3 -c "import json,sys; print([m['digest'][:12] for m in json.load(sys.stdin)['models'] if m['name']=='qwen2.5:3b-instruct'][0])" 2>/dev/null)
[ "$dig" = "357c53fb659c" ] || fail "qwen2.5:3b-instruct digest is '$dig', expected 357c53fb659c"
python3 - <<'PY' || fail "power sampler"
import sys, time
sys.path.insert(0, ".")
from harness.energy import make_sampler
s = make_sampler().start(); time.sleep(3)
t1 = time.time() - 0.6
print(type(s).__name__, "samples", len(s.ws), "mean W last 2 s", round(s.mean_watts(t1 - 2, t1) or -1, 1)); s.stop()
sys.exit(0 if len(s.ws) > 10 else 1)
PY
nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv,noheader

say "PREFLIGHT OK; launching run_gsm8k_energy.py (n=$N, seed 1)"
for i in $(seq 1 50); do
  if python3 -u scripts/run_gsm8k_energy.py --n "$N"; then say "GSM8K_ENERGY_RUN_DONE"; exit 0; fi
  say "RETRY $i"; sleep 30
done
fail "50 retries"
