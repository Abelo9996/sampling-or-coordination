#!/bin/bash
# HumanEval x 7 configurations with per-call GPU energy, on the RTX 4060 PC.
# Design and reporting rule: paper/eaai/CODE_ENERGY_EXPERIMENT.md.
#
# DO NOT RUN while the clinical experiment occupies the PC (until about
# 2026-10-14). The preflight below refuses to start before 2026-10-14, or if
# any non-Ollama process holds the GPU, unless FORCE_AFTER_CLINICAL=1 is set
# by a person who has confirmed the clinical run has finished.
#
# Launch from the Mac (one line; the harness runs ON the PC, next to
# nvidia-smi, Docker, and the PC's own Ollama at 127.0.0.1:11434):
#   ssh <gpu-host> 'cd <repository root on the GPU host> && bash scripts/run_code_energy_pc.sh'
# Resumable: rerun the same command after any interruption.
set -u
cd "$(dirname "$0")/.." || exit 1
mkdir -p results/logs
LOG=results/logs/code_energy_pc.log
OUT=results/humaneval_matrix_pc.jsonl
export OLLAMA_URL=http://127.0.0.1:11434
export AGENTS_SANDBOX=docker
export AGENTS_HOST_LABEL=rtx4060-pc
unset AGENTS_BACKEND AGENTS_NUM_PREDICT

fail() { echo "PREFLIGHT FAIL: $*" >&2; exit 2; }

# 1. date gate (clinical experiment)
if [ "${FORCE_AFTER_CLINICAL:-0}" != "1" ] && [ "$(date -u +%Y%m%d)" -lt 20261014 ]; then
  fail "before 2026-10-14; the clinical experiment may still be running"
fi
# 2. GPU present and not used by anything but Ollama
command -v nvidia-smi >/dev/null || fail "nvidia-smi not found"
others=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | grep -vi ollama | grep -v '^\s*$' || true)
[ -z "$others" ] || [ "${FORCE_AFTER_CLINICAL:-0}" = "1" ] || fail "GPU in use by: $others"
nvidia-smi --query-gpu=name,power.draw,utilization.gpu --format=csv,noheader
# 3. Docker sandbox image, and the sandbox really has no network
command -v docker >/dev/null || fail "docker not found"
docker image inspect python:3.11-slim >/dev/null 2>&1 || docker pull -q python:3.11-slim || fail "cannot pull python:3.11-slim"
docker run --rm --network none python:3.11-slim python -c "import socket,sys
try: socket.create_connection(('1.1.1.1',80),timeout=3); sys.exit(1)
except OSError: sys.exit(0)" || fail "sandbox network is not isolated"
# 4. Ollama and model digest
curl -s -m 5 "$OLLAMA_URL/api/version" || fail "Ollama not reachable at $OLLAMA_URL"; echo
dig=$(curl -s "$OLLAMA_URL/api/tags" | python3 -c "import json,sys; print([m['digest'][:12] for m in json.load(sys.stdin)['models'] if m['name']=='qwen2.5:3b-instruct'][0])" 2>/dev/null)
[ "$dig" = "357c53fb659c" ] || fail "qwen2.5:3b-instruct digest is '$dig', expected 357c53fb659c"
# 5. canonical solutions pass in the Docker sandbox (harness self-test)
python3 - <<'PY' || fail "canonical-solution self-test failed"
import json, sys
sys.path.insert(0, ".")
from harness import tasks
t = tasks.get_task("humaneval")
sols = {json.loads(l)["task_id"]: json.loads(l)["canonical_solution"]
        for l in open("tasks_data/HumanEval.jsonl")}
items = t["load"]()[:5]
ok = sum(t["score"](t["extract_item"]("```python\n" + sols[i["gold"]["task_id"]] + "```", i), i["gold"]) for i in items)
print("canonical self-test", ok, "/", len(items)); sys.exit(0 if ok == len(items) else 1)
PY

echo "PREFLIGHT OK $(date -u +%FT%TZ); launching -> $OUT (log $LOG)"
nohup bash -c "for i in \$(seq 1 200); do python3 scripts/run_code_energy.py --energy nvidia --out $OUT && { echo CODE_RUN_DONE; exit 0; }; echo RETRY \$i; sleep 30; done" >> "$LOG" 2>&1 &
echo "started pid $!"
