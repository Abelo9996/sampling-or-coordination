#!/bin/sh
# Resumable driver for scripts/run_scale70b.py: reruns passes until the run is
# complete (exit 0) or the spend cap stops it (exit 3). Infrastructure errors
# (exit 2) are retried after a pause. Usage, from compute-optimal-agents/:
#   nohup sh scripts/run_scale70b.sh >> results/gsm8k_llama70b.log 2>&1 &
cd "$(dirname "$0")/.." || exit 1
pass=0
while [ $pass -lt 40 ]; do
  pass=$((pass + 1))
  echo "=== pass $pass ($(date -u +%Y-%m-%dT%H:%M:%SZ)) ==="
  python3 scripts/run_scale70b.py
  rc=$?
  if [ $rc -eq 0 ]; then echo "=== driver: complete ==="; exit 0; fi
  if [ $rc -eq 3 ]; then echo "=== driver: stopped by spend cap ==="; exit 3; fi
  sleep 120
done
echo "=== driver: gave up after $pass passes ==="
exit 2
