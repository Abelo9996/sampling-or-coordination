#!/usr/bin/env bash
# Tier-C GPU campaign (round-2 reviewer fixes). Stages are independent and
# checkpointed; a killed run resumes by re-running this script.
cd "$(dirname "$0")/.."
echo "== STAGE 1: fixed topologies + debate R=2 on GSM8K (640 cap) =="
AGENTS_NUM_PREDICT=640 python3 scripts/run_matrix.py --task gsm8k_tierc --set spec
echo "== STAGE 2: MATH 1024-cap second seed =="
AGENTS_NUM_PREDICT=1024 python3 scripts/run_matrix.py --task math500_trunc2 --set spec
echo "== STAGE 3: fresh-seed router probes (seed 7) =="
AGENTS_NUM_PREDICT=640 python3 scripts/run_probe.py --task gsm8k   --n 80 --seed 7 --suffix _fresh
AGENTS_NUM_PREDICT=640 python3 scripts/run_probe.py --task mmlu    --n 80 --seed 7 --suffix _fresh
AGENTS_NUM_PREDICT=1024 python3 scripts/run_probe.py --task math500 --n 60 --seed 7 --suffix _fresh
echo "== STAGE 4: 7B scale probe on GSM8K =="
ollama pull qwen2.5:7b-instruct
AGENTS_NUM_PREDICT=640 python3 scripts/run_matrix.py --task gsm8k_7b --set spec
echo "== TIERC_ALL_DONE =="
