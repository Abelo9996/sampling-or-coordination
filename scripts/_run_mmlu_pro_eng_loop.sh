# Resumable driver for the MMLU-Pro engineering main run (cap 1024).
# Waits for the Ollama endpoint before each attempt; the run resumes from the log.
cd "$(dirname "$0")/.."
export OLLAMA_URL=http://localhost:21434
export AGENTS_NUM_PREDICT=1024
export AGENTS_ABORT_LOG=results/mmlu_pro_eng_aborts.jsonl
echo "MAIN_START $(date -u +%FT%TZ) cap=$AGENTS_NUM_PREDICT"
for i in $(seq 1 2000); do
  until curl -s -m 5 -o /dev/null "$OLLAMA_URL/api/version"; do sleep 30; done
  .venv/bin/python scripts/run_mmlu_pro_eng.py --task mmlu_pro_eng && { echo "MAIN_DONE $(date -u +%FT%TZ)"; exit 0; }
  echo "MAIN_RETRY $i $(date -u +%FT%TZ)"; sleep 20
done
echo "MAIN_FAILED $(date -u +%FT%TZ)"
