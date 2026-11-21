#!/bin/zsh
cd "$(dirname "$0")/.."
while kill -0 66556 2>/dev/null; do sleep 60; done
echo "[chain] cross-family pipeline exited, resuming mechanistic $(date)" >> results/overnight_xfamily.log
ollama stop llama3.2:3b 2>/dev/null; ollama stop gemma2:2b 2>/dev/null; sleep 3
caffeinate -i .venv/bin/python harness/mechanistic.py >> results/mechanistic_run.log 2>&1
echo "[chain] mechanistic COMPLETE $(date)" >> results/overnight_xfamily.log
