"""Run the MMLU-Pro engineering matrix with the repeat-limit handler installed.

Same as `scripts/run_matrix.py --task mmlu_pro_eng`, except that a call Ollama
aborts with "token repeat limit reached" is recorded instead of stopping the run
(see harness/repeat_abort.py). Arguments are passed through to run_matrix.

  AGENTS_NUM_PREDICT=1024 AGENTS_ABORT_LOG=results/mmlu_pro_eng_aborts.jsonl \
    .venv/bin/python scripts/run_mmlu_pro_eng.py --task mmlu_pro_eng
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from harness import repeat_abort  # noqa: E402

repeat_abort.install()

import run_matrix  # noqa: E402

if __name__ == "__main__":
    run_matrix.main()
