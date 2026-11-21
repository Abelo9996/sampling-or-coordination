# Benchmark data: sources and licenses

These files are unmodified subsets or copies of public benchmarks, redistributed under
their licenses, which permit redistribution with the copyright and license notice.
Cite the original papers when you use them.

| File | Benchmark | Source | License |
|---|---|---|---|
| `gsm8k_test.jsonl` | GSM8K test split (Cobbe et al., 2021) | github.com/openai/grade-school-math | MIT |
| `math500.jsonl` | MATH-500 subset (Lightman et al., 2023) of MATH (Hendrycks et al., 2021) | github.com/openai/prm800k; github.com/hendrycks/math | MIT (both) |
| `mmlu.jsonl` | MMLU (Hendrycks et al., 2021) | github.com/hendrycks/test | MIT |
| `mmlu_pro_engineering.jsonl` | MMLU-Pro engineering category, 150 items (Wang et al., 2024); sampling recorded in `mmlu_pro_engineering.meta.json` | huggingface.co/datasets/TIGER-Lab/MMLU-Pro | Apache-2.0 |
| `HumanEval.jsonl` | HumanEval (Chen et al., 2021) | github.com/openai/human-eval | MIT |

Note on MATH: the MATH problems were collected by its authors from mathematics
competitions (AMC, AIME and others) and are distributed by them under MIT. If you
need to avoid redistributing problem text, delete `math500.jsonl` and rebuild it
from the PRM800K repository; the item ids in the logs (`math-<index>`) refer to
the line order of the MATH-500 test file.
