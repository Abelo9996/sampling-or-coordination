# Du et al. (2023) multiagent debate: upstream code (fetched, not vendored)

Paper: Yilun Du, Shuang Li, Antonio Torralba, Joshua B. Tenenbaum, Igor Mordatch,
"Improving Factuality and Reasoning in Language Models through Multiagent Debate",
arXiv:2305.14325 (2023).

- Source: https://github.com/composable-models/llm_multiagent_debate
- Commit: `9846749350eb917ae5bfaaff4c645fc705b8d3af` (HEAD of `main`, committed 24 Apr 2025)
- Retrieved: 16 Nov 2026 (`git clone --depth 1`)
- License: the upstream repository has no LICENSE file and GitHub reports no license
  (`license: null`), so default copyright applies. **The upstream files are not included
  in this release.** Run `sh third_party/du2023_debate/fetch_upstream.sh` to clone them at
  the commit above into `upstream/`; the script checks the SHA-256 values listed below.
  Only `scripts/extra_workflows.py` (the Du et al. debate run) needs them; no analysis
  script does.

## Files (unmodified copies under `upstream/`, SHA-256)

```
7fcae591f15a0d14545d1fcbcfe41c3c090a92305a01e13064f443b4d1b04733  README.md
7f37c317e9a0fe5636423b099f11355eca899c69a657981610a28fcc5b3c673b  requirements.txt
3a4adafb7a2563e70eb86855aad4d99507f12f8382f39dda2af29baa4bf1900a  gsm/gen_gsm.py
dd500b00b7be92a7cc584a2190349704dfe5416abc3079ddf2aae56ba38ff1e6  gsm/eval_gsm.py
cbaa63faa2cb7b0bcb7afb954fd3cb1c31fc37c6a3d927c34c51a48c13a2836e  math/gen_math.py
23ffcf0b0f78da2bb99744946844f93477371fc5eaaa4ce1969fa6e42b9cf3c2  mmlu/gen_mmlu.py
0987d705729c591467f06eacbf32db9038ed0c7c402348181641e14994b73018  mmlu/eval_mmlu.py
```

The `biography/` folder of the upstream repository was not copied (not used).
`math/` is their arithmetic task (a+b*c+d-e*f), not the MATH competition dataset.

## How the reimplementation uses them (`scripts/extra_workflows.py`)

Nothing is retyped except one inline string. At import time the module compiles,
from the files above, `construct_message` and `construct_assistant_message`
(`gsm/gen_gsm.py` lines 6 to 24) and `solve_math_problems`, `parse_answer`,
`most_frequent` (`gsm/eval_gsm.py` lines 46 to 66 and 109 to 119). The initial
question prompt is inline in their `__main__` (`gsm/gen_gsm.py` line 45); it is
copied and asserted equal to the string constant in that file.

Kept as in their GSM script: 3 agents, `rounds = 2` (round 0 is the independent
answer, so one debate round and 6 calls); each agent's own chat history (user and
assistant turns) is the message list of its next call; in round r each agent gets
`construct_message` over the other agents' contexts at index `2*r - 1` (their
previous-round replies); agents processed in order. Final answer: for each agent's
last message, `parse_answer` (last `{...}` holding only digits, `.`, `,` or `$`,
non-digits removed) with fallback `solve_math_problems` (last number in the text),
then `most_frequent` over the three (first-seen value wins ties; `None` entries are
kept in the vote), as in `eval_gsm.py` `compute_accuracy` lines 75 to 87.

## Departures from their script

| Item | Theirs | Here | Why |
|---|---|---|---|
| Model and API | `openai.ChatCompletion`, `gpt-3.5-turbo-0301`, `n=1` | Ollama `/api/chat`, Qwen2.5-3B | the paper's model |
| Temperature | not set: OpenAI default 1.0 (top_p 1) | `--du-temperature`, default 0.7 | harness debate value, so that only the protocol differs from the harness debate |
| Max tokens | not set: OpenAI default, up to the context (4,096 for that model) | `--du-max-tokens`, default `--cap` (640 GSM8K, 2,048 MATH) | every harness call uses the cap; same cost meter |
| Seeds | none | round 0 agent a: `seed*1000+a`; round r: `seed*1000+100*r+a` | harness debate rule, replicates labelled alike |
| Items | `random.seed(0)` shuffle of GSM8K test, first 100 | harness order, first 100 (same ids as the existing records) | paired comparison |
| Scoring | `float(gt) == float(pred)`; a `None` vote counts as correct (`eval_gsm.py` lines 95 to 96) | their parsed and voted answer scored by the harness scorer; `None` is wrong | the `None` rule is a bug that would inflate accuracy |
| MATH-500 | no script | ADAPTATION, below | |

Also logged: the harness extractor applied to each final-round message with the
harness majority rule (`pred_harness`, `correct_harness`).

### MATH-500 adaptation (not in Du et al.)
- Prompts: the GSM prompts with the words `a single numerical number, ` deleted
  (one occurrence in the initial prompt, two in `construct_message`), applied as a
  string replacement on their source, nothing else changed.
- Parse: their `parse_answer` keeps only digits inside braces, so `\frac{1}{2}`
  would read as `2`. The last `\boxed{...}` (balanced braces, harness `_boxed`) is
  read instead, with their fallback (last number) kept. Vote: their `most_frequent`
  on the raw strings.
- Context: at cap 2,048 a round-1 prompt holds the agent's own reply and two other
  replies; it can approach 8,192 tokens, so the run uses the 8k-context alias and
  calls with `in_tokens + out_tokens >= 8192` are reported as possibly truncated.

## Upstream quirks, not reproduced or not reached
- `eval_gsm.py` reads `gsm_debate_3_3.json` while `gen_gsm.py` writes `gsm_3_2.json`.
- `compute_accuracy` iterates a global `pred_solutions`; here the per-agent
  parse and `most_frequent` are called directly.
- The zero-agent branch of `construct_message` (unused with 3 agents) is not
  passed through `.format`, so it would show `\boxed{{answer}}` with double braces.
