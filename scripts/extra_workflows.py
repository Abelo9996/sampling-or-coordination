"""Two extra workflows for the reviewer points, kept outside harness/ (unchanged).

1. DU_DEBATE: the multiagent debate of Du et al. (2023), "Improving Factuality
   and Reasoning in Language Models through Multiagent Debate", reproduced from
   the authors' released GSM8K script, vendored unmodified under
   third_party/du2023_debate/upstream/ (commit 9846749, see the README there).
   The prompt-building function, the answer parser and the vote are NOT retyped:
   they are compiled from the vendored source files (AST extraction of the named
   functions), and the one prompt that lives inline in their __main__ block is
   copied here and asserted equal to the string constant in their file.

2. HIERFIX: harness.topologies.hier (M=3) with one change, the answer-format
   reminder at the end of the aggregator prompt (see hierfix() below).

Model calls for DU_DEBATE go to Ollama /api/chat (their script is a chat
conversation with per-agent history); HIERFIX calls go through the unchanged
harness client (/api/generate).
"""
import ast
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPSTREAM = os.path.join(ROOT, "third_party", "du2023_debate", "upstream")
GEN_GSM = os.path.join(UPSTREAM, "gsm", "gen_gsm.py")
EVAL_GSM = os.path.join(UPSTREAM, "gsm", "eval_gsm.py")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from harness import client, topologies  # noqa: E402


# ---------------------------------------------------------------------------
# Verbatim code from the vendored Du et al. files
# ---------------------------------------------------------------------------
def _load_functions(path, names, replace=None):
    """Compile the named top-level functions of `path` exactly as written there
    (source segments taken from the file). `replace` = (old, new) is applied to
    the source text of these functions only; used solely for the MATH adaptation."""
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src)
    ns = {"re": re}
    found = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            seg = ast.get_source_segment(src, node)
            if replace is not None:
                seg = seg.replace(*replace)
            exec(compile(seg, f"{path}:{node.name}", "exec"), ns)
            found.append(node.name)
    assert sorted(found) == sorted(names), (path, found)
    return ns


def _string_constants(path):
    tree = ast.parse(open(path, encoding="utf-8").read())
    return {n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)}


# gen_gsm.py line 45, inline in their __main__ block (copied; asserted below).
DU_GSM_INITIAL = ("""Can you solve the following math problem? {} Explain your reasoning. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response. """)
assert DU_GSM_INITIAL in _string_constants(GEN_GSM), "initial prompt differs from gen_gsm.py"

# gen_gsm.py lines 6-19 (construct_message) and 22-24 (construct_assistant_message)
_GSM_GEN = _load_functions(GEN_GSM, ["construct_message", "construct_assistant_message"])
# eval_gsm.py lines 46-53 (solve_math_problems), 55-66 (parse_answer), 109-119 (most_frequent)
_GSM_EVAL = _load_functions(EVAL_GSM, ["solve_math_problems", "parse_answer", "most_frequent"])
du_construct_message_gsm = _GSM_GEN["construct_message"]
du_construct_assistant_message = _GSM_GEN["construct_assistant_message"]
du_solve_math_problems = _GSM_EVAL["solve_math_problems"]
du_parse_answer = _GSM_EVAL["parse_answer"]
du_most_frequent = _GSM_EVAL["most_frequent"]

# ADAPTATION (not in Du et al.; they release no MATH-competition script): the
# smallest change of the GSM prompts for MATH-500, namely deleting the words
# "a single numerical number, " so that the answer is any \boxed{} expression.
# Applied to the initial prompt and to the source of construct_message (where
# it occurs twice: the unused zero-agent branch and the debate message).
MATH_ADAPT = ("a single numerical number, ", "")
assert DU_GSM_INITIAL.count(MATH_ADAPT[0]) == 1
assert open(GEN_GSM, encoding="utf-8").read().count(MATH_ADAPT[0]) == 3
DU_MATH_INITIAL = DU_GSM_INITIAL.replace(*MATH_ADAPT)
du_construct_message_math = _load_functions(
    GEN_GSM, ["construct_message"], replace=MATH_ADAPT)["construct_message"]


def du_parse_gsm(text):
    """Their per-agent reading (eval_gsm.compute_accuracy, lines 79-82):
    parse_answer, falling back to the last number in the text."""
    a = du_parse_answer(text)
    if a is None:
        a = du_solve_math_problems(text)
    return a


def du_parse_math(text):
    """ADAPTATION for MATH: their parse_answer reads only digits inside braces
    (pattern \\{([0-9.,$]*)\\}), which mangles \\frac{1}{2}; the boxed content is
    read instead (last \\boxed{...}, balanced braces, harness tasks._boxed),
    with their fallback (last number in the text) kept."""
    from harness.tasks import _boxed
    b = _boxed(text)
    if b is not None and b.strip():
        return b.strip()
    return du_solve_math_problems(text)


DU_TASKS = {
    "gsm8k": {"initial": DU_GSM_INITIAL, "construct_message": du_construct_message_gsm,
              "parse": du_parse_gsm, "adapted": False},
    "math500": {"initial": DU_MATH_INITIAL, "construct_message": du_construct_message_math,
                "parse": du_parse_math, "adapted": True},
}


# ---------------------------------------------------------------------------
# Ollama /api/chat call (same options as harness.client.generate)
# ---------------------------------------------------------------------------
def _chat_url():
    u = os.environ.get("OLLAMA_URL", "").strip() or "http://localhost:11434"
    for suf in ("/api/generate", "/api/chat"):
        if u.endswith(suf):
            u = u[: -len(suf)]
    return u.rstrip("/") + "/api/chat"


def chat(model, messages, temperature, num_predict, seed=None, timeout=900, retries=2):
    """One chat call. Sends options temperature, num_predict, seed (as the harness
    generate call does) and returns text, exact token counts and latency."""
    body = {"model": model, "messages": messages, "stream": False,
            "options": {"temperature": float(temperature), "num_predict": int(num_predict)}}
    if seed is not None:
        body["options"]["seed"] = int(seed)
    data = json.dumps(body).encode("utf-8")
    last = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(_chat_url(), data=data,
                                         headers={"Content-Type": "application/json"})
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d = json.loads(r.read().decode("utf-8"))
            return client.CallRecord(
                text=(d.get("message") or {}).get("content", ""),
                in_tokens=int(d.get("prompt_eval_count", 0) or 0),
                out_tokens=int(d.get("eval_count", 0) or 0),
                latency_s=round(time.time() - t0, 4),
                model=model, temperature=float(temperature), seed=seed,
                num_predict=int(num_predict))
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Ollama chat call failed after {retries + 1} attempts: {last}")


# ---------------------------------------------------------------------------
# DU_DEBATE
# ---------------------------------------------------------------------------
def du_debate(model, task_name, task, item, params, seed):
    """Du et al. (2023) GSM debate, gen_gsm.py lines 32-61, on Ollama chat.

    agents=3, rounds=2 (their `rounds` counts the initial answer as round 0, so
    this is one debate round and 6 calls, like harness DEBATE_N3_R1). Each agent
    keeps its own chat history; in round r>=1 it receives construct_message over
    the OTHER agents' contexts at index 2*r-1, i.e. their previous-round replies.
    Agents are processed in order 0..2 as in their loop.

    Departures (documented in third_party/du2023_debate/README.md): model API
    (Ollama chat instead of gpt-3.5-turbo-0301), temperature and max tokens (they
    rely on OpenAI defaults; we pass params temperature and max_tokens), seeds
    (harness debate rule: round 0 agent a -> seed*1000+a, round r -> seed*1000+
    r*100+a), MATH prompt/parse adaptation (marked above).

    Returns the answer by their parse + their vote (most_frequent over the
    final-round agents, None entries included as in eval_gsm.py) and, separately,
    the harness extractor + harness majority over the same final messages."""
    spec = DU_TASKS[task_name]
    agents = int(params.get("N", 3))
    rounds = int(params.get("R", 2))
    temp = float(params["temperature"])
    max_tokens = int(params["max_tokens"])
    question = item["question"]
    agent_contexts = [[{"role": "user", "content": spec["initial"].format(question)}]
                      for agent in range(agents)]
    calls = []
    for round in range(rounds):
        for i, agent_context in enumerate(agent_contexts):
            if round != 0:
                agent_contexts_other = agent_contexts[:i] + agent_contexts[i + 1:]
                message = spec["construct_message"](agent_contexts_other, question, 2 * round - 1)
                agent_context.append(message)
            s = seed * 1000 + round * 100 + i  # harness debate seed rule
            r = chat(model, agent_context, temperature=temp, num_predict=max_tokens, seed=s)
            r["agent"], r["round"], r["n_messages_in"] = i, round, len(agent_context)
            calls.append(r)
            completion = {"choices": [{"message": {"content": r["text"]}}]}
            agent_context.append(du_construct_assistant_message(completion))
    finals = [ctx[-1]["content"] for ctx in agent_contexts]
    du_parsed = [spec["parse"](t) for t in finals]
    du_answer = du_most_frequent(du_parsed)
    h_final = [topologies._extract(task, item, t) for t in finals]
    return {"answer": du_answer, "du_parsed_final": du_parsed,
            "answer_harness": topologies._majority(h_final),
            "harness_extracted_final": h_final,
            "contexts": agent_contexts, "calls": calls}


# ---------------------------------------------------------------------------
# HIERFIX
# ---------------------------------------------------------------------------
# The harness HIER aggregator prompt ends with task['reminder']. For GSM8K that is
#   "End with exactly one line: 'The answer is N.'"
# and the temperature-0 aggregator often returns that line verbatim, with the
# letter N and no number. HIERFIX replaces only this reminder.
HIERFIX_OLD_REMINDER = {"gsm8k": "End with exactly one line: 'The answer is N.'"}
HIERFIX_REMINDER = {
    "gsm8k": "End with exactly one line: 'The answer is N.' with N replaced by the final number.",
}


def hierfix(model, task_name, task, item, params, seed):
    """HIERFIX_M3: harness.topologies.hier, unchanged and called as is, with a
    task copy whose `reminder` (used by hier only at the end of the aggregator
    prompt) is HIERFIX_REMINDER[task]. Planner, executors, aggregator template,
    temperatures (0.7 / 0.7 / 0.0), seeds (seed*1000, +100+i, +999), call count
    (1 + M + 1), extractor and scorer are the harness ones.

    The repair wording was written after seeing the failure (aggregator echoes
    the template line "The answer is N."). It was not tuned on any accuracy: it
    was fixed before any HIERFIX output was generated, and before the protocol
    runs it was executed on at most 3 items, only to check that the code runs."""
    if task_name not in HIERFIX_REMINDER:
        raise SystemExit(f"HIERFIX is defined for {sorted(HIERFIX_REMINDER)} only")
    assert task["reminder"] == HIERFIX_OLD_REMINDER[task_name], "harness reminder changed"
    t2 = dict(task, reminder=HIERFIX_REMINDER[task_name])
    return topologies.hier(model, t2, item, params, seed)
