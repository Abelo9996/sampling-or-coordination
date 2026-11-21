"""Task loaders + verifiable, automatic scorers.

Each task exposes:
  - load(n, split) -> list of {id, question, gold}
  - prompt(question) -> str        (instruction eliciting a parseable answer)
  - extract(text) -> parsed answer
  - score(pred, gold) -> bool      (deterministic correctness)
  - reminder: str                  (answer-format reminder for multi-step topologies)

No LLM judge is used: scoring is regex/numeric/normalized-string based, so
results are free and reproducible. All three tasks have a *votable* discrete
answer, so self-consistency (the paper's control) is well-defined for each.
"""
import json
import os
import re

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "tasks_data")


# ----------------------------------------------------------------------------
# shared
# ----------------------------------------------------------------------------
def _to_number(s):
    s = str(s).replace(",", "").rstrip(".")
    try:
        f = float(s)
        return int(f) if f.is_integer() else f
    except ValueError:
        return None


def _split_take(rows, n, split, seed):
    import random
    rng = random.Random(seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    cut = int(0.6 * len(idx))
    if split == "train":
        chosen = idx[:cut]
    elif split == "test":
        chosen = idx[cut:]
    else:
        chosen = idx
    if n is not None:
        chosen = chosen[:n]
    return [rows[i] for i in chosen]


# ----------------------------------------------------------------------------
# GSM8K
# ----------------------------------------------------------------------------
_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def gsm8k_load(n=None, split="all", seed=0):
    rows = []
    with open(os.path.join(DATA_DIR, "gsm8k_test.jsonl")) as f:
        for i, line in enumerate(f):
            d = json.loads(line)
            gold = d["answer"].split("####")[-1].strip()
            rows.append({"id": f"gsm8k-{i}", "question": d["question"],
                         "gold": _to_number(gold)})
    return _split_take(rows, n, split, seed)


def gsm8k_prompt(q):
    return ("Solve the following grade-school math problem step by step. Show "
            "brief reasoning, then on the final line write exactly: 'The answer "
            f"is N.' where N is the final number.\n\nProblem: {q}\n")


def gsm8k_extract(text):
    m = re.search(r"answer is\s*:?\s*(-?\d[\d,]*(?:\.\d+)?)", text, re.IGNORECASE)
    if m:
        return _to_number(m.group(1))
    nums = _NUM.findall(text)
    return _to_number(nums[-1]) if nums else None


def gsm8k_score(pred, gold):
    if pred is None or gold is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-4
    except (TypeError, ValueError):
        return False


# ----------------------------------------------------------------------------
# MATH-500  (competition math; \boxed answer, normalized equivalence)
# ----------------------------------------------------------------------------
def _boxed(text):
    """Extract the content of the last \\boxed{...} with balanced braces."""
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None
    i = idx + len("\\boxed")
    while i < len(text) and text[i] != "{":
        i += 1
    if i >= len(text):
        return None
    depth, start = 0, i
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1:j]
    return None


def _math_norm(s):
    if s is None:
        return None
    s = str(s).strip()
    for tok in ["\\left", "\\right", "\\!", "\\,", "\\;", "\\ ", "\\$", "$",
                "\\text", "\\mathrm", "\\displaystyle", "^\\circ", "\\%"]:
        s = s.replace(tok, "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = s.replace("dfrac", "frac").replace("tfrac", "frac")
    s = s.replace("\\frac{", "frac{")
    s = re.sub(r"\\(text|mbox|mathbf|mathrm){([^}]*)}", r"\2", s)
    s = s.replace("{", "").replace("}", "").replace(" ", "")
    s = s.replace("\\", "").rstrip(".").rstrip("%")
    return s.lower()


def math_load(n=None, split="all", seed=0, max_level=3):
    """Load MATH-500, filtered to levels <= max_level (where a small model has
    real signal; levels 4-5 are near-floor for a 3B model)."""
    rows = []
    with open(os.path.join(DATA_DIR, "math500.jsonl")) as f:
        for i, line in enumerate(f):
            d = json.loads(line)
            if int(d.get("level", 5)) > max_level:
                continue
            rows.append({"id": f"math-{i}", "question": d["problem"],
                         "gold": d["answer"]})
    return _split_take(rows, n, split, seed)


def math_prompt(q):
    return ("Solve the following competition math problem. Show brief reasoning, "
            "then give your final answer in \\boxed{}.\n\n"
            f"Problem: {q}\n")


def math_extract(text):
    # Prefer an explicit \boxed{}; else a short "answer is/= X" token; else None
    # (returning None for unparseable output is honest -- we do not grab prose).
    b = _boxed(text)
    if b is not None:
        return b.strip()
    m = re.search(r"(?:final answer|answer)\s*(?:is|:|=)\s*\$?([^.\n$]{1,40})",
                  text, re.IGNORECASE)
    if m:
        cand = m.group(1).strip().strip("$")
        # reject obvious prose (multiple words of plain text)
        if cand and not re.match(r"^[A-Za-z]+\s+[A-Za-z]+", cand):
            return cand
    return None


def math_score(pred, gold):
    if pred is None or gold is None:
        return False
    if _math_norm(pred) == _math_norm(gold):
        return True
    p, g = _to_number(_math_norm(pred)), _to_number(_math_norm(gold))
    if p is not None and g is not None:
        return abs(float(p) - float(g)) < 1e-4
    return False


# ----------------------------------------------------------------------------
# MMLU  (knowledge multiple-choice; letter answer)
# ----------------------------------------------------------------------------
def mmlu_load(n=None, split="all", seed=0):
    rows = []
    with open(os.path.join(DATA_DIR, "mmlu.jsonl")) as f:
        for i, line in enumerate(f):
            d = json.loads(line)
            choices = d["choices"]
            block = "\n".join(f"{chr(65+j)}) {c}" for j, c in enumerate(choices))
            rows.append({"id": f"mmlu-{i}",
                         "question": f"{d['question']}\n{block}",
                         "gold": "ABCD"[int(d["answer"])]})
    return _split_take(rows, n, split, seed)


def mmlu_prompt(q):
    return ("Answer the following multiple-choice question. Think briefly, then "
            "on the final line write exactly: 'The answer is X.' where X is the "
            f"letter (A, B, C, or D) of the correct choice.\n\n{q}\n")


def mmlu_extract(text):
    m = re.search(r"answer is\s*:?\s*\(?([A-D])\)?", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m2 = re.findall(r"\b([A-D])\b", text)
    return m2[-1].upper() if m2 else None


def mmlu_score(pred, gold):
    return pred is not None and gold is not None and str(pred).upper() == str(gold).upper()


# ----------------------------------------------------------------------------
# MMLU-Pro, engineering category  (TIGER-Lab/MMLU-Pro, Wang et al. 2024,
# arXiv:2406.01574; up to ten options A..J, some items have fewer)
#
# The MMLU functions above are deliberately left untouched (prior logs must
# stay reproducible byte for byte). The parser below is a separate, letter-set
# parametrized extractor; `mc_extract(text, 4)` is NOT guaranteed to equal
# `mmlu_extract(text)`, and the MMLU task never calls it.
# Data: tasks_data/mmlu_pro_engineering.jsonl, built by
#       scripts/prep_mmlu_pro_eng.py (fixed seed, n=150).
# ----------------------------------------------------------------------------
MC_LETTERS = "ABCDEFGHIJ"

# "answer is" / "answer:" / "answer =" lead-in, optionally "option"/"choice"
# and markdown bold. Case-insensitive on the words only.
_MC_LEAD = (r"(?i:\banswer\s*(?:is\b\s*:?|:|=))\s*\**\s*"
            r"(?i:(?:option|choice)\s*)?")
# Uppercase letter after the lead-in, optionally parenthesized, not followed by
# a letter/digit (so "Answer: H" and "The answer is (J)." both parse).
_MC_UP = re.compile(_MC_LEAD + r"\(?\s*([A-J])\s*\)?(?![A-Za-z0-9])")
# Lowercase letter is accepted only when parenthesized or line/sentence-final,
# so "the answer is a capacitor" is not read as option A.
_MC_LOW = re.compile(_MC_LEAD + r"(?:\(\s*([a-j])\s*\)|([a-j])"
                     r"(?=[ \t]*(?:[.,;!)\]*\n]|\Z)))")
_MC_BOXED = re.compile(r"\\boxed\{\s*(?:\\text(?:bf)?\{\s*)?\(?\s*([A-Ja-j])"
                       r"\s*\)?\s*\}")
# Fallbacks (used only when no explicit answer statement exists).
_MC_OPT = re.compile(r"(?i:\b(?:option|choice)\s*)\(?([A-J])\)?(?![A-Za-z0-9])")
# Standalone uppercase letter, excluding pronoun/article use ("I think",
# "A capacitor", "I'm").
_MC_BARE = re.compile(r"\b([A-J])\b(?!\s+[a-z])(?!['\u2019])")


def mc_extract(text, n_options=10):
    """Parse a multiple-choice letter from model output, restricted to the
    first `n_options` letters of A..J.

    1. Explicit statements ("The answer is (J)", "Answer: H", \\boxed{J},
       lowercase "the answer is (h)"): the LAST such statement wins (multi-agent
       outputs often quote other agents' answers before the final one). If that
       letter is beyond the option count the output is rejected (None); there
       is no fall-through to a different letter.
    2. Only if no explicit statement exists: the last "option X"/"choice X" or
       standalone uppercase letter within range.
    Returns an uppercase letter or None.
    """
    if text is None:
        return None
    n = max(1, min(int(n_options), len(MC_LETTERS)))
    valid = MC_LETTERS[:n]
    text = str(text)
    best = None  # (position, letter)
    for rx in (_MC_UP, _MC_LOW, _MC_BOXED):
        for m in rx.finditer(text):
            g = next(x for x in m.groups() if x)
            if best is None or m.start() >= best[0]:
                best = (m.start(), g.upper())
    if best is not None:
        return best[1] if best[1] in valid else None
    cands = []
    for rx in (_MC_OPT, _MC_BARE):
        for m in rx.finditer(text):
            if m.group(1) in valid:
                cands.append((m.start(1), m.group(1)))
    return max(cands)[1] if cands else None


def _mc_letter_list(n):
    ls = list(MC_LETTERS[:n])
    return ls[0] if n == 1 else ", ".join(ls[:-1]) + ", or " + ls[-1]


_MC_OPTLINE = re.compile(r"(?m)^([A-J])\) ")


def mmlu_pro_load(n=None, split="all", seed=0,
                  fname="mmlu_pro_engineering.jsonl"):
    rows = []
    with open(os.path.join(DATA_DIR, fname)) as f:
        for line in f:
            d = json.loads(line)
            opts = d["options"]
            k = len(opts)
            if not 1 <= k <= len(MC_LETTERS):
                raise ValueError(f"{d['question_id']}: {k} options")
            gold = MC_LETTERS[int(d["answer_index"])]
            if d.get("answer") and d["answer"] != gold:
                raise ValueError(f"{d['question_id']}: answer/answer_index mismatch")
            block = "\n".join(f"{MC_LETTERS[j]}) {c}" for j, c in enumerate(opts))
            rows.append({"id": f"mmlupro-{d['question_id']}",
                         "question": f"{d['question']}\n{block}",
                         "gold": gold, "n_options": k})
    return _split_take(rows, n, split, seed)


def mmlu_pro_prompt(q):
    # Same wording as mmlu_prompt; only the letter set differs. The number of
    # options is read from the option block that mmlu_pro_load appends.
    # Use the final run "A) ... B) ..." so a stray "X) " line inside the
    # question body cannot change the count.
    letters = _MC_OPTLINE.findall(q)
    run = letters[len(letters) - 1 - letters[::-1].index("A"):] if "A" in letters else []
    k = len(run) if run and "".join(run) == MC_LETTERS[:len(run)] else len(MC_LETTERS)
    return ("Answer the following multiple-choice question. Think briefly, then "
            "on the final line write exactly: 'The answer is X.' where X is the "
            f"letter ({_mc_letter_list(k)}) of the correct choice.\n\n{q}\n")


def mmlu_pro_extract(text):
    """Item-agnostic extractor (A..J). Topologies prefer `extract_item`, which
    also rejects letters beyond the item's own option count."""
    return mc_extract(text, len(MC_LETTERS))


def mmlu_pro_extract_item(text, item):
    return mc_extract(text, item.get("n_options", len(MC_LETTERS)))



# ----------------------------------------------------------------------------
# HumanEval (Chen et al. 2021; canonical data/HumanEval.jsonl.gz of
# github.com/openai/human-eval, sha256 of the uncompressed file
# 1d49078ba3e2b196b9344535bef34a43021f038fad9561d6ee7c53450609a6a2).
# Design and voting rule: paper/eaai/CODE_ENERGY_EXPERIMENT.md. Scoring runs the
# hidden tests on the model's program in harness/sandbox.py (never unsandboxed).
# ----------------------------------------------------------------------------
import ast as _ast
import functools as _functools

_FENCE = re.compile(r"```[ \t]*[A-Za-z0-9_+.-]*[ \t]*\n(.*?)```", re.DOTALL)
_OPEN_FENCE = re.compile(r"```[ \t]*[A-Za-z0-9_+.-]*[ \t]*\n(.*)\Z", re.DOTALL)


def code_block(text):
    """The last fenced code block; an unterminated final fence (output cut at
    the generation cap) counts as a block; else the whole text."""
    if text is None:
        return None
    blocks = _FENCE.findall(text)
    tail = text[text.rfind("```") :] if text.count("```") % 2 == 1 else ""
    if tail:
        m = _OPEN_FENCE.match(tail)
        if m:
            return m.group(1)
    if blocks:
        return blocks[-1]
    return text


def _strip_docstrings(tree):
    for node in _ast.walk(tree):
        if isinstance(node, (_ast.Module, _ast.FunctionDef, _ast.AsyncFunctionDef,
                             _ast.ClassDef)):
            b = node.body
            if b and isinstance(b[0], _ast.Expr) and \
                    isinstance(getattr(b[0], "value", None), _ast.Constant) and \
                    isinstance(b[0].value.value, str):
                node.body = b[1:] or [_ast.Pass()]
    return tree


def _defines(tree, name):
    return any(isinstance(n, (_ast.FunctionDef, _ast.AsyncFunctionDef))
               and n.name == name for n in tree.body)


def code_canonical(text, item):
    """Model output -> canonical executable program (str) or None.

    If the extracted code defines the entry point, the program is the prompt's
    preamble (every top-level statement of the prompt except the entry-point
    stub: imports and helper functions) followed by the code; otherwise the
    code is read as a body-only completion appended to the prompt. Docstrings
    are removed and the program is re-emitted by ast.unparse, so formatting,
    comments, and docstrings do not affect votes."""
    code = code_block(text)
    if code is None:
        return None
    entry, prompt = item["entry_point"], item["prompt"]
    try:
        tree = _ast.parse(code)
    except SyntaxError:
        tree = None
    if tree is not None and _defines(tree, entry):
        pre = [n for n in _ast.parse(prompt).body
               if not (isinstance(n, (_ast.FunctionDef, _ast.AsyncFunctionDef))
                       and n.name == entry)]
        prog = _ast.Module(body=pre + tree.body, type_ignores=[])
    else:
        try:
            prog = _ast.parse(prompt + code)
        except SyntaxError:
            return None
        if not _defines(prog, entry):
            return None
    try:
        return _ast.unparse(_strip_docstrings(prog))
    except Exception:
        return None


# ---- rev3 extractor (protocols/REV3_RERUN_PROTOCOL.md) -----------------------
# Added for the rev3 rerun. code_block / code_canonical above are unchanged and
# remain the extractor of the "humaneval" task; the functions below are only
# used where they are named (task "humaneval_entryblock", scripts/run_rev3_campaign.py).
def code_blocks_all(text):
    """Every fenced code block of `text`, in order of appearance; when the text
    has an odd number of fences, the content after the last one (an
    unterminated final fence, as in code_block) is the last element."""
    if text is None:
        return []
    blocks = list(_FENCE.findall(text))
    if text.count("```") % 2 == 1:
        m = _OPEN_FENCE.match(text[text.rfind("```"):])
        if m:
            blocks.append(m.group(1))
    return blocks


def code_block_entry(text, entry_point):
    """rev3 block choice: the LAST fenced block (code_blocks_all order) that
    parses as Python and defines `entry_point` as a top-level function. If no
    block does, the old choice code_block(text) is returned unchanged, so the
    two extractors differ only when the old choice does not define the entry
    function while an earlier block does."""
    if text is None:
        return None
    for code in reversed(code_blocks_all(text)):
        try:
            tree = _ast.parse(code)
        except (SyntaxError, ValueError):
            continue
        if _defines(tree, entry_point):
            return code
    return code_block(text)


def _program_from_code(code, item):
    """The program assembly of code_canonical, applied to an already chosen
    code string (same statements, same order; tests check equality with
    code_canonical when `code` is code_block(text))."""
    if code is None:
        return None
    entry, prompt = item["entry_point"], item["prompt"]
    try:
        tree = _ast.parse(code)
    except SyntaxError:
        tree = None
    if tree is not None and _defines(tree, entry):
        pre = [n for n in _ast.parse(prompt).body
               if not (isinstance(n, (_ast.FunctionDef, _ast.AsyncFunctionDef))
                       and n.name == entry)]
        prog = _ast.Module(body=pre + tree.body, type_ignores=[])
    else:
        try:
            prog = _ast.parse(prompt + code)
        except SyntaxError:
            return None
        if not _defines(prog, entry):
            return None
    try:
        return _ast.unparse(_strip_docstrings(prog))
    except Exception:
        return None


def code_canonical_v2(text, item):
    """code_canonical with the rev3 block choice (code_block_entry)."""
    return _program_from_code(code_block_entry(text, item["entry_point"]), item)


def humaneval_load(n=None, split="all", seed=0):
    rows = []
    with open(os.path.join(DATA_DIR, "HumanEval.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            num = d["task_id"].split("/")[-1]
            rows.append({"id": f"humaneval-{num}", "question": d["prompt"],
                         "prompt": d["prompt"], "entry_point": d["entry_point"],
                         "gold": {"task_id": d["task_id"],
                                  "entry_point": d["entry_point"],
                                  "test": d["test"]}})
    return _split_take(rows, n, split, seed)


def humaneval_prompt(q):
    return ("Complete the following Python function. Give the complete "
            "function, including its signature and any imports it needs, in a "
            f"single ```python code block.\n\n```python\n{q}```\n")


def humaneval_extract(text):
    raise RuntimeError("humaneval needs the item: use extract_item")


def humaneval_extract_item(text, item):
    return code_canonical(text, item)


def humaneval_extract_item_v2(text, item):
    """rev3 extractor: the block that defines the entry function."""
    return code_canonical_v2(text, item)


@_functools.lru_cache(maxsize=4096)
def _run_check(program, test, entry_point):
    from . import sandbox
    res = sandbox.run_program(f"{program}\n\n{test}\n\ncheck({entry_point})\n")
    return bool(res["ok"])


def humaneval_score(pred, gold):
    """Pass iff the program plus the hidden tests exits 0 in the sandbox."""
    if pred is None or gold is None:
        return False
    return _run_check(pred, gold["test"], gold["entry_point"])


_DOCTEST = re.compile(r"^\s*>>>\s*(.+?)\s*$", re.MULTILINE)


def humaneval_examples(item):
    """Expressions of the prompt's own `>>> f(...)` examples (public, not the
    hidden tests) that call the entry point."""
    return [e for e in _DOCTEST.findall(item["prompt"])
            if item["entry_point"] + "(" in e]


def example_signature(program, item):
    """Secondary voting key: sandboxed outputs of `program` on the prompt's
    examples, or None if the prompt has no usable examples or no program."""
    ex = humaneval_examples(item)
    if program is None or not ex:
        return None
    from . import sandbox
    lines = [program, "", "def _sig():"]
    for e in ex:
        lines += ["    try:", f"        print(repr({e}))",
                  "    except Exception as _e:",
                  "        print('EXC', type(_e).__name__)"]
    lines += ["_sig()", ""]
    res = sandbox.run_program("\n".join(lines))
    return res["stdout"] if not res["timeout"] else "TIMEOUT"

# ----------------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------------
TASKS = {
    "gsm8k": {"load": gsm8k_load, "prompt": gsm8k_prompt, "extract": gsm8k_extract,
              "score": gsm8k_score,
              "reminder": "End with exactly one line: 'The answer is N.'"},
    "math500": {"load": math_load, "prompt": math_prompt, "extract": math_extract,
                "score": math_score,
                "reminder": "Give your final answer in \\boxed{}."},
    "mmlu": {"load": mmlu_load, "prompt": mmlu_prompt, "extract": mmlu_extract,
             "score": mmlu_score,
             "reminder": "End with exactly one line: 'The answer is X.' (a letter A-D)."},
    # MMLU-Pro engineering (EAAI engineering-domain experiment). `extract_item`
    # is an optional per-item extractor; topologies use it when present and
    # fall back to `extract` otherwise, so the tasks above behave as before.
    "mmlu_pro_eng": {"load": mmlu_pro_load, "prompt": mmlu_pro_prompt,
                     "extract": mmlu_pro_extract,
                     "extract_item": mmlu_pro_extract_item,
                     "score": mmlu_score,
                     "reminder": "End with exactly one line: 'The answer is X.' "
                                 "(the letter of one of the listed options, A-J)."},
    # HumanEval code generation (executable hidden tests, sandboxed). The
    # answer is the canonical program, so majority vote = exact-program vote.
    "humaneval": {"load": humaneval_load, "prompt": humaneval_prompt,
                  "extract": humaneval_extract,
                  "extract_item": humaneval_extract_item,
                  "score": humaneval_score,
                  "reminder": "Give the complete final function in one "
                              "```python code block."},
}
# rev3: HumanEval with the entry-block extractor; everything else (loader,
# prompt, reminder, scorer) is the "humaneval" entry. Selecting "humaneval"
# keeps the original last-block extractor.
TASKS["humaneval_entryblock"] = dict(TASKS["humaneval"],
                                     extract_item=humaneval_extract_item_v2)
HUMANEVAL_EXTRACTORS = {"last_block": humaneval_extract_item,
                        "entry_block": humaneval_extract_item_v2}


def get_task(name):
    return TASKS[name]


if __name__ == "__main__":
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else "math500"
    for it in get_task(name)["load"](n=3, split="all"):
        print(it["id"], "gold=", repr(it["gold"]), "|", it["question"][:60])
