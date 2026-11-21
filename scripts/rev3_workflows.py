"""HIERFIX on every task (rev3 protocol, protocols/REV3_RERUN_PROTOCOL.md).

HIERFIX_M3 is harness.topologies.hier (M=3), called unchanged, with a task copy
whose `reminder` (used by hier only at the end of the aggregator prompt) is
replaced by one rule, fixed before any rev3 output exists:

  If the task's reminder asks for the template line "The answer is P." with a
  placeholder letter P, the HIERFIX reminder is
      End with exactly one line: 'The answer is P.' with P replaced by <D>.
  where <D> is the definition of P that the task's own first-call prompt gives
  ("where P is <D>."). Otherwise (no template line with a placeholder) the
  reminder is unchanged and HIERFIX_M3 is byte-identical to HIER_M3.

For GSM8K this reproduces the HIERFIX wording of protocols/EXTRA_WORKFLOWS_PROTOCOL.md
exactly (asserted below against scripts/extra_workflows.py). MMLU-Pro engineering
items have 2 to 10 options; the first-call prompt lists the item's own letters, so
the HIERFIX reminder does too. MATH-500 (\\boxed{}) and HumanEval (code block) have
no placeholder: HIERFIX = HIER there.
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from harness import tasks, topologies  # noqa: E402

# Reminders as they are in harness/tasks.py (asserted at call time).
OLD_REMINDER = {
    "gsm8k": "End with exactly one line: 'The answer is N.'",
    "mmlu": "End with exactly one line: 'The answer is X.' (a letter A-D).",
    "mmlu_pro_eng": "End with exactly one line: 'The answer is X.' "
                    "(the letter of one of the listed options, A-J).",
    "math500": "Give your final answer in \\boxed{}.",
    "humaneval": "Give the complete final function in one ```python code block.",
}
IDENTICAL = {"math500", "humaneval"}  # no placeholder: HIERFIX = HIER
_DEF = re.compile(r"'The answer is ([A-Z])\.' where \1 is (.+?)\.\n")


def placeholder_definition(task_name, item):
    """(P, D) read from the task's first-call prompt for this item."""
    p = tasks.get_task(task_name)["prompt"](item["question"])
    m = _DEF.search(p)
    assert m, (task_name, p[:200])
    return m.group(1), m.group(2)


def hierfix_reminder(task_name, item):
    if task_name in IDENTICAL:
        return OLD_REMINDER[task_name]
    P, D = placeholder_definition(task_name, item)
    return f"End with exactly one line: 'The answer is {P}.' with {P} replaced by {D}."


def hierfix(model, task_name, task, item, params, seed):
    """HIERFIX_M3: topologies.hier with only the aggregator reminder replaced
    (see module docstring). Returns hier's result plus the reminder used."""
    assert task["reminder"] == OLD_REMINDER[task_name], "harness reminder changed"
    rem = hierfix_reminder(task_name, item)
    t2 = task if rem == task["reminder"] else dict(task, reminder=rem)
    res = topologies.hier(model, t2, item, params, seed)
    res["agg_reminder"] = rem
    res["identical_to_hier"] = rem == task["reminder"]
    return res


def _check_gsm8k_matches_extra():
    """The GSM8K HIERFIX wording equals the one already run (extra protocol)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import extra_workflows as xw
    it = tasks.get_task("gsm8k")["load"](n=1)[0]
    assert hierfix_reminder("gsm8k", it) == xw.HIERFIX_REMINDER["gsm8k"], \
        (hierfix_reminder("gsm8k", it), xw.HIERFIX_REMINDER["gsm8k"])
    assert OLD_REMINDER["gsm8k"] == xw.HIERFIX_OLD_REMINDER["gsm8k"]
    return True


if __name__ == "__main__":
    print("gsm8k matches extra_workflows:", _check_gsm8k_matches_extra())
    for t in ("gsm8k", "mmlu", "mmlu_pro_eng", "math500", "humaneval"):
        it = tasks.get_task(t)["load"](n=1)[0]
        assert tasks.get_task(t)["reminder"] == OLD_REMINDER[t], t
        print(f"{t:13s} old: {OLD_REMINDER[t]}\n{'':13s} fix: {hierfix_reminder(t, it)}")
