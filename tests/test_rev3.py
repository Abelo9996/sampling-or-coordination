"""Offline tests for the rev3 rerun code (protocols/REV3_RERUN_PROTOCOL.md).
  /usr/bin/python3 -m unittest tests.test_rev3 -v
No model is called. The HumanEval matrix run did not log model output text
(results/audit/humaneval_unparsed_audit.py), so the logged records can only be
used through their extracted programs: each logged program is re-wrapped in the
output shapes the two extractors see, and the counts are printed."""
import json
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from harness import tasks  # noqa: E402

T = tasks.get_task("humaneval")
ITEMS = {it["id"]: it for it in T["load"]()}
I0 = ITEMS["humaneval-0"]
SOL0 = ("from typing import List\n\n"
        "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n"
        "    for i, a in enumerate(numbers):\n"
        "        for j, b in enumerate(numbers):\n"
        "            if i != j and abs(a - b) < threshold:\n"
        "                return True\n"
        "    return False\n")
TEST0 = "print(has_close_elements([1.0, 2.0], 0.5))\n"


def fence(code, lang="python"):
    return f"```{lang}\n{code}```\n"


class TestEntryBlock(unittest.TestCase):
    def test_solution_then_test_block(self):
        t = "Solution:\n" + fence(SOL0) + "Example usage:\n" + fence(TEST0)
        old, new = tasks.code_canonical(t, I0), tasks.code_canonical_v2(t, I0)
        self.assertIn("print(has_close_elements", old)       # old: stub + test code
        self.assertNotIn("abs(a - b)", old)
        self.assertIn("abs(a - b)", new)                      # new: the solution
        self.assertTrue(T["score"](new, I0["gold"]) if _sbx() else True)
        self.assertFalse(T["score"](old, I0["gold"]) if _sbx() else False)

    def test_last_defining_block_wins(self):
        quoted = SOL0.replace("< threshold", "<= threshold")  # another agent's version
        t = "Agent 1 wrote:\n" + fence(quoted) + "My updated solution:\n" + fence(SOL0) + fence(TEST0)
        self.assertIn("< threshold", tasks.code_canonical_v2(t, I0))
        self.assertNotIn("<= threshold", tasks.code_canonical_v2(t, I0))

    def test_unterminated_final_fence(self):
        t = fence(SOL0) + "Tests:\n```python\nassert has_close_elements([1.0], 0.1) =="  # cut at cap
        self.assertIsNotNone(tasks.code_canonical_v2(t, I0))
        self.assertIn("abs(a - b)", tasks.code_canonical_v2(t, I0))
        t2 = "```python\n" + SOL0  # unterminated block that defines the function
        self.assertEqual(tasks.code_canonical_v2(t2, I0), tasks.code_canonical(t2, I0))

    def test_identical_when_last_block_defines_entry(self):
        for t in (fence(SOL0), "text\n" + fence(TEST0) + fence(SOL0), SOL0, fence(SOL0, "")):
            self.assertEqual(tasks.code_canonical_v2(t, I0), tasks.code_canonical(t, I0))

    def test_no_defining_block_falls_back_to_old(self):
        body = "    return any(abs(a-b) < threshold for i, a in enumerate(numbers) " \
               "for j, b in enumerate(numbers) if i != j)\n"
        for t in (fence(body), fence(TEST0) + fence(body), "no code at all", None):
            self.assertEqual(tasks.code_canonical_v2(t, I0) if t else tasks.code_block_entry(t, "x"),
                             tasks.code_canonical(t, I0) if t else None)

    def test_program_from_code_equals_code_canonical(self):
        cases = [fence(SOL0), fence(TEST0), fence(SOL0) + fence(TEST0), SOL0, "junk ```",
                 "```python\n" + SOL0, fence("def broken(:\n"), ""]
        for t in cases:
            self.assertEqual(tasks._program_from_code(tasks.code_block(t), I0),
                             tasks.code_canonical(t, I0), t[:40])

    def test_task_registry(self):
        self.assertIs(tasks.TASKS["humaneval"]["extract_item"], tasks.humaneval_extract_item)
        self.assertIs(tasks.TASKS["humaneval_entryblock"]["extract_item"],
                      tasks.humaneval_extract_item_v2)
        for k in ("load", "prompt", "score", "reminder"):
            self.assertIs(tasks.TASKS["humaneval_entryblock"][k], tasks.TASKS["humaneval"][k])


def _sbx():
    from harness import sandbox
    return sandbox.default_backend() in ("macos", "docker")


class TestLoggedPrograms(unittest.TestCase):
    """Logged HumanEval programs (results/humaneval_matrix_pc.jsonl: pred of every
    record, and the per-sample programs SC and debate keep in `raw`)."""

    def test_logged(self):
        path = os.path.join(ROOT, "results", "humaneval_matrix_pc.jsonl")
        recs = [json.loads(l) for l in open(path)]
        progs, unparsed = [], 0
        for r in recs:
            it = ITEMS[r["item_id"]]
            if r["pred"] in (None, ""):
                unparsed += 1
            else:
                progs.append((r["pred"], it))
            for p in (r.get("raw") or []):
                if p:
                    progs.append((p, it))
        same_single = same_with_test = rescued_shape = 0
        for p, it in progs:
            t1 = fence(p + "\n")
            if tasks.code_canonical_v2(t1, it) == tasks.code_canonical(t1, it):
                same_single += 1
            call = f"{it['entry_point']}()\n"
            t2 = fence(p + "\n") + "Test:\n" + fence(f"print({call.strip()})\n")
            v1, v2 = tasks.code_canonical(t2, it), tasks.code_canonical_v2(t2, it)
            if v2 == tasks.code_canonical(t1, it):
                same_with_test += 1
            if v1 != v2:
                rescued_shape += 1
        n = len(progs)
        print(f"\n[logged] records {len(recs)}, unparsed records {unparsed} (text not logged: "
              f"not recoverable offline); logged programs {n}; "
              f"single block: v2 == v1 for {same_single}/{n}; "
              f"solution + trailing test block: v2 == solution for {same_with_test}/{n}, "
              f"v1 != v2 for {rescued_shape}/{n}")
        self.assertEqual(same_single, n)
        self.assertEqual(same_with_test, n)


_DU_UPSTREAM = os.path.join(ROOT, "third_party", "du2023_debate", "upstream", "gsm", "gen_gsm.py")


class TestHierfix(unittest.TestCase):
    # Release note: scripts/extra_workflows.py compiles code from the Du et al. files, which
    # are not redistributed; fetch them with third_party/du2023_debate/fetch_upstream.sh.
    @unittest.skipUnless(os.path.exists(_DU_UPSTREAM), "Du et al. upstream not fetched")
    def test_reminders(self):
        import rev3_workflows as rw
        self.assertTrue(rw._check_gsm8k_matches_extra())
        for t in ("math500", "humaneval"):
            it = tasks.get_task(t)["load"](n=1)[0]
            self.assertEqual(rw.hierfix_reminder(t, it), tasks.get_task(t)["reminder"])
        for t in ("gsm8k", "mmlu", "mmlu_pro_eng"):
            for it in tasks.get_task(t)["load"](n=200):
                r = rw.hierfix_reminder(t, it)
                self.assertNotEqual(r, tasks.get_task(t)["reminder"])
                self.assertIn("replaced by", r)
                self.assertIn("where " + r[len("End with exactly one line: 'The answer is ")] + " is "
                              + r.split(" replaced by ")[1],
                              tasks.get_task(t)["prompt"](it["question"]))

    def test_final_call_idx(self):
        from run_rev3_campaign import final_call_idx
        self.assertEqual(final_call_idx("SA", {}, 1), [0])
        self.assertEqual(final_call_idx("SC", {"k": 8}, 8), list(range(8)))
        self.assertEqual(final_call_idx("DEBATE", {"N": 3, "R": 1}, 6), [3, 4, 5])
        self.assertEqual(final_call_idx("GE", {"R": 2}, 5), [4])
        self.assertEqual(final_call_idx("HIERFIX", {"M": 3}, 5), [4])


if __name__ == "__main__":
    unittest.main()
