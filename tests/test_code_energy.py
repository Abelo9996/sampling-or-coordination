"""Tests for the HumanEval task, the sandbox, and the energy integrator.
  python3 -m unittest tests.test_code_energy -v
Sandbox tests run only where a sandbox backend exists (macOS sandbox-exec or
Docker); the code they execute is written here, not model output."""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from harness import energy, sandbox, tasks, topologies  # noqa: E402

T = tasks.get_task("humaneval")
ITEMS = {it["gold"]["task_id"]: it for it in T["load"]()}
I0 = ITEMS["HumanEval/0"]      # has_close_elements, imports List
HAVE_SBX = sandbox.default_backend() in ("macos", "docker")

FULL = '''Here is my solution:
```python
from typing import List

def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """Docstring that should not affect the vote."""
    # a comment
    for i, a in enumerate(numbers):
        for j, b in enumerate(numbers):
            if i != j and abs(a - b) < threshold:
                return True
    return False
```
'''
FULL_REFORMATTED = '''```python
def has_close_elements(numbers, threshold):
    for i, a in enumerate(numbers):
        for j, b in enumerate(numbers):
            if i != j and abs(a - b) < threshold:   # same logic
                return True
    return False
```'''


class CodeExtractTests(unittest.TestCase):
    def test_last_block_and_unterminated(self):
        self.assertEqual(tasks.code_block("a\n```python\nx=1\n```\nb\n```\ny=2\n```"), "y=2\n")
        self.assertEqual(tasks.code_block("```python\nx=1\n```\n```python\ny=2"), "y=2")
        self.assertEqual(tasks.code_block("x = 3"), "x = 3")

    def test_full_function_gets_prompt_preamble(self):
        prog = T["extract_item"](FULL, I0)
        self.assertIn("from typing import List", prog)
        self.assertNotIn("Docstring", prog)
        self.assertNotIn("comment", prog)
        self.assertEqual(prog.count("def has_close_elements"), 1)

    def test_vote_ignores_comments_and_docstrings(self):
        a = T["extract_item"](FULL, I0)
        b = T["extract_item"](FULL.replace("# a comment", "# another"), I0)
        self.assertEqual(a, b)
        # signature annotations differ -> a different program (exact vote)
        c = T["extract_item"](FULL_REFORMATTED, I0)
        self.assertNotEqual(a, c)
        self.assertEqual(topologies._majority([c, a, a]), a)

    def test_body_only_completion(self):
        body = "```python\n    return any(abs(a-b) < threshold for i, a in enumerate(numbers) for j, b in enumerate(numbers) if i != j)\n```"
        prog = T["extract_item"](body, I0)
        self.assertIsNotNone(prog)
        self.assertIn("def has_close_elements", prog)

    def test_unparseable_is_none(self):
        self.assertIsNone(T["extract_item"]("```python\ndef has_close_elements(:\n```", I0))
        self.assertIsNone(T["extract_item"]("I cannot solve this.", I0))

    def test_examples_parsed(self):
        ex = tasks.humaneval_examples(I0)
        self.assertEqual(len(ex), 2)
        self.assertTrue(all(e.startswith("has_close_elements(") for e in ex))


@unittest.skipUnless(HAVE_SBX, "no sandbox backend")
class SandboxTests(unittest.TestCase):
    def test_pass_fail_timeout(self):
        good = T["extract_item"](FULL, I0)
        self.assertTrue(T["score"](good, I0["gold"]))
        bad = good.replace("< threshold", "> threshold")
        self.assertFalse(T["score"](bad, I0["gold"]))
        r = sandbox.run_program("while True: pass", timeout=2)
        self.assertFalse(r["ok"])
        self.assertTrue(r["timeout"])

    def test_network_blocked(self):
        r = sandbox.run_program(
            "import socket\nsocket.create_connection(('1.1.1.1', 80), timeout=3)\n")
        self.assertFalse(r["ok"])

    def test_example_signature(self):
        good = T["extract_item"](FULL, I0)
        self.assertEqual(tasks.example_signature(good, I0).split(), ["False", "True"])


class EnergyTests(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(energy.parse_watts("52.31\n"), 52.31)
        self.assertEqual(energy.parse_watts("52.31 W"), 52.31)
        self.assertIsNone(energy.parse_watts("[N/A]"))

    def test_integrate_constant_and_ramp(self):
        ts = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        self.assertAlmostEqual(energy.integrate(ts, [50.0] * 6, 0.1, 0.9), 40.0)
        ramp = [100.0 * t for t in ts]          # P = 100 t, integral t0..t1
        self.assertAlmostEqual(energy.integrate(ts, ramp, 0.1, 0.9), 50 * (0.81 - 0.01))
        self.assertIsNone(energy.integrate(ts, ramp, -1.0, 0.5))   # not covered


if __name__ == "__main__":
    unittest.main()
