"""Tests for harness/traces.py and the optional trace hooks in topologies.

A fake client returns scripted texts, so no model is called. Run:
  python3 -m unittest tests.test_traces -v
"""
import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from harness import client, tasks, topologies, traces  # noqa: E402


class FakeGen:
    """Replace client.generate; returns texts in call order."""
    def __init__(self, texts):
        self.texts = list(texts)
        self.prompts = []

    def __call__(self, model, prompt, temperature=0.0, seed=None, **kw):
        self.prompts.append((prompt, temperature, seed))
        t = self.texts.pop(0)
        return client.CallRecord(text=t, in_tokens=10, out_tokens=5,
                                 latency_s=0.5, model=model,
                                 temperature=temperature, seed=seed)


ITEM = {"id": "gsm8k-x", "question": "What is 2+2?", "gold": 4}
GSM = tasks.get_task("gsm8k")


_ORIG = client.generate


def _patch(_unused, texts):
    fake = FakeGen(texts)
    client.generate = fake
    return fake


class TraceTests(unittest.TestCase):
    def tearDown(self):
        client.generate = _ORIG


    def test_debate_trace_corruption(self, monkeypatch=None):
        # round 0: 4, 4, 5 (majority 4, correct); round 1: 5, 5, 4 (majority 5)
        texts = ["The answer is 4.", "The answer is 4.", "The answer is 5.",
                 "The answer is 5.", "The answer is 5.", "The answer is 4."]
        _patch(monkeypatch, texts)
        rec = traces.trace_record("m", "gsm8k", GSM, ITEM, "debate", 1)
        assert rec["agent_answers"] == [[4, 4, 5], [5, 5, 4]]
        assert rec["agent_correct"] == [[True, True, False], [False, False, True]]
        assert rec["init_answer"] == 4 and rec["final_answer"] == 5
        assert rec["init_correct"] is True and rec["final_correct"] is False
        assert rec["n_calls"] == 6 and rec["total_tokens"] == 90
        assert len(rec["calls"]) == 6 and rec["calls"][0]["latency_s"] == 0.5
        json.dumps(rec)


    def test_trace_hook_does_not_change_calls(self, monkeypatch=None):
        texts = ["The answer is 3.", "The answer is 4.", "The answer is 3.",
                 "The answer is 4.", "The answer is 4.", "The answer is 4."]
        f1 = _patch(monkeypatch, texts)
        r1 = topologies.debate("m", GSM, ITEM, {"N": 3, "R": 1}, 1)
        f2 = _patch(monkeypatch, texts)
        r2 = topologies.debate("m", GSM, ITEM, {"N": 3, "R": 1}, 1, trace={})
        assert f1.prompts == f2.prompts
        assert r1["answer"] == r2["answer"] == 4
        g = ["The answer is 3.", "Wrong step.", "The answer is 4.",
             "No errors.", "The answer is 4."]
        f1 = _patch(monkeypatch, g)
        r1 = topologies.ge("m", GSM, ITEM, {"R": 2}, 2)
        f2 = _patch(monkeypatch, g)
        r2 = topologies.ge("m", GSM, ITEM, {"R": 2}, 2, trace={})
        assert f1.prompts == f2.prompts and r1["answer"] == r2["answer"] == 4


    def test_ge_trace_fix_and_verdicts(self, monkeypatch=None):
        # draft 3 (wrong), critic flags, revise to 4; critic accepts, revise 4
        texts = ["The answer is 3.", "Mistake: 2+2 is not 3.", "The answer is 4.",
                 "No errors. The solution is right.", "The answer is 4."]
        _patch(monkeypatch, texts)
        rec = traces.trace_record("m", "gsm8k", GSM, ITEM, "ge", 1)
        assert rec["draft_answer"] == 3 and rec["final_answer"] == 4
        assert rec["init_correct"] is False and rec["final_correct"] is True
        assert [r["critic_found_errors"] for r in rec["rounds"]] == [True, False]
        assert [(r["before_correct"], r["after_correct"]) for r in rec["rounds"]] \
            == [(False, True), (True, True)]
        assert rec["n_calls"] == 5


    def test_critic_verdict_matches_ges_gate(self):
        assert topologies.critic_found_errors("No errors.") is False
        assert topologies.critic_found_errors("The solution is fully correct.") is False
        assert topologies.critic_found_errors("Step 2 is wrong.") is True
        assert topologies.critic_found_errors("x" * 250 + "no errors") is True


    def test_mmlu_pro_item_extractor_used(self, monkeypatch=None):
        task = tasks.get_task("mmlu_pro_eng")
        item = {"id": "mmlupro-1", "question": "Q\nA) a\nB) b\nC) c\nD) d",
                "gold": "B", "n_options": 4}
        # "J" is out of range for a 4-option item and must parse to None
        texts = ["The answer is J."] + ["The answer is B."] * 5
        _patch(monkeypatch, texts)
        rec = traces.trace_record("m", "mmlu_pro_eng", task, item, "debate", 1)
        assert rec["agent_answers"][0] == [None, "B", "B"]
        assert rec["init_correct"] and rec["final_correct"]


    def test_resume_repairs_partial_line(self):
        p = os.path.join(tempfile.mkdtemp(), "t.jsonl")
        traces.append_record(str(p), {"item_id": "a", "wf": "ge", "seed": 1})
        with open(p, "a") as f:
            f.write('{"item_id": "b", "wf": "ge", "se')   # power loss mid-write
        assert traces.load_done(str(p)) == {("a", "ge", 1)}
        assert traces.repair_tail(str(p)) > 0
        traces.append_record(str(p), {"item_id": "b", "wf": "ge", "seed": 1})
        lines = open(p).read().splitlines()
        assert [json.loads(x)["item_id"] for x in lines] == ["a", "b"]
        assert traces.repair_tail(str(p)) == 0


if __name__ == "__main__":
    unittest.main()
