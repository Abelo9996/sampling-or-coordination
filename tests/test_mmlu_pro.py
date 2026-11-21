"""Parser and wiring tests for the MMLU-Pro engineering task (A..J), plus a
guard that the original MMLU (A..D) task is unchanged. No model calls: the
topology test replaces client.generate with canned outputs.

Run (stdlib only):  .venv/bin/python -m unittest discover -s tests -v
"""
import inspect
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from harness import tasks, topologies  # noqa: E402

# Verbatim source of the MMLU functions at commit ae6d74d (before this change).
MMLU_SOURCE_AT_AE6D74D = {
    "mmlu_prompt": '''def mmlu_prompt(q):
    return ("Answer the following multiple-choice question. Think briefly, then "
            "on the final line write exactly: 'The answer is X.' where X is the "
            f"letter (A, B, C, or D) of the correct choice.\\n\\n{q}\\n")
''',
    "mmlu_extract": '''def mmlu_extract(text):
    m = re.search(r"answer is\\s*:?\\s*\\(?([A-D])\\)?", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m2 = re.findall(r"\\b([A-D])\\b", text)
    return m2[-1].upper() if m2 else None
''',
    "mmlu_score": '''def mmlu_score(pred, gold):
    return pred is not None and gold is not None and str(pred).upper() == str(gold).upper()
''',
}


class TestMMLUUnchanged(unittest.TestCase):
    def test_source_identical(self):
        for name, src in MMLU_SOURCE_AT_AE6D74D.items():
            self.assertEqual(inspect.getsource(getattr(tasks, name)), src, name)

    def test_registry_entry(self):
        t = tasks.TASKS["mmlu"]
        self.assertIs(t["extract"], tasks.mmlu_extract)
        self.assertIs(t["prompt"], tasks.mmlu_prompt)
        self.assertNotIn("extract_item", t)
        self.assertEqual(t["reminder"],
                         "End with exactly one line: 'The answer is X.' (a letter A-D).")

    def test_a_to_d_cases(self):
        ex = tasks.mmlu_extract
        self.assertEqual(ex("Reasoning...\nThe answer is C."), "C")
        self.assertEqual(ex("The answer is (B)."), "B")
        self.assertEqual(ex("the answer is: d"), "D")
        self.assertEqual(ex("I pick A over B, final: D"), "D")
        self.assertIsNone(ex("no letter here"))

    def test_e_still_rejected(self):
        self.assertIsNone(tasks.mmlu_extract("The answer is E."))
        self.assertIsNone(tasks.mmlu_extract("The answer is (J)."))
        # falls back to the last standalone A-D letter, as before
        self.assertEqual(tasks.mmlu_extract("Not B. The answer is E."), "B")

    def test_old_tasks_use_plain_extract_in_topologies(self):
        item = {"id": "x", "question": "q", "gold": "A", "n_options": 2}
        for name in ("gsm8k", "math500", "mmlu"):
            t = tasks.TASKS[name]
            for s in ("The answer is C.", "The answer is 42.", "\\boxed{7}"):
                self.assertEqual(topologies._extract(t, item, s), t["extract"](s))


class TestMCExtractAtoJ(unittest.TestCase):
    def test_typical_outputs(self):
        ex = tasks.mmlu_pro_extract
        self.assertEqual(ex("Some steps.\nThe answer is (J)."), "J")
        self.assertEqual(ex("The answer is J."), "J")
        self.assertEqual(ex("Answer: H"), "H")
        self.assertEqual(ex("**Answer:** G"), "G")
        self.assertEqual(ex("Final answer: \\boxed{I}"), "I")
        self.assertEqual(ex("\\boxed{\\text{E}}"), "E")
        self.assertEqual(ex("so the answer is (h)."), "H")
        self.assertEqual(ex("the answer is f"), "F")
        self.assertEqual(ex("The correct answer is J) 5 kW."), "J")
        self.assertEqual(ex("The answer is option F."), "F")

    def test_last_explicit_statement_wins(self):
        txt = ("Agent 1 says the answer is (C). Agent 2: answer is D.\n"
               "After checking, the answer is (G).")
        self.assertEqual(tasks.mmlu_pro_extract(txt), "G")

    def test_lowercase_article_not_a_letter(self):
        self.assertIsNone(tasks.mmlu_pro_extract("the answer is a capacitor bank"))

    def test_pronoun_not_a_letter_in_fallback(self):
        self.assertIsNone(tasks.mmlu_pro_extract("I think it depends on the load."))
        self.assertEqual(tasks.mmlu_pro_extract("I think option I fits best."), "I")

    def test_fallback_without_statement(self):
        self.assertEqual(tasks.mmlu_pro_extract("Comparing, (B) is close but (H) matches"), "H")
        self.assertIsNone(tasks.mmlu_pro_extract("no choice given"))
        self.assertIsNone(tasks.mmlu_pro_extract(None))

    def test_rejects_beyond_option_count(self):
        ex = tasks.mc_extract
        self.assertIsNone(ex("The answer is (J).", 4))
        self.assertIsNone(ex("The answer is E.", 4))
        self.assertIsNone(ex("Answer: J", 9))
        self.assertEqual(ex("Answer: I", 9), "I")
        self.assertEqual(ex("The answer is D.", 4), "D")
        # an out-of-range explicit answer is rejected, not replaced by another letter
        self.assertIsNone(ex("Between A and C... The answer is (F).", 4))
        # without an explicit statement, out-of-range bare letters are ignored
        self.assertEqual(ex("(C) or maybe (H)", 4), "C")

    def test_item_aware_extractor(self):
        t = tasks.TASKS["mmlu_pro_eng"]
        four = {"id": "x", "question": "q", "gold": "A", "n_options": 4}
        ten = {"id": "y", "question": "q", "gold": "J", "n_options": 10}
        self.assertIsNone(topologies._extract(t, four, "The answer is (J)."))
        self.assertEqual(topologies._extract(t, ten, "The answer is (J)."), "J")

    def test_score(self):
        s = tasks.TASKS["mmlu_pro_eng"]["score"]
        self.assertTrue(s("J", "J"))
        self.assertFalse(s("I", "J"))
        self.assertFalse(s(None, "J"))


class TestPromptAndData(unittest.TestCase):
    def test_prompt_letter_list(self):
        q4 = "Q?\nA) 1\nB) 2\nC) 3\nD) 4"
        q10 = "Q?\n" + "\n".join(f"{c}) x" for c in "ABCDEFGHIJ")
        self.assertIn("letter (A, B, C, or D) of", tasks.mmlu_pro_prompt(q4))
        self.assertIn("letter (A, B, C, D, E, F, G, H, I, or J) of",
                      tasks.mmlu_pro_prompt(q10))
        # identical wording to MMLU apart from the options
        self.assertEqual(tasks.mmlu_pro_prompt(q4), tasks.mmlu_prompt(q4))

    @unittest.skipUnless(os.path.exists(os.path.join(
        tasks.DATA_DIR, "mmlu_pro_engineering.jsonl")), "data not prepared")
    def test_loaded_items(self):
        items = tasks.TASKS["mmlu_pro_eng"]["load"]()
        self.assertEqual(len(items), 150)
        self.assertEqual(len({i["id"] for i in items}), 150)
        for it in items:
            k = it["n_options"]
            self.assertIn(it["gold"], tasks.MC_LETTERS[:k])
            p = tasks.mmlu_pro_prompt(it["question"])
            self.assertIn(f"or {tasks.MC_LETTERS[k - 1]}) of the correct choice", p)
            # a model echoing the gold letter is scored correct end to end
            pred = tasks.mmlu_pro_extract_item(f"The answer is {it['gold']}.", it)
            self.assertTrue(tasks.mmlu_score(pred, it["gold"]))


class TestMajorityVote(unittest.TestCase):
    def test_majority_over_j_type_letters(self):
        self.assertEqual(topologies._majority(["J", "H", "J", None]), "J")
        self.assertEqual(topologies._majority(["I", "I", "J", "J", "I"]), "I")
        self.assertIsNone(topologies._majority([None, None]))

    def test_sc_votes_with_canned_outputs(self):
        outs = iter(["The answer is (J).", "Answer: H", "the answer is (j).",
                     "\\boxed{J}"])

        def fake_generate(model, prompt, temperature=0.0, seed=None, **kw):
            return {"text": next(outs), "in_tokens": 1, "out_tokens": 1,
                    "latency_s": 0.0}

        task = tasks.TASKS["mmlu_pro_eng"]
        item = {"id": "z", "question": "Q?\n" + "\n".join(
            f"{c}) x" for c in "ABCDEFGHIJ"), "gold": "J", "n_options": 10}
        orig = topologies.client.generate
        topologies.client.generate = fake_generate
        try:
            res = topologies.sc("m", task, item, {"k": 4}, seed=1)
        finally:
            topologies.client.generate = orig
        self.assertEqual(res["raw"], ["J", "H", "J", "J"])
        self.assertEqual(res["answer"], "J")
        self.assertTrue(task["score"](res["answer"], item["gold"]))


if __name__ == "__main__":
    unittest.main()
