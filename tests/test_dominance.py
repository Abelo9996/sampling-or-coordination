"""Tests for soc/dominance.py (no model call).
  python3 -m unittest tests.test_dominance -v
TestToy: small synthetic cases (instant). TestPaper: the released logs give the paper's values
(GSM8K 500 x 3 cost-matched dominance test under total tokens; break-even price ratios of
SC k=8 against debate on GSM8K and MATH levels 1 to 3), about a minute."""
import importlib.util
import json
import os
import re
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from soc.dominance import break_even_price_ratio, cost_matched_dominance, hier_paired_bootstrap  # noqa: E402


def _mod(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", name + ".py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _macros():
    out = {}
    for f in ("rev3_reruns_macros.tex", "rev3_macros.tex"):
        for m in re.finditer(r"\\newcommand\{\\(\w+)\}\{(.*)\}", open(os.path.join(ROOT, "paper/arxiv_rev3", f)).read()):
            out[m.group(1)] = m.group(2)
    return out


class TestToy(unittest.TestCase):
    def test_rule(self):
        rng = np.random.default_rng(1)
        n = 300
        good = (rng.random((n, 3)) < 0.9).astype(float)
        bad = (rng.random((n, 3)) < 0.6).astype(float)
        R = cost_matched_dominance(bad, 1000.0, {"cheap_good": good}, {"cheap_good": 500.0}, B=2000)
        self.assertEqual((R["verdict"], R["dominator"]), ("R", "cheap_good"))
        N = cost_matched_dominance(good, 1000.0, {"cheap_bad": bad}, {"cheap_bad": 500.0}, B=2000)
        self.assertEqual(N["verdict"], "N")
        C = cost_matched_dominance(good, 1000.0, {"cheap_same": good}, {"cheap_same": 500.0}, B=2000)
        self.assertEqual(C["verdict"], "C")
        none = cost_matched_dominance(bad, 100.0, {"dear": good}, {"dear": 500.0}, B=2000)
        self.assertEqual((none["verdict"], none["cheaper"]), ("N", []))

    def test_break_even(self):
        r = break_even_price_ratio(np.full(50, 100.0), np.full(50, 50.0), np.full(50, 300.0), np.full(50, 30.0), B=500)
        self.assertAlmostEqual(r["point"], 10.0)  # (300 - 100) / (50 - 30)
        r = break_even_price_ratio(np.full(50, 100.0), np.full(50, 20.0), np.full(50, 300.0), np.full(50, 30.0), B=500)
        self.assertEqual((r["point"], r["share_infinite"]), (np.inf, 1.0))  # cheaper at every ratio

    def test_matches_harness(self):
        from harness import stats
        rng = np.random.default_rng(2)
        a = rng.integers(0, 2, (120, 3)).astype(float)
        b = rng.integers(0, 2, (120, 3)).astype(float)
        ref = stats.hier_paired_test(dict(enumerate(a)), dict(enumerate(b)), B=300, seed=0)
        got = hier_paired_bootstrap(a, b, B=300, seed=0)
        self.assertEqual(ref, (got["diff"], got["p"], *got["ci95"]))


class TestPaper(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.M = _macros()

    def test_gsm8k_dominance(self):
        rr = _mod("analyze_rev3_reruns")
        from harness import tasks
        g = tasks.get_task("gsm8k")
        S3 = [1, 2, 3]
        ids = [it["id"] for it in g["load"](n=500, split="all")]
        PA = rr.load(["results/power/gsm8k_qwen3b_s[123].jsonl", "results/power/gsm8k_qwen3b_s[123]_sh*of2.jsonl"])
        R1 = rr.load(["results/rev3/gsm8k_qwen3b_s*.jsonl"])
        cells = {"SA": rr.cell_from(PA, "SA", ids, S3)}
        cells.update(rr.sc_prefix_cells(PA, "SC_k8", ids, S3, g, [2, 4, 8]))
        for w in ("DEBATE_N3_R1", "GE_R2"):
            cells[w] = rr.cell_from(PA, w, ids, S3)
        cells["HIERFIX_M3"] = rr.cell_from(R1, "HIERFIX_M3", ids, S3)
        order = sorted(ids)  # harness.stats.hier_paired_test resamples items in sorted-id order
        acc = {c: np.stack([v.M[i] for i in order]) for c, v in cells.items()}
        tot = {c: np.array([v.tot[i] for i in order]) for c, v in cells.items()}
        with open(os.path.join(ROOT, "results/extra/rev3_reruns_analysis.json")) as f:
            J = json.load(f)["tasks"]["gsm8k"]
        samp = ["SA", "SC_k2", "SC_k4", "SC_k8"]
        expect = {"DEBATE_N3_R1": ("R", "SC_k4"), "GE_R2": ("R", "SC_k4"), "HIERFIX_M3": ("R", "SA")}
        for w, (v, dom) in expect.items():
            res = cost_matched_dominance(acc[w], tot[w], {s: acc[s] for s in samp}, {s: tot[s] for s in samp})
            self.assertEqual((res["verdict"], res["dominator"]), (v, dom), w)
            self.assertEqual(res["verdict"], J["verdict_total"][w]["verdict"])
            self.assertEqual(res["robust_by"], J["verdict_total"][w]["robust_by"])
            for s, c in res["comparisons"].items():
                t = J["tests"][f"{w}|{s}"]  # workflow minus sampling, stored rounded to 2 places
                self.assertAlmostEqual(-c["adv"], t["diff"], places=2)
                self.assertEqual([-c["ci95"][1], -c["ci95"][0]], t["ci95"])
                self.assertEqual(c["p"], t["p"])
        # The paper's printed values (Table 2): SC k=4 minus debate +4.2 [+2.1,+6.2].
        c = cost_matched_dominance(acc["DEBATE_N3_R1"], tot["DEBATE_N3_R1"], {"SC_k4": acc["SC_k4"]},
                                   {"SC_k4": tot["SC_k4"]})["comparisons"]["SC_k4"]
        self.assertEqual(f"$[{c['ci95'][0]:+.1f},{c['ci95'][1]:+.1f}]$", self.M["RrIGsmScfourMinusDeb"])
        self.assertEqual(f"${c['adv']:+.1f}$", self.M["RrDGsmScfourMinusDeb"])

    def test_break_even_price_ratio(self):
        co = _mod("analyze_rev3_costs")
        from harness import tasks
        S3 = [1, 2, 3]
        gids = [it["id"] for it in tasks.get_task("gsm8k")["load"](n=500, split="all")]
        PA = co.load(["results/power/gsm8k_qwen3b_s[123].jsonl", "results/power/gsm8k_qwen3b_s[123]_sh*of2.jsonl"])
        mids = [it["id"] for it in tasks.get_task("math500")["load"](n=238, split="all")]
        PB = co.load(["results/power/math500_qwen3b_s*.jsonl"])
        with open(os.path.join(ROOT, "results/extra/rev3_costs_analysis.json")) as f:
            J = json.load(f)["crossover"]
        runs = {"gsm8k": (PA, gids, "SC_k8", ("RrCrossGsm", "RrCrossCIGsm")),
                "math": (PB, mids, "SC_k12", ("RthCrossMath", "RrCrossCIMath"))}
        for name, (recs, ids, src, (mp, mci)) in runs.items():
            sc = co.costs(recs, "SC_k8", ids, S3, k=8, src=src)
            deb = co.costs(recs, "DEBATE_N3_R1", ids, S3)
            arr = lambda d, j: np.stack([d[i][:, j] for i in ids])
            r = break_even_price_ratio(arr(sc, 0), arr(sc, 1), arr(deb, 0), arr(deb, 1))
            self.assertAlmostEqual(r["point"], J[name]["SC_k8"]["point"], places=9)
            for x, y in zip(r["ci95"], J[name]["SC_k8"]["ci95"]):
                self.assertAlmostEqual(x, y, places=9)
            self.assertEqual(f"{r['point']:.2f}", self.M[mp])
            self.assertEqual(f"$[{r['ci95'][0]:.1f},{r['ci95'][1]:.1f}]$", self.M[mci])
            sc4 = co.costs(recs, "SC_k4", ids, S3, k=4, src=src)
            r4 = break_even_price_ratio(arr(sc4, 0), arr(sc4, 1), arr(deb, 0), arr(deb, 1))
            self.assertEqual((r4["point"], r4["share_infinite"]), (np.inf, 1.0))  # SC k=4 cheaper at every ratio


if __name__ == "__main__":
    unittest.main()
