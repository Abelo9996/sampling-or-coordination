"""Tests for the OpenAI-compatible backend (harness/openai_client.py) and the
AGENTS_BACKEND switch in harness/client.py. No network: urllib is replaced by
a fake, and the switch is tested in subprocesses.

Run (stdlib only):  .venv/bin/python -m unittest discover -s tests -v
"""
import hashlib
import inspect
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
import urllib.error

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.pop("AGENTS_BACKEND", None)

from harness import client, tasks, topologies  # noqa: E402
from harness import openai_client as oc  # noqa: E402

# sha256 of harness/client.py at commit c350a6a (before the backend switch was
# added), and of the Ollama generate / cost_of sources in that file.
CLIENT_PY_SHA_C350A6A = "f6f7c96f6d5c61c10ea7df5ff300af834d048cd6b2121e1275aeb82f09e6ca5b"
OLLAMA_GENERATE_SHA = "f2fb47446edc9108900a50d06a2ea0ae5e1f229639ec088726a27e2926ea6501"
COST_OF_SHA = "6de52a035e7590ec8b58fae8840bda6bbd499fecc816247d8b3f02b5411afa56"
SWITCH_START = "# Optional OpenAI-compatible backend"


def _sha(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _completion(text="The answer is 42.", pt=120, ct=30, cost=0.0000123,
                provider="DeepInfra", finish="stop"):
    return {"id": "gen-abc", "model": "meta-llama/llama-3.3-70b-instruct",
            "provider": provider,
            "choices": [{"message": {"role": "assistant", "content": text},
                         "finish_reason": finish}],
            "usage": {"prompt_tokens": pt, "completion_tokens": ct,
                      "total_tokens": pt + ct, "cost": cost}}


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeUrlopen:
    """Sequence of outcomes: dict -> 200 JSON body; (code, body, headers) -> HTTPError."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.requests = []

    def __call__(self, req, timeout=None):
        self.requests.append(req)
        out = self.outcomes.pop(0)
        if isinstance(out, tuple):
            code, body, hdrs = out
            raise urllib.error.HTTPError(req.full_url, code, "err", hdrs,
                                         io.BytesIO(body.encode("utf-8")))
        return _Resp(json.dumps(out).encode("utf-8"))


class OAIBase(unittest.TestCase):
    def setUp(self):
        self._urlopen = oc.urllib.request.urlopen
        self._limiter = oc.LIMITER
        self._cap = oc.SPEND_CAP_USD
        oc.LIMITER = oc._RateLimiter(10_000)
        oc.set_spend(0.0, 0)
        oc.set_ledger(None)
        os.environ["OPENROUTER_API_KEY"] = "sk-test-not-a-real-key"
        os.environ.pop("AGENTS_OPENAI_PROVIDER", None)
        self.sleeps = []

    def tearDown(self):
        oc.urllib.request.urlopen = self._urlopen
        oc.LIMITER = self._limiter
        oc.SPEND_CAP_USD = self._cap
        oc.set_ledger(None)
        os.environ.pop("OPENROUTER_API_KEY", None)
        os.environ.pop("AGENTS_OPENAI_PROVIDER", None)

    def fake(self, outcomes):
        f = FakeUrlopen(outcomes)
        oc.urllib.request.urlopen = f
        return f

    def gen(self, *a, **k):
        k.setdefault("_sleep", self.sleeps.append)
        return oc.generate(*a, **k)


class TestOllamaPathUnchanged(unittest.TestCase):
    def test_client_file_is_old_file_plus_switch_block(self):
        with open(os.path.join(ROOT, "harness", "client.py")) as fh:
            src = fh.read()
        i, j = src.index(SWITCH_START), src.index('if __name__ == "__main__":')
        self.assertEqual(_sha(src[:i] + src[j:]), CLIENT_PY_SHA_C350A6A)

    def test_default_generate_is_ollama_function(self):
        self.assertEqual(client.generate.__module__, "harness.client")
        self.assertEqual(_sha(inspect.getsource(client.generate)), OLLAMA_GENERATE_SHA)
        self.assertEqual(_sha(inspect.getsource(client.cost_of)), COST_OF_SHA)
        self.assertFalse(hasattr(client, "ollama_generate"))

    def _probe(self, backend):
        env = dict(os.environ)
        env.pop("AGENTS_BACKEND", None)
        if backend is not None:
            env["AGENTS_BACKEND"] = backend
        code = ("import sys; sys.path.insert(0, %r); from harness import client;"
                "print(client.generate.__module__)" % ROOT)
        return subprocess.run([sys.executable, "-c", code], env=env, check=True,
                              capture_output=True, text=True).stdout.strip()

    def test_switch_by_env(self):
        self.assertEqual(self._probe(None), "harness.client")
        self.assertEqual(self._probe(""), "harness.client")
        self.assertEqual(self._probe("ollama"), "harness.client")
        self.assertEqual(self._probe("openai"), "harness.openai_client")
        self.assertEqual(self._probe("OpenAI"), "harness.openai_client")


class TestRequestAndParse(OAIBase):
    def test_body(self):
        b = oc.build_body("m", "PROMPT", 0.7, 1003, 640, None)
        self.assertEqual(b["model"], "m")
        self.assertEqual(b["messages"], [{"role": "user", "content": "PROMPT"}])
        self.assertEqual(b["temperature"], 0.7)
        self.assertEqual(b["max_tokens"], 640)
        self.assertEqual(b["seed"], 1003)
        self.assertEqual(b["usage"], {"include": True})
        self.assertNotIn("stop", b)
        self.assertNotIn("provider", b)
        self.assertNotIn("seed", oc.build_body("m", "p", 0.0, None, 640, None))

    def test_provider_prefs_from_env(self):
        prefs = {"sort": "price", "max_price": {"prompt": 0.25, "completion": 0.6}}
        os.environ["AGENTS_OPENAI_PROVIDER"] = json.dumps(prefs)
        self.assertEqual(oc.build_body("m", "p", 0, 1, 640, None)["provider"], prefs)

    def test_parse(self):
        r = oc.parse_response(_completion(), "m", 0.0, 1, 1.5)
        self.assertEqual(r["text"], "The answer is 42.")
        self.assertEqual((r["in_tokens"], r["out_tokens"]), (120, 30))
        self.assertEqual(r["provider"], "DeepInfra")
        self.assertEqual(r["served_model"], "meta-llama/llama-3.3-70b-instruct")
        self.assertEqual(r["finish_reason"], "stop")
        self.assertAlmostEqual(r["cost_usd"], 0.0000123)
        self.assertEqual(r["model"], "m")
        self.assertEqual(r.text, "The answer is 42.")

    def test_parse_errors(self):
        with self.assertRaises(RuntimeError):
            oc.parse_response({"error": {"code": 502}}, "m", 0, 1, 0)
        with self.assertRaises(RuntimeError):
            oc.parse_response({"choices": []}, "m", 0, 1, 0)
        d = _completion()
        del d["usage"]
        with self.assertRaises(RuntimeError):
            oc.parse_response(d, "m", 0, 1, 0)

    def test_cost_of_unchanged_semantics(self):
        a = oc.parse_response(_completion(pt=100, ct=20), "m", 0, 1, 1.0)
        b = oc.parse_response(_completion(pt=300, ct=50), "m", 0, 1, 2.0)
        c = client.cost_of([a, b])
        self.assertEqual(c, {"in_tokens": 400, "out_tokens": 70, "total_tokens": 470,
                             "latency_s": 3.0, "n_calls": 2})


class TestGenerate(OAIBase):
    def test_success_and_headers(self):
        f = self.fake([_completion()])
        r = self.gen("model-x", "PROMPT", temperature=0.0, seed=1)
        self.assertEqual(r["in_tokens"], 120)
        req = f.requests[0]
        self.assertTrue(req.full_url.endswith("/chat/completions"))
        self.assertEqual(req.get_header("Authorization"), "Bearer sk-test-not-a-real-key")
        body = json.loads(req.data)
        self.assertEqual(body["max_tokens"], client.DEFAULT_NUM_PREDICT)
        self.assertNotIn("sk-test", json.dumps(dict(r)))

    def test_429_backoff_then_success(self):
        self.fake([(429, '{"error":"rate limited"}', {"Retry-After": "7"}),
                   (429, '{"error":"rate limited"}', {}),
                   _completion()])
        r = self.gen("m", "p", seed=1)
        self.assertEqual(r["out_tokens"], 30)
        self.assertEqual(self.sleeps[0], 7.0)
        self.assertEqual(self.sleeps[1], 10.0)   # 5 * 2**1

    def test_daily_limit_sleeps_until_reset(self):
        self.fake([(429, '{"error":{"message":"Rate limit exceeded: '
                         'free-models-per-day"}}', {}),
                   _completion()])
        self.gen("m:free", "p", seed=1, retries=0)  # daily wait uses no retry
        self.assertGreater(self.sleeps[0], 300)
        self.assertLessEqual(self.sleeps[0], 86400 + 300)

    def test_nonretryable(self):
        self.fake([(404, '{"error":"No endpoints found"}', {})])
        with self.assertRaises(oc.NonRetryableError):
            self.gen("m:free", "p")

    def test_5xx_and_bad_body_retry(self):
        self.fake([(502, "bad gateway", {}), {"error": {"code": 500}}, _completion()])
        r = self.gen("m", "p")
        self.assertEqual(r["in_tokens"], 120)
        self.assertEqual(len(self.sleeps), 2)

    def test_gives_up(self):
        self.fake([(503, "x", {})] * 3)
        with self.assertRaises(RuntimeError):
            self.gen("m", "p", retries=2)

    def test_empty_generation_retried(self):
        self.fake([_completion(text="", ct=0), _completion()])
        r = self.gen("m", "p")
        self.assertEqual(r["text"], "The answer is 42.")

    def test_spend_cap_and_ledger(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ledger.jsonl")
            oc.set_ledger(path)
            oc.SPEND_CAP_USD = 0.00002
            self.fake([_completion(cost=0.000015), _completion(cost=0.000015)])
            self.gen("m", "p")
            self.gen("m", "p")                        # 0.000015 < cap: allowed
            self.assertAlmostEqual(oc.SPEND["usd"], 0.00003)
            with self.assertRaises(oc.SpendCapReached):
                self.gen("m", "p")                    # cap reached: no request
            with open(path) as fh:
                lines = [json.loads(l) for l in fh]
            self.assertEqual(len(lines), 2)
            self.assertNotIn("text", lines[0])
            self.assertEqual(lines[0]["provider"], "DeepInfra")
            self.assertAlmostEqual(lines[1]["cum_usd"], 0.00003)

    def test_rate_limiter(self):
        t = [0.0]
        slept = []

        def sleep(s):
            slept.append(s)
            t[0] += s
        lim = oc._RateLimiter(2)
        for _ in range(3):
            lim.wait(clock=lambda: t[0], sleep=sleep)
        self.assertEqual(len(slept), 1)
        self.assertGreaterEqual(t[0], 60.0)

    def test_seconds_until_reset(self):
        import datetime as dt
        now = dt.datetime(2026, 10, 11, 23, 0, 0, tzinfo=dt.timezone.utc)
        self.assertEqual(oc._seconds_until_utc_reset(now, margin_s=0), 3600)


class TestTopologiesOverOpenAI(OAIBase):
    """The unchanged topologies drive the API client with the harness's
    temperatures, seeds, and output cap."""

    def setUp(self):
        super().setUp()
        self._gen = client.generate
        client.generate = lambda *a, **k: oc.generate(*a, _sleep=self.sleeps.append, **k)
        self.task = tasks.get_task("gsm8k")
        self.item = {"id": "gsm8k-x", "question": "What is 40 + 2?", "gold": 42}

    def tearDown(self):
        client.generate = self._gen
        super().tearDown()

    def bodies(self, f):
        return [json.loads(r.data) for r in f.requests]

    def test_sa(self):
        f = self.fake([_completion()])
        res = topologies.sa("m", self.task, self.item, {}, 1)
        self.assertEqual(res["answer"], 42)
        b = self.bodies(f)[0]
        self.assertEqual((b["temperature"], b["seed"], b["max_tokens"]), (0.0, 1, 640))
        self.assertEqual(b["messages"][0]["content"], self.task["prompt"](self.item["question"]))

    def test_sc(self):
        f = self.fake([_completion("The answer is 41."), _completion(), _completion(),
                       _completion("The answer is 7.")])
        res = topologies.sc("m", self.task, self.item, {"k": 4}, 1)
        self.assertEqual(res["answer"], 42)
        bs = self.bodies(f)
        self.assertEqual([b["seed"] for b in bs], [1000, 1001, 1002, 1003])
        self.assertTrue(all(b["temperature"] == 0.7 for b in bs))

    def test_debate(self):
        outs = [_completion("The answer is 41."), _completion(), _completion(),
                _completion(), _completion(), _completion("The answer is 5.")]
        f = self.fake(outs)
        res = topologies.debate("m", self.task, self.item, {"N": 3, "R": 1}, 1)
        self.assertEqual(res["answer"], 42)
        bs = self.bodies(f)
        self.assertEqual([b["seed"] for b in bs], [1000, 1001, 1002, 1100, 1101, 1102])
        self.assertIn("Here are other agents' solutions", bs[3]["messages"][0]["content"])
        self.assertEqual(len(res["calls"]), 6)
        self.assertEqual(client.cost_of(res["calls"])["total_tokens"], 6 * 150)
        self.assertTrue(all(c["provider"] == "DeepInfra" for c in res["calls"]))


if __name__ == "__main__":
    unittest.main()
