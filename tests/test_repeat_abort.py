"""Tests for harness/repeat_abort.py against a local fake Ollama server (no model)."""
import json
import os
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from harness import client, repeat_abort  # noqa: E402

ABORT = json.dumps({"error": "prediction aborted, token repeat limit reached"})
MODE = {"m": "abort"}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        H.seen.append(body)
        if body["options"]["num_predict"] == 1:
            return self._json(200, {"response": "x", "prompt_eval_count": 42, "eval_count": 1})
        if MODE["m"] == "other500":
            return self._raw(500, json.dumps({"error": "something else"}))
        if not body["stream"]:
            if MODE["m"] == "ok":
                return self._json(200, {"response": "The answer is (B).",
                                        "prompt_eval_count": 42, "eval_count": 7})
            return self._raw(500, ABORT)
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()
        for piece in ["The", " value", " is", " 0."] + ["0"] * 31:
            self.wfile.write((json.dumps({"response": piece, "done": False}) + "\n").encode())
        self.wfile.write((ABORT + "\n").encode())

    def _json(self, code, d):
        self._raw(code, json.dumps(d))

    def _raw(self, code, s):
        b = s.encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)


class TestRepeatAbort(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.url = client.OLLAMA_URL
        client.OLLAMA_URL = f"http://127.0.0.1:{cls.srv.server_port}/api/generate"
        cls.sleep = client.time.sleep
        client.time.sleep = lambda s: None

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        client.OLLAMA_URL = cls.url
        client.time.sleep = cls.sleep

    def setUp(self):
        H.seen = []

    def test_abort_recorded(self):
        MODE["m"] = "abort"
        with tempfile.TemporaryDirectory() as d:
            log = os.path.join(d, "a.jsonl")
            os.environ["AGENTS_ABORT_LOG"] = log
            try:
                r = repeat_abort.generate("m", "prompt", temperature=0.7, seed=3)
            finally:
                del os.environ["AGENTS_ABORT_LOG"]
            self.assertTrue(r["aborted"])
            self.assertEqual(r["out_tokens"], 35)
            self.assertEqual(r["in_tokens"], 42)
            self.assertTrue(r["text"].startswith("The value is 0.000"))
            rows = [json.loads(l) for l in open(log)]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["seed"], 3)
        # 3 normal attempts, 1 streamed retry, 1 prompt-count call; same options
        self.assertEqual([b["stream"] for b in H.seen], [False, False, False, True, False])
        self.assertEqual(H.seen[0]["options"], H.seen[3]["options"])
        self.assertEqual(H.seen[0]["options"]["seed"], 3)

    def test_normal_path_unchanged(self):
        MODE["m"] = "ok"
        r = repeat_abort.generate("m", "prompt", seed=1)
        self.assertNotIn("aborted", r)
        self.assertEqual((r["in_tokens"], r["out_tokens"]), (42, 7))
        self.assertEqual(len(H.seen), 1)

    def test_other_500_still_raises(self):
        MODE["m"] = "other500"
        with self.assertRaises(RuntimeError):
            repeat_abort.generate("m", "prompt", seed=1)


if __name__ == "__main__":
    unittest.main()
