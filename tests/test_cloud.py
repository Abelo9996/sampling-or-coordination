"""Offline tests for the vLLM backend and the cloud budget plan (no server, no GPU).
Run: python3 -m pytest tests/test_cloud.py -q   (or python3 tests/test_cloud.py)"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts", "cloud"))

from harness import vllm_client as V  # noqa: E402
import budget as B  # noqa: E402


def test_split_thinking_closed():
    v, r, c = V.split_thinking("<think>\nlet me add\n</think>\n\nThe answer is 5.", None, True, "stop")
    assert v == "The answer is 5." and r == "let me add" and c is True


def test_split_thinking_truncated():
    v, r, c = V.split_thinking("<think>\nstill thinking", None, True, "length")
    assert v == "" and r == "still thinking" and c is False


def test_split_thinking_template_opened():
    # templates that put <think> in the prompt: completion has only the close tag
    v, r, c = V.split_thinking("reasoning here</think>\nThe answer is 7.", None, True, "stop")
    assert v == "The answer is 7." and r == "reasoning here" and c is True


def test_split_reasoning_field():
    v, r, c = V.split_thinking("The answer is 3.", "thought", True, "stop")
    assert (v, r, c) == ("The answer is 3.", "thought", True)
    v, r, c = V.split_thinking("", "thought", True, "length")
    assert (v, c) == ("", False)


def test_non_thinking_passthrough():
    assert V.split_thinking("The answer is 2.", None, None, "stop") == ("The answer is 2.", None, None)


def test_build_body():
    b = V.build_body("m", "p", 0.7, 1003, 4096, None, True)
    assert b["seed"] == 1003 and b["max_tokens"] == 4096 and b["temperature"] == 0.7
    assert b["chat_template_kwargs"] == {"enable_thinking": True}
    assert b["messages"] == [{"role": "user", "content": "p"}]
    assert "chat_template_kwargs" not in V.build_body("m", "p", 0.0, 1, 640, None, None)


def test_parse_response_usage():
    d = {"model": "Qwen/Qwen3-4B", "choices": [{"message": {"content": "<think>x</think>The answer is 4."},
                                                 "finish_reason": "stop"}],
         "usage": {"prompt_tokens": 50, "completion_tokens": 120}}
    r = V.parse_response(d, "Qwen/Qwen3-4B", 0.7, 5, 1.0, True)
    assert r["text"] == "The answer is 4." and r["reasoning"] == "x"
    assert r["in_tokens"] == 50 and r["out_tokens"] == 120  # thinking counts as output


def test_thread_cap():
    V.set_thread_cap(8192)
    assert V.thread_cap() == 8192
    V.set_thread_cap(None)
    assert V.thread_cap() is None


def test_budget_plan_under_cap():
    for spent in (0.0, 5.0, 20.0, 24.0):
        p = B.plan(spent=spent)
        tot = spent + sum(u for _, _, u, _ in p)
        assert tot <= max(spent, B.BUDGET_USD - B.RESERVE_USD) + 1e-6, (spent, p)
    # hard function timeouts stay within the cap with zero prior spend
    worst = sum(B.usd(t + 120) for k, t, _, _ in B.plan() if t > 0) + B.usd(B.SMOKE_TIMEOUT_S + 120)
    assert worst <= B.BUDGET_USD, worst


def test_rate_bound():
    assert abs(B.USD_PER_H - (0.001097 + 4 * 0.0000131 + 40 * 0.00000222) * 3600) < 1e-9


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f()
    print("ok")
