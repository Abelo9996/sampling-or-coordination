"""Self-hosted OpenAI-compatible chat backend (vLLM server; also Ollama's /v1
endpoint for local tests), as a drop-in for the Ollama `client.generate`.

Selected by `AGENTS_BACKEND=vllm` (switch at the end of client.py). With the
variable unset the harness uses Ollama exactly as before. Standard library only,
thread safe (the cloud driver runs many work units in parallel threads).

Same contract as client.generate: one user message (no system prompt, as with
Ollama's /api/generate), per-request `temperature`, `seed`, `max_tokens`, and a
CallRecord with text, in_tokens, out_tokens, latency_s, model, temperature, seed.
Token counts are the server's `usage.prompt_tokens` / `usage.completion_tokens`;
for a thinking model completion_tokens includes the thinking tokens, so
thinking counts as output.

Thinking models (AGENTS_THINKING=on): the request carries
`chat_template_kwargs: {"enable_thinking": true}` (Qwen3 hybrid template). The
server is run WITHOUT a reasoning parser; the client splits the raw completion
itself (split_thinking): everything up to the first `</think>` is the reasoning,
the rest is the visible text. CallRecord["text"] is the visible text only, so the
unchanged extractors and every downstream agent (debate peers, GE critic and
reviser, HIER executors and aggregator) see only the final visible answer and
explanation, never the think block. The reasoning is kept in
CallRecord["reasoning"] for the raw log. A completion that stops inside the
think block (cap reached) has visible text "" (no answer) and
think_closed=False. If a server does parse reasoning into
`message.reasoning_content` or `message.reasoning` (e.g. Ollama), that field is
used as the reasoning and `content` as the visible text.

Environment:
  AGENTS_VLLM_URL      base URL with /v1, default http://127.0.0.1:8000/v1
  AGENTS_VLLM_API_KEY  optional bearer token (never logged)
  AGENTS_THINKING      on | off | "" (omit chat_template_kwargs; default)
  AGENTS_VLLM_TIMEOUT  per-request timeout in seconds, default 1800
"""
import json
import os
import threading
import time
import urllib.error
import urllib.request

BASE_URL = os.environ.get("AGENTS_VLLM_URL", "http://127.0.0.1:8000/v1").strip().rstrip("/")
THINKING = os.environ.get("AGENTS_THINKING", "").strip().lower()
TIMEOUT = float(os.environ.get("AGENTS_VLLM_TIMEOUT", "1800"))

_TL = threading.local()


class NonRetryableError(RuntimeError):
    """HTTP 400/401/403/404/422: the request itself cannot succeed."""


def set_thread_cap(cap):
    """Per-thread output cap used when the caller passes no num_predict (the
    topologies never do). Lets one process run tasks with different caps."""
    _TL.cap = int(cap) if cap else None


def thread_cap():
    return getattr(_TL, "cap", None)


def thinking_flag(value=None):
    v = THINKING if value is None else (value or "").strip().lower()
    if v in ("on", "1", "true", "yes"):
        return True
    if v in ("off", "0", "false", "no"):
        return False
    return None


def build_body(model, prompt, temperature, seed, num_predict, stop, thinking=None):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": float(temperature),
        "max_tokens": int(num_predict),
        "stream": False,
    }
    if seed is not None:
        body["seed"] = int(seed)
    if stop:
        body["stop"] = stop
    if thinking is not None:
        body["chat_template_kwargs"] = {"enable_thinking": bool(thinking)}
    return body


def split_thinking(content, reasoning_field=None, thinking=None, finish_reason=None):
    """(visible, reasoning, think_closed). See the module docstring."""
    content = content or ""
    if reasoning_field:
        closed = not (finish_reason == "length" and not content.strip())
        return content, reasoning_field, closed
    if thinking is not True and "<think>" not in content and "</think>" not in content:
        return content, None, None
    if "</think>" in content:
        r, v = content.split("</think>", 1)
        r = r.replace("<think>", "", 1).strip("\n")
        return v.lstrip("\n"), r, True
    if content.lstrip().startswith("<think>") or finish_reason == "length":
        # stopped inside the think block: no visible answer
        return "", content.replace("<think>", "", 1).strip("\n"), False
    # thinking requested but the model emitted no think block at all
    return content, None, None


def parse_response(d, model, temperature, seed, latency, thinking=None):
    from .client import CallRecord
    if d.get("error"):
        raise RuntimeError(f"API error in body: {str(d['error'])[:300]}")
    choices = d.get("choices") or []
    if not choices:
        raise RuntimeError("API response has no choices")
    ch = choices[0]
    msg = ch.get("message") or {}
    fr = ch.get("finish_reason")
    rf = msg.get("reasoning_content") or msg.get("reasoning")
    visible, reasoning, closed = split_thinking(msg.get("content"), rf, thinking, fr)
    u = d.get("usage") or {}
    if "prompt_tokens" not in u or "completion_tokens" not in u:
        raise RuntimeError("API response has no usage token counts")
    return CallRecord(
        text=visible,
        in_tokens=int(u.get("prompt_tokens") or 0),
        out_tokens=int(u.get("completion_tokens") or 0),
        latency_s=round(latency, 4),
        model=model,
        temperature=float(temperature),
        seed=seed,
        reasoning=reasoning,
        think_closed=closed,
        finish_reason=fr,
        served_model=d.get("model"),
    )


def generate(model, prompt, temperature=0.0, seed=None, num_predict=None,
             stop=None, timeout=None, retries=6, _sleep=time.sleep):
    """One chat-completion call. Same contract as client.generate."""
    from .client import DEFAULT_NUM_PREDICT
    if num_predict is None:
        num_predict = thread_cap() or DEFAULT_NUM_PREDICT
    thinking = thinking_flag()
    body = build_body(model, prompt, temperature, seed, num_predict, stop, thinking)
    data = json.dumps(body).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    key = os.environ.get("AGENTS_VLLM_API_KEY", "").strip()
    if key:
        headers["Authorization"] = "Bearer " + key
    url = BASE_URL + "/chat/completions"
    last_err = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, data=data, headers=headers)
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=timeout or TIMEOUT) as r:
                d = json.loads(r.read().decode("utf-8"))
            return parse_response(d, model, temperature, seed, time.time() - t0, thinking)
        except urllib.error.HTTPError as e:
            try:
                txt = e.read().decode("utf-8", "replace")
            except Exception:  # noqa: BLE001
                txt = ""
            last_err = f"HTTP {e.code}: {txt[:300]}"
            if e.code in (400, 401, 403, 404, 422):
                raise NonRetryableError(last_err)
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError,
                json.JSONDecodeError, RuntimeError) as e:
            last_err = f"{type(e).__name__}: {str(e)[:300]}"
        _sleep(min(60.0, 2.0 * (2 ** attempt)))
    raise RuntimeError(f"vLLM call failed after {retries + 1} attempts: {last_err}")
