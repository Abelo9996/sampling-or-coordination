"""OpenAI-compatible chat-completions client (e.g. OpenRouter) with exact token
accounting, as a drop-in alternative to the Ollama `client.generate`.

Selected by `AGENTS_BACKEND=openai` (see the switch at the end of client.py);
with that variable unset the harness uses Ollama exactly as before. Standard
library only.

`generate` has the same signature and returns the same CallRecord keys as the
Ollama client (text, in_tokens, out_tokens, latency_s, model, temperature,
seed), so `client.cost_of` and every topology work unchanged. The token counts
are the provider-reported `usage.prompt_tokens` / `usage.completion_tokens`,
the API analogue of Ollama's `prompt_eval_count` / `eval_count` (both count the
full templated prompt on every call). Extra keys record the serving provider,
the served model id, the finish reason, and the cost reported by the API.

The harness prompt is sent as the single user message, with no system prompt,
matching Ollama's /api/generate, which wraps the prompt as the user turn of the
model's chat template.

Environment:
  OPENAI_BASE_URL        default https://openrouter.ai/api/v1
  OPENAI_API_KEY or OPENROUTER_API_KEY   bearer token (never logged)
  AGENTS_OPENAI_RPM      max requests per rolling 60 s, all threads (default 18)
  AGENTS_OPENAI_PROVIDER optional JSON object sent as OpenRouter `provider`
  AGENTS_OPENAI_SPEND_CAP_USD  optional hard cap on cumulative reported cost
"""
import datetime as _dt
import json
import os
import threading
import time
import urllib.error
import urllib.request

# CallRecord / DEFAULT_NUM_PREDICT come from client.py; imported inside the
# functions so that importing either module first works (client.py imports
# this module at load time when AGENTS_BACKEND=openai).

BASE_URL = os.environ.get("OPENAI_BASE_URL",
                          "https://openrouter.ai/api/v1").strip().rstrip("/")
RPM = int(os.environ.get("AGENTS_OPENAI_RPM", "18"))


class SpendCapReached(RuntimeError):
    """Raised before a request would be sent once the spend cap is reached."""


class NonRetryableError(RuntimeError):
    """HTTP 400/401/402/403/404: the request itself cannot succeed."""


def _api_key():
    k = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if not k:
        raise RuntimeError("no OPENAI_API_KEY / OPENROUTER_API_KEY in environment")
    return k.strip()


def _provider_prefs():
    raw = os.environ.get("AGENTS_OPENAI_PROVIDER", "").strip()
    return json.loads(raw) if raw else None


# --------------------------------------------------------------------------
# global rate limiter and spend meter (thread-safe)
# --------------------------------------------------------------------------
class _RateLimiter:
    def __init__(self, per_minute):
        self.per_minute = max(1, int(per_minute))
        self.stamps = []
        self.lock = threading.Lock()

    def wait(self, clock=time.time, sleep=time.sleep):
        while True:
            with self.lock:
                now = clock()
                self.stamps = [t for t in self.stamps if now - t < 60.0]
                if len(self.stamps) < self.per_minute:
                    self.stamps.append(now)
                    return
                delay = 60.0 - (now - self.stamps[0]) + 0.05
            sleep(max(delay, 0.05))


LIMITER = _RateLimiter(RPM)

_SPEND_LOCK = threading.Lock()
SPEND = {"usd": 0.0, "calls": 0}
_cap_env = os.environ.get("AGENTS_OPENAI_SPEND_CAP_USD", "").strip()
SPEND_CAP_USD = float(_cap_env) if _cap_env else None
_LEDGER = {"path": None}


def set_spend(usd, calls=0):
    """Initialise the cumulative spend (e.g. from a ledger on resume)."""
    with _SPEND_LOCK:
        SPEND["usd"] = float(usd)
        SPEND["calls"] = int(calls)


def set_ledger(path):
    """Append one JSON line per successful call (cost, tokens, provider; no
    prompt or response text, no credentials)."""
    _LEDGER["path"] = path


def _check_cap():
    if SPEND_CAP_USD is not None:
        with _SPEND_LOCK:
            if SPEND["usd"] >= SPEND_CAP_USD:
                raise SpendCapReached(
                    f"spend {SPEND['usd']:.6f} >= cap {SPEND_CAP_USD:.2f} USD")


def _record_spend(rec):
    with _SPEND_LOCK:
        SPEND["usd"] += float(rec.get("cost_usd") or 0.0)
        SPEND["calls"] += 1
        if _LEDGER["path"]:
            line = {"ts": round(time.time(), 3), "gen_id": rec.get("gen_id"),
                    "served_model": rec.get("served_model"),
                    "provider": rec.get("provider"),
                    "in_tokens": rec["in_tokens"], "out_tokens": rec["out_tokens"],
                    "cost_usd": rec.get("cost_usd"),
                    "cum_usd": round(SPEND["usd"], 8)}
            with open(_LEDGER["path"], "a") as f:
                f.write(json.dumps(line) + "\n")


# --------------------------------------------------------------------------
def _seconds_until_utc_reset(now=None, margin_s=300):
    """Seconds until 00:00 UTC of the next day plus a margin (free-tier daily
    request counters reset at 00:00 UTC)."""
    now = now or _dt.datetime.now(_dt.timezone.utc)
    nxt = (now + _dt.timedelta(days=1)).replace(hour=0, minute=0, second=0,
                                                microsecond=0)
    return (nxt - now).total_seconds() + margin_s


def _is_daily_limit(body_text, headers):
    t = (body_text or "").lower()
    if "per-day" in t or "per day" in t or "daily" in t:
        return True
    rem = headers.get("X-RateLimit-Remaining") if headers else None
    reset = headers.get("X-RateLimit-Reset") if headers else None
    if rem == "0" and reset:
        try:
            return float(reset) / 1000.0 - time.time() > 3600
        except ValueError:
            return False
    return False


def build_body(model, prompt, temperature, seed, num_predict, stop):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": float(temperature),
        "max_tokens": int(num_predict),
        "stream": False,
        "usage": {"include": True},
    }
    if seed is not None:
        body["seed"] = int(seed)
    if stop:
        body["stop"] = stop
    prov = _provider_prefs()
    if prov:
        body["provider"] = prov
    return body


def parse_response(d, model, temperature, seed, latency):
    """Map an OpenAI-style chat-completion JSON to a CallRecord."""
    from .client import CallRecord
    if d.get("error"):
        raise RuntimeError(f"API error in body: {str(d['error'])[:300]}")
    choices = d.get("choices") or []
    if not choices:
        raise RuntimeError("API response has no choices")
    ch = choices[0]
    msg = ch.get("message") or {}
    text = msg.get("content") or ""
    u = d.get("usage") or {}
    if "prompt_tokens" not in u or "completion_tokens" not in u:
        raise RuntimeError("API response has no usage token counts")
    cost = u.get("cost")
    return CallRecord(
        text=text,
        in_tokens=int(u.get("prompt_tokens") or 0),
        out_tokens=int(u.get("completion_tokens") or 0),
        latency_s=round(latency, 4),
        model=model,
        temperature=float(temperature),
        seed=seed,
        served_model=d.get("model"),
        provider=d.get("provider"),
        finish_reason=ch.get("finish_reason") or ch.get("native_finish_reason"),
        cost_usd=float(cost) if cost is not None else None,
        gen_id=d.get("id"),
    )


def generate(model, prompt, temperature=0.0, seed=None, num_predict=None,
             stop=None, timeout=300, retries=8, _sleep=time.sleep):
    """One chat-completion call. Same contract as client.generate."""
    from .client import DEFAULT_NUM_PREDICT
    if num_predict is None:
        num_predict = DEFAULT_NUM_PREDICT
    body = build_body(model, prompt, temperature, seed, num_predict, stop)
    data = json.dumps(body).encode("utf-8")
    url = BASE_URL + "/chat/completions"
    last_err = None
    attempt = 0
    empty_retries = 0
    while attempt <= retries:
        _check_cap()
        LIMITER.wait()
        try:
            req = urllib.request.Request(url, data=data, headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer " + _api_key(),
                "X-Title": "compute-optimal-agents"})
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d = json.loads(r.read().decode("utf-8"))
            rec = parse_response(d, model, temperature, seed, time.time() - t0)
            _record_spend(rec)
            if not rec["text"].strip() and rec["out_tokens"] == 0 and empty_retries < 3:
                # empty generation: infrastructure failure, retry (cost counted)
                empty_retries += 1
                _sleep(2.0 * empty_retries)
                continue
            return rec
        except urllib.error.HTTPError as e:
            try:
                txt = e.read().decode("utf-8", "replace")
            except Exception:  # noqa: BLE001
                txt = ""
            last_err = f"HTTP {e.code}: {txt[:300]}"
            if e.code == 429:
                if _is_daily_limit(txt, e.headers):
                    _sleep(_seconds_until_utc_reset())
                    continue  # daily reset does not consume a retry
                ra = e.headers.get("Retry-After") if e.headers else None
                try:
                    delay = float(ra)
                except (TypeError, ValueError):
                    delay = min(300.0, 5.0 * (2 ** attempt))
                _sleep(delay)
                attempt += 1
                continue
            if e.code in (400, 401, 402, 403, 404):
                raise NonRetryableError(last_err)
            _sleep(min(300.0, 5.0 * (2 ** attempt)))
            attempt += 1
        except (urllib.error.URLError, TimeoutError, ConnectionError,
                json.JSONDecodeError, RuntimeError) as e:
            if isinstance(e, (SpendCapReached, NonRetryableError)):
                raise
            last_err = f"{type(e).__name__}: {str(e)[:300]}"
            _sleep(min(300.0, 5.0 * (2 ** attempt)))
            attempt += 1
    raise RuntimeError(f"API call failed after {retries + 1} attempts: {last_err}")


def key_info():
    """OpenRouter /key metadata (usage counters, free-tier daily requests)."""
    req = urllib.request.Request(BASE_URL + "/key",
                                 headers={"Authorization": "Bearer " + _api_key()})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8")).get("data", {})
