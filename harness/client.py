"""Ollama client with exact token accounting.

Uses only the Python standard library (urllib) so it runs on the system
Python with no extra dependencies. Every call returns the exact input/output
token counts reported by Ollama (`prompt_eval_count` / `eval_count`), which are
the cost-meter primitive for the whole project.
"""
import json
import os
import time
import urllib.request
import urllib.error

# Default to the local Ollama; override with OLLAMA_URL to target a remote host
# (e.g. a Tailscale GPU box). Accepts either a base host URL or a full /api/generate URL.
_OLLAMA_ENV = os.environ.get("OLLAMA_URL", "").strip()
if _OLLAMA_ENV:
    OLLAMA_URL = _OLLAMA_ENV if _OLLAMA_ENV.endswith("/api/generate") \
        else _OLLAMA_ENV.rstrip("/") + "/api/generate"
else:
    OLLAMA_URL = "http://localhost:11434/api/generate"
# Default generation cap; override via env for the Tier-B truncation sensitivity
# run (MATH at a higher cap). Kept at 640 for the consistent v1 results.
DEFAULT_NUM_PREDICT = int(os.environ.get("AGENTS_NUM_PREDICT", "640"))


class CallRecord(dict):
    """A single model call's result + exact cost. Behaves like a dict for JSONL."""
    @property
    def text(self):
        return self["text"]


def generate(model, prompt, temperature=0.0, seed=None, num_predict=None,
             stop=None, timeout=900, retries=2):
    if num_predict is None:
        num_predict = DEFAULT_NUM_PREDICT
    """One model call. Returns a CallRecord with text + exact token counts."""
    body = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": float(temperature), "num_predict": int(num_predict)},
    }
    if seed is not None:
        body["options"]["seed"] = int(seed)
    if stop:
        body["options"]["stop"] = stop
    data = json.dumps(body).encode("utf-8")

    last_err = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(
                OLLAMA_URL, data=data,
                headers={"Content-Type": "application/json"})
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d = json.loads(r.read().decode("utf-8"))
            latency = time.time() - t0
            return CallRecord(
                text=d.get("response", ""),
                in_tokens=int(d.get("prompt_eval_count", 0) or 0),
                out_tokens=int(d.get("eval_count", 0) or 0),
                latency_s=round(latency, 4),
                model=model,
                temperature=float(temperature),
                seed=seed,
            )
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last_err = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Ollama call failed after {retries+1} attempts: {last_err}")


def cost_of(calls):
    """Aggregate a list of CallRecords into total tokens / latency / #calls."""
    return {
        "in_tokens": sum(c["in_tokens"] for c in calls),
        "out_tokens": sum(c["out_tokens"] for c in calls),
        "total_tokens": sum(c["in_tokens"] + c["out_tokens"] for c in calls),
        "latency_s": round(sum(c["latency_s"] for c in calls), 4),
        "n_calls": len(calls),
    }


# Optional OpenAI-compatible backend (e.g. OpenRouter), selected by
# AGENTS_BACKEND=openai. With the variable unset (the default) nothing below
# runs and `generate` is the Ollama function above, unchanged.
if os.environ.get("AGENTS_BACKEND", "").strip().lower() == "openai":
    from . import openai_client as _openai_client
    ollama_generate = generate
    generate = _openai_client.generate
# Self-hosted OpenAI-compatible server (vLLM; Ollama /v1 for local tests),
# selected by AGENTS_BACKEND=vllm. See harness/vllm_client.py.
elif os.environ.get("AGENTS_BACKEND", "").strip().lower() == "vllm":
    from . import vllm_client as _vllm_client
    ollama_generate = generate
    generate = _vllm_client.generate


if __name__ == "__main__":
    import sys
    m = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5:3b-instruct"
    r = generate(m, "What is 17 + 25? Reply with only the number.", temperature=0, seed=1)
    print(r)
