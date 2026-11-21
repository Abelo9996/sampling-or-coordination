"""Handling of Ollama's "token repeat limit" abort, for the MMLU-Pro engineering run.

Ollama stops a generation whose recent streamed pieces are all identical and
answers HTTP 500 {"error":"prediction aborted, token repeat limit reached"}
(the string is in both 0.30.10, used for the earlier results, and 0.40.1).
`client.generate` retries an HTTP 500 three times and then raises RuntimeError,
so a single such call stops a run. The GSM8K, MMLU and MATH runs completed, so
it never fired there; on MMLU-Pro engineering it fires on generations that loop
(the first case was a long run of zeros in a unit conversion), and because the
call is deterministic for a fixed seed, a resumed run stops on it again.

`install()` wraps `client.generate` without changing it: every call goes
through the unchanged function first. Only when that raises after an HTTP 500
is the same request body sent once more with "stream": true, and

  * if the stream ends with the repeat-limit error, the call is recorded as
    what the runtime produced: text = the pieces streamed before the abort,
    out_tokens = the number of streamed pieces (Ollama streams one token per
    piece), in_tokens = prompt_eval_count of a 1-token call on the same prompt
    (that count does not depend on the KV cache: checked on four prompts);
    the record carries aborted=True and is appended to $AGENTS_ABORT_LOG;
  * if the stream completes, its final counts are used (a normal call);
  * any other error re-raises the original RuntimeError.

Used only by scripts/run_mmlu_pro_eng.py; harness/client.py is untouched.
"""
import hashlib
import json
import os
import time
import urllib.error
import urllib.request

from . import client

REPEAT_ABORT = "token repeat limit reached"
_orig_generate = client.generate


def _body(model, prompt, temperature, seed, num_predict, stop):
    # identical to the body built in client.generate
    if num_predict is None:
        num_predict = client.DEFAULT_NUM_PREDICT
    body = {"model": model, "prompt": prompt, "stream": False,
            "options": {"temperature": float(temperature),
                        "num_predict": int(num_predict)}}
    if seed is not None:
        body["options"]["seed"] = int(seed)
    if stop:
        body["options"]["stop"] = stop
    return body


def _req(body):
    return urllib.request.Request(
        client.OLLAMA_URL, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"})


def _streamed(body, model, temperature, seed, timeout):
    """Returns (CallRecord or None, error string or None)."""
    text, pieces, err = [], 0, None
    t0 = time.time()
    with urllib.request.urlopen(_req(dict(body, stream=True)), timeout=timeout) as r:
        for line in r:
            if not line.strip():
                continue
            d = json.loads(line.decode("utf-8"))
            if "error" in d:
                err = d["error"]
                break
            text.append(d.get("response", ""))
            pieces += 1
            if d.get("done"):
                return client.CallRecord(
                    text="".join(text),
                    in_tokens=int(d.get("prompt_eval_count", 0) or 0),
                    out_tokens=int(d.get("eval_count", 0) or 0),
                    latency_s=round(time.time() - t0, 4), model=model,
                    temperature=float(temperature), seed=seed), None
    latency = time.time() - t0
    if err is None or REPEAT_ABORT not in err:
        return None, err or "stream ended without done"
    pbody = dict(body, options=dict(body["options"], num_predict=1))
    with urllib.request.urlopen(_req(pbody), timeout=timeout) as r:
        in_tok = int(json.loads(r.read().decode("utf-8")).get("prompt_eval_count", 0) or 0)
    rec = client.CallRecord(text="".join(text), in_tokens=in_tok, out_tokens=pieces,
                            latency_s=round(latency, 4), model=model,
                            temperature=float(temperature), seed=seed, aborted=True)
    log = os.environ.get("AGENTS_ABORT_LOG", "").strip()
    if log:
        with open(log, "a") as f:
            f.write(json.dumps({
                "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "model": model, "seed": seed, "temperature": float(temperature),
                "num_predict": body["options"]["num_predict"], "error": err,
                "prompt_sha256": hashlib.sha256(body["prompt"].encode()).hexdigest(),
                "prompt_head": body["prompt"][:400], "in_tokens": in_tok,
                "out_pieces": pieces, "text_tail": rec["text"][-200:]}) + "\n")
    return rec, None


def generate(model, prompt, temperature=0.0, seed=None, num_predict=None,
             stop=None, timeout=900, retries=2):
    try:
        return _orig_generate(model, prompt, temperature=temperature, seed=seed,
                              num_predict=num_predict, stop=stop,
                              timeout=timeout, retries=retries)
    except RuntimeError as e:
        if "HTTP Error 500" not in str(e):
            raise
        body = _body(model, prompt, temperature, seed, num_predict, stop)
        try:
            rec, err = _streamed(body, model, temperature, seed, timeout)
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            raise e
        if rec is None:
            raise RuntimeError(f"{e}; streamed retry: {err}") from e
        return rec


def install():
    client.generate = generate
