#!/usr/bin/env python3
"""Hard-MATH answer-extraction audit (protocols/HARD_MATH_PROTOCOL.md, section
"Extraction" and Amendment 1).

Input: results/power/math500hard_qwen3b_s*.jsonl (all files, shard files
included), read in sorted path order, deduplicated by (config, item_id, seed),
first occurrence wins (the same rule as scripts/analyze_hard_math.py load()).
MATH-500 levels 4-5, 262 items (scripts/run_power_campaign.py load_math_levels),
seeds 1-3, cap 2,048, Qwen2.5-3B. Configs: SA (seed 1), SC_k12, DEBATE_N3_R1,
GE_R2, HIER_M3.

Deciding (answer-bearing) calls, following harness/topologies.py:
  * SA (sa): answer = _extract of the one call -> idx 0.
  * SC (sc, k=12): answer = _majority of the extraction of the k samples. The
    analysis uses k=2, 4, 8, 12 as prefixes of the same 12 samples, so all 12
    samples are deciding calls (idx 0..11); SC k=j uses idx 0..j-1.
  * DEBATE (debate, N=3, R=1): answer = _majority of cur_answers after the last
    critique round = the N final-round calls -> idx 3..5.
  * GE (ge, R=2): answer = _extract of the last revision text -> idx 4
    (calls: gen, crit1, rev1, crit2, rev2).
  * HIER (hier, M=3): answer = _extract of the aggregator text -> idx 4
    (calls: plan, exec1..3, agg).

Flag rule (the GSM8K audit rule adapted to \\boxed{} answers), on deciding calls:
  (a) no_boxed: the text contains no complete \\boxed{...} (harness.tasks._boxed
      returns None);
  (b) cap: the call reached the output cap (hit_cap);
  (c) stated_differs: the last \\boxed answer differs from a final answer stated
      elsewhere in the text, i.e. from (c1) an answer statement ('answer is',
      'answer:', 'answer =', or a 'Final Answer' heading, followed by a \\boxed,
      a $...$ / \\(...\\) / \\[...\\] span or a short token), or (c2) an earlier
      \\boxed{...} in the same text. 'Differs' = not equal under
      harness.tasks.math_score in either direction.
A call is flagged if any of (a), (b), (c) holds. The flag rule reads only the
call text and hit_cap (no workflow label, harness reading or gold).

Steps (subcommands, run from the repo root):
  build    load, replicate, flag; write hardmath_extraction_cases.json
           (shuffled opaque case ids, call text ONLY, plus the list of case ids
           to read) and hardmath_extraction_key.json (case id -> record/call).
  freeze   merge the reader's per-batch reading files (written from the cases
           file text alone, see READER_BATCHES) into
           hardmath_extraction_readings.json and print its sha256. This runs
           before the key or any gold answer is opened.
  candidates  (after freeze) list, from the joined data, the (item, stated
           answer, gold) pairs where math_score and a sympy-based equality test
           disagree, for the scorer-problem review (SCORER_* sets below).
  analyze  join the frozen readings with the key, the records and the gold;
           rescore; write hardmath_extraction_summary.json.

Equivalence of a reading and a harness extraction: harness.tasks.math_score in
either direction; both None counts as agreement.

Usage:  /usr/bin/python3 results/audit/hardmath_extraction_audit.py build|freeze|candidates|analyze
No model is called.
"""
import glob
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from harness import tasks  # noqa: E402
from harness.tasks import math_score, math_extract, _boxed  # noqa: E402
from harness.topologies import _majority  # noqa: E402

PATTERN = "results/power/math500hard_qwen3b_s*.jsonl"
OUT = os.path.join(ROOT, "results", "audit")
CASES = os.path.join(OUT, "hardmath_extraction_cases.json")
KEY = os.path.join(OUT, "hardmath_extraction_key.json")
READINGS = os.path.join(OUT, "hardmath_extraction_readings.json")
SUMMARY = os.path.join(OUT, "hardmath_extraction_summary.json")

CONFIGS = ["SA", "SC_k12", "DEBATE_N3_R1", "GE_R2", "HIER_M3"]
NCALLS = {"SA": 1, "SC_k12": 12, "DEBATE_N3_R1": 6, "GE_R2": 5, "HIER_M3": 5}
DECIDING = {"SA": [0], "SC_k12": list(range(12)), "DEBATE_N3_R1": [3, 4, 5],
            "GE_R2": [4], "HIER_M3": [4]}
KS = [2, 4, 8, 12]
COORD = ["DEBATE_N3_R1", "GE_R2", "HIER_M3"]
SEEDS = [1, 2, 3]
SHUFFLE_SEED = 20261117
SAMPLE_SEED = 20261117
SAMPLE_PER_WORKFLOW = 150
# Read every flagged call if the total is at most this; else a per-workflow sample.
READ_ALL_MAX = 1500


def load_records():
    recs = {}
    files = sorted(glob.glob(os.path.join(ROOT, PATTERN)))
    for p in files:
        for line in open(p):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            recs.setdefault((r["config"], r["item_id"], r["seed"]), r)
    return recs, [os.path.relpath(p, ROOT) for p in files]


def items():
    import run_power_campaign as camp
    return camp.load_math_levels(tasks, 4, 5, 262)


def eq(a, b):
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return bool(math_score(a, b) or math_score(b, a))


def all_boxed(text):
    """Contents of every complete \\boxed{...}, in order (balanced braces)."""
    out = []
    pos = 0
    while True:
        idx = text.find("\\boxed", pos)
        if idx < 0:
            return out
        i = idx + len("\\boxed")
        while i < len(text) and text[i] != "{":
            i += 1
        if i >= len(text):
            return out
        depth = 0
        end = None
        for j in range(i, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    end = j
                    break
        if end is None:
            return out
        out.append(text[i + 1:end].strip())
        pos = end + 1


STATE_RE = re.compile(r"(final\s+answer|answer)\s*(?:is|:|=)?\s*[:\-]?\s*", re.IGNORECASE)
SPAN_RES = [re.compile(r"^\$\$(.{1,80}?)\$\$", re.S), re.compile(r"^\$(.{1,80}?)\$", re.S),
            re.compile(r"^\\\((.{1,80}?)\\\)", re.S), re.compile(r"^\\\[(.{1,80}?)\\\]", re.S)]
TOKEN_RE = re.compile(r"^([^.\n$]{1,40})")


def stated_answers(text):
    """Answers stated by an 'answer is/:/=' phrase or a 'Final Answer' heading."""
    out = []
    for m in STATE_RE.finditer(text):
        word = m.group(1).lower()
        sep = text[m.start():m.end()]
        has_sep = bool(re.search(r"(is|:|=)", sep[len(m.group(1)):], re.IGNORECASE))
        if not has_sep and not word.startswith("final"):
            continue
        rest = text[m.end():m.end() + 200]
        b_at = rest.find("\\boxed")
        if 0 <= b_at <= 40:
            b = all_boxed(rest[b_at:])
            if b:
                out.append(b[0])
            continue
        if word.startswith("final") and not has_sep:
            continue
        got = None
        for rx in SPAN_RES:
            mm = rx.match(rest)
            if mm:
                got = mm.group(1).strip()
                break
        if got is None and has_sep:
            mm = TOKEN_RE.match(rest)
            if mm:
                cand = mm.group(1).strip().strip("$")
                if cand and not re.match(r"^[A-Za-z]+\s+[A-Za-z]+", cand):
                    got = cand
        if got:
            # a math span whose content is 'x = v' states v
            if "=" in got:
                got = got.split("=")[-1].strip()
            out.append(got)
    return out


def flag(text, hit_cap):
    reasons = []
    last = _boxed(text)
    if last is None:
        reasons.append("no_boxed")
    if hit_cap:
        reasons.append("cap")
    if last is not None:
        lb = last.strip()
        boxes = all_boxed(text)
        if any(not eq(b, lb) for b in boxes[:-1]):
            reasons.append("stated_differs_earlier_boxed")
        if any(not eq(s, lb) for s in stated_answers(text)):
            reasons.append("stated_differs_answer_phrase")
    return reasons


def build():
    recs, files = load_records()
    its = items()
    ids = [it["id"] for it in its]
    missing = [(c, i, s) for i in ids for c in CONFIGS for s in ([1] if c == "SA" else SEEDS)
               if (c, i, s) not in recs]
    assert not missing, f"{len(missing)} records missing, e.g. {missing[:3]}"
    keys = [(c, i, s) for c in CONFIGS for i in ids for s in ([1] if c == "SA" else SEEDS)]
    rep_calls = 0
    rep_pred = 0
    rows = []
    cnt = {c: Counter() for c in CONFIGS}
    for k in keys:
        r = recs[k]
        c = r["config"]
        assert len(r["calls"]) == NCALLS[c], (k, len(r["calls"]))
        for call in r["calls"]:
            rep_calls += math_extract(call["text"]) != call["extracted"]
        ans = [r["calls"][i]["extracted"] for i in DECIDING[c]]
        pred = _majority(ans) if c in ("SC_k12", "DEBATE_N3_R1") else ans[0]
        rep_pred += pred != r["pred"]
        for ci in DECIDING[c]:
            call = r["calls"][ci]
            assert call["idx"] == ci
            reasons = flag(call["text"], call["hit_cap"])
            cnt[c]["deciding"] += 1
            if reasons:
                cnt[c]["flagged"] += 1
                for x in reasons:
                    cnt[c][x] += 1
                rows.append((k, ci, call["text"], reasons))
    rng = random.Random(SHUFFLE_SEED)
    rng.shuffle(rows)
    cases, key = [], {}
    for pos, (k, ci, text, reasons) in enumerate(rows):
        cid = "h" + hashlib.sha256(f"{SHUFFLE_SEED}:{pos}".encode()).hexdigest()[:9]
        cases.append({"case": cid, "text": text})
        key[cid] = {"config": k[0], "item_id": k[1], "seed": k[2], "call_idx": ci,
                    "flag_reasons": reasons}
    # which cases are read
    by_cfg = defaultdict(list)
    for cid, v in key.items():
        by_cfg[v["config"]].append(cid)
    if len(cases) <= READ_ALL_MAX:
        to_read = set(key)
        plan = {"mode": "all", "n": len(cases)}
    else:
        srng = random.Random(SAMPLE_SEED)
        to_read = set()
        per = {}
        for c in CONFIGS:
            pool = sorted(by_cfg[c])
            pick = pool if len(pool) <= SAMPLE_PER_WORKFLOW else srng.sample(pool, SAMPLE_PER_WORKFLOW)
            to_read |= set(pick)
            per[c] = {"flagged": len(pool), "read": len(pick)}
        plan = {"mode": "sample", "per_workflow": SAMPLE_PER_WORKFLOW,
                "sampler": f"random.Random({SAMPLE_SEED}).sample(sorted(case ids of the workflow), "
                           f"{SAMPLE_PER_WORKFLOW}) per workflow in order {CONFIGS}; all if fewer",
                "per_config": per}
    for cid in key:
        key[cid]["read"] = cid in to_read
    read_order = [c["case"] for c in cases if c["case"] in to_read]
    json.dump({"note": "flagged deciding calls, shuffled; call text only. 'read' lists the "
                       "case ids the reader reads, in file order.",
               "read": read_order, "cases": cases}, open(CASES, "w"), indent=1)
    json.dump({"inputs": files, "flag_counts": {c: dict(cnt[c]) for c in CONFIGS},
               "replication": {"calls_logged_extracted_ne_math_extract": rep_calls,
                               "records_logged_pred_ne_recomputed_vote": rep_pred},
               "read_plan": plan, "key": key}, open(KEY, "w"), indent=1)
    lens = [len(c["text"]) for c in cases if c["case"] in to_read]
    print(json.dumps({"records": len(keys), "flag_counts": {c: dict(cnt[c]) for c in CONFIGS},
                      "flagged": len(cases), "to_read": len(read_order),
                      "read_chars_total": sum(lens),
                      "replication": [rep_calls, rep_pred], "plan": plan}, indent=1))


# ---------------------------------------------------------------------------
# Reading. The reader (Claude, as eight parallel reader instances of the
# auditing session) was given only /tmp/hardmath_read/batch_<b>.json, a
# contiguous slice of the cases file in file order (case id + text), with the
# instruction to open no other file, and wrote /tmp/hardmath_read/readings_<b>.json.
READER_DIR = "/tmp/hardmath_read"
READER_BATCHES = 8
READER = ("AI assistant (Claude; eight reader instances, each given one contiguous slice "
          "of hardmath_extraction_cases.json: case id and call text only); blind to "
          "workflow, item, gold and harness reading at reading time; no human check")


def freeze():
    cases = json.load(open(CASES))
    want = list(cases["read"])
    merged = {}
    for b in range(READER_BATCHES):
        part = json.load(open(os.path.join(READER_DIR, f"readings_{b}.json")))
        batch = [c["case"] for c in json.load(open(os.path.join(READER_DIR, f"batch_{b}.json")))]
        assert set(part) == set(batch), (b, set(batch) ^ set(part))
        for cid in batch:
            v = part[cid]
            assert v["status"] in ("answer", "none", "truncated"), (cid, v)
            if v["status"] == "answer":
                assert isinstance(v["answer"], str) and v["answer"].strip(), (cid, v)
            merged[cid] = {"status": v["status"],
                           "answer": v["answer"] if v["status"] == "answer" else None,
                           "cut_off": bool(v.get("cut_off", False)),
                           "note": v.get("note", ""), "batch": b}
    assert set(merged) == set(want) and len(merged) == len(want)
    out = {"reader": READER,
           "status_codes": {"answer": "text states a final answer (verbatim in 'answer')",
                            "none": "text states no single final answer",
                            "truncated": "text cut off before stating a final answer"},
           "cut_off_flag": "answer status but text cut off after an explicit final-answer "
                           "statement (sensitivity variant counts it as truncated)",
           "batches": {"n": READER_BATCHES,
                       "rule": "cases file 'read' list in file order, batch b = "
                               "items [b*n//8:(b+1)*n//8]"},
           "pre_freeze_checks": {
               "text_consistency": "answer readings whose text (whitespace removed) does not "
                                   "contain the answer, and none/truncated readings whose text "
                                   "contains '\\boxed': 12 cases listed, each re-read from the "
                                   "text, no reading changed",
               "spot_check": {"n": 25, "sampling": "random.Random(7).sample(sorted(case ids), 25)",
                              "method": "re-read from the text by the auditing session (Claude), "
                                        "blind as above", "agreed": 25, "disagreed": 0}},
           "clerical_fixes": [],
           "n": len(merged), "readings": {cid: merged[cid] for cid in want}}
    json.dump(out, open(READINGS, "w"), indent=1, ensure_ascii=False)
    print(hashlib.sha256(open(READINGS, "rb").read()).hexdigest(), len(merged),
          dict(Counter(v["status"] for v in merged.values())))


# ---------------------------------------------------------------------------
# Equality test used only to find scorer-problem candidates (each candidate is
# then judged and listed by hand in the SCORER_* sets below).
def _brace_arg(s, i):
    """s[i] == '{' -> (content, index after the closing brace)."""
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
    raise ValueError("unbalanced")


def _latex_to_py(s):
    s = s.strip().strip("$").strip()
    for a, b in [("\\left", ""), ("\\right", ""), ("\\dfrac", "\\frac"), ("\\tfrac", "\\frac"),
                 ("\\!", ""), ("\\,", ""), ("\\;", ""), ("\\ ", ""), ("^\\circ", ""),
                 ("^{\\circ}", ""), ("\\%", ""), ("\\cdot", "*"), ("\\times", "*"),
                 ("\\pi", "(pi)"), ("\\infty", "oo"), ("\\displaystyle", "")]:
        s = s.replace(a, b)
    s = re.sub(r"\\(text|mathrm|mbox|mathbf)\{([^{}]*)\}", r"\2", s)
    out, i = "", 0
    while i < len(s):
        if s.startswith("\\frac", i):
            j = i + 5
            if j < len(s) and s[j] == "{":
                a, j = _brace_arg(s, j)
            else:
                a, j = s[j], j + 1
            if j < len(s) and s[j] == "{":
                b, j = _brace_arg(s, j)
            else:
                b, j = s[j], j + 1
            out += f"(({_latex_to_py(a)})/({_latex_to_py(b)}))"
            i = j
        elif s.startswith("\\sqrt", i):
            j = i + 5
            n = "2"
            if j < len(s) and s[j] == "[":
                k = s.index("]", j)
                n, j = s[j + 1:k], k + 1
            if j < len(s) and s[j] == "{":
                a, j = _brace_arg(s, j)
            else:
                a, j = s[j], j + 1
            out += f"(({_latex_to_py(a)})**(1/({n})))"
            i = j
        elif s[i] == "{":
            a, j = _brace_arg(s, i)
            out += f"({_latex_to_py(a)})"
            i = j
        elif s[i] == "\\":
            raise ValueError("unhandled macro")
        else:
            out += s[i]
            i += 1
    return out.replace("^", "**")


def sym_value(s):
    import sympy
    from sympy.parsing.sympy_parser import (parse_expr, standard_transformations,
                                            implicit_multiplication_application)
    if s is None:
        return None
    t = s.strip()
    if "=" in t:
        t = t.split("=")[-1]
    py = _latex_to_py(t)
    return parse_expr(py, transformations=standard_transformations +
                      (implicit_multiplication_application,), evaluate=True)


def sym_equal(a, b):
    """True/False when decidable, None when either side does not parse."""
    import sympy
    try:
        pa = [x for x in _split_top(a)]
        pb = [x for x in _split_top(b)]
        if len(pa) != len(pb):
            return False if len(pa) > 1 or len(pb) > 1 else None
        va = [sym_value(x) for x in pa]
        vb = [sym_value(x) for x in pb]
        used = [False] * len(vb)
        for x in va:
            hit = False
            for k, y in enumerate(vb):
                if not used[k] and sympy.simplify(x - y) == 0:
                    used[k] = hit = True
                    break
            if not hit:
                return False
        return True
    except Exception:
        return None


def _split_top(s):
    s = s.strip()
    if len(s) >= 2 and s[0] in "([" and s[-1] in ")]" and "," in s:
        s = s[1:-1]
    parts, depth, cur = [], 0, ""
    for ch in s:
        if ch in "({[":
            depth += 1
        elif ch in ")}]":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    return [p for p in parts if p.strip()]


# ---------------------------------------------------------------------------
def joined():
    recs, files = load_records()
    K = json.load(open(KEY))
    key = K["key"]
    rd = json.load(open(READINGS))
    readings = rd["readings"]
    its = items()
    ids = [it["id"] for it in its]
    gold = {it["id"]: it["gold"] for it in its}
    level = {it["id"]: it["level"] for it in its}
    # reading per (config, item, seed, call_idx)
    sub = {}
    for cid, k in key.items():
        sub[(k["config"], k["item_id"], k["seed"], k["call_idx"])] = (cid, k)
    return recs, files, K, key, rd, readings, ids, gold, level, sub


def candidates():
    recs, files, K, key, rd, readings, ids, gold, level, sub = joined()
    pairs = Counter()
    for (cfg, item, seed), r in recs.items():
        if cfg not in CONFIGS or item not in gold:
            continue
        assert r["gold"] == gold[item]
        for ci in DECIDING[cfg]:
            h = r["calls"][ci]["extracted"]
            s = sub.get((cfg, item, seed, ci))
            vals = [h]
            if s and readings.get(s[0], {}).get("status") == "answer":
                vals.append(readings[s[0]]["answer"])
            for v in vals:
                if v is not None:
                    pairs[(item, v.strip())] += 1
    eqw, uqr, unparsed = [], [], 0
    for (item, v), n in sorted(pairs.items()):
        g = gold[item]
        ms = bool(math_score(v, g))
        se = sym_equal(v, g)
        if se is None:
            unparsed += 1
        if not ms and se is True:
            eqw.append((item, v, g, n))
        if ms and (se is False or (tasks._math_norm(v) != tasks._math_norm(g))):
            uqr.append((item, v, g, n))
    print(f"distinct (item, answer) pairs {len(pairs)}, sympy unparsed {unparsed}")
    print("== math_score False, sympy equal")
    for x in eqw:
        print(json.dumps(x, ensure_ascii=False))
    print("== math_score True, norm differs or sympy unequal")
    for x in uqr:
        print(json.dumps(x, ensure_ascii=False))
    if len(sys.argv) > 2 and sys.argv[2] == "--wrong-nonnumeric":
        # every wrong-marked answer whose gold is not a plain number, for review
        print("== math_score False, gold not a plain number")
        for (item, v), n in sorted(pairs.items()):
            g = gold[item]
            if not math_score(v, g) and tasks._to_number(tasks._math_norm(g)) is None:
                print(json.dumps((item, v, g, n), ensure_ascii=False))


# Post-freeze adjudications (written after hardmath_extraction_readings.json was
# frozen and joined with the key and the gold; the readings file is not edited).
# Found with `candidates` (math_score False but sympy-equal; wrong-marked pairs
# sympy cannot parse, all 177 read; wrong-marked pairs equal after dropping a
# \text{} unit or a 'var =' prefix), then judged one by one.
SCORER_EQUAL_MARKED_WRONG = {
    ("math-23", "5"),                    # gold 'x=5'; question asks for all x
    ("math-25", "-2, 1"),                # gold '1,-2'; 'enter all such integers'
    ("math-33", "\\frac{2187}{5625}"),   # = 243/625 (unreduced common fraction)
    ("math-92", "\\frac{2240}{78125}"),  # = 448/15625
    ("math-92", "0.028672"),             # = 448/15625 exactly
    ("math-163", "\\frac{-35}{9}"),      # gold '-\frac{35}{9}'
    ("math-163", "\\frac{1}{9} - 4 = \\frac{-35}{9}"),
    ("math-163", "\\frac{1}{9} - 4 = \\frac{1 - 36}{9} = \\frac{-35}{9}"),
    ("math-248", "-\\mathbf{I}"),        # gold is the 2x2 matrix -I
    ("math-273", "1250 \\text{ dollars}"),
    ("math-313", "12\\pi \\text{ inches per second}"),
    ("math-321", "10 \\text{ meters}"),
    ("math-329", "4 \\text{ inches/second}"),
    ("math-388", "5.5"),                 # gold '\frac{11}{2}'
    ("math-408", "\\begin{pmatrix} \\frac{1}{5} \\\\ -\\frac{18}{5} \\end{pmatrix}"),
    ("math-408", "\\begin{pmatrix} 0.2 \\\\ -3.6 \\end{pmatrix}"),  # gold uses 1/5, -18/5
    ("math-441", "550 \\text{ gallons}"),
}
SCORER_UNEQUAL_MARKED_RIGHT = set()
SCORER_NOTES = {
    "not_counted": "math-179: '-5x + 7y - 11z - 4 = 0' and '20x - 28y + 44z + 16 = 0' describe "
                   "the gold plane but violate the form the question requires (A > 0, gcd 1); "
                   "math-293 and math-431: interval endpoints open/closed differ from gold; "
                   "math-238 '-4 and 1', math-391 '0 or 5': extra values; not counted",
    "unequal_marked_right": "math_score True with differing normal forms: math-184, math-217, "
                            "math-242, all equal (spacing, thousands separators, dollar sign); "
                            "no tolerance case found",
}
# Reading vs harness pairs that are mathematically equal but math_score calls
# unequal (formatting/normalization class).
FORMAT_EQUAL = set()


def corrected_score(item_id, pred, gold):
    if pred is not None and (item_id, pred.strip()) in SCORER_EQUAL_MARKED_WRONG:
        return True
    if pred is not None and (item_id, pred.strip()) in SCORER_UNEQUAL_MARKED_RIGHT:
        return False
    return bool(math_score(pred, gold))


def analyze():
    recs, files, K, key, rd, readings, ids, gold, level, sub = joined()
    readings_sha256 = hashlib.sha256(open(READINGS, "rb").read()).hexdigest()
    assert set(readings) == {c for c, v in key.items() if v["read"]}

    def reading_ans(cid, cutoff_as_truncated=False):
        v = readings[cid]
        if v["status"] != "answer":
            return None
        if cutoff_as_truncated and v.get("cut_off"):
            return None
        return v["answer"]

    # workflow views: (name, record config, deciding idx, vote?)
    views = [("SA", "SA", [0], False)]
    views += [(f"SC_k{k}", "SC_k12", list(range(k)), True) for k in KS]
    views += [("DEBATE_N3_R1", "DEBATE_N3_R1", [3, 4, 5], True),
              ("GE_R2", "GE_R2", [4], False), ("HIER_M3", "HIER_M3", [4], False)]

    per_wf, item_scores, flips_all, disagreements = {}, {}, [], []
    scorer = {}
    scorer_cases = []
    for name, cfg, idxs, is_vote in views:
        seeds = [1] if cfg == "SA" else SEEDS
        t = Counter()
        sc_item = defaultdict(lambda: defaultdict(list))
        cls = Counter()
        scorer[name] = Counter()
        for item in ids:
            for s in seeds:
                r = recs[(cfg, item, s)]
                g = gold[item]
                h = {i: r["calls"][i]["extracted"] for i in idxs}
                a, a_s = dict(h), dict(h)
                for i in idxs:
                    c = r["calls"][i]
                    t["deciding"] += 1
                    t["cap"] += bool(c["hit_cap"])
                    sk = sub.get((cfg, item, s, i))
                    if sk is None:
                        continue
                    cid, kk = sk
                    t["flagged"] += 1
                    if not kk["read"]:
                        continue
                    t["read"] += 1
                    v = readings[cid]
                    a[i] = reading_ans(cid)
                    a_s[i] = reading_ans(cid, True)
                    t["reading_" + v["status"]] += 1
                    t["reading_cut_off_after_answer"] += bool(v.get("cut_off"))
                    if not eq(a[i], h[i]):
                        t["differs"] += 1
                        if a[i] is None:
                            k2 = ("truncated" if v["status"] == "truncated"
                                  else "no_answer_stated")
                        elif h[i] is None:
                            k2 = "extractor_none_answer_stated"
                        elif (a[i].strip(), h[i].strip()) in FORMAT_EQUAL:
                            k2 = "format_normalization"
                        else:
                            k2 = "extractor_nonfinal_expression"
                        cls[k2] += 1
                        if name in ("SA", "SC_k12", "DEBATE_N3_R1", "GE_R2", "HIER_M3"):
                            disagreements.append({
                                "case": cid, "config": cfg, "item_id": item, "seed": s,
                                "call_idx": i, "flag_reasons": kk["flag_reasons"],
                                "reading_status": v["status"], "reading": a[i],
                                "reading_note": v.get("note"), "harness_extracted": h[i],
                                "hit_cap": c["hit_cap"], "class": k2})
                vote = (lambda d: _majority([d[i] for i in idxs])) if is_vote else \
                       (lambda d: d[idxs[0]])
                p_log, p_new, p_new_s = vote(h), vote(a), vote(a_s)
                if name == "SC_k12" or not name.startswith("SC"):
                    assert p_log == r["pred"], (name, item, s)
                ok_log = bool(math_score(p_log, g))
                if name == "SC_k12" or not name.startswith("SC"):
                    assert ok_log == bool(r["correct"]), (name, item, s)
                ok_new = bool(math_score(p_new, g))
                ok_new_s = bool(math_score(p_new_s, g))
                ok_log_fix = corrected_score(item, p_log, g)
                ok_new_fix = corrected_score(item, p_new, g)
                t["records"] += 1
                t["logged_correct"] += ok_log
                t["rescored_correct"] += ok_new
                t["logged_none"] += p_log is None
                t["rescored_none"] += p_new is None
                t["logged_answered_correct"] += ok_log and p_log is not None
                t["rescored_answered_correct"] += ok_new and p_new is not None
                t["flip_wrong_to_right"] += ok_new and not ok_log
                t["flip_right_to_wrong"] += ok_log and not ok_new
                if ok_new != ok_log:
                    flips_all.append({"workflow": name, "item_id": item, "seed": s,
                                      "gold": g, "logged_pred": p_log, "rescored_pred": p_new,
                                      "logged_correct": ok_log, "rescored_correct": ok_new})
                if ok_log_fix != ok_log:
                    scorer[name]["records_logged_pred_" + ("equal_marked_wrong" if ok_log_fix
                                                           else "unequal_marked_right")] += 1
                if ok_new_fix != ok_new:
                    scorer[name]["records_rescored_pred_" + ("equal_marked_wrong" if ok_new_fix
                                                             else "unequal_marked_right")] += 1
                for i in idxs:
                    x = a[i]
                    if x is None:
                        continue
                    if (item, x.strip()) in SCORER_EQUAL_MARKED_WRONG:
                        scorer[name]["calls_equal_marked_wrong"] += 1
                        if name in ("SA", "SC_k12", "DEBATE_N3_R1", "GE_R2", "HIER_M3"):
                            scorer_cases.append({"config": cfg, "item_id": item, "seed": s,
                                                 "call_idx": i, "gold": g, "stated": x,
                                                 "verdict": "equal_but_marked_wrong"})
                    elif (item, x.strip()) in SCORER_UNEQUAL_MARKED_RIGHT:
                        scorer[name]["calls_unequal_marked_right"] += 1
                        if name in ("SA", "SC_k12", "DEBATE_N3_R1", "GE_R2", "HIER_M3"):
                            scorer_cases.append({"config": cfg, "item_id": item, "seed": s,
                                                 "call_idx": i, "gold": g, "stated": x,
                                                 "verdict": "unequal_but_marked_right"})
                for lab, ok in (("logged", ok_log), ("rescored", ok_new),
                                ("rescored_cutoff_as_truncated", ok_new_s),
                                ("logged_scorer_fixed", ok_log_fix),
                                ("rescored_scorer_fixed", ok_new_fix)):
                    sc_item[lab][item].append(float(ok))
        item_scores[name] = sc_item
        N, D = t["records"], t["deciding"]
        acc = {lab: round(100 * sum(sum(v) / len(v) for v in sc_item[lab].values()) / len(ids), 2)
               for lab in sc_item}
        nf = t["reading_none"] + t["reading_truncated"]
        per_wf[name] = {
            "records": N,
            "logged_acc_pct": acc["logged"], "rescored_acc_pct": acc["rescored"],
            "rescored_acc_cutoff_as_truncated_pct": acc["rescored_cutoff_as_truncated"],
            "logged_acc_scorer_fixed_pct": acc["logged_scorer_fixed"],
            "rescored_acc_scorer_fixed_pct": acc["rescored_scorer_fixed"],
            "logged_correct": t["logged_correct"], "rescored_correct": t["rescored_correct"],
            "flip_wrong_to_right": t["flip_wrong_to_right"],
            "flip_right_to_wrong": t["flip_right_to_wrong"],
            "deciding_calls": D, "flagged_calls": t["flagged"], "read_calls": t["read"],
            "reading_differs_from_harness_calls": t["differs"],
            "cap_calls": t["cap"], "cap_share": round(t["cap"] / D, 4),
            "no_final_answer_calls": nf, "no_final_answer_share": round(nf / D, 4),
            "reading_none_calls": t["reading_none"],
            "reading_truncated_calls": t["reading_truncated"],
            "reading_cut_off_after_answer_calls": t["reading_cut_off_after_answer"],
            "record_no_answer_logged": t["logged_none"],
            "record_no_answer_logged_share": round(t["logged_none"] / N, 4),
            "record_no_answer_rescored": t["rescored_none"],
            "record_no_answer_rescored_share": round(t["rescored_none"] / N, 4),
            "acc_among_answered_logged_pct": round(100 * t["logged_answered_correct"]
                                                   / max(1, N - t["logged_none"]), 2),
            "acc_among_answered_rescored_pct": round(100 * t["rescored_answered_correct"]
                                                     / max(1, N - t["rescored_none"]), 2),
            "disagreement_classes": dict(cls),
        }
    labs = ["logged", "rescored", "rescored_cutoff_as_truncated", "logged_scorer_fixed",
            "rescored_scorer_fixed"]
    diffs = {}
    for w in COORD:
        for base in ("SC_k8", "SC_k12", "SA"):
            diffs[f"{w}_minus_{base}"] = {
                lab: round(_acc(item_scores[w][lab], ids) - _acc(item_scores[base][lab], ids), 2)
                for lab in labs}
    summary = {
        "inputs": files,
        "dedupe": "sorted path order, (config, item_id, seed), first occurrence wins",
        "items": len(ids),
        "reader": rd["reader"],
        "readings_file_sha256_at_analysis": readings_sha256,
        "deciding_calls_rule": DECIDING,
        "flag_rule": "(a) no complete \\boxed{} (b) hit_cap (c) last \\boxed differs (math_score "
                     "both directions) from an earlier \\boxed or from an 'answer is/:/=' or "
                     "'Final Answer' statement",
        "flag_counts": K["flag_counts"],
        "read_plan": K["read_plan"],
        "replication": K["replication"],
        "equivalence_reading_vs_harness": "math_score(reading, extracted) or "
                                          "math_score(extracted, reading); both None = agree",
        "rescoring": "flagged calls take the reading (status none/truncated -> None); unflagged "
                     "calls keep the harness reading; vote = harness.topologies._majority over "
                     "deciding calls (SC prefixes, DEBATE), single deciding call otherwise; "
                     "scored with harness.tasks.math_score",
        "accuracy_definition": "mean over the 262 items of the per-item mean over seeds "
                               "(SA: seed 1), as in scripts/analyze_hard_math.py",
        "per_workflow": per_wf,
        "differences_points": diffs,
        "scorer_problems": {k: dict(v) for k, v in scorer.items()},
        "scorer_problem_rules": {
            "equal_but_marked_wrong": sorted(map(list, SCORER_EQUAL_MARKED_WRONG)),
            "unequal_but_marked_right": sorted(map(list, SCORER_UNEQUAL_MARKED_RIGHT)),
            "notes": SCORER_NOTES},
        "scorer_problem_calls": scorer_cases,
        "flips": flips_all,
        "disagreements": sorted(disagreements, key=lambda d: (d["config"], d["item_id"],
                                                              d["seed"], d["call_idx"])),
    }
    json.dump(summary, open(SUMMARY, "w"), indent=1, ensure_ascii=False)
    show = {k: {kk: vv for kk, vv in v.items() if kk != "disagreement_classes"}
            for k, v in per_wf.items()}
    print(json.dumps({"per_workflow": show, "classes": {k: v["disagreement_classes"]
                                                         for k, v in per_wf.items()},
                      "diffs": diffs, "scorer": summary["scorer_problems"],
                      "sha": readings_sha256}, indent=1))


def _acc(m, ids):
    return 100 * sum(sum(m[i]) / len(m[i]) for i in ids) / len(ids)


if __name__ == "__main__":
    {"build": build, "freeze": freeze, "candidates": candidates,
     "analyze": analyze}[sys.argv[1]]()
