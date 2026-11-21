#!/usr/bin/env python3
"""MATH answer-extraction audit (results/math500_qwen3b_coord_trace.jsonl).

Input: 240 records (60 MATH level 1-3 items x SC_k8, DEBATE_N3_R1, GE_R2,
HIER_M3; seed 1; cap 1,024; Qwen2.5-3B). Each call logs its full text and the
value the unchanged harness extractor (harness.tasks.math_extract via
harness.topologies._extract) read from it.

Deciding calls, following harness/topologies.py:
  * SC (sc): answer = _majority of the extraction of all k=8 samples -> idx 0..7.
  * DEBATE (debate, N=3, R=1): answer = _majority of cur_answers after the last
    critique round, i.e. the N final-round calls -> idx 3..5.
  * GE (ge, R=2): answer = _extract of the last revision text -> idx 4
    (calls: gen, crit1, rev1, crit2, rev2).
  * HIER (hier, M=3): answer = _extract of the aggregator text -> idx 4
    (calls: plan, exec1..3, agg).

Steps (subcommands, run from the repo root):
  build    write math_extraction_cases.json (shuffled opaque case id + call
           text ONLY) and math_extraction_key.json (case id -> record/call).
  auto     print auto-resolvable case ids and boxed content, from the cases
           file alone (text only): exactly one \\boxed{...} in the text, and
           the text after its closing brace is under 300 characters and
           contains no 'answer', no '=', and no further '\\boxed'.
  analyze  join the frozen math_extraction_readings.json with the key and the
           trace, classify disagreements, rescore, write
           math_extraction_summary.json.

Equivalence of a reading and a harness extraction: harness.tasks.math_score
(reading, extracted), i.e. _math_norm string equality or |difference| < 1e-4
after _to_number. Both None counts as agreement.

Usage:  /usr/bin/python3 results/audit/math_extraction_audit.py build|auto|analyze
No model is called.
"""
import hashlib
import json
import os
import random
import sys
from collections import Counter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from harness.tasks import math_score, _boxed  # noqa: E402
from harness.topologies import _majority  # noqa: E402

TRACE = os.path.join(ROOT, "results", "math500_qwen3b_coord_trace.jsonl")
OUT = os.path.join(ROOT, "results", "audit")
CASES = os.path.join(OUT, "math_extraction_cases.json")
KEY = os.path.join(OUT, "math_extraction_key.json")
READINGS = os.path.join(OUT, "math_extraction_readings.json")
SUMMARY = os.path.join(OUT, "math_extraction_summary.json")

CONFIGS = ["SC_k8", "DEBATE_N3_R1", "GE_R2", "HIER_M3"]
DECIDING = {"SC_k8": list(range(8)), "DEBATE_N3_R1": [3, 4, 5],
            "GE_R2": [4], "HIER_M3": [4]}
COORD = ["DEBATE_N3_R1", "GE_R2", "HIER_M3"]
SHUFFLE_SEED = 20261116


def load():
    return [json.loads(l) for l in open(TRACE)]


def build():
    recs = load()
    rows = []
    for ri, r in enumerate(recs):
        n = len(r["calls"])
        exp = {"SC_k8": 8, "DEBATE_N3_R1": 6, "GE_R2": 5, "HIER_M3": 5}[r["config"]]
        assert n == exp, (r["item_id"], r["config"], n)
        for ci in DECIDING[r["config"]]:
            c = r["calls"][ci]
            assert c["idx"] == ci
            rows.append((ri, ci, c["text"]))
    rng = random.Random(SHUFFLE_SEED)
    rng.shuffle(rows)
    cases, key = [], {}
    for pos, (ri, ci, text) in enumerate(rows):
        cid = "c" + hashlib.sha256(f"{SHUFFLE_SEED}:{pos}".encode()).hexdigest()[:8]
        cases.append({"case": cid, "text": text})
        r = recs[ri]
        key[cid] = {"record_index": ri, "item_id": r["item_id"],
                    "config": r["config"], "seed": r["seed"], "call_idx": ci}
    json.dump(cases, open(CASES, "w"), indent=1)
    json.dump(key, open(KEY, "w"), indent=1)
    print(f"{len(cases)} cases written")


def auto_rule(text):
    if text.count("\\boxed") != 1:
        return None
    idx = text.find("\\boxed")
    b = _boxed(text)
    if b is None:
        return None
    end = text.find(b, idx) + len(b) + 1
    tail = text[end:]
    if len(tail) >= 300 or "answer" in tail.lower() or "=" in tail or "\\boxed" in tail:
        return None
    return b.strip()


def auto():
    cases = json.load(open(CASES))
    res = {}
    for c in cases:
        a = auto_rule(c["text"])
        if a is not None:
            res[c["case"]] = a
    print(json.dumps(res, indent=1))
    print(f"# auto-resolved {len(res)} of {len(cases)}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Post-freeze adjudications (written after math_extraction_readings.json was
# frozen and joined with the key and the gold; the readings file is not edited).
#
# Scorer problems: (item_id, stated answer) pairs where harness.tasks.math_score
# against the gold disagrees with mathematical equality.
#   math-227 gold '\text{(C)}': _math_norm keeps the parentheses -> '(c)', so a
#     stated 'C' or '\text{C}' (-> 'c') is marked wrong although it names the
#     same option.
#   math-255 gold '\text{(E)}': same mechanism for 'E' / '\text{E}'.
#   math-176 gold '.0000672' (= 21/312500 exactly): '0.0000671875' differs by
#     1.25e-8, inside the absolute 1e-4 tolerance of _to_number comparison, so it
#     is marked right although it is a different number.
SCORER_EQUAL_MARKED_WRONG = {("math-227", "C"), ("math-227", "\\text{C}"),
                             ("math-255", "E"), ("math-255", "\\text{E}")}
SCORER_UNEQUAL_MARKED_RIGHT = {("math-176", "0.0000671875")}
# Reading vs harness pairs that are mathematically equal but math_score calls
# unequal (formatting/normalization class). None found in this file.
FORMAT_EQUAL = set()


def eq(a, b):
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return math_score(a, b) or math_score(b, a)


def corrected_score(item_id, pred, gold):
    if pred is not None and (item_id, pred.strip()) in SCORER_EQUAL_MARKED_WRONG:
        return True
    if pred is not None and (item_id, pred.strip()) in SCORER_UNEQUAL_MARKED_RIGHT:
        return False
    return math_score(pred, gold)


def vote(cfg, answers):
    idxs = DECIDING[cfg]
    if cfg in ("SC_k8", "DEBATE_N3_R1"):
        return _majority([answers[i] for i in idxs])
    return answers[idxs[0]]


def analyze():
    recs = load()
    key = json.load(open(KEY))
    rd = json.load(open(READINGS))
    readings_sha256 = hashlib.sha256(open(READINGS, "rb").read()).hexdigest()
    readings = rd["readings"]
    assert set(readings) == set(key), "readings must cover every case"

    def ans(cid, cutoff_as_truncated=False):
        v = readings[cid]
        if v["status"] != "answer":
            return None
        if cutoff_as_truncated and v.get("cut_off"):
            return None
        return v["answer"]

    # replication: logged extracted == math_extract(text) for every call
    from harness.tasks import math_extract
    rep_calls = sum(1 for r in recs for c in r["calls"]
                    if math_extract(c["text"]) != c["extracted"])

    cnt = {c: Counter() for c in CONFIGS}
    disagreements = []
    by_rec, by_rec_s = {}, {}
    for cid, k in key.items():
        r = recs[k["record_index"]]
        c = r["calls"][k["call_idx"]]
        cfg = r["config"]
        h = c["extracted"]
        v = readings[cid]
        a = ans(cid)
        n = cnt[cfg]
        n["calls"] += 1
        n["cap"] += bool(c["hit_cap"])
        n["reading_none"] += v["status"] == "none"
        n["reading_truncated"] += v["status"] == "truncated"
        n["reading_cut_off_after_answer"] += bool(v.get("cut_off"))
        n["harness_none"] += h is None
        n["auto"] += v["method"] == "auto"
        if not eq(a, h):
            if a is None:
                klass = ("truncated" if v["status"] == "truncated"
                         else "no_answer_stated")
            elif h is None:
                klass = "extractor_none_answer_stated"
            elif (a, h) in FORMAT_EQUAL:
                klass = "format_normalization"
            else:
                klass = "extractor_nonfinal_expression"
            n["misread"] += 1
            n["class:" + klass] += 1
            disagreements.append({"case": cid, "config": cfg, "item_id": r["item_id"],
                                  "call_idx": k["call_idx"],
                                  "reading_status": v["status"], "reading": a,
                                  "reading_note": v.get("note"),
                                  "harness_extracted": h, "hit_cap": c["hit_cap"],
                                  "class": klass})
        by_rec.setdefault(k["record_index"], {})[k["call_idx"]] = a
        by_rec_s.setdefault(k["record_index"], {})[k["call_idx"]] = ans(cid, True)

    table, flips = {}, []
    scorer = {c: Counter() for c in CONFIGS}
    scorer_cases = []
    pred_rep = Counter()
    for ri, r in enumerate(recs):
        cfg, gold, item = r["config"], r["gold"], r["item_id"]
        harness_pred = vote(cfg, {i: r["calls"][i]["extracted"] for i in DECIDING[cfg]})
        pred_rep["match" if harness_pred == r["pred"] else "mismatch"] += 1
        pred_rep["correct_match" if bool(math_score(r["pred"], gold)) == bool(r["correct"])
                 else "correct_mismatch"] += 1
        new_pred = vote(cfg, by_rec[ri])
        new_pred_s = vote(cfg, by_rec_s[ri])
        ok_log = bool(r["correct"])
        ok_new = bool(math_score(new_pred, gold))
        ok_new_s = bool(math_score(new_pred_s, gold))
        ok_log_fix = bool(corrected_score(item, r["pred"], gold))
        ok_new_fix = bool(corrected_score(item, new_pred, gold))
        t = table.setdefault(cfg, Counter())
        t["n"] += 1
        t["logged_correct"] += ok_log
        t["rescored_correct"] += ok_new
        t["rescored_cutoff_as_truncated_correct"] += ok_new_s
        t["logged_scorer_fixed_correct"] += ok_log_fix
        t["rescored_scorer_fixed_correct"] += ok_new_fix
        t["flip_wrong_to_right"] += ok_new and not ok_log
        t["flip_right_to_wrong"] += ok_log and not ok_new
        if ok_new != ok_log:
            flips.append({"item_id": item, "config": cfg, "gold": gold,
                          "logged_pred": r["pred"], "rescored_pred": new_pred,
                          "logged_correct": ok_log, "rescored_correct": ok_new})
        if ok_log_fix != ok_log:
            scorer[cfg]["records_logged_pred"] += 1
            scorer[cfg]["records_logged_pred_" + ("equal_marked_wrong" if ok_log_fix
                                                 else "unequal_marked_right")] += 1
        for i in DECIDING[cfg]:
            a = by_rec[ri][i]
            if a is None:
                continue
            if (item, a.strip()) in SCORER_EQUAL_MARKED_WRONG:
                scorer[cfg]["calls_equal_marked_wrong"] += 1
                scorer_cases.append({"item_id": item, "config": cfg, "call_idx": i,
                                     "gold": gold, "stated": a,
                                     "verdict": "equal_but_marked_wrong"})
            elif (item, a.strip()) in SCORER_UNEQUAL_MARKED_RIGHT:
                scorer[cfg]["calls_unequal_marked_right"] += 1
                scorer_cases.append({"item_id": item, "config": cfg, "call_idx": i,
                                     "gold": gold, "stated": a,
                                     "verdict": "unequal_but_marked_right"})

    per_wf = {}
    for cfg in CONFIGS:
        t, n = table[cfg], cnt[cfg]
        N = t["n"]
        per_wf[cfg] = {
            "records": N,
            "logged_acc": round(t["logged_correct"] / N, 4),
            "rescored_acc": round(t["rescored_correct"] / N, 4),
            "rescored_acc_cutoff_as_truncated": round(
                t["rescored_cutoff_as_truncated_correct"] / N, 4),
            "logged_acc_scorer_fixed": round(t["logged_scorer_fixed_correct"] / N, 4),
            "rescored_acc_scorer_fixed": round(t["rescored_scorer_fixed_correct"] / N, 4),
            "logged_correct": t["logged_correct"],
            "rescored_correct": t["rescored_correct"],
            "flip_wrong_to_right": t["flip_wrong_to_right"],
            "flip_right_to_wrong": t["flip_right_to_wrong"],
            "deciding_calls": n["calls"],
            "auto_resolved_calls": n["auto"],
            "misread_calls": n["misread"],
            "misread_share": round(n["misread"] / n["calls"], 4),
            "cap_calls": n["cap"],
            "cap_share": round(n["cap"] / n["calls"], 4),
            "no_final_answer_calls": n["reading_none"] + n["reading_truncated"],
            "no_final_answer_share": round((n["reading_none"] + n["reading_truncated"])
                                           / n["calls"], 4),
            "reading_none_calls": n["reading_none"],
            "reading_truncated_calls": n["reading_truncated"],
            "reading_cut_off_after_answer_calls": n["reading_cut_off_after_answer"],
            "harness_none_calls": n["harness_none"],
            "disagreement_classes": {k[6:]: v for k, v in n.items() if k.startswith("class:")},
        }
    gaps = {}
    for i, a in enumerate(CONFIGS):
        for b in CONFIGS[i + 1:]:
            gaps[f"{a}_minus_{b}"] = {
                f: round((table[a][f + "_correct"] - table[b][f + "_correct"]) / 60, 4)
                for f in ("logged", "rescored", "rescored_cutoff_as_truncated",
                          "logged_scorer_fixed", "rescored_scorer_fixed")}
    sc_m, sc_n = cnt["SC_k8"]["misread"], cnt["SC_k8"]["calls"]
    co_m = sum(cnt[c]["misread"] for c in COORD)
    co_n = sum(cnt[c]["calls"] for c in COORD)

    summary = {
        "input": "results/math500_qwen3b_coord_trace.jsonl",
        "reader": rd["reader"],
        "readings_file_sha256_at_analysis": readings_sha256,
        "deciding_calls_rule": DECIDING,
        "equivalence_reading_vs_harness": "harness.tasks.math_score(reading, extracted) "
                                          "or math_score(extracted, reading); both None = agree",
        "rescoring": "vote = harness.topologies._majority over readings of the deciding "
                     "calls (SC_k8, DEBATE_N3_R1); reading of the last call (GE_R2, HIER_M3); "
                     "scored against gold with harness.tasks.math_score",
        "n_cases": len(key),
        "auto_resolved": rd["auto_resolved_count"],
        "manual": rd["manual_count"],
        "spot_check": rd["spot_check"],
        "clerical_fixes": rd["clerical_fixes"],
        "replication": {"calls_where_logged_extracted_differs_from_math_extract": rep_calls,
                        "records_logged_pred_vs_recomputed_vote": dict(pred_rep)},
        "per_workflow": per_wf,
        "gaps": gaps,
        "misread_rate_SC_samples": {"misread": sc_m, "calls": sc_n,
                                    "rate": round(sc_m / sc_n, 4)},
        "misread_rate_coordination_final_calls": {"misread": co_m, "calls": co_n,
                                                  "rate": round(co_m / co_n, 4)},
        "scorer_problems": {c: dict(scorer[c]) for c in CONFIGS},
        "scorer_problem_rules": {
            "equal_but_marked_wrong": sorted(map(list, SCORER_EQUAL_MARKED_WRONG)),
            "unequal_but_marked_right": sorted(map(list, SCORER_UNEQUAL_MARKED_RIGHT))},
        "scorer_problem_calls": scorer_cases,
        "flips": flips,
        "disagreements": sorted(disagreements, key=lambda d: (d["config"], d["item_id"],
                                                              d["call_idx"])),
    }
    json.dump(summary, open(SUMMARY, "w"), indent=1)
    print(json.dumps({k: summary[k] for k in (
        "replication", "per_workflow", "gaps", "misread_rate_SC_samples",
        "misread_rate_coordination_final_calls", "scorer_problems", "flips",
        "disagreements")}, indent=1))


if __name__ == "__main__":
    {"build": build, "auto": auto, "analyze": analyze}[sys.argv[1]]()
