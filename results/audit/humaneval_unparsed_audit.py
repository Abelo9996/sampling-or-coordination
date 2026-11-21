#!/usr/bin/env python3
"""HumanEval unparsed-program audit (results/humaneval_matrix_pc.jsonl).

Regenerates the mechanical parts of
results/audit/humaneval_unparsed_adjudication.json:
  * counts of records with no parseable program, per config and seed;
  * output tokens of the final call(s) and whether they hit the generation cap;
  * per-sample unparsed rates (SC samples, debate final-round agents), which
    are recoverable because those workflows log their per-sample programs;
  * the seed-0 random sample of 40 wrong coordination records with a parsed
    program, with mechanical flags on the extracted program;
  * accuracy variants per config and SA minus workflow gaps;
  * rescoring of any stored literal extraction (`literal_extraction` field of
    an entry in the existing JSON) with the harness's own scorer and sandbox.

Finding that shapes everything below: the run script (scripts/run_code_energy.py)
stores `raw` only when the topology's `raw` is a list, i.e. the list of
extracted programs for SC and debate. For SA, GE and HIER `raw` is None, and the
per-call records keep only token counts and timings. The model's output text was
never written to disk, so no record can be read, classified by reading, or
rescued by a literal extraction from this file. Categories are therefore
`other` (text not logged) for every unparsed record.

Usage (repo root):  /usr/bin/python3 results/audit/humaneval_unparsed_audit.py
Scoring uses harness.tasks.humaneval_score (sandbox: harness/sandbox.py, macOS
sandbox-exec backend on this Mac; the logged scores came from the docker
backend on the PC, so a replication check against the logged scores is run).
No model is called.
"""
import ast
import json
import os
import random
import sys
from collections import Counter, defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from harness import tasks  # noqa: E402

SRC = os.path.join(ROOT, "results", "humaneval_matrix_pc.jsonl")
OUT = os.path.join(ROOT, "results", "audit", "humaneval_unparsed_adjudication.json")
CONFIGS = ["SA", "SC_k2", "SC_k4", "SC_k8", "DEBATE_N3_R1", "GE_R2", "HIER_M3"]
COORD = ["DEBATE_N3_R1", "GE_R2", "HIER_M3"]
NO_TEXT = ("model output text not logged: scripts/run_code_energy.py stores raw "
           "only when it is a list of extracted programs (SC, debate); for SA, GE "
           "and HIER raw is None and calls keep token counts only")


def load():
    recs = [json.loads(l) for l in open(SRC)]
    items = {}
    for it in tasks.humaneval_load():
        items[it["id"]] = it
    return recs, items


def unparsed(r):
    return r["pred"] in (None, "")


def final_calls(r):
    """Calls whose text is the final answer: debate = the N=3 round-1 agents
    (calls 3..5), SC = all k samples, SA/GE/HIER = the last call."""
    c, cfg = r["calls"], r["config"]
    if cfg.startswith("DEBATE"):
        return c[3:6]
    if cfg.startswith("SC"):
        return c
    return [c[-1]]


def hit_cap(r):
    return [c["out_tokens"] >= r["num_predict"] for c in final_calls(r)]


def score(prog, item):
    return bool(tasks.humaneval_score(prog, item["gold"]))


# --- mechanical flags on an extracted (canonical) program -------------------
KEEP = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef,
        ast.ClassDef, ast.Assign, ast.AnnAssign)


def prog_flags(pred, item):
    entry = item["entry_point"]
    t = ast.parse(pred)
    defs = [n for n in t.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            and n.name == entry]
    f = defs[-1]
    stub = all(isinstance(s, ast.Pass) for s in f.body)
    toplevel = [type(n).__name__ for n in t.body if not isinstance(n, KEEP)]
    stub_kind = None
    if stub:
        # The extracted block did not define the entry point, so code_canonical
        # appended it to the prompt's stub. What did the block hold?
        pre_names = {n.name for n in ast.parse(item["prompt"]).body
                     if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        idx = t.body.index(f)
        after = t.body[idx + 1:]
        other_defs = [n.name for n in after if isinstance(n, (ast.FunctionDef,
                      ast.AsyncFunctionDef)) and n.name not in pre_names
                      and not n.name.lower().startswith(("check", "test", "main"))]
        if other_defs:
            stub_kind = "block_defines_other_name:" + ",".join(other_defs)
        elif after:
            stub_kind = "block_is_test_or_usage_code"
        else:
            stub_kind = "block_empty_or_body_only"
    return {"entry_stub": stub, "stub_kind": stub_kind,
            "toplevel_nondef_stmts": toplevel, "n_entry_defs": len(defs)}


def sanitize(pred):
    """Keep imports, defs, classes and top-level assignments; drop other
    top-level statements (the model's own example calls, prints, asserts,
    __main__ blocks). Diagnostic only, in the spirit of EvalPlus's sanitizer;
    it never edits a function."""
    t = ast.parse(pred)
    t.body = [n for n in t.body if isinstance(n, KEEP)]
    return ast.unparse(t)


def acc_block(rows, extra_pass=0):
    n = len(rows)
    c = sum(r["correct"] for r in rows) + extra_pass
    return {"num": c, "den": n, "acc": round(c / n, 4) if n else None}


def main():
    recs, items = load()
    old = {}
    if os.path.exists(OUT):
        try:
            old = json.load(open(OUT))
        except Exception:
            old = {}
    by = defaultdict(list)
    for r in recs:
        by[(r["config"], r["seed"])].append(r)

    # 1. counts ---------------------------------------------------------------
    counts = []
    for cfg in CONFIGS:
        for s in sorted({k[1] for k in by if k[0] == cfg}):
            rows = by[(cfg, s)]
            u = [r for r in rows if unparsed(r)]
            counts.append({
                "config": cfg, "seed": s, "n": len(rows),
                "unparsed": len(u),
                "unparsed_final_call_hit_cap_all": sum(all(hit_cap(r)) for r in u),
                "unparsed_final_call_hit_cap_any": sum(any(hit_cap(r)) for r in u),
                "correct": sum(r["correct"] for r in rows)})

    # per-sample unparsed rates where per-sample programs are logged ------------
    per_sample = {}
    for cfg in ["SC_k2", "SC_k4", "SC_k8", "DEBATE_N3_R1"]:
        n = nn = cap = nncap = 0
        for r in recs:
            if r["config"] != cfg:
                continue
            for a, c in zip(r["raw"], final_calls(r)):
                hc = c["out_tokens"] >= r["num_predict"]
                n += 1; cap += hc
                if a in (None, ""):
                    nn += 1; nncap += hc
        per_sample[cfg] = {"samples": n, "unparsed": nn, "hit_cap": cap,
                           "unparsed_and_hit_cap": nncap,
                           "note": ("SC samples: first-pass answers to the plain prompt "
                                    "at T=0.7" if cfg.startswith("SC") else
                                    "debate: the 3 round-1 (final) agent outputs")}

    # 2-3. per-record entries for unparsed records ------------------------------
    old_lit = {(e["item_id"], e["config"], e["seed"]): e
               for e in old.get("records", []) if e.get("literal_extraction")}
    entries = []
    for r in recs:
        if not unparsed(r):
            continue
        hc = hit_cap(r)
        toks = [c["out_tokens"] for c in final_calls(r)]
        e = {"item_id": r["item_id"], "config": r["config"], "seed": r["seed"],
             "category": "other",
             "reason": NO_TEXT,
             "out_tokens": toks if len(toks) > 1 else toks[0],
             "num_predict": r["num_predict"],
             "hit_cap": all(hc) if len(hc) == 1 else hc,
             "mechanical_hint": ("final output hit the cap: consistent with truncated, "
                                 "but a complete program followed by a cut second "
                                 "block would look the same" if all(hc) else
                                 "final output below the cap: not truncation; "
                                 "no_code vs code_present_unextracted undeterminable"),
             "debate_round1_programs_logged": (r["raw"] if isinstance(r["raw"], list)
                                               else None),
             "rescued_pass": None,
             "excerpt": None}
        lit = old_lit.get((r["item_id"], r["config"], r["seed"]))
        if lit:  # rescore a stored literal extraction, if one was ever added
            e["literal_extraction"] = lit["literal_extraction"]
            prog = tasks.code_canonical("```python\n" + lit["literal_extraction"] + "```",
                                        items[r["item_id"]])
            e["rescued_pass"] = score(prog, items[r["item_id"]]) if prog else False
        entries.append(e)

    # 4. seed-0 sample of 40 wrong coordination records with a parsed program ---
    pop = [r for r in recs if r["config"] in COORD and not unparsed(r) and not r["correct"]]
    rng = random.Random(0)
    sample = rng.sample(pop, 40)
    sample_entries = []
    for r in sample:
        it = items[r["item_id"]]
        fl = prog_flags(r["pred"], it)
        sample_entries.append({
            "item_id": r["item_id"], "config": r["config"], "seed": r["seed"],
            "final_out_tokens": [c["out_tokens"] for c in final_calls(r)],
            **fl,
            "presented_program_matches_pred": None,
            "reason": NO_TEXT + "; the presented program cannot be compared",
            "pred_excerpt": r["pred"][-300:]})

    # extractor-signature scan over ALL parsed records (every config) ----------
    scan = {}
    for cfg in CONFIGS:
        c = Counter()
        kinds = Counter()
        for r in recs:
            if r["config"] != cfg or unparsed(r):
                continue
            fl = prog_flags(r["pred"], items[r["item_id"]])
            k = "pass" if r["correct"] else "fail"
            c[k] += 1
            if fl["entry_stub"]:
                c[k + "_entry_stub"] += 1
                kinds[fl["stub_kind"].split(":")[0]] += 1
            if fl["toplevel_nondef_stmts"]:
                c[k + "_toplevel_nondef"] += 1
        scan[cfg] = {**dict(c), "stub_kinds": dict(kinds)}

    # diagnostic: sanitized rescoring + replication of logged scores ----------
    repl = Counter()
    san_correct = {}
    for r in recs:
        if unparsed(r):
            continue
        it = items[r["item_id"]]
        fl = prog_flags(r["pred"], it)
        key = (r["item_id"], r["config"], r["seed"])
        if not fl["toplevel_nondef_stmts"] and r["config"] != "SA":
            san_correct[key] = r["correct"]
            continue
        mine = score(r["pred"], it)
        repl["agree" if mine == r["correct"] else "disagree"] += 1
        if not fl["toplevel_nondef_stmts"]:
            san_correct[key] = r["correct"]
        else:
            san_correct[key] = score(sanitize(r["pred"]), it)

    # 5. accuracy variants ------------------------------------------------------
    def variants(rows):
        u = [r for r in rows if unparsed(r)]
        u_nocap = [r for r in u if not all(hit_cap(r))]
        parsed = [r for r in rows if not unparsed(r)]
        san = sum(san_correct[(r["item_id"], r["config"], r["seed"])] for r in parsed)
        return {
            "logged": acc_block(rows),
            "literal_rescue": None,
            "literal_rescue_upper_bound_all_noncap_unparsed_pass":
                acc_block(rows, extra_pass=len(u_nocap)),
            "parsed_only": acc_block(parsed),
            "diag_sanitized_toplevel_dropped": {"num": san, "den": len(rows),
                                                "acc": round(san / len(rows), 4)},
        }

    acc = {}
    for cfg in CONFIGS:
        rows = [r for r in recs if r["config"] == cfg]
        acc[cfg] = {"pooled": variants(rows)}
        for s in sorted({r["seed"] for r in rows}):
            acc[cfg][f"seed{s}"] = variants([r for r in rows if r["seed"] == s])
    gaps = {}
    vkeys = ["logged", "literal_rescue_upper_bound_all_noncap_unparsed_pass",
             "parsed_only", "diag_sanitized_toplevel_dropped"]
    for cfg in CONFIGS[1:]:
        gaps[cfg] = {}
        for scope in ["pooled", "seed1"]:
            gaps[cfg][scope] = {
                v: round(acc["SA"]["seed1"][v]["acc"] - acc[cfg][scope][v]["acc"], 4)
                for v in vkeys}
            gaps[cfg][scope]["literal_rescue"] = None

    table = []
    for cfg in CONFIGS:
        es = [e for e in entries if e["config"] == cfg]
        table.append({"config": cfg,
                      "unparsed": len(es),
                      "no_code": None, "truncated": None,
                      "code_present_unextracted": None,
                      "other_text_not_logged": sum(e["category"] == "other" for e in es),
                      "of_which_final_call_hit_cap": sum(
                          (e["hit_cap"] if isinstance(e["hit_cap"], bool)
                           else all(e["hit_cap"])) for e in es),
                      "rescued_passes": None})

    out = {
        "method": {
            "reader": "an AI assistant (Claude), unattended; no human checked any reading",
            "saw_workflow_label": True,
            "looked_at_canonical_solution": False,
            "text_read": False,
            "why_no_text_read": NO_TEXT + ". No record could be read, classified by "
                                "reading, or rescued; categories no_code / truncated / "
                                "code_present_unextracted are undetermined for all records.",
            "scorer": "harness.tasks.humaneval_score (sandbox harness/sandbox.py; macOS "
                      "sandbox-exec backend here, docker python:3.11-slim on the PC for "
                      "the logged scores), 10 s timeout as in the harness",
            "cap": "num_predict = 1024 for every record; hit_cap = out_tokens >= 1024 on "
                   "the final call(s): debate = the three round-1 agents, GE = the second "
                   "revision, HIER = the aggregator, SA = its single call",
            "sample_step4": "random.Random(0).sample(pop, 40), pop = DEBATE_N3_R1, GE_R2, "
                            "HIER_M3 records with a parsed pred and correct=False, in file "
                            "order",
            "diag_sanitized": "diagnostic, not a rescue: drop top-level statements other "
                              "than imports, defs, classes and assignments from the logged "
                              "pred (the model's own example calls, prints, asserts, "
                              "__main__ blocks), then score; no function is edited",
            "regenerate": "/usr/bin/python3 results/audit/humaneval_unparsed_audit.py",
        },
        "summary": {
            "counts_by_config_seed": counts,
            "counts_table": table,
            "per_sample_unparsed": per_sample,
            "extractor_signature_scan_parsed_records": scan,
            "step4_sample": {"n": 40, "mismatches": None,
                             "determinable": False,
                             "entry_stub_in_sample": sum(e["entry_stub"] for e in sample_entries),
                             "stub_kinds_in_sample": dict(Counter(
                                 e["stub_kind"].split(":")[0] for e in sample_entries
                                 if e["stub_kind"]))},
            "replication_of_logged_scores_on_this_mac": dict(repl),
            "accuracy": acc,
            "sa_minus_workflow_gap": gaps,
        },
        "records": entries,
        "step4_sample_records": sample_entries,
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out["summary"]["counts_table"], indent=1))
    print(json.dumps(out["summary"]["step4_sample"], indent=1))
    print("replication", dict(repl))
    for cfg in CONFIGS:
        p = acc[cfg]["pooled"]
        print(cfg, {k: (v["acc"] if v else None) for k, v in p.items()})
    print(json.dumps(gaps, indent=1))
    print(json.dumps(scan, indent=1))
    print(json.dumps(per_sample, indent=1))


if __name__ == "__main__":
    main()
