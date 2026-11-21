"""Compare regenerated outputs with the reference copies used by the paper (rev3).

The rev3 macro files and table fragments in paper/arxiv_rev3/ are copies of files written by
the emitters of stages 2 and 3 of reproduce/regenerate_all.sh. For each reference file this
script finds the regenerated file it was copied from and reports:
  EXACT          identical after dropping LaTeX comment lines
  SAME-NUMBERS   macro values (macro files) or the sequence of numbers (tables) are identical;
                 only text differs (captions and headers were edited for rev3)
  DIFF           a value differs (printed)
  DIFF-UNUSED    only macros that the rev3 manuscript and scripts do not use differ
                 (reproduce/rev3_used_macros.txt; e.g. stale router macros of an earlier
                 draft that the emitters still write); informational, not a failure
  MISSING        the regenerated file does not exist (run regenerate_all.sh first)
Files that stage 4 regenerates in place (rev3_macros.tex, math_easy_macros.tex, tables/rev3/,
tables/math_easy.json, ...) are checked against reproduce/rev3_reference.sha256, the hashes
of the files as used in the paper. Figures are not compared byte for byte (PDF metadata
differs); every figure script asserts its plotted numbers against the macros and fails
otherwise.

Usage: python3 reproduce/check_rev3.py [--only NAME ...]
Exit status 0 iff nothing is DIFF or MISSING.
"""
import hashlib
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R3 = "paper/arxiv_rev3"
AL = "paper/arxiv_long"

# reference file (relative to paper/arxiv_rev3) -> regenerated source (relative to ROOT)
COPIES = {
    "audit_macros.tex": f"{AL}/audit_macros.tex",
    "context_macros.tex": f"{AL}/context_macros.tex",
    "equivalence_macros.tex": f"{AL}/equivalence_macros.tex",
    "extra_noise_aware_macros.tex": f"{AL}/extra_noise_aware_macros.tex",
    "extra_workflows_macros.tex": f"{AL}/extra_workflows_macros.tex",
    "hard_math_macros.tex": f"{AL}/hard_math_macros.tex",
    "humaneval_audit_macros.tex": f"{AL}/humaneval_audit_macros.tex",
    "math_audit_macros.tex": f"{AL}/math_audit_macros.tex",
    "math_interaction_macros.tex": f"{AL}/math_interaction_macros.tex",
    "review_revision_macros.tex": f"{AL}/review_revision_macros.tex",
    "criterion_predictive_macros.tex": "paper/eaai/criterion_predictive_macros.tex",
    "evidence_macros.tex": "paper/eaai/evidence_macros.tex",
    "figure_macros.tex": "paper/eaai/figure_macros.tex",
    "humaneval_macros.tex": "paper/eaai/humaneval_macros.tex",
    "mmlu_pro_eng_macros.tex": "paper/eaai/mmlu_pro_eng_macros.tex",
    "throughput_macros.tex": "paper/eaai/throughput_macros.tex",
    "crosstask_macros.tex": "paper/crosstask_macros.tex",
    "results_macros.tex": "paper/results_macros.tex",
    "scale70b_macros.tex": "paper/scale70b_macros.tex",
    "rerun_macros.tex": "paper/aaibd/rerun_macros.tex",
    "data/pareto_membership.out": f"{AL}/scripts/pareto_membership.out",
    "tables/audit.json": f"{AL}/tables/audit.json",
    "tables/audit_drift_table.tex": f"{AL}/tables/audit_drift_table.tex",
    "tables/audit_extraction_table.tex": f"{AL}/tables/audit_extraction_table.tex",
    "tables/audit_rerun_table.tex": f"{AL}/tables/audit_rerun_table.tex",
    "tables/context.json": f"{AL}/tables/context.json",
    "tables/cost_verdict_table.tex": f"{AL}/tables/cost_verdict_table.tex",
    "tables/equivalence.json": f"{AL}/tables/equivalence.json",
    "tables/equivalence_table.tex": f"{AL}/tables/equivalence_table.tex",
    "tables/output_verdict_table.tex": f"{AL}/tables/output_verdict_table.tex",
    "tables/extra_domination_compact.tex": f"{AL}/tables/extra_domination_compact.tex",
    "tables/extra_domination_table.tex": f"{AL}/tables/extra_domination_table.tex",
    "tables/extra_noise_aware.json": f"{AL}/tables/extra_noise_aware.json",
    "tables/extra_workflows.json": f"{AL}/tables/extra_workflows.json",
    "tables/extra_workflows_table.tex": f"{AL}/tables/extra_workflows_table.tex",
    "tables/hard_math.json": f"{AL}/tables/hard_math.json",
    "tables/hard_math_table.tex": f"{AL}/tables/hard_math_table.tex",
    "tables/humaneval_audit.json": f"{AL}/tables/humaneval_audit.json",
    "tables/humaneval_audit_table.tex": f"{AL}/tables/humaneval_audit_table.tex",
    "tables/review_revision.json": f"{AL}/tables/review_revision.json",
    "tables/criterion_pooled_table.tex": "paper/eaai/criterion_pooled_table.tex",
    "tables/criterion_predictive_table.tex": "paper/eaai/criterion_predictive_table.tex",
    "tables/criterion_splits_table.tex": "paper/eaai/criterion_splits_table.tex",
    "tables/humaneval_table.tex": "paper/eaai/humaneval_table.tex",
    "tables/humaneval_supp_table.tex": "paper/eaai/humaneval_supp_table.tex",
    "tables/mmlu_pro_eng_table.tex": "paper/eaai/mmlu_pro_eng_table.tex",
    "tables/poster_main_table.tex": "paper/aaibd/main_table.tex",
    "tables/poster_cost_table.tex": "paper/aaibd/cost_table.tex",
    "tables/poster_domination_table.tex": "paper/aaibd/domination_table.tex",
    "tables/poster_robustness_table.tex": "paper/aaibd/robustness_table.tex",
    "tables/poster_round0_table.tex": "paper/aaibd/round0_table.tex",
    "tables/scale70b_table.tex": "paper/scale70b_table.tex",
    "tables/throughput_table.tex": "paper/eaai/throughput_table.tex",
    "tables/trunc_table.tex": "paper/trunc_table.tex",
}
# Table bodies that are hand-maintained copies (values cross-checked in
# paper/aaibd/revision_stats.json section J) or a hand-written comparison table.
NOT_REGENERATED = ["tables/latency_table.tex", "tables/scale_table.tex",
                   "tables/diversity_table.tex", "tables/positioning_table.tex"]

NUM = re.compile(r"[-+]?\d[\d{},]*\.?\d*")
MACRO = re.compile(r"\\(?:re)?newcommand\{\\([A-Za-z]+)\}\{(.*)\}\s*$")


def body(path):
    with open(path, encoding="utf-8") as f:
        return [l.rstrip() for l in f if not l.lstrip().startswith("%")]


def macros(lines):
    out = {}
    for l in lines:
        m = MACRO.match(l.strip())
        if m:
            out[m.group(1)] = m.group(2)
    return out


def numbers(lines):
    return [n.replace("{,}", "").replace(",", "") for l in lines for n in NUM.findall(l)]


USED = None


def used_macros():
    global USED
    if USED is None:
        p = os.path.join(ROOT, "reproduce", "rev3_used_macros.txt")
        USED = {l.strip() for l in open(p) if l.strip() and not l.startswith("#")} if os.path.exists(p) else None
    return USED


def compare(ref, new):
    a, b = body(ref), body(new)
    if a == b:
        return "EXACT", ""
    if ref.endswith(".json"):
        import json

        def norm(x):
            if isinstance(x, dict):
                return {k: norm(v) for k, v in x.items()}
            if isinstance(x, list):
                return sorted((norm(v) for v in x), key=lambda v: json.dumps(v, sort_keys=True))
            return x
        if norm(json.load(open(ref))) == norm(json.load(open(new))):
            return "SAME-NUMBERS", "(JSON equal up to list order: set iteration order varies)"
    ma, mb = macros(a), macros(b)
    if ma:
        diff = [k for k in ma if mb.get(k) != ma[k]]
        if not diff:
            return "SAME-NUMBERS", ""
        msg = "; ".join(f"\\{k}: paper {ma[k]!r} vs regenerated {mb.get(k)!r}" for k in diff[:6])
        used = used_macros()
        if used is not None and not any(k in used for k in diff):
            return "DIFF-UNUSED", msg
        return "DIFF", msg
    na, nb = numbers(a), numbers(b)
    if na == nb:
        return "SAME-NUMBERS", ""
    if ref.endswith(".tex"):
        # rev3 copies of tables may differ only in caption or header text: compare the
        # numbers of the data rows (lines ending in \\)
        ra = numbers([l for l in a if l.endswith("\\\\")])
        rb = numbers([l for l in b if l.endswith("\\\\")])
        if ra and ra == rb:
            return "SAME-NUMBERS", "(data rows)"
        # rev3 may also drop a footnote row: every numeric data row of the reference must
        # be a row of the regenerated table
        rows_a = [tuple(numbers([l])) for l in a if l.endswith("\\\\") and numbers([l])]
        rows_b = {tuple(numbers([l])) for l in b if l.endswith("\\\\") and numbers([l])}
        if rows_a and all(r in rows_b for r in rows_a):
            return "SAME-NUMBERS", f"(all {len(rows_a)} data rows present; regenerated has {len(rows_b)})"
    i = next((k for k, (x, y) in enumerate(zip(na, nb)) if x != y), min(len(na), len(nb)))
    return "DIFF", f"first differing number #{i}: paper {na[i:i+3]} vs regenerated {nb[i:i+3]}"


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    only = sys.argv[sys.argv.index("--only") + 1:] if "--only" in sys.argv else None
    counts, bad = {}, 0
    for ref_rel, src_rel in COPIES.items():
        if only and not any(o in ref_rel for o in only):
            continue
        ref, src = os.path.join(ROOT, R3, ref_rel), os.path.join(ROOT, src_rel)
        if not os.path.exists(src):
            status, note = "MISSING", src_rel
        else:
            status, note = compare(ref, src)
        counts[status] = counts.get(status, 0) + 1
        bad += status in ("DIFF", "MISSING")
        print(f"{status:13s} {R3}/{ref_rel}  <-  {src_rel} {note}")
    manifest = os.path.join(ROOT, "reproduce", "rev3_reference.sha256")
    if os.path.exists(manifest) and not only:
        for line in open(manifest):
            h, rel = line.split(None, 1)
            rel = rel.strip()
            p = os.path.join(ROOT, rel)
            status = "EXACT" if os.path.exists(p) and sha(p) == h else "DIFF"
            counts[status] = counts.get(status, 0) + 1
            bad += status == "DIFF"
            print(f"{status:13s} {rel}  (regenerated in place; sha256 vs reference)")
    for rel in NOT_REGENERATED:
        print(f"{'NOT-REGEN':13s} {R3}/{rel}  (copied body; see file header)")
    print("summary:", ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
