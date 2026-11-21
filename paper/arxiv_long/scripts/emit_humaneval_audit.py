"""HumanEval unparsed-program audit: numbers for the paper (third-review revision).

Inputs (read only; no model calls, no sandbox runs):
  results/humaneval_matrix_pc.jsonl                     the logged run
  results/audit/humaneval_unparsed_adjudication.json    blocks `method`, `summary`
      (written by results/audit/humaneval_unparsed_audit.py, commit fbc6e17e)

Recomputed here from the raw log and asserted equal to the audit JSON:
  * no call text was logged (calls carry token counts and timings only; `raw` is
    a list of extracted programs for SC and debate and None otherwise);
  * unparsed records (pred None or empty) per config, and how many of them had
    every final call at the 1024-token cap (debate: the 3 round-1 agents;
    SC: all samples; SA, GE, HIER: the last call);
  * per-sample unparsed programs for SC samples and debate round-1 agents;
  * accuracy variants, pooled over seeds: logged; upper bound if every unparsed
    record whose final call did not hit the cap had passed; parsed records only.
    Gaps: the single call (one greedy seed) minus the workflow, as in the audit
    JSON, and SC k=8 (two seeds, the paper's fixed comparator) minus the
    workflow. These are bounds and sensitivity values, not intervals.
Taken from the audit JSON without recomputation (they need the extractor scan or
the sandbox): parsed wrong records whose extracted block is test or usage code
(entry function left as a stub), and the rescoring of logged programs on the Mac.

Writes tables/humaneval_audit.json, tables/humaneval_audit_table.tex and
humaneval_audit_macros.tex. Usage: python3 scripts/emit_humaneval_audit.py
"""
import json
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
SRC = os.path.join(ROOT, "results", "humaneval_matrix_pc.jsonl")
AUD = os.path.join(ROOT, "results", "audit", "humaneval_unparsed_adjudication.json")
CONFIGS = ["SA", "SC_k2", "SC_k4", "SC_k8", "DEBATE_N3_R1", "GE_R2", "HIER_M3"]
COORD = ["DEBATE_N3_R1", "GE_R2", "HIER_M3"]
SC = ["SC_k2", "SC_k4", "SC_k8"]


def unparsed(r):
    return r["pred"] in (None, "")


def final_calls(r):
    c, cfg = r["calls"], r["config"]
    if cfg.startswith("DEBATE"):
        return c[3:6]
    if cfg.startswith("SC"):
        return c
    return [c[-1]]


def all_cap(r):
    return all(c["out_tokens"] >= r["num_predict"] for c in final_calls(r))


def main():
    recs = [json.loads(l) for l in open(SRC) if l.strip()]
    aud = json.load(open(AUD))
    S = aud["summary"]
    assert aud["method"]["text_read"] is False
    # 1. no output text logged
    assert all(set(c) <= {"in_tokens", "out_tokens", "latency_s", "t0", "t1", "energy_j"}
               for r in recs for c in r["calls"]), "a call carries more than counts and timings"
    assert all(r["raw"] is None for r in recs if r["config"] in ("SA", "GE_R2", "HIER_M3"))
    assert all(isinstance(r["raw"], list) for r in recs if r["config"] in SC + ["DEBATE_N3_R1"])
    by = defaultdict(list)
    for r in recs:
        by[r["config"]].append(r)
    out = {"configs": {}}
    # 2. unparsed records and cap hits
    ref = {e["config"]: e for e in S["counts_table"]}
    for c in CONFIGS:
        rows = by[c]
        u = [r for r in rows if unparsed(r)]
        capped = sum(all_cap(r) for r in u)
        assert len(u) == ref[c]["unparsed"] and capped == ref[c]["of_which_final_call_hit_cap"], c
        parsed = [r for r in rows if not unparsed(r)]
        n, k = len(rows), sum(bool(r["correct"]) for r in rows)
        out["configs"][c] = {
            "records": n, "unparsed": len(u), "unparsed_at_cap": capped,
            "acc_logged": 100 * k / n,
            "acc_upper": 100 * (k + len(u) - capped) / n,
            "acc_parsed": 100 * sum(bool(r["correct"]) for r in parsed) / len(parsed)}
    # 3. per-sample unparsed programs
    ps = {}
    for c in SC + ["DEBATE_N3_R1"]:
        n = sum(len(r["raw"]) for r in by[c])
        nn = sum(a in (None, "") for r in by[c] for a in r["raw"])
        assert (n, nn) == (S["per_sample_unparsed"][c]["samples"], S["per_sample_unparsed"][c]["unparsed"]), c
        ps[c] = {"samples": n, "unparsed": nn}
    out["per_sample"] = ps
    # 4. gaps (points): SA minus workflow (as the audit JSON) and SC k=8 minus workflow
    keys = {"acc_logged": "logged", "acc_upper": "literal_rescue_upper_bound_all_noncap_unparsed_pass",
            "acc_parsed": "parsed_only"}
    A = out["configs"]
    gaps = {}
    for ref_c in ("SA", "SC_k8"):
        for c in COORD:
            g = {k: A[ref_c][k] - A[c][k] for k in keys}
            if ref_c == "SA":
                for k, jk in keys.items():
                    assert abs(g[k] / 100 - S["sa_minus_workflow_gap"][c]["pooled"][jk]) < 1.5e-4, (c, k)  # the JSON rounds each accuracy to 4 decimals
            gaps[f"{ref_c}|{c}"] = g
    out["gaps"] = gaps
    # 5. extractor scan and rescoring, from the audit JSON
    scan = S["extractor_signature_scan_parsed_records"]
    stub = {c: scan[c].get("stub_kinds", {}).get("block_is_test_or_usage_code", 0) for c in CONFIGS}
    out["test_block_wrong"] = stub
    out["rescored_agree"] = S["replication_of_logged_scores_on_this_mac"]["agree"]
    json.dump(out, open(os.path.join(HERE, "..", "tables", "humaneval_audit.json"), "w"), indent=1)

    def f1(x, sign=True):
        x = round(x, 1) + 0.0
        return f"{'+' if sign and x > 0 else ''}{x:.1f}"
    NM = {"SA": "SA", "SC_k8": "SC $k{=}8$", "DEBATE_N3_R1": "Debate", "GE_R2": "GE", "HIER_M3": "HIER"}
    KEY = {"SA": "Sa", "SC_k8": "Sc", "DEBATE_N3_R1": "Deb", "GE_R2": "Ge", "HIER_M3": "Hier"}
    # table
    L = ["% AUTO-GENERATED by paper/arxiv_long/scripts/emit_humaneval_audit.py. Do not edit."]
    for c in ["SA", "SC_k8"] + COORD:
        e = A[c]
        cells = [NM[c], f"{e['unparsed']}/{e['records']}", str(e["unparsed_at_cap"]),
                 str(stub[c])]
        for ref_c in ("SA", "SC_k8"):
            if c in COORD:
                g = gaps[f"{ref_c}|{c}"]
                cells += [f"${f1(g['acc_logged'])}$", f"${f1(g['acc_upper'])}$", f"${f1(g['acc_parsed'])}$"]
            else:
                cells += ["--", "--", "--"]
        L.append(" & ".join(cells) + " \\\\")
    open(os.path.join(HERE, "..", "tables", "humaneval_audit_table.tex"), "w").write("\n".join(L) + "\n")
    # macros
    M = [L[0]]
    add = lambda k, v: M.append(f"\\newcommand{{\\He{k}}}{{{v}}}")  # noqa: E731
    for c in ["SA", "SC_k8"] + COORD:
        add(f"Unp{KEY[c]}", A[c]["unparsed"])
        add(f"Rec{KEY[c]}", A[c]["records"])
        add(f"Cap{KEY[c]}", A[c]["unparsed_at_cap"])
        add(f"Stub{KEY[c]}", stub[c])
    sc_unp = sum(A[c]["unparsed"] for c in SC)
    add("UnpScAll", sc_unp)
    tot = sum(A[c]["unparsed"] for c in ["SA"] + SC + COORD)
    totcap = sum(A[c]["unparsed_at_cap"] for c in ["SA"] + SC + COORD)
    add("UnpTot", tot)
    add("UnpCoord", sum(A[c]["unparsed"] for c in COORD))
    add("CapCoord", sum(A[c]["unparsed_at_cap"] for c in COORD))
    add("UnpCoordNotCap", sum(A[c]["unparsed"] - A[c]["unparsed_at_cap"] for c in COORD))
    add("UnpNotCap", tot - totcap)
    n = sum(ps[c]["samples"] for c in SC)
    nn = sum(ps[c]["unparsed"] for c in SC)
    add("PsScUnp", nn)
    add("PsScN", f"{n:,}".replace(",", "{,}"))
    add("PsScPct", f"{100 * nn / n:.1f}")
    d = ps["DEBATE_N3_R1"]
    add("PsDebUnp", d["unparsed"])
    add("PsDebN", d["samples"])
    add("PsDebPct", f"{100 * d['unparsed'] / d['samples']:.1f}")
    scs = sorted(stub[c] for c in SC)
    add("StubScRange", f"{scs[0]} to {scs[-1]}" if scs[0] != scs[-1] else str(scs[0]))
    add("Rescored", out["rescored_agree"])
    for ref_c in ("SA", "SC_k8"):
        for c in COORD:
            g = gaps[f"{ref_c}|{c}"]
            for k, t in (("acc_logged", "Log"), ("acc_upper", "Up"), ("acc_parsed", "Par")):
                add(f"Gap{KEY[ref_c]}{KEY[c]}{t}", f"${f1(g[k])}$")
    # statements in the text
    assert all(gaps[f"SA|{c}"]["acc_upper"] > 8 for c in ("DEBATE_N3_R1", "HIER_M3"))
    assert min(gaps[f"SA|{c}"]["acc_upper"] for c in ("DEBATE_N3_R1", "HIER_M3")) == min(
        gaps[f"SA|{c}"]["acc_upper"] for c in COORD if c != "GE_R2")
    add("MinUpDebHier", f"{min(gaps[f'SA|{c}']['acc_upper'] for c in ('DEBATE_N3_R1', 'HIER_M3')):.1f}")
    assert gaps["SC_k8|GE_R2"]["acc_upper"] < 0  # GE could exceed SC k=8 under the bound
    assert A["SA"]["unparsed_at_cap"] == A["SA"]["unparsed"] == 1
    assert all(A[c]["unparsed"] == 0 for c in SC)
    open(os.path.join(HERE, "..", "humaneval_audit_macros.tex"), "w").write("\n".join(M) + "\n")


if __name__ == "__main__":
    main()
