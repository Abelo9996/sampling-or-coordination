"""GSM8K raw-output rerun, cap-2048 check and extraction audit (third revision).

Reads only logged records and the hand adjudication (no model calls):
  results/power/gsm8k_qwen3b_s{1,2,3}.jsonl         RTX 4060, Ollama 0.40.1, cap 640,
      raw text of every call; only the first 100 items of the GSM8K loader order
      are used (SA: seed 1 only, in the s1 file);
  results/power/gsm8k_cap2048_qwen3b_s{1,2,3}.jsonl same items and seeds, cap 2048,
      SC k=8 and debate only (model alias with num_ctx 8192, same weights);
  results/gsm8k_qwen3b_coord_trace.jsonl           Apple M4 rerun, seed 1, the 80
      matrix items, with ref_pred = the July matrix prediction. The audit read
      this file while SC k=8 was still being written: it covers debate, GE and
      HIER (80 records each) and the first 12 SC k=8 records;
  results/gsm8k_matrix.jsonl, results/mmlu_matrix.jsonl   July matrix records
      (no call texts), for the HIER no-answer counts quoted in the paper;
  results/audit/gsm8k_extraction_{cases,denominators,adjudication}.json and
      gsm8k_rescored.json: the flagged calls and their hand reading.

Answer-bearing calls: every call of SA and SC; the three round-1 calls of
debate (idx 3, 4, 5); the last call of GE and HIER. A call is a "template-only"
output if its whole text is "The answer is N." with a literal N. A call is
flagged for hand reading if the harness read no number, or no explicitly stated
answer is found (last of: "answer is <num>", "N = <num>", "\\boxed{<num>}"), or
the stated answer differs from the harness reading, or the call hit the cap.
This script re-derives the flagged set and the denominators from the raw texts
and asserts that they equal the stored ones, checks every adjudicated call
against the raw record, and rescores each record with the hand reading in place
of the harness reading for the adjudicated calls (majority vote for SC and
debate, as in harness/topologies.py). The rescored correctness is asserted equal
to results/audit/gsm8k_rescored.json.

Paired tests: the loop of harness.stats.hier_paired_test (items, then seeds
within item; 10,000 replicates, seed 0), via emit_equivalence.hier_boot, which
also returns the bootstrap distribution (one test is checked against
hier_paired_test itself). SA has one greedy seed; its value is repeated to three
seeds, which leaves its per-item mean unchanged. 95% interval = 2.5th and 97.5th
percentiles; 90% interval = 5th and 95th, whose upper limit for coordination
minus SC k=8 is the one-sided 95% bound on the gain (the 5-point margin of
Table equiv, chosen after seeing the matrix data).

Writes tables/audit.json, tables/audit_rerun_table.tex,
tables/audit_extraction_table.tex and audit_macros.tex.
Usage: python3 scripts/emit_audit.py [--render] [--workers N]
"""
import json
import os
import re
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from harness import stats as S  # noqa: E402
from harness import tasks  # noqa: E402
from harness.topologies import _majority  # noqa: E402
from emit_equivalence import hier_boot  # noqa: E402

B = 10000
SEED = 0
MARGIN = 5.0
T = tasks.get_task("gsm8k")
SEEDS = (1, 2, 3)
CFGS = ["SA", "SC_k8", "DEBATE_N3_R1", "GE_R2", "HIER_M3"]
COORD = ["DEBATE_N3_R1", "GE_R2", "HIER_M3"]
RES = os.path.join(ROOT, "results")
AUD = os.path.join(RES, "audit")
OUT = os.path.join(HERE, "..")
MAC_SC_AUDITED = 12  # SC k=8 records of the Mac trace present when the audit ran

PAT = re.compile(r"answer is\s*:?\s*(-?\d[\d,]*(?:\.\d+)?)", re.I)  # = tasks.gsm8k_extract
NUM = r"(-?\$?\\?\$?\d[\d,]*(?:\.\d+)?)"
STATED = [re.compile(p, re.I) for p in (
    r"answer is\s*:?\s*\**\s*\\?\(?\s*\\?\$?\s*" + NUM,
    r"\bN\s*(?:=|is|:)\s*\**\s*\\?\$?\s*" + NUM,
    r"answer is\s*:?\s*\\?\(?\s*N\s*=\s*\\?\$?\s*" + NUM,
    r"\\boxed\{\s*\\?\$?\s*" + NUM,
)]
TEMPLATE = re.compile(r"\s*The answer is N\.?\s*")


def stated(text):
    """Last explicitly stated answer by position, independent of the harness rule."""
    best = None
    for p in STATED:
        for m in p.finditer(text):
            if best is None or m.start() > best[0]:
                best = (m.start(), m.group(1))
    if best is None:
        return None
    return tasks._to_number(best[1].replace("$", "").replace("\\", ""))


def bearing(cfg, n):
    if cfg == "SA" or cfg.startswith("SC"):
        return list(range(n))
    if cfg.startswith("DEBATE"):
        return [3, 4, 5]
    return [n - 1]


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def items100():
    return [it["id"] for it in T["load"](n=100, split="all")]


def data():
    keep = set(items100())
    pc = [r for s in SEEDS for r in load(os.path.join(RES, "power", f"gsm8k_qwen3b_s{s}.jsonl"))
          if r["item_id"] in keep]
    hi = [r for s in SEEDS for r in load(os.path.join(RES, "power", f"gsm8k_cap2048_qwen3b_s{s}.jsonl"))
          if r["item_id"] in keep]
    mac_all = load(os.path.join(RES, "gsm8k_qwen3b_coord_trace.jsonl"))
    mac = [r for r in mac_all if r["config"] != "SC_k8"] + \
          [r for r in mac_all if r["config"] == "SC_k8"][:MAC_SC_AUDITED]
    assert all(r["cap"] == 640 and r["host"] == "rtx4060-pc" for r in pc)
    assert all(r["cap"] == 2048 and r["host"] == "rtx4060-pc" for r in hi)
    assert all(r["cap"] == 640 and r["seed"] == 1 for r in mac_all)
    return pc, hi, mac, mac_all


def by_cfg(rs):
    d = defaultdict(list)
    for r in rs:
        d[r["config"]].append(r)
    return d


def arrays(rs, cfg, field="correct", repeat=None):
    """item -> per-seed array; SA (one seed) repeated to `repeat` seeds."""
    d = defaultdict(list)
    for r in rs:
        if r["config"] == cfg:
            d[r["item_id"]].append(float(r[field]))
    if repeat:
        d = {i: v * repeat if len(v) == 1 else v for i, v in d.items()}
    return {i: np.asarray(v, float) for i, v in d.items()}


def mean_of(arr):
    return float(np.mean([v.mean() for v in arr.values()]))


# ==== audit checks
def audit(pc, mac):
    srcs = {"gpu": pc, "mac": mac}
    cases = json.load(open(os.path.join(AUD, "gsm8k_extraction_cases.json")))
    adj = json.load(open(os.path.join(AUD, "gsm8k_extraction_adjudication.json")))
    den_ref = json.load(open(os.path.join(AUD, "gsm8k_extraction_denominators.json")))
    key = lambda c: (c["source"], c["config"], c["item_id"], c["seed"], c["call_idx"])  # noqa: E731
    ref_flags = {key(c): sorted(c["flags"]) for c in cases}
    A = {key(a): a for a in adj}
    assert set(A) == set(ref_flags) and len(A) == len(adj) == len(cases)
    den, tmpl, flags, first_last = Counter(), Counter(), {}, Counter()
    recmap = {}
    for name, rs in srcs.items():
        for r in rs:
            recmap[(name, r["config"], r["item_id"], r["seed"])] = r
            for i in bearing(r["config"], len(r["calls"])):
                c = r["calls"][i]
                k = f"{name}|{r['config']}"
                den[k] += 1
                t, a = c["text"], c["extracted"]
                assert str(a) == str(T["extract"](t))
                ms = PAT.findall(t)
                if ms and str(tasks._to_number(ms[0])) != str(tasks._to_number(ms[-1])):
                    first_last[k] += 1  # harness takes the first match
                if TEMPLATE.fullmatch(t):
                    tmpl[k] += 1
                    continue
                s, f = stated(t), []
                if a is None:
                    f.append("harness_none")
                elif s is None:
                    f.append("no_explicit_statement")
                elif abs(float(s) - float(a)) > 1e-6:
                    f.append("readings_differ")
                if c["hit_cap"]:
                    f.append("hit_cap")
                if f:
                    flags[(name, r["config"], r["item_id"], r["seed"], i)] = sorted(f)
    assert dict(den) == den_ref["answer_bearing_calls"], (dict(den), den_ref)
    assert dict(tmpl) == den_ref["template_only_outputs"], (dict(tmpl), den_ref)
    assert flags == ref_flags, "flag rule does not reproduce the stored cases"
    for c in cases:  # stored texts and readings are the raw ones
        call = recmap[key(c)[:4]]["calls"][c["call_idx"]]
        assert call["text"] == c["text"] and str(call["extracted"]) == str(c["harness_reading"])
    tab = defaultdict(Counter)
    for k, a in A.items():
        g = tab[f"{k[0]}|{k[1]}"]
        g["flagged"] += 1
        cat = a["category"]
        hc, uc = bool(a["harness_correct"]), bool(a["human_correct"])
        assert str(a["harness_reading"]) == str(recmap[k[:4]]["calls"][k[4]]["extracted"])
        if cat == "A_same":
            assert hc == uc and str(a["human_answer"]) == str(a["harness_reading"])
            g["A"] += 1
        elif cat == "B_misread":
            g["B"] += 1
            g["B_against" if (uc and not hc) else ("B_for" if (hc and not uc) else "B_either")] += 1
        else:
            assert cat == "C_no_answer" and a["human_answer"] is None and not uc
            g["C"] += 1
            g["C_trunc"] += bool(a["truncated"])
            g["C_luck"] += hc
    return srcs, A, den, tmpl, tab, first_last


def rescore(srcs, A):
    """Per record: (harness correct, hand-reading correct)."""
    out = {}
    for name, rs in srcs.items():
        for r in rs:
            idx = bearing(r["config"], len(r["calls"]))
            hr = [r["calls"][i]["extracted"] for i in idx]
            ur = []
            for i, h in zip(idx, hr):
                k = (name, r["config"], r["item_id"], r["seed"], i)
                ur.append(A[k]["human_answer"] if k in A else h)
            vote = len(idx) > 1
            ph = _majority(hr) if vote else hr[0]
            pu = _majority(ur) if vote else ur[0]
            assert str(ph) == str(r["pred"]) and bool(T["score"](ph, r["gold"])) == bool(r["correct"])
            out[(name, r["config"], r["item_id"], r["seed"])] = (bool(r["correct"]), bool(T["score"](pu, r["gold"])))
    ref = json.load(open(os.path.join(AUD, "gsm8k_rescored.json")))
    n = 0
    for (name, cfg, it, sd), v in out.items():
        rv = ref[name][cfg].get(f"{it}|{sd}")
        if rv is not None:
            assert tuple(rv) == v, (name, cfg, it, sd, rv, v)
            n += 1
    return out, n


# ==== bootstrap
def job(args):
    name, a, b = args
    obs, p, boot = hier_boot(a, b)
    lo95, hi95, lo90, hi90 = np.percentile(boot, [2.5, 97.5, 5, 95]) * 100
    return name, {"diff": obs * 100, "p": p, "ci95": [float(lo95), float(hi95)],
                  "ci90": [float(lo90), float(hi90)]}


def check_job(args):
    a, b = args
    return S.hier_paired_test(a, b, B=B, seed=SEED)


def compute(workers):
    pc, hi, mac, mac_all = data()
    srcs, A, den, tmpl, tab, first_last = audit(pc, mac)
    sc, n_ref = rescore(srcs, A)
    for r in pc:
        r["correct_h"] = sc[("gpu", r["config"], r["item_id"], r["seed"])][1]
    for r in mac:
        r["correct_h"] = sc[("mac", r["config"], r["item_id"], r["seed"])][1]
    out = {"meta": {"B": B, "seed": SEED, "margin": MARGIN, "doc": __doc__.split("\n\n")[0],
                    "n_items": len(items100()), "rescored_records_checked": n_ref}}
    P, H = by_cfg(pc), by_cfg(hi)
    assert {c: len(v) for c, v in P.items()} == {"SA": 100, "SC_k8": 300, "DEBATE_N3_R1": 300,
                                                 "GE_R2": 300, "HIER_M3": 300}
    assert {c: len(v) for c, v in H.items()} == {"SC_k8": 300, "DEBATE_N3_R1": 300}
    cor = {c: arrays(pc, c, repeat=3) for c in CFGS}
    corh = {c: arrays(pc, c, "correct_h", repeat=3) for c in CFGS}
    cor2 = {c: arrays(hi, c) for c in ("SC_k8", "DEBATE_N3_R1")}
    for d in list(cor.values()) + list(corh.values()) + list(cor2.values()):
        assert len(d) == 100 and all(len(v) == 3 for v in d.values())
    # descriptive
    cfg = {}
    for c in CFGS:
        rs = P[c]
        cfg[c] = {"acc": 100 * mean_of(cor[c]), "acc_hand": 100 * mean_of(corh[c]),
                  "total_tokens": float(np.mean([r["total_tokens"] for r in rs])),
                  "out_tokens": float(np.mean([r["out_tokens"] for r in rs])),
                  "records": len(rs), "no_answer": sum(r["pred"] is None for r in rs)}
    cfg2 = {}
    for c in ("SC_k8", "DEBATE_N3_R1"):
        rs = H[c]
        cfg2[c] = {"acc": 100 * mean_of(cor2[c]),
                   "total_tokens": float(np.mean([r["total_tokens"] for r in rs])),
                   "out_tokens": float(np.mean([r["out_tokens"] for r in rs])),
                   "records": len(rs), "no_answer": sum(r["pred"] is None for r in rs)}

    def caps(rs, idxs):
        n = h = rec = 0
        for r in rs:
            sel = r["calls"] if idxs is None else [r["calls"][j] for j in idxs]
            k = sum(bool(c["hit_cap"]) for c in sel)
            n, h, rec = n + len(sel), h + k, rec + (k > 0)
        return {"calls": n, "hit": h, "records_with_hit": rec, "records": len(rs)}
    cap = {"640": {"debate_final": caps(P["DEBATE_N3_R1"], [3, 4, 5]),
                   "debate_all": caps(P["DEBATE_N3_R1"], None), "sc8": caps(P["SC_k8"], None)},
           "2048": {"debate_final": caps(H["DEBATE_N3_R1"], [3, 4, 5]),
                    "debate_all": caps(H["DEBATE_N3_R1"], None), "sc8": caps(H["SC_k8"], None)}}
    # HIER no-answer output: fresh reruns vs July records (no texts in July logs)
    jul = load(os.path.join(RES, "gsm8k_matrix.jsonl"))
    mm = load(os.path.join(RES, "mmlu_matrix.jsonl"))
    hier = {}
    for nm, rs in (("pc", P["HIER_M3"]), ("mac", by_cfg(mac)["HIER_M3"]),
                   ("july_gsm8k", by_cfg(jul)["HIER_M3"]), ("july_mmlu", by_cfg(mm)["HIER_M3"])):
        ans = [r for r in rs if r["pred"] is not None]
        e = {"records": len(rs), "no_answer": len(rs) - len(ans),
             "acc": 100 * np.mean([bool(r["correct"]) for r in rs]),
             "acc_answered": 100 * np.mean([bool(r["correct"]) for r in ans])}
        if "calls" in rs[0]:
            e["template_only"] = sum(bool(TEMPLATE.fullmatch(r["calls"][-1]["text"])) for r in rs)
            assert e["template_only"] == e["no_answer"]
        hier[nm] = e
    sa1 = {r["item_id"]: bool(r["correct"]) for r in P["SA"]}
    ans = [r for r in P["HIER_M3"] if r["pred"] is not None]
    na = [r for r in P["HIER_M3"] if r["pred"] is None]
    n = len(P["HIER_M3"])
    hier["pc"]["sa_acc_same_answered"] = 100 * np.mean([sa1[r["item_id"]] for r in ans])
    hier["pc"]["loss_from_no_answer_pts"] = -100 * sum(sa1[r["item_id"]] for r in na) / n
    hier["pc"]["loss_from_answered_pts"] = 100 * sum(bool(r["correct"]) - sa1[r["item_id"]] for r in ans) / n
    julsa = {r["item_id"]: bool(r["correct"]) for r in by_cfg(jul)["SA"]}
    mans = [r for r in by_cfg(mac)["HIER_M3"] if r["pred"] is not None]
    hier["mac"]["july_sa_acc_same_answered"] = 100 * np.mean([julsa[r["item_id"]] for r in mans])
    for c in CFGS:
        hier.setdefault("july_acc", {})[c] = 100 * np.mean([bool(r["correct"]) for r in by_cfg(jul)[c]])
    # Mac rerun vs July predictions (same items, seed 1)
    repro = {}
    for c, rs in by_cfg(mac_all).items():
        repro[c] = {"match": sum(str(r["pred"]) == str(r["ref_pred"]) for r in rs), "n": len(rs),
                    "acc": 100 * np.mean([bool(r["correct"]) for r in rs]),
                    "ref_acc": 100 * np.mean([bool(r["ref_correct"]) for r in rs])}
    jul1 = {(r["config"], r["item_id"]): r["pred"] for r in jul if r["seed"] == 1}
    for r in mac_all:  # ref_pred is the July seed-1 prediction
        assert str(jul1[(r["config"], r["item_id"])]) == str(r["ref_pred"])
    mac_hand = {c: {"acc": 100 * np.mean([bool(r["correct"]) for r in rs]),
                    "acc_hand": 100 * np.mean([bool(r["correct_h"]) for r in rs]), "n": len(rs)}
                for c, rs in by_cfg(mac).items()}
    # Drift vs item mix: the RTX rerun restricted to the 80 matrix items, next to
    # the July matrix (M4, two seeds; SA one greedy seed) on the same items, and the
    # rerun on the 20 other items.
    mat_ids = sorted({r["item_id"] for r in jul})
    assert len(mat_ids) == 80 and set(mat_ids) <= set(items100())
    m80 = set(mat_ids)
    sub = lambda d, keep: {i: v for i, v in d.items() if i in keep}  # noqa: E731
    o20 = set(items100()) - m80
    drift = {}
    for c in CFGS:
        jm = arrays([r for r in jul if r["config"] == c], c)
        assert len(jm) == 80
        drift[c] = {"matrix80": 100 * mean_of(jm), "rerun80": 100 * mean_of(sub(cor[c], m80)),
                    "rerun20": 100 * mean_of(sub(cor[c], o20)), "rerun100": cfg[c]["acc"],
                    "matrix_seeds": sorted({r["seed"] for r in jul if r["config"] == c})}
    for c in COORD + ["SA"]:
        drift[c]["sc8_minus_matrix80"] = drift["SC_k8"]["matrix80"] - drift[c]["matrix80"]
        drift[c]["sc8_minus_rerun80"] = drift["SC_k8"]["rerun80"] - drift[c]["rerun80"]
    out.update({"configs_640": cfg, "configs_2048": cfg2, "cap_hits": cap, "hier": hier,
                "drift80": drift,
                "mac_repro": repro, "mac_hand": mac_hand,
                "audit": {"answer_bearing_calls": dict(den), "template_only": dict(tmpl),
                          "first_match_differs_from_last": dict(first_last),
                          "by_source_config": {k: dict(v) for k, v in tab.items()}}})
    # paired tests
    plan = []
    for c in COORD + ["SA"]:
        plan.append((f"640|SC_k8|{c}", cor["SC_k8"], cor[c]))
        plan.append((f"hand|SC_k8|{c}", corh["SC_k8"], corh[c]))
    for c in COORD:
        plan.append((f"640|{c}|SA", cor[c], cor["SA"]))
        plan.append((f"hand|{c}|SA", corh[c], corh["SA"]))
    for c in COORD + ["SA"]:  # the rerun on the 80 matrix items only
        plan.append((f"r80|SC_k8|{c}", sub(cor["SC_k8"], m80), sub(cor[c], m80)))
    plan.append(("2048|SC_k8|DEBATE_N3_R1", cor2["SC_k8"], cor2["DEBATE_N3_R1"]))
    plan.append(("2048-640|DEBATE_N3_R1", cor2["DEBATE_N3_R1"], cor["DEBATE_N3_R1"]))
    plan.append(("2048-640|SC_k8", cor2["SC_k8"], cor["SC_k8"]))
    with Pool(workers) as pool:
        chk = pool.apply_async(check_job, ((cor["SC_k8"], cor["DEBATE_N3_R1"]),))
        res = dict(pool.map(job, plan, chunksize=1))
        chk = chk.get()
    t = res["640|SC_k8|DEBATE_N3_R1"]
    assert abs(chk[0] - t["diff"]) < 1e-9 and abs(chk[1] - t["p"]) < 1e-12 and \
        abs(chk[2] - t["ci95"][0]) < 1e-9 and abs(chk[3] - t["ci95"][1]) < 1e-9, (chk, t)
    out["meta"]["check_hier_paired_test"] = list(chk)
    out["tests"] = res
    json.dump(out, open(os.path.join(OUT, "tables", "audit.json"), "w"), indent=1, default=float)
    return out


# ==== rendering
def f1(x, sign=True):
    x = round(x, 1) + 0.0
    return f"{'+' if (sign and x > 0) else ''}{x:.1f}"


def iv(lo, hi):
    return f"$[{f1(lo)},{f1(hi)}]$"


def pv(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}" if p < 0.1 else f"{p:.2f}"


def comma(x):
    return f"{x:,.0f}".replace(",", "{,}")


NM = {"SA": "SA", "SC_k8": "SC $k{=}8$", "DEBATE_N3_R1": "Debate", "GE_R2": "GE", "HIER_M3": "HIER"}
KEY = {"SA": "Sa", "SC_k8": "Sc", "DEBATE_N3_R1": "Deb", "GE_R2": "Ge", "HIER_M3": "Hier"}
WORD = {0: "no", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven",
        8: "eight", 9: "nine", 10: "ten"}


def render(out):
    hdr = "% AUTO-GENERATED by paper/arxiv_long/scripts/emit_audit.py. Do not edit."
    R, R2, Tt = out["configs_640"], out["configs_2048"], out["tests"]
    dash = "--"

    def cell(name):
        t = Tt[name]
        return f"${f1(t['diff'])}$ {{\\scriptsize {iv(*t['ci95'])}}}"
    L = [hdr]
    for c in CFGS:
        r = R[c]
        vs_sc = cell(f"640|SC_k8|{c}") if c != "SC_k8" else dash
        vs_sa = cell(f"640|{c}|SA") if c in COORD else dash
        L.append(" & ".join([NM[c], "640", f"{r['acc']:.1f}", f"{r['acc_hand']:.1f}",
                             comma(r["total_tokens"]), comma(r["out_tokens"]), vs_sc, vs_sa]) + " \\\\")
    L.append("\\midrule")
    for c in ("SC_k8", "DEBATE_N3_R1"):
        r = R2[c]
        vs_sc = cell("2048|SC_k8|DEBATE_N3_R1") if c != "SC_k8" else dash
        L.append(" & ".join([NM[c], "2048", f"{r['acc']:.1f}", dash, comma(r["total_tokens"]),
                             comma(r["out_tokens"]), vs_sc, dash]) + " \\\\")
    open(os.path.join(OUT, "tables", "audit_rerun_table.tex"), "w").write("\n".join(L) + "\n")

    # drift vs item mix: matrix (July, M4) and RTX rerun on the same 80 items
    D = out["drift80"]
    L = [hdr]
    for c in CFGS:
        d = D[c]
        gm = f"${f1(d['sc8_minus_matrix80'])}$" if c != "SC_k8" else dash
        gr = cell(f"r80|SC_k8|{c}") if c != "SC_k8" else dash
        L.append(" & ".join([NM[c], f"{d['matrix80']:.1f}", f"{d['rerun80']:.1f}",
                             f"{d['rerun20']:.1f}", f"{d['rerun100']:.1f}", gm, gr]) + " \\\\")
    open(os.path.join(OUT, "tables", "audit_drift_table.tex"), "w").write("\n".join(L) + "\n")

    au = out["audit"]
    tab, den, tm = au["by_source_config"], au["answer_bearing_calls"], au["template_only"]
    L = [hdr]
    tot = Counter()
    for src, lab in (("gpu", "RTX~4060"), ("mac", "Apple M4")):
        cf = ["SA", "SC_k8", "DEBATE_N3_R1", "GE_R2", "HIER_M3"] if src == "gpu" else \
             ["SC_k8", "DEBATE_N3_R1", "GE_R2", "HIER_M3"]
        first = True
        for c in cf:
            k = f"{src}|{c}"
            g = Counter(tab.get(k, {}))
            row = {"calls": den[k], "tmpl": tm.get(k, 0), **g}
            tot.update(row)
            nm = NM[c] + ("$^{*}$" if (src == "mac" and c == "SC_k8") else "")
            L.append(" & ".join([lab if first else "", nm] + [comma(row.get(x, 0)) for x in
                     ("calls", "tmpl", "flagged", "A", "B", "B_against", "C", "C_trunc", "C_luck")]) + " \\\\")
            first = False
        L.append("\\midrule")
    L.append(" & ".join(["Total", ""] + [comma(tot.get(x, 0)) for x in
             ("calls", "tmpl", "flagged", "A", "B", "B_against", "C", "C_trunc", "C_luck")]) + " \\\\")
    open(os.path.join(OUT, "tables", "audit_extraction_table.tex"), "w").write("\n".join(L) + "\n")

    M = [hdr]
    add = lambda k, v: M.append(f"\\newcommand{{\\Au{k}}}{{{v}}}")  # noqa: E731
    add("Nitems", out["meta"]["n_items"])
    for c in CFGS:
        add(f"Acc{KEY[c]}", f"{R[c]['acc']:.1f}")
        add(f"Hand{KEY[c]}", f"{R[c]['acc_hand']:.1f}")
        add(f"Tok{KEY[c]}", comma(R[c]["total_tokens"]))
        add(f"Out{KEY[c]}", comma(R[c]["out_tokens"]))
    for c in ("SC_k8", "DEBATE_N3_R1"):
        add(f"TwoAcc{KEY[c]}", f"{R2[c]['acc']:.1f}")
        add(f"TwoTok{KEY[c]}", comma(R2[c]["total_tokens"]))
        add(f"TwoOut{KEY[c]}", comma(R2[c]["out_tokens"]))
    # tests: "+6.0, [+1.3,+10.7]" and p
    for name, t in Tt.items():
        k = "".join(KEY.get(x, x) for x in name.replace("r80", "Eighty")
                    .replace("2048-640", "Cap").replace("640", "")
                    .replace("2048", "Two").replace("hand", "Hand").split("|"))
        add(f"T{k}", f"${f1(t['diff'])}$, {iv(*t['ci95'])}")
        add(f"T{k}P", "p{<}0.001" if t["p"] < 0.001 else f"p{{=}}{pv(t['p'])}")  # use inside $...$
    # one-sided bound on the gain of each workflow over SC k=8 (coordination minus SC)
    upper = {}
    for c in COORD:
        for tag, nm in (("", f"640|SC_k8|{c}"), ("Hand", f"hand|SC_k8|{c}")):
            u = -Tt[nm]["ci90"][0]
            upper[tag + c] = u
            add(f"Up{tag}{KEY[c]}", f"${f1(u)}$")
    u2 = -Tt["2048|SC_k8|DEBATE_N3_R1"]["ci90"][0]
    upper["2048"] = u2
    add("UpTwoDeb", f"${f1(u2)}$")
    out["gain_upper_bounds"] = upper
    assert all(u < MARGIN for u in upper.values()), upper
    add("MaxUp", f"${f1(max(upper.values()))}$")
    # cap hits
    cp = out["cap_hits"]
    for cp_k, tag in (("640", "Lo"), ("2048", "Hi")):
        for part, pt in (("debate_final", "DebFinal"), ("sc8", "Sc")):
            e = cp[cp_k][part]
            add(f"Cap{tag}{pt}Hit", e["hit"])
            add(f"Cap{tag}{pt}Calls", comma(e["calls"]))
            add(f"Cap{tag}{pt}Recs", e["records_with_hit"])
    # audit totals
    tab = out["audit"]["by_source_config"]
    tot = Counter()
    for v in tab.values():
        tot.update(v)
    den = out["audit"]["answer_bearing_calls"]
    add("Calls", comma(sum(den.values())))
    add("CallsGpu", comma(sum(v for k, v in den.items() if k.startswith("gpu"))))
    add("CallsMac", comma(sum(v for k, v in den.items() if k.startswith("mac"))))
    for k in ("flagged", "A", "B", "B_against", "B_for", "B_either", "C", "C_trunc", "C_luck"):
        add("N" + k.replace("_", ""), tot.get(k, 0))
    luck_cfg = sorted({k.split("|")[1] for k, v in tab.items() if v.get("C_luck")})
    assert luck_cfg == ["DEBATE_N3_R1"], luck_cfg
    assert tot.get("B_for", 0) == 0
    fl = out["audit"]["first_match_differs_from_last"]
    assert list(fl) == ["gpu|DEBATE_N3_R1"], fl
    add("FirstMatch", fl["gpu|DEBATE_N3_R1"])
    add("FirstMatchCalls", comma(den["gpu|DEBATE_N3_R1"]))
    add("MacScRecs", MAC_SC_AUDITED)
    # HIER no-answer output
    h = out["hier"]
    for nm, tag in (("pc", "Pc"), ("mac", "Mac"), ("july_gsm8k", "Jul"), ("july_mmlu", "Mmlu")):
        add(f"Hier{tag}NoAns", h[nm]["no_answer"])
        add(f"Hier{tag}Recs", h[nm]["records"])
        add(f"Hier{tag}NoAnsPct", f"{100 * h[nm]['no_answer'] / h[nm]['records']:.1f}")
        add(f"Hier{tag}Acc", f"{h[nm]['acc']:.1f}")
        add(f"Hier{tag}AccAns", f"{h[nm]['acc_answered']:.1f}")
    add("HierPcSaSameAns", f"{h['pc']['sa_acc_same_answered']:.1f}")
    add("HierPcLossNoAns", f"${f1(h['pc']['loss_from_no_answer_pts'])}$")
    add("HierPcLossAns", f"${f1(h['pc']['loss_from_answered_pts'])}$")
    add("HierMacJulSaSameAns", f"{h['mac']['july_sa_acc_same_answered']:.1f}")
    for c in CFGS:
        add(f"Jul{KEY[c]}", f"{h['july_acc'][c]:.1f}")
        d = out["drift80"][c]
        add(f"DriftMat{KEY[c]}", f"{d['matrix80']:.1f}")
        add(f"DriftRe{KEY[c]}", f"{d['rerun80']:.1f}")
        add(f"DriftOth{KEY[c]}", f"{d['rerun20']:.1f}")
        if c != "SC_k8":
            add(f"DriftGapMat{KEY[c]}", f"${f1(d['sc8_minus_matrix80'])}$")
    same = [abs(out["drift80"][c]["rerun80"] - out["drift80"][c]["matrix80"])
            for c in ("SC_k8", "DEBATE_N3_R1", "GE_R2")]
    add("DriftMaxAbsSamePts", f"{max(same):.1f}")
    # the text says SA and HIER fell by more than that on the same items
    assert all(out["drift80"][c]["matrix80"] - out["drift80"][c]["rerun80"] > max(same)
               for c in ("SA", "HIER_M3"))
    rp = out["mac_repro"]
    for c in ("SC_k8",) + tuple(COORD):
        add(f"Repro{KEY[c]}", rp[c]["match"])
        add(f"Repro{KEY[c]}N", rp[c]["n"])
        add(f"Mac{KEY[c]}", f"{rp[c]['acc']:.1f}")
        add(f"MacRef{KEY[c]}", f"{rp[c]['ref_acc']:.1f}")
    for c in COORD:
        add(f"MacHand{KEY[c]}", f"{out['mac_hand'][c]['acc_hand']:.1f}")
    open(os.path.join(OUT, "audit_macros.tex"), "w").write("\n".join(M) + "\n")
    json.dump(out, open(os.path.join(OUT, "tables", "audit.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--render" in args:
        render(json.load(open(os.path.join(OUT, "tables", "audit.json"))))
    else:
        w = int(args[args.index("--workers") + 1]) if "--workers" in args else 6
        render(compute(w))
