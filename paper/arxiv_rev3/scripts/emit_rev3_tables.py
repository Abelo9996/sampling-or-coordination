#!/usr/bin/env python3
"""rev3 table fragments, written to tables/rev3/, from the generated rev2 fragments in tables/
(byte-identical copies of paper/arxiv_rev2/tables, themselves copies of the arxiv_long emit outputs)
and from generated JSON. The original emit scripts are not touched.

1. Formatting only (every fragment): thousands separators on integers of four or more digits
   (3224 -> 3{,}224, -1648 -> -1{,}648), one workflow naming style ("debate" for Debate/DEBATE,
   "SC $k{=}8$" for SC$_8$). The script asserts that the multiset of numeric values in each
   fragment is unchanged by the rewrite (commas removed), so formatting cannot alter a value.
2. Statistics reframing (decided for rev3: one primary family, the MATH levels 4 to 5 protocol;
   everything else exploratory with unadjusted values):
   * equivalence table: the two Holm columns (over 16 rows) are replaced by the unadjusted
     one-sided p for a gain of 5 points (equivalence.json comparators.main.p_gain5); the other
     cells are re-rendered from equivalence.json with the formatting functions of
     paper/arxiv_long/scripts/emit_equivalence.py and asserted equal to the rev2 cells.
   * hard-MATH table: the Du et al. debate row's p (Holm over the extra-workflow protocol's
     family) is replaced by its unadjusted p (results/extra/hard_math_analysis.json
     du.minus_SC_k8.p), after asserting the old cell equals the Holm value in that JSON.
3. New Table 1 (main_table.tex): GSM8K and MMLU cells of the July matrix (poster_main_table.tex)
   with the MATH levels 1 to 3 column from the 238-item, three-seed run (math_easy_macros.tex,
   scripts/emit_math_easy.py).
4. MATH L1-3 (238 items) verdict rows for Table 2 (math_easy_domination.tex) from
   tables/math_easy.json, in the format of poster_domination_table.tex.
"""
import json, os, re, sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
REV3 = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(REV3))
SRC = os.path.join(REV3, "tables")
DST = os.path.join(REV3, "tables", "rev3")
os.makedirs(DST, exist_ok=True)

NUM = re.compile(r"(?<![\d.{,~])(?<!RTX~)(?<!RTX )(-?)(\d{4,})(?![\d.])")


def numbers(s):
    s = s.replace("{,}", "").replace(",", "")
    return Counter(re.findall(r"\d+(?:\.\d+)?", s))


def thousands(s):
    def rep(m):
        n = m.group(2)
        if n in ("4060",):  # RTX 4060 is a product name
            return m.group(0)
        return m.group(1) + f"{int(n):,}".replace(",", "{,}")
    return NUM.sub(rep, s)


def names(s):
    s = re.sub(r"SC\$_\{?(\d+)\}?\$", r"SC $k{=}\1$", s)
    s = re.sub(r"\bDEBATE\b", "debate", s)
    s = re.sub(r"(^|&|\\quad|\{)(\s*)Debate\b", lambda m: m.group(1) + m.group(2) + "debate", s, flags=re.M)
    return s


def fmt(text):
    out = thousands(names(text))
    a, b = numbers(text), numbers(out)
    assert a == b, (a - b, b - a)
    return out


# == 1. formatting of every fragment ==
for f in sorted(os.listdir(SRC)):
    if f.endswith(".tex"):
        t = open(os.path.join(SRC, f)).read()
        if f == "positioning_table.tex":
            out = names(t)  # arXiv ids and years: no separators
        else:
            out = fmt(t)
        if f in ("poster_domination_table.tex", "cost_verdict_table.tex", "output_verdict_table.tex"):
            out = re.sub(r"^MATH &", "MATH$_{60}$ &", out, flags=re.M)
        open(os.path.join(DST, f), "w").write(
            "% rev3: formatted copy of tables/" + f + " by scripts/emit_rev3_tables.py (separators, names).\n" + out)

# == 2a. equivalence table: unadjusted p for a 5-point gain in place of the Holm columns ==
E = json.load(open(os.path.join(SRC, "equivalence.json")))
TN = {"gsm8k": "GSM8K", "math": "MATH$_{60}$", "mmlu": "MMLU", "gsm8k200": "GSM8K-200", "proeng": "MMLU-Pro eng.",
      "code": "HumanEval"}
CN = {"DEBATE_N3_R1": "debate", "GE_R2": "GE", "HIER_M3": "HIER"}
LBL = {"SA": "SA", "SC_k2": "SC $k{=}2$", "SC_k4": "SC $k{=}4$", "SC_k8": "SC $k{=}8$", "SC_k12": "SC $k{=}12$"}
VW = {"upper": "upper", "neither": "neither", "equiv": "equivalent"}


def f1(x):
    x = round(x, 1) + 0.0
    return f"${'+' if x > 0 else ''}{x:.1f}$"


def iv(lo, hi):
    return f"$[{f1(lo).strip('$')},{f1(hi).strip('$')}]$"


def pv(p):
    return "${<}0.001$" if p < 0.001 else f"${p:.3f}$" if p < 0.1 else f"${p:.2f}$"


old_rows = [l for l in open(os.path.join(SRC, "equivalence_table.tex")).read().splitlines()
            if l.strip() and not l.startswith("%") and l.strip() != "\\midrule"]
L, prev, j = ["% rev3: rendered by scripts/emit_rev3_tables.py from tables/equivalence.json."], None, 0
for r in E["rows"]:
    t, c = r["task"], r["coord"]
    if prev is not None and t != prev:
        L.append("\\midrule")
    m, b = r["comparators"]["main"], r["comparators"]["best_cheaper"]
    assert m["config"] == "SC_k8"
    old = [x.strip() for x in old_rows[j].rstrip("\\ ").split("&")]
    assert old[2:7] == [f1(m["est"]), iv(*m["ci90"]), pv(m["p_gain0"]), pv(m["p_gain0_holm"]), pv(m["p_gain5_holm"])], (old, r)
    cells = [TN[t] if t != prev else "", CN[c], f1(m["est"]), iv(*m["ci90"]), pv(m["p_gain0"]), pv(m["p_gain5"])]
    if b["config"] == m["config"]:
        cells += ["same", "", "", ""]
    else:
        dag = "$^\\ddagger$" if b["one_seed"] else ""
        cells += [LBL[b["config"]] + dag, f1(b["est"]), iv(*b["ci90"]), VW[b["verdict"]]]
        assert old[8:11] == [f1(b["est"]), iv(*b["ci90"]), VW[b["verdict"]]]
    L.append(" & ".join(cells) + " \\\\")
    prev, j = t, j + 1
open(os.path.join(DST, "equivalence_table.tex"), "w").write("\n".join(L) + "\n")

# == 2b. hard-MATH table: Du et al. p unadjusted ==
HM = json.load(open(os.path.join(ROOT, "results", "extra", "hard_math_analysis.json")))
t = open(os.path.join(DST, "hard_math_table.tex")).read()
holm = HM["extra_primary_family_holm"]["MATH45 DU - SC_k8"]
raw = HM["du"]["minus_SC_k8"]["p"]
old_cell = f"${holm:.3f}$"
assert old_cell in t.split("Du et al.")[1], (old_cell,)
t = t.replace("Du et al." + t.split("Du et al.")[1],
              "Du et al." + t.split("Du et al.")[1].replace(old_cell, f"${raw:.3f}$"))
t = t.replace("Single call (greedy, 1 seed)", "SA (greedy, 1 seed)")
t = re.sub(r"^debate \(\$N\{=\}3\$, \$R\{=\}1\$\)", "debate", t, flags=re.M)
open(os.path.join(DST, "hard_math_table.tex"), "w").write(t)

# == 3. Table 1 ==
rows = [l for l in open(os.path.join(SRC, "poster_main_table.tex")).read().splitlines()
        if l.strip() and not l.startswith("%")]
ME = {"SA": "Sa", "SC $k{=}2$": "Sctwo", "SC $k{=}4$": "Scfour", "SC $k{=}8$": "Sceight", "SC $k{=}12$": "Sctwelve",
      "debate": "Deb", "GE": "Ge", "HIER": "Hier"}
T1 = ["% rev3 Table 1: GSM8K and MMLU from tables/poster_main_table.tex (July matrix);",
      "% MATH levels 1 to 3 from math_easy_macros.tex (238 items, 3 seeds). scripts/emit_rev3_tables.py."]
for l in rows:
    if l.strip() == "\\midrule":
        T1.append(l); continue
    cells = [x.strip() for x in fmt(l).rstrip("\\ ").split("&")]
    cfg = cells[0]
    n = ME[cfg]
    T1.append(" & ".join(cells[:5] + [f"\\Me{n}Acc{{}} {{\\tiny\\Me{n}CI}}", f"\\Me{n}Tok"]) + " \\\\")
open(os.path.join(DST, "main_table.tex"), "w").write("\n".join(T1) + "\n")

# == 4. MATH L1-3 (238) verdict rows ==
J = json.load(open(os.path.join(SRC, "math_easy.json")))


def d1(x):
    x = round(x, 1) + 0.0
    return f"{'+' if x > 0 else ''}{x:.1f}"


def tok(x):
    v = int(round(x))
    return ("-" if v < 0 else "") + f"{abs(v):,}".replace(",", "{,}")


D = ["% rev3: MATH L1-3, 238 items, 3 seeds; from tables/math_easy.json (scripts/emit_math_easy.py)."]
first = True
for w in ["DEBATE_N3_R1", "GE_R2", "HIER_M3"]:
    v = J["verdicts"][w]
    show = []
    if v["cheapest_robust"]:
        show.append(v["cheapest_robust"])
    if "SC_k8" in v["cheaper"] and "SC_k8" not in show:
        show.append("SC_k8")
    if not show:
        show = [v["cheaper"][-1]]
    for i, a in enumerate(show):
        t_ = J["tests"][f"{a}|{w}"]
        dt = J["configs"][a]["total_tokens"] - J["configs"][w]["total_tokens"]
        assert abs(dt - t_["dtok"]) < 1e-6
        dlo, dhi = t_["dtok_ci95"]
        dag = "$^\\dagger$" if dlo <= 0 <= dhi else ""
        verdict = ("R" if t_["ci95"][0] > 0 else ("N" if t_["ci95"][1] < 0 else "C")) + dag
        cells = ["MATH$_{238}$" if first else "", CN[w] if i == 0 else "", LBL[a],
                 f"${d1(t_['diff'])}$ {{\\tiny$[{d1(t_['ci95'][0])},{d1(t_['ci95'][1])}]$}}",
                 f"${tok(dt)}$ {{\\tiny$[{tok(dlo)},{tok(dhi)}]$}}", verdict]
        D.append(" & ".join(cells) + " \\\\")
        first = False
open(os.path.join(DST, "math_easy_domination.tex"), "w").write("\n".join(D) + "\n")

# == 4b. audit fixes (AUDIT_INDEPENDENT.md) ==
# L4: Table 3 marks: one-seed SC k=12 (section mark), HumanEval GE fails the unparsed-output bound
# (pilcrow mark), HumanEval output-token verdicts filled from the output-token table (all R SA).
p3 = os.path.join(DST, "cost_verdict_table.tex")
t = open(p3).read()
ov = open(os.path.join(DST, "output_verdict_table.tex")).read()
he = [l for l in ov.splitlines() if l.startswith("HumanEval") or (l.startswith(" &") and "SA~271" in l)]
assert len(he) == 3 and all(l.rstrip(" \\").endswith("R {\\scriptsize SA}") for l in he), he
old = "MATH$_{60}$ & debate & R {\\scriptsize SC $k{=}12$} &"
assert t.count(old) == 1
t = t.replace(old, "MATH$_{60}$ & debate & R {\\scriptsize SC $k{=}12$$^{\\S}$} &")
for wf, mark in (("debate", ""), ("GE", "$^{\\P}$"), ("HIER", "")):
    pre = "HumanEval & debate" if wf == "debate" else " & " + wf
    old = pre + " & R {\\scriptsize SA} & -- & -- & -- & R {\\scriptsize SA} \\\\"
    assert t.count(old) == 1, old
    t = t.replace(old, pre + f" & R{mark} {{\\scriptsize SA}} & R {{\\scriptsize SA}} & -- & -- & R{mark} {{\\scriptsize SA}} \\\\")
open(p3, "w").write(t)

# H3: positioning table, "Released" cell of this study.
pp = os.path.join(DST, "positioning_table.tex")
t = open(pp).read()
old = "Not yet public; code and every per-record log will be released \\\\"
assert t.count(old) == 1
t = t.replace(old, "Code, prompts and per-record logs are at \\CodeURL \\\\")
open(pp, "w").write(t)

# L6: one direction per comparison. Table 24's rerun debate row (debate minus SC k=8) is the
# negation of the rerun test of Table 21 (tables/audit.json, 640|SC_k8|DEBATE_N3_R1).
A = json.load(open(os.path.join(SRC, "audit.json")))
pe = os.path.join(DST, "extra_workflows_table.tex")
t = open(pe).read()
tt = A["tests"]["640|SC_k8|DEBATE_N3_R1"]
old_cell = "$-6.0$ $[-10.7,-1.7]$ & $0.006$"
assert t.count(old_cell) == 1
new_cell = f"${f1(-tt['diff']).strip('$')}$ {iv(-tt['ci95'][1], -tt['ci95'][0])} & {pv(tt['p'])}"
t = t.replace(old_cell, new_cell)
for cfg, key in (("Single call", "640|SC_k8|SA"), ("HIER, original", "640|SC_k8|HIER_M3")):
    tt = A["tests"][key]
    row = [l for l in t.splitlines() if l.startswith(cfg)][0]
    assert iv(-tt["ci95"][1], -tt["ci95"][0]) in row, (cfg, row)
open(pe, "w").write(t)

# M2: MATH 238-item output-token verdict rows for the output-token table (same rule as its
# other rows: cheaper = sampling configurations with fewer mean output tokens; nearest = the most
# expensive of them; V = R if any cheaper one has a 95% interval above zero).
def tokc(x):
    return f"{int(round(x)):,}".replace(",", "{,}")
O = ["\\midrule"]
first = True
SAMP = ["SA", "SC_k2", "SC_k4", "SC_k8", "SC_k12"]
for w in ["DEBATE_N3_R1", "GE_R2", "HIER_M3"]:
    wo = J["configs"][w]["out_tokens"]
    ch = [a for a in SAMP if J["configs"][a]["out_tokens"] < wo]
    near = ch[-1]
    rob = [a for a in ch if J["tests"][f"{a}|{w}"]["ci95"][0] > 0]
    v = "R {\\scriptsize " + LBL[rob[0]] + "}" if rob else (
        "N" if all(J["tests"][f"{a}|{w}"]["ci95"][1] < 0 for a in ch) else "C")
    assert v == "C", (w, v)  # independent audit: all three C on the 238-item run
    tn = J["tests"][f"{near}|{w}"]
    cheaper = ", ".join(f"{LBL[a]}~{tokc(J['configs'][a]['out_tokens'])}" for a in ch)
    O.append(" & ".join(["MATH$_{238}$" if first else "", CN[w], tokc(wo), cheaper, LBL[near],
                         f"${d1(tn['diff'])}$ {{\\scriptsize $[{d1(tn['ci95'][0])},{d1(tn['ci95'][1])}]$}}", v]) + " \\\\")
    first = False
po = os.path.join(DST, "output_verdict_table.tex")
open(po, "a").write("\n".join(O) + "\n")

# == 5. HIER mark (HIER as run, unrepaired; explained once in Section 3) ==
MARKED = ["main_table.tex", "poster_main_table.tex", "poster_domination_table.tex", "math_easy_domination.tex",
          "cost_verdict_table.tex", "poster_robustness_table.tex", "output_verdict_table.tex",
          "extra_domination_compact.tex", "extra_domination_table.tex", "equivalence_table.tex",
          "audit_drift_table.tex"]
# rev3 rerun fold-in: hard_math_table.tex no longer carries the mark (on MATH the HIERFIX prompt
# equals the HIER prompt, protocols/REV3_RERUN_PROTOCOL.md Section 2, so its HIER row is HIERFIX).
for f in MARKED:
    p = os.path.join(DST, f)
    t = open(p).read()
    t2 = re.sub(r"(^|&\s*|\\quad\s*)HIER\b(?!FIX|\\hiermark)", r"\1HIER\\hiermark{}", t, flags=re.M)
    assert t2.count("\\hiermark") >= 1, f
    open(p, "w").write(t2)
print("wrote", sorted(os.listdir(DST)))
