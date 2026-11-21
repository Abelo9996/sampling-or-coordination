"""Analyze results/mechanistic.jsonl into correction/corruption rates for the
paper's mechanism paragraph. Prints a compact summary; no plotting.

Debate (init majority vs final majority):
  correction   = init wrong  -> final correct
  corruption   = init correct -> final wrong
  minority-win = correct answer present but a minority at init, final correct
  diversity collapse = final unique answers < init unique answers

GE (per revision round, before vs after):
  critic TP = found_errors on a before-wrong solution
  critic FP = found_errors on a before-correct solution
  correction = before wrong -> after correct
  corruption = before correct -> after wrong
"""
import json, os

path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "results", "mechanistic.jsonl")
deb, ge = [], []
with open(path) as f:
    for line in f:
        r = json.loads(line)
        (deb if r["wf"] == "debate" else ge).append(r)


def pct(a, b):
    return f"{100.0*a/b:.1f}% ({a}/{b})" if b else "n/a (0)"


print(f"=== DEBATE (n={len(deb)}) ===")
if deb:
    init_wrong = [r for r in deb if not r["init_correct"]]
    init_right = [r for r in deb if r["init_correct"]]
    corr = sum(1 for r in init_wrong if r["final_correct"])
    corp = sum(1 for r in init_right if not r["final_correct"])
    minority = [r for r in deb if r["correct_minority_init"]]
    minority_win = sum(1 for r in minority if r["final_correct"])
    collapse = sum(1 for r in deb if r["final_unique"] < r["init_unique"])
    net = sum(1 for r in deb if r["final_correct"]) - sum(1 for r in deb if r["init_correct"])
    print(f"  init acc:  {pct(len(init_right), len(deb))}")
    print(f"  final acc: {pct(sum(1 for r in deb if r['final_correct']), len(deb))}")
    print(f"  correction (wrong->right): {pct(corr, len(init_wrong))}")
    print(f"  corruption (right->wrong): {pct(corp, len(init_right))}")
    print(f"  correct-minority items:    {len(minority)}; won: {pct(minority_win, len(minority))}")
    print(f"  diversity collapse:        {pct(collapse, len(deb))}")
    print(f"  net accuracy change:       {net:+d} items")

print(f"\n=== GE (n={len(ge)}) ===")
if ge:
    rounds = [rd for r in ge for rd in r["rounds"]]
    before_wrong = [rd for rd in rounds if not rd["before_correct"]]
    before_right = [rd for rd in rounds if rd["before_correct"]]
    tp = sum(1 for rd in before_wrong if rd["critic_found_errors"])
    fp = sum(1 for rd in before_right if rd["critic_found_errors"])
    corr = sum(1 for rd in before_wrong if rd["after_correct"])
    corp = sum(1 for rd in before_right if not rd["after_correct"])
    net = sum(1 for r in ge if r["final_correct"]) - sum(1 for r in ge if r["init_correct"])
    print(f"  init acc:  {pct(sum(1 for r in ge if r['init_correct']), len(ge))}")
    print(f"  final acc: {pct(sum(1 for r in ge if r['final_correct']), len(ge))}")
    print(f"  critic true-positive (flags a wrong soln):  {pct(tp, len(before_wrong))}")
    print(f"  critic false-positive (flags a right soln): {pct(fp, len(before_right))}")
    print(f"  revision correction (wrong->right): {pct(corr, len(before_wrong))}")
    print(f"  revision corruption (right->wrong): {pct(corp, len(before_right))}")
    print(f"  net accuracy change (init->final):  {net:+d} items")
