"""Generate paper/results_table.tex (booktabs) from a results JSONL.
Every number comes straight from the logs -> no manual transcription.
  .venv/bin/python scripts/emit_table.py results/gsm8k_matrix.jsonl
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from harness import analyze  # noqa: E402


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        ROOT, "results", "gsm8k_matrix.jsonl")
    rows = analyze.load(path)
    agg = analyze.aggregate(rows)
    order = sorted(agg, key=lambda c: agg[c]["tokens"])
    lines = [
        r"\begin{table}[t]\centering\small",
        r"\caption{Accuracy and inference cost per topology on GSM8K "
        r"(Qwen2.5-3B-Instruct). Accuracy is mean over items with 95\% "
        r"bootstrap CIs; cost is mean total tokens per query. \$ is a "
        r"price-model estimate.}",
        r"\label{tab:results}",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Config & Topology & Accuracy (\%) & Tokens/query & \$/1k queries \\",
        r"\midrule",
    ]
    for c in order:
        a = agg[c]
        lines.append(
            f"{c.replace('_',chr(92)+'_')} & {a['topology']} & "
            f"{a['accuracy']*100:.1f} [{a['acc_lo']*100:.1f}, {a['acc_hi']*100:.1f}] & "
            f"{a['tokens']:.0f} & {a['usd_per_item']*1000:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out = os.path.join(ROOT, "paper", "results_table.tex")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
