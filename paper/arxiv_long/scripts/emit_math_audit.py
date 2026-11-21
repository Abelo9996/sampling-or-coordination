"""Macros for the blind MATH extraction audit (results/audit/math_extraction_summary.json)."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(HERE, "../../../results/audit/math_extraction_summary.json")))
pw = d["per_workflow"]; coord = ["DEBATE_N3_R1", "GE_R2", "HIER_M3"]
assert all(pw[k]["flip_wrong_to_right"] == pw[k]["flip_right_to_wrong"] == 0 for k in pw)
assert all(pw[k]["logged_correct"] == pw[k]["rescored_correct"] for k in pw)
assert all(d["scorer_problems"][k]["records_logged_pred_equal_marked_wrong"] == 2 for k in pw)
m = {
 "MaCalls": d["n_cases"], "MaAuto": d["auto_resolved"], "MaManual": d["manual"],
 "MaSpot": d["spot_check"]["n"], "MaSpotAgree": d["spot_check"]["agreed_with_auto_reading"],
 "MaRecords": sum(pw[k]["records"] for k in pw), "MaItems": pw["SC_k8"]["records"],
 "MaMisSc": d["misread_rate_SC_samples"]["misread"], "MaCallsSc": d["misread_rate_SC_samples"]["calls"],
 "MaMisCoord": d["misread_rate_coordination_final_calls"]["misread"],
 "MaCallsCoord": d["misread_rate_coordination_final_calls"]["calls"],
 "MaFlips": len(d["flips"]),
 "MaAccSc": f'{100*pw["SC_k8"]["logged_acc"]:.1f}', "MaAccDeb": f'{100*pw["DEBATE_N3_R1"]["logged_acc"]:.1f}',
 "MaAccGe": f'{100*pw["GE_R2"]["logged_acc"]:.1f}', "MaAccHier": f'{100*pw["HIER_M3"]["logged_acc"]:.1f}',
 "MaScorerRecords": 2,
}
with open(os.path.join(HERE, "../math_audit_macros.tex"), "w") as f:
    for k, v in m.items():
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(m)
