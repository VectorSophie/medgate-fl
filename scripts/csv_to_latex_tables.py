#!/usr/bin/env python3
"""Convert the committed result CSVs into LaTeX tabular snippets under
paper/tables/*.tex, \\input by paper/main.tex. This is the only path any
empirical number takes into the paper — nothing is hand-typed.

Usage: PYTHONPATH=. python scripts/csv_to_latex_tables.py
"""
import csv
from pathlib import Path

TABLES = Path("paper/tables")


def esc(s: str) -> str:
    return s.replace("_", "\\_").replace("±", "$\\pm$")


def fmt_num(s: str, nd: int = 3) -> str:
    if s in ("", "None", None):
        return "n/a"
    try:
        return f"{float(s):.{nd}f}"
    except ValueError:
        return esc(s)


def write_tabular(out_name: str, header: list, rows: list, colspec: str):
    lines = [f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    for r in rows:
        lines.append(" & ".join(r) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (TABLES / out_name).write_text("\n".join(lines) + "\n")


def phase1():
    rows = list(csv.DictReader(open(TABLES / "phase1_utility_synthetic.csv")))
    out = [[esc(r["baseline"]), r["n_seeds"],
            f"{fmt_num(r['coarse_macro_f1_mean'])} $\\pm$ {fmt_num(r['coarse_macro_f1_std'])}",
            f"{fmt_num(r['fine_macro_f1_mean'])} $\\pm$ {fmt_num(r['fine_macro_f1_std'])}",
            f"{fmt_num(r['worst_institution_fine_macro_f1_mean'])} $\\pm$ {fmt_num(r['worst_institution_fine_macro_f1_std'])}",
            f"{fmt_num(r['wall_clock_seconds_mean'], 1)}"] for r in rows]
    write_tabular("phase1_utility_synthetic.tex",
                  ["Baseline", "Seeds", "Coarse macro-F1", "Fine macro-F1", "Worst-inst. fine macro-F1", "Wall-clock (s)"],
                  out, "lccccc")


def phase2():
    rows = list(csv.DictReader(open(TABLES / "phase2_capability_isolation_synthetic.csv")))
    out = []
    for r in rows:
        arr = "n/a" if r["arr_mean"] == "" else f"{fmt_num(r['arr_mean'])} $\\pm$ {fmt_num(r['arr_std'])}"
        out.append([esc(r["method"]), r["n_seeds"],
                    f"{fmt_num(r['fine_macro_f1_mean'])} $\\pm$ {fmt_num(r['fine_macro_f1_std'])}",
                    f"{fmt_num(r['u_public_mean'])} $\\pm$ {fmt_num(r['u_public_std'])}",
                    f"{fmt_num(r['rfc_mean'])} $\\pm$ {fmt_num(r['rfc_std'])}",
                    f"{fmt_num(r['ucg_mean'])} $\\pm$ {fmt_num(r['ucg_std'])}",
                    arr])
    write_tabular("phase2_capability_isolation_synthetic.tex",
                  ["Method", "Seeds", "Auth. fine F1", "$U_{\\text{public}}$", "RFC", "UCG", "ARR"],
                  out, "lcccccc")


def phase3():
    rows = list(csv.DictReader(open(TABLES / "phase3_attacks_synthetic.csv")))
    out = [[esc(r["attack"]), r["n_seeds"], esc(r["metric"]), fmt_num(r["mean"]), fmt_num(r["std"]), esc(r["note"])] for r in rows]
    write_tabular("phase3_attacks_synthetic.tex", ["Attack", "Seeds", "Metric", "Mean", "Std", "Note"], out, "lcllll")


def phase3_integrity():
    rows = list(csv.DictReader(open(TABLES / "phase3_integrity_synthetic.csv")))
    out = []
    for r in rows:
        bd = "n/a" if r["backdoor_success_rate_mean"] == "" else fmt_num(r["backdoor_success_rate_mean"])
        out.append([esc(r["attack"]), esc(r["aggregator"]), r["n_seeds"],
                    f"{fmt_num(r['fine_macro_f1_mean'])} $\\pm$ {fmt_num(r['fine_macro_f1_std'])}",
                    fmt_num(r["model_corrupted_rate"], 2), bd])
    write_tabular("phase3_integrity_synthetic.tex",
                  ["Attack", "Aggregator", "Seeds", "Fine macro-F1", "Model corrupted rate", "Backdoor success rate"],
                  out, "llccccc")


def phase4():
    rows = list(csv.DictReader(open(TABLES / "phase4_privacy_synthetic.csv")))
    out = []
    for r in rows:
        eps = "n/a" if r["epsilon"] == "" else fmt_num(r["epsilon"], 2)
        nm = "n/a" if r["noise_multiplier"] == "" else r["noise_multiplier"]
        out.append([esc(r["arm"]), nm, eps,
                    f"{fmt_num(r['fine_macro_f1_mean'])} $\\pm$ {fmt_num(r['fine_macro_f1_std'])}", r["n_seeds"]])
    write_tabular("phase4_privacy_synthetic.tex",
                  ["Arm", "Noise mult.", "$\\varepsilon$ ($\\delta=10^{-5}$)", "Fine macro-F1", "Seeds"], out, "lcccc")


def phase5():
    rows = list(csv.DictReader(open(TABLES / "phase5_unlearning_synthetic.csv")))
    out = [[esc(r["scenario"]), esc(r["method"]), r["n_seeds"],
            f"{fmt_num(r['retained_fine_macro_f1_mean'])} $\\pm$ {fmt_num(r['retained_fine_macro_f1_std'])}",
            f"{fmt_num(r['gap_to_gold_standard_mean'], 3)} $\\pm$ {fmt_num(r['gap_to_gold_standard_std'])}",
            f"{fmt_num(r['forgetting_auc_mean'])} $\\pm$ {fmt_num(r['forgetting_auc_std'])}"] for r in rows]
    write_tabular("phase5_unlearning_synthetic.tex",
                  ["Scenario", "Method", "Seeds", "Retained fine F1", "Gap to gold", "Forgetting AUC"], out, "llcccc")


if __name__ == "__main__":
    phase1(); phase2(); phase3(); phase3_integrity(); phase4(); phase5()
    print("wrote LaTeX tables to", TABLES)
