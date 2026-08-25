#!/usr/bin/env python3
"""Convert the committed result CSVs into LaTeX tabular snippets under
paper/tables/*.tex, \\input by paper/main.tex. This is the only path any
empirical number takes into the paper — nothing is hand-typed.

P2-12/13 repair (docs/execution_plan.md): code-style identifiers
(snake_case attack/method/arm names) are mapped to human-readable
publication labels via DISPLAY_NAMES below before they reach a table —
the underlying CSV/JSON keep the code identifier for traceability, only
the rendered table changes. Wide tables (many columns, or long text
cells) are wrapped in \\resizebox by the .tex snippet itself where needed
so nothing overflows the page margin — verified by rendering and visually
inspecting the compiled PDF, not just by generating the LaTeX.

Usage: PYTHONPATH=. python scripts/csv_to_latex_tables.py
"""
import csv
from pathlib import Path

from display_names import disp

TABLES = Path("paper/tables")


def esc(s: str) -> str:
    return str(s).replace("_", "\\_").replace("±", "$\\pm$")


def fmt_num(s, nd: int = 3) -> str:
    if s in ("", "None", None):
        return "n/a"
    try:
        return f"{float(s):.{nd}f}"
    except (ValueError, TypeError):
        return esc(s)


def write_tabular(out_name: str, header: list, rows: list, colspec: str, wide: bool = False):
    lines = [f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    for r in rows:
        lines.append(" & ".join(r) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    body = "\n".join(lines) + "\n"
    if wide:
        body = f"\\resizebox{{\\textwidth}}{{!}}{{%\n{body}}}\n"
    (TABLES / out_name).write_text(body)


def phase1():
    rows = list(csv.DictReader(open(TABLES / "phase1_utility_synthetic.csv")))
    out = [[disp(r["baseline"]), r["n_seeds"],
            f"{fmt_num(r['coarse_macro_f1_mean'])} $\\pm$ {fmt_num(r['coarse_macro_f1_std'])}",
            f"{fmt_num(r['fine_macro_f1_mean'])} $\\pm$ {fmt_num(r['fine_macro_f1_std'])}",
            f"{fmt_num(r['worst_institution_fine_macro_f1_mean'])} $\\pm$ {fmt_num(r['worst_institution_fine_macro_f1_std'])}",
            f"{fmt_num(r['wall_clock_seconds_mean'], 1)}"] for r in rows]
    write_tabular("phase1_utility_synthetic.tex",
                  ["Baseline", "Seeds", "Coarse macro-F1", "Fine macro-F1", "Worst-inst. fine macro-F1", "Wall-clock (s)"],
                  out, "lccccc")


def phase1_hierarchical():
    path = TABLES / "phase1_hierarchical.csv"
    if not path.exists():
        return
    rows = list(csv.DictReader(open(path)))
    out = []
    for r in rows:
        out.append([disp(r["method"]), disp(r["kind"]), r["n_seeds"],
                    f"{fmt_num(r['coarse_f1_mean'])} $\\pm$ {fmt_num(r['coarse_f1_std'])}",
                    f"{fmt_num(r['fine_f1_mean'])} $\\pm$ {fmt_num(r['fine_f1_std'])}",
                    f"{fmt_num(r['u_public_mean'])} $\\pm$ {fmt_num(r['u_public_std'])}",
                    f"{fmt_num(r['rfc_mean'])} $\\pm$ {fmt_num(r['rfc_std'])}",
                    f"{fmt_num(r['arr_mean'])} $\\pm$ {fmt_num(r['arr_std'])}"])
    write_tabular("phase1_hierarchical.tex",
                  ["Method", "Kind", "Seeds", "Coarse F1", "Fine F1", "OutputLeak", "BestProbeRFC", "ARR"],
                  out, "llcccccc", wide=True)


def phase2():
    rows = list(csv.DictReader(open(TABLES / "phase2_capability_isolation_synthetic.csv")))
    out = []
    for r in rows:
        arr = "n/a" if r["arr_mean"] == "" else f"{fmt_num(r['arr_mean'])} $\\pm$ {fmt_num(r['arr_std'])}"
        out.append([disp(r["method"]), r["n_seeds"],
                    f"{fmt_num(r['fine_macro_f1_mean'])} $\\pm$ {fmt_num(r['fine_macro_f1_std'])}",
                    f"{fmt_num(r['u_public_mean'])} $\\pm$ {fmt_num(r['u_public_std'])}",
                    f"{fmt_num(r['rfc_mean'])} $\\pm$ {fmt_num(r['rfc_std'])}",
                    f"{fmt_num(r['ucg_mean'])} $\\pm$ {fmt_num(r['ucg_std'])}",
                    arr])
    write_tabular("phase2_capability_isolation_synthetic.tex",
                  ["Method", "Seeds", "AuthorizedFineUtility", "OutputLeak", "BestProbeRFC", "UCG", "ARR"],
                  out, "lcccccc", wide=True)


def phase3():
    rows = list(csv.DictReader(open(TABLES / "phase3_attacks_synthetic.csv")))
    out = [[disp(r["attack"]), r["n_seeds"], esc(r["metric"]), fmt_num(r["mean"]), fmt_num(r["std"]), esc(r["note"])] for r in rows]
    write_tabular("phase3_attacks_synthetic.tex", ["Attack", "Seeds", "Metric", "Mean", "Std", "Note"], out, "lcllll", wide=True)


def phase3_adapter_recovery():
    path = TABLES / "phase3_adapter_recovery_synthetic.csv"
    if not path.exists():
        return
    rows = list(csv.DictReader(open(path)))
    out = [[disp(r["scenario"]), r["reveal_fraction"], r["n_seeds"],
            f"{fmt_num(r['cosine_similarity_mean'])} $\\pm$ {fmt_num(r['cosine_similarity_std'])}",
            f"{fmt_num(r['normalized_frobenius_error_mean'])} $\\pm$ {fmt_num(r['normalized_frobenius_error_std'])}",
            f"{fmt_num(r['functional_fine_macro_f1_mean'])} $\\pm$ {fmt_num(r['functional_fine_macro_f1_std'])}"]
           for r in rows]
    write_tabular("phase3_adapter_recovery_synthetic.tex",
                  ["Scenario", "Reveal fraction", "Seeds", "Cosine sim.", "Frob. error", "Functional fine-F1"],
                  out, "lccccc", wide=True)


def phase3_integrity():
    rows = list(csv.DictReader(open(TABLES / "phase3_integrity_synthetic.csv")))
    out = []
    for r in rows:
        bd = "n/a" if r["backdoor_success_rate_mean"] == "" else fmt_num(r["backdoor_success_rate_mean"])
        out.append([disp(r["attack"]), disp(r["aggregator"]), r["n_seeds"],
                    f"{fmt_num(r['fine_macro_f1_mean'])} $\\pm$ {fmt_num(r['fine_macro_f1_std'])}",
                    fmt_num(r["model_corrupted_rate"], 2), bd])
    write_tabular("phase3_integrity_synthetic.tex",
                  ["Attack", "Aggregator", "Seeds", "Fine macro-F1", "Model corrupted rate", "Backdoor success rate"],
                  out, "llcccc", wide=True)


def phase4():
    rows = list(csv.DictReader(open(TABLES / "phase4_privacy_synthetic.csv")))
    out = []
    for r in rows:
        eps = "n/a" if r["epsilon"] == "" else fmt_num(r["epsilon"], 2)
        nm = "n/a" if r["noise_multiplier"] == "" else r["noise_multiplier"]
        out.append([disp(r["arm"]), nm, eps,
                    f"{fmt_num(r['fine_macro_f1_mean'])} $\\pm$ {fmt_num(r['fine_macro_f1_std'])}", r["n_seeds"]])
    write_tabular("phase4_privacy_synthetic.tex",
                  ["Arm", "Noise mult.", "$\\varepsilon$ ($\\delta=10^{-5}$)", "Fine macro-F1", "Seeds"], out, "lcccc")


def phase5():
    rows = list(csv.DictReader(open(TABLES / "phase5_unlearning_synthetic.csv")))
    out = [[disp(r["scenario"]), disp(r["method"]), r["n_seeds"],
            f"{fmt_num(r['retained_fine_macro_f1_mean'])} $\\pm$ {fmt_num(r['retained_fine_macro_f1_std'])}",
            f"{fmt_num(r['gap_to_gold_standard_mean'], 3)} $\\pm$ {fmt_num(r['gap_to_gold_standard_std'])}",
            f"{fmt_num(r['forgetting_symmetric_auc_mean'])} $\\pm$ {fmt_num(r['forgetting_symmetric_auc_std'])}",
            f"{fmt_num(r['forgetting_attack_advantage_mean'])} $\\pm$ {fmt_num(r['forgetting_attack_advantage_std'])}"]
           for r in rows]
    write_tabular("phase5_unlearning_synthetic.tex",
                  ["Scenario", "Method", "Seeds", "Retained fine F1", "Gap to gold", "SymmetricAUC", "Attack advantage"],
                  out, "llccccc", wide=True)


def phase5_confounded_appendix():
    path = TABLES / "phase5_unlearning_confounded_appendix.csv"
    if not path.exists():
        return
    rows = list(csv.DictReader(open(path)))
    out = [[disp(r["method"]), r["n_seeds"],
            f"{fmt_num(r['confounded_symmetric_auc_mean'])} $\\pm$ {fmt_num(r['confounded_symmetric_auc_std'])}"] for r in rows]
    write_tabular("phase5_unlearning_confounded_appendix.tex",
                  ["Method", "Seeds", "Confounded SymmetricAUC (DO NOT USE AS EVIDENCE)"], out, "lcc")


if __name__ == "__main__":
    phase1(); phase1_hierarchical(); phase2(); phase3(); phase3_adapter_recovery(); phase3_integrity()
    phase4(); phase5(); phase5_confounded_appendix()
    print("wrote LaTeX tables to", TABLES)
