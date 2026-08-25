#!/usr/bin/env python3
"""Generate paper/figures/*.pdf (vector) from the committed result tables
under paper/tables/*.csv — never hand-drawn numbers. Two schematic figures
(architecture, pipeline) are diagrams, not data plots, and are built from
this project's own module/phase structure, not measurements.

Categorical color order follows the dataviz skill's validated default
palette (light mode, fixed order, never cycled): blue, orange, aqua,
yellow — see references/palette.md in that skill.

Usage: PYTHONPATH=. python scripts/make_figures.py
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from display_names import disp

OUT = Path("paper/figures")
OUT.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d8d7d0"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "svg.fonttype": "none",
})


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------- diagrams

def fig_architecture():
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")

    def box(x, y, w, h, text, color, alpha=0.18):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                     linewidth=1.3, edgecolor=color, facecolor=color, alpha=alpha))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5, color=INK)

    def arrow(x1, y1, x2, y2, color=INK):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12,
                                      linewidth=1.2, color=color))

    box(0.3, 2.4, 2.0, 1.2, "input\n$x$", MUTED, 0.12)
    box(2.7, 2.4, 2.2, 1.2, "backbone\n$f_\\theta(x)$", BLUE)
    arrow(2.3, 3.0, 2.7, 3.0)

    box(5.4, 4.0, 2.4, 1.1, "coarse head $h_c$\n(public)", AQUA)
    box(5.4, 0.9, 2.4, 1.1, "adapter $A_\\phi$ + fine head $h_f$\n(restricted, encrypted at rest)", ORANGE)
    arrow(4.9, 3.2, 5.4, 4.5)
    arrow(4.9, 2.8, 5.4, 1.5)

    box(8.2, 4.0, 1.5, 1.1, "$\\hat{y}_c$", AQUA, 0.28)
    box(8.2, 0.9, 1.5, 1.1, "$\\hat{y}_f$", ORANGE, 0.28)
    arrow(7.8, 4.55, 8.2, 4.55)
    arrow(7.8, 1.45, 8.2, 1.45)

    ax.text(5.0, 5.5, "public path (unauthorized users)", fontsize=8, color=AQUA, ha="center")
    ax.text(5.0, 0.25, "authorized path (decrypted adapter required)", fontsize=8, color=ORANGE, ha="center")
    fig.tight_layout()
    fig.savefig(OUT / "fig_architecture.pdf")
    plt.close(fig)


def fig_pipeline():
    phases = [
        ("0", "repo &\ndata validation", "DONE"),
        ("1", "primary\nutility", "null+hier. tier"),
        ("2", "capability\nisolation", "null+hier. tier"),
        ("3", "security\nattacks", "mostly done"),
        ("4", "privacy\nmechanisms", "synth. tier"),
        ("5", "revocation &\nunlearning", "synth. tier"),
        ("6", "external\nvalidation", "pending"),
        ("7", "IXI\nextension", "pending"),
        ("8", "blockchain\n(optional)", "pending"),
    ]
    fig, ax = plt.subplots(figsize=(9.5, 1.9))
    ax.set_xlim(0, len(phases)); ax.set_ylim(0, 1); ax.axis("off")
    for i, (num, label, status) in enumerate(phases):
        done = "pending" not in status
        color = BLUE if done else MUTED
        alpha = 0.20 if done else 0.08
        ax.add_patch(FancyBboxPatch((i + 0.06, 0.15), 0.88, 0.7, boxstyle="round,pad=0.03",
                                     linewidth=1.1, edgecolor=color, facecolor=color, alpha=alpha))
        ax.text(i + 0.5, 0.62, f"P{num}", ha="center", va="center", fontsize=9, fontweight="bold", color=INK)
        ax.text(i + 0.5, 0.42, label, ha="center", va="center", fontsize=6.3, color=INK)
        ax.text(i + 0.5, 0.05, status, ha="center", va="top", fontsize=6, color=MUTED, style="italic")
        if i < len(phases) - 1:
            ax.annotate("", xy=(i + 1.02, 0.5), xytext=(i + 0.96, 0.5),
                        arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.0))
    fig.tight_layout()
    fig.savefig(OUT / "fig_pipeline.pdf")
    plt.close(fig)


# ------------------------------------------------------------- data plots

def fig_phase1_utility():
    rows = read_csv("paper/tables/phase1_utility_synthetic.csv")
    baselines = [r["baseline"] for r in rows]
    coarse = [float(r["coarse_macro_f1_mean"]) for r in rows]
    coarse_std = [float(r["coarse_macro_f1_std"]) for r in rows]
    fine = [float(r["fine_macro_f1_mean"]) for r in rows]
    fine_std = [float(r["fine_macro_f1_std"]) for r in rows]

    x = range(len(baselines)); w = 0.35
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.bar([i - w / 2 for i in x], coarse, width=w, yerr=coarse_std, color=BLUE, label="coarse macro-F1", capsize=2)
    ax.bar([i + w / 2 for i in x], fine, width=w, yerr=fine_std, color=ORANGE, label="fine macro-F1", capsize=2)
    ax.set_xticks(list(x)); ax.set_xticklabels([disp(b) for b in baselines], rotation=25, ha="right")
    ax.set_ylabel("macro-F1 (mean ± std, 5 seeds)")
    ax.set_title("Phase 1 primary utility — SYNTHETIC TIER (pipeline validation only)", fontsize=8)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_phase1_utility.pdf")
    plt.close(fig)


def fig_phase2_capability_isolation():
    rows = read_csv("paper/tables/phase2_capability_isolation_synthetic.csv")
    methods = [r["method"] for r in rows]
    u_public = [float(r["u_public_mean"]) for r in rows]
    rfc = [float(r["rfc_mean"]) for r in rows]
    rfc_std = [float(r["rfc_std"]) for r in rows]

    x = range(len(methods)); w = 0.35
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.bar([i - w / 2 for i in x], u_public, width=w, color=AQUA, label="OutputLeak (public-output-only probe)")
    ax.bar([i + w / 2 for i in x], rfc, width=w, yerr=rfc_std, color=ORANGE, label="BestProbeRFC (best representation probe)", capsize=2)
    ax.set_xticks(list(x)); ax.set_xticklabels([disp(m) for m in methods], rotation=25, ha="right", fontsize=7.5)
    ax.set_ylabel("fine-label macro-F1 (mean ± std, 5 seeds)")
    ax.set_title("Phase 2 capability decomposition — SYNTHETIC TIER (pipeline validation only)", fontsize=8)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_phase2_capability_isolation.pdf")
    plt.close(fig)


def fig_phase1_hierarchical():
    path = Path("paper/tables/phase1_hierarchical.csv")
    if not path.exists():
        return
    rows = read_csv(str(path))
    methods = [r["method"] for r in rows]
    fine = [float(r["fine_f1_mean"]) if r["fine_f1_mean"] not in ("", None) else 0.0 for r in rows]
    fine_std = [float(r["fine_f1_std"]) if r["fine_f1_std"] not in ("", None) else 0.0 for r in rows]
    rfc = [float(r["rfc_mean"]) if r["rfc_mean"] not in ("", None) else 0.0 for r in rows]

    x = range(len(methods)); w = 0.35
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    ax.bar([i - w / 2 for i in x], fine, width=w, yerr=fine_std, color=BLUE, label="AuthorizedFineUtility", capsize=2)
    ax.bar([i + w / 2 for i in x], rfc, width=w, color=ORANGE, label="BestProbeRFC")
    ax.set_xticks(list(x)); ax.set_xticklabels([disp(m) for m in methods], rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("fine macro-F1 (mean ± std, 5 seeds)")
    ax.set_title("Phase 1+2 fair baselines — HIERARCHICAL FIXTURE (real learnable signal, still synthetic)", fontsize=7.5)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_phase1_hierarchical.pdf")
    plt.close(fig)


def fig_phase4_privacy_pareto():
    EPS_COL = "epsilon_full_training_max_per_client_record_level"
    rows = read_csv("paper/tables/phase4_privacy_synthetic.csv")
    dp_rows = [r for r in rows if r["arm"] in ("dp_sgd", "secure_agg_plus_dp") and r[EPS_COL] != ""]
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    for arm, color, marker in [("dp_sgd", BLUE, "o"), ("secure_agg_plus_dp", ORANGE, "s")]:
        arm_rows = sorted([r for r in dp_rows if r["arm"] == arm], key=lambda r: float(r[EPS_COL]))
        eps = [float(r[EPS_COL]) for r in arm_rows]
        f1 = [float(r["fine_macro_f1_mean"]) for r in arm_rows]
        ax.plot(eps, f1, marker=marker, color=color, label=disp(arm), linewidth=1.5, markersize=5)
    no_prot = [r for r in rows if r["arm"] == "no_protection"][0]
    ax.axhline(float(no_prot["fine_macro_f1_mean"]), color=MUTED, linestyle="--", linewidth=1,
               label=f"{disp('no_protection')} (no DP noise)")
    ax.set_xlabel(r"full-training privacy budget $\varepsilon$ (record-level, $\delta=10^{-5}$, lower = more private)")
    ax.set_ylabel("fine macro-F1 (mean, 3 seeds)")
    ax.set_title("Phase 4 privacy-utility Pareto — SYNTHETIC TIER (pipeline validation only)", fontsize=8)
    ax.legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig_phase4_privacy_pareto.pdf")
    plt.close(fig)


def fig_phase5_unlearning():
    rows = read_csv("paper/tables/phase5_unlearning_synthetic.csv")
    methods = sorted({r["method"] for r in rows})
    scenarios = ["institution", "class"]
    x = range(len(methods)); w = 0.35
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    for i, (scenario, color) in enumerate(zip(scenarios, [BLUE, ORANGE])):
        vals, errs = [], []
        for m in methods:
            match = [r for r in rows if r["scenario"] == scenario and r["method"] == m][0]
            vals.append(float(match["retained_fine_macro_f1_mean"]))
            errs.append(float(match["retained_fine_macro_f1_std"]))
        offset = (i - 0.5) * w
        ax.bar([xi + offset for xi in x], vals, width=w, yerr=errs, color=color, label=f"{disp(scenario)} scenario", capsize=2)
    ax.set_xticks(list(x)); ax.set_xticklabels([disp(m) for m in methods], rotation=25, ha="right", fontsize=7.5)
    ax.set_ylabel("retained-data fine macro-F1 (mean ± std, 3 seeds)")
    ax.set_title("Phase 5 unlearning vs. retraining — SYNTHETIC TIER (pipeline validation only)", fontsize=8)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_phase5_unlearning.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig_architecture()
    fig_pipeline()
    fig_phase1_utility()
    fig_phase1_hierarchical()
    fig_phase2_capability_isolation()
    fig_phase4_privacy_pareto()
    fig_phase5_unlearning()
    print(f"wrote figures to {OUT}/")
