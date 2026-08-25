#!/usr/bin/env python3
"""Phase 3 P0-A genuine parameter-space adapter-recovery attack (repair
pass 4 rewrite -- see medgate/attacks/adapter_recovery.py's module
docstring and configs/phase3_adapter_recovery_synthetic.yaml for why this
whole experiment was rebuilt, not just relabeled).

For each of >=5 seeds, trains ONE "combined"-method authorized model on
each of two fixtures (null_signal, hierarchical -- same protocol as
scripts/run_phase1_hierarchical.py for the latter) and against that
model's true delta_w = up @ down:
  - sweeps fill method x reveal fraction (parameter-space geometry, both
    fixtures -- fixture-independent claim),
  - sweeps rank misspecification at one fixed reveal fraction (hard_impute
    only, both fixtures),
  - runs the two-attacker collusion variant (hard_impute, both fixtures),
  - evaluates FUNCTIONAL recovery (fine macro-F1) on each fixture's own
    fine task -- informative only on the hierarchical fixture, which has
    real signal; kept on null_signal too for continuity/comparison.

Usage: PYTHONPATH=. python scripts/run_phase3_adapter_recovery_synthetic.py [config.yaml]
"""
import json
import sys
import time
from pathlib import Path

import torch
import yaml

from medgate.attacks.adapter_recovery import low_rank_completion_attack, two_attacker_collusion_completion_attack
from medgate.data.hierarchical_synthetic import HierarchicalConfig, make_hierarchical_institutions, split_by_patient
from medgate.data.synthetic import COARSE_CLASSES, FINE_CLASSES, make_synthetic_train_test_centers
from medgate.federated.capability_isolation import train_capability_isolation
from medgate.federated.pretrain import build_coarse_pretrained_checkpoint
from medgate.metrics import evaluate_fine
from scripts.run_phase1_synthetic import git_commit, manifest_hash


def train_null_signal_combined(cfg, seed):
    d, t = cfg["data"], cfg["train"]
    model_kwargs = dict(num_coarse=len(COARSE_CLASSES), num_fine=len(FINE_CLASSES), **cfg["model"])
    train_centers, test_centers = make_synthetic_train_test_centers(
        samples_per_center=d["samples_per_center"], image_size=d["image_size"], seed=d["data_seed"]
    )
    model = train_capability_isolation(
        "combined", train_centers, model_kwargs, t["rounds"], t["epochs_per_round"], t["batch_size"], t["lr"], seed
    )
    test_pool = torch.utils.data.ConcatDataset(test_centers)
    return model, test_pool


def train_hierarchical_combined(cfg, seed):
    d, pt, t = cfg["data"], cfg["pretrain"], cfg["train"]
    model_kwargs = dict(num_coarse=len(COARSE_CLASSES), num_fine=len(FINE_CLASSES), **cfg["model"])
    hcfg = HierarchicalConfig(
        image_size=d["image_size"],
        num_patients_per_institution=d["num_patients_per_institution"],
        observations_per_patient=d["observations_per_patient"],
        class_imbalance_strength=d["class_imbalance_strength"],
        sensitive_property_correlation=d["sensitive_property_correlation"],
    )
    insts = make_hierarchical_institutions(hcfg, seed=seed)
    train, _val, test = split_by_patient(insts, train_frac=d["train_frac"], val_frac=d["val_frac"], seed=seed)
    test_pool = torch.utils.data.ConcatDataset(test)

    coarse_ckpt = build_coarse_pretrained_checkpoint(
        torch.utils.data.ConcatDataset(train), model_kwargs, pt["epochs"], pt["batch_size"], pt["lr"], seed
    )
    model = train_capability_isolation(
        "combined", train, model_kwargs, t["rounds"], t["epochs_per_round"], t["batch_size"], t["lr"], seed,
        init_state_dict=coarse_ckpt.state_dict(),
    )
    return model, test_pool


def run_arm(arm_name, train_fn, cfg, seed, commit) -> dict:
    start = time.time()
    model, test_pool = train_fn(cfg[arm_name], seed)
    u_authorized = evaluate_fine(model, test_pool)["fine_macro_f1"]

    result = {"arm": arm_name, "seed": seed, "git_commit": commit, "u_authorized_fine_macro_f1": u_authorized}

    # --- Fill-method x reveal-fraction sweep (P0-A #3, #4) ---
    sweep = []
    for method in cfg["fill_methods"]:
        for reveal_fraction in cfg["reveal_fractions"]:
            r = low_rank_completion_attack(
                model, test_pool, reveal_fraction, seed,
                method=method, soft_lam=cfg["soft_impute_lam"], completion_iterations=cfg["completion_iterations"],
            )
            sweep.append(r)
    result["fill_method_sweep"] = sweep

    # --- Rank misspecification sweep (P0-A #4), hard_impute only ---
    rank_sweep = []
    fixed_reveal = cfg["rank_misspecification_reveal_fraction"]
    true_rank = model.adapter.down.out_features
    for delta in cfg["rank_misspecification_deltas"]:
        candidate_rank = max(1, true_rank + delta)
        r = low_rank_completion_attack(
            model, test_pool, fixed_reveal, seed,
            method="hard_impute", candidate_rank=candidate_rank, completion_iterations=cfg["completion_iterations"],
        )
        rank_sweep.append(r)
    result["rank_misspecification_sweep"] = rank_sweep

    # --- Collusion (hard_impute) ---
    result["collusion"] = two_attacker_collusion_completion_attack(
        model, test_pool, cfg["collusion_reveal_fraction_each"], seed,
        method="hard_impute", completion_iterations=cfg["completion_iterations"],
    )

    result["wall_clock_seconds"] = round(time.time() - start, 2)
    return result


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/phase3_adapter_recovery_synthetic.yaml"
    cfg = yaml.safe_load(Path(config_path).read_text())
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    commit = git_commit()

    for seed in cfg["seeds"]:
        for arm_name, train_fn in (("null_signal", train_null_signal_combined), ("hierarchical", train_hierarchical_combined)):
            result = run_arm(arm_name, train_fn, cfg, seed, commit)
            result["config"] = cfg
            result["dataset_manifest_hash"] = manifest_hash({"arm": arm_name, **cfg[arm_name]["data"]})
            out_path = out_dir / f"{arm_name}_seed{seed}.json"
            out_path.write_text(json.dumps(result, indent=2))
            best = next(r for r in result["fill_method_sweep"] if r["method"] == "hard_impute" and r["compute_budget"]["reveal_fraction"] == 0.9)
            print(
                f"arm={arm_name} seed={seed} u_authorized={result['u_authorized_fine_macro_f1']:.3f} "
                f"hard_impute@90pct_cos_sim={best['parameter_space_recovery']['cosine_similarity']:.3f} "
                f"unobs_err={best['parameter_space_recovery']['unobserved_entry_normalized_error']:.3f} "
                f"gain_over_zerofill={best['completion_gain_over_zero_fill']['absolute_gain']:.3f} "
                f"({result['wall_clock_seconds']:.1f}s) -> {out_path}"
            )


if __name__ == "__main__":
    main()
