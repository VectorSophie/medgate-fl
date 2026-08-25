#!/usr/bin/env python3
"""Phase 5 revocation and unlearning, synthetic tier
(docs/execution_plan.md). Two removal scenarios (institution-level,
class-level — patient-level/patient-group are NOT run, see
configs/phase5_synthetic.yaml) x 5 methods (full_retrain gold standard,
checkpoint_rollback, adapter_deletion_and_retrain, gradient_ascent_unlearning,
key_revocation_only), each scored on retained-data utility (vs the gold
standard) and a membership-inference-style forgetting signal on the
removed data.

Usage: PYTHONPATH=. python scripts/run_phase5_synthetic.py [config.yaml]
"""
import json
import sys
import time
from pathlib import Path

import torch
import yaml

from medgate.attacks.membership_inference import loss_threshold_membership_inference
from medgate.data.synthetic import (
    COARSE_CLASSES,
    FINE_CLASSES,
    make_synthetic_train_test_centers,
    remove_fine_class,
    select_fine_class,
)
from medgate.federated.capability_isolation import train_capability_isolation
from medgate.metrics import evaluate_fine
from medgate.unlearning.methods import (
    adapter_deletion_and_retrain,
    checkpoint_rollback,
    full_retrain,
    gradient_ascent_unlearning,
    key_revocation_only,
)
from scripts.run_phase1_synthetic import git_commit, manifest_hash


def build_institution_scenario(train_centers, test_centers, institution_index):
    removed_data = train_centers[institution_index]
    data_after_removal = [c for i, c in enumerate(train_centers) if i != institution_index]
    retained_test = torch.utils.data.ConcatDataset(
        [c for i, c in enumerate(test_centers) if i != institution_index]
    )
    return data_after_removal, removed_data, retained_test


def build_class_scenario(train_centers, test_centers, fine_class_index):
    data_after_removal = [remove_fine_class(c, fine_class_index) for c in train_centers]
    removed_data = torch.utils.data.ConcatDataset([select_fine_class(c, fine_class_index) for c in train_centers])
    retained_test = torch.utils.data.ConcatDataset([remove_fine_class(c, fine_class_index) for c in test_centers])
    return data_after_removal, removed_data, retained_test


def run_scenario(scenario_name: str, seed: int, cfg: dict, train_centers, test_centers) -> dict:
    model_kwargs = dict(num_coarse=len(COARSE_CLASSES), num_fine=len(FINE_CLASSES), **cfg["model"])
    t = cfg["train"]
    u = cfg["unlearning"]

    authorized_model = train_capability_isolation(
        "combined", train_centers, model_kwargs, t["rounds"], t["epochs_per_round"], t["batch_size"], t["lr"], seed
    )

    if scenario_name == "institution":
        data_after_removal, removed_data, retained_test = build_institution_scenario(
            train_centers, test_centers, cfg["removal"]["institution_index"]
        )
    elif scenario_name == "class":
        data_after_removal, removed_data, retained_test = build_class_scenario(
            train_centers, test_centers, cfg["removal"]["fine_class_index"]
        )
    else:
        raise ValueError(scenario_name)

    retained_pool = torch.utils.data.ConcatDataset(data_after_removal)

    methods = {}
    gold = full_retrain(data_after_removal, model_kwargs, t["rounds"], t["epochs_per_round"], t["batch_size"], t["lr"], seed)
    methods["full_retrain"] = gold
    methods["checkpoint_rollback"] = checkpoint_rollback(model_kwargs, seed)
    methods["adapter_deletion_and_retrain"] = adapter_deletion_and_retrain(
        authorized_model, data_after_removal, u["adapter_deletion_epochs"], u["batch_size"], u["lr"], seed
    )
    methods["gradient_ascent_unlearning"] = gradient_ascent_unlearning(
        authorized_model, removed_data, retained_pool, u["ascent_steps"], u["descent_steps"], u["batch_size"], u["lr"], seed
    )
    methods["key_revocation_only"] = key_revocation_only(authorized_model)

    gold_retained_utility = evaluate_fine(gold, retained_test)["fine_macro_f1"]

    results = {}
    for name, model in methods.items():
        retained_utility = evaluate_fine(model, retained_test)["fine_macro_f1"]
        mi = loss_threshold_membership_inference(model, removed_data, retained_test)
        results[name] = {
            "retained_fine_macro_f1": retained_utility,
            "gap_to_gold_standard": retained_utility - gold_retained_utility,
            "forgetting_auc": mi["attack_auc"],  # near 0.5 = removed data no longer distinguishable from never-seen data
            "mean_removed_data_loss": mi["mean_member_loss"],
            "mean_retained_test_loss": mi["mean_nonmember_loss"],
        }

    return {"scenario": scenario_name, "seed": seed, "gold_retained_utility": gold_retained_utility, "methods": results}


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/phase5_synthetic.yaml"
    cfg = yaml.safe_load(Path(config_path).read_text())
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    commit = git_commit()

    d = cfg["data"]
    train_centers, test_centers = make_synthetic_train_test_centers(
        samples_per_center=d["samples_per_center"], image_size=d["image_size"], seed=d["data_seed"]
    )

    for seed in cfg["seeds"]:
        for scenario_name in ("institution", "class"):
            start = time.time()
            result = run_scenario(scenario_name, seed, cfg, train_centers, test_centers)
            result["config"] = cfg
            result["git_commit"] = commit
            result["dataset_manifest_hash"] = manifest_hash(cfg["data"])
            result["wall_clock_seconds"] = round(time.time() - start, 2)
            out_path = out_dir / f"{scenario_name}_seed{seed}.json"
            out_path.write_text(json.dumps(result, indent=2))
            print(f"{scenario_name:12s} seed={seed} ({result['wall_clock_seconds']:.1f}s) -> {out_path}")
            for name, m in result["methods"].items():
                print(f"    {name:28s} retained_f1={m['retained_fine_macro_f1']:.3f} gap={m['gap_to_gold_standard']:+.3f} forget_auc={m['forgetting_auc']:.3f}")


if __name__ == "__main__":
    main()
