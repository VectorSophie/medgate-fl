#!/usr/bin/env python3
"""P1 requirement #5 (repair pass 4): patient-level and patient-group
unlearning, run on the HIERARCHICAL fixture, which (unlike the null-signal
fixture used by scripts/run_phase5_synthetic.py) actually carries
patient_ids and real learnable signal -- the null-signal script's docstring
explicitly noted patient-level removal was "NOT RUN: the synthetic fixture
has no patient/lesion manifest to remove by," which was true for THAT
fixture but is no longer true project-wide once the hierarchical fixture
exists. Same five methods, same primary metric
(medgate.attacks.membership_inference SymmetricAUC/AttackAdvantage) as
scripts/run_phase5_synthetic.py, applied to two new scenarios:
  - patient: removes ONE patient's observations (a handful of images).
  - patient_group: removes a small GROUP of patients from one institution
    (a coarser removal granularity than a single patient, finer than a
    whole institution).
Neither scenario has the class-composition confound scripts/run_phase5_synthetic.py's
class scenario needed a special control pool for: a single patient (or
small patient group)'s fine-label draw is itself i.i.d. sampled
(medgate.data.hierarchical_synthetic._sample_fine_labels is called at
PATIENT granularity), so comparing removed-patient data directly against
the ordinary held-out test pool carries no systematic class-imbalance bias
the way removing an entire class does.

Usage: PYTHONPATH=. python scripts/run_phase5_hierarchical_unlearning.py [config.yaml]
"""
import json
import sys
import time
from pathlib import Path

import torch
import yaml

from medgate.attacks.membership_inference import loss_threshold_membership_inference
from medgate.data.hierarchical_synthetic import HierarchicalConfig, make_hierarchical_institutions, split_by_patient
from medgate.data.synthetic import COARSE_CLASSES, FINE_CLASSES
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


def _subset(ds, idx: torch.Tensor):
    return torch.utils.data.Subset(ds, idx.tolist())


def build_patient_scenario(train_insts: list, institution_index: int, n_patients: int):
    """Removes `n_patients` patient(s) from ONE institution (n_patients=1
    for the 'patient' scenario, >1 for 'patient_group'). Returns
    (data_after_removal: list[Dataset-like per institution],
    removed_data: Dataset, removed_patient_ids: list)."""
    target = train_insts[institution_index]
    unique_patients = torch.unique(target.patient_ids)
    removed_ids = set(unique_patients[:n_patients].tolist())  # deterministic given the (seeded) institution build
    removed_mask = torch.tensor([pid in removed_ids for pid in target.patient_ids.tolist()])
    kept_mask = ~removed_mask
    removed_idx = torch.nonzero(removed_mask, as_tuple=True)[0]
    kept_idx = torch.nonzero(kept_mask, as_tuple=True)[0]

    removed_data = _subset(target, removed_idx)
    kept_target = _subset(target, kept_idx)
    data_after_removal = [kept_target if i == institution_index else inst for i, inst in enumerate(train_insts)]
    return data_after_removal, removed_data, sorted(removed_ids)


def run_scenario(scenario_name: str, seed: int, cfg: dict, train_insts, test_pool) -> dict:
    model_kwargs = dict(num_coarse=len(COARSE_CLASSES), num_fine=len(FINE_CLASSES), **cfg["model"])
    t, u = cfg["train"], cfg["unlearning"]
    n_patients = 1 if scenario_name == "patient" else cfg["removal"]["patient_group_size"]
    institution_index = cfg["removal"]["institution_index"]

    authorized_model = train_capability_isolation(
        "combined", train_insts, model_kwargs, t["rounds"], t["epochs_per_round"], t["batch_size"], t["lr"], seed
    )
    data_after_removal, removed_data, removed_patient_ids = build_patient_scenario(train_insts, institution_index, n_patients)
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

    gold_retained_utility = evaluate_fine(gold, test_pool)["fine_macro_f1"]

    results = {}
    for name, model in methods.items():
        retained_utility = evaluate_fine(model, test_pool)["fine_macro_f1"]
        mi = loss_threshold_membership_inference(model, removed_data, test_pool)
        results[name] = {
            "retained_fine_macro_f1": retained_utility,
            "gap_to_gold_standard": retained_utility - gold_retained_utility,
            "forgetting_symmetric_auc": mi["symmetric_auc"],
            "forgetting_attack_advantage": mi["attack_advantage"],
            "forgetting_raw_auc_diagnostic_only": mi["attack_auc"],
        }
    return {
        "scenario": scenario_name, "seed": seed, "n_removed_patients": n_patients,
        "removed_patient_ids": removed_patient_ids, "n_removed_examples": len(removed_data),
        "gold_retained_utility": gold_retained_utility, "methods": results,
    }


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/phase5_hierarchical_unlearning.yaml"
    cfg = yaml.safe_load(Path(config_path).read_text())
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    commit = git_commit()
    d = cfg["data"]

    for seed in cfg["seeds"]:
        hcfg = HierarchicalConfig(
            image_size=d["image_size"], num_patients_per_institution=d["num_patients_per_institution"],
            observations_per_patient=d["observations_per_patient"], class_imbalance_strength=d["class_imbalance_strength"],
            sensitive_property_correlation=d["sensitive_property_correlation"],
        )
        insts = make_hierarchical_institutions(hcfg, seed=seed)
        train_insts, _val, test_insts = split_by_patient(insts, train_frac=d["train_frac"], val_frac=d["val_frac"], seed=seed)
        test_pool = torch.utils.data.ConcatDataset(test_insts)

        for scenario_name in ("patient", "patient_group"):
            start = time.time()
            result = run_scenario(scenario_name, seed, cfg, train_insts, test_pool)
            result["config"] = cfg
            result["git_commit"] = commit
            result["dataset_manifest_hash"] = manifest_hash({"fixture": "hierarchical_unlearning", **d})
            result["wall_clock_seconds"] = round(time.time() - start, 2)
            out_path = out_dir / f"{scenario_name}_seed{seed}.json"
            out_path.write_text(json.dumps(result, indent=2))
            print(f"{scenario_name:14s} seed={seed} n_removed_examples={result['n_removed_examples']} ({result['wall_clock_seconds']:.1f}s) -> {out_path}")
            for name, m in result["methods"].items():
                print(f"    {name:28s} retained_f1={m['retained_fine_macro_f1']:.3f} gap={m['gap_to_gold_standard']:+.3f} "
                      f"forget_symAUC={m['forgetting_symmetric_auc']:.3f} advantage={m['forgetting_attack_advantage']:.3f}")


if __name__ == "__main__":
    main()
