#!/usr/bin/env python3
"""P1 requirement #6 (repair pass 4): sensitivity sweeps on the
hierarchical fixture -- site_shift_strength, fine_signal_strength,
class_imbalance_strength, and an alternative coarse ontology
(medgate/data/coarse_ontology.py). Trains the "combined" objective (the
full proposed method) at 2 seeds per sweep point against
coarse_pretrained_fedlora as the fair-baseline reference, using the same
attack-validation/attack-test probe-selection split as
scripts/run_phase1_hierarchical.py (medgate.attacks.probes.selected_probe_attack)
-- kept to 2 seeds/point (not the main sweep's 5) because this is a
sensitivity CHECK across many configurations, not a single statistical
comparison; the main sweep is the one whose numbers get 5-seed error bars.

Usage: PYTHONPATH=. python scripts/run_phase1_hierarchical_sensitivity.py [config.yaml]
"""
import json
import sys
import time
from pathlib import Path

import torch
import yaml

from medgate.attacks.probes import output_only_probe, selected_probe_attack
from medgate.data.coarse_ontology import ALTERNATIVE_FINE_TO_COARSE_IDX
from medgate.data.hierarchical_synthetic import HierarchicalConfig, make_hierarchical_institutions, split_by_patient
from medgate.data.synthetic import COARSE_CLASSES, FINE_CLASSES
from medgate.federated.baselines import train_pretrained_fedlora
from medgate.federated.capability_isolation import train_capability_isolation
from medgate.federated.pretrain import build_coarse_pretrained_checkpoint
from medgate.metrics import evaluate_both
from scripts.run_phase1_synthetic import git_commit, manifest_hash


def run_one(dimension: str, value, seed: int, cfg: dict, use_alt_ontology: bool = False) -> dict:
    d, pt, t = dict(cfg["data"]), cfg["pretrain"], cfg["train"]
    if dimension != "coarse_ontology":
        d[dimension] = value
    model_kwargs = dict(num_coarse=len(COARSE_CLASSES), num_fine=len(FINE_CLASSES), **cfg["model"])

    hcfg = HierarchicalConfig(
        image_size=d["image_size"], num_patients_per_institution=d["num_patients_per_institution"],
        observations_per_patient=d["observations_per_patient"], class_imbalance_strength=d["class_imbalance_strength"],
        sensitive_property_correlation=d["sensitive_property_correlation"],
        site_shift_strength=d.get("site_shift_strength", 1.0), fine_signal_strength=d.get("fine_signal_strength", 0.6),
    )
    fine_to_coarse = ALTERNATIVE_FINE_TO_COARSE_IDX if use_alt_ontology else None
    insts = make_hierarchical_institutions(hcfg, seed=seed, fine_to_coarse_idx=fine_to_coarse)
    train, val, test = split_by_patient(insts, train_frac=d["train_frac"], val_frac=d["val_frac"], seed=seed)
    utility_test, _unused, attack_test = split_by_patient(test, train_frac=0.5, val_frac=0.0, seed=seed + 900_000)
    train_pool = torch.utils.data.ConcatDataset(train)
    val_pool = torch.utils.data.ConcatDataset(val)
    utility_test_pool = torch.utils.data.ConcatDataset(utility_test)
    attack_test_pool = torch.utils.data.ConcatDataset(attack_test)

    coarse_ckpt = build_coarse_pretrained_checkpoint(train_pool, model_kwargs, pt["epochs"], pt["batch_size"], pt["lr"], seed)
    coarse_state = coarse_ckpt.state_dict()

    def leak(model):
        utility = evaluate_both(model, utility_test_pool)
        u_public = output_only_probe(model, train_pool, attack_test_pool, seed=seed)["macro_f1"]
        attack = selected_probe_attack(model, train_pool, val_pool, attack_test_pool, seed=seed, include_slow=True)
        return utility, u_public, attack

    baseline_model, _ = train_pretrained_fedlora(train, coarse_state, model_kwargs, t["rounds"], t["epochs_per_round"], t["batch_size"], t["lr"], seed)
    baseline_utility, baseline_u_public, baseline_attack = leak(baseline_model)

    combined_model = train_capability_isolation(
        "combined", train, model_kwargs, t["rounds"], t["epochs_per_round"], t["batch_size"], t["lr"], seed, init_state_dict=coarse_state
    )
    combined_utility, combined_u_public, combined_attack = leak(combined_model)

    return {
        "dimension": dimension, "value": value if dimension != "coarse_ontology" else ("alternative" if use_alt_ontology else "primary"),
        "seed": seed,
        "coarse_pretrained_fedlora": {
            "fine_macro_f1": baseline_utility["fine_macro_f1"], "coarse_macro_f1": baseline_utility["coarse_macro_f1"],
            "u_public": baseline_u_public, "selected_probe": baseline_attack["selected_probe"],
            "rfc": baseline_attack["attack_test_result"]["macro_f1"],
        },
        "combined": {
            "fine_macro_f1": combined_utility["fine_macro_f1"], "coarse_macro_f1": combined_utility["coarse_macro_f1"],
            "u_public": combined_u_public, "selected_probe": combined_attack["selected_probe"],
            "rfc": combined_attack["attack_test_result"]["macro_f1"],
        },
    }


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/phase1_hierarchical_sensitivity.yaml"
    cfg = yaml.safe_load(Path(config_path).read_text())
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    commit = git_commit()

    rows = []
    start = time.time()
    for dimension, values in cfg["sweeps"].items():
        for value in values:
            for seed in cfg["seeds"]:
                if dimension == "coarse_ontology":
                    row = run_one(dimension, value, seed, cfg, use_alt_ontology=(value == "alternative"))
                else:
                    row = run_one(dimension, value, seed, cfg)
                row["git_commit"] = commit
                rows.append(row)
                print(f"{dimension}={value} seed={seed}: combined fine_f1={row['combined']['fine_macro_f1']:.3f} "
                      f"rfc={row['combined']['rfc']:.3f} ({row['combined']['selected_probe']})")

    out = {
        "git_commit": commit, "config": cfg,
        "dataset_manifest_hash": manifest_hash({"fixture": "hierarchical_sensitivity", **cfg["data"]}),
        "wall_clock_seconds": round(time.time() - start, 2),
        "rows": rows,
    }
    out_path = out_dir / "sensitivity.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"({out['wall_clock_seconds']:.1f}s) -> {out_path}")


if __name__ == "__main__":
    main()
