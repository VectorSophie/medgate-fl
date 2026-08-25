#!/usr/bin/env python3
"""Phase 3 continued: A4 integrity attacks vs. aggregators, plus property
inference, synthetic tier (docs/execution_plan.md). One client (index
`malicious_client_index`) is replaced by each attack in turn; the
resulting global model (after 3 rounds) is compared across four
aggregators: plaintext FedAvg, validated FedAvg (drops malformed updates),
coordinate-wise median, and trimmed mean.

Usage: PYTHONPATH=. python scripts/run_phase3_integrity_synthetic.py [config.yaml]
"""
import copy
import json
import sys
import time
from pathlib import Path

import torch
import yaml

from medgate.attacks.integrity import (
    backdoor_success_rate,
    backdoor_train,
    free_rider_update,
    label_flipping_train,
    malformed_update,
    model_replacement_update,
    sign_flipping_update,
)
from medgate.attacks.property_inference import majority_fine_class_property, property_inference_attack
from medgate.data.synthetic import COARSE_CLASSES, FINE_CLASSES, make_synthetic_train_test_centers
from medgate.federated.fedavg import fedavg_aggregate, local_train
from medgate.federated.robust_aggregation import (
    coordinate_median_aggregate,
    trimmed_mean_aggregate,
    validated_fedavg_aggregate,
)
from medgate.metrics import evaluate_fine
from medgate.models.backbone import MedGateModel
from scripts.run_phase1_synthetic import git_commit, manifest_hash

AGGREGATORS = {
    "fedavg": lambda states, weights: fedavg_aggregate(states, weights),
    "validated_fedavg": lambda states, weights: validated_fedavg_aggregate(states, weights),
    "coordinate_median": lambda states, weights: coordinate_median_aggregate(states),
    "trimmed_mean": lambda states, weights: trimmed_mean_aggregate(states, trim_fraction=0.2),
}


def malicious_state(attack: str, local_model, dataset, global_state: dict, epochs, batch_size, lr, cfg) -> dict:
    if attack == "none":
        return local_train(local_model, dataset, epochs, batch_size, lr)
    if attack == "label_flip":
        return label_flipping_train(local_model, dataset, epochs, batch_size, lr)
    if attack == "backdoor":
        return backdoor_train(local_model, dataset, epochs, batch_size, lr, cfg["backdoor_target_fine_class"])
    if attack == "free_rider":
        return free_rider_update(global_state)
    honest = local_train(local_model, dataset, epochs, batch_size, lr)
    if attack == "sign_flip":
        return sign_flipping_update(honest, global_state)
    if attack == "model_replacement":
        return model_replacement_update(honest, global_state, cfg["model_replacement_boost"])
    if attack == "malformed":
        return malformed_update(honest, "nan")
    raise ValueError(attack)


def run_attack(attack: str, aggregator_name: str, seed: int, cfg: dict, train_centers, test_pool, backdoor_eval_dataset) -> dict:
    model_kwargs = dict(num_coarse=len(COARSE_CLASSES), num_fine=len(FINE_CLASSES), **cfg["model"])
    t = cfg["train"]
    torch.manual_seed(seed)
    global_model = MedGateModel(**model_kwargs)
    mal_idx = cfg["malicious_client_index"]
    aggregate_fn = AGGREGATORS[aggregator_name]

    for _ in range(t["rounds"]):
        global_state = {k: v.clone() for k, v in global_model.state_dict().items()}
        states, weights = [], []
        for i, ds in enumerate(train_centers):
            local_model = copy.deepcopy(global_model)
            if i == mal_idx:
                state = malicious_state(attack, local_model, ds, global_state, t["epochs_per_round"], t["batch_size"], t["lr"], cfg)
            else:
                state = local_train(local_model, ds, t["epochs_per_round"], t["batch_size"], t["lr"])
            states.append(state)
            weights.append(float(len(ds)))
        try:
            aggregated = aggregate_fn(states, weights)
        except ValueError as e:
            return {"attack": attack, "aggregator": aggregator_name, "seed": seed, "aggregation_failed": str(e)}
        global_model.load_state_dict(aggregated)

    utility = evaluate_fine(global_model, test_pool)
    result = {
        "attack": attack, "aggregator": aggregator_name, "seed": seed,
        "aggregation_failed": None,
        "fine_macro_f1": utility["fine_macro_f1"],
        "model_has_nan": any(torch.isnan(v).any().item() for v in global_model.state_dict().values()),
    }
    if attack == "backdoor" and not result["model_has_nan"]:
        result["backdoor_success_rate"] = backdoor_success_rate(
            global_model, backdoor_eval_dataset, cfg["backdoor_target_fine_class"]
        )
    return result


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/phase3_integrity_synthetic.yaml"
    cfg = yaml.safe_load(Path(config_path).read_text())
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    commit = git_commit()

    d = cfg["data"]
    train_centers, test_centers = make_synthetic_train_test_centers(
        samples_per_center=d["samples_per_center"], image_size=d["image_size"], seed=d["data_seed"]
    )
    test_pool = torch.utils.data.ConcatDataset(test_centers)
    backdoor_eval_dataset = test_centers[0]  # a real SyntheticFedISIC, needed by backdoor_success_rate

    full_attacks = ["none", "label_flip", "backdoor", "sign_flip", "model_replacement", "free_rider"]
    full_aggregators = ["fedavg", "coordinate_median", "trimmed_mean"]
    malformed_aggregators = ["fedavg", "validated_fedavg"]

    for seed in cfg["seeds"]:
        results = []
        start = time.time()
        for attack in full_attacks:
            for agg in full_aggregators:
                results.append(run_attack(attack, agg, seed, cfg, train_centers, test_pool, backdoor_eval_dataset))
        for agg in malformed_aggregators:
            results.append(run_attack("malformed", agg, seed, cfg, train_centers, test_pool, backdoor_eval_dataset))

        # property inference: one honest round, attacker observes plaintext per-client updates
        torch.manual_seed(seed)
        model_kwargs = dict(num_coarse=len(COARSE_CLASSES), num_fine=len(FINE_CLASSES), **cfg["model"])
        pi_model = MedGateModel(**model_kwargs)
        t = cfg["train"]
        client_states = [
            local_train(copy.deepcopy(pi_model), ds, t["epochs_per_round"], t["batch_size"], t["lr"])
            for ds in train_centers
        ]
        target_class = cfg["backdoor_target_fine_class"]
        property_labels = [majority_fine_class_property(ds, target_class) for ds in train_centers]
        pi_result = property_inference_attack(client_states, property_labels, seed=seed)

        out = {
            "seed": seed, "git_commit": commit, "config": cfg,
            "dataset_manifest_hash": manifest_hash(cfg["data"]),
            "wall_clock_seconds": round(time.time() - start, 2),
            "integrity_results": results,
            "property_inference": pi_result,
        }
        out_path = out_dir / f"seed{seed}.json"
        out_path.write_text(json.dumps(out, indent=2))
        print(f"seed={seed} ({out['wall_clock_seconds']:.1f}s) -> {out_path}")
        for r in results:
            if r.get("aggregation_failed"):
                print(f"    {r['attack']:18s} {r['aggregator']:18s} AGGREGATION FAILED: {r['aggregation_failed']}")
            else:
                print(f"    {r['attack']:18s} {r['aggregator']:18s} fine_f1={r['fine_macro_f1']:.3f} nan={r['model_has_nan']}")
        print(f"    property_inference AUC={pi_result.get('attack_auc')}")


if __name__ == "__main__":
    main()
