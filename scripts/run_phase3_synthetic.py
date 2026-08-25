#!/usr/bin/env python3
"""Phase 3 security attacks, synthetic tier (docs/execution_plan.md).
Trains one "combined"-method authorized model per seed, then runs:
gradient inversion (A1), loss-threshold membership inference (A1/A3),
adapter reconstruction at several auxiliary-data budgets (A2), black-box
extraction at several query budgets (A3), and two-attacker collusion.

Usage: PYTHONPATH=. python scripts/run_phase3_synthetic.py [config.yaml]
"""
import json
import sys
import time
from pathlib import Path

import torch
import yaml

from medgate.attacks.gradient_inversion import dlg_attack
from medgate.attacks.membership_inference import loss_threshold_membership_inference
from medgate.attacks.reconstruction import adapter_reconstruction_attack, black_box_extraction_attack, collusion_attack
from medgate.capability_metrics import capability_recovery_efficiency
from medgate.data.synthetic import COARSE_CLASSES, FINE_CLASSES, make_synthetic_centers, make_synthetic_train_test_centers
from medgate.federated.capability_isolation import train_capability_isolation
from medgate.metrics import evaluate_fine
from scripts.run_phase1_synthetic import git_commit, manifest_hash


def run_seed(seed: int, cfg: dict, train_centers, test_centers, aux_centers, commit: str) -> dict:
    model_kwargs = dict(num_coarse=len(COARSE_CLASSES), num_fine=len(FINE_CLASSES), **cfg["model"])
    t = cfg["train"]
    test_pool = torch.utils.data.ConcatDataset(test_centers)
    aux_pool = torch.utils.data.ConcatDataset(aux_centers)

    authorized_model = train_capability_isolation(
        "combined", train_centers, model_kwargs, t["rounds"], t["epochs_per_round"], t["batch_size"], t["lr"], seed
    )
    u_authorized = evaluate_fine(authorized_model, test_pool)["fine_macro_f1"]

    result = {"seed": seed, "git_commit": commit, "u_authorized_fine_macro_f1": u_authorized}

    # --- A1: gradient inversion (DLG) on one example from client 0 ---
    dlg_cfg = cfg["dlg"]
    img, y_fine, y_coarse = train_centers[0][0]
    dlg_result = dlg_attack(
        authorized_model, img.unsqueeze(0), y_coarse.unsqueeze(0), y_fine.unsqueeze(0),
        steps=dlg_cfg["steps"], lr=dlg_cfg["lr"], seed=seed,
    )
    result["gradient_inversion"] = dlg_result

    # --- A1/A3: loss-threshold membership inference (member=client 0's
    # train data, non-member=held-out test pool) ---
    mi_result = loss_threshold_membership_inference(authorized_model, train_centers[0], test_pool)
    result["membership_inference"] = mi_result

    # --- A2: adapter reconstruction at several auxiliary-data budgets ---
    ar_cfg = cfg["adapter_reconstruction"]
    result["adapter_reconstruction"] = []
    for budget in ar_cfg["budgets"]:
        aux_subset = torch.utils.data.Subset(aux_pool, range(min(budget, len(aux_pool))))
        attacker_model, meta = adapter_reconstruction_attack(
            authorized_model, aux_subset, ar_cfg["epochs"], min(8, budget), t["lr"], seed
        )
        u_attack = evaluate_fine(attacker_model, test_pool)["fine_macro_f1"]
        result["adapter_reconstruction"].append({
            **meta, "u_attack_fine_macro_f1": u_attack,
            "capability_recovery_efficiency": capability_recovery_efficiency(u_attack, 0.0, budget),
        })

    # --- A3: black-box extraction at several query budgets ---
    ex_cfg = cfg["extraction"]
    result["extraction"] = []
    for q in ex_cfg["query_budgets"]:
        query_images = torch.stack([aux_pool[i][0] for i in range(min(q, len(aux_pool)))])
        student, meta = black_box_extraction_attack(
            authorized_model, query_images, model_kwargs, ex_cfg["epochs"], min(8, q), t["lr"], seed
        )
        u_attack = evaluate_fine(student, test_pool)["fine_macro_f1"]
        result["extraction"].append({
            **meta, "u_attack_fine_macro_f1": u_attack,
            "capability_recovery_efficiency": capability_recovery_efficiency(u_attack, 0.0, q),
        })

    # --- Collusion: two attackers, disjoint auxiliary halves ---
    col_cfg = cfg["collusion"]
    b = col_cfg["budget_each"]
    aux_a = torch.utils.data.Subset(aux_pool, range(0, b))
    aux_b = torch.utils.data.Subset(aux_pool, range(b, 2 * b))
    ensemble, col_meta = collusion_attack(authorized_model, aux_a, aux_b, col_cfg["epochs"], min(8, b), t["lr"], seed)
    u_colluded = evaluate_fine(ensemble, test_pool)["fine_macro_f1"]
    u_solo_a = evaluate_fine(_rebuild_solo(authorized_model, aux_a, col_cfg, t), test_pool)["fine_macro_f1"]
    result["collusion"] = {
        **col_meta,
        "u_colluded_fine_macro_f1": u_colluded,
        "u_solo_attacker_fine_macro_f1": u_solo_a,
        "collusion_gain_over_solo": u_colluded - u_solo_a,
    }

    return result


def _rebuild_solo(authorized_model, aux_a, col_cfg, t):
    model, _ = adapter_reconstruction_attack(authorized_model, aux_a, col_cfg["epochs"], min(8, len(aux_a)), t["lr"], 0)
    return model


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/phase3_synthetic.yaml"
    cfg = yaml.safe_load(Path(config_path).read_text())
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    commit = git_commit()

    d = cfg["data"]
    train_centers, test_centers = make_synthetic_train_test_centers(
        samples_per_center=d["samples_per_center"], image_size=d["image_size"], seed=d["data_seed"]
    )
    aux_centers = make_synthetic_centers(
        samples_per_center=d["auxiliary_samples_per_center"], image_size=d["image_size"], seed=d["data_seed"] + 20_000
    )

    for seed in cfg["seeds"]:
        start = time.time()
        result = run_seed(seed, cfg, train_centers, test_centers, aux_centers, commit)
        result["config"] = cfg
        result["dataset_manifest_hash"] = manifest_hash(cfg["data"])
        result["wall_clock_seconds"] = round(time.time() - start, 2)
        out_path = out_dir / f"seed{seed}.json"
        out_path.write_text(json.dumps(result, indent=2))
        print(
            f"seed={seed} u_authorized={result['u_authorized_fine_macro_f1']:.3f} "
            f"dlg_psnr={result['gradient_inversion']['psnr_db']:.1f}dB "
            f"mi_auc={result['membership_inference']['attack_auc']:.3f} "
            f"({result['wall_clock_seconds']:.1f}s) -> {out_path}"
        )


if __name__ == "__main__":
    main()
