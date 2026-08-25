#!/usr/bin/env python3
"""Phase 3 P1-7 genuine parameter-space adapter-recovery attack, synthetic
tier (docs/execution_plan.md). Trains one "combined"-method authorized
model per seed (same protocol as scripts/run_phase3_synthetic.py), then
runs single-attacker low-rank completion at several reveal fractions and a
two-attacker collusion variant -- distinct from A2/A3 in
scripts/run_phase3_synthetic.py, which never touch the true adapter
weights at all.

Usage: PYTHONPATH=. python scripts/run_phase3_adapter_recovery_synthetic.py [config.yaml]
"""
import json
import sys
import time
from pathlib import Path

import torch
import yaml

from medgate.attacks.adapter_recovery import low_rank_completion_attack, two_attacker_collusion_completion_attack
from medgate.data.synthetic import COARSE_CLASSES, FINE_CLASSES, make_synthetic_train_test_centers
from medgate.federated.capability_isolation import train_capability_isolation
from medgate.metrics import evaluate_fine
from scripts.run_phase1_synthetic import git_commit, manifest_hash


def run_seed(seed: int, cfg: dict, train_centers, test_centers, commit: str) -> dict:
    model_kwargs = dict(num_coarse=len(COARSE_CLASSES), num_fine=len(FINE_CLASSES), **cfg["model"])
    t = cfg["train"]
    test_pool = torch.utils.data.ConcatDataset(test_centers)

    authorized_model = train_capability_isolation(
        "combined", train_centers, model_kwargs, t["rounds"], t["epochs_per_round"], t["batch_size"], t["lr"], seed
    )
    u_authorized = evaluate_fine(authorized_model, test_pool)["fine_macro_f1"]

    result = {"seed": seed, "git_commit": commit, "u_authorized_fine_macro_f1": u_authorized, "solo": [], "collusion": None}

    for reveal_fraction in cfg["reveal_fractions"]:
        r = low_rank_completion_attack(authorized_model, test_pool, reveal_fraction, seed, cfg["completion_iterations"])
        result["solo"].append(r)

    result["collusion"] = two_attacker_collusion_completion_attack(
        authorized_model, test_pool, cfg["collusion_reveal_fraction_each"], seed, cfg["completion_iterations"]
    )
    return result


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/phase3_adapter_recovery_synthetic.yaml"
    cfg = yaml.safe_load(Path(config_path).read_text())
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    commit = git_commit()

    d = cfg["data"]
    train_centers, test_centers = make_synthetic_train_test_centers(
        samples_per_center=d["samples_per_center"], image_size=d["image_size"], seed=d["data_seed"]
    )

    for seed in cfg["seeds"]:
        start = time.time()
        result = run_seed(seed, cfg, train_centers, test_centers, commit)
        result["config"] = cfg
        result["dataset_manifest_hash"] = manifest_hash(cfg["data"])
        result["wall_clock_seconds"] = round(time.time() - start, 2)
        out_path = out_dir / f"seed{seed}.json"
        out_path.write_text(json.dumps(result, indent=2))
        best = result["solo"][-1]["parameter_space_recovery"]
        print(
            f"seed={seed} u_authorized={result['u_authorized_fine_macro_f1']:.3f} "
            f"90pct_reveal_cos_sim={best['cosine_similarity']:.3f} "
            f"90pct_reveal_frob_err={best['normalized_frobenius_error']:.3f} "
            f"({result['wall_clock_seconds']:.1f}s) -> {out_path}"
        )


if __name__ == "__main__":
    main()
