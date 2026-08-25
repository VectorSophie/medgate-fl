#!/usr/bin/env python3
"""Phase 4 privacy mechanisms, synthetic tier (docs/execution_plan.md).
Compares four arms per seed: no protection (plain FedAvg), secure
aggregation only (pairwise masking), DP-SGD only (at several noise
multipliers), and secure aggregation + DP-SGD combined. Reports utility and
(for DP arms) the achieved epsilon, so a privacy-utility Pareto curve can
be drawn from the raw numbers.

Usage: PYTHONPATH=. python scripts/run_phase4_synthetic.py [config.yaml]
"""
import json
import sys
import time
from pathlib import Path

import torch
import yaml

from medgate.data.synthetic import COARSE_CLASSES, FINE_CLASSES, make_synthetic_train_test_centers
from medgate.federated.fedavg import run_fedavg_round
from medgate.metrics import evaluate_both
from medgate.models.backbone import MedGateModel
from medgate.privacy.dp_sgd import dp_fedavg_round, secure_dp_fedavg_round
from medgate.privacy.secure_aggregation import secure_fedavg_round
from scripts.run_phase1_synthetic import git_commit, manifest_hash


def run_arm(arm_name: str, seed: int, cfg: dict, train_centers, test_pool, noise_multiplier=None) -> dict:
    torch.manual_seed(seed)
    model_kwargs = dict(num_coarse=len(COARSE_CLASSES), num_fine=len(FINE_CLASSES), **cfg["model"])
    model = MedGateModel(**model_kwargs)
    t = cfg["train"]
    dp = cfg["dp"]
    epsilons = []

    for _ in range(t["rounds"]):
        if arm_name == "no_protection":
            state = run_fedavg_round(model, train_centers, t["epochs_per_round"], t["batch_size"], t["lr"])
        elif arm_name == "secure_agg":
            state = secure_fedavg_round(model, train_centers, t["epochs_per_round"], t["batch_size"], t["lr"], seed)
        elif arm_name == "dp_sgd":
            state, eps = dp_fedavg_round(
                model, train_centers, t["epochs_per_round"], t["batch_size"], t["lr"],
                noise_multiplier, dp["max_grad_norm"], dp["delta"],
            )
            epsilons.append(eps)
        elif arm_name == "secure_agg_plus_dp":
            state, eps = secure_dp_fedavg_round(
                model, train_centers, t["epochs_per_round"], t["batch_size"], t["lr"],
                noise_multiplier, seed, dp["max_grad_norm"], dp["delta"],
            )
            epsilons.append(eps)
        else:
            raise ValueError(arm_name)
        model.load_state_dict(state)

    utility = evaluate_both(model, test_pool)
    result = {
        "arm": arm_name,
        "noise_multiplier": noise_multiplier,
        "seed": seed,
        "epsilon": max(epsilons) if epsilons else None,
        "utility": utility,
    }
    if arm_name in ("dp_sgd", "secure_agg_plus_dp"):
        # P1-10: accountant metadata recorded in every DP result, not just
        # documented in prose -- see medgate/privacy/dp_sgd.py's module
        # docstring for what each field means and does not mean.
        result["dp_accountant_metadata"] = {
            "adjacency": "example-level (record-level), NOT client-level",
            "accountant": "Opacus RDP accountant (default)",
            "sampling_mechanism": "shuffled mini-batches, poisson_sampling=False (not Poisson subsampling)",
            "delta": dp["delta"],
            "max_grad_norm": dp["max_grad_norm"],
            "noise_multiplier": noise_multiplier,
            "local_epochs_per_round": t["epochs_per_round"],
            "federated_rounds": t["rounds"],
            "composition_scope": "within one client's local dp_local_train call only; "
                                  "reported epsilon = max over clients THIS ROUND, "
                                  "NOT cumulative across rounds (see dp_fedavg_round docstring)",
            "params_covered": "backbone, coarse_head, adapter, fine_head (medgate.attacks.gradient_inversion.attack_params)",
            "params_excluded": "adversary_head (architecturally unused by this training path -- receives no gradient at all)",
        }
    return result


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/phase4_synthetic.yaml"
    cfg = yaml.safe_load(Path(config_path).read_text())
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    commit = git_commit()

    d = cfg["data"]
    train_centers, test_centers = make_synthetic_train_test_centers(
        samples_per_center=d["samples_per_center"], image_size=d["image_size"], seed=d["data_seed"]
    )
    test_pool = torch.utils.data.ConcatDataset(test_centers)

    for seed in cfg["seeds"]:
        results = []
        start = time.time()
        results.append(run_arm("no_protection", seed, cfg, train_centers, test_pool))
        results.append(run_arm("secure_agg", seed, cfg, train_centers, test_pool))
        for nm in cfg["dp"]["noise_multipliers"]:
            results.append(run_arm("dp_sgd", seed, cfg, train_centers, test_pool, noise_multiplier=nm))
            results.append(run_arm("secure_agg_plus_dp", seed, cfg, train_centers, test_pool, noise_multiplier=nm))

        from medgate.privacy.secure_aggregation import empirical_concealment_sanity_check
        concealment = empirical_concealment_sanity_check(shape=(64,), seed=seed, n_samples=100, mean_shift=2.0, n_bootstrap=200)

        out = {
            "seed": seed, "git_commit": commit, "config": cfg,
            "dataset_manifest_hash": manifest_hash(cfg["data"]),
            "wall_clock_seconds": round(time.time() - start, 2),
            "arms": results,
            "secure_agg_empirical_concealment_sanity_check": concealment,  # NOT a security proof -- see its docstring
        }
        out_path = out_dir / f"seed{seed}.json"
        out_path.write_text(json.dumps(out, indent=2))
        print(f"seed={seed} ({out['wall_clock_seconds']:.1f}s) -> {out_path}")
        for r in results:
            eps_str = f"eps={r['epsilon']:.2f}" if r["epsilon"] is not None else "eps=n/a"
            nm_str = f"nm={r['noise_multiplier']}" if r["noise_multiplier"] is not None else ""
            print(f"    {r['arm']:20s} {nm_str:8s} {eps_str:12s} fine_f1={r['utility']['fine_macro_f1']:.3f}")


if __name__ == "__main__":
    main()
