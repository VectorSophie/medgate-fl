#!/usr/bin/env python3
"""Phase 2 capability-isolation ablation, synthetic tier
(docs/execution_plan.md). For each method (coarse_only, hidden_fine_head,
adapter_isolation, adversarial, orthogonal, combined) x seed: train, then
run every Phase-2 probe (linear/nonlinear/knn/few-shot) against the frozen
PUBLIC representation, and compute the capability-isolation composite
metrics (ARR/UCG/RFC — CRE per probe) alongside the raw numbers.

Usage: PYTHONPATH=. python scripts/run_phase2_synthetic.py [config.yaml]
"""
import json
import sys
import time
from pathlib import Path

import torch
import yaml

from medgate.attacks.probes import output_only_probe, run_all_probes
from medgate.capability_metrics import (
    authorized_recovery_ratio,
    capability_recovery_efficiency,
    residual_fine_capability,
    unauthorized_capability_gain,
)
from medgate.data.synthetic import COARSE_CLASSES, FINE_CLASSES, make_synthetic_train_test_centers
from medgate.federated.capability_isolation import METHODS, train_capability_isolation
from medgate.metrics import evaluate_both
from scripts.run_phase1_synthetic import git_commit, manifest_hash  # reuse Phase 1 helpers


def run_one(method_name: str, seed: int, cfg: dict, train_centers, test_centers, train_pool, test_pool, commit: str) -> dict:
    model_kwargs = dict(num_coarse=len(COARSE_CLASSES), num_fine=len(FINE_CLASSES), **cfg["model"])
    t = cfg["train"]
    start = time.time()

    model = train_capability_isolation(
        method_name, train_centers, model_kwargs, t["rounds"], t["epochs_per_round"], t["batch_size"], t["lr"], seed
    )
    train_seconds = time.time() - start

    utility = evaluate_both(model, test_pool)
    u_public = output_only_probe(model, train_pool, test_pool, seed=seed)["macro_f1"]
    probes = run_all_probes(model, train_pool, test_pool, seed=seed)
    rfc = residual_fine_capability(probes)
    ucg = unauthorized_capability_gain(rfc, u_public)
    cre_by_probe = {
        name: capability_recovery_efficiency(r["macro_f1"], u_public, max(r["fit_seconds"], 1e-6))
        for name, r in probes.items()
    }

    return {
        "method": method_name,
        "seed": seed,
        "git_commit": commit,
        "config": cfg,
        "dataset_manifest_hash": manifest_hash(cfg["data"]),
        "train_wall_clock_seconds": round(train_seconds, 3),
        "utility": utility,             # authorized-path coarse/fine macro-F1 + balanced accuracy
        "u_public": u_public,           # fine macro-F1 recoverable from public OUTPUT alone
        "probes": probes,               # linear/nonlinear/knn/fewshot on the frozen PUBLIC REPRESENTATION
        "residual_fine_capability": rfc,
        "unauthorized_capability_gain": ucg,
        "capability_recovery_efficiency_by_probe": cre_by_probe,
    }


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/phase2_synthetic.yaml"
    cfg = yaml.safe_load(Path(config_path).read_text())
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    commit = git_commit()

    d = cfg["data"]
    train_centers, test_centers = make_synthetic_train_test_centers(
        samples_per_center=d["samples_per_center"], image_size=d["image_size"], seed=d["data_seed"]
    )
    train_pool = torch.utils.data.ConcatDataset(train_centers)
    test_pool = torch.utils.data.ConcatDataset(test_centers)

    assert set(cfg["methods"]) <= set(METHODS), f"unknown method(s) in config: {set(cfg['methods']) - set(METHODS)}"

    results_by_seed = {}
    for seed in cfg["seeds"]:
        results_by_seed[seed] = {}
        for method_name in cfg["methods"]:
            result = run_one(method_name, seed, cfg, train_centers, test_centers, train_pool, test_pool, commit)

            # ARR needs the plain-adapter reference (adapter_isolation) from
            # the SAME seed; compute it once that method has run this seed.
            if "adapter_isolation" in results_by_seed[seed]:
                u_plain_adapter = results_by_seed[seed]["adapter_isolation"]["utility"]["fine_macro_f1"]
                result["authorized_recovery_ratio"] = authorized_recovery_ratio(
                    result["utility"]["fine_macro_f1"], result["u_public"], u_plain_adapter
                )
            else:
                result["authorized_recovery_ratio"] = None  # filled in below once adapter_isolation exists

            results_by_seed[seed][method_name] = result
            out_path = out_dir / f"{method_name}_seed{seed}.json"
            out_path.write_text(json.dumps(result, indent=2))
            print(
                f"{method_name:18s} seed={seed} "
                f"fine_f1={result['utility']['fine_macro_f1']:.3f} "
                f"u_public={result['u_public']:.3f} rfc={result['residual_fine_capability']:.3f} "
                f"ucg={result['unauthorized_capability_gain']:+.3f} "
                f"({result['train_wall_clock_seconds']:.1f}s) -> {out_path}"
            )

        # backfill ARR for methods that ran before adapter_isolation this seed
        if "adapter_isolation" in results_by_seed[seed]:
            u_plain_adapter = results_by_seed[seed]["adapter_isolation"]["utility"]["fine_macro_f1"]
            for method_name, result in results_by_seed[seed].items():
                if result["authorized_recovery_ratio"] is None:
                    result["authorized_recovery_ratio"] = authorized_recovery_ratio(
                        result["utility"]["fine_macro_f1"], result["u_public"], u_plain_adapter
                    )
                    (out_dir / f"{method_name}_seed{seed}.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
