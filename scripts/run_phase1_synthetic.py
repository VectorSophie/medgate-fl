#!/usr/bin/env python3
"""Phase 1 primary utility baseline, synthetic tier (docs/execution_plan.md).

Trains centralized / local-only / FedAvg / FedProx / plain-FedLoRA on the
synthetic fixture across multiple seeds, evaluates each, and writes one
machine-readable JSON per (baseline, seed) run under experiments/ so every
number in a later table traces back to a config + seed + commit + raw
metrics file (reproducibility requirement).

Usage: PYTHONPATH=. python scripts/run_phase1_synthetic.py [config.yaml]
"""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml

from medgate.data.synthetic import COARSE_CLASSES, FINE_CLASSES, make_synthetic_train_test_centers
from medgate.federated import baselines
from medgate.metrics import evaluate_both


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5
        ).stdout.strip() or "uncommitted"
    except FileNotFoundError:
        return "unknown"


def manifest_hash(data_cfg: dict) -> str:
    """Synthetic tier has no real dataset manifest (docs/research_scope.md
    dataset-manifest section applies once real data is used); stand in
    with a hash of the exact generation parameters so runs are still
    traceable to what data they saw."""
    blob = json.dumps(data_cfg, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def evaluate_federated_model(model, test_centers) -> dict:
    pooled_test = torch.utils.data.ConcatDataset(test_centers)
    metrics = {"pooled": evaluate_both(model, pooled_test)}
    per_institution = [evaluate_both(model, ds) for ds in test_centers]
    metrics["per_institution"] = per_institution
    metrics["worst_institution_fine_macro_f1"] = min(m["fine_macro_f1"] for m in per_institution)
    return metrics


def evaluate_local_only_models(models, test_centers) -> dict:
    per_institution = [evaluate_both(m, ds) for m, ds in zip(models, test_centers)]
    pooled = {
        k: sum(m[k] for m in per_institution) / len(per_institution)
        for k in per_institution[0]
    }
    return {
        "pooled": pooled,  # average of per-client (train-on-own, test-on-own) scores; no single shared model exists
        "per_institution": per_institution,
        "worst_institution_fine_macro_f1": min(m["fine_macro_f1"] for m in per_institution),
    }


def run_one(baseline_name: str, seed: int, cfg: dict, train_centers, test_centers, commit: str) -> dict:
    model_kwargs = dict(
        num_coarse=len(COARSE_CLASSES), num_fine=len(FINE_CLASSES), **cfg["model"]
    )
    t = cfg["train"]
    start = time.time()

    if baseline_name == "centralized":
        pooled_train = torch.utils.data.ConcatDataset(train_centers)
        model = baselines.train_centralized(
            pooled_train, model_kwargs, t["epochs_per_round"] * t["rounds"], t["batch_size"], t["lr"], seed
        )
        eval_metrics = evaluate_federated_model(model, test_centers)
    elif baseline_name == "local_only":
        models = baselines.train_local_only(
            train_centers, model_kwargs, t["epochs_per_round"] * t["rounds"], t["batch_size"], t["lr"], seed
        )
        eval_metrics = evaluate_local_only_models(models, test_centers)
    elif baseline_name == "fedavg":
        model = baselines.train_fedavg(
            train_centers, model_kwargs, t["rounds"], t["epochs_per_round"], t["batch_size"], t["lr"], seed
        )
        eval_metrics = evaluate_federated_model(model, test_centers)
    elif baseline_name == "fedprox":
        model = baselines.train_fedprox(
            train_centers, model_kwargs, t["rounds"], t["epochs_per_round"], t["batch_size"], t["lr"], seed,
            mu=t["fedprox_mu"],
        )
        eval_metrics = evaluate_federated_model(model, test_centers)
    elif baseline_name == "fedlora":
        model = baselines.train_fedlora(
            train_centers, model_kwargs, t["rounds"], t["epochs_per_round"], t["batch_size"], t["lr"], seed
        )
        eval_metrics = evaluate_federated_model(model, test_centers)
    else:
        raise ValueError(f"unknown baseline {baseline_name}")

    wall_clock_s = time.time() - start
    return {
        "baseline": baseline_name,
        "seed": seed,
        "git_commit": commit,
        "config": cfg,
        "dataset_manifest_hash": manifest_hash(cfg["data"]),
        "wall_clock_seconds": round(wall_clock_s, 3),
        "metrics": eval_metrics,
    }


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/phase1_synthetic.yaml"
    cfg = yaml.safe_load(Path(config_path).read_text())
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    commit = git_commit()

    d = cfg["data"]
    train_centers, test_centers = make_synthetic_train_test_centers(
        samples_per_center=d["samples_per_center"], image_size=d["image_size"], seed=d["data_seed"]
    )

    for baseline_name in cfg["baselines"]:
        for seed in cfg["seeds"]:
            result = run_one(baseline_name, seed, cfg, train_centers, test_centers, commit)
            out_path = out_dir / f"{baseline_name}_seed{seed}.json"
            out_path.write_text(json.dumps(result, indent=2))
            m = result["metrics"]["pooled"]
            print(
                f"{baseline_name:12s} seed={seed} "
                f"coarse_f1={m['coarse_macro_f1']:.3f} fine_f1={m['fine_macro_f1']:.3f} "
                f"({result['wall_clock_seconds']:.1f}s) -> {out_path}"
            )


if __name__ == "__main__":
    main()
