#!/usr/bin/env python3
"""P1 requirement #4 (repair pass 4): convergence curves + a larger-budget
full-fine-tune oracle, hierarchical fixture. The main sweep
(scripts/run_phase1_hierarchical.py) trains every method for a fixed,
small round/epoch budget chosen to fit this project's CPU-only hardware
across 10 methods x 5 seeds -- that budget was never checked for whether
ANY method has actually converged by the time training stops, so
"full_finetune (upper bound)" may understate what a properly-converged
model could reach. This script:
  1. trains coarse_pretrained_fedlora and full_finetune at a
     SUBSTANTIALLY larger round budget (single seed -- this is a
     diagnostic curve, not a statistical comparison across seeds),
     evaluating utility after EVERY round, to produce a convergence curve;
  2. reports full_finetune's LARGE-BUDGET final utility as the
     near-convergence oracle upper bound, to be read alongside (not
     instead of) the main sweep's matched-budget full_finetune number --
     the gap between them is itself the answer to "how much is the main
     sweep's upper bound understating true convergence."

Usage: PYTHONPATH=. python scripts/run_phase1_hierarchical_convergence.py [config.yaml]
"""
import json
import sys
import time
from pathlib import Path

import torch
import yaml

from medgate.data.hierarchical_synthetic import HierarchicalConfig, make_hierarchical_institutions, split_by_patient
from medgate.data.synthetic import COARSE_CLASSES, FINE_CLASSES
from medgate.federated.baselines import _freeze, _lora_trainable_params_fn, set_seed
from medgate.federated.fedavg import run_round
from medgate.federated.pretrain import build_coarse_pretrained_checkpoint
from medgate.metrics import evaluate_both
from medgate.models.backbone import MedGateModel
from scripts.run_phase1_synthetic import git_commit, manifest_hash


def train_with_curve(method: str, client_train_datasets, init_state_dict, model_kwargs, rounds, epochs, batch_size, lr, seed, test_pool):
    set_seed(seed)
    model = MedGateModel(**model_kwargs)
    model.load_state_dict(init_state_dict)
    trainable_fn = None
    if method == "coarse_pretrained_fedlora":
        _freeze(model.backbone)
        _freeze(model.coarse_head)
        trainable_fn = _lora_trainable_params_fn
    elif method == "full_finetune":
        pass  # every parameter trainable, no freezing
    else:
        raise ValueError(method)

    curve = [{"round": 0, **evaluate_both(model, test_pool)}]
    for r in range(1, rounds + 1):
        agg = run_round(model, client_train_datasets, epochs, batch_size, lr, trainable_params_fn=trainable_fn)
        model.load_state_dict(agg)
        curve.append({"round": r, **evaluate_both(model, test_pool)})
    return model, curve


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/phase1_hierarchical_convergence.yaml"
    cfg = yaml.safe_load(Path(config_path).read_text())
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    commit = git_commit()
    seed = cfg["seed"]
    model_kwargs = dict(num_coarse=len(COARSE_CLASSES), num_fine=len(FINE_CLASSES), **cfg["model"])
    d, pt, t = cfg["data"], cfg["pretrain"], cfg["train"]

    hcfg = HierarchicalConfig(
        image_size=d["image_size"], num_patients_per_institution=d["num_patients_per_institution"],
        observations_per_patient=d["observations_per_patient"], class_imbalance_strength=d["class_imbalance_strength"],
        sensitive_property_correlation=d["sensitive_property_correlation"],
    )
    insts = make_hierarchical_institutions(hcfg, seed=seed)
    train, _val, test = split_by_patient(insts, train_frac=d["train_frac"], val_frac=d["val_frac"], seed=seed)
    train_pool, test_pool = torch.utils.data.ConcatDataset(train), torch.utils.data.ConcatDataset(test)

    start = time.time()
    coarse_ckpt = build_coarse_pretrained_checkpoint(train_pool, model_kwargs, pt["epochs"], pt["batch_size"], pt["lr"], seed)
    coarse_state = coarse_ckpt.state_dict()

    curves = {}
    for method in ("coarse_pretrained_fedlora", "full_finetune"):
        _model, curve = train_with_curve(
            method, train, coarse_state, model_kwargs, t["rounds"], t["epochs_per_round"], t["batch_size"], t["lr"], seed, test_pool
        )
        curves[method] = curve
        print(f"{method}: round0 fine_f1={curve[0]['fine_macro_f1']:.3f} -> "
              f"round{t['rounds']} fine_f1={curve[-1]['fine_macro_f1']:.3f} "
              f"(coarse {curve[0]['coarse_macro_f1']:.3f} -> {curve[-1]['coarse_macro_f1']:.3f})")

    out = {
        "seed": seed, "git_commit": commit, "config": cfg,
        "dataset_manifest_hash": manifest_hash({"fixture": "hierarchical_convergence", **d}),
        "wall_clock_seconds": round(time.time() - start, 2),
        "curves": curves,
        "large_budget_full_finetune_oracle_fine_macro_f1": curves["full_finetune"][-1]["fine_macro_f1"],
        "large_budget_full_finetune_oracle_coarse_macro_f1": curves["full_finetune"][-1]["coarse_macro_f1"],
        "rounds": t["rounds"],
    }
    out_path = out_dir / f"seed{seed}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"({out['wall_clock_seconds']:.1f}s) -> {out_path}")


if __name__ == "__main__":
    main()
