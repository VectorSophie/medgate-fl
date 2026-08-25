#!/usr/bin/env python3
"""Phase 1+2 fair-baseline repair, hierarchical-signal fixture
(docs/execution_plan.md). For each seed: build a coarse-pretrained
SmallBackbone checkpoint and an ImageNet-pretrained MobileNet checkpoint
(medgate/federated/pretrain.py, medgate/federated/checkpoints.py), then run
the FedLoRA family (random_frozen_lora as a negative control,
coarse_pretrained_fedlora, imagenet_pretrained_fedlora, full_finetune) and
all six Phase-2 capability-isolation methods from the SAME coarse
checkpoint, each evaluated with the full probe suite (RFC/UCG/ARR).

Usage: PYTHONPATH=. python scripts/run_phase1_hierarchical.py [config.yaml]
"""
import json
import sys
import time
from pathlib import Path

import torch
import yaml

from medgate.attacks.probes import output_only_probe, run_all_probes
from medgate.capability_metrics import authorized_recovery_ratio, residual_fine_capability, unauthorized_capability_gain
from medgate.data.hierarchical_synthetic import HierarchicalConfig, make_hierarchical_institutions, split_by_patient
from medgate.data.synthetic import COARSE_CLASSES, FINE_CLASSES
from medgate.federated.baselines import (
    train_full_finetune,
    train_pretrained_fedlora,
    train_random_frozen_lora,
)
from medgate.federated.capability_isolation import train_capability_isolation
from medgate.federated.checkpoints import save_checkpoint
from medgate.federated.pretrain import build_coarse_pretrained_checkpoint, build_imagenet_pretrained_checkpoint
from medgate.metrics import evaluate_both
from medgate.models.backbone import PretrainedMobileNetBackbone
from scripts.run_phase1_synthetic import git_commit, manifest_hash


def build_fixture(cfg, seed):
    d = cfg["data"]
    hcfg = HierarchicalConfig(
        image_size=d["image_size"],
        num_patients_per_institution=d["num_patients_per_institution"],
        observations_per_patient=d["observations_per_patient"],
        class_imbalance_strength=d["class_imbalance_strength"],
        sensitive_property_correlation=d["sensitive_property_correlation"],
    )
    insts = make_hierarchical_institutions(hcfg, seed=seed)
    train, _val, test = split_by_patient(insts, train_frac=d["train_frac"], val_frac=d["val_frac"], seed=seed)
    return train, test, torch.utils.data.ConcatDataset(train), torch.utils.data.ConcatDataset(test)


def eval_and_leak(model, train_pool, test_pool, seed) -> dict:
    utility = evaluate_both(model, test_pool)
    u_public = output_only_probe(model, train_pool, test_pool, seed=seed)["macro_f1"]
    probes = run_all_probes(model, train_pool, test_pool, seed=seed)
    rfc = residual_fine_capability(probes)
    return {
        "utility": utility,
        "u_public": u_public,
        "probes": probes,
        "residual_fine_capability": rfc,
        "unauthorized_capability_gain": unauthorized_capability_gain(rfc, u_public),
    }


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/phase1_hierarchical.yaml"
    cfg = yaml.safe_load(Path(config_path).read_text())
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    commit = git_commit()
    model_kwargs = dict(num_coarse=len(COARSE_CLASSES), num_fine=len(FINE_CLASSES), **cfg["model"])
    pt, t = cfg["pretrain"], cfg["train"]

    for seed in cfg["seeds"]:
        seed_start = time.time()
        train, test, train_pool, test_pool = build_fixture(cfg, seed)

        coarse_ckpt = build_coarse_pretrained_checkpoint(train_pool, model_kwargs, pt["epochs"], pt["batch_size"], pt["lr"], seed)
        coarse_state = coarse_ckpt.state_dict()
        coarse_meta = save_checkpoint(coarse_ckpt, f"hier_coarse_pretrained_seed{seed}", {
            "kind": "coarse_pretrained", "seed": seed, "config": pt, "git_commit": commit,
            "pretrain_coarse_macro_f1": evaluate_both(coarse_ckpt, test_pool)["coarse_macro_f1"],
        })

        imagenet_ckpt = build_imagenet_pretrained_checkpoint(train_pool, model_kwargs, pt["epochs"], pt["batch_size"], pt["lr"], seed)
        imagenet_state = imagenet_ckpt.state_dict()
        imagenet_meta = save_checkpoint(imagenet_ckpt, f"hier_imagenet_pretrained_seed{seed}", {
            "kind": "imagenet_pretrained", "seed": seed, "config": pt, "git_commit": commit,
            "pretrain_coarse_macro_f1": evaluate_both(imagenet_ckpt, test_pool)["coarse_macro_f1"],
        })

        results = []

        model, summary = train_random_frozen_lora(train, model_kwargs, t["rounds"], t["epochs_per_round"], t["batch_size"], t["lr"], seed)
        results.append({"method": "random_frozen_lora", "kind": "baseline", "param_summary": summary, **eval_and_leak(model, train_pool, test_pool, seed)})

        model, summary = train_pretrained_fedlora(train, coarse_state, model_kwargs, t["rounds"], t["epochs_per_round"], t["batch_size"], t["lr"], seed)
        results.append({"method": "coarse_pretrained_fedlora", "kind": "baseline", "param_summary": summary,
                         "checkpoint_sha256": coarse_meta["state_dict_sha256"], **eval_and_leak(model, train_pool, test_pool, seed)})

        model, summary = train_pretrained_fedlora(
            train, imagenet_state, model_kwargs, t["rounds"], t["epochs_per_round"], t["batch_size"], t["lr"], seed,
            backbone=PretrainedMobileNetBackbone(feature_dim=model_kwargs["feature_dim"], freeze=True),
        )
        results.append({"method": "imagenet_pretrained_fedlora", "kind": "baseline", "param_summary": summary,
                         "checkpoint_sha256": imagenet_meta["state_dict_sha256"], **eval_and_leak(model, train_pool, test_pool, seed)})

        model, summary = train_full_finetune(train, coarse_state, model_kwargs, t["rounds"], t["epochs_per_round"], t["batch_size"], t["lr"], seed)
        results.append({"method": "full_finetune", "kind": "baseline", "param_summary": summary,
                         "checkpoint_sha256": coarse_meta["state_dict_sha256"], **eval_and_leak(model, train_pool, test_pool, seed)})

        for method_name in cfg["capability_isolation_methods"]:
            model = train_capability_isolation(
                method_name, train, model_kwargs, t["rounds"], t["epochs_per_round"], t["batch_size"], t["lr"], seed,
                init_state_dict=coarse_state,
            )
            results.append({"method": method_name, "kind": "capability_isolation",
                             "checkpoint_sha256": coarse_meta["state_dict_sha256"], **eval_and_leak(model, train_pool, test_pool, seed)})

        # ARR needs the plain-adapter reference: use coarse_pretrained_fedlora's fine utility as U_plain_adapter
        plain_adapter_fine_f1 = next(r["utility"]["fine_macro_f1"] for r in results if r["method"] == "coarse_pretrained_fedlora")
        for r in results:
            r["authorized_recovery_ratio"] = authorized_recovery_ratio(r["utility"]["fine_macro_f1"], r["u_public"], plain_adapter_fine_f1)

        out = {
            "seed": seed, "git_commit": commit, "config": cfg,
            "dataset_manifest_hash": manifest_hash({"fixture": "hierarchical", **cfg["data"]}),
            "wall_clock_seconds": round(time.time() - seed_start, 2),
            "results": results,
        }
        out_path = out_dir / f"seed{seed}.json"
        out_path.write_text(json.dumps(out, indent=2))
        print(f"seed={seed} ({out['wall_clock_seconds']:.1f}s) -> {out_path}")
        for r in results:
            print(f"    {r['method']:28s} coarse_f1={r['utility']['coarse_macro_f1']:.3f} fine_f1={r['utility']['fine_macro_f1']:.3f} "
                  f"u_public={r['u_public']:.3f} rfc={r['residual_fine_capability']:.3f} arr={r['authorized_recovery_ratio']}")


if __name__ == "__main__":
    main()
