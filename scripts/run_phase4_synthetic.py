#!/usr/bin/env python3
"""Phase 4 privacy mechanisms, synthetic tier (docs/execution_plan.md).
Compares four arms per seed: no protection (plain FedAvg), simulated
pairwise additive masking only, DP-SGD only (at several noise
multipliers), and simulated masking + DP-SGD combined. Reports utility and
(for DP arms) the achieved FULL-TRAINING epsilon, so a privacy-utility
Pareto curve can be drawn from the raw numbers.

REPAIR PASS 4 / P0-C FIX: earlier drafts of this script reported the max
of each ROUND's independent, freshly-reset per-client epsilon as if it
were the whole run's budget -- see medgate/privacy/dp_sgd.py's module
docstring for the mechanism and why that was wrong. This script now
creates one PrivacyEngine per client BEFORE the round loop and threads the
SAME `engines` dict through every round, so the epsilon reported after the
last round is the genuine cumulative, full-training, per-client-max,
RECORD-level (not client/hospital-level) epsilon at the configured delta.

Usage: PYTHONPATH=. python scripts/run_phase4_synthetic.py [config.yaml]
"""
import json
import sys
import time
from pathlib import Path

import opacus
import torch
import yaml

from medgate.data.synthetic import COARSE_CLASSES, FINE_CLASSES, make_synthetic_train_test_centers
from medgate.federated.fedavg import run_fedavg_round
from medgate.metrics import evaluate_both
from medgate.models.backbone import MedGateModel
from medgate.privacy.dp_sgd import dp_fedavg_round, secure_dp_fedavg_round
from medgate.privacy.secure_aggregation import secure_fedavg_round
from scripts.run_phase1_synthetic import git_commit, manifest_hash


def dp_accountant_metadata(dp: dict, t: dict, noise_multiplier, batch_size: int, samples_per_center: int) -> dict:
    """P0-C requirement #6: record accountant type, Opacus version,
    sampling assumptions and adjacency alongside every DP arm's result,
    not only documented in prose in medgate/privacy/dp_sgd.py."""
    sample_rate = batch_size / samples_per_center  # per-client-dataset sampling fraction per step, mini-batch (not Poisson) sampling
    return {
        "adjacency": "example-level (record-level), NOT client-level / hospital-level",
        "opacus_version": opacus.__version__,
        "accountant": "Opacus RDP accountant (opacus.accountants.rdp.RDPAccountant, the PrivacyEngine default)",
        "sampling_mechanism": "shuffled mini-batches, poisson_sampling=False (NOT Poisson subsampling -- "
                               "the accountant's numeric epsilon is therefore less tight than the Poisson-sampled "
                               "analysis DP-SGD literature usually reports)",
        "approx_sample_rate_per_step": sample_rate,
        "delta": dp["delta"],
        "max_grad_norm": dp["max_grad_norm"],
        "noise_multiplier": noise_multiplier,
        "local_epochs_per_round": t["epochs_per_round"],
        "federated_rounds": t["rounds"],
        "composition_scope": "CUMULATIVE across every federated round this client participated in (P0-C fix, "
                              "repair pass 4) -- the SAME PrivacyEngine per client is reused round over round via "
                              "the `engines` dict threaded through dp_fedavg_round/secure_dp_fedavg_round, so this "
                              "IS a full-training-run epsilon, not a single round's -- see "
                              "medgate/privacy/dp_sgd.py's module docstring for the mechanism and the bug it fixes",
        "epsilon_reported_is": "max over clients of each client's OWN cumulative epsilon after the LAST round -- "
                                "record-level DP for the worst-case (most-composed) client, not an average",
        "params_covered": "backbone, coarse_head, adapter, fine_head (medgate.attacks.gradient_inversion.attack_params)",
        "params_excluded": "adversary_head (architecturally unused by this training path -- receives no gradient at all)",
    }


def run_arm(arm_name: str, seed: int, cfg: dict, train_centers, test_pool, noise_multiplier=None) -> dict:
    torch.manual_seed(seed)
    model_kwargs = dict(num_coarse=len(COARSE_CLASSES), num_fine=len(FINE_CLASSES), **cfg["model"])
    model = MedGateModel(**model_kwargs)
    t = cfg["train"]
    dp = cfg["dp"]
    engines = None  # persists across the round loop below for dp_sgd/secure_agg_plus_dp -- see module docstring
    final_epsilon = None

    for _ in range(t["rounds"]):
        if arm_name == "no_protection":
            state = run_fedavg_round(model, train_centers, t["epochs_per_round"], t["batch_size"], t["lr"])
        elif arm_name == "secure_agg":
            state = secure_fedavg_round(model, train_centers, t["epochs_per_round"], t["batch_size"], t["lr"], seed)
        elif arm_name == "dp_sgd":
            state, final_epsilon, engines = dp_fedavg_round(
                model, train_centers, t["epochs_per_round"], t["batch_size"], t["lr"],
                noise_multiplier, dp["max_grad_norm"], dp["delta"], engines=engines,
            )
        elif arm_name == "secure_agg_plus_dp":
            state, final_epsilon, engines = secure_dp_fedavg_round(
                model, train_centers, t["epochs_per_round"], t["batch_size"], t["lr"],
                noise_multiplier, seed, dp["max_grad_norm"], dp["delta"], engines=engines,
            )
        else:
            raise ValueError(arm_name)
        model.load_state_dict(state)

    utility = evaluate_both(model, test_pool)
    result = {
        "arm": arm_name,
        "noise_multiplier": noise_multiplier,
        "seed": seed,
        # P0-C requirement #3: precise column name -- this is the FULL-TRAINING
        # (all rounds composed), record-level (not client-level), max-over-clients
        # epsilon at the configured delta. Never "epsilon" unqualified.
        "epsilon_full_training_max_per_client_record_level": final_epsilon,
        "utility": utility,
    }
    if arm_name in ("dp_sgd", "secure_agg_plus_dp"):
        result["dp_accountant_metadata"] = dp_accountant_metadata(
            dp, t, noise_multiplier, t["batch_size"], cfg["data"]["samples_per_center"]
        )
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
            # NOT "secure aggregation" unqualified -- P0-B: this is a
            # single-process simulated-pairwise-additive-masking heuristic
            # check, not a cryptographic security proof.
            "simulated_pairwise_masking_empirical_concealment_sanity_check": concealment,
        }
        out_path = out_dir / f"seed{seed}.json"
        out_path.write_text(json.dumps(out, indent=2))
        print(f"seed={seed} ({out['wall_clock_seconds']:.1f}s) -> {out_path}")
        for r in results:
            eps = r["epsilon_full_training_max_per_client_record_level"]
            eps_str = f"eps={eps:.2f}" if eps is not None else "eps=n/a"
            nm_str = f"nm={r['noise_multiplier']}" if r["noise_multiplier"] is not None else ""
            print(f"    {r['arm']:20s} {nm_str:8s} {eps_str:12s} fine_f1={r['utility']['fine_macro_f1']:.3f}")


if __name__ == "__main__":
    main()
