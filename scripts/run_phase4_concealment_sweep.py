#!/usr/bin/env python3
"""P0-B requirement: sweep mask scale and check a nonlinear attacker, as
an actual experiment producing a committed result (not only a unit-test
assertion) -- scripts/run_phase4_synthetic.py's single mask_scale=1.0,
logistic-only concealment check stays as a smoke-check embedded in that
script's own JSON; THIS script is the dedicated sweep the paper's
secure-aggregation subsection actually cites. Cheap: no model training,
pure synthetic Gaussian data (medgate.privacy.secure_aggregation
.empirical_concealment_sanity_check), runs in well under a minute.

Usage: PYTHONPATH=. python scripts/run_phase4_concealment_sweep.py [config.yaml]
"""
import json
import sys
import time
from pathlib import Path

import yaml

from medgate.privacy.secure_aggregation import empirical_concealment_sanity_check
from scripts.run_phase1_synthetic import git_commit


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/phase4_concealment_sweep.yaml"
    cfg = yaml.safe_load(Path(config_path).read_text())
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    commit = git_commit()

    start = time.time()
    rows = []
    for mask_scale in cfg["mask_scales"]:
        for attacker in cfg["attackers"]:
            for seed in cfg["seeds"]:
                r = empirical_concealment_sanity_check(
                    shape=tuple(cfg["shape"]), seed=seed, n_samples=cfg["n_samples"],
                    mean_shift=cfg["mean_shift"], n_bootstrap=cfg["n_bootstrap"],
                    mask_scale=mask_scale, attacker=attacker,
                )
                r["seed"] = seed
                rows.append(r)
                print(f"mask_scale={mask_scale:6.2f} attacker={attacker:9s} seed={seed} "
                      f"masked_auc={r['masked_case_attack_auc']:.3f} unmasked_auc={r['unmasked_control_attack_auc']:.3f}")

    out = {
        "git_commit": commit, "config": cfg,
        "wall_clock_seconds": round(time.time() - start, 2),
        "rows": rows,
    }
    out_path = out_dir / "sweep.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"({out['wall_clock_seconds']:.1f}s) -> {out_path}")


if __name__ == "__main__":
    main()
