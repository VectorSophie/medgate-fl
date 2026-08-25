"""Versioned checkpoint save/load for the pretrained-baseline repair
(docs/execution_plan.md Phase 1): coarse_pretrained_fedlora,
imagenet_pretrained_fedlora, full_finetune, and proposed_isolation all
need to provably start from the SAME initial weights. Checkpoints live
under experiments/checkpoints/ (gitignored -- see .gitignore: they are
exactly reproducible by rerunning the generating script with the same
seed, so committing the binary is redundant; what's committed is each
result JSON's checkpoint_sha256 + regenerating command, per this
project's existing config/seed/commit/manifest-hash traceability pattern).
"""
import hashlib
import io
import json
from pathlib import Path

import torch

CHECKPOINT_DIR = Path("experiments/checkpoints")


def _state_dict_sha256(state_dict: dict) -> str:
    buf = io.BytesIO()
    torch.save(state_dict, buf)
    return hashlib.sha256(buf.getvalue()).hexdigest()


def save_checkpoint(model, name: str, metadata: dict) -> dict:
    """Saves model.state_dict() to experiments/checkpoints/{name}.pt plus
    a sidecar {name}.json with metadata + the state_dict's sha256 (so a
    caller can verify a loaded checkpoint is bit-identical to the one a
    result JSON claims it used). Returns the sidecar metadata dict."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    state_dict = model.state_dict()
    sha = _state_dict_sha256(state_dict)
    torch.save(state_dict, CHECKPOINT_DIR / f"{name}.pt")
    full_metadata = {**metadata, "checkpoint_name": name, "state_dict_sha256": sha}
    (CHECKPOINT_DIR / f"{name}.json").write_text(json.dumps(full_metadata, indent=2))
    return full_metadata


def load_checkpoint(name: str) -> tuple[dict, dict]:
    """Returns (state_dict, metadata). Raises FileNotFoundError with a
    clear message if the checkpoint was never generated (e.g. a config
    references it before its generating script has been run)."""
    pt_path = CHECKPOINT_DIR / f"{name}.pt"
    json_path = CHECKPOINT_DIR / f"{name}.json"
    if not pt_path.exists():
        raise FileNotFoundError(
            f"checkpoint '{name}' not found at {pt_path} -- run its generating "
            f"script first (see the checkpoint's expected producer in the config that references it)"
        )
    state_dict = torch.load(pt_path, weights_only=True)
    metadata = json.loads(json_path.read_text()) if json_path.exists() else {}
    if metadata.get("state_dict_sha256") and metadata["state_dict_sha256"] != _state_dict_sha256(state_dict):
        raise ValueError(f"checkpoint '{name}': on-disk state_dict does not match its recorded sha256 (corrupted or manually edited?)")
    return state_dict, metadata
