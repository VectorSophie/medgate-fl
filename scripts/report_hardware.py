#!/usr/bin/env python3
"""Capture a machine-readable hardware/environment snapshot.

ponytail: stdlib + one optional torch probe, no dependency added just to
read CPU/RAM/disk. Run this at the start of every phase so results are
traceable to the hardware that produced them (reproducibility requirement).
"""
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone


def _cmd(args):
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=10).stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def gpu_info():
    out = _cmd(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
    if out:
        return out.splitlines()
    return []


def mem_info():
    try:
        with open("/proc/meminfo") as f:
            lines = {l.split(":")[0]: l.split(":")[1].strip() for l in f if ":" in l}
        return {k: lines[k] for k in ("MemTotal", "MemAvailable", "SwapTotal") if k in lines}
    except OSError:
        return {}


def torch_info():
    try:
        import torch
        return {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "mps_available": getattr(torch.backends, "mps", None) and torch.backends.mps.is_available(),
        }
    except ImportError:
        return {"torch_version": None}


def main():
    report = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "cpu_count_logical": os.cpu_count(),
        "gpus": gpu_info(),
        "mem": mem_info(),
        "disk_free_gb": round(shutil.disk_usage(".").free / 2**30, 1),
        "disk_total_gb": round(shutil.disk_usage(".").total / 2**30, 1),
        "torch": torch_info(),
        "git_commit": _cmd(["git", "rev-parse", "HEAD"]) or "uncommitted",
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
