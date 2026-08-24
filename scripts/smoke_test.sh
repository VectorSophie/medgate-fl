#!/usr/bin/env bash
# One-command Phase 0 smoke test: synthetic-fixture forward/backward pass +
# two-client FedAvg round. Must pass before any Phase 1+ run starts
# (docs/execution_plan.md).
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -d .venv ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

echo "== hardware snapshot =="
python scripts/report_hardware.py

echo "== Phase 0 tests =="
PYTHONPATH=. pytest tests/ -v
