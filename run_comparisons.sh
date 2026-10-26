#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
exec "$PROJECT_ROOT/.venv/bin/python" -u "$PROJECT_ROOT/run_comparisons.py" "$@"
