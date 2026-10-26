#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"
export MPLCONFIGDIR="$PROJECT_ROOT/.cache/matplotlib"
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
mkdir -p "$MPLCONFIGDIR"
PYTHON="$PROJECT_ROOT/.venv/bin/python"
"$PYTHON" -c 'import torch; assert torch.cuda.is_available(), "CUDA is not accessible from this terminal"; print(torch.cuda.get_device_name(0)); print(torch.ones(1, device="cuda"))'
"$PYTHON" validate_real_clip.py --device cuda
"$PYTHON" run_clip_federated.py --device cuda --transport hybrid "$@"
