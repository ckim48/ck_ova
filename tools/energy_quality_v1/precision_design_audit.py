"""Joint-transceiver checks for lower digital precisions on saved updates."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    args = parser.parse_args()
    root = args.project.resolve(); sys.path.insert(0, str(root))
    from transport import design_transport, energy_accounting, recover
    source = root / "results/failed_round_diagnostic"
    metadata = json.loads((source / "metadata.json").read_text()); config = metadata["config"]
    with np.load(source / "updates.npz") as data:
        l, u, channel = [data[k].copy() for k in ("learning", "unlearning", "channels")]
    block = config["block_dimension"]
    learning_blocks = np.pad(l, ((0, 0), (0, (-l.shape[1]) % block))).reshape(len(l), -1, block)
    epsilon = max(config["absolute_tolerance_floor"], config["relative_tolerance"] * np.linalg.norm(np.linalg.norm(learning_blocks, axis=-1).sum(axis=0)))
    report = {"round": metadata["round"], "epsilon": float(epsilon),
        "updates_sha256": hashlib.sha256((source / "updates.npz").read_bytes()).hexdigest(),
        "source_sha256": {n: hashlib.sha256((root / n).read_bytes()).hexdigest() for n in ("wireless.py", "transport.py")},
        "interpretation": "Saved legacy-round CPU reconstruction, Hybrid only, same absolute error and time budget. Different quantizer precision is explicitly an experimental configuration. No training, task-loss or forgetting conclusion.", "rows": []}
    destination = root / "results/energy_quality_radio_v1/precision_design_audit.json"
    for bits in (4, 6, 8):
        began = time.monotonic()
        design, lb, ub, decoded, heads = design_transport(l, u, channel, dict(config, quantization_bits=bits), "hybrid", epsilon)
        row = {"bits": bits, "status": "no_feasible_design_found", "feasible_heads": heads}
        if design is not None:
            row.update(energy_accounting(design, config), status="feasible")
            restored_u = ub[design.head] + decoded[design.users].sum(axis=0)
            row["unlearning_component_quantization_error"] = float(np.linalg.norm(restored_u.reshape(-1)[:l.shape[1]] - u.sum(axis=0)))
            row["realized_total_update_error"] = float(np.linalg.norm(recover(design, lb, ub, decoded, np.random.default_rng(12345))[:l.shape[1]] - (l+u).sum(axis=0)))
        row["solve_seconds"] = time.monotonic() - began
        report["rows"].append(row)
        destination.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
