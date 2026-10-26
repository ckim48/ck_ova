"""Same-update radio energy/error audit using saved actual CLIP updates."""
import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tolerances", type=float, nargs="+", default=[0.1, 0.2, 0.4])
    args = parser.parse_args()
    root = args.project.resolve()
    sys.path.insert(0, str(root))
    from transport import design_transport, energy_accounting, recover
    source = args.input or root / "results/failed_round_diagnostic"
    output = args.output or root / "results/energy_quality_radio_v1"
    output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((source / "metadata.json").read_text())
    config = metadata["config"]
    with np.load(source / "updates.npz") as arrays:
        learning, unlearning, channels = [arrays[k].copy() for k in ("learning", "unlearning", "channels")]
    block = config["block_dimension"]
    padding = (-learning.shape[1]) % block
    blocked = np.pad(learning, ((0, 0), (0, padding))).reshape(len(learning), -1, block)
    scale = float(np.linalg.norm(np.linalg.norm(blocked, axis=-1).sum(axis=0)))
    ideal = (learning + unlearning).sum(axis=0)
    expected_time = blocked.shape[1] * block * config["symbols_per_coordinate"] / config["symbol_rate_hz"]
    initializations = [config["receive_initializations"], 12]
    manifest = {
        "experiment": "paired fixed-update radio audit; not end-to-end task evaluation",
        "round": metadata["round"], "source": str(source),
        "updates_sha256": hashlib.sha256((source / "updates.npz").read_bytes()).hexdigest(),
        "source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                          for name in ("wireless.py", "transport.py")},
        "audit_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "config": config, "tolerances": args.tolerances,
        "receive_initializations": initializations,
        "methods": ["hybrid", "orthogonal_equal_time", "mixed_aircomp"],
        "note": "All cases included, including failed searches. Same updates, absolute error tolerance and total transmission time. No VLM-loss or forgetting conclusion follows from this radio-only audit. The source is CPU reconstruction of a legacy GPU checkpoint, not current-trajectory replay.",
    }
    path = output / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Existing audit settings differ. Use a new output directory.")
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    rows = []
    if (output / "radio_audit.csv").exists():
        with (output / "radio_audit.csv").open() as handle:
            rows = list(csv.DictReader(handle))
    done = {(float(x["relative_tolerance"]), x["method"], int(x["receive_initializations"])) for x in rows}
    for relative in args.tolerances:
        epsilon = max(config["absolute_tolerance_floor"], relative * scale)
        for method in manifest["methods"]:
            for starts in initializations:
                if (relative, method, starts) in done:
                    continue
                candidate_config = dict(config, relative_tolerance=relative, receive_initializations=starts)
                began = time.monotonic()
                design, l, u, decoded, heads = design_transport(
                    learning, unlearning, channels, candidate_config, method, epsilon)
                row = {"relative_tolerance": relative, "method": method,
                       "receive_initializations": starts, "epsilon": epsilon,
                       "status": "no_feasible_design_found", "feasible_heads": heads,
                       "energy_bound_j": None, "expected_energy_j": None,
                       "uplink_s": None, "head": None, "error_bound": None,
                       "analytic_total_rmse": None, "realized_total_error": None,
                       "unlearning_component_error": None, "ao_energy_history": None,
                       "solve_seconds": None}
                if design is not None:
                    stats = energy_accounting(design, candidate_config)
                    if not np.isclose(stats["uplink_s"], expected_time):
                        raise ValueError("Transmission times differ across radio methods.")
                    # Actual mismatch/quantization bias and the implemented independent
                    # Gaussian post-SIC noise model; this is not a waveform measurement.
                    bias = l[design.head] + u[design.head]
                    variance = 0.0
                    for m in range(len(design.dimensions)):
                        rho = design.rho[:, m]
                        gain = design.receiver[m] * design.normalized_channels[:, m]
                        gain *= design.analog_rms[:, m] / np.where(rho > 0, rho, 1)
                        bias[m] += (gain[:, None] * l[design.users, m]).sum(axis=0)
                        bias[m] += decoded[design.users, m].sum(axis=0)
                        residual = np.sum(design.chi * (design.normalized_channels[:, m]
                                                       * design.digital_sqrt_power[:, m]) ** 2)
                        coordinates = max(0, min(block, learning.shape[1] - m * block))
                        variance += coordinates * design.receiver[m] ** 2 * (1 + residual) / (2 * design.symbols[m])
                    deterministic_error = bias.reshape(-1)[:len(ideal)] - ideal
                    rmse = float(np.sqrt(deterministic_error @ deterministic_error + variance))
                    # Only Hybrid/phase-separated links expose a distinct decoded U.
                    u_error = None
                    if method != "mixed_aircomp":
                        u_hat = u[design.head] + decoded[design.users].sum(axis=0)
                        u_error = float(np.linalg.norm(u_hat.reshape(-1)[:len(ideal)] - unlearning.sum(axis=0)))
                    recovered = recover(design, l, u, decoded, np.random.default_rng(12345))[:len(ideal)]
                    if rmse > design.error_bound * (1 + 1e-4) + 1e-12:
                        raise ValueError("Analytic RMSE exceeds the purported conservative error bound.")
                    row.update(status="feasible", energy_bound_j=stats["energy_bound_j"],
                               expected_energy_j=stats["expected_energy_random_codewords_j"],
                               uplink_s=stats["uplink_s"], head=stats["head"], error_bound=stats["error_bound"],
                               analytic_total_rmse=rmse, realized_total_error=float(np.linalg.norm(recovered - ideal)),
                               unlearning_component_error=u_error, ao_energy_history=json.dumps(design.history))
                row["solve_seconds"] = time.monotonic() - began
                rows.append(row)
                temporary = output / "radio_audit.partial.csv"
                with temporary.open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(row))
                    writer.writeheader(); writer.writerows(rows)
                temporary.replace(output / "radio_audit.csv")
                print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
