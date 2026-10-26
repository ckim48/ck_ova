import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from run_wireless import scenario
from transport import design_transport, energy_accounting


ROOT = Path(__file__).resolve().parent


def main():
    config = json.loads((ROOT / "configs/pilot.json").read_text())
    destination = ROOT / "results/equal_time"
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in config["seeds"]:
        channels, learning, unlearning, _, _, _, _, symbols, _ = scenario(config, seed)
        rho = np.linalg.norm(learning, axis=-1)
        learning = learning.reshape(config["users"], -1)
        unlearning = unlearning.reshape(config["users"], -1)
        for relative in config["relative_tolerances"]:
            epsilon = relative * np.linalg.norm(rho.sum(axis=0))
            for chi in config["sic_residual_fractions"]:
                config["sic_residual_fraction"] = chi
                for method in ("hybrid", "orthogonal_equal_time", "mixed_aircomp"):
                    design, _, _, _, feasible_heads = design_transport(
                        learning, unlearning, channels, config, method, epsilon
                    )
                    record = {
                        "seed": seed, "relative_tolerance": relative, "chi": chi,
                        "method": method, "epsilon": epsilon,
                        "status": "no_feasible_design_found", "feasible_heads": feasible_heads,
                        "energy_bound_j": None, "expected_energy_random_codewords_j": None,
                        "overhead_j": None, "transmit_energy_bound_j": None,
                        "uplink_s": None, "head": None, "error_bound": None,
                    }
                    if design is not None:
                        record.update(energy_accounting(design, config), status="feasible")
                        if not np.isclose(record["uplink_s"], symbols.sum() / config["symbol_rate_hz"]):
                            raise AssertionError("Total transmission time differs across methods.")
                    rows.append(record)
                print(f"seed={seed} tolerance={relative} chi={chi} completed", flush=True)
                with (destination / "metrics.csv").open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
    (destination / "manifest.json").write_text(json.dumps({
        "experiment": "equal-total-symbols synthetic communication comparison",
        "config": config,
        "orthogonal": "50/50 analog/digital time split in blocks with remote digital payload",
        "energy": "conservative bound and expected energy under independent zero-mean codewords",
        "not_a_real_clip_experiment": True,
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                          for name in ("wireless.py", "transport.py", "run_equal_time.py")},
    }, indent=2) + "\n")
    for method in ("hybrid", "orthogonal_equal_time", "mixed_aircomp"):
        selected = [row for row in rows if row["method"] == method]
        feasible = [row for row in selected if row["status"] == "feasible"]
        print(method, f"feasible={len(feasible)}/{len(selected)}", flush=True)


if __name__ == "__main__":
    main()
