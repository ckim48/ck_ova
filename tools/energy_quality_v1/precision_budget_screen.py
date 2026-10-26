"""Screen digital precision against energy lower bounds and quantizer bounds."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    args = parser.parse_args()
    root = args.project.resolve(); sys.path.insert(0, str(root))
    from wireless import quantize, sic_system
    source = root / "results/failed_round_diagnostic"
    config = json.loads((source / "metadata.json").read_text())["config"]
    with np.load(source / "updates.npz") as data:
        l, u, channel = [data[k].copy() for k in ("learning", "unlearning", "channels")]
    b = config["block_dimension"]; dimension = l.shape[1]
    shape = (len(l), -1, b)
    l = np.pad(l, ((0, 0), (0, (-dimension) % b))).reshape(shape)
    u = np.pad(u, ((0, 0), (0, (-dimension) % b))).reshape(shape)
    epsilon = max(config["absolute_tolerance_floor"], config["relative_tolerance"] * np.linalg.norm(np.linalg.norm(l, axis=-1).sum(axis=0)))
    symbols = b * config["symbols_per_coordinate"]
    duration = symbols / config["symbol_rate_hz"]
    overhead = config["users"] * config["base_overhead_j"] + config["head_overhead_j"]
    rows = []
    for bits in (2, 4, 6, 8):
        decoded, delta, payload = quantize(u, bits, config["scale_metadata_bits"])
        heads = []
        for head in range(config["users"]):
            users = np.delete(np.arange(len(l)), head)
            power = np.zeros_like(payload[users], dtype=float)
            for m in range(l.shape[1]):
                active = np.flatnonzero(payload[users, m] > 0)
                active = active[np.argsort(-channel[users[active], head], kind="stable")]
                gamma = np.expm1(np.log(2) * payload[users[active], m] / symbols)
                power[active, m] = sic_system(gamma, config["sic_residual_fraction"]) / channel[users[active], head] ** 2
            floor = float(duration * power.sum() + overhead)
            quantization_bound = float(np.linalg.norm(delta[users].sum(axis=0)))
            actual_error = float(np.linalg.norm((decoded[users] - u[users]).sum(axis=0).reshape(-1)[:dimension]))
            heads.append({"head": head, "energy_floor_j": floor,
                          "quantization_only_bound_over_epsilon": quantization_bound / epsilon,
                          "unlearning_component_quantization_error": actual_error,
                          "passes_digital_power_necessary_condition": bool(np.max(power, initial=0) <= config["power_limit_w"] * (1 + 1e-6))})
        eligible = [h for h in heads if h["passes_digital_power_necessary_condition"] and h["quantization_only_bound_over_epsilon"] <= 1]
        rows.append({"bits": bits, "minimum_digital_energy_floor_j": min(h["energy_floor_j"] for h in heads),
                     "minimum_quantization_bound_over_epsilon": min(h["quantization_only_bound_over_epsilon"] for h in heads),
                     "heads_passing_both_screens": len(eligible),
                     "minimum_floor_among_screened_heads_j": min((h["energy_floor_j"] for h in eligible), default=None), "heads": heads})
    report = {"round": 91, "epsilon": float(epsilon), "relative_tolerance": config["relative_tolerance"],
              "updates_sha256": hashlib.sha256((source / "updates.npz").read_bytes()).hexdigest(),
              "interpretation": "Fixed-update Hybrid precision screen only. Bounds do not establish feasible joint transceiver designs or end-to-end forgetting quality. A quantization bound above epsilon excludes this conservative formulation, not necessarily the original MSE problem. No training configuration was changed.",
              "rows": rows}
    output = root / "results/energy_quality_radio_v1/precision_budget_screen.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    for r in rows:
        print(json.dumps({k: v for k, v in r.items() if k != "heads"}), flush=True)


if __name__ == "__main__":
    main()
