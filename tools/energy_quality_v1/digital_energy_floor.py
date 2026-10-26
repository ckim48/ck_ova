"""Necessary digital energy/power bounds for the fixed saved CLIP updates."""
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
    root = args.project.resolve()
    sys.path.insert(0, str(root))
    from wireless import quantize, sic_system
    source = root / "results/failed_round_diagnostic"
    output = root / "results/energy_quality_radio_v1"
    metadata = json.loads((source / "metadata.json").read_text())
    config = metadata["config"]
    with np.load(source / "updates.npz") as data:
        u, channels = data["unlearning"], data["channels"]
    block = config["block_dimension"]
    u = np.pad(u, ((0, 0), (0, (-u.shape[1]) % block))).reshape(len(u), -1, block)
    _, _, payloads = quantize(u, config["quantization_bits"], config["scale_metadata_bits"])
    symbols = np.full(u.shape[1], block * config["symbols_per_coordinate"])
    overhead = config["users"] * config["base_overhead_j"] + config["head_overhead_j"]
    report = {"round": metadata["round"], "updates_sha256": hashlib.sha256((source / "updates.npz").read_bytes()).hexdigest(),
              "assumptions": "Saved payloads, fixed SIC order, noise and channel model. Remove all analog interference and analog energy. Hence these are necessary lower bounds, not attainable designs. Phase separation uses the current fixed 50/50 split.",
              "power_limit_w": config["power_limit_w"], "methods": {}}
    for method in ("hybrid", "orthogonal_equal_time"):
        heads = []
        for head in range(config["users"]):
            users = np.delete(np.arange(config["users"]), head)
            payload = payloads[users]
            digital_symbols = symbols.copy()
            if method == "orthogonal_equal_time":
                digital_symbols[np.any(payload > 0, axis=0)] //= 2
            power = np.zeros_like(payload, dtype=float)
            for m in range(u.shape[1]):
                active = np.flatnonzero(payload[:, m] > 0)
                active = active[np.argsort(-channels[users[active], head], kind="stable")]
                gamma = np.expm1(np.log(2) * payload[active, m] / digital_symbols[m])
                multiplier = sic_system(gamma, config["sic_residual_fraction"])
                power[active, m] = multiplier / channels[users[active], head] ** 2
            user_energy = (power * (digital_symbols / config["symbol_rate_hz"])).sum(axis=1)
            passes = (np.max(power, initial=0) <= config["power_limit_w"] * (1 + 1e-6)
                      and np.all(user_energy + config["base_overhead_j"] <= config["available_energy_j"] * (1 + 1e-6)))
            heads.append({"head": head, "passes_necessary_power_and_energy": bool(passes),
                          "maximum_digital_power_floor_w": float(np.max(power, initial=0)),
                          "digital_energy_floor_j": float(user_energy.sum()),
                          "total_energy_floor_including_fixed_overhead_j": float(user_energy.sum() + overhead)})
        eligible = [h for h in heads if h["passes_necessary_power_and_energy"]]
        report["methods"][method] = {"heads": heads, "heads_passing_necessary_conditions": len(eligible),
            "minimum_necessary_energy_floor_j": min((h["total_energy_floor_including_fixed_overhead_j"] for h in eligible), default=None)}
    diagnostic = json.loads((source / "comparison.json").read_text())
    if diagnostic.get("solver_sha256") == hashlib.sha256((root / "wireless.py").read_bytes()).hexdigest():
        mixed = diagnostic["methods"]["mixed_aircomp"]
        if mixed["status"] == "feasible":
            bound = report["methods"]["hybrid"]["minimum_necessary_energy_floor_j"]
            report["saved_mixed_feasible_energy_bound_j"] = mixed["energy_bound_j"]
            report["hybrid_necessary_floor_over_saved_mixed_energy"] = bound / mixed["energy_bound_j"] if bound is not None else None
    output.mkdir(parents=True, exist_ok=True)
    (output / "digital_energy_floor.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"hybrid_floor_j": report["methods"]["hybrid"]["minimum_necessary_energy_floor_j"],
                      "phase_separated_eligible_heads": report["methods"]["orthogonal_equal_time"]["heads_passing_necessary_conditions"],
                      "floor_over_saved_mixed": report.get("hybrid_necessary_floor_over_saved_mixed_energy")}), flush=True)


if __name__ == "__main__":
    main()
