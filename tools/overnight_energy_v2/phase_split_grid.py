"""Fair fixed-total-time phase-separation grid; does not modify training code."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np


def make_inputs(learning, unlearning, config):
    from wireless import quantize
    block = config["block_dimension"]
    padding = (-learning.shape[1]) % block
    shape = (len(learning), -1, block)
    l = np.pad(learning, ((0, 0), (0, padding))).reshape(shape)
    u = np.pad(unlearning, ((0, 0), (0, padding))).reshape(shape)
    decoded, delta, payload = quantize(u, config["quantization_bits"], config["scale_metadata_bits"])
    dims = np.full(l.shape[1], block)
    total_symbols = dims * config["symbols_per_coordinate"]
    return l, u, decoded, delta, payload, np.linalg.norm(l, axis=-1), dims, total_symbols


def split_symbols(total, payload, head, fraction):
    if not 0 < fraction < 1:
        raise ValueError("Analog fraction must lie strictly between zero and one.")
    remote = np.delete(np.arange(len(payload)), head)
    active = np.any(payload[remote] > 0, axis=0)
    analog, digital = total.copy(), total.copy()
    analog[active] = np.floor(total[active] * fraction).astype(int)
    digital[active] = total[active] - analog[active]
    return analog, digital, active


def necessary_power(channels, payload, head, digital_symbols, config):
    from wireless import sic_system
    users = np.delete(np.arange(len(payload)), head)
    power = np.zeros_like(payload[users], dtype=float)
    for block in range(payload.shape[1]):
        active = np.flatnonzero(payload[users, block] > 0)
        active = active[np.argsort(-channels[users[active], head], kind="stable")]
        if not len(active):
            continue
        if np.any(channels[users[active], head] <= 0):
            return None
        gamma = np.expm1(np.log(2) * payload[users[active], block] / digital_symbols[block])
        try:
            multiplier = sic_system(gamma, config["sic_residual_fraction"])
        except ValueError:
            return None
        power[active, block] = multiplier / channels[users[active], head] ** 2
    return power


def optimize_fraction(learning, unlearning, channels, config, epsilon, fraction):
    from wireless import optimize_head
    inputs = make_inputs(learning, unlearning, config)
    l, u, decoded, delta, payload, rho, dims, total_symbols = inputs
    best = None; diagnostics = []
    for head in range(config["users"]):
        analog_symbols, digital_symbols, active = split_symbols(total_symbols, payload, head, fraction)
        item = {"head": head, "status": "not_attempted"}
        if np.any(analog_symbols < dims) or np.any(digital_symbols <= 0):
            item["status"] = "symbol_dimension_condition_failed"
            diagnostics.append(item); continue
        floor = necessary_power(channels, payload, head, digital_symbols, config)
        if floor is None or np.max(floor, initial=0) > config["power_limit_w"] * (1 + 1e-6):
            item["status"] = "necessary_digital_power_condition_failed"
            diagnostics.append(item); continue
        item["maximum_digital_power_floor_w"] = float(np.max(floor, initial=0))
        design = optimize_head(head, channels, rho, delta, payload, dims, analog_symbols,
            epsilon, config, config["sic_residual_fraction"], "orthogonal", digital_symbols=digital_symbols)
        item["status"] = "feasible" if design is not None else "no_feasible_design_found"
        if design is not None:
            item["energy_bound_j"] = design.energy_j
            if best is None or design.energy_j < best.energy_j:
                best = design
        diagnostics.append(item)
    return best, inputs, diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--bits", type=int, nargs="+", default=[8, 6])
    parser.add_argument("--analog-fractions", type=float, nargs="+", default=[0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875])
    parser.add_argument("--relative-tolerance", type=float, default=0.2)
    args = parser.parse_args()
    root = args.project.resolve(); sys.path.insert(0, str(root))
    from transport import energy_accounting
    source = root / "results/failed_round_diagnostic"
    metadata = json.loads((source / "metadata.json").read_text()); baseline = metadata["config"]
    with np.load(source / "updates.npz") as data:
        learning, unlearning, channels = [data[k].copy() for k in ("learning", "unlearning", "channels")]
    output = root / "results/overnight_energy_v2/phase_split_grid"
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"round": metadata["round"], "bits": args.bits, "analog_fractions": args.analog_fractions,
        "relative_tolerance": args.relative_tolerance,
        "source_sha256": {n: hashlib.sha256((root/n).read_bytes()).hexdigest() for n in ("wireless.py", "transport.py")},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "updates_sha256": hashlib.sha256((source/"updates.npz").read_bytes()).hexdigest(),
        "interpretation": "Exploratory fixed-update CPU diagnostic. Same channel, power, total time and absolute RMSE budget for each bit setting. Grid optimization of analog/digital phase time only, with digital NOMA retained. This is not a global optimum or full OMA/TDMA implementation. No VLM-loss or forgetting result."}
    f = output / "manifest.json"
    if f.exists() and json.loads(f.read_text()) != manifest:
        raise ValueError("Existing experiment differs; choose a new version.")
    f.write_text(json.dumps(manifest, indent=2) + "\n")
    results_path = output / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else []
    done = {(r["bits"], r["analog_fraction"]) for r in results}
    for bits in args.bits:
        config = dict(baseline, quantization_bits=bits)
        inputs = make_inputs(learning, unlearning, config)
        epsilon = max(config["absolute_tolerance_floor"], args.relative_tolerance * np.linalg.norm(inputs[5].sum(axis=0)))
        expected_time = inputs[-1].sum() / config["symbol_rate_hz"]
        for fraction in args.analog_fractions:
            if (bits, fraction) in done:
                continue
            began = time.monotonic()
            design, _, heads = optimize_fraction(learning, unlearning, channels, config, epsilon, fraction)
            row = {"bits": bits, "analog_fraction": fraction, "epsilon": float(epsilon),
                   "status": "no_feasible_design_found", "heads": heads}
            if design is not None:
                stats = energy_accounting(design, config)
                assert np.isclose(stats["uplink_s"], expected_time)
                assert stats["error_bound"] <= epsilon * (1+1e-5)
                row.update(stats, status="feasible")
            row["seconds"] = time.monotonic() - began
            results.append(row)
            temporary = results_path.with_suffix(".partial.json")
            temporary.write_text(json.dumps(results, indent=2) + "\n"); temporary.replace(results_path)
            print(json.dumps({k:v for k,v in row.items() if k != "heads"}), flush=True)
    best = []
    for bits in args.bits:
        feasible = [r for r in results if r["bits"] == bits and r["status"] == "feasible"]
        if feasible:
            r = min(feasible, key=lambda x:x["energy_bound_j"])
            best.append({k:v for k,v in r.items() if k != "heads"})
    (output / "best_observed.json").write_text(json.dumps(best, indent=2) + "\n")
    print("BEST_OBSERVED", json.dumps(best), flush=True)


if __name__ == "__main__":
    main()
