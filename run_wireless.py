import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import cvxpy
import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from wireless import empirical_error, optimize_head, quantize


ROOT = Path(__file__).resolve().parent


def scenario(config, seed):
    rng = np.random.default_rng(seed)
    count = config["users"]
    dimension = config["dimension"]
    block_size = config["block_dimension"]
    if dimension % block_size:
        raise ValueError("Pilot dimension must be divisible by block_dimension.")
    positions = np.column_stack((
        rng.uniform(0, config["area_m"], (count, 2)),
        np.full(count, config["altitude_m"]),
    ))
    distances = np.linalg.norm(positions[:, None] - positions[None], axis=-1)
    noise_w = 10 ** ((config["noise_dbm"] - 30) / 10)
    reference_gain = 10 ** (config["reference_gain_db"] / 10)
    channels = np.sqrt(reference_gain / noise_w) / np.maximum(distances, 1)
    np.fill_diagonal(channels, 0)
    blocks = dimension // block_size
    learning = rng.normal(0, 0.002 / np.sqrt(dimension), (count, blocks, block_size))
    unlearning = np.zeros_like(learning)
    active = rng.choice(count, config["active_unlearning_users"], replace=False)
    unlearning[active] = rng.normal(0, 0.001 / np.sqrt(dimension),
                                   (len(active), blocks, block_size))
    decoded, delta, payload = quantize(
        unlearning, config["quantization_bits"], config["scale_metadata_bits"]
    )
    dims = np.full(blocks, block_size)
    symbols = dims * config["symbols_per_coordinate"]
    return channels, learning, unlearning, decoded, delta, payload, dims, symbols, positions


def execute(config, destination):
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    traces = []
    started = time.monotonic()
    for seed in config["seeds"]:
        channels, learning, unlearning, decoded, delta, payload, dims, symbols, positions = scenario(
            config, seed
        )
        np.savez_compressed(destination / f"inputs_seed{seed}.npz", channels=channels,
                            learning=learning, unlearning=unlearning, positions=positions,
                            decoded=decoded, delta=delta, payload=payload)
        rho = np.linalg.norm(learning, axis=-1)
        scale = float(np.linalg.norm(rho.sum(axis=0)))
        for relative in config["relative_tolerances"]:
            epsilon = relative * scale
            for chi in config["sic_residual_fractions"]:
                selected = {}
                for mode in ("hybrid", "orthogonal", "mixed_aircomp"):
                    if mode == "mixed_aircomp":
                        transmitted_learning = learning + unlearning
                        transmitted_unlearning = np.zeros_like(unlearning)
                        reconstructed = transmitted_unlearning
                        mode_rho = np.linalg.norm(transmitted_learning, axis=-1)
                        mode_delta = np.zeros_like(delta)
                        mode_payload = np.zeros_like(payload)
                    else:
                        transmitted_learning = learning
                        transmitted_unlearning = unlearning
                        reconstructed = decoded
                        mode_rho, mode_delta, mode_payload = rho, delta, payload
                    candidates = []
                    for head in range(config["users"]):
                        design = optimize_head(head, channels, mode_rho, mode_delta, mode_payload,
                                               dims, symbols, epsilon, config, chi, mode)
                        if design is not None:
                            candidates.append(design)
                            traces.append({
                                "seed": seed, "relative_tolerance": relative, "chi": chi,
                                "mode": mode, "head": head, "energy_history_j": design.history,
                            })
                            if mode == "hybrid" and head == 0:
                                selected["fixed_head"] = (
                                    design, learning, unlearning, decoded, 1
                                )
                    if candidates:
                        design = min(candidates, key=lambda item: item.energy_j)
                        selected[mode] = (design, transmitted_learning, transmitted_unlearning,
                                          reconstructed, len(candidates))
                for mode in ("hybrid", "fixed_head", "orthogonal", "mixed_aircomp"):
                    row = {
                        "seed": seed, "relative_tolerance": relative, "chi": chi,
                        "method": mode, "epsilon": epsilon,
                        "status": "no_feasible_design_found",
                        "head": None, "feasible_heads": 0, "energy_bound_j": None,
                        "uplink_s": None, "error_bound": None, "empirical_rmse": None,
                        "analytic_rmse": None, "mse_standard_error": None,
                    }
                    if mode in selected:
                        design, transmitted_learning, transmitted_unlearning, reconstructed, count = selected[mode]
                        measured, analytic, standard_error = empirical_error(
                            design, transmitted_learning, transmitted_unlearning, reconstructed,
                            np.random.default_rng(seed + 10000),
                            config["monte_carlo_trials"],
                        )
                        duration = symbols.sum() / config["symbol_rate_hz"]
                        if mode == "orthogonal":
                            duration += symbols[np.any(payload[design.users] > 0, axis=0)].sum() / config[
                                "symbol_rate_hz"
                            ]
                        row.update(
                            status="feasible", head=design.head, feasible_heads=count,
                            energy_bound_j=design.energy_j, uplink_s=duration,
                            error_bound=design.error_bound, empirical_rmse=float(np.sqrt(measured)),
                            analytic_rmse=float(np.sqrt(analytic)), mse_standard_error=standard_error,
                        )
                        if analytic > design.error_bound ** 2 * (1 + 1e-7):
                            raise AssertionError("Analytic MSE exceeds conservative bound.")
                        if abs(measured - analytic) > 6 * standard_error + 1e-15:
                            raise AssertionError("Monte Carlo MSE disagrees with analytic MSE.")
                        np.savez_compressed(
                            destination / f"design_s{seed}_e{relative}_c{chi}_{mode}.npz",
                            head=design.head, users=design.users, analog_rms=design.analog_rms,
                            digital_sqrt_power=design.digital_sqrt_power, receiver=design.receiver,
                        )
                    rows.append(row)
                print(f"seed={seed} tolerance={relative} chi={chi}: "
                      f"{', '.join(selected)}; elapsed={time.monotonic() - started:.1f}s", flush=True)
                write_csv(rows, destination / "metrics.csv")
    provenance = {
        "experiment": "synthetic_update_wireless_pilot",
        "not_a_clip_or_paper_baseline_reproduction": True,
        "config": config,
        "python": sys.version,
        "host": platform.node(),
        "cvxpy": cvxpy.__version__,
        "numpy": np.__version__,
        "elapsed_s": time.monotonic() - started,
        "source_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in ("wireless.py", "run_wireless.py")
        },
    }
    (destination / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (destination / "convergence.json").write_text(json.dumps(traces, indent=2) + "\n")
    plot(rows, destination)
    return rows


def write_csv(rows, path):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(rows, destination):
    methods = ("hybrid", "fixed_head", "orthogonal", "mixed_aircomp")
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.6), layout="constrained")
    for method in methods:
        records = [row for row in rows if row["method"] == method]
        feasible = [row for row in records if row["status"] == "feasible"]
        axes[0].bar(method, len(feasible) / len(records))
        if feasible:
            energy = np.array([row["energy_bound_j"] for row in feasible]) * 1e3
            axes[1].bar(method, energy.mean(), yerr=energy.std())
            axes[2].scatter([row["error_bound"] for row in feasible],
                            [row["empirical_rmse"] for row in feasible], label=method, s=20)
    for axis in axes[:2]:
        axis.tick_params(axis="x", rotation=25)
    axes[0].set_ylabel("Feasible design fraction")
    axes[1].set_ylabel("Energy bound (mJ), feasible cases only")
    axes[2].set_xlabel("Conservative RMSE bound")
    axes[2].set_ylabel("Monte Carlo RMSE")
    axes[2].legend(fontsize=7)
    figure.suptitle("Synthetic communication pilot — not CLIP results")
    figure.savefig(destination / "wireless_pilot.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/pilot.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/wireless")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    settings = json.loads(args.config.read_text())
    if args.smoke:
        settings.update(seeds=[0], users=4, dimension=32, relative_tolerances=[0.5],
                        sic_residual_fractions=[0.0], max_ao_iterations=3,
                        receive_initializations=3)
    execute(settings, args.output)
