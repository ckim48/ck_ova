import json
from pathlib import Path

import cvxpy as cp
import numpy as np

from wireless import block_bounds, quantize, sic_system


ROOT = Path(__file__).resolve().parent / "results/failed_round_diagnostic"


def main():
    metadata = json.loads((ROOT / "metadata.json").read_text())
    config = metadata["config"]
    arrays = np.load(ROOT / "updates.npz")
    learning, unlearning, channels = (arrays[key] for key in ("learning", "unlearning", "channels"))
    dimension = config["block_dimension"]
    padding = (-learning.shape[1]) % dimension
    shape = (len(learning), -1, dimension)
    rho_all = np.linalg.norm(np.pad(learning, ((0, 0), (0, padding))).reshape(shape), axis=-1)
    _, delta_all, payload_all = quantize(
        np.pad(unlearning, ((0, 0), (0, padding))).reshape(shape),
        config["quantization_bits"], config["scale_metadata_bits"],
    )
    epsilon = max(config["absolute_tolerance_floor"],
                  config["relative_tolerance"] * np.linalg.norm(rho_all.sum(axis=0)))
    blocks = rho_all.shape[1]
    symbols = np.full(blocks, dimension * config["symbols_per_coordinate"])
    duration = symbols / config["symbol_rate_hz"]
    report = {
        "round": metadata["round"], "epsilon": float(epsilon),
        "note": "Numerical convex feasibility diagnostic for hybrid only, not an energy optimizer. "
                "Uses receiver-scaled powers; battery constraints omitted for relaxation. "
                "A feasible witness is separately checked in original variables.",
        "battery_constraints_redundant_from_peak": bool(
            duration.sum() * config["power_limit_w"] + config["base_overhead_j"]
            + config["head_overhead_j"] <= config["available_energy_j"]
        ),
        "heads": [],
    }
    for head in range(config["users"]):
        users = np.delete(np.arange(config["users"]), head)
        channel = np.broadcast_to(channels[users, head, None], (len(users), blocks))
        rho, delta, payload = rho_all[users], delta_all[users], payload_all[users]
        signal = cp.Variable(rho.shape, nonneg=True)
        digital = cp.Variable(rho.shape, nonneg=True)
        receiver = cp.Variable(blocks, nonneg=True)
        constraints = [
            signal + digital <= np.sqrt(config["power_limit_w"]) * cp.vstack(
                [receiver] * len(users)),
            cp.multiply(rho == 0, signal) == 0,
        ]
        bounds, orders, multipliers = [], [], []
        for block in range(blocks):
            active = np.flatnonzero(payload[:, block] > 0)
            active = active[np.argsort(-channel[active, block], kind="stable")]
            gamma = np.expm1(np.log(2) * payload[active, block] / symbols[block])
            multiplier = sic_system(gamma, config["sic_residual_fraction"])
            orders.append(active)
            multipliers.append(multiplier)
            inactive = np.setdiff1d(np.arange(len(users)), active)
            if len(inactive):
                constraints.append(digital[inactive, block] == 0)
            interference = cp.sum(cp.multiply(channel[:, block], signal[:, block]))
            for rank, user in enumerate(active):
                constraints.append(
                    np.sqrt(multiplier[rank]) * cp.norm(cp.hstack([receiver[block], interference]))
                    <= channel[user, block] * digital[user, block]
                )
            bounds.append(
                cp.norm1(cp.multiply(channel[:, block], signal[:, block]) - rho[:, block] / epsilon)
                + receiver[block] * np.sqrt(dimension / (2 * symbols[block]))
                + cp.sum(cp.multiply(np.sqrt(config["sic_residual_fraction"]) * channel[:, block],
                                     digital[:, block]))
                + delta[:, block].sum() / epsilon
            )
        problem = cp.Problem(cp.Minimize(cp.norm(cp.hstack(bounds))), constraints)
        try:
            problem.solve(solver="CLARABEL", max_iter=300)
            result = {"head": head, "solver_status": problem.status,
                      "minimum_relaxed_error_over_epsilon": problem.value,
                      "original_constraints_feasible_witness": False}
            if signal.value is not None and receiver.value is not None:
                normalized_receiver = np.maximum(receiver.value, 0)
                analog = np.maximum(signal.value, 0) / np.maximum(normalized_receiver, 1e-15)
                digital_power = np.zeros_like(analog)
                for block, active in enumerate(orders):
                    digital_power[active, block] = (
                        (1 + (channel[:, block] @ analog[:, block]) ** 2)
                        * multipliers[block] / channel[active, block] ** 2
                    )
                digital_amplitude = np.sqrt(digital_power)
                physical_receiver = epsilon * normalized_receiver
                bound = np.linalg.norm(block_bounds(
                    channel, analog, digital_amplitude, physical_receiver, rho, delta,
                    np.full(blocks, dimension), symbols, config["sic_residual_fraction"],
                ))
                peak = float(np.max((analog + digital_amplitude) ** 2))
                energy = (duration * (analog + digital_amplitude) ** 2).sum(axis=1)
                valid = (
                    bound <= epsilon * (1 + 1e-5)
                    and peak <= config["power_limit_w"] * (1 + 1e-6)
                    and np.all(energy + config["base_overhead_j"]
                               <= config["available_energy_j"] * (1 + 1e-6))
                    and config["base_overhead_j"] + config["head_overhead_j"]
                    <= config["available_energy_j"]
                    and duration.sum() <= config["round_uplink_deadline_s"]
                    and np.all(duration <= config["coherence_time_s"])
                    and np.all(symbols >= dimension)
                )
                result.update(original_constraints_feasible_witness=bool(valid),
                              witness_error_over_epsilon=float(bound / epsilon), witness_peak_w=peak)
        except cp.error.SolverError as error:
            result = {"head": head, "solver_status": "error", "error": str(error)}
        report["heads"].append(result)
        (ROOT / "convex_feasibility.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
