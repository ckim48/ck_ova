from dataclasses import dataclass

import cvxpy as cp
import numpy as np


@dataclass
class Design:
    head: int
    users: np.ndarray
    analog_rms: np.ndarray
    digital_sqrt_power: np.ndarray
    receiver: np.ndarray
    energy_j: float
    error_bound: float
    history: list
    normalized_channels: np.ndarray
    rho: np.ndarray
    quantization_bound: np.ndarray
    symbols: np.ndarray
    dimensions: np.ndarray
    chi: float
    mode: str
    digital_symbols: np.ndarray


def quantize(updates, bits=8, metadata_bits=32):
    if not 2 <= bits <= 16:
        raise ValueError("Quantization bits must be in [2, 16].")
    levels = 2 ** (bits - 1) - 1
    maxima = np.max(np.abs(updates), axis=-1)
    scale = (maxima / levels).astype(np.float32).astype(np.float64)
    safe_scale = np.where(scale > 0, scale, 1)
    rounded = np.clip(np.rint(updates / safe_scale[..., None]), -levels, levels)
    decoded = rounded * safe_scale[..., None]
    bound = np.sqrt(updates.shape[-1]) * (
        scale / 2 + np.maximum(maxima - levels * scale, 0)
    )
    payload = np.where(maxima > 0, updates.shape[-1] * bits + metadata_bits, 0)
    return decoded, bound, payload


def sic_system(gamma, chi):
    count = len(gamma)
    matrix = np.zeros((count, count))
    for user in range(count):
        matrix[user, user + 1:] = gamma[user]
        matrix[user, :user] = gamma[user] * chi
    spectral_radius = max(np.abs(np.linalg.eigvals(matrix)), default=0.0)
    if spectral_radius >= 1 - 1e-10:
        raise ValueError("SIC spectral radius is at least one.")
    return np.linalg.solve(np.eye(count) - matrix, gamma)


def receive_optimum(channels, analog, rho, noise):
    gain = channels * analog
    positive = gain > 1e-15
    locations = np.r_[0.0, rho[positive] / gain[positive]]
    weights = np.r_[noise, gain[positive]]
    ordering = np.argsort(locations)
    index = np.searchsorted(np.cumsum(weights[ordering]), weights.sum() / 2)
    return float(locations[ordering[min(index, len(ordering) - 1)]])


def block_bounds(channels, analog, digital, receiver, rho, delta, dims, symbols, chi):
    noise = np.sqrt(dims / (2 * symbols))
    noise += np.sum(np.sqrt(chi) * channels * digital, axis=0)
    return (
        np.abs(receiver * channels * analog - rho).sum(axis=0)
        + receiver * noise
        + delta.sum(axis=0)
    )


def optimize_head(head, channels, rho_all, delta_all, payload_all, dims, symbols,
                  epsilon, config, chi, mode="hybrid", digital_symbols=None):
    users = np.delete(np.arange(len(channels)), head)
    channel = channels[users, head, None] * np.ones((1, len(dims)))
    rho = rho_all[users]
    delta = delta_all[users]
    payload = payload_all[users]
    digital_symbols = np.array(symbols if digital_symbols is None else digital_symbols)
    if np.any(digital_symbols <= 0) or epsilon <= 0:
        raise ValueError("Positive digital symbol counts and epsilon are required.")
    active_orders = []
    multipliers = []
    for block in range(len(dims)):
        active = np.flatnonzero(payload[:, block] > 0)
        active = active[np.argsort(-channel[active, block], kind="stable")]
        active_orders.append(active)
        gamma = np.expm1(np.log(2) * payload[active, block] / digital_symbols[block])
        try:
            multipliers.append(sic_system(gamma, chi))
        except ValueError:
            return None
    if np.linalg.norm(delta.sum(axis=0)) > epsilon:
        return None
    overhead = np.full(len(channels), config["base_overhead_j"])
    overhead[head] += config["head_overhead_j"]
    if np.any(overhead > config["available_energy_j"]):
        return None
    duration = symbols / config["symbol_rate_hz"]
    digital_duration = digital_symbols / config["symbol_rate_hz"]
    if mode == "orthogonal":
        duration_total = duration.sum() + digital_duration[np.any(payload > 0, axis=0)].sum()
    else:
        duration_total = duration.sum()
    if duration_total > config["round_uplink_deadline_s"]:
        return None
    if np.any(duration > config["coherence_time_s"]) or np.any(symbols < dims):
        return None
    if np.any(digital_duration > config["coherence_time_s"]):
        return None

    def min_digital(analog):
        power = np.zeros_like(rho)
        for block, active in enumerate(active_orders):
            interference = 0 if mode == "orthogonal" else (
                channel[:, block] @ analog[:, block]
            ) ** 2
            power[active, block] = (
                (1 + interference) * multipliers[block] / channel[active, block] ** 2
            )
        return np.sqrt(power)

    effective_chi = 0 if mode == "orthogonal" else chi

    def energy_blocks(analog, digital):
        if mode == "orthogonal":
            return duration * analog ** 2 + digital_duration * digital ** 2
        return duration * (analog + digital) ** 2

    def feasible(analog, digital, receiver):
        if not all(np.all(np.isfinite(value)) for value in (analog, digital, receiver)):
            return False
        if min(analog.min(), digital.min(), receiver.min()) < -1e-9:
            return False
        peak = np.maximum(analog ** 2, digital ** 2) if mode == "orthogonal" else (
            analog + digital
        ) ** 2
        if np.max(peak) > config["power_limit_w"] * (1 + 1e-6):
            return False
        if np.any(energy_blocks(analog, digital).sum(axis=1) + overhead[users]
                  > config["available_energy_j"] * (1 + 1e-6)):
            return False
        if np.any(digital + 1e-7 < min_digital(analog)):
            return False
        bound = block_bounds(channel, analog, digital, receiver, rho, delta,
                             dims, symbols, effective_chi)
        return np.linalg.norm(bound) <= epsilon * (1 + 1e-5)

    def conic_step(receiver, fixed_digital=None):
        analog = cp.Variable(rho.shape, nonneg=True)
        digital = cp.Variable(rho.shape, nonneg=True) if fixed_digital is None else cp.Constant(fixed_digital)
        constraints = [cp.multiply(rho == 0, analog) == 0]
        if mode == "orthogonal":
            energy = cp.multiply(duration, cp.square(analog))
            energy += cp.multiply(digital_duration, cp.square(digital))
            constraints += [analog <= np.sqrt(config["power_limit_w"]),
                            digital <= np.sqrt(config["power_limit_w"])]
        else:
            power = cp.square(analog + digital)
            energy = cp.multiply(duration, power)
            constraints.append(analog + digital <= np.sqrt(config["power_limit_w"]))
        constraints.append(cp.sum(energy, axis=1) + overhead[users]
                           <= config["available_energy_j"])
        bounds = []
        for block, active in enumerate(active_orders):
            if fixed_digital is None:
                inactive = np.setdiff1d(np.arange(len(users)), active)
                if len(inactive):
                    constraints.append(digital[inactive, block] == 0)
            analog_interference = cp.sum(cp.multiply(channel[:, block], analog[:, block]))
            for rank, user in enumerate(active):
                gamma = np.expm1(np.log(2) * payload[user, block] / digital_symbols[block])
                terms = [cp.Constant(1.0)]
                if mode != "orthogonal":
                    terms.append(analog_interference)
                for other_rank, other in enumerate(active):
                    if other == user:
                        continue
                    factor = np.sqrt(chi) if other_rank < rank else 1.0
                    terms.append(factor * channel[other, block] * digital[other, block])
                constraints.append(np.sqrt(gamma) * cp.norm(cp.hstack(terms))
                                   <= channel[user, block] * digital[user, block])
            noise = np.sqrt(dims[block] / (2 * symbols[block]))
            noise += cp.sum(cp.multiply(
                np.sqrt(effective_chi) * channel[:, block], digital[:, block]
            ))
            mismatch = cp.norm1(
                cp.multiply(receiver[block] * channel[:, block], analog[:, block])
                - rho[:, block]
            )
            bounds.append((mismatch + receiver[block] * noise + delta[:, block].sum()) / epsilon)
        constraints.append(cp.norm(cp.hstack(bounds)) <= 1)
        objective = cp.sum(energy) / duration.sum()
        problem = cp.Problem(cp.Minimize(objective), constraints)
        try:
            problem.solve(solver="CLARABEL", max_iter=150)
        except cp.error.SolverError:
            return None
        if problem.status not in ("optimal", "optimal_inaccurate") or analog.value is None:
            return None
        analog_value = np.maximum(analog.value, 0)
        digital_value = min_digital(analog_value)
        if not feasible(analog_value, digital_value, receiver):
            return None
        return analog_value, digital_value

    reference_receiver = np.max(
        rho / np.maximum(channel * np.sqrt(config["power_limit_w"]) * 0.3, 1e-15),
        axis=0,
    )
    reference_receiver = np.maximum(reference_receiver, 1e-10)
    def convex_initialization():
        signal = cp.Variable(rho.shape, nonneg=True)
        digital_signal = cp.Variable(rho.shape, nonneg=True)
        scaled_receiver = cp.Variable(len(dims), nonneg=True)
        receiver_rows = cp.vstack([scaled_receiver] * len(users))
        constraints = [cp.multiply(rho == 0, signal) == 0]
        if mode == "orthogonal":
            constraints += [
                signal <= np.sqrt(config["power_limit_w"]) * receiver_rows,
                digital_signal <= np.sqrt(config["power_limit_w"]) * receiver_rows,
            ]
        else:
            constraints.append(signal + digital_signal
                               <= np.sqrt(config["power_limit_w"]) * receiver_rows)
        bounds = []
        for block, active in enumerate(active_orders):
            inactive = np.setdiff1d(np.arange(len(users)), active)
            if len(inactive):
                constraints.append(digital_signal[inactive, block] == 0)
            terms = [scaled_receiver[block]]
            if mode != "orthogonal":
                terms.append(cp.sum(cp.multiply(channel[:, block], signal[:, block])))
            for rank, user in enumerate(active):
                constraints.append(
                    np.sqrt(multipliers[block][rank]) * cp.norm(cp.hstack(terms))
                    <= channel[user, block] * digital_signal[user, block]
                )
            bounds.append(
                cp.norm1(cp.multiply(channel[:, block], signal[:, block]) - rho[:, block] / epsilon)
                + scaled_receiver[block] * np.sqrt(dims[block] / (2 * symbols[block]))
                + cp.sum(cp.multiply(np.sqrt(effective_chi) * channel[:, block],
                                     digital_signal[:, block]))
                + delta[:, block].sum() / epsilon
            )
        problem = cp.Problem(cp.Minimize(cp.norm(cp.hstack(bounds))), constraints)
        try:
            problem.solve(solver="CLARABEL", max_iter=300)
        except cp.error.SolverError:
            return None
        if problem.status not in ("optimal", "optimal_inaccurate") or signal.value is None:
            return None
        normalized_receiver = np.maximum(scaled_receiver.value, 0)
        analog = np.maximum(signal.value, 0) / np.maximum(normalized_receiver, 1e-15)
        digital = min_digital(analog)
        receiver = epsilon * normalized_receiver
        if not feasible(analog, digital, receiver):
            return None
        return analog, digital, receiver

    def improve(analog, digital, receiver):
        history = [float(energy_blocks(analog, digital).sum() + overhead.sum())]
        for iteration in range(config["max_ao_iterations"]):
            noise = np.sqrt(dims / (2 * symbols))
            noise += (np.sqrt(effective_chi) * channel * digital).sum(axis=0)
            receiver = np.array([
                receive_optimum(channel[:, block], analog[:, block], rho[:, block], noise[block])
                for block in range(len(dims))
            ])
            candidate = conic_step(receiver, digital)
            if candidate is None:
                break
            updated_analog, updated_digital = candidate
            energy = float(energy_blocks(updated_analog, updated_digital).sum() + overhead.sum())
            if energy > history[-1] + max(1e-12, history[-1] * 1e-6):
                break
            analog, digital = updated_analog, updated_digital
            history.append(energy)
            if abs(history[-2] - energy) <= max(1e-12, history[-2] * 1e-5):
                break
        if not feasible(analog, digital, receiver):
            return None
        energy = float(energy_blocks(analog, digital).sum() + overhead.sum())
        bound = float(np.linalg.norm(block_bounds(
            channel, analog, digital, receiver, rho, delta, dims, symbols, effective_chi
        )))
        return Design(head, users, analog, digital, receiver, energy, bound,
                      history, channel, rho, delta, symbols, dims, effective_chi, mode,
                      digital_symbols)

    best = None
    for multiplier in np.geomspace(0.3, 6, config["receive_initializations"]):
        receiver = reference_receiver * multiplier
        initial = conic_step(receiver)
        if initial is None:
            continue
        candidate = improve(*initial, receiver)
        if candidate is not None and (best is None or candidate.energy_j < best.energy_j):
            best = candidate
    if best is None:
        initial = convex_initialization()
        if initial is not None:
            best = improve(*initial)
    return best


def empirical_error(design, learning, unlearning, decoded, rng, trials):
    errors = np.zeros(trials)
    expected_mse = 0.0
    for block, dimension in enumerate(design.dimensions):
        users = design.users
        rho = design.rho[:, block]
        gain = design.receiver[block] * design.normalized_channels[:, block]
        gain *= design.analog_rms[:, block] / np.where(rho > 0, rho, 1)
        mismatch = ((gain[:, None] - 1) * learning[users, block]).sum(axis=0)
        quantization_error = (decoded[users, block] - unlearning[users, block]).sum(axis=0)
        bias = mismatch + quantization_error
        residual_power = np.sum(
            design.chi * (design.normalized_channels[:, block]
                          * design.digital_sqrt_power[:, block]) ** 2
        )
        variance = design.receiver[block] ** 2 * (1 + residual_power) / (
            2 * design.symbols[block]
        )
        noise = rng.normal(0, np.sqrt(variance), (trials, dimension))
        errors += ((bias + noise) ** 2).sum(axis=1)
        expected_mse += float(bias @ bias + dimension * variance)
    standard_error = float(errors.std(ddof=1) / np.sqrt(trials))
    return float(errors.mean()), expected_mse, standard_error
