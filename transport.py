import numpy as np

from wireless import block_bounds, optimize_head, quantize


def link_matrix(users, seed, config):
    generator = np.random.default_rng(seed)
    positions = generator.uniform(0, config["area_m"], (users, 2))
    distances = np.linalg.norm(positions[:, None] - positions[None], axis=-1)
    noise = 10 ** ((config["noise_dbm"] - 30) / 10)
    reference = 10 ** (config["reference_gain_db"] / 10)
    channels = np.sqrt(reference / noise) / np.maximum(distances, 1)
    np.fill_diagonal(channels, 0)
    return channels


def design_transport(learning, unlearning, channels, config, method, epsilon):
    users, dimension = learning.shape
    block_dimension = config["block_dimension"]
    padding = (-dimension) % block_dimension
    learning_blocks = np.pad(learning, ((0, 0), (0, padding))).reshape(users, -1, block_dimension)
    unlearning_blocks = np.pad(unlearning, ((0, 0), (0, padding))).reshape(users, -1, block_dimension)
    if method == "mixed_aircomp":
        learning_blocks = learning_blocks + unlearning_blocks
        unlearning_blocks = np.zeros_like(unlearning_blocks)
    decoded, delta, payload = quantize(
        unlearning_blocks, config["quantization_bits"], config["scale_metadata_bits"]
    )
    rho = np.linalg.norm(learning_blocks, axis=-1)
    dims = np.full(learning_blocks.shape[1], block_dimension)
    total_symbols = dims * config["symbols_per_coordinate"]
    best = None
    feasible_heads = 0
    heads = [0] if method == "fixed_head" else range(users)
    for head in heads:
        analog_symbols = total_symbols.copy()
        digital_symbols = total_symbols.copy()
        mode = "orthogonal" if method == "orthogonal_equal_time" else "hybrid"
        if method == "mixed_aircomp":
            mode = method
        if mode == "orthogonal":
            remote_users = np.delete(np.arange(users), head)
            active = np.any(payload[remote_users] > 0, axis=0)
            analog_symbols[active] = total_symbols[active] // 2
            digital_symbols[active] = total_symbols[active] - analog_symbols[active]
        design = optimize_head(
            head, channels, rho, delta, payload, dims, analog_symbols, epsilon, config,
            config["sic_residual_fraction"], mode, digital_symbols=digital_symbols,
        )
        if design is not None:
            feasible_heads += 1
            if best is None or design.energy_j < best.energy_j:
                best = design
    return best, learning_blocks, unlearning_blocks, decoded, feasible_heads


def recover(design, learning, unlearning, decoded, generator):
    result = learning[design.head] + unlearning[design.head]
    for block in range(len(design.dimensions)):
        users = design.users
        rho = design.rho[:, block]
        gain = design.receiver[block] * design.normalized_channels[:, block]
        gain *= design.analog_rms[:, block] / np.where(rho > 0, rho, 1)
        result[block] += (gain[:, None] * learning[users, block]).sum(axis=0)
        result[block] += decoded[users, block].sum(axis=0)
        residual = np.sum(design.chi * (
            design.normalized_channels[:, block] * design.digital_sqrt_power[:, block]
        ) ** 2)
        variance = design.receiver[block] ** 2 * (1 + residual) / (2 * design.symbols[block])
        result[block] += generator.normal(0, np.sqrt(variance), result.shape[1])
    return result.reshape(-1)


def energy_accounting(design, config):
    overhead = config["users"] * config["base_overhead_j"] + config["head_overhead_j"]
    analog = (design.symbols / config["symbol_rate_hz"] * design.analog_rms ** 2).sum()
    digital = (design.digital_symbols / config["symbol_rate_hz"]
               * design.digital_sqrt_power ** 2).sum()
    uplink = design.symbols.sum()
    if design.mode == "orthogonal":
        active = np.any(design.digital_sqrt_power > 0, axis=0)
        uplink += design.digital_symbols[active].sum()
    return {
        "energy_bound_j": design.energy_j,
        "expected_energy_random_codewords_j": float(analog + digital + overhead),
        "overhead_j": overhead,
        "transmit_energy_bound_j": design.energy_j - overhead,
        "uplink_s": float(uplink / config["symbol_rate_hz"]),
        "head": design.head,
        "error_bound": design.error_bound,
    }


def communicate(learning, unlearning, channels, config, method, seed):
    ideal = (learning + unlearning).sum(axis=0)
    if method == "ideal":
        return ideal, {"status": "ideal", "energy_bound_j": 0.0,
                       "expected_energy_random_codewords_j": 0.0,
                       "overhead_j": 0.0, "transmit_energy_bound_j": 0.0,
                       "uplink_s": 0.0, "error_bound": 0.0, "realized_error": 0.0,
                       "head": None}
    block_size = config["block_dimension"]
    padding = (-learning.shape[1]) % block_size
    padded = np.pad(learning, ((0, 0), (0, padding))).reshape(len(learning), -1, block_size)
    epsilon = config["relative_tolerance"] * np.linalg.norm(
        np.linalg.norm(padded, axis=-1).sum(axis=0)
    )
    epsilon = max(float(epsilon), config.get("absolute_tolerance_floor", 1e-8))
    design, learning_blocks, unlearning_blocks, decoded, feasible_heads = design_transport(
        learning, unlearning, channels, config, method, epsilon
    )
    if design is None:
        raise RuntimeError("No feasible design found; model not updated and deletion not acknowledged.")
    recovered = recover(design, learning_blocks, unlearning_blocks, decoded,
                        np.random.default_rng(seed))[:len(ideal)]
    statistics = energy_accounting(design, config)
    statistics.update(status="feasible", epsilon=epsilon, feasible_heads=feasible_heads,
                      realized_error=float(np.linalg.norm(recovered - ideal)))
    return recovered, statistics
