"""Joint-power transport with a common, request-dependent RMS envelope."""
import numpy as np

from phase_split_grid import make_inputs, necessary_power, split_symbols
from transport import energy_accounting, recover
from wireless_joint_v3 import optimize_head


FRACTIONS = (.125, .25, .375, .5, .625, .75, .875)
MODES = ('hybrid', 'mixed_aircomp', 'phase_grid')


def design(learning, unlearning, channels, config, requested_mode):
    if requested_mode not in MODES:
        raise ValueError('Unknown experimental transport mode.')
    base_inputs = make_inputs(learning, unlearning, config)
    active_request = bool(np.any(unlearning != 0))
    relative = config['deletion_relative_tolerance'] if active_request else config['relative_tolerance']
    epsilon = max(config['absolute_tolerance_floor'],
                  float(relative*np.linalg.norm(base_inputs[5].sum(axis=0))))
    # A single canonical path makes all pre-request learning rounds identical.
    mode = requested_mode if active_request else 'hybrid'
    inputs = make_inputs(learning+unlearning, np.zeros_like(unlearning), config) if mode == 'mixed_aircomp' else base_inputs
    l, u, decoded, delta, payload, rho, dims, total_symbols = inputs
    best = None; best_fraction = None; diagnostics = []
    for fraction in FRACTIONS if mode == 'phase_grid' else (None,):
        local_best = None; feasible_heads = 0
        for head in range(config['users']):
            analog_symbols, digital_symbols = total_symbols.copy(), total_symbols.copy()
            physical_mode = 'orthogonal' if mode == 'phase_grid' else mode
            if mode == 'phase_grid':
                analog_symbols, digital_symbols, _ = split_symbols(total_symbols, payload, head, fraction)
            if np.any(analog_symbols < dims):
                continue
            floor = necessary_power(channels, payload, head, digital_symbols, config)
            if floor is None or np.max(floor, initial=0) > config['power_limit_w']*(1+1e-6):
                continue
            candidate = optimize_head(head, channels, rho, delta, payload, dims,
                analog_symbols, epsilon, config, config['sic_residual_fraction'], physical_mode,
                digital_symbols=digital_symbols)
            if candidate is not None:
                feasible_heads += 1
                if local_best is None or candidate.energy_j < local_best.energy_j:
                    local_best = candidate
        diagnostics.append({'analog_fraction': fraction, 'feasible_heads': feasible_heads,
            'energy_bound_j': local_best.energy_j if local_best is not None else None})
        if local_best is not None and (best is None or local_best.energy_j < best.energy_j):
            best = local_best; best_fraction = fraction
    metadata = {'requested_mode': requested_mode, 'selected_mode': mode,
        'active_unlearning_update': active_request, 'relative_tolerance_used': relative,
        'epsilon': epsilon, 'analog_fraction': best_fraction, 'fraction_search': diagnostics}
    return best, inputs, metadata


def communicate(learning, unlearning, channels, config, method, seed):
    selected, inputs, metadata = design(learning, unlearning, channels, config, method)
    if selected is None:
        raise RuntimeError('No feasible experimental design found; model not updated and deletion not acknowledged.')
    l, u, decoded = inputs[:3]
    ideal = (learning+unlearning).sum(axis=0)
    update = recover(selected, l, u, decoded, np.random.default_rng(seed))[:len(ideal)]
    stats = energy_accounting(selected, config)
    assert stats['error_bound'] <= metadata['epsilon']*(1+1e-5)
    assert np.isclose(stats['uplink_s'], inputs[-1].sum()/config['symbol_rate_hz'])
    stats.update(metadata, status='feasible', realized_error=float(np.linalg.norm(update-ideal)))
    return update, stats
