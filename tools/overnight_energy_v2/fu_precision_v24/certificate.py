"""Independent physical arithmetic for uncompressed FU61 precision designs."""
import numpy as np


def check(learning, unlearning, channels, config, bits, mode, epsilon, design, row):
    assert config['sic_residual_fraction'] == float(design['chi']) == 0.
    width = config['block_dimension']; dimension = learning.shape[1]
    block = lambda x: np.pad(x, ((0, 0), (0, (-dimension) % width))).reshape(len(x), -1, width)
    head = int(design['head']); users = np.delete(np.arange(len(learning)), head)
    np.testing.assert_array_equal(users, design['users'])
    analog, digital = block(learning), block(unlearning)
    rho = np.linalg.norm(analog[users], axis=-1)
    levels = 2**(bits-1)-1; maxima = np.max(np.abs(digital), axis=-1)
    scale = (maxima/levels).astype(np.float32).astype(float)
    decoded = np.clip(np.rint(digital/np.where(scale > 0, scale, 1.)[..., None]), -levels, levels)*scale[..., None]
    delta = np.sqrt(width)*(scale/2+np.maximum(maxima-levels*scale, 0))
    payload = np.where(maxima > 0, width*bits+config['scale_metadata_bits'], 0)
    np.testing.assert_allclose(design['rho'], rho, rtol=1e-11, atol=1e-15)
    np.testing.assert_allclose(design['quantization_bound'], delta[users], rtol=1e-12, atol=1e-15)
    na, nd = design['symbols'], design['digital_symbols']; a, s = design['analog_rms'], design['receiver']
    p = design['digital_sqrt_power']**2
    sigma2 = 10**((config['noise_dbm']-30)/10); sigma = np.sqrt(sigma2)
    h = channels[users, head, None]*sigma
    b = np.divide(a*np.sqrt(na), rho, out=np.zeros_like(a), where=rho > 0)
    c = s/(sigma*np.sqrt(na)); gain = c*h*b
    ea = b*b*rho*rho/config['symbol_rate_hz']; ed = nd*p/config['symbol_rate_hz']
    tx = ea+ed if mode == 'phase_grid' else (np.sqrt(ea)+np.sqrt(ed))**2
    peak = np.maximum(a*a, p) if mode == 'phase_grid' else (a+np.sqrt(p))**2
    assert np.max(peak) <= config['power_limit_w']*(1+1e-5)
    overhead = np.full(len(learning), config['base_overhead_j']); overhead[head] += config['head_overhead_j']
    assert np.all(tx.sum(axis=1)+overhead[users] <= config['available_energy_j']*(1+1e-5))
    assert overhead[head] <= config['available_energy_j']
    upper = float(tx.sum()+overhead.sum()); expected = float(ea.sum()+ed.sum()+overhead.sum())
    np.testing.assert_allclose(upper, row['energy_bound_j'], rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(expected, row['expected_energy_random_codewords_j'], rtol=1e-8, atol=1e-12)
    minimum_margin = None
    for m in range(len(na)):
        order = np.flatnonzero(payload[users, m] > 0)
        order = order[np.argsort(-h[order, 0], kind='stable')]
        for j, u in enumerate(order):
            analog_interference = 0. if mode == 'phase_grid' else float((h[:, 0]@a[:, m])**2)
            after = float(np.sum(h[order[j+1:], 0]**2*p[order[j+1:], m]))
            sinr = h[u, 0]**2*p[u, m]/(sigma2+analog_interference+after)
            margin = float(nd[m]*np.log2(1+sinr)-payload[users[u], m])
            assert margin >= -max(1e-3, payload[users[u], m]*1e-5)
            minimum_margin = margin if minimum_margin is None else min(minimum_margin, margin)
    bound = float(np.linalg.norm((rho*np.abs(gain-1)).sum(axis=0)+c*sigma*np.sqrt(width/2)+delta[users].sum(axis=0)))
    np.testing.assert_allclose(bound, row['error_bound'], rtol=1e-7, atol=1e-12)
    assert bound <= epsilon*(1+1e-5)
    bias = (((gain-1)[..., None]*analog[users]).sum(axis=0)+(decoded[users]-digital[users]).sum(axis=0)).reshape(-1)[:dimension]
    valid = np.minimum(width, np.maximum(dimension-np.arange(len(na))*width, 0))
    rmse = float(np.sqrt(bias@bias+np.sum(valid*sigma2*c*c/2)))
    np.testing.assert_allclose(rmse, row['analytic_total_rmse'], rtol=1e-6, atol=1e-12)
    assert rmse <= bound*(1+1e-5)
    active = np.any(payload[users] > 0, axis=0)
    symbols = na.sum()+(nd[active].sum() if mode == 'phase_grid' else 0)
    assert int(symbols) == width*len(na)*config['symbols_per_coordinate']
    assert symbols/config['symbol_rate_hz'] <= config['round_uplink_deadline_s']
    assert np.all(na >= width)
    assert max(na.max(), nd.max())/config['symbol_rate_hz'] <= config['coherence_time_s']
    return dict(head=head, energy_upper_j=upper, expected_energy_j=expected, root_error_bound=bound,
                analytic_rmse=rmse, minimum_payload_margin_bits=minimum_margin, uplink_s=float(symbols/config['symbol_rate_hz']))
