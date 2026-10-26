"""Reconstruct completed v11 incumbents in the manuscript's physical units.

This verifier never imports or calls the optimizer. It checks saved candidate
designs, not unsaved heads or a globally optimal solution.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    root = parser.parse_args().project.resolve()
    source = root/'results/overnight_energy_v2/initialization_v11'
    out = root/'results/overnight_energy_v2/completed_transport_audit_v14'
    out.mkdir(exist_ok=True)
    manifest = json.loads((source/'manifest.json').read_text())
    status = json.loads((source/'status.json').read_text())
    assert status['status'] == 'completed_cpu_only_not_promoted'
    config = manifest['config']
    assert sha(root/'results/failed_round_diagnostic/updates.npz') == manifest['updates_sha256']
    assert sha(root/'tools/overnight_energy_v2/initialization_v11/wireless_init_v11.py') == manifest['variant_sha256']
    for name, expected in manifest['source_sha256'].items():
        assert sha(root/name) == expected
    with np.load(root/'results/failed_round_diagnostic/updates.npz') as data:
        learning, unlearning, channels = [data[k].copy() for k in ('learning', 'unlearning', 'channels')]
    dimension = learning.shape[1]
    block = config['block_dimension']
    def blocks(v):
        return np.pad(v, ((0, 0), (0, (-dimension) % block))).reshape(len(v), -1, block)
    sigma2 = 10**((config['noise_dbm']-30)/10)
    sigma = np.sqrt(sigma2)
    rows = json.loads((source/'results.json').read_text())
    certificates = []
    for row in rows:
        if row['status'] != 'feasible':
            continue
        mode = row['mode']
        name = f"b{row['bits']}_r{row['relative_tolerance']:g}_{mode}_f{row['analog_fraction']}_design.npz"
        path = source/name
        with np.load(path) as data:
            arrays = {k: data[k].copy() for k in data.files}
        head = int(arrays['head'])
        users = np.delete(np.arange(config['users']), head)
        np.testing.assert_array_equal(arrays['users'], users)
        analog_vectors = blocks(learning+unlearning if mode == 'mixed_aircomp' else learning)
        digital_vectors = blocks(np.zeros_like(unlearning) if mode == 'mixed_aircomp' else unlearning)
        rho = np.linalg.norm(analog_vectors[users], axis=-1)
        n = arrays['symbols']; nd = arrays['digital_symbols']; dims = arrays['dimensions']
        a = arrays['analog_rms']; t = arrays['digital_sqrt_power']; s = arrays['receiver']
        # Independently reproduce the prescribed integer codec and scale rounding.
        levels = 2**(row['bits']-1)-1
        maxima = np.max(np.abs(digital_vectors), axis=-1)
        scale = (maxima/levels).astype(np.float32).astype(np.float64)
        safe = np.where(scale > 0, scale, 1.)
        decoded = np.clip(np.rint(digital_vectors/safe[..., None]), -levels, levels)*scale[..., None]
        delta = np.sqrt(block)*(scale/2+np.maximum(maxima-levels*scale, 0))
        payload = np.where(maxima > 0, block*row['bits']+config['scale_metadata_bits'], 0)
        np.testing.assert_allclose(arrays['rho'], rho, rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(arrays['quantization_bound'], delta[users], rtol=1e-12, atol=1e-15)
        # Original complex phases are compensated; reconstruct positive magnitudes.
        b = np.divide(a*np.sqrt(n), rho, out=np.zeros_like(a), where=rho > 0)
        c = s/(sigma*np.sqrt(n))
        h = channels[users, head, None]*sigma
        p = t*t
        gain = c*h*b
        analog_block_j = b*b*rho*rho/config['symbol_rate_hz']
        digital_block_j = nd/config['symbol_rate_hz']*p
        if mode == 'orthogonal':
            bound_block_j = analog_block_j+digital_block_j
            peak = np.maximum(a*a, p)
            effective_chi = 0.
        else:
            bound_block_j = (np.sqrt(analog_block_j)+np.sqrt(digital_block_j))**2
            peak = (a+t)**2
            effective_chi = config['sic_residual_fraction']
        assert np.max(peak) <= config['power_limit_w']*(1+1e-5)
        overhead = np.full(config['users'], config['base_overhead_j'])
        overhead[head] += config['head_overhead_j']
        assert np.all(bound_block_j.sum(axis=1)+overhead[users] <= config['available_energy_j']*(1+1e-5))
        assert overhead[head] <= config['available_energy_j']
        energy = float(bound_block_j.sum()+overhead.sum())
        assert np.isclose(energy, row['energy_bound_j'], rtol=1e-8, atol=1e-12)
        min_rate_margin = None
        active_blocks = np.any(payload[users] > 0, axis=0)
        for m in range(len(n)):
            order = np.flatnonzero(payload[users, m] > 0)
            order = order[np.argsort(-h[order, 0], kind='stable')]
            for j, u in enumerate(order):
                analog_interference = 0. if mode == 'orthogonal' else float((h[:, 0]@a[:, m])**2)
                after = float(np.sum(h[order[j+1:], 0]**2*p[order[j+1:], m]))
                before = float(config['sic_residual_fraction']*np.sum(h[order[:j], 0]**2*p[order[:j], m]))
                sinr = float(h[u, 0]**2*p[u, m]/(sigma2+analog_interference+after+before))
                margin = float(nd[m]*np.log2(1+sinr)-payload[users[u], m])
                assert margin >= -max(1e-3, payload[users[u], m]*1e-5)
                min_rate_margin = margin if min_rate_margin is None else min(min_rate_margin, margin)
        residual = c*np.sqrt(n)*np.sum(np.sqrt(effective_chi)*h*t, axis=0)
        noise = np.abs(c)*sigma*np.sqrt(dims/2)
        bound = float(np.linalg.norm(np.sum(rho*np.abs(gain-1), axis=0)+noise+residual+delta[users].sum(axis=0)))
        assert np.isclose(bound, row['error_bound'], rtol=1e-7, atol=1e-12)
        assert bound <= row['epsilon']*(1+1e-5)
        assert effective_chi == 0., 'Exact MSE audit currently requires the recorded zero-SIC-residual protocol.'
        bias = ((gain-1)[..., None]*analog_vectors[users]).sum(axis=0)
        bias += (decoded[users]-digital_vectors[users]).sum(axis=0)
        bias = bias.reshape(-1)[:dimension]
        valid = np.minimum(block, np.maximum(dimension-np.arange(len(n))*block, 0))
        rmse = float(np.sqrt(bias@bias+np.sum(valid*sigma2*c*c/2)))
        assert rmse <= bound*(1+1e-5)+1e-12
        assert np.isclose(rmse, row['analytic_total_rmse'], rtol=1e-6, atol=1e-12)
        total_symbols = n.sum()+(nd[active_blocks].sum() if mode == 'orthogonal' else 0)
        assert total_symbols == block*len(n)*config['symbols_per_coordinate']
        assert total_symbols/config['symbol_rate_hz'] <= config['round_uplink_deadline_s']
        assert np.all(n >= dims)
        assert np.max(n)/config['symbol_rate_hz'] <= config['coherence_time_s']
        assert np.max(nd)/config['symbol_rate_hz'] <= config['coherence_time_s']
        history = next(x['history'] for x in row['heads'] if x['head'] == head and x['status'] == 'feasible')
        assert all(y <= x+max(1e-12, abs(x)*1e-6) for x, y in zip(history, history[1:]))
        assert row['selected_energy_bound_j'] <= row['energy_bound_j']+1e-12
        certificates.append(dict(bits=row['bits'], relative_tolerance=row['relative_tolerance'], mode=mode,
                                 analog_fraction=row['analog_fraction'], head=head, energy_bound_j=energy,
                                 rms_bound=bound, analytic_rmse=rmse, min_payload_margin_bits=min_rate_margin,
                                 candidate_design_sha256=sha(path)))
    result = dict(status='completed_saved_candidates_passed', at_kst=datetime.now().isoformat(timespec='seconds'),
                  observed_cases=len(rows), checked_saved_feasible_candidates=len(certificates),
                  selected_solver_counts={k: sum(x.get('selected_solver') == k for x in rows)
                                          for k in ['original_incumbent', 'joint_update']},
                  certificates=certificates, verifier_sha256=sha(__file__),
                  limits='Independent arithmetic reconstruction of saved v11 candidates, not new optimization, hardware measurement, task quality, or proof of global optimality. Original incumbent selection does not discard failed new candidates.')
    (out/'certificates.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'certificates'}))


if __name__ == '__main__':
    main()
