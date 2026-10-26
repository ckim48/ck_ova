"""Independent physical-unit audit of paid half-coefficient designs."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', type=Path, required=True)
    root = p.parse_args().project.resolve()
    out = root/'results/overnight_energy_v2/public_history_early_v17/radio'
    assert json.loads((out/'status.json').read_text())['status'].startswith('completed')
    plan = json.loads((out/'manifest.json').read_text()); cfg = plan['config']
    for name, expected in plan['code_sha256'].items():
        assert sha(root/'tools/overnight_energy_v2/public_history_early_v17'/name) == expected
    for name, expected in plan['core_sha256'].items():
        assert sha(root/name) == expected
    source = root/'results/overnight_energy_v2/public_history_early_v17/screen/vectors.npz'
    assert sha(source) == plan['parent_vectors_sha256']
    with np.load(source) as data:
        original = data['learning'].copy(); channels = data['channels'].copy()
        bases = {n[:-6]: data[n].copy() for n in data.files if n.endswith('_basis')}
    block = cfg['block_dimension']; dimension = original.shape[1]
    padded = lambda v: np.pad(v, ((0, 0), (0, (-dimension) % block))).reshape(len(v), -1, block)
    sigma2 = 10**((cfg['noise_dbm']-30)/10); sigma = np.sqrt(sigma2)
    certificates = []; codec_certificates = []
    for item in json.loads((out/'results.json').read_text()):
        name = item['variant']; rank = item['rank']
        vectors = original
        if rank:
            with np.load(out/(name+'_codec.npz')) as data:
                vectors = data['residual'].copy(); bits = data['coefficient_bits'].copy()
                decoded = data['decoded_coefficients'].copy(); scale = float(data['public_scale'])
            assert scale == plan['public_scale'] and bits.dtype == np.uint16
            np.testing.assert_array_equal(bits.view(np.float16).astype(float)*scale, decoded)
            assert bits.nbytes == cfg['users']*rank*2
            reconstructed = vectors+decoded@bases[name].T
            error = float(np.linalg.norm(reconstructed-original))
            assert error < 1e-12*np.linalg.norm(original)
            codec_certificates.append(dict(variant=name, exact_reconstruction_error=error, coefficient_bytes_per_client=rank*2))
        blocked = padded(vectors)
        rows = json.loads((out/(name+'_heads.json')).read_text())
        assert len(rows) == cfg['users']
        for row in rows:
            if row['status'] != 'feasible_radio_only':
                continue
            head = row['head']; users = np.delete(np.arange(cfg['users']), head)
            with np.load(out/f'{name}_h{head}.npz') as data:
                a, s, n, pc, ec = [data[k].copy() for k in ['analog_rms', 'receiver', 'data_symbols', 'control_power', 'control_energy']]
                np.testing.assert_array_equal(data['users'], users)
            rho = np.linalg.norm(blocked[users], axis=-1)
            b = np.divide(a*np.sqrt(n), rho, out=np.zeros_like(a), where=rho > 0)
            c = s/(sigma*np.sqrt(n)); h = channels[users, head, None]*sigma
            gain = c*h*b
            tx = b*b*rho*rho/cfg['symbol_rate_hz']
            power = tx*cfg['symbol_rate_hz']/n
            assert np.max(power) <= cfg['power_limit_w']*(1+1e-6)
            control_symbols = 32*rank
            assert row['coefficient_payload_bits'] == 16*rank*len(users)
            if rank:
                bits_delivered = control_symbols*np.log2(1+h[:, 0]**2*pc/sigma2)
                np.testing.assert_allclose(bits_delivered, 16*rank, rtol=1e-12)
                np.testing.assert_allclose(ec, control_symbols/cfg['symbol_rate_hz']*pc, rtol=1e-12)
            else:
                np.testing.assert_array_equal(ec, 0.)
            assert np.max(pc) <= cfg['power_limit_w']*(1+1e-6)
            overhead = np.full(cfg['users'], cfg['base_overhead_j']); overhead[head] += cfg['head_overhead_j']
            assert np.all(tx.sum(axis=1)+ec+overhead[users] <= cfg['available_energy_j']*(1+1e-6))
            joules = float(tx.sum()+ec.sum()+overhead.sum())
            assert np.isclose(joules, row['total_modeled_energy_j'], rtol=1e-8, atol=1e-12)
            assert int(n.sum())+control_symbols*len(users) == block*len(n)*cfg['symbols_per_coordinate']
            assert np.all(n >= block)
            assert np.max(n)/cfg['symbol_rate_hz'] <= cfg['coherence_time_s']
            assert control_symbols/cfg['symbol_rate_hz'] <= cfg['coherence_time_s']
            assert (n.sum()+control_symbols*len(users))/cfg['symbol_rate_hz'] <= cfg['round_uplink_deadline_s']
            bound = float(np.linalg.norm((rho*np.abs(gain-1)).sum(axis=0)+c*sigma*np.sqrt(block/2)))
            bias = ((gain-1)[..., None]*blocked[users]).sum(axis=0).reshape(-1)[:dimension]
            valid = np.minimum(block, np.maximum(dimension-np.arange(len(n))*block, 0))
            rmse = float(np.sqrt(bias@bias+np.sum(valid*sigma2*c*c/2)))
            assert bound <= plan['epsilon']*(1+1e-5)
            assert rmse <= bound*(1+1e-5)
            np.testing.assert_allclose([bound, rmse], [row['error_bound'], row['analytic_rmse']], rtol=1e-7, atol=1e-14)
            assert all(y <= x+max(1e-12, abs(x)*1e-6) for x, y in zip(row['history'], row['history'][1:]))
            certificates.append(dict(variant=name, head=head, status='passed_physical_units', energy_j=joules, rms_bound=bound, analytic_rmse=rmse))
    summary = json.loads((out/'summary.json').read_text())
    saving = next(x['rf_plus_original_overhead_saving_j'] for x in summary if x['variant'] == 'recent_public_span_rank3')
    minimum_extra_projection_flops = 4*cfg['users']*dimension*3
    result = dict(status='completed_independent_paid_codec_audit', at_kst=datetime.now().isoformat(timespec='seconds'),
                  checked_saved_radio_designs=len(certificates), certificates=certificates, codec_certificates=codec_certificates,
                  saved_network_rf_energy_j=saving, extra_projection_reconstruction_flops_lower_estimate=minimum_extra_projection_flops,
                  maximum_extra_energy_per_listed_flop_j=saving/minimum_extra_projection_flops,
                  verifier_sha256=sha(__file__), limits='Arithmetic audit of a single learning-only fixture, not hardware energy or application/unlearning proof. FLOP estimate counts only each clients projection/residual construction; excludes public basis generation, receiver, memory, coding and bookkeeping. Break-even is for extra NETWORK energy, not a hardware measurement.')
    (out/'physical_certificates.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ['certificates', 'codec_certificates']}))


if __name__ == '__main__':
    main()
