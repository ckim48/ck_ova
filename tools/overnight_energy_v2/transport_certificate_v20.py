"""Physical-unit certificate of available saved FU61 candidate designs.

Does not import optimizers, quantizers or metadata-energy helpers.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time
import numpy as np


def read(p):
    for attempt in range(10):
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            if attempt == 9:
                raise
            time.sleep(.02)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--label', required=True)
    args = parser.parse_args(); root = args.project.resolve()
    out = root/'results/overnight_energy_v2/transport_certificate_v20'; out.mkdir(exist_ok=True)
    target = out/(args.label+'.json'); assert not target.exists(), 'Preserve earlier certificate snapshots.'
    source = root/'results/overnight_energy_v2/public_history_fu_v19/screen'
    registered = read(source/'plan.json')
    for n, value in registered['core_sha256'].items():
        assert sha(root/n) == value
    for n, value in registered['code_sha256'].items():
        assert sha(root/'tools/overnight_energy_v2/public_history_fu_v19'/n) == value
    with np.load(source/'vectors.npz') as data:
        learning, unlearning, channels = [data[k].copy() for k in ['learning', 'unlearning', 'channels']]
        bases = {rank: data[f'rank{rank}_basis'].copy() for rank in [1, 3]}; public_scale = float(data['public_scale'])
    cfg = registered['config']; width = cfg['block_dimension']; dimension = learning.shape[1]
    def block(v):
        return np.pad(v, ((0, 0), (0, (-dimension) % width))).reshape(len(v), -1, width)
    base_rho = np.linalg.norm(block(learning), axis=-1)
    sigma2 = 10**((cfg['noise_dbm']-30)/10); sigma = np.sqrt(sigma2)
    certificates = []; statuses = []; errors = []
    for version in ['public_history_fu_v19', 'public_history_fu_v19b']:
        directory = root/'results/overnight_energy_v2'/version
        radio = directory/'radio' if version.endswith('v19') else directory
        if not (radio/'manifest.json').exists():
            statuses.append(dict(version=version, status='not_yet_started')); continue
        manifest = read(radio/'manifest.json')
        for name, expected in manifest['code_sha256'].items():
            assert sha(root/'tools/overnight_energy_v2'/version/name) == expected
        rows = read(radio/'results.json') if (radio/'results.json').exists() else []
        statuses.append(dict(version=version, observed_cases=len(rows), status=read(radio/'status.json')))
        for row in rows:
            if row['status'] != 'feasible_radio_only':
                continue
            rank = row['rank']; mode = row['mode']; fraction = row['analog_fraction']; relative = row['relative_tolerance']
            path = radio/f'r{relative:g}_rank{rank}_{mode}_f{fraction}_design.npz'
            with np.load(path) as data:
                d = {k: data[k].copy() for k in data.files}
            head = int(d['head']); users = np.delete(np.arange(cfg['users']), head)
            np.testing.assert_array_equal(d['users'], users)
            raw = learning+unlearning if mode == 'mixed_aircomp' else learning
            residual = raw
            if rank:
                decoded_coefficient = ((raw@bases[rank])/public_scale).astype(np.float16).astype(float)*public_scale
                residual = raw-decoded_coefficient@bases[rank].T
                np.testing.assert_allclose(residual+decoded_coefficient@bases[rank].T, raw, rtol=1e-11, atol=1e-15)
            analog = block(residual); digital = block(np.zeros_like(unlearning) if mode == 'mixed_aircomp' else unlearning)
            rho = np.linalg.norm(analog[users], axis=-1)
            levels = 2**(cfg['quantization_bits']-1)-1
            maxima = np.max(np.abs(digital), axis=-1)
            scale = (maxima/levels).astype(np.float32).astype(float)
            safe = np.where(scale > 0, scale, 1.)
            decoded = np.clip(np.rint(digital/safe[..., None]), -levels, levels)*scale[..., None]
            delta = np.sqrt(width)*(scale/2+np.maximum(maxima-levels*scale, 0))
            payload = np.where(maxima > 0, width*cfg['quantization_bits']+cfg['scale_metadata_bits'], 0)
            np.testing.assert_allclose(d['rho'], rho, rtol=1e-11, atol=1e-15)
            np.testing.assert_allclose(d['quantization_bound'], delta[users], rtol=1e-12, atol=1e-15)
            na = d['symbols']; nd = d['digital_symbols']; a = d['analog_rms']; s = d['receiver']; p = d['digital_sqrt_power']**2
            b = np.divide(a*np.sqrt(na), rho, out=np.zeros_like(a), where=rho > 0)
            c = s/(sigma*np.sqrt(na)); h = channels[users, head, None]*sigma
            gain = c*h*b
            ea = b*b*rho*rho/cfg['symbol_rate_hz']; ed = nd/cfg['symbol_rate_hz']*p
            if mode == 'phase_grid':
                tx = ea+ed; peak = np.maximum(a*a, p); effective_chi = 0.
            else:
                tx = (np.sqrt(ea)+np.sqrt(ed))**2; peak = (a+np.sqrt(p))**2; effective_chi = cfg['sic_residual_fraction']
            assert np.max(peak) <= cfg['power_limit_w']*(1+1e-5)
            pc = d['control_power']; ec = d['control_energy']; ncontrol = 32*rank
            if rank:
                np.testing.assert_allclose(ncontrol*np.log2(1+h[:, 0]**2*pc/sigma2), 16*rank, rtol=1e-12)
                np.testing.assert_allclose(ec, pc*ncontrol/cfg['symbol_rate_hz'], rtol=1e-12)
            else:
                np.testing.assert_array_equal(pc, 0.); np.testing.assert_array_equal(ec, 0.)
            assert np.max(pc) <= cfg['power_limit_w']*(1+1e-6)
            overhead = np.full(cfg['users'], cfg['base_overhead_j']); overhead[head] += cfg['head_overhead_j']
            assert np.all(tx.sum(axis=1)+ec+overhead[users] <= cfg['available_energy_j']*(1+1e-5))
            assert overhead[head] <= cfg['available_energy_j']
            joules = float(tx.sum()+ec.sum()+overhead.sum())
            assert np.isclose(joules, row['energy_bound_j'], rtol=1e-8, atol=1e-12)
            minimum_payload_margin = None
            for m in range(len(na)):
                order = np.flatnonzero(payload[users, m] > 0)
                order = order[np.argsort(-h[order, 0], kind='stable')]
                for j, u in enumerate(order):
                    ia = 0. if mode == 'phase_grid' else float((h[:, 0]@a[:, m])**2)
                    after = float(np.sum(h[order[j+1:], 0]**2*p[order[j+1:], m]))
                    before = float(cfg['sic_residual_fraction']*np.sum(h[order[:j], 0]**2*p[order[:j], m]))
                    sinr = h[u, 0]**2*p[u, m]/(sigma2+ia+after+before)
                    margin = float(nd[m]*np.log2(1+sinr)-payload[users[u], m])
                    assert margin >= -max(1e-3, payload[users[u], m]*1e-5)
                    minimum_payload_margin = margin if minimum_payload_margin is None else min(margin, minimum_payload_margin)
            noise = c*sigma*np.sqrt(width/2)
            sic = c*np.sqrt(na)*np.sum(np.sqrt(effective_chi)*h*np.sqrt(p), axis=0)
            bound = float(np.linalg.norm((rho*np.abs(gain-1)).sum(axis=0)+noise+sic+delta[users].sum(axis=0)))
            epsilon = max(cfg['absolute_tolerance_floor'], relative*np.linalg.norm(base_rho.sum(axis=0)))
            np.testing.assert_allclose([bound, epsilon], [row['error_bound'], row['epsilon']], rtol=1e-7, atol=1e-12)
            assert bound <= epsilon*(1+1e-5)
            assert effective_chi == 0., 'Recorded exact MSE certificate uses the zero residual protocol.'
            bias = (((gain-1)[..., None]*analog[users]).sum(axis=0)+(decoded[users]-digital[users]).sum(axis=0)).reshape(-1)[:dimension]
            valid = np.minimum(width, np.maximum(dimension-np.arange(len(na))*width, 0))
            rmse = float(np.sqrt(bias@bias+np.sum(valid*sigma2*c*c/2)))
            np.testing.assert_allclose(rmse, row['analytic_total_rmse'], rtol=1e-6, atol=1e-12)
            assert rmse <= bound*(1+1e-5)
            active = np.any(payload[users] > 0, axis=0)
            symbols = na.sum()+(nd[active].sum() if mode == 'phase_grid' else 0)+ncontrol*len(users)
            assert int(symbols) == width*len(na)*cfg['symbols_per_coordinate']
            assert symbols/cfg['symbol_rate_hz'] <= cfg['round_uplink_deadline_s']
            assert np.all(na >= width)
            assert max(na.max(), nd.max(), ncontrol)/cfg['symbol_rate_hz'] <= cfg['coherence_time_s']
            selected_head = next(x for x in row['heads'] if x['head'] == head)
            assert all(y <= x+max(1e-12, abs(x)*1e-6) for x, y in zip(selected_head['history'], selected_head['history'][1:]))
            certificates.append(dict(version=version, relative_tolerance=relative, rank=rank, mode=mode,
                analog_fraction=fraction, head=head, energy_j=joules, rms_bound=bound, analytic_rmse=rmse,
                minimum_payload_margin_bits=minimum_payload_margin, design_sha256=sha(path)))
    result = dict(status='all_observed_saved_feasible_candidates_passed', at_kst=datetime.now().isoformat(timespec='seconds'),
                  observed_statuses=statuses, checked_saved_candidates=len(certificates), certificates=certificates,
                  source_sha256=sha(__file__), limits='Arithmetic reconstruction of available best saved candidates, not all unsaved heads, global optimality, real hardware energy or task/deletion proof. Partial grids are not eligible for architecture ranking; include v19b minimum phase boundary.')
    write = json.dumps(result, indent=2)+'\n'; target.write_text(write)
    print(json.dumps({k: v for k, v in result.items() if k != 'certificates'}))


if __name__ == '__main__':
    main()
