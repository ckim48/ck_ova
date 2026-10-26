"""Pre-registered earlier round30 validation of paid half coefficients."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from codec import encode_residual, reserve_symbols, control_resources


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', type=Path, required=True)
    p.add_argument('--plan-only', action='store_true')
    args = p.parse_args(); root = args.project.resolve()
    sys.path[:0] = [str(root), str(root/'tools/overnight_energy_v2/initialization_v11')]
    from wireless_init_v11 import optimize_head
    source = root/'results/overnight_energy_v2/public_history_early_v17/screen'
    report = json.loads((source/'results.json').read_text())
    assert report['round'] == 30 and report['status'] == 'completed_actual_early_cpu_replay'
    c = dict(report['config'], max_ao_iterations=30)
    core = json.loads((root/'results/comparisons_seed0_solver_v2/suite_metadata.json').read_text())['source_sha256']
    assert all(sha(root/n) == v for n, v in core.items())
    observer = root/'results/overnight_energy_v2/public_history_early_v17/observer'
    with np.load(source/'vectors.npz') as data:
        learning = data['learning'].copy(); channels = data['channels'].copy()
        public_scale = float(data['public_scale'])
        variants = [('unprojected_baseline', 0, learning, None, None)]
        for row in report['rows']:
            name = row['basis']; basis = data[name+'_basis'].copy()
            residual, bits, decoded = encode_residual(learning, basis, public_scale)
            error = float(np.linalg.norm(residual.sum(axis=0)+decoded.sum(axis=0)@basis.T-learning.sum(axis=0)))
            assert error < 1e-12*max(np.linalg.norm(learning), 1e-12)
            variants.append((name, row['rank'], residual, bits, decoded))
    block = c['block_dimension']; dimension = learning.shape[1]
    blocked = lambda v: np.pad(v, ((0, 0), (0, (-dimension) % block))).reshape(len(v), -1, block)
    original = blocked(learning); dims = np.full(original.shape[1], block)
    symbols = dims*c['symbols_per_coordinate']
    epsilon = max(c['absolute_tolerance_floor'], .2*np.linalg.norm(np.linalg.norm(original, axis=-1).sum(axis=0)))
    out = root/'results/overnight_energy_v2/public_history_early_v17/radio'; out.mkdir(exist_ok=True)
    def write(name, obj):
        target = out/name; tmp = target.with_suffix('.tmp')
        tmp.write_text(json.dumps(obj, indent=2)+'\n'); tmp.replace(target)
    plan = dict(version='public_history_early_v17_radio', core_sha256=core,
                code_sha256={n: sha(Path(__file__).with_name(n)) for n in ['radio.py', 'codec.py', 'replay.py']},
                solver_sha256=sha(root/'tools/overnight_energy_v2/initialization_v11/wireless_init_v11.py'),
                parent_vectors_sha256=sha(source/'vectors.npz'), parent_report_sha256=sha(source/'results.json'),
                public_states={str(k): sha(observer/f'round_{k:03}.pt') for k in [28, 29]},
                config=c, public_scale=public_scale, epsilon=float(epsilon), total_time_s=float(symbols.sum()/c['symbol_rate_hz']),
                metadata='Each remote client sends16bits/rank coefficient using IEEEhalf with a single shared scale from the norm of public theta29-theta28. No private data determine scale. Fixed TDMA32symbols/rank at the same ideal0.5bit/symbol, all symbols and RFenergy debited. Residual is computed from the DECODED half coefficient, so coefficient rounding cancels algebraically; no free quantization-error assumption.',
                limits=['Actual round30 burst-Hybrid learning-only update, no unlearning request or full trajectory.',
                        'All comparators may use identical past public states26..29, half coefficients, radio budgets and AO30/5starts; generic FL candidate, no Hybrid-over-Mixed superiority.',
                        'Original absolute total-update RMS tolerance unchanged. All10 candidate heads, same total time/power/battery/overheads.',
                        'Ideal fixed-length coefficient delivery uses original Shannon abstraction. Extra projection/decoding computation and finite-block headers/errors are uncalibrated; RFsaving only with an explicit extra-compute break-even.',
                        'No validation or test outputs used and no automatic GPU promotion.'])
    manifest = out/'manifest.json'
    if manifest.exists():
        assert json.loads(manifest.read_text()) == plan
    else:
        write('manifest.json', plan)
    if args.plan_only:
        print(json.dumps(plan)); return
    assert not (out/'results.json').exists(), 'Do not duplicate an active or completed radio job.'
    results = []
    for name, rank, vectors, bits, decoded in variants:
        data = blocked(vectors); rho = np.linalg.norm(data, axis=-1); zero = np.zeros_like(rho)
        ndata, ncontrol, all_control = reserve_symbols(symbols, dims, rank, c['users']-1)
        if bits is not None:
            np.savez_compressed(out/(name+'_codec.npz'), residual=vectors, coefficient_bits=bits,
                                decoded_coefficients=decoded, public_scale=public_scale)
        candidates = []
        for head in range(c['users']):
            write('status.json', dict(status='solving_cpu_head', variant=name, head=head, at_kst=datetime.now().isoformat(timespec='seconds')))
            users = np.delete(np.arange(c['users']), head); h = channels[users, head]
            pc, ec = control_resources(h, 16*rank, ncontrol, c['symbol_rate_hz'])
            row = dict(variant=name, rank=rank, head=head, status='no_feasible_design_found')
            if np.any(pc > c['power_limit_w']*(1+1e-6)):
                row['status'] = 'control_peak_power_failed'; candidates.append(row); write(name+'_heads.json', candidates); continue
            cfg = dict(c, available_energy_j=c['available_energy_j']-float(ec.max(initial=0)))
            begin = time.monotonic()
            design = optimize_head(head, channels, rho, zero, zero, dims, ndata, epsilon, cfg, 0., 'mixed_aircomp')
            row['solve_seconds'] = time.monotonic()-begin
            if design is not None:
                a, s = design.analog_rms, design.receiver; channel = channels[users, head, None]
                data_energy = (ndata/c['symbol_rate_hz']*a*a).sum(axis=1)
                overhead = np.full(c['users'], c['base_overhead_j']); overhead[head] += c['head_overhead_j']
                assert np.max(a*a) <= c['power_limit_w']*(1+1e-6)
                assert np.all(data_energy+ec+overhead[users] <= c['available_energy_j']*(1+1e-6))
                quality = float(np.linalg.norm(np.abs(s*channel*a-rho[users]).sum(axis=0)+s*np.sqrt(dims/(2*ndata))))
                gain = np.divide(s*channel*a, rho[users], out=np.zeros_like(a), where=rho[users] > 0)
                bias = ((gain-1)[..., None]*data[users]).sum(axis=0).reshape(-1)[:dimension]
                valid = np.minimum(block, np.maximum(dimension-np.arange(len(dims))*block, 0))
                rmse = float(np.sqrt(bias@bias+np.sum(valid*s*s/(2*ndata))))
                assert rmse <= quality*(1+1e-5)+1e-15
                assert quality <= epsilon*(1+1e-5)
                energy = float(data_energy.sum()+overhead.sum()+ec.sum())
                assert np.isclose(energy-ec.sum(), design.energy_j, rtol=1e-8, atol=1e-12)
                row.update(status='feasible_radio_only', total_modeled_energy_j=energy,
                           analog_data_plus_original_overhead_j=float(data_energy.sum()+overhead.sum()),
                           coefficient_tx_energy_j=float(ec.sum()), error_bound=quality, analytic_rmse=rmse,
                           total_symbols=int(ndata.sum()+all_control), coefficient_payload_bits=16*rank*len(users),
                           coefficient_control_symbols=all_control, history=design.history)
                np.savez_compressed(out/f'{name}_h{head}.npz', analog_rms=a, receiver=s, users=users,
                                    rho=rho[users], data_symbols=ndata, control_power=pc, control_energy=ec, channels=channel)
            candidates.append(row); write(name+'_heads.json', candidates)
        good = [x for x in candidates if x['status'] == 'feasible_radio_only']
        best = min(good, key=lambda x: x['total_modeled_energy_j']) if good else None
        results.append(dict(variant=name, rank=rank, completed_heads=len(candidates), best=best))
        write('results.json', results); print('RESULT', json.dumps(results[-1]), flush=True)
    baseline = results[0]['best']; summary = []
    for item in results[1:]:
        if item['best'] is not None and baseline is not None:
            saving = baseline['total_modeled_energy_j']-item['best']['total_modeled_energy_j']
            summary.append(dict(variant=item['variant'], rf_plus_original_overhead_saving_j=saving,
                                saving_fraction=saving/baseline['total_modeled_energy_j'],
                                maximum_extra_compute_energy_for_net_gain_j=max(saving, 0.)))
    write('summary.json', summary)
    write('status.json', dict(status='completed_learning_only_radio_no_gpu_promotion', at_kst=datetime.now().isoformat(timespec='seconds')))


if __name__ == '__main__':
    main()
