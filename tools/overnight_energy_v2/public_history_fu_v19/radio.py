"""Paid codec with concurrentL/U: paired54-case all-head radio diagnostic."""
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
    args = p.parse_args(); root = args.project.resolve()
    sys.path[:0] = [str(root), str(root/'tools/overnight_energy_v2'), str(root/'tools/overnight_energy_v2/initialization_v11')]
    from phase_split_grid import make_inputs, split_symbols, necessary_power
    from wireless_init_v11 import optimize_head
    from transport import energy_accounting
    from adaptive_precision import exact_statistics
    family = root/'results/overnight_energy_v2/public_history_fu_v19'; source = family/'screen'
    registered = json.loads((source/'plan.json').read_text()); c = dict(registered['config'], max_ao_iterations=30)
    assert json.loads((source/'status.json').read_text())['status'] == 'completed_actual_request_cpu_replay'
    for n, expected in registered['core_sha256'].items():
        assert sha(root/n) == expected
    for n, expected in registered['code_sha256'].items():
        assert sha(Path(__file__).with_name(n)) == expected
    solver = root/'tools/overnight_energy_v2/initialization_v11/wireless_init_v11.py'
    assert sha(solver) == registered['parent_solver_sha256']
    with np.load(source/'vectors.npz') as data:
        learning, unlearning, channels = [data[k].copy() for k in ['learning', 'unlearning', 'channels']]
        bases = {rank: data[f'rank{rank}_basis'].copy() for rank in [1, 3]}; public_scale = float(data['public_scale'])
    out = family/'radio'; out.mkdir(exist_ok=True)
    def write(name, value):
        target = out/name; tmp = target.with_suffix('.tmp')
        tmp.write_text(json.dumps(value, indent=2)+'\n'); tmp.replace(target)
    manifest = dict(version='public_history_fu_v19_radio', config=c, registered_replay_plan_sha256=sha(source/'plan.json'),
                    vectors_sha256=sha(source/'vectors.npz'), solver_sha256=sha(solver),
                    code_sha256={n: sha(Path(__file__).with_name(n)) for n in ['radio.py', 'codec.py']},
                    ranks=[0, 1, 3], relative_tolerances=[.2, .6], modes=['hybrid', 'mixed_aircomp', 'phase_grid'],
                    fractions=registered['phase_fractions'], head_candidates=c['users'],
                    limits=registered['limits']+['Conservative per-user battery margin reserves maximum coefficient RF energy in the solver, followed by actual per-user battery checking. Extra computation/basis/headers uncalibrated.',
                        'Every mode/rank uses epsilon computed from the SAME uncompressed learning vectors; no quality loosening by projection or by Mixed norm changes.',
                        'All mode/rank cases use original absolute error/time/power/battery/symbol conventions. Finite feasible incumbents, not global optimality or actual task quality. All failures retained.'])
    assert not (out/'manifest.json').exists(), 'Do not duplicate this radio diagnostic.'
    write('manifest.json', manifest)
    base = make_inputs(learning, unlearning, c); total_symbols = base[-1]; dims = base[-2]
    rows = []; dimension = learning.shape[1]
    for relative in [.2, .6]:
        epsilon = max(c['absolute_tolerance_floor'], relative*np.linalg.norm(base[5].sum(axis=0)))
        for rank in [0, 1, 3]:
            for mode in manifest['modes']:
                analog = learning+unlearning if mode == 'mixed_aircomp' else learning
                digital = np.zeros_like(unlearning) if mode == 'mixed_aircomp' else unlearning
                coefficient = None
                if rank:
                    analog, bits, coefficient = encode_residual(analog, bases[rank], public_scale)
                    np.testing.assert_allclose(analog+coefficient@bases[rank].T,
                                               learning+unlearning if mode == 'mixed_aircomp' else learning,
                                               rtol=1e-10, atol=1e-15)
                l, u, decoded, delta, payload, rho, _, _ = make_inputs(analog, digital, c)
                nmain, ncontrol, control_total = reserve_symbols(total_symbols, dims, rank, c['users']-1)
                fractions = manifest['fractions'] if mode == 'phase_grid' else [None]
                for fraction in fractions:
                    started = time.monotonic(); heads = []; best = None; best_row = None
                    key = f'r{relative:g}_rank{rank}_{mode}_f{fraction}'
                    for head in range(c['users']):
                        write('status.json', dict(status='solving_cpu_head', relative_tolerance=relative, rank=rank, mode=mode,
                                                 analog_fraction=fraction, head=head, completed_cases=len(rows), at_kst=datetime.now().isoformat(timespec='seconds')))
                        users = np.delete(np.arange(c['users']), head)
                        pc, ec = control_resources(channels[users, head], 16*rank, ncontrol, c['symbol_rate_hz'])
                        item = dict(head=head, status='no_feasible_design_found')
                        if np.any(pc > c['power_limit_w']*(1+1e-6)):
                            item['status'] = 'coefficient_peak_power_failed'; heads.append(item); continue
                        na = nmain.copy(); nd = nmain.copy(); physical_mode = mode
                        if mode == 'phase_grid':
                            na, nd, active = split_symbols(nmain, payload, head, fraction); physical_mode = 'orthogonal'
                        else:
                            active = np.any(payload[users] > 0, axis=0)
                        if np.any(na < dims):
                            item['status'] = 'symbol_dimension_condition_failed'; heads.append(item); continue
                        floor = necessary_power(channels, payload, head, nd, c)
                        if floor is None or np.max(floor, initial=0) > c['power_limit_w']*(1+1e-6):
                            item['status'] = 'necessary_digital_power_failed'; heads.append(item); continue
                        cfg = dict(c, available_energy_j=c['available_energy_j']-float(ec.max(initial=0)))
                        design = optimize_head(head, channels, rho, delta, payload, dims, na, epsilon, cfg,
                                               c['sic_residual_fraction'], physical_mode, digital_symbols=nd)
                        if design is not None:
                            stats = energy_accounting(design, c); stats.update(exact_statistics(design, l, u, decoded, dimension))
                            time_main = na.sum()+(nd[active].sum() if mode == 'phase_grid' else 0)
                            assert int(time_main)+control_total == int(total_symbols.sum())
                            assert np.isclose(stats['uplink_s'], time_main/c['symbol_rate_hz'])
                            assert stats['error_bound'] <= epsilon*(1+1e-5)
                            assert stats['analytic_total_rmse'] <= stats['error_bound']*(1+1e-5)
                            duration_a = na/c['symbol_rate_hz']; duration_d = nd/c['symbol_rate_hz']
                            if mode == 'phase_grid':
                                energy_blocks = duration_a*design.analog_rms**2+duration_d*design.digital_sqrt_power**2
                            else:
                                energy_blocks = duration_a*(design.analog_rms+design.digital_sqrt_power)**2
                            assert np.all(energy_blocks.sum(axis=1)+ec+c['base_overhead_j'] <= c['available_energy_j']*(1+1e-5))
                            energy = stats['energy_bound_j']+float(ec.sum())
                            item.update(status='feasible_radio_only', energy_bound_j=energy, data_plus_original_overhead_j=stats['energy_bound_j'],
                                        coefficient_rf_energy_j=float(ec.sum()), error_bound=stats['error_bound'], history=design.history)
                            if best is None or energy < best_row['energy_bound_j']:
                                best = design
                                best_row = {**stats, 'energy_bound_j': energy,
                                    'expected_energy_including_coefficients_j': stats['expected_energy_random_codewords_j']+float(ec.sum()),
                                    'data_plus_original_overhead_j': stats['energy_bound_j'], 'coefficient_rf_energy_j': float(ec.sum()),
                                    'total_uplink_s': float(total_symbols.sum()/c['symbol_rate_hz']), 'coefficient_payload_bits': 16*rank*len(users),
                                    'coefficient_control_symbols': control_total, 'head': head}
                                best_control = (pc, ec)
                        heads.append(item)
                    row = dict(relative_tolerance=relative, rank=rank, mode=mode, analog_fraction=fraction, epsilon=float(epsilon),
                               status='feasible_radio_only' if best is not None else 'no_feasible_design_found',
                               heads=heads, seconds=time.monotonic()-started)
                    if best is not None:
                        row.update(best_row)
                        if mode == 'mixed_aircomp':
                            row['unlearning_component_error'] = None
                        arrays = {field: getattr(best, field) for field in best.__dataclass_fields__ if field != 'history'}
                        arrays.update(control_power=best_control[0], control_energy=best_control[1])
                        np.savez_compressed(out/(key+'_design.npz'), **arrays)
                    rows.append(row); write('results.json', rows)
                    print('CASE', json.dumps({k: v for k, v in row.items() if k != 'heads'}), flush=True)
    comparisons = []
    for relative in [.2, .6]:
        for mode in manifest['modes']:
            result = dict(relative_tolerance=relative, mode=mode)
            for rank in [0, 1, 3]:
                eligible = [x for x in rows if x['relative_tolerance'] == relative and x['mode'] == mode and x['rank'] == rank and x['status'] == 'feasible_radio_only']
                result['rank'+str(rank)] = min(eligible, key=lambda x: x['energy_bound_j']) if eligible else None
            comparisons.append(result)
    write('comparisons.json', comparisons)
    write('status.json', dict(status='completed_cpu_request61_radio_no_gpu_promotion', cases=len(rows), at_kst=datetime.now().isoformat(timespec='seconds')))


if __name__ == '__main__':
    main()
