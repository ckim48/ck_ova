"""Fixed-update comparison of joint and frozen digital-power AO updates."""
import argparse
import hashlib
from datetime import datetime
import json
from pathlib import Path
import sys
import time

import numpy as np


EXPECTED_WIRELESS = '0d981861984500a14890ff5391d1d826068fb879cedf76039f4403b8729a8606'


def read_baselines(root):
    cache, hashes = {}, {}
    paths = ['results/overnight_energy_v2/phase_split_grid/results.json',
             'results/overnight_energy_v2/error_budget_regimes/results.json',
             'results/overnight_energy_v2/joint_power_v3/results.json']
    for name in paths:
        path = root/name
        if not path.exists():
            continue
        raw = path.read_bytes(); hashes[name] = hashlib.sha256(raw).hexdigest()
        for row in json.loads(raw):
            row = dict(row)
            if 'selected_energy_bound_j' in row:
                row.update(status='feasible', energy_bound_j=row['selected_energy_bound_j'])
            if 'relative_tolerance' in row:
                key = (row['bits'], row['relative_tolerance'], row['mode'], row.get('analog_fraction'))
            else:
                key = (row['bits'], .2, 'orthogonal', row['analog_fraction'])
            cache[key] = dict(row)
    precision = root/'results/energy_quality_radio_v1/precision_design_audit.json'
    raw = precision.read_bytes(); hashes[str(precision.relative_to(root))] = hashlib.sha256(raw).hexdigest()
    for row in json.loads(raw)['rows']:
        cache.setdefault((row['bits'], .2, 'hybrid', None), dict(row))
    return cache, hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--bits', type=int, nargs='+', default=[8, 6])
    parser.add_argument('--relative-tolerances', type=float, nargs='+', default=[.2, .6])
    parser.add_argument('--analog-fractions', type=float, nargs='+', default=[.125, .25, .375, .5, .625, .75, .875])
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args(); root = args.project.resolve()
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root/'tools/overnight_energy_v2'))
    from phase_split_grid import make_inputs, necessary_power, split_symbols
    from adaptive_precision import exact_statistics
    from transport import energy_accounting
    import wireless_init_v11 as joint
    source_hash = {name: hashlib.sha256((root/name).read_bytes()).hexdigest()
                   for name in ('wireless.py', 'transport.py')}
    if source_hash['wireless.py'] != EXPECTED_WIRELESS:
        raise ValueError('The copied solver no longer matches the original source.')
    source = root/'results/failed_round_diagnostic'
    metadata = json.loads((source/'metadata.json').read_text()); original_config = metadata['config']; config = dict(original_config, max_ao_iterations=30)
    with np.load(source/'updates.npz') as data:
        learning, unlearning, channels = [data[name].copy() for name in ('learning', 'unlearning', 'channels')]
    output = root/'results/overnight_energy_v2/initialization_v11'
    output.mkdir(parents=True, exist_ok=True)
    baselines, baseline_hashes = read_baselines(root)
    manifest = {'bits': args.bits, 'relative_tolerances': args.relative_tolerances,
        'analog_fractions': args.analog_fractions, 'config': config,
        'source_sha256': source_hash, 'baseline_config': original_config,
        'parent_solver_sha256': hashlib.sha256((root/'tools/overnight_energy_v2/joint_power_v3/wireless_joint_v3.py').read_bytes()).hexdigest(),
        'updates_sha256': hashlib.sha256((source/'updates.npz').read_bytes()).hexdigest(),
        'variant_sha256': hashlib.sha256(Path(joint.__file__).read_bytes()).hexdigest(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'baseline_snapshot_sha256': baseline_hashes,
        'interpretation': 'CPU fixed original legacy state. Always compare convex minimum-error initialization with unchanged four reference starts; increase AO from8 to30 for Hybrid, Mixed and all seven digital-phase fractions equally. Parent v3 and original feasible incumbents remain eligible. Physical time/power/battery/sum-RMS/codec definitions unchanged. Finite local optimization, not a global optimum or task/deletion guarantee. No GPU promotion or alteration of frozen training sources.'}
    path = output/'manifest.json'
    if path.exists():
        saved = json.loads(path.read_text())
        # An ongoing baseline grid can append rows; all other inputs must remain identical.
        a, b = dict(saved), dict(manifest)
        a.pop('baseline_snapshot_sha256'); b.pop('baseline_snapshot_sha256')
        if a != b:
            raise ValueError('Existing joint-power experiment differs.')
    else:
        path.write_text(json.dumps(manifest, indent=2)+'\n')
    if args.plan_only:
        print(json.dumps(manifest, indent=2)); return
    def status(state, **extra):
        target=output/'status.json'; temporary=output/'status.tmp'
        temporary.write_text(json.dumps(dict(status=state,at_kst=datetime.now().isoformat(timespec='seconds'),**extra),indent=2)+'\n'); temporary.replace(target)
    status('running_cpu_only')
    path = output/'results.json'; rows = json.loads(path.read_text()) if path.exists() else []
    done = {(r['bits'], r['relative_tolerance'], r['mode'], r.get('analog_fraction')) for r in rows}
    for bits in args.bits:
        for relative in args.relative_tolerances:
            c = dict(config, quantization_bits=bits, relative_tolerance=relative)
            base_inputs = make_inputs(learning, unlearning, c)
            epsilon = max(c['absolute_tolerance_floor'], relative*np.linalg.norm(base_inputs[5].sum(axis=0)))
            total_time = base_inputs[-1].sum()/c['symbol_rate_hz']
            for mode, fraction in [('hybrid', None), ('mixed_aircomp', None)]+[('orthogonal', x) for x in args.analog_fractions]:
                key = (bits, relative, mode, fraction)
                if key in done:
                    continue
                started = time.monotonic()
                if mode == 'mixed_aircomp':
                    inputs = make_inputs(learning+unlearning, np.zeros_like(unlearning), c)
                else:
                    inputs = base_inputs
                l, u, decoded, delta, payload, rho, dims, symbols = inputs
                best = None; heads = []
                for head in range(c['users']):
                    status('solving_cpu_head',bits=bits,relative_tolerance=relative,mode=mode,analog_fraction=fraction,head=head,completed_cases=len(rows))
                    analog_symbols, digital_symbols = symbols.copy(), symbols.copy()
                    if mode == 'orthogonal':
                        analog_symbols, digital_symbols, _ = split_symbols(symbols, payload, head, fraction)
                    item = {'head': head, 'status': 'no_feasible_design_found'}
                    if np.any(analog_symbols < dims):
                        item['status'] = 'symbol_dimension_condition_failed'; heads.append(item); continue
                    floor = necessary_power(channels, payload, head, digital_symbols, c)
                    if floor is None or np.max(floor, initial=0) > c['power_limit_w']*(1+1e-6):
                        item['status'] = 'necessary_digital_power_condition_failed'; heads.append(item); continue
                    design = joint.optimize_head(head, channels, rho, delta, payload, dims,
                        analog_symbols, epsilon, c, c['sic_residual_fraction'], mode,
                        digital_symbols=digital_symbols)
                    if design is not None:
                        item.update(status='feasible', energy_bound_j=design.energy_j,
                                    error_bound=design.error_bound, history=design.history)
                        assert np.all(np.diff(design.history) <= np.maximum(1e-12, np.array(design.history[:-1])*1e-6))
                        if best is None or design.energy_j < best.energy_j:
                            best = design
                    heads.append(item)
                row = {'bits': bits, 'relative_tolerance': relative, 'mode': mode,
                       'analog_fraction': fraction, 'epsilon': float(epsilon),
                       'status': 'no_feasible_design_found', 'heads': heads,
                       'original_incumbent': baselines.get(key)}
                if best is not None:
                    row.update(energy_accounting(best, c))
                    row.update(exact_statistics(best, l, u, decoded, learning.shape[1]))
                    if mode == 'mixed_aircomp':
                        row['unlearning_component_error'] = None
                    row['status'] = 'feasible'
                    assert np.isclose(row['uplink_s'], total_time)
                    assert row['error_bound'] <= epsilon*(1+1e-5)
                    assert row['analytic_total_rmse'] <= row['error_bound']*(1+1e-4)+1e-12
                    assert row['expected_energy_random_codewords_j'] <= row['energy_bound_j']*(1+1e-6)
                    name = f"b{bits}_r{relative:g}_{mode}_f{fraction}"
                    np.savez_compressed(output/f'{name}_design.npz', **{
                        field: getattr(best, field) for field in best.__dataclass_fields__ if field != 'history'})
                original = row['original_incumbent']
                eligible = []
                if original and original['status'] == 'feasible':
                    eligible.append(('original_incumbent', original['energy_bound_j']))
                if best is not None:
                    eligible.append(('joint_update', best.energy_j))
                if eligible:
                    row['selected_solver'], row['selected_energy_bound_j'] = min(eligible, key=lambda p:p[1])
                row['seconds'] = time.monotonic()-started; rows.append(row)
                tmp = path.with_suffix('.partial.json'); tmp.write_text(json.dumps(rows, indent=2)+'\n'); tmp.replace(path)
                print(json.dumps({k:v for k,v in row.items() if k not in ('heads', 'original_incumbent')}), flush=True)
    comparisons = []
    for bits in args.bits:
        for relative in args.relative_tolerances:
            group = [r for r in rows if r['bits'] == bits and r['relative_tolerance'] == relative]
            record = {'bits': bits, 'relative_tolerance': relative}
            for mode in ['hybrid', 'orthogonal', 'mixed_aircomp']:
                feasible = [r for r in group if r['mode'] == mode and 'selected_energy_bound_j' in r]
                if feasible:
                    best = min(feasible, key=lambda r:r['selected_energy_bound_j'])
                    record[mode] = {k:best.get(k) for k in ('selected_energy_bound_j', 'selected_solver', 'analog_fraction')}
            comparisons.append(record)
    (output/'comparisons.json').write_text(json.dumps(comparisons, indent=2)+'\n')
    status('completed_cpu_only_not_promoted',cases=len(rows))
    print('COMPARISONS', json.dumps(comparisons), flush=True)


if __name__ == '__main__':
    main()
