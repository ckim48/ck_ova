"""FU61 low-bit screen with equal epsilon and strong phase-separated baseline."""
import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from certificate import check


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True); parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args(); root = args.project.resolve()
    sys.path[:0] = [str(root), str(root/'tools/overnight_energy_v2'), str(root/'tools/overnight_energy_v2/initialization_v11')]
    from phase_split_grid import make_inputs, split_symbols, necessary_power
    from wireless_init_v11 import optimize_head
    from transport import energy_accounting
    from adaptive_precision import exact_statistics
    source = root/'results/overnight_energy_v2/public_history_fu_v19/screen'
    registered = json.loads((source/'plan.json').read_text()); cfg = dict(registered['config'], max_ao_iterations=30)
    assert all(sha(root/n) == value for n, value in registered['core_sha256'].items())
    solver = root/'tools/overnight_energy_v2/initialization_v11/wireless_init_v11.py'
    assert sha(solver) == registered['parent_solver_sha256']
    out = root/'results/overnight_energy_v2/fu_precision_v24'; out.mkdir(exist_ok=True)
    def write(name, value):
        target = out/name; tmp = target.with_suffix('.tmp'); tmp.write_text(json.dumps(value, indent=2)+'\n'); tmp.replace(target)
    with np.load(source/'vectors.npz') as data:
        learning, unlearning, channels = [data[k].copy() for k in ['learning', 'unlearning', 'channels']]
    base = make_inputs(learning, unlearning, cfg); epsilon = float(.2*np.linalg.norm(base[5].sum(axis=0)))
    plan = dict(version='fu_precision_v24', actual_round=61, relative_tolerance=.2, epsilon=epsilon,
        bits=[6, 4, 2], modes=['hybrid', 'phase_grid'], phase_fractions=registered['phase_fractions'],
        ranks=[0], head_candidates=10, cases=24, config=cfg, core_sha256=registered['core_sha256'],
        input_vector_sha256=sha(source/'vectors.npz'), parent_solver_sha256=sha(solver),
        code_sha256={n: sha(Path(__file__).with_name(n)) for n in ['radio.py', 'certificate.py']},
        existing_comparator_results_sha256=sha(root/'results/overnight_energy_v2/public_history_fu_v19/radio/results.json'),
        limits=['Targeted fixed-actual-FU61 diagnostic, not a new training run.4/6bit room follows the quantization-only sufficient-bound preview, not any task/test outputs.',
            'All10heads, AO30/convex+4initializations, original absolute epsilon from uncompressedL, identical .167936s/.32W/.1J budgets. All7phase fractions retained, including exactminimum L_A=d at0.125 for rank0.',
            'Apply every codec to both Hybrid and phase-separated baseline. Mixed transmits no digitalU and its completed exact same-vector radio comparator remains eligible unchanged; existing8bit is imported, not rerun.',
            '2bit quantization lower floor alone exceeds the conservative envelope for every head. Report this sufficient-bound failure without declaring original physical/task infeasibility.',
            'Fixed bit setting uses the preconfigured codec;32bit scale metadata included. No new adaptive per-block bit-depth headers, history coefficients or omitted delivery airtime.',
            'Independent non-optimizer physical arithmetic checks every saved feasible incumbent, with startup reproduction of known8bit Hybrid/phase designs and deliberate invalid-power rejection.',
            'Modeled communication energy, unmeasured codec/compute/circuit extras. No NOMA/deletion/global-optimality claim, no test/model-output selection or automatic full-GPU promotion.',
            'No new head optimization starts after noonKST. If the deadline cuts the grid, preserve partial results and do not rank an incomplete codec/architecture grid.'])
    target = out/'plan.json'
    if target.exists(): assert json.loads(target.read_text()) == plan
    else: write('plan.json', plan)
    if args.plan_only:
        print(json.dumps(dict(cases=24, epsilon=epsilon, source_hashes_verified=True))); return
    assert not (out/'results.json').exists(), 'Do not duplicate this screen.'
    # Independent checker must reproduce two established physical certificates,
    # and must reject an actual power violation before new solver work begins.
    old = root/'results/overnight_energy_v2/public_history_fu_v19/radio'
    old_rows = json.loads((old/'results.json').read_text()); checks = []
    for mode, fraction in [('hybrid', None), ('phase_grid', .125)]:
        row = next(x for x in old_rows if x['relative_tolerance'] == .2 and x['rank'] == 0 and x['mode'] == mode and x['analog_fraction'] == fraction)
        with np.load(old/f'r0.2_rank0_{mode}_f{fraction}_design.npz') as data:
            d = {k: data[k].copy() for k in data.files}
        checks.append(check(learning, unlearning, channels, cfg, 8, mode, epsilon, d, row))
        invalid = {k: v.copy() for k, v in d.items()}; invalid['analog_rms'][0, 0] += 1000.
        try:
            check(learning, unlearning, channels, cfg, 8, mode, epsilon, invalid, row)
        except AssertionError:
            pass
        else: raise AssertionError('Certificate accepted invalid transmit power.')
    write('checker_startup_validation.json', dict(status='two_known_designs_and_two_invalid_power_rejections_passed', references=checks))
    rows = []; certificates = []; deadline = datetime(2026, 10, 21, 12, tzinfo=timezone(timedelta(hours=9)))
    for bits in plan['bits']:
        c = dict(cfg, quantization_bits=bits)
        l, u, decoded, delta, payload, rho, dims, total = make_inputs(learning, unlearning, c)
        for mode in plan['modes']:
            for fraction in plan['phase_fractions'] if mode == 'phase_grid' else [None]:
                started = time.monotonic(); heads = []; best = None; selected = None
                key = f'q{bits}_{mode}_f{fraction}'
                for head in range(c['users']):
                    if datetime.now(timezone(timedelta(hours=9))) >= deadline:
                        write('status.json', dict(status='partial_grid_noon_cutoff_no_ranking', completed_cases=len(rows), at_kst=datetime.now().isoformat(timespec='seconds'))); return
                    write('status.json', dict(status='solving_cpu_head', quantization_bits=bits, mode=mode, analog_fraction=fraction, head=head,
                        completed_cases=len(rows), at_kst=datetime.now().isoformat(timespec='seconds')))
                    users = np.delete(np.arange(c['users']), head); item = dict(head=head)
                    floor = float(np.linalg.norm(delta[users].sum(axis=0)))
                    item['quantization_bound_floor'] = floor
                    if floor >= epsilon:
                        item['status'] = 'conservative_quantization_bound_floor_exceeds_budget'; heads.append(item); continue
                    na, nd = total.copy(), total.copy(); active = np.any(payload[users] > 0, axis=0)
                    if mode == 'phase_grid': na, nd, active = split_symbols(total, payload, head, fraction)
                    if np.any(na < dims):
                        item['status'] = 'symbol_dimension_condition_failed'; heads.append(item); continue
                    necessary = necessary_power(channels, payload, head, nd, c)
                    if necessary is None or np.max(necessary, initial=0) > c['power_limit_w']*(1+1e-6):
                        item['status'] = 'necessary_digital_power_failed'; heads.append(item); continue
                    design = optimize_head(head, channels, rho, delta, payload, dims, na, epsilon, c,
                        c['sic_residual_fraction'], 'orthogonal' if mode == 'phase_grid' else 'hybrid', digital_symbols=nd)
                    item['status'] = 'no_feasible_design_found'
                    if design is not None:
                        stats = energy_accounting(design, c); stats.update(exact_statistics(design, l, u, decoded, learning.shape[1]))
                        item.update(status='feasible_radio_only', energy_bound_j=stats['energy_bound_j'], history=design.history)
                        if best is None or stats['energy_bound_j'] < selected['energy_bound_j']:
                            best, selected = design, stats
                    heads.append(item)
                row = dict(quantization_bits=bits, relative_tolerance=.2, rank=0, mode=mode, analog_fraction=fraction,
                    epsilon=epsilon, status='feasible_radio_only' if best is not None else 'no_feasible_design_found', heads=heads, seconds=time.monotonic()-started)
                if best is not None:
                    row.update(selected)
                    arrays = {field: getattr(best, field) for field in best.__dataclass_fields__ if field != 'history'}
                    cert = check(learning, unlearning, channels, c, bits, mode, epsilon, arrays, row)
                    cert.update(quantization_bits=bits, mode=mode, analog_fraction=fraction)
                    path = out/(key+'_design.npz'); np.savez_compressed(path, **arrays)
                    cert['design_sha256'] = sha(path); certificates.append(cert); write('certificates.json', certificates)
                rows.append(row); write('results.json', rows)
                print('CASE', json.dumps({k: v for k, v in row.items() if k != 'heads'}), flush=True)
    write('status.json', dict(status='completed_cpu_lowbit_request61_radio_no_gpu_promotion', cases=len(rows),
        saved_feasible_candidates=len(certificates), at_kst=datetime.now().isoformat(timespec='seconds')))


if __name__ == '__main__': main()
