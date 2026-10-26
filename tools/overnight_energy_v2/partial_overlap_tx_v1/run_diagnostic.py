"""Versioned CPU fixed-update experiment requested by the user."""
from pathlib import Path
import hashlib, json, sys, time
import numpy as np
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[2]
sys.path[:0] = [str(HERE), str(ROOT/'tools/overnight_energy_v2/deletion_window_screen_v1'),
               str(ROOT), str(ROOT/'tools/overnight_energy_v2'),
               str(ROOT/'tools/overnight_energy_v2/initialization_v11'),
               str(ROOT/'tools/overnight_energy_v2/receiver_control_v1')]
from overlap_solver import optimize_partial, audit, received_coefficients
from phase_split_grid import make_inputs
from receiver_control import sensitivity_bound
from trajectory_transport import oma_solve

OUT = ROOT/'results/overnight_energy_v2/partial_overlap_tx_v1'

def write(p, x):
    t = p.with_suffix('.tmp'); t.write_text(json.dumps(x, indent=2)+'\n'); t.replace(p)

def tests():
    # Analytic single-user and multi-user SIC rates, zero-bit phases, plus
    # an independent audit that rejects inadequate rate and power violations.
    h = np.array([2., 1.]); bits = np.array([[20.], [10.]])
    co = received_coefficients(bits, np.array([10.]), h)
    assert np.allclose(co[:, 0], [6., 1.])
    assert received_coefficients(bits, np.array([0.]), h) is None
    assert np.all(received_coefficients(bits*0, np.array([0.]), h) == 0)
    c = dict(power_limit_w=100., available_energy_j=100., symbol_rate_hz=100., coherence_time_s=1.)
    a = np.array([[.1], [.2]]); rho = h[:, None]*a; rx = np.ones(1)
    la = np.array([10.]); ld = np.array([20.]); eta = .5; dims = np.array([1])
    co = received_coefficients(eta*bits, la, h); ct = received_coefficients((1-eta)*bits, ld-la, h)
    po = co/h[:, None]**2*(1+(h[:, None]*a).sum(axis=0)**2); pt = ct/h[:, None]**2
    good = audit(a, po, pt, rx, h, rho, bits, la, ld, eta, dims, 1., c)
    assert good['passed']
    assert not audit(a, po*.9, pt, rx, h, rho, bits, la, ld, eta, dims, 1., c)['passed']
    assert not audit(a, po, pt, rx, h, rho, bits, la, ld, eta, dims, 1., dict(c, power_limit_w=.01))['passed']
    print('PHYSICS_TESTS_PASSED', flush=True)

def main():
    tests(); OUT.mkdir(exist_ok=False)
    prior = json.loads((ROOT/'results/overnight_energy_v2/deletion_window_screen_v1/plan.json').read_text())
    c = dict(prior['config'], base_overhead_j=0., head_overhead_j=0.)
    source = ROOT/'results/overnight_energy_v2/crossmodal_ablation_v1/prepared/crossmodal/k5/prepared.npz'
    with np.load(source) as z:
        learning, unlearning, channels, gradient = [z[k].copy() for k in ['learning', 'unlearning', 'channels', 'gradient']]
    l, u, decoded, delta, payload, rho, dims, total = make_inputs(learning, unlearning, c)
    la = np.floor(total*.125).astype(int); ld = total-la
    active = np.flatnonzero(np.any(unlearning != 0, axis=1))
    canonical = decoded.reshape(len(channels), -1)[:, :learning.shape[1]]
    lip = float(sensitivity_bound(canonical[active], gradient))
    epsilon = max(c['absolute_tolerance_floor'], float(c['relative_tolerance']*np.linalg.norm(rho.sum(axis=0))))
    eps = epsilon/lip
    plan = dict(config=c, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')},
                metric_definition='Total delay = sum of occupied uplink frame durations. Energy = uplink transmit waveform energy only. Receiver, SIC, model reconstruction, optimization, training, evaluation, control and downlink excluded for every method.',
                scope='Fixed identical updates, canonical Q8, same channels/quality/power/battery. CPU diagnostic; not a learned trajectory or deletion-quality certificate.',
                analog_time_s=float(la.sum()/c['symbol_rate_hz']), digital_time_s=float(ld.sum()/c['symbol_rate_hz']),
                separated_time_s=float(total.sum()/c['symbol_rate_hz']), overlap_time_s=float(ld.sum()/c['symbol_rate_hz']),
                change_from_original='Analog occupies only the initial 1/8 original frame inside the 7/8 digital frame. Digital payload/power differ by overlap/tail. SIC assumed perfect. A prescribed split grid and alternating receiver optimization yield feasible candidates, not global optima.',
                expected_energy_scope='Uncorrelated random codeword expectation reaccounts the robust-bound design; not a separate average-power optimum.',
                application_error_budget=epsilon, controller_lipschitz=lip)
    write(OUT/'plan.json', plan)
    best = None; rows = []
    # Evaluate the formerly selected head first, then all other heads fairly.
    for head in [5]+[i for i in range(len(channels)) if i != 5]:
        write(OUT/'progress.json', dict(status='running', method='partial_overlap', head=head, completed_heads=len(rows)))
        warm = []
        old = ROOT/f'results/overnight_energy_v2/deadline_frontier_v1/k5_spc16_oma_design.npz'
        if head == 5 and old.exists():
            with np.load(old) as z: warm = [z['receiver'].copy()]
        started = time.perf_counter()
        d = optimize_partial(head, channels, rho, payload, dims, la, ld, eps, c, warm_receivers=warm)
        row = dict(head=head, status='feasible' if d else 'no_feasible_design_found', solver_wall_s_excluded=time.perf_counter()-started)
        if d:
            row.update({k:v for k,v in d.items() if not isinstance(v, np.ndarray)})
            row['application_error_bound'] = d['error_bound']*lip
            if best is None or d['tx_energy_bound_j'] < best['tx_energy_bound_j']:
                best = row.copy(); np.savez_compressed(OUT/'best_design.npz', **{k:v for k,v in d.items() if isinstance(v, np.ndarray)})
        rows.append(row); write(OUT/'head_results.json', rows)
        write(OUT/'partial_result.json', dict(best=best, total_delay_s=plan['overlap_time_s'], completed_heads=len(rows)))
        print(json.dumps(row), flush=True)
    reference = json.loads((ROOT/'results/overnight_energy_v2/deadline_frontier_v1/results.json').read_text())
    comparisons = []
    overhead = c['users']*prior['config']['base_overhead_j']+prior['config']['head_overhead_j']
    for r in reference:
        if r['K'] != 5 or r['symbols_per_coordinate'] not in [14, 16] or r['status'] != 'feasible': continue
        comparisons.append(dict(method=r['method']+'_existing_fixed_update_design', total_delay_s=r['deadline_s'],
                                tx_energy_bound_j=r['energy_bound_j']-overhead,
                                tx_energy_expected_j=r['expected_energy_j']-overhead,
                                note='Same fixed input; original feasible design with fixed communication overhead removed. Not reoptimized under relaxed battery overhead.'))
    if best:
        comparisons.append(dict(method='partial_overlap_tx', total_delay_s=plan['overlap_time_s'], **best))
    write(OUT/'comparison.json', comparisons)
    write(OUT/'progress.json', dict(status='completed', completed_heads=len(rows), feasible=best is not None))

if __name__ == '__main__':
    if '--test-only' in sys.argv: tests()
    else: main()
