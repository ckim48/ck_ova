"""Physical follow-up with state-transfer-aware head selection and paired seeds."""
import argparse
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import adaptive_iq_pilot_v2 as pilot

previous, parent = pilot.previous, pilot.parent
controller = pilot.controller
ROOT, read, write, sha = pilot.ROOT, pilot.read, pilot.write, pilot.sha
OUT = ROOT/'results/adaptive_iq_physical_v1_20261030'
METHODS = dict(combined='C-RMS', combined_reset='C-Reset', iq_shared='IQ-Shared', iq_separate='IQ-Separate')
SEED = 0
SETTING = None
STATE = {}
ORIGINAL_RADIO = parent.radio_module


def register():
    path = OUT/'protocol.json'
    if path.exists():
        p = read(path)
        for f, h in p['source_sha256'].items():
            assert sha(ROOT/f) == h, f
        return p
    selection = read(pilot.OUT/'selection.json')
    files = [__file__, controller.__file__, pilot.__file__, previous.__file__, parent.__file__]
    p = dict(registered_unix=time.time(), selected=selection['selected'], methods=METHODS, seeds=[0, 1, 2],
        source_sha256={str(Path(f).relative_to(ROOT)):sha(Path(f)) for f in files},
        selection_sha256=sha(pilot.OUT/'selection.json'), pilot_selection=selection['rankings'],
        design='Four algorithms x two datasets x three paired post-warm-start stochastic seeds. Seed0 is development;1 and2 are unseen minibatch/radio draws, conditional on the same cohorts and warm20. All24 attempts are retained.',
        learning='Unchanged manuscript cosine FU and uniform16 FL from the pilot. No future-forget proxy/retrained predictions used by training. Same200FL rounds,45requests,225FU cycles and10-UAV/five-region physical setup.',
        comparison='C-RMS has one persistent optimizer state; C-Reset resets that state on cohort changes. IQ-Shared collapses I/Q before the same single-state optimizer. IQ-Separate independently adapts recovered FL/FU with FU state reset on cohort changes. Same normalization and selected hyperparameters.',
        noise='Same finite-variance AWGN/phase inversion as experiment A. Known branch noise variance regularizes the RMS rule and debiases squared observations. No hand-picked channel failures or stronger noise only for a comparator.',
        overhead='Optimizer state is stored as float32 second moments plus8-byte counters per branch. On a head change, the surviving state plus128-byte header is sent digitally from old to new head. Rate=W log2(1+Pmax*|h_old,new|^2) using the actual noise-normalized link. PA/circuits/hover and post-reception head arithmetic are charged.',
        head_selection='The handover energy and hovering duration are added to the candidate-head objective. The handover RF energy is reserved from the0.1J network cap during fixed-head SOCP; the per-UAV budget is conservatively reduced by the same reserve for all remote users. Uplink plus handover must fit the original communication deadline.',
        timing='Local compute, model uplink, optimizer-state handover if required, head adaptation, then model broadcast. No additional vector transmission is hidden.',
        evaluation='Full native retained/forget cohorts, common and new optimizer-matched retained-only reference for each seed, mR/loss, active energy/delay and handover overhead. Shared reference is valid because all four policies are identical without FU. No-FU control will be exported for the selected I/Q setting.',
        limitations='Empirical curvature/HVP transport model and perfect synchronization inherited. State-aware BCD remains local, not globally optimal. Old linear-sum distortion theorem does not certify this nonlinear stateful optimizer. This is not a measured UAV deployment.')
    write(path, p)
    return p


def vectors(engine, theta, plan, queue, t, fl, kind, ids, proxy, teachers, crossmodal):
    engine.joint_base = JOINT_BASE
    engine.joint_rng_fl = lambda k: np.random.default_rng(k+1000000*SEED)
    engine.joint_rng_fu = lambda k: np.random.default_rng(k+1000000*SEED)
    if fl == 0:
        STATE.clear()
        STATE['optimizer'] = controller.RMS(controller.SETTINGS[SETTING], MODE if MODE != 'iq_shared' else 'combined')
    l, u, stats = CALL_VECTORS(engine, theta, plan, queue, t, fl, kind, ids, None, teachers, crossmodal)
    engine.iq_batches = [(r['user'], r['batch'], r['round']) for r in engine.joint_fl_records] if kind in ['fl', 'joint'] else []
    parent.curvature.ACTIVE = 'paper_iq'
    parent.curvature.complete_vectors(engine, theta, plan, queue, t, fl, kind, ids, teachers, l, u, stats)
    STATE.update(ids=ids, lr=engine.joint_base.core.lr_at(21+fl))
    return l, u, stats


def fu_objective(engine, theta, fids, rids, step, teacher):
    fm, rm, cm = previous.manuscript_terms(engine, theta, fids, rids, step, teacher)
    return 3*fm+rm+3*cm


def handover(head, channels, c):
    optimizer = STATE['optimizer']
    old = optimizer.head
    size = optimizer.state_bytes()
    if old is None or old == head or size == 0:
        return dict(bytes=0, duration_s=0., RF_j=0., draw_j=0., old_head=old, new_head=int(head))
    size += 128
    rate = c['symbol_rate_hz']*np.log2(1+c['power_limit_w']*abs(channels[old, head])**2)
    seconds = size*8/max(rate, 1e-30)
    p = parent.learning.quality.infra.PHYSICAL.PARAMETERS['communication']
    rf = c['power_limit_w']*seconds
    return dict(bytes=int(size), duration_s=float(seconds), RF_j=float(rf),
        draw_j=float(rf/p['pa_efficiency']+(p['tx_circuit_w']+p['rx_circuit_w'])*seconds),
        old_head=int(old), new_head=int(head), rate_bps=float(rate))


def radio_module(mode):
    module = ORIGINAL_RADIO('combined' if mode.startswith('combined') else 'iq')
    measure = module.measure
    solve = module.transceiver
    original = module.analog

    def state_measure(head, n, y, channels, rho, eps, c, weights, targets):
        result = measure(head, n, y, channels, rho, eps, c, weights, targets)
        if result is None:
            return None
        extra = handover(head, channels, c)
        if float(result['energy'].sum())+extra['RF_j'] > .1*(1+1e-6):
            return None
        if result['uplink_s']+extra['duration_s'] > c['round_uplink_deadline_s']+1e-12:
            return None
        result['objective'] += module.HOVER*extra['duration_s']+extra['draw_j']
        return result

    def state_solve(head, n, channels, rho, eps, c, weights, targets):
        extra = handover(head, channels, c)
        if extra['RF_j'] >= .1:
            return None
        module.CAP = .1-extra['RF_j']
        limited = dict(c, available_energy_j=c['available_energy_j']-extra['RF_j'])
        try:
            return solve(head, n, channels, rho, eps, limited, weights, targets)
        finally:
            module.CAP = .1

    module.measure, module.transceiver = state_measure, state_solve

    def analog(l, u, channels, c, seed, ell):
        optimizer = STATE['optimizer']
        optimizer.prepare(STATE['ids'])
        received, meta = original(l, u, channels, c, seed+1000000*SEED, ell)
        meta.update(adaptive_method=mode, optimizer_setting=SETTING, stochastic_seed=SEED,
                    constraint_scope='pre-optimizer transport; total RF/deadline also include state transfer')
        if received is None:
            return received, meta
        extra = handover(meta['head'], channels, c)
        var = np.array(meta['block_variance'])
        dm = c['block_dimension']
        noise = np.random.default_rng(seed+1000000*SEED).normal(size=(2, var.shape[1], dm))*np.sqrt(var[:, :, None]/dm)
        variances = np.repeat(var/dm, dm, axis=1)
        if mode == 'iq_separate':
            left, right = l.sum(0)+noise[0].ravel(), u.sum(0)+noise[1].ravel()
            assert np.allclose(left+right, received, rtol=1e-10, atol=1e-14)
        else:
            left, right = received, np.zeros_like(received)
        applied, info = optimizer.apply(left, right, variances, STATE['lr'])
        optimizer.head = meta['head']
        meta.update(optimizer=info, state_handover=extra,
            head_MAC_equivalents=info['MAC_equivalents'],
            transport_realized_error=float(np.linalg.norm(received-(l+u).sum(0))),
            applied_update_norm=float(np.linalg.norm(applied)))
        return applied, meta
    module.analog = analog
    return module


def timing(clock, compute, radio):
    result = clock.transaction_timing(compute, radio)
    macs = np.zeros(10)
    macs[radio['head']] = radio['head_MAC_equivalents']
    energy, seconds = parent.learning.quality.infra.PHYSICAL.compute_cost(macs)
    extra = radio['state_handover']
    result.update(head_compute_s=float(seconds.max()), state_handover_s=extra['duration_s'],
        head_compute_j=float(energy.sum()), state_handover_j=extra['draw_j'],
        head_extra_s=float(seconds.max())+extra['duration_s'])
    result['duration_s'] += result['head_extra_s']
    result['policy'] = 'Local compute, uplink, state handover, head optimization, model broadcast.'
    radio['physical_mode'] = 'combined_real_AirComp' if radio['adaptive_method'].startswith('combined') else 'parallel_IQ_AirComp'
    return result


def run(mode, seed):
    global MODE, SEED, SETTING
    protocol = register()
    MODE, SEED, SETTING = mode, seed, protocol['selected']
    parent.OUT = OUT/f'seed{seed}'
    parent.METHODS = METHODS
    parent.register = lambda: protocol
    parent.vectors = vectors
    parent.fu_objective = fu_objective
    parent.radio_module = radio_module
    parent.CFG = parent.learning.spec(late_uniform=0)
    parent.learning.request_gradient = previous.manuscript_gradient
    q = parent.learning.quality
    parent.learning.fl_vectors = q.infra.cached_clone(parent.learning.fl_vectors,
        [('len(pool)/total', 'len(batch)/sum(min(16,len(p)) for p in pools)', 1)], {}, live_globals=parent.learning.__dict__)
    make_original = q.make_arm

    def make(base, deployment, radios, cap):
        fn = make_original(base, deployment, radios, cap)
        return q.infra.cached_clone(fn, [
            ('clock.transaction_timing(compute,rm)', 'adaptive_timing(clock,compute,rm)', 1),
            ("start+timing['compute_s']+timing['uplink_s']", "start+timing['compute_s']+timing['uplink_s']+timing['head_extra_s']", 2)],
            dict(adaptive_timing=timing))
    q.make_arm = make
    parent.run(mode)


def queue():
    protocol = register()
    for seed in protocol['seeds']:
        for mode in METHODS:
            dest = OUT/f'seed{seed}'/mode
            if (dest/'exit.json').exists():
                continue
            dest.mkdir(parents=True, exist_ok=True)
            with (dest/'console.log').open('a') as log:
                status = subprocess.run([sys.executable, __file__, 'run', '--method', mode, '--seed', str(seed)],
                    stdout=log, stderr=subprocess.STDOUT)
            write(dest/'exit.json', dict(returncode=status.returncode, unix=time.time()))
            print(seed, mode, 'exit', status.returncode, flush=True)
    write(OUT/'queue_finished.json', dict(unix=time.time()))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['register', 'run', 'queue'])
    p.add_argument('--method', choices=list(METHODS)); p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    if a.action == 'run': run(a.method, a.seed)
    else: {'register': register, 'queue': queue}[a.action]()
