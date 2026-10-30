"""Registered head-fusion development test; preserve every attempted result."""
import argparse
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import combined_iq_experiment_a as parent
import head_fusion_math_v1 as fusion

ROOT, read, write, sha = parent.ROOT, parent.read, parent.write, parent.sha
OUT = ROOT/'results/head_fusion_v1_20261030'
METHODS = dict(combined_sum='C-Sum', iq_sum='IQ-Sum', combined_local_pc='C-Local',
               iq_head_pc='IQ-Head', iq_head_nr='IQ-NR')
FAMILY = MODE = None
RAW = {}
ORIGINAL_RADIO = parent.radio_module
ORIGINAL_FU = parent.fu_objective


def register():
    path = OUT/'protocol.json'
    sources = [__file__, fusion.__file__, parent.__file__, parent.learning.__file__,
               parent.curvature.__file__, parent.iq.__file__]
    if path.exists():
        p = read(path)
        for name, digest in p['source_sha256'].items():
            assert sha(ROOT/name) == digest, name
        return p
    p = dict(registered_unix=time.time(), seed=0, datasets=['CapERA', 'AirScape'],
        methods=METHODS, families=['current', 'manuscript'],
        source_sha256={str(Path(f).relative_to(ROOT)): sha(Path(f)) for f in sources},
        design='2 learning families x 5 arms x 2 datasets. Reuse only the four frozen current-family sum controls from experiment A. Sixteen fresh 200-round arms. No outcome-dependent parameter search or omitted losses.',
        current='Unchanged priority16 FL through100 then uniform16, pool-size FL weights; 3 relu(forget proxy retrieval KL-.01)+.95 retained CE+3 request-start retrieval KL.',
        manuscript='Uniform16 FL with actual minibatch-size weights; 3 positive-pair cosine +1 retained CE+3 embedding-softmax consistency KL against request-start teacher. Numerical weights are disclosed operational choices inherited from the earlier manuscript pilot, not supplied by the user equations.',
        controls='C-Sum: one combined real stream. IQ-Sum: two branches and linear sum. C-Local: two-task PCGrad independently at each UAV, then one combined real stream. IQ-Head: two-task PCGrad on noisy global I/Q estimates. IQ-NR: same with fixed 1.645 dot-noise margin and variance-regularized denominators. Both AirComp transports use one simultaneous interval.',
        fusion='For d=L dot U and q=max(0,-d), apply L+U+q*(L/||L||^2+U/||U||^2). IQ-NR replaces q by max(0,-d-1.645 estimated_dot_SD) and denominators by estimated-vector norm squared plus known noise variance. No claimed statistical confidence guarantee.',
        fairness='Same warm20, native samples, request schedule, seeds,200 FL rounds,225 FU steps, channels, hard caps and empirical HVP procedure. Raw local updates define shared error budgets; actual transmitted vectors define RF costs. Unmodified baselines retained. Differences across learning families are not attributed to transport.',
        information='Head uses only recovered branch vectors and designed noise variances. Clean vectors are diagnostic-only. No mR, evaluation KL or retrained-model output enters fusion. Current-family inherited known-future-forget-cohort proxy assumption remains.',
        accounting='Local surgery charges32*d+128 scalar MAC equivalents per active UAV. Head PCGrad charges32*d+128; head NR64*d+10*M+128, after uplink and before broadcast on the selected head. Added time incurs hovering and added CPU energy is exported explicitly. Constants are conservative modeling choices, not measured UAV hardware profiling. Curvature and optimizer scope unchanged from experiment A.',
        theory='Component-MSE and C8 curvature constraints are transport pre-fusion quantities. Nonlinear fusion may bias/amplify noise. Log post-fusion error; do not reuse the zero-mean linear-sum quality theorem. No forgetting certificate.',
        evaluation='Full native gallery; retained mR/loss, common and policy-matched forget KL, energy, delay, conflict/correction rates. Same-family retained-only references are evaluation-only. One already-inspected development seed, not independent confirmation.',
        citation='Tianhe Yu et al., Gradient Surgery for Multi-Task Learning, NeurIPS2020, https://proceedings.neurips.cc/paper_files/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html')
    write(path, p)
    write(OUT/'math_checks.json', fusion.check())
    return p


def manuscript_terms(engine, theta, fids, rids, step, teacher):
    import torch
    import torch.nn.functional as F
    fcids = [c for i in fids for c in engine.data.caption_ids[i]]
    rcids = engine.data.training_caption_ids(rids, 21+step)
    im, tx, scale = engine.features(theta, fids, fcids)
    ri, rt, _ = engine.features(theta, rids, rcids)
    old = teacher.features(rids, rcids)
    owner = torch.tensor([j for j, i in enumerate(fids) for _ in engine.data.caption_ids[i]], device=theta.device)
    fm = (im[owner]*tx).sum(-1).mean()
    rm = parent.curvature.symmetric_ce(ri, rt, scale)
    cm = F.kl_div(F.log_softmax(ri, dim=-1), F.softmax(old[0], dim=-1), reduction='batchmean')
    cm += F.kl_div(F.log_softmax(rt, dim=-1), F.softmax(old[1], dim=-1), reduction='batchmean')
    return fm, rm, cm


def manuscript_gradient(engine, theta, fids, rids, step, teacher):
    import torch
    variable = theta.detach().requires_grad_(True)
    fm, rm, cm = manuscript_terms(engine, variable, fids, rids, step, teacher)
    g = torch.autograd.grad(3*fm+rm+3*cm, variable)[0].detach()
    g[-1] = 0
    nc = sum(len(engine.data.caption_ids[i]) for i in fids)
    extra = int((len(fids)*nc+len(rids)**2)*(4*512+32)+
        2*10240*(len(rids)*2)+16*(len(theta)-1)+64)
    parent.learning.quality.infra.lib.infra.charge(engine, fids, extra)
    return g, dict(FU_variant='manuscript_cosine', forget_loss=float(fm.detach()),
        RM=float(rm.detach()), CM=float(cm.detach()), RM_weight=1.,
        FU_gradient_norm=float(g.norm()), retained_images=rids, extra_control_MAC_equivalents=extra)


def objective(engine, theta, fids, rids, step, teacher):
    if FAMILY == 'current':
        return ORIGINAL_FU(engine, theta, fids, rids, step, teacher)
    fm, rm, cm = manuscript_terms(engine, theta, fids, rids, step, teacher)
    return 3*fm+rm+3*cm


def vectors(engine, theta, plan, queue, t, fl, kind, ids, proxy, teachers, crossmodal):
    engine.joint_base = JOINT_BASE
    engine.joint_rng_fl = lambda k: np.random.default_rng(k)
    engine.joint_rng_fu = lambda k: np.random.default_rng(k)
    l, u, stats = CALL_VECTORS(engine, theta, plan, queue, t, fl, kind, ids, proxy, teachers, crossmodal)
    engine.iq_batches = [(r['user'], r['batch'], r['round']) for r in engine.joint_fl_records] if kind in ['fl', 'joint'] else []
    parent.curvature.ACTIVE = 'iq_only' if FAMILY == 'current' else 'paper_iq'
    parent.curvature.complete_vectors(engine, theta, plan, queue, t, fl, kind, ids, teachers, l, u, stats)
    RAW.clear()
    RAW.update(l=l.copy(), u=u.copy(), local=[])
    if MODE == 'combined_local_pc' and np.any(u):
        l, u, RAW['local'] = fusion.local_fuse(l, u)
        for row in engine.joint_fl_records:
            user = row['user']
            if np.any(RAW['u'][user]):
                parent.learning.quality.infra.lib.infra.charge(engine, row['batch'], 32*l.shape[1]+128)
    return l, u, stats


def radio_module(mode):
    module = ORIGINAL_RADIO('combined' if mode.startswith('combined') else 'iq')
    if mode == 'combined_local_pc':
        def inputs(l, u, c, ell):
            _, eps, weights, targets = parent.iq.inputs(RAW['l'], RAW['u'], c, ell)
            rho = np.zeros((2, len(l), l.shape[1]//c['block_dimension']))
            rho[0] = np.linalg.norm((l+u).reshape(len(l), -1, c['block_dimension']), axis=-1)
            return rho, np.array([np.linalg.norm(eps), 0.]), weights, targets
        module.inputs = inputs
    original = module.analog

    def analog(l, u, channels, c, seed, ell):
        received, meta = original(l, u, channels, c, seed, ell)
        meta.update(fusion_method=mode, learning_family=FAMILY,
                    physical_mode='combined_real_AirComp' if mode.startswith('combined') else 'parallel_IQ_AirComp',
                    head_MAC_equivalents=0, constraint_scope='pre-fusion transport only')
        if received is None:
            return received, meta
        clean_l, clean_u = RAW['l'].sum(0), RAW['u'].sum(0)
        _, rawstats = fusion.fuse(clean_l, clean_u)
        meta['clean_global_diagnostic'] = rawstats
        meta['local_fusion_diagnostics'] = RAW['local']
        meta['pre_fusion_realized_error_norm'] = float(np.linalg.norm(received-(l+u).sum(0)))
        ideal = (l+u).sum(0)
        if mode in ['iq_head_pc', 'iq_head_nr'] and np.any(u):
            variance = np.array(meta['block_variance'])
            m, dm = variance.shape[1], c['block_dimension']
            noise = np.random.default_rng(seed).normal(size=(2, m, dm))*np.sqrt(variance[:, :, None]/dm)
            lh, uh = l.sum(0)+noise[0].ravel(), u.sum(0)+noise[1].ravel()
            assert np.allclose(lh+uh, received, rtol=1e-10, atol=1e-14)
            nr = mode == 'iq_head_nr'
            fused, info = fusion.fuse(lh, uh, variance, nr)
            # Preserve original receive arithmetic exactly if there is no correction.
            if info['corrected']:
                received = received + (fused-(lh+uh))
            ideal = fusion.fuse(l.sum(0), u.sum(0), variance, nr)[0]
            meta['head_fusion'] = info
            meta['head_MAC_equivalents'] = int((64*l.shape[1]+10*m+128) if nr else (32*l.shape[1]+128))
        meta['post_fusion_error_norm'] = float(np.linalg.norm(received-ideal))
        meta['applied_update_norm'] = float(np.linalg.norm(received))
        return received, meta
    module.analog = analog
    return module


def fusion_timing(clock, compute, radio):
    result = clock.transaction_timing(compute, radio)
    counts = np.zeros(10)
    counts[radio['head']] = radio['head_MAC_equivalents']
    energies, seconds = parent.learning.quality.infra.PHYSICAL.compute_cost(counts)
    result['head_fusion_s'] = float(seconds.max())
    result['head_fusion_energy_j'] = float(energies.sum())
    result['duration_s'] += result['head_fusion_s']
    result['policy'] = 'Local compute, uplink, selected-head fusion, then broadcast.'
    radio['head_fusion_energy_j'] = result['head_fusion_energy_j']
    radio['head_fusion_s'] = result['head_fusion_s']
    return result


def run(family, mode):
    global FAMILY, MODE
    FAMILY, MODE = family, mode
    protocol = register()
    assert not (family == 'current' and mode in ['combined_sum', 'iq_sum']), 'Reuse frozen controls.'
    parent.OUT = OUT/family
    parent.METHODS = METHODS
    parent.register = lambda: protocol
    parent.vectors = vectors
    parent.fu_objective = objective
    parent.radio_module = radio_module
    q = parent.learning.quality
    if family == 'manuscript':
        parent.CFG = parent.learning.spec(late_uniform=0)
        parent.learning.request_gradient = manuscript_gradient
        parent.learning.fl_vectors = q.infra.cached_clone(parent.learning.fl_vectors,
            [('len(pool)/total', 'len(batch)/sum(min(16,len(p)) for p in pools)', 1)], {},
            live_globals=parent.learning.__dict__)
    old_make = q.make_arm

    def make(base, deployment, radios, cap):
        fn = old_make(base, deployment, radios, cap)
        return q.infra.cached_clone(fn, [
            ('clock.transaction_timing(compute,rm)', 'fusion_timing(clock,compute,rm)', 1),
            ("start+timing['compute_s']+timing['uplink_s']", "start+timing['compute_s']+timing['uplink_s']+timing['head_fusion_s']", 2)],
            dict(fusion_timing=fusion_timing))
    q.make_arm = make
    parent.run(mode)


def queue():
    register()
    for family in ['current', 'manuscript']:
        for mode in METHODS:
            if family == 'current' and mode in ['combined_sum', 'iq_sum']:
                continue
            dest = OUT/family/mode
            if (dest/'exit.json').exists():
                continue
            dest.mkdir(parents=True, exist_ok=True)
            with (dest/'console.log').open('a') as log:
                status = subprocess.run([sys.executable, __file__, 'run', '--family', family, '--method', mode],
                    stdout=log, stderr=subprocess.STDOUT)
            write(dest/'exit.json', dict(returncode=status.returncode, unix=time.time()))
            print(family, mode, 'exit', status.returncode, flush=True)
    write(OUT/'queue_finished.json', dict(unix=time.time()))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['register', 'run', 'queue'])
    p.add_argument('--family', choices=['current', 'manuscript'])
    p.add_argument('--method', choices=list(METHODS))
    a = p.parse_args()
    if a.action == 'register':
        register()
    elif a.action == 'queue':
        queue()
    else:
        run(a.family, a.method)
