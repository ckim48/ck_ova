"""Paired transport-only experiment; preserve the registered learning policy."""
import argparse
import importlib.util
from pathlib import Path
import time
import numpy as np
import joint_quality_control_v1 as learning
import joint_quality_physical_v2 as physical
import manuscript_iq_study_v1 as curvature
import manuscript_iq_radio_v1 as iq

ROOT, read, write, sha = learning.ROOT, learning.read, learning.write, learning.sha
OUT = ROOT / 'results/combined_iq_experiment_a_20261030'
METHODS = {'combined': 'Combined AirComp', 'iq': 'I/Q AirComp'}
CURRENT_ELL = None
CFG = learning.spec(late_uniform=100, retention=.95)


def register():
    p = OUT / 'protocol.json'
    if p.exists():
        result = read(p)
        for name, digest in result['source_sha256'].items():
            assert sha(ROOT / name) == digest, name
        return result
    files = [__file__, learning.__file__, physical.__file__, curvature.__file__, iq.__file__]
    result = dict(registered_unix=time.time(), methods=METHODS, datasets=['CapERA', 'AirScape'],
        seed=0, learning_configuration=CFG,
        source_sha256={str(Path(f).relative_to(ROOT)): sha(Path(f)) for f in files},
        fixed='Same warm20 checkpoint, native features, data partitions, minibatch seeds, FL/FU objectives, LR, 200 FL rounds, 45 requests x 5 steps, five-region channels, bandwidth, head candidates, expansion grid, power0.32W and network RF0.1J. No outcome-dependent tuning.',
        shared_control='Both arms estimate the same empirical task curvatures using identical HVP procedure and charge its executed computation. Both use the same joint-update MSE budget epsilon_L^2+epsilon_U^2 and task-distortion limits, head enumeration and BCD solver.',
        combined='One real AirComp stream transmits g_L+g_U. Its per-device block energy uses ||g_L+g_U||^2, including the true cross term. One receiver noise stream. No invented separate FL/FU error measurements.',
        iq='Two real updates occupy I and Q of one complex AirComp stream. Joint device block power and RF cap include BOTH branches. In addition to the common total-update constraints, individual component-MSE limits apply.',
        tolerances='epsilon_a=max(1e-8,0.2*B_a) for active tasks. 2*delta_t=0.5*(epsilon_L^2+epsilon_U^2) after mean-normalizing empirical curvature, inherited from the prior manuscript pilot; not tuned for this comparison.',
        accounting='Active FL/FU computation, uplink and broadcasts, circuits and hovering; fixed travel energy excluded. Same model for both. Head-optimizer wall time logged, not converted to hardware energy.',
        evaluation='Report retained mR/loss, common-reference and policy-matched forget-set KL, total energy and active delay. Same uniform_after100 retained-only matched reference for both; neither evaluation reference is a training teacher.',
        limits='One paired development seed. Empirical HVP curvature is not a certified bound. I/Q separation alone need not improve learning: both finally apply the sum. This isolates radio representation/controllability, not simultaneous versus serial scheduling. Both arms are parallel. No forgetting guarantee or forced winning outcome.')
    write(p, result)
    return result


def radio_module(mode):
    spec = importlib.util.spec_from_file_location('experiment_a_radio_' + mode, iq.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if mode == 'combined':
        original = module.inputs

        def combined_inputs(l, u, c, ell):
            _, eps, weights, targets = original(l, u, c, ell)
            rho = np.zeros((2, len(l), l.shape[1] // c['block_dimension']))
            rho[0] = np.linalg.norm((l+u).reshape(len(l), -1, c['block_dimension']), axis=-1)
            return rho, np.array([np.linalg.norm(eps), 0.]), weights, targets

        module.inputs = combined_inputs
    return module


def fu_objective(engine, theta, fids, rids, step, teacher):
    import torch
    fcids = [c for i in fids for c in engine.data.caption_ids[i]]
    rcids = engine.data.training_caption_ids(rids, 21+step)
    im, tx, scale = engine.features(theta, fids, fcids)
    ri, rt, scale = engine.features(theta, rids, rcids)
    kl = learning.quality.infra.lib.adaptive.retrieval_kl
    fk = kl(im, tx, *engine.joint_proxy.features(fids, fcids), scale)
    ck = kl(ri, rt, *teacher.features(rids, rcids), scale)
    return 3*torch.relu(fk-CFG['threshold']) + CFG['retention']*curvature.symmetric_ce(ri, rt, scale) + CFG['consistency']*ck


ORIGINAL_VECTORS = learning.vectors


def vectors(engine, theta, plan, queue, t, fl, kind, ids, proxy, teachers, crossmodal):
    engine.joint_base = JOINT_BASE
    engine.joint_rng_fl = lambda k: np.random.default_rng(k)
    engine.joint_rng_fu = lambda k: np.random.default_rng(k)
    l, u, stats = CALL_VECTORS(engine, theta, plan, queue, t, fl, kind, ids, proxy, teachers, crossmodal)
    engine.iq_batches = [(r['user'], r['batch'], r['round']) for r in engine.joint_fl_records] if kind in ['fl', 'joint'] else []
    curvature.complete_vectors(engine, theta, plan, queue, t, fl, kind, ids, teachers, l, u, stats)
    global CURRENT_ELL
    CURRENT_ELL = curvature.CURRENT_ELL
    return l, u, stats


def run(mode):
    protocol = register()
    q = learning.quality
    learning.VARIANTS = {mode: CFG}
    learning.register = lambda: protocol
    learning.set_variant(mode)
    q.register = lambda: protocol
    q.OUT = OUT
    q.VARIANTS = {mode: dict(label=METHODS[mode], objective='gaft', selection=16, equal=False, exact=False)}
    q.cases = lambda: [dict(id='baseline', axis=None, value=None)]
    curvature.ACTIVE = 'iq_only'
    curvature.fu_loss = fu_objective
    learning.fl_vectors = q.infra.cached_clone(learning.fl_vectors,
        [('np.random.default_rng(', 'engine.joint_rng_fl(', 1)], {}, live_globals=learning.__dict__)
    call_vectors = q.infra.cached_clone(ORIGINAL_VECTORS,
        [('np.random.default_rng(', 'engine.joint_rng_fu(', 1)], {}, live_globals=learning.__dict__)

    def factory(base, objective):
        if objective != 'cliperase':
            return base.control.synchronized_vectors(objective)
        return q.infra.cached_clone(vectors, [], dict(JOINT_BASE=base, CALL_VECTORS=call_vectors))

    q.vectors = factory
    parent_wrapper = q.engine_wrapper

    def wrapper(parent):
        class Engine(parent_wrapper(parent)):
            def finish_work(self):
                result = super().finish_work()
                result['manuscript_curvature'] = self.iq_curvature
                return result
        return Engine

    q.engine_wrapper = wrapper
    original_make = q.make_arm
    module = radio_module(mode)

    def make(base, deployment, radios, cap):
        class Radio:
            @staticmethod
            def analog(l, u, channels, c, seed):
                received, meta = module.analog(l, u, channels, c, seed, curvature.CURRENT_ELL)
                meta['experiment_A_mode'] = mode
                meta['physical_mode'] = 'combined_real_AirComp' if mode == 'combined' else 'parallel_IQ_AirComp'
                return received, meta
        base.base.parent.guard = Radio
        return original_make(base, deployment, radios, cap)

    q.make_arm = make
    dest = OUT / mode
    assert not (dest / 'started.json').exists(), 'Preserve all attempted runs.'
    write(dest / 'started.json', dict(unix=time.time(), method=mode))
    q.run(mode, 'baseline')
    write(dest / 'finished.json', dict(unix=time.time(), method=mode, completed=True))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['register', 'run'])
    p.add_argument('--method', choices=list(METHODS))
    a = p.parse_args()
    if a.action == 'register':
        print(register())
    else:
        run(a.method)
