"""Bounded development search before radio runs; all outcomes retained."""
import argparse
from pathlib import Path
from types import SimpleNamespace
import time
import numpy as np
import head_fusion_study_v1 as previous
import adaptive_iq_math_v1 as controller

parent = previous.parent
learning, audit = parent.learning, parent.learning.audit
ROOT, read, write, sha = parent.ROOT, parent.read, parent.write, parent.sha
OUT = ROOT/'results/adaptive_iq_pilot_v1_20261030'
MODES = ['combined', 'combined_reset', 'iq_separate']


def register():
    path = OUT/'protocol.json'
    sources = [__file__, controller.__file__, previous.__file__, learning.__file__]
    if path.exists():
        p = read(path)
        for f, h in p['source_sha256'].items():
            assert sha(ROOT/f) == h, f
        return p
    p = dict(registered_unix=time.time(), settings=controller.SETTINGS, modes=MODES,
        source_sha256={str(Path(f).relative_to(ROOT)): sha(Path(f)) for f in sources},
        design='Four predeclared settings x three policies x two datasets, seed0. All24 complete200-round exact-aggregation development runs and8 policy-matched retained-only retrains are reported. No condition-specific parameter choice.',
        learning='Supplied manuscript equations: uniform16 FL, actual minibatch-size weights;3 positive-pair cosine +1 retained CE+3 embedding-softmax request-start consistency KL. Same warm20, request timing,45 requests/225 FU steps. No future-deletion proxy is used by this objective.',
        head='RMS-type second-moment adaptation, group128 or coordinate groups; predeclared beta/blend pairs. FL and FU have separate states only for iq_separate. FU state resets when request cohort changes. combined has one shared state; combined_reset also resets its state when the request cohort changes. Same per-round total update norm, no larger step budget.',
        selection='Development mR and common/matched KL are already-inspected cohort metrics, NOT a hidden test. For each setting compare I/Q to BOTH Combined policies on BOTH datasets. Joint success requires mR no lower and both KLs no larger everywhere, with at least one strict gain. Select max worst normalized margin, tie-break by smaller state. If no joint winner, explicitly report that and select the best tradeoff for physical falsification, never call it a win.',
        score='Minimum of mR difference/0.1 percentage point, common KL fractional reduction/0.01, matched KL fractional reduction/0.01, across both datasets and both baselines.',
        confirm='One common selected setting is frozen before physical outcomes. Same-candidate physical comparisons use new minibatch/radio seeds1,2 plus development seed0, same data partition/warm start. These are stochastic repeats conditional on existing cohorts, not independent data validation. Report all three seeds and all baselines.',
        realism='Exact pilot makes no radio/energy claim. Physical follow-up must include noisy observations, actual RF caps, head computation, and digital transfer of float32 optimizer statistics if the aggregation head changes. Reference predictions never enter optimization.',
        limitation='New nonlinear optimizer; original linear-sum zero-mean distortion theorem does not certify post-optimizer output. Literature motivates adaptive OTA optimization but does not establish this FL/FU rule.',
        references=[dict(title='Adaptive Federated Learning Over the Air', venue='IEEE TSP2025', doi='10.1109/TSP.2025.3585002', url='https://arxiv.org/pdf/2403.06528'),
                    dict(title='Adaptive Federated Optimization', venue='ICLR2021', url='https://research.google/pubs/adaptive-federated-optimization/')])
    write(path, p); write(OUT/'math_checks.json', controller.checks())
    return p


def configure():
    learning.CURRENT = learning.spec(late_uniform=0)
    learning.quality.CONFIG = dict(objective='gaft', selection=16, equal=False, exact=False)
    learning.request_gradient = previous.manuscript_gradient
    learning.fl_vectors = learning.quality.infra.cached_clone(learning.fl_vectors,
        [('len(pool)/total', 'len(batch)/sum(min(16,len(p)) for p in pools)', 1)], {}, live_globals=learning.__dict__)


def retrain(ds, setting, seed=0, destination=None):
    """Shared reference: without FU all three policies reduce to identical FL RMS."""
    import torch
    from train import update_theta
    base, data, parent_engine, plan, payload, anchor = audit.runtime(ds)
    dest = destination or OUT/'references'/ds.lower()/setting
    if (dest/'finished.json').exists():
        return dest
    assert not (dest/'started.json').exists(), 'Preserve interrupted references.'
    write(dest/'started.json', dict(unix=time.time(), seed=seed))
    records = read((anchor.parent/'reference200').resolve().parent/'reference_records.json')
    engine = learning.quality.engine_wrapper(parent_engine)(data, plan, payload)
    theta = engine.theta0.clone()
    pools = [sorted(set(p)-set(plan['forgotten_images'])) for p in plan['partitions']]
    ctrl = controller.RMS(controller.SETTINGS[setting], 'combined')
    for k in range(1, 201):
        engine.start_work()
        if k <= 20:
            engine.plan['learning_rate'] = base.core.lr_at(k)
            l, _, _ = engine.updates(theta, records[k-1], k, 'retrain', [[] for _ in range(10)])
            delta = l.sum(0)
        else:
            l = np.zeros((10, len(theta)-1))
            for user, pool in enumerate(pools):
                batch = np.random.default_rng(730000+k*10+user+1000000*seed).choice(pool, 16, replace=False).astype(int).tolist()
                g, _ = engine.gradient(theta, batch, k)
                l[user] = (-base.core.lr_at(k)*.1*g[:-1]).cpu().numpy()
            ctrl.prepare([])
            delta, _ = ctrl.apply(l.sum(0), l.sum(0)*0, np.zeros((2, len(theta)-1)), base.core.lr_at(k))
        theta = update_theta(theta, delta)
        engine.finish_work()
    base.core.save_theta(dest/'final.pt', theta, 200)
    write(dest/'finished.json', dict(seed=seed, rounds=200, checkpoint_sha256=sha(dest/'final.pt'), evaluation_only=True))
    return dest


def run():
    import torch
    from train import update_theta
    register(); configure(); torch.set_num_threads(1)
    assert torch.cuda.is_available()
    for ds in ['CapERA', 'AirScape']:
        base, data, parent_engine, plan, payload, anchor = audit.runtime(ds)
        records = [read(p) for p in sorted((anchor/'transactions').glob('*.json'))]
        requests = read(anchor.parent/'requests.json')
        for setting in controller.SETTINGS:
            refdir = retrain(ds, setting)
            for mode in MODES:
                dest = OUT/'runs'/ds.lower()/setting/mode
                if (dest/'finished.json').exists():
                    continue
                assert not (dest/'started.json').exists(), 'Preserve attempted candidates.'
                write(dest/'started.json', dict(unix=time.time()))
                start = time.time()
                engine = learning.quality.engine_wrapper(parent_engine)(data, plan, payload)
                engine.joint_base = base
                theta = base.core.load_theta(anchor.parent/'initial20/final.pt', engine.device)
                engine.set_teacher(theta)
                queue, teachers = base.RoundQueue(requests), {}
                ctrl = controller.RMS(controller.SETTINGS[setting], mode)
                fl = 0; work = []
                for tx in records:
                    queue.sync(fl, tx['start_s'], 0.)
                    engine.start_work()
                    l, u, local = learning.vectors(engine, theta, plan, queue, tx['start_s'], fl,
                        tx['kind'], tx['FU_request_ids'], None, teachers, False)
                    ctrl.prepare(tx['FU_request_ids'])
                    delta, info = ctrl.apply(l.sum(0), u.sum(0), np.zeros((2, len(theta)-1)), base.core.lr_at(21+fl))
                    theta = update_theta(theta, delta)
                    counted = engine.finish_work()
                    queue.commit(tx['FU_request_ids'], tx['start_s'], tx['end_s'])
                    fl += int(tx['kind'] in ['fl', 'joint'])
                    work.append(dict(round=20+fl, local=local, control=info, device_macs=counted['device_macs']))
                assert fl == 180 and sum(len(w['local']) for w in work) == 225
                base.core.save_theta(dest/'final.pt', theta, 200)
                refs = {label: audit.per_video(engine, base.core.load_theta(path, engine.device), plan)[1]
                    for label, path in dict(common=anchor.parent/'reference200/final.pt', matched=refdir/'final.pt').items()}
                metrics, _ = audit.store_eval(engine, theta, plan, dest, refs)
                result = dict(dataset=ds, setting=setting, mode=mode, mR=100*metrics['retained']['mR'],
                    loss=metrics['retained']['loss'], common_KL=metrics['forget']['common_KL'],
                    matched_KL=metrics['forget']['matched_KL'], wall_s=time.time()-start)
                write(dest/'work.json', work); write(dest/'finished.json', result)
                print(result, flush=True)
    select()


def select():
    rows = [read(p) for p in sorted((OUT/'runs').glob('*/*/*/finished.json'))]
    assert len(rows) == 24
    ranking = []
    for setting, config in controller.SETTINGS.items():
        margins = []; strict = False
        for ds in ['CapERA', 'AirScape']:
            cand = next(r for r in rows if r['setting'] == setting and r['dataset'] == ds and r['mode'] == 'iq_separate')
            for mode in ['combined', 'combined_reset']:
                b = next(r for r in rows if r['setting'] == setting and r['dataset'] == ds and r['mode'] == mode)
                values = [(cand['mR']-b['mR'])/.1,
                    (1-cand['common_KL']/b['common_KL'])/.01,
                    (1-cand['matched_KL']/b['matched_KL'])/.01]
                margins.extend(values); strict |= max(values) > 1e-8
        ranking.append(dict(setting=setting, worst_margin=min(margins), joint_winner=min(margins)>=-1e-8 and strict,
            state_size=20480//config['group']))
    ranking.sort(key=lambda r: (-r['worst_margin'], r['state_size'], r['setting']))
    result = dict(selected=ranking[0]['setting'], rankings=ranking, results=rows,
        interpretation='Development selection only; physical repeats are required. A negative worst margin is an explicit failed joint-win criterion.')
    write(OUT/'selection.json', result)
    print('SELECTION', {k:v for k,v in result.items() if k!='results'}, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['register', 'run', 'select'])
    a = p.parse_args()
    {'register': register, 'run': run, 'select': select}[a.action]()
