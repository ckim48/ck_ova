"""Paired training/random-noise seeds, with frozen data/request/topology policy.

No running or historical source is edited. The adapters below change only
explicit RNG seeds in private, per-process function namespaces.
"""
from pathlib import Path
import os, sys, json, time, inspect, textwrap, importlib.util, traceback
import numpy as np
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / 'results/overnight_energy_v2/three_seed_cap100_v1'

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

base = load('seed_parent', ROOT / 'tools/overnight_energy_v2/rsicd_cap100_v1/campaign.py')
DATASETS = {'RSITMD': 'retrieval_long200_v3', 'RSICD': 'rsicd_cap100_v1'}
RUNS = [(dataset, seed) for seed in [1, 2] for dataset in DATASETS]

def write(path, value):
    base.core.write(path, value)

def clone_function(fn, edits, extra=None, live_globals=None):
    source = textwrap.dedent(inspect.getsource(fn))
    for old, new, count in edits:
        assert source.count(old) == count, (fn.__name__, old)
        source = source.replace(old, new)
    namespace = live_globals if live_globals is not None else dict(fn.__globals__)
    if extra:
        namespace.update(extra)
    exec(compile(source, str(HERE / 'repeat.py') + ':' + fn.__name__, 'exec'), namespace)
    return namespace[fn.__name__]

def seed_rng(seed, offset):
    assert isinstance(seed, (int, np.integer)), type(seed)
    return np.random.default_rng(int(seed) + offset)

def register():
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / 'protocol.json'
    if p.exists():
        verify_registration()
        return
    # Verify that the only supported frozen source expressions are changed.
    # This is CPU-only; it does not initialize another GPU model.
    import train
    clone_function(train.Engine.__init__,
        [('torch.manual_seed(0)', "torch.manual_seed(plan['seed'])", 1),
         ('torch.cuda.manual_seed_all(0)', "torch.cuda.manual_seed_all(plan['seed'])", 1)])
    for fn in [base.clip.vectors, base.clip.ORIGINAL_VECTORS]:
        clone_function(fn, [('np.random.default_rng(', 'seeded_rng(', 2)],
                       {'seeded_rng': lambda key: seed_rng(key, 10000000)})
    clone_function(base.core.proxy_schedule, [('np.random.default_rng(', 'seeded_rng(', 1)],
                   {'seeded_rng': lambda key: seed_rng(key, 10000000)})
    clone_function(base.pure_trajectory, [('else 100000)+k', 'else 100000)+k+SEED_OFFSET', 1)], {'SEED_OFFSET': 10000000})
    clone_function(base.runtime.run_arm, [('seed=500000+seq', 'seed=500000+seq+SEED_OFFSET', 1)], {'SEED_OFFSET': 10000000})
    hashes = dict(json.loads((base.OUT / 'protocol.json').read_text())['source_hashes'])
    for path in HERE.glob('*.py'):
        hashes[str(path.relative_to(ROOT))] = base.core.sha(path)
    for dataset, seed in RUNS:
        source = ROOT / 'results/overnight_energy_v2' / DATASETS[dataset]
        plan = json.loads((source / 'plan.json').read_text())
        records = json.loads((source / 'records.json').read_text())
        assert len(records) == 200 and len(plan['partitions']) == 10
        reqpath = (source if dataset == 'RSICD' else ROOT / 'results/overnight_energy_v2/fourway_cap100_v1') / 'requests.json'
        requests = json.loads(reqpath.read_text())
        assert len(requests) == 45
        assert set(i for r in requests for i in r['images']) == set(plan['forgotten_images'])
        seen = [set() for _ in range(10)]
        for rec in records[:20]:
            for u, batch in enumerate(rec['batches']):
                seen[u].update(batch)
        assert all(set(r['images']) <= seen[r['user']] for r in requests)
        for path in [source / 'plan.json', source / 'records.json', source / 'channels.npy', reqpath]:
            hashes[str(path.relative_to(ROOT))] = base.core.sha(path)
        # Actual batch selections: seed0 parity, paired-method identity, and
        # seed diversity on every UAV's unchanged retained training pool.
        checks = []
        for u, pool in enumerate(records[0]['retained']):
            key = 730000 + 21 * 10 + u
            original = np.random.default_rng(key).choice(pool, min(16, len(pool)), replace=False)
            zero = seed_rng(key, 0).choice(pool, min(16, len(pool)), replace=False)
            a = seed_rng(key, seed * 10000000).choice(pool, min(16, len(pool)), replace=False)
            b = seed_rng(key, seed * 10000000).choice(pool, min(16, len(pool)), replace=False)
            assert np.array_equal(original, zero) and np.array_equal(a, b)
            assert not np.array_equal(a, zero) and set(a) <= set(pool)
            checks.append(dict(user=u, paired_equal=True, different_from_seed0=True))
        write(OUT / f'{dataset.lower()}_seed{seed}/registration_checks.json',
              dict(passed=True, seed=seed, dataset=dataset, batch_checks=checks,
                   fixed_warmup_preserves_requested_images_seen=True,
                   source_plan_sha256=base.core.sha(source / 'plan.json')))
    protocol = dict(version='three_seed_cap100_v1', registered_unix=time.time(),
        authorization='User requested three seeds instead of moving-average curves, 2026-10-24.',
        runs=[dict(dataset=d, seed=s) for d, s in RUNS],
        seed0='Reuse completed RSITMD fourway/sequential0.10J and current RSICD rsicd_cap100_v1. No rerun or replacement of seed0.',
        seed_scope='Training/prompt-initialization and sampled channel-noise variability conditional on fixed data partition, validation/gallery, request content/arrivals, and physical channels. Not a random-partition/topology robustness study.',
        changed_rng='Torch initialization seed=1/2; adaptation FL/FU-retention minibatch, safe-proxy minibatch, retained-reference minibatch and analog-noise seed offsets=seed*10000000. Same seeds paired across the four methods.',
        warmup='Keep the original first20 data batches so every requested sample remains previously learned. Rebuild model/proxy/reference independently for each seed; no copied trained checkpoint.',
        invariant='All four methods, losses, schedulers,200FL,45requests,5FUsteps,Q6 digital,128tones,1MHz,0.10J TX cap,0.32W power,head/radio solver/quality budgets unchanged.',
        evaluation='Mean and sample SD(ddof=1) over seeds0,1,2 at matched actual evaluation rounds. No moving-average smoothing. Per-seed raw target crossings before aggregation; never compute target from the mean curve. Keep failures/missing targets explicit.',
        cost='All initial20/proxy20 and subsequent uploads charged. TX is supplementary; total communication+computation energy remains pending hardware calibration. Pure uplink is not end-to-end time.',
        dependency=dict(pid=243150,start_ticks='884549997',argv_suffix='rsicd_cap100_v1/campaign.py'),
        execution='Wait for the registered RSICD seed0 campaign to finish, then four dataset/seed campaigns serially under the existing global GPU lock. Do not resume old campaigns or tune on results.',
        source_hashes=hashes)
    write(p, protocol)
    write(OUT / 'queue_registered.json', dict(status='registered', runs=protocol['runs'], unix=time.time()))

def verify_registration():
    p = json.loads((OUT / 'protocol.json').read_text())
    for name, value in p['source_hashes'].items():
        assert base.core.sha(ROOT / name) == value, name
    return p

def prepare_child(dataset, seed):
    protocol = verify_registration()
    dest = OUT / f'{dataset.lower()}_seed{seed}'
    source = ROOT / 'results/overnight_energy_v2' / DATASETS[dataset]
    plan = json.loads((source / 'plan.json').read_text())
    plan.update(seed=seed, version='three_seed_cap100_v1', dataset=dataset)
    plan['config'] = json.loads((base.parent.OUT / 'preflight.json').read_text())['config']
    records = json.loads((source / 'records.json').read_text())
    reqpath = (source if dataset == 'RSICD' else ROOT / 'results/overnight_energy_v2/fourway_cap100_v1') / 'requests.json'
    requests = json.loads(reqpath.read_text())
    refs = []
    forgotten = set(plan['forgotten_images'])
    for rec in records:
        pools = [sorted(set(p) - forgotten) for p in rec['retained']]
        batches = [seed_rng(key, seed * 10000000).choice(pool, min(16, len(pool)), replace=False).astype(int).tolist()
                   for key, pool in zip(rec['batch_seeds'], pools)]
        assert all(not (set(b) & forgotten) for b in batches)
        refs.append(dict(round=rec['round'], retained=pools, batches=batches))
    assert not (dest / 'protocol.json').exists(), 'Refuse duplicate child preparation'
    for name, value in [('plan.json', plan), ('records.json', records), ('requests.json', requests), ('retrain_records.json', refs)]:
        write(dest / name, value)
    np.save(dest / 'channels.npy', np.load(source / 'channels.npy'))
    hashes = dict(protocol['source_hashes'])
    for name in ['plan.json', 'records.json', 'requests.json', 'retrain_records.json', 'channels.npy']:
        hashes[str((dest / name).relative_to(ROOT))] = base.core.sha(dest / name)
    write(dest / 'protocol.json', dict(dataset=dataset, seed=seed, source_hashes=hashes,
          seed_policy=protocol['seed_scope'], inherited_protocol=str((OUT / 'protocol.json').relative_to(ROOT))))
    inherited = ROOT / 'results/overnight_energy_v2/rsicd_cap100_v1/cpu_preflight.json'
    assert json.loads(inherited.read_text())['passed']
    write(dest / 'cpu_preflight.json', dict(passed=True, inherited_physical_checks=str(inherited.relative_to(ROOT)),
          inherited_sha256=base.core.sha(inherited), randomization_checks='registration_checks.json',
          actual_new_seed_update_checks='Run after the fresh20+20 models, before reference and ranked arms.'))
    return dest, plan, requests

def run(dataset, seed):
    dest, plan, requests = prepare_child(dataset, seed)
    offset = seed * 10000000
    base.OUT = dest
    base.prepare = lambda: (plan, requests)
    if dataset == 'RSITMD':
        from dataset import RetrievalData
        base.RetrievalData = RetrievalData
    import train
    init = clone_function(train.Engine.__init__,
        [('torch.manual_seed(0)', "torch.manual_seed(plan['seed'])", 1),
         ('torch.cuda.manual_seed_all(0)', "torch.cuda.manual_seed_all(plan['seed'])", 1)])
    train.Engine = type('SeededEngine', (train.Engine,), {'__init__': init})
    rng = lambda key: seed_rng(key, offset)
    for name in ['vectors', 'ORIGINAL_VECTORS']:
        fn = getattr(base.clip, name)
        setattr(base.clip, name, clone_function(fn, [('np.random.default_rng(', 'seeded_rng(', 2)], {'seeded_rng': rng}))
    base.core.proxy_schedule = clone_function(base.core.proxy_schedule,
        [('np.random.default_rng(', 'seeded_rng(', 1)], {'seeded_rng': rng})
    base.pure_trajectory = clone_function(base.pure_trajectory,
        [('else 100000)+k', 'else 100000)+k+SEED_OFFSET', 1)],
        {'SEED_OFFSET': offset}, live_globals=base.__dict__)
    # The radio is unchanged; only the sampling seed at the frozen runtime's
    # call boundary changes. Digital OFDMA packet values remain deterministic.
    base.runtime.run_arm = clone_function(base.runtime.run_arm,
        [('seed=500000+seq', 'seed=500000+seq+SEED_OFFSET', 1)],
        {'SEED_OFFSET': offset}, live_globals=base.runtime.__dict__)
    write(dest / 'seed_adapter.json', dict(seed=seed, offset=offset,
          modifications=['Torch prompt seed', 'FL/FU-retention minibatch RNG', 'proxy minibatch RNG',
                         'reference minibatches', 'warmup/proxy/adaptation analog-noise RNG'],
          fixed=['partitions', 'validation/gallery', 'request contents and times', 'first20 warmup data batches', 'physical channel matrix']))
    base.main()

if __name__ == '__main__':
    error_dest = OUT
    try:
        if '--register' in sys.argv:
            register()
            print('REPEAT_REGISTRATION_PASSED', flush=True)
        else:
            import argparse
            p = argparse.ArgumentParser()
            p.add_argument('--dataset', choices=list(DATASETS), required=True)
            p.add_argument('--seed', type=int, choices=[1, 2], required=True)
            a = p.parse_args()
            error_dest = OUT / f'{a.dataset.lower()}_seed{a.seed}'
            run(a.dataset, a.seed)
    except Exception as exc:
        write(error_dest / 'status.json', dict(stage='failed', error=repr(exc), unix=time.time()))
        write(error_dest / 'failure.json', dict(error=repr(exc), traceback=traceback.format_exc()))
        raise
    finally:
        base.parent.common.base.close()
