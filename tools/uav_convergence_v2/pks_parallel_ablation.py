"""Controlled parallel-versus-immediate-serial scheduling of unchanged P-KS.

Both modes retain the same learning objectives, sampler, radio optimization,
head selection, data and physical costs. No artificial periodic waiting is added.
The zero-change parallel mode must reproduce the published P-KS exactly.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

import learning_quality_v3 as study
import learning_selection_fix_v2 as selection
import learning_quality_campaign as campaign

ROOT = study.ROOT
OUT = ROOT / 'results/pks_parallel_ablation_v1_20261028'
ORIGINAL_OPTIONS = study.options
MODE = None
read, write, sha = study.read, study.write, study.sha


def register():
    path = OUT / 'protocol.json'
    if path.exists():
        value = read(path)
        for name, digest in value['source_sha256'].items():
            assert sha(ROOT / name) == digest, name
        return value
    parent = selection.register()
    sources = dict(parent['source_sha256'])
    sources[str(Path(__file__).relative_to(ROOT))] = sha(Path(__file__))
    result = dict(registered_unix=time.time(), modes=['parallel', 'serial'],
                  datasets=['CapERA', 'AirScape'], seed=0, new_arms=4,
                  algorithm='Unchanged P-KS: proxy-KL-gated FU, causal priority FL16, sample-weighted aggregation, joint-family optimized AirComp/head selection.',
                  intervention='Only options.parallel changes. Serial mode uses period=0: pending FU is served immediately without FL in that transaction. All participating FU clients still aggregate over AirComp. Parallel mode combines FL and pending FU in the same update. Same radio solver is run on each actual update; optimized values and learned parameters may differ as consequences of the scheduling change.',
                  fixed='Identical warm20, feature cache, splits, 45 requests, five regions, 200 total FL rounds, 225 request-local FU steps, learning-rate rules, proxy/teacher construction, radio bounds and physical coefficients.',
                  clocks='Same full compute/uplink/broadcast clock and completion-driven departure. No claim that two local gradient evaluations execute simultaneously on one UAV. Parallelism here is FL/FU co-aggregation and application in one service transaction.',
                  evaluation='Retain all outcomes and failures. The parallel control must match the original P-KS final model, full quality output, transaction order, radio costs and physical timestamps before publication. Serial quality is measured independently, never copied.',
                  scope='Single development seed and round-triggered deletion releases. Requests are tied to completed FL ordinals, not common wall-clock arrival times. This is a scheduling ablation, not a Poisson-arrival throughput benchmark or a deletion certificate.',
                  publication='Code, figures, aggregate results and provenance only; raw samples/checkpoints/execution logs stay local.',
                  source_sha256=sources)
    write(path, result)
    return result


def options(base, method):
    value = dict(ORIGINAL_OPTIONS(base, method))
    assert value['parallel'] and value['period'] == 0 and not value['digital']
    value['parallel'] = MODE == 'parallel'
    value['period'] = 0
    return value


def source(mode, dataset):
    return OUT / mode / 'gate_select16/cases/baseline' / (dataset.lower() + '_seed0') / study.PROPOSED


def verify():
    import torch
    results = []
    for ds in ['CapERA', 'AirScape']:
        original = selection.OUT / 'gate_select16/cases/baseline' / (ds.lower() + '_seed0') / study.PROPOSED
        control = source('parallel', ds)
        assert torch.equal(torch.load(original / 'final.pt', weights_only=True)['theta'],
                           torch.load(control / 'final.pt', weights_only=True)['theta'])
        assert read(original / 'service_completion.json')['evaluation'] == read(control / 'service_completion.json')['evaluation']
        before = [read(p) for p in sorted((original / 'transactions').glob('*.json'))]
        after = [read(p) for p in sorted((control / 'transactions').glob('*.json'))]
        assert len(before) == len(after)
        for a, b in zip(before, after):
            for key in ['kind', 'FU_request_ids', 'start_s', 'end_s', 'timing', 'completed_FL_rounds']:
                assert a[key] == b[key], (ds, key)
            for key in ['energy_bound_j', 'uplink_s', 'head', 'conditional_rmse']:
                assert a['radio'][key] == b['radio'][key], (ds, key)
        for mode in ['parallel', 'serial']:
            arm = source(mode, ds)
            assert read(arm / 'outcome.json')['status'] == 'completed'
            check = read(arm / 'verification.json')
            assert check['passed'] and check['FL_rounds'] == 200 and check['FU_steps'] == 225
            for name in ['plan.json', 'requests.json', 'initial20/final.pt', 'reference200/final.pt', 'feature_cache/features.pt']:
                assert sha(arm.parent / name) == sha(original.parent / name), (mode, ds, name)
            assert read(arm.parents[1] / 'settings.json')['energy_model'] == read(original.parents[1] / 'settings.json')['energy_model']
            tx = [read(p) for p in sorted((arm / 'transactions').glob('*.json'))]
            works = [read(p) for p in (arm / 'work_attempts').glob('*.json')]
            assert sum(t['kind'] in ['fl', 'joint'] for t in tx) == 180
            assert sum(len(t['FU_request_ids']) for t in tx) == 225
            if mode == 'parallel':
                assert all(t['kind'] != 'fu' for t in tx)
                assert all(w['FL_clients_apply_s'] == w['FU_clients_apply_s'] for w in works if w['kind'] == 'joint')
            else:
                assert all(t['kind'] in ['fl', 'fu'] for t in tx)
                assert sum(t['kind'] == 'fu' for t in tx) == 45
            results.append(dict(dataset=ds, mode=mode, passed=True, transactions=len(tx),
                                joint_transactions=sum(t['kind'] == 'joint' for t in tx),
                                zero_change_control_exact=mode == 'parallel'))
    result = dict(passed=True, original_PKS_reproduced_exactly=True, rows=results)
    write(OUT / 'verification.json', result)
    return result


def child(mode):
    global MODE
    assert int(os.environ['OVA_PKS_PARALLEL_PARENT']) == os.getppid()
    original_lock = fcntl.flock

    def lock(handle, flag):
        if getattr(handle, 'name', None) == str(campaign.LOCK):
            assert flag == fcntl.LOCK_EX
            return
        return original_lock(handle, flag)

    fcntl.flock = lock
    MODE = mode
    study.OUT = OUT / mode
    study.register = register
    study.vectors = selection.vectors
    study.options = options
    study.run('gate_select16', 'baseline')


def run():
    register()

    def launch(mode):
        path = OUT / 'logs' / (mode + '.log')
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a') as log:
            done = subprocess.run([sys.executable, str(Path(__file__)), 'child', '--mode', mode],
                                  cwd=ROOT, env=dict(os.environ, OVA_PKS_PARALLEL_PARENT=str(os.getpid())),
                                  stdout=log, stderr=subprocess.STDOUT)
        return dict(mode=mode, returncode=done.returncode)

    rows = []
    with campaign.LOCK.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with ThreadPoolExecutor(max_workers=2) as pool:
            for future in as_completed([pool.submit(launch, mode) for mode in ['parallel', 'serial']]):
                row = future.result()
                rows.append(row)
                print(row, flush=True)
    write(OUT / 'execution.json', dict(rows=rows, completed=all(r['returncode'] == 0 for r in rows)))
    assert all(r['returncode'] == 0 for r in rows), rows
    print(verify(), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['run', 'child', 'verify'])
    parser.add_argument('--mode', choices=['parallel', 'serial'])
    args = parser.parse_args()
    if args.action == 'child':
        child(args.mode)
    elif args.action == 'verify':
        print(verify())
    else:
        run()
