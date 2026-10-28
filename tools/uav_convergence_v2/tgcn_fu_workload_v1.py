"""Matched FU-step sensitivity of the four currently published methods.

New wrapper only: preserve all registered training implementations and attempts.
The five-step controls must exactly reproduce the current baseline.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

import downlink_bandwidth_v1 as inherited

study, selection, infra = inherited.study, inherited.selection, inherited.infra
ROOT, read, write, sha = inherited.ROOT, inherited.read, inherited.write, inherited.sha
OUT = ROOT / 'results/tgcn_fu_workload_v1_20261028'
VALUES = [1, 3, 5, 7, 9]
METHODS, DATASETS = inherited.METHODS, inherited.MODEL_DATASETS
ORIGINAL_VALIDATE = infra.validate


def case_id(value):
    return f'fu_steps_{value}'


def cases():
    return [dict(id=case_id(v), axis='fu_steps', value=v) for v in VALUES]


def source(value, dataset, method):
    group = method if method in ['gate_select16', 'salun'] else 'papers'
    root = OUT / group
    if group != 'papers':
        root /= method
    internal = {'gate_select16': study.PROPOSED, 'iscc': study.DIGITAL,
                'afl': 'uav_afl_ota_gaft', 'salun': study.DIGITAL}[method]
    return root / 'cases' / case_id(value) / (dataset.lower() + '_seed0') / internal


def register():
    path = OUT / 'protocol.json'
    if path.exists():
        p = read(path)
        for name, digest in p['source_sha256'].items():
            assert sha(ROOT / name) == digest, name
        return p
    parent = selection.register()
    sources = dict(parent['source_sha256'])
    sources[str(Path(__file__).relative_to(ROOT))] = sha(Path(__file__))
    p = dict(registered_unix=time.time(), methods=METHODS, datasets=DATASETS,
             seed=0, values=VALUES, new_arms=40, baseline_steps=5,
             intervention='Only required_steps per deletion request changes to 1/3/5/7/9; 45 unchanged requests, same request images, release FL ordinals, routes, 200 total FL rounds, initial features/model and all original learning/radio rules. This counts outer FU cycles; the GA->FT comparators perform two charged local optimizer substeps per cycle, as before.',
             controls='Eight fresh five-step controls must exactly reproduce the published baseline final parameters, quality and transaction clocks/radio values. All other arms are fresh actual learning executions, not time or energy rescaling.',
             validation='Replace only the inherited validator constants225/5 with the declared45*steps/steps. All workload, power, aggregation-error, energy, clock and joint-application checks remain.',
             costs='Communication + executed adapter computation + active hovering. Travel, IoT collection and preflight feature encoding excluded, as in the existing charts.',
             interpretation='FU work increases but quality need not improve. Show every outcome and all quality diagnostics. Prescribed FU completion is not certified deletion. Same known-cohort proxy assumption, native video captions, single development seed and scoped literature transfers as the existing baseline.',
             publication='Code, charts, aggregate metrics, source hashes only; raw videos, checkpoints and execution logs stay local.',
             source_sha256=sources)
    write(path, p)
    return p


def install_validation(value):
    infra.validate = infra.cached_clone(ORIGINAL_VALIDATE, [
        ("==225", "==45*FU_STEPS", 1),
        ("==5 and math.isclose", "==FU_STEPS and math.isclose", 1),
        ("FU_steps=225", "FU_steps=45*FU_STEPS", 1)],
        dict(FU_STEPS=value), live_globals=infra.__dict__)


def child(group, value):
    assert int(os.environ['OVA_TGCN_FU_PARENT']) == os.getppid()
    original_lock = fcntl.flock
    def lock(handle, mode):
        if getattr(handle, 'name', None) == str(inherited.campaign.LOCK):
            assert mode == fcntl.LOCK_EX
            return
        return original_lock(handle, mode)
    fcntl.flock = lock
    register()
    install_validation(value)
    infra.cases, infra.register = cases, register
    if group == 'papers':
        infra.OUT = OUT / group
        infra.run_case(case_id(value), 'papers')
    else:
        selection.install()
        study.OUT, study.register, study.cases = OUT / group, register, cases
        study.run(group, case_id(value))


def preflight():
    import copy
    import numpy as np
    _, base = infra.lib.infra.prior.load_base()
    base.repeat.clone_function = infra.cached_clone
    import common
    original = read(common.SOURCE / 'capera_seed0/requests.json')
    plan = read(common.SOURCE / 'capera_seed0/plan.json')
    geometry = read(common.GEOMETRY / 'mission_spec.json')
    deployment = read(common.GEOMETRY / 'deployment.json')
    results = []
    for case in cases():
        requests = copy.deepcopy(original)
        infra.configure(case, copy.deepcopy(plan), requests, copy.deepcopy(geometry),
                        copy.deepcopy(deployment), {}, np.ones((5, 10, 10)))
        assert all(r['required_steps'] == case['value'] for r in requests)
        check = infra.replay(infra.queues(base, 1e6), requests, geometry, deployment, plan['config'])
        assert all(r['FU_cycles'] == 45*case['value'] and r['FL_rounds'] == 200 for r in check)
        install_validation(case['value'])
        results.append(dict(value=case['value'], schedules=check))
    write(OUT / 'preflight.json', dict(passed=True, rows=results))
    print(dict(preflight=True, values=VALUES), flush=True)


def verify_controls():
    import torch
    import proposed_pks_report as current
    rows = []
    for ds in DATASETS:
        for m in METHODS:
            old, new = current.source('baseline', ds, m), source(5, ds, m)
            assert torch.equal(torch.load(old/'final.pt', weights_only=True)['theta'],
                               torch.load(new/'final.pt', weights_only=True)['theta'])
            assert read(old/'service_completion.json')['evaluation'] == read(new/'service_completion.json')['evaluation']
            before = [read(f) for f in sorted((old/'transactions').glob('*.json'))]
            after = [read(f) for f in sorted((new/'transactions').glob('*.json'))]
            assert len(before) == len(after)
            for a, b in zip(before, after):
                for key in ['kind', 'FU_request_ids', 'start_s', 'end_s', 'timing', 'completed_FL_rounds']:
                    assert a[key] == b[key], (ds, m, key)
                for key in ['energy_bound_j', 'uplink_s', 'head', 'conditional_rmse']:
                    assert a['radio'].get(key) == b['radio'].get(key), (ds, m, key)
            rows.append(dict(dataset=ds, method=m, exact_model_quality_clock_radio=True))
    write(OUT/'control_checks.json', dict(passed=True, rows=rows))


def run():
    preflight()
    register()
    # Baseline controls run first, followed by the four workload interventions.
    items = [(g, v) for v in [5, 1, 3, 7, 9] for g in ['gate_select16', 'salun', 'papers']]
    for g, v in items:
        methods = ['iscc', 'afl'] if g == 'papers' else [g]
        assert not any(source(v, ds, m).exists() for ds in DATASETS for m in methods), 'Preserve previous attempts.'
    def launch(g, v):
        path = OUT / 'logs' / f'{g}_{v}.log'
        path.parent.mkdir(exist_ok=True)
        with path.open('x') as log:
            result = subprocess.run([sys.executable, str(Path(__file__)), 'child', '--group', g, '--value', str(v)],
                cwd=ROOT, env=dict(os.environ, OVA_TGCN_FU_PARENT=str(os.getpid())), stdout=log, stderr=subprocess.STDOUT)
        return dict(group=g, steps=v, returncode=result.returncode)
    rows = []
    with inherited.campaign.LOCK.open('a') as gpu:
        fcntl.flock(gpu, fcntl.LOCK_EX)
        with ThreadPoolExecutor(max_workers=3) as pool:
            for future in as_completed([pool.submit(launch, g, v) for g, v in items]):
                row = future.result()
                rows.append(row)
                write(OUT/'campaign_status.json', dict(rows=rows, total_jobs=len(items)))
                print(row, flush=True)
    passed = all(r['returncode'] == 0 for r in rows)
    write(OUT/'campaign_finished.json', dict(rows=rows, passed=passed))
    assert passed, rows
    verify_controls()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['run', 'child', 'preflight', 'verify'])
    parser.add_argument('--group', choices=['gate_select16', 'salun', 'papers'])
    parser.add_argument('--value', type=int, choices=VALUES)
    args = parser.parse_args()
    if args.action == 'child':
        child(args.group, args.value)
    elif args.action == 'preflight':
        preflight()
    elif args.action == 'verify':
        verify_controls()
    else:
        run()
