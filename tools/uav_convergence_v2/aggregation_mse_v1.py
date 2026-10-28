"""Squared relative aggregation-error budget sweep, with frozen algorithms.

tau = relative_tolerance**2. This is a dimensionless MSE budget coefficient,
not an absolute coordinate MSE. Each method keeps its published norm/guard rule.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import downlink_bandwidth_v1 as inherited

study, selection, infra = inherited.study, inherited.selection, inherited.infra
ROOT, read, write, sha = inherited.ROOT, inherited.read, inherited.write, inherited.sha
OUT = ROOT / 'results/aggregation_mse_v1_20261028'
VALUES = [.01, .025, .04, .055, .07]
BASE = .04
METHODS, DATASETS = inherited.METHODS, inherited.MODEL_DATASETS
ORIGINAL_CONFIGURE = inherited.ORIGINAL_CONFIGURE


def case_id(value):
    return 'mse_' + str(value).replace('.', 'p')


def cases():
    return [dict(id=case_id(v), axis='aggregation_mse', value=v) for v in VALUES]


def configure(case, plan, requests, geometry, deployment, parameters, channels):
    result = ORIGINAL_CONFIGURE(case, plan, requests, geometry, deployment, parameters, channels)
    assert case['axis'] == 'aggregation_mse'
    plan['config']['relative_tolerance'] = math.sqrt(case['value'])
    return result


def register():
    path = OUT / 'protocol.json'
    if path.exists():
        p = read(path)
        for name, digest in p['source_sha256'].items():
            assert sha(ROOT / name) == digest, name
        return p
    parent = inherited.register()
    sources = dict(parent['source_sha256'])
    sources[str(Path(__file__).relative_to(ROOT))] = sha(Path(__file__))
    p = dict(registered_unix=time.time(), methods=METHODS, datasets=DATASETS, seed=0,
             values=VALUES, baseline=BASE, fresh_arms=32, reused_baseline_arms=8,
             intervention='Only config.relative_tolerance = sqrt(tau). Same all-method setting; downlink and uplink fixed at original1MHz/1Mbit/s baseline, same200FL/225FU workload, inputs, learning, head/radio rules and hard power limits. Baseline tau=.04 exactly reproduces coefficient.2 and reuses unchanged baseline arms.',
             definition='tau is the squared relative aggregation-RMSE coefficient. Thus conditional expected squared L2 reconstruction error <= epsilon_k**2, with epsilon_k=max(1e-8,sqrt(tau)*S_k) under the original method budget/guard. This is a dimensionless normalized MSE BUDGET, not an absolute per-coordinate MSE or the measured realized error. The1e-8 RMS floor implies1e-16 squared error floor.',
             normalizers='Proposed keeps the block-norm bound S_L=||sum_u ||g^L_ub||_2||_2 in FL-only updates, and its unchanged FU-relative total-noise guard during FU (minimum of joint and FU-block-norm budgets). AFL-OTA uses ||sum_u update_u||_2 and analytic coherent repetition. ISCC-P2 and F-SalUn keep their fixedFP32 ideal digital transport, with zero modeled reconstruction error; tightening this coefficient need not change their costs. Consequently the x value is a shared relative-budget coefficient, not an equal absolute error budget across methods.',
             integrity='Every nonbaseline condition reruns actual learning/radio/scheduling; no slope injection, noise-realization rejection or rescaling another trajectory. Retain all outcomes, quality tradeoffs and QoS exceedances. Single development seed; no statistical superiority claims.',
             publication='Code, charts, aggregate metrics and provenance only; raw samples, checkpoints and execution logs local.',
             source_sha256=sources)
    assert math.sqrt(BASE) == .2
    write(path, p)
    return p


def source(value, dataset, method):
    if value == BASE:
        import proposed_pks_report as current
        return current.source('baseline', dataset, method)
    group = method if method in ['gate_select16', 'salun'] else 'papers'
    root = OUT / group
    if group != 'papers':
        root = root / method
    internal = {'gate_select16': study.PROPOSED, 'iscc': study.DIGITAL,
                'afl': 'uav_afl_ota_gaft', 'salun': study.DIGITAL}[method]
    return root / 'cases' / case_id(value) / (dataset.lower() + '_seed0') / internal


def child(group, value):
    assert int(os.environ['OVA_MSE_PARENT']) == os.getppid()
    original_lock = fcntl.flock
    def lock(handle, mode):
        if getattr(handle, 'name', None) == str(inherited.campaign.LOCK):
            assert mode == fcntl.LOCK_EX
            return
        return original_lock(handle, mode)
    fcntl.flock = lock
    register()
    infra.configure, infra.cases, infra.register = configure, cases, register
    if group == 'papers':
        infra.OUT = OUT / group
        infra.run_case(case_id(value), 'papers')
    else:
        selection.install()
        study.OUT, study.register, study.cases = OUT / group, register, cases
        study.run(group, case_id(value))


def run():
    register()
    items = [(g, v) for v in VALUES if v != BASE for g in ['gate_select16', 'salun', 'papers']]
    for g, v in items:
        methods = ['iscc', 'afl'] if g == 'papers' else [g]
        assert not any(source(v, ds, m).exists() for ds in DATASETS for m in methods), 'Do not overwrite attempts.'
    def launch(g, v):
        logfile = OUT / 'logs' / (g + '_' + case_id(v) + '.log')
        logfile.parent.mkdir(exist_ok=True)
        with logfile.open('x') as log:
            result = subprocess.run([sys.executable, str(Path(__file__)), 'child', '--group', g, '--value', str(v)],
                cwd=ROOT, env=dict(os.environ, OVA_MSE_PARENT=str(os.getpid())), stdout=log, stderr=subprocess.STDOUT)
        return dict(group=g, threshold=v, returncode=result.returncode)
    completed = []
    with inherited.campaign.LOCK.open('a') as gpu:
        fcntl.flock(gpu, fcntl.LOCK_EX)
        with ThreadPoolExecutor(max_workers=3) as pool:
            for future in as_completed([pool.submit(launch, g, v) for g, v in items]):
                row = future.result()
                completed.append(row)
                write(OUT / 'campaign_status.json', dict(rows=completed, total_jobs=len(items)))
                print(row, flush=True)
    write(OUT / 'campaign_finished.json', dict(rows=completed, passed=all(r['returncode'] == 0 for r in completed)))
    assert all(r['returncode'] == 0 for r in completed), completed


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['run', 'child'])
    parser.add_argument('--group', choices=['gate_select16', 'salun', 'papers'])
    parser.add_argument('--value', type=float, choices=VALUES)
    args = parser.parse_args()
    run() if args.action == 'run' else child(args.group, args.value)
