"""Declared fixed-spectral-efficiency downlink bandwidth intervention.

Frozen training extension: preserve existing rate studies and algorithm code.
The 1 MHz controls and 3/5 MHz conditions execute the complete FL/FU workload.
Existing 2/4 Mbit/s runs are reused only after exact input and cost equivalence
checks under the explicitly assumed 1 bit/s/Hz downlink model.
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

import learning_quality_v3 as study
import learning_selection_fix_v2 as selection
import learning_quality_campaign as campaign

ROOT = study.ROOT
OUT = ROOT / 'results/downlink_bandwidth_v1_20261028'
infra = study.infra
read, write, sha = study.read, study.write, study.sha
ORIGINAL_CONFIGURE = infra.configure
VALUES = [1, 2, 3, 4, 5]
FRESH_VALUES = [1, 3, 5]
EFFICIENCY = 1.0
METHODS = {'gate_select16': 'Proposed', 'iscc': 'ISCC-P2',
           'afl': 'AFL-OTA', 'salun': 'F-SalUn'}
MODEL = dict(name='fixed_effective_spectral_efficiency',
             equation='R_dl = B_dl * eta_dl',
             effective_spectral_efficiency_bps_hz=EFFICIENCY,
             bandwidth_values_mhz=VALUES, baseline_bandwidth_mhz=1,
             power_w=0.32, uplink_bandwidth_hz=1e6,
             assumption='All intended receivers support the same fixed net spectral efficiency throughout 1–5 MHz. Effective efficiency includes modulation/coding/overhead; it is a declared service model, not an SNR-calibrated Shannon capacity or measured channel. No adaptive MCS or bandwidth-dependent packet error is modeled. Baseline aggregate packet erasure is zero. Downlink power and all other coefficients remain fixed.',
             scope='Only downlink bandwidth changes. Full 20480-coordinate FP32 model plus512 control bits per nonzero update; actual broadcast and hover costs are charged. No claim that increasing bandwidth at fixed power preserves reliability in an arbitrary physical channel.')


def case_id(value):
    return f'downlink_bandwidth_{value}'


def cases():
    return [dict(id=case_id(v), axis='downlink_bandwidth', value=v) for v in VALUES]


def rate_bps(bandwidth_hz, efficiency=EFFICIENCY):
    assert math.isfinite(bandwidth_hz) and bandwidth_hz > 0
    assert math.isfinite(efficiency) and efficiency > 0
    return bandwidth_hz * efficiency


def configure(case, plan, requests, geometry, deployment, parameters, channels):
    result = ORIGINAL_CONFIGURE(case, plan, requests, geometry, deployment, parameters, channels)
    assert case['axis'] == 'downlink_bandwidth'
    p = parameters['communication']
    p['downlink_bandwidth_hz'] = float(case['value']) * 1e6
    p['downlink_spectral_efficiency_bps_hz'] = EFFICIENCY
    p['downlink_rate_bps'] = rate_bps(p['downlink_bandwidth_hz'], EFFICIENCY)
    p['downlink_rate_model'] = MODEL['name']
    parameters['downlink_bandwidth_model'] = MODEL
    return result


def communication(meta):
    p = infra.PHYSICAL.PARAMETERS['communication']
    rate = rate_bps(p['downlink_bandwidth_hz'], p['downlink_spectral_efficiency_bps_hz'])
    assert p['downlink_rate_bps'] == rate
    up = meta['uplink_s']
    energy = meta['energy_bound_j'] / p['pa_efficiency'] + up * (9*p['tx_circuit_w'] + p['rx_circuit_w'])
    active = up > 0 or meta.get('broadcast_required', False)
    count = meta.get('broadcast_attempts', int(active))
    down = count * (p['update_coordinates']*p['broadcast_bits_per_coordinate'] + p['control_bits_per_update']) / rate
    energy += down * (p['downlink_power_w']/p['pa_efficiency'] + p['tx_circuit_w'] + 9*p['rx_circuit_w'])
    return energy, down


def register():
    path = OUT / 'protocol.json'
    if path.exists():
        p = read(path)
        for name, digest in p['source_sha256'].items():
            assert sha(ROOT / name) == digest, name
        return p
    inherited = selection.register()
    sources = dict(inherited['source_sha256'])
    sources[str(Path(__file__).relative_to(ROOT))] = sha(Path(__file__))
    p = dict(registered_unix=time.time(), model=MODEL, methods=METHODS,
             datasets=['CapERA', 'AirScape'], seed=0,
             new_arms=24, reused_arms=16, cases=cases(),
             fresh_bandwidth_mhz=FRESH_VALUES, reused_bandwidth_mhz=[2, 4],
             fixed='Original P-KS, ISCC-P2, AFL-OTA and F-SalUn learning and transport rules. Same features, warm20, 10 UAVs, five regions, 45 requests, 200 total FL rounds and225 request-local FU steps. No P-KB25 or old Proposed substitution.',
             reuse='Only2/4MHz reuse existing2/4Mbit/s arms, under this explicit eta=1 assumption. Every original parameter, input hash, transaction broadcast duration and cost must match. Reused runs are identified as equivalent-rate reuse, never new bandwidth executions. Fresh1MHz controls must reproduce all four original baseline models and measured clocks before release.',
             interpretation='This study adds an explicit bandwidth-to-rate assumption; equal numerical MHz and Mbit/s values are conditional on eta=1, not an identity between bandwidth and throughput. Curves are conditional simulations, not empirical bandwidth measurements.',
             publication='Code, charts, plot arrays, aggregate metrics and provenance only; checkpoints, raw samples and execution logs stay local.',
             source_sha256=sources)
    write(path, p)
    return p


def fresh_source(value, dataset, method):
    group = method if method in ['gate_select16', 'salun'] else 'papers'
    variant = method if method in ['gate_select16', 'salun'] else None
    root = OUT / group
    if variant:
        root = root / variant
    internal = {'gate_select16': study.PROPOSED, 'iscc': study.DIGITAL,
                'afl': 'uav_afl_ota_gaft', 'salun': study.DIGITAL}[method]
    return root / 'cases' / case_id(value) / (dataset.lower() + '_seed0') / internal


def child(group, value):
    assert int(os.environ['OVA_BANDWIDTH_PARENT']) == os.getppid()
    original_lock = fcntl.flock
    def lock(handle, mode):
        if getattr(handle, 'name', None) == str(campaign.LOCK):
            assert mode == fcntl.LOCK_EX
            return
        return original_lock(handle, mode)
    fcntl.flock = lock
    register()
    infra.configure = configure
    infra.communication = communication
    infra.cases = cases
    infra.register = register
    if group == 'papers':
        infra.OUT = OUT / group
        infra.run_case(case_id(value), 'papers')
    else:
        selection.install()
        study.OUT = OUT / group
        study.register = register
        study.cases = cases
        study.run(group, case_id(value))


def run():
    register()
    items = [(group, v) for v in FRESH_VALUES for group in ['gate_select16', 'salun', 'papers']]
    for group, v in items:
        methods = ['iscc', 'afl'] if group == 'papers' else [group]
        assert not any(fresh_source(v, ds, m).exists() for ds in MODEL_DATASETS for m in methods), 'Preserve previous attempts; do not overwrite.'
    def launch(group, v):
        logfile = OUT / 'logs' / f'{group}_{v}.log'
        logfile.parent.mkdir(exist_ok=True)
        with logfile.open('x') as log:
            result = subprocess.run([sys.executable, str(Path(__file__)), 'child', '--group', group, '--value', str(v)],
                cwd=ROOT, env=dict(os.environ, OVA_BANDWIDTH_PARENT=str(os.getpid())), stdout=log, stderr=subprocess.STDOUT)
        return dict(group=group, bandwidth_mhz=v, returncode=result.returncode)
    rows = []
    with campaign.LOCK.open('a') as gpu:
        fcntl.flock(gpu, fcntl.LOCK_EX)
        with ThreadPoolExecutor(max_workers=3) as pool:
            for future in as_completed([pool.submit(launch, g, v) for g, v in items]):
                row = future.result()
                rows.append(row)
                write(OUT / 'campaign_status.json', dict(completed_jobs=rows, total_jobs=len(items)))
                print(row, flush=True)
    write(OUT / 'campaign_finished.json', dict(rows=rows, passed=all(r['returncode'] == 0 for r in rows)))
    assert all(r['returncode'] == 0 for r in rows), rows


MODEL_DATASETS = ['CapERA', 'AirScape']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['run', 'child', 'register'])
    parser.add_argument('--group', choices=['gate_select16', 'salun', 'papers'])
    parser.add_argument('--value', type=int, choices=FRESH_VALUES)
    args = parser.parse_args()
    if args.action == 'child':
        child(args.group, args.value)
    elif args.action == 'run':
        run()
    else:
        register()
