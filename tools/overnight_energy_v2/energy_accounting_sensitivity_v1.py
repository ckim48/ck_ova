"""Conditional circuit/PA accounting on frozen designs; no new energy winner."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def adjusted(row, eta, tx_w, rx_w):
    return (row['overhead_j'] + row['rf_bound_j'] / eta
            + tx_w * row['scheduled_tx_device_s']
            + rx_w * row['head_rx_s'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', type=Path, required=True)
    root = p.parse_args().project.resolve()
    suite = root / 'results/comparisons_seed0_solver_v2'
    manifest_path = root / 'results/overnight_energy_v2/joint_power_v3/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    c = manifest['config']; speed = c['symbol_rate_hz']
    for name, expected in json.loads((suite/'suite_metadata.json').read_text())['source_sha256'].items():
        assert sha(root/name) == expected, name
    input_paths = [manifest_path]
    out = root/'results/overnight_energy_v2/energy_accounting_sensitivity_v1'
    out.mkdir(parents=True, exist_ok=True)
    full = []
    for method in ['coupled_hybrid', 'coupled_mixed_aircomp', 'coupled_fixed_head']:
        directory = suite/method
        input_paths.extend([directory/'manifest.json', directory/'rounds.jsonl'])
        m = json.loads((directory/'manifest.json').read_text())
        assert m['status'] == 'completed'
        records = [json.loads(s) for s in (directory/'rounds.jsonl').read_text().splitlines() if s]
        assert [r['round'] for r in records] == list(range(1, c['rounds']+1))
        overhead = sum(r['transmission']['overhead_j'] for r in records)
        energy = sum(r['transmission']['energy_bound_j'] for r in records)
        duration = sum(r['transmission']['uplink_s'] for r in records)
        assert np.isclose(energy, records[-1]['cumulative_energy_bound_j'])
        full.append({'method': method, 'rounds': len(records), 'energy_bound_j': energy,
                     'overhead_j': overhead, 'rf_bound_j': energy-overhead,
                     'head_rx_s': duration, 'scheduled_tx_device_s': (c['users']-1)*duration})
    assert all(np.isclose(x['head_rx_s'], full[0]['head_rx_s']) for x in full)
    updates_path = root/'results/failed_round_diagnostic/updates.npz'
    assert sha(updates_path) == manifest['updates_sha256']
    input_paths.append(updates_path)
    with np.load(updates_path) as data:
        requesters = np.flatnonzero(np.any(data['unlearning'] != 0, axis=1))
    rows_path = manifest_path.parent/'results.json'; input_paths.append(rows_path)
    rows = json.loads(rows_path.read_text()); fixed = []; break_even = []
    for bits in [8, 6]:
        for relative in [.2, .6]:
            for mode in ['hybrid', 'orthogonal', 'mixed_aircomp']:
                candidates = [x for x in rows if x['bits']==bits and x['relative_tolerance']==relative
                              and x['mode']==mode and x['status']=='feasible']
                best = min(candidates, key=lambda x:x['energy_bound_j'])
                path = manifest_path.parent/f"b{bits}_r{relative:g}_{mode}_f{best['analog_fraction']}_design.npz"
                input_paths.append(path)
                with np.load(path) as d:
                    a, digital = d['symbols'].copy(), d['digital_symbols'].copy()
                    head = int(d['head']); users = d['users'].copy()
                    assert head == best['head'] and np.isclose(float(d['energy_j']), best['energy_bound_j'])
                remote_requesters = int(np.sum(requesters != head))
                if mode == 'orthogonal':
                    # This frozen round has a nonzero payload in every block for each requester.
                    assert remote_requesters > 0
                    rx_s = float((a+digital).sum()/speed)
                    tx_s = float((len(users)*a.sum()+remote_requesters*digital.sum())/speed)
                else:
                    rx_s = float(a.sum()/speed)
                    tx_s = len(users)*rx_s
                assert np.isclose(rx_s, best['uplink_s'])
                fixed.append({'bits': bits, 'relative_tolerance': relative, 'mode': mode,
                              'analog_fraction': best['analog_fraction'], 'head': head,
                              'remote_requesters': remote_requesters,
                              'energy_bound_j': best['energy_bound_j'],
                              'overhead_j': best['overhead_j'],
                              'rf_bound_j': best['energy_bound_j']-best['overhead_j'],
                              'head_rx_s': rx_s, 'scheduled_tx_device_s': tx_s})
            h, phase, mixed = fixed[-3:]
            extra_tx = h['scheduled_tx_device_s']-phase['scheduled_tx_device_s']
            saving = phase['energy_bound_j']-h['energy_bound_j']
            assert extra_tx > 0 and np.isclose(h['head_rx_s'], phase['head_rx_s'])
            threshold = saving/extra_tx if saving > 0 else None
            if threshold is not None:
                assert np.isclose(adjusted(h,1,threshold,0), adjusted(phase,1,threshold,0))
            break_even.append({'bits': bits, 'relative_tolerance': relative,
                               'hybrid_rf_bound_saving_over_phase_j': saving,
                               'additional_scheduled_tx_device_s': extra_tx,
                               'extra_tx_circuit_w_at_tie_eta1': threshold,
                               'extra_tx_circuit_w_at_tie_eta025': threshold/.25 if threshold else None,
                               'hybrid_below_mixed_before_extra_cost': h['energy_bound_j']<mixed['energy_bound_j']})
    sensitivity = []
    for eta in [1, .5, .25]:
        for tx_w, rx_w in [(0,0), (.001,.001), (.01,.01), (.05,.05)]:
            energies = {x['method']:adjusted(x,eta,tx_w,rx_w) for x in full}
            sensitivity.append({'pa_efficiency':eta, 'extra_tx_circuit_w':tx_w,
                                'extra_head_rx_circuit_w':rx_w, 'conditional_energy_j':energies,
                                'mixed_saving_vs_hybrid_fraction':1-energies['coupled_mixed_aircomp']/energies['coupled_hybrid']})
    result = {'at_kst': datetime.now().isoformat(timespec='seconds'),
              'status':'conditional_frozen_design_sensitivity_not_hardware_measurement',
              'formula':'E = fixed_OH + RF_waveform_bound/eta + extra_TX_W * scheduled_TX_device_seconds + extra_head_RX_W * head_RX_seconds',
              'assumptions': [
                  'Powers are hypothetical homogeneous incremental costs beyond overhead already counted. No hardware value is asserted.',
                  'All scheduled analog participants keep TX circuitry on for the analog phase, even if an optimizer coefficient is zero. Digital-only phase keeps only remote requesters on. Coefficient-aware power gating is a different implementation.',
                  'Head RX remains on throughout each complete uplink. PA efficiency is homogeneous; waveform-bound RF peak constraints are unchanged.',
                  'These are frozen feasible RF designs. Adding circuit energy can change optimal head, time split and feasibility under the 0.1 J battery limit; no augmented optimum or battery-feasibility claim is made.',
                  'Additional SIC, codec, control-RX, device computation, and flight costs are unmeasured. Equal overhead is not evidence these costs are physically equal.',
                  'Fixed-update diagnostic is one legacy round. Final retraining-reference utility/forgetting evaluation remains pending.'],
              'full_completed_runs':full,'full_run_sensitivity':sensitivity,
              'fixed_update_best_rf_designs':fixed,'hybrid_phase_break_even':break_even,
              'inputs_sha256':{str(x.relative_to(root)):sha(x) for x in input_paths},
              'script_sha256':sha(__file__)}
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    lines = ['에너지 계상 민감도 — 추가 회로 비용의 조건부 계산', '',
             '측정된 하드웨어 전력이나 새로운 에너지 순위가 아니다. 기존 overhead에 이미 포함된 비용을 다시 더하지 않고, 그 밖의 추가 비용을 가정했다.',
             '가정: 아날로그 참여자는 해당 구간에 송신 회로를 켜고, 분리 전송의 디지털 구간에는 원격 요청자만 송신 회로를 켠다. 모든 방식에 동일한 추가 W를 적용한다.',
             '', '동일 고정 업데이트, RF 최적 후보 간 손익분기점 (PA 효율 1 가정):',
             '비트/상대오차 | Hybrid의 분리 전송 대비 RF 상한 절감 mJ | 추가 송신 회로 전력의 동률 지점 mW']
    for x in break_even:
        threshold = x['extra_tx_circuit_w_at_tie_eta1']
        lines.append(f"{x['bits']}/{x['relative_tolerance']} | {1000*x['hybrid_rf_bound_saving_over_phase_j']:.6f} | {1000*threshold:.6f}" if threshold else f"{x['bits']}/{x['relative_tolerance']} | {1000*x['hybrid_rf_bound_saving_over_phase_j']:.6f} | 원래부터 분리 전송이 낮음")
    lines += ['', '상대오차 0.6에서의 소폭 Hybrid RF 절감은 위 추가 회로 비용 가정에 민감하다. 양의 임계값을 넘으면 이미 찾은 분리 전송 설계가 더 낮아진다. 이 숫자로 실제 UAV 회로 전력을 추정하지 않는다.',
              '효율 eta가 1보다 작으면 RF 차이는 1/eta배이므로 동률 회로 전력도 1/eta배가 된다. 별도 SIC 계산 비용은 미포함이다.',
              '', '완료된 100라운드 비교의 공통 추가 송수신 비용은 에너지 절감률의 분모를 늘린다. 공통 비용만 추가하면 Mixed와 Hybrid의 에너지 차이 부호는 바뀌지 않는다.',
              '회로가 꺼지는 동작과 에너지 계측이 확인되기 전에는 동시 송신에 따른 절감이나 배터리 수명 우위를 주장할 수 없다.',
              '이 진단은 학습 코드, 대기 큐 또는 논문의 원래 문제를 변경하지 않았다. 회로를 포함한 최적화는 별도 문제로 정의하고 모든 방식의 head와 시간 배분을 다시 최적화해야 한다.']
    (out/'report_ko.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'status':result['status'],'hybrid_phase_break_even':break_even,'full_run_sensitivity':sensitivity}))


if __name__ == '__main__':
    main()
