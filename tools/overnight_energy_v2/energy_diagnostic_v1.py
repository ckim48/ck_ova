"""Separate measured simulation components from unvalidated method claims."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    args = parser.parse_args(); root = args.project.resolve()
    sys.path[:0] = [str(root), str(root/'tools/overnight_energy_v2')]
    from phase_split_grid import make_inputs, necessary_power
    suite = root/'results/comparisons_seed0_solver_v2'
    config = json.loads((suite/'config.json').read_text())
    expected_sources = json.loads((suite/'suite_metadata.json').read_text())['source_sha256']
    assert all(digest(root/n) == h for n,h in expected_sources.items())
    source = suite/'coupled_hybrid'; source_schedule = digest(source/'schedule.json')
    completed = []; excluded = []; request_rows = []
    for mode in ['coupled_hybrid','coupled_mixed_aircomp','coupled_fixed_head','coupled_orthogonal_equal_time']:
        directory = suite/mode; manifest = json.loads((directory/'manifest.json').read_text())
        records = [json.loads(line) for line in (directory/'rounds.jsonl').read_text().splitlines() if line.strip()]
        assert manifest['config'] == config and manifest['source_sha256'] == expected_sources
        assert digest(directory/'schedule.json') == source_schedule
        if manifest['status'] != 'completed':
            excluded.append({'method':mode, 'status':manifest['status'], 'round':records[-1]['round'], 'reason':'Failed/incomplete run excluded from energy comparisons.'}); continue
        assert [r['round'] for r in records] == list(range(1,config['rounds']+1))
        energy = sum(r['transmission']['energy_bound_j'] for r in records)
        expected = sum(r['transmission']['expected_energy_random_codewords_j'] for r in records)
        assert np.isclose(energy, records[-1]['cumulative_energy_bound_j'])
        assert np.isclose(expected, records[-1]['cumulative_expected_energy_j'])
        assert all(r['transmission']['expected_energy_random_codewords_j'] <= r['transmission']['energy_bound_j']*(1+1e-6) for r in records)
        requests = [r for r in records if r['newly_deleted']]
        head_counts = {}
        for r in records:
            h = str(r['transmission']['head']); head_counts[h] = head_counts.get(h,0)+1
        schedule = json.loads((directory/'schedule.json').read_text())
        for r in requests:
            s = schedule[r['round']-1]; requesters = [u for u,ids in enumerate(s['forget']) if ids]
            head = r['transmission']['head']
            request_rows.append({'method':mode, 'round':r['round'], 'requesters':requesters,
                'head':head, 'head_is_requester':head in requesters,
                'scheduled_remote_requesters':sum(u!=head for u in requesters),
                'energy_bound_j':r['transmission']['energy_bound_j'],
                'expected_energy_j':r['transmission']['expected_energy_random_codewords_j']})
        request_energy = sum(r['transmission']['energy_bound_j'] for r in requests)
        completed.append({'method':mode, 'energy_bound_j':energy, 'expected_energy_j':expected,
            'request_round_energy_bound_j':request_energy, 'learning_only_energy_bound_j':energy-request_energy,
            'request_energy_fraction':request_energy/energy, 'request_rounds_delivered':len(requests),
            'head_counts':head_counts, 'validation':records[-1]['validation']})
    source = root/'results/failed_round_diagnostic'
    metadata = json.loads((source/'metadata.json').read_text())
    with np.load(source/'updates.npz') as data:
        learning, unlearning, channels = [data[k].copy() for k in ['learning','unlearning','channels']]
    audit = root/'results/overnight_energy_v2/joint_power_v3'
    audit_manifest = json.loads((audit/'manifest.json').read_text())
    assert audit_manifest['updates_sha256'] == digest(source/'updates.npz')
    assert audit_manifest['source_sha256'] == {n:expected_sources[n] for n in ['wireless.py','transport.py']}
    candidates = json.loads((audit/'results.json').read_text()); radio = []
    for bits in [8,6]:
        for relative in [.2,.6]:
            c = dict(metadata['config'], quantization_bits=bits)
            for mode in ['hybrid','orthogonal','mixed_aircomp']:
                rows = [r for r in candidates if r['bits']==bits and r['relative_tolerance']==relative and r['mode']==mode and r['status']=='feasible']
                best = min(rows,key=lambda r:r['energy_bound_j'])
                path = audit/f"b{bits}_r{relative:g}_{mode}_f{best['analog_fraction']}_design.npz"
                with np.load(path,allow_pickle=False) as data:
                    a,d,h,rho,symbols,dsymbols,users = [data[k].copy() for k in ['analog_rms','digital_sqrt_power','normalized_channels','rho','symbols','digital_symbols','users']]
                    head = int(data['head']); design_energy = float(data['energy_j'])
                    assert str(data['mode']) == mode
                assert head == best['head'] and np.isclose(design_energy,best['energy_bound_j'])
                inputs = make_inputs(learning+unlearning,np.zeros_like(unlearning),c) if mode=='mixed_aircomp' else make_inputs(learning,unlearning,c)
                l,_,_,_,payload = inputs[:5]
                duration = symbols/c['symbol_rate_hz']; digital_duration = dsymbols/c['symbol_rate_hz']
                analog_e = float(np.sum(duration*a*a)); digital_e = float(np.sum(digital_duration*d*d))
                cross_e = 0. if mode=='orthogonal' else float(2*np.sum(duration*a*d))
                overhead = c['users']*c['base_overhead_j']+c['head_overhead_j']
                assert np.isclose(analog_e+digital_e+cross_e+overhead,design_energy)
                normalized_vectors = l[users]/np.where(rho>0,rho,1)[...,None]
                actual_interference = np.sum(np.sum((h*a)[...,None]*normalized_vectors,axis=0)**2,axis=1)
                bound_interference = np.sum(h*a,axis=0)**2
                assert np.all(actual_interference <= bound_interference*(1+1e-8)+1e-12)
                floor = necessary_power(channels,payload,head,dsymbols,c)
                assert floor is not None
                digital_floor = float(np.sum(digital_duration*floor))
                assert digital_e >= digital_floor*(1-1e-6)
                radio.append({'bits':bits,'relative_tolerance':relative,'mode':mode,'head':head,
                    'analog_fraction':best['analog_fraction'],'joint_candidate_energy_bound_j':design_energy,
                    'analog_energy_j':analog_e,'digital_energy_j':digital_e,'waveform_cross_bound_j':cross_e,
                    'overhead_j':overhead,'expected_random_codeword_energy_j':analog_e+digital_e+overhead,
                    'cross_bound_fraction':cross_e/design_energy,
                    'digital_fraction_of_expected':digital_e/(analog_e+digital_e+overhead),
                    'same_head_digital_only_necessary_floor_j':digital_floor+overhead,
                    'remote_payload_bits':int(payload[users].sum()),
                    'power_weighted_actual_over_bound_analog_interference':float(actual_interference.sum()/bound_interference.sum()) if mode=='hybrid' and bound_interference.sum()>0 else None,
                    'diagnostic_only':'Actual analog interference uses full saved local updates unavailable to the original optimizer. It is not a deployable bound or a free-geometry energy competitor.'})
    by_name = {r['method']:r for r in completed}
    head_reduction = 1-by_name['coupled_hybrid']['energy_bound_j']/by_name['coupled_fixed_head']['energy_bound_j']
    result = {'at_kst':datetime.now().isoformat(timespec='seconds'), 'core_source_sha256':expected_sources,
        'script_sha256':digest(__file__), 'schedule_sha256':source_schedule,'updates_sha256':digest(source/'updates.npz'),
        'completed_trajectories':completed,'excluded_trajectories':excluded,'request_rounds':request_rows,
        'head_selection_energy_reduction_fraction':head_reduction,'fixed_update_radio_components':radio,
        'limits':['One seed; same schedule and local algorithm, but head-dependent wireless errors lead to different trajectories.',
            'Final retrain-reference forgetting metrics remain pending; validation closeness is not a deletion guarantee.',
            'Communication model only; no UAV flight or model-computation energy and no lifetime result.',
            'The same-head digital floor is necessary in the current conservative SIC model, not a global physical lower bound.',
            'No source code or running queue was modified. All original results, including failures, remain visible.']}
    out = root/'results/overnight_energy_v2/energy_diagnostic_v1';out.mkdir(parents=True,exist_ok=True)
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    lines = [f"통신 에너지 원인 진단 — {result['at_kst']} KST",'',
        '100라운드 완료본 (에너지 상한, J; 동일 seed·스케줄·로컬 알고리즘):',
        '방식 | 전체 | 삭제 라운드 | 학습만 있는 라운드 | 검증 loss']
    for r in completed:
        lines.append(f"{r['method']} | {r['energy_bound_j']:.6f} | {r['request_round_energy_bound_j']:.6f} | {r['learning_only_energy_bound_j']:.6f} | {r['validation']['loss']:.8f}")
    lines += ['',f'헤드 선택 Hybrid는 고정 헤드보다 누적 상한이 {100*head_reduction:.2f}% 낮다.',
        '헤드에 따른 무선 오차 때문에 학습 궤적은 달라질 수 있으므로, 같은 업데이트에 대한 정확한 전역 절감률은 아니다.',
        '전체 방식의 우위·삭제 보장·네트워크 수명 연장을 뜻하지 않는다. 최종 재학습 기준 품질 평가는 대기 중이다.',
        '', '고정 업데이트 Hybrid의 전력 비용 (공동 AO 후보):',
        '비트/오차 예산 | 아날로그 mJ | 디지털 mJ | 파형 상관 에너지 상한 mJ | 디지털 비중(기대 에너지) | 실제/상한 아날로그 간섭']
    for r in radio:
        if r['mode']=='hybrid':
            lines.append(f"{r['bits']}/{r['relative_tolerance']} | {1000*r['analog_energy_j']:.3f} | {1000*r['digital_energy_j']:.3f} | {1000*r['waveform_cross_bound_j']:.3f} | {100*r['digital_fraction_of_expected']:.2f}% | {r['power_weighted_actual_over_bound_analog_interference']:.4f}")
    lines += ['', '디지털 payload 전송 에너지와 그에 부과되는 아날로그 간섭 제약을 함께 봐야 한다.',
        '파형 상관 상한을 기대 에너지로 바꾸는 것만으로 전체 비용 차이가 사라진다고 볼 수 없다.',
        '실제/상한 간섭 비율은 전체 로컬 업데이트를 사후에 읽은 진단값이다. 원래 옵티마이저가 공짜로 아는 정보가 아니다.',
        '만약 구조/스케치 기반 개선을 시도하면 통계의 가용성·오차 보장·추가 전송 시간과 에너지를 포함해 비교군에도 같은 정보를 줘야 한다.',
        '이 진단만으로 새로운 통신 계상을 학습 실험에 적용하지 않았다.']
    (out/'report_ko.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'head_selection_energy_reduction_fraction':head_reduction,'completed':completed,'hybrid_radio_components':[r for r in radio if r['mode']=='hybrid']}))


if __name__=='__main__':
    main()
