"""Save a compact, source-traceable snapshot for the 01:42 heartbeat."""
import hashlib
import json
from pathlib import Path
from datetime import datetime


def main(root):
    def read(path, fallback=None):
        path = root/path
        return json.loads(path.read_text()) if path.exists() else fallback
    suite = 'results/comparisons_seed0_solver_v2'
    records = {}
    for name in ['coupled_hybrid', 'coupled_orthogonal_equal_time', 'coupled_mixed_aircomp',
                 'coupled_fixed_head', 'uncoupled_hybrid', 'puf_regular_adapted_hybrid',
                 'coupled_ideal', 'retrain_reference_ideal']:
        manifest = read(f'{suite}/{name}/manifest.json', {})
        p = root/f'{suite}/{name}/rounds.jsonl'
        rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()] if p.exists() else []
        records[name] = {'status': manifest.get('status', 'not_started'),
                         'last': rows[-1] if rows else None,
                         'latest_validation': next((r['validation'] for r in reversed(rows) if 'validation' in r), None),
                         'error': manifest.get('error')}
    regimes = read('results/overnight_energy_v2/error_budget_regimes/results.json', [])
    joint = read('results/overnight_energy_v2/joint_power_v3/results.json', [])
    pilot = read('results/overnight_energy_v2/single_step_quality_pilot/results.json', [])
    expected = read(f'{suite}/suite_metadata.json')['source_sha256']
    hashes = {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in expected}
    now = datetime.now().isoformat(timespec='seconds')
    snapshot = {'at_kst': now, 'suite': records, 'regime_completed_cases': len(regimes),
                'joint_completed_cases': len(joint), 'pilot_completed_vectors': len(pilot),
                'core_sha256': hashes, 'core_unchanged': hashes == expected,
                'jobs': read('tools/overnight_energy_v2/launched_jobs.json')}
    out = root/'results/overnight_energy_v2'
    (out/'heartbeat_0142_snapshot.json').write_text(json.dumps(snapshot, indent=2)+'\n')
    lines = [f'ova_CK 밤샘 연구 중간 기록 — {now} KST', '',
        '새 개선: 기존 AO에서 디지털 전력을 고정하던 갱신을, 수신 계수가 고정된 상태에서',
        '아날로그·디지털 전력을 함께 최적화하도록 별도 버전으로 바꿨다.',
        '초기화에 이미 쓰던 동일 SOCP를 재사용하며 원래 전력·배터리·SIC·오차 제약을 유지한다.',
        '독립 제약 검증 테스트 3개 통과. 원본 학습 소스의 모든 SHA256 일치 확인.', '']
    row = next((r for r in joint if (r['bits'], r['relative_tolerance'], r['mode']) == (8, .2, 'hybrid')), None)
    if row and row['status'] == 'feasible':
        old = row['original_incumbent']; reduction = 100*(1-row['energy_bound_j']/old['energy_bound_j'])
        lines += [f"고정 업데이트 91라운드, 8비트, 기존 상대 RMS 예산 0.2: 에너지 상한 {1000*old['energy_bound_j']:.3f} → {1000*row['energy_bound_j']:.3f} mJ ({reduction:.2f}% 감소).",
            f"기대 에너지 {1000*old['expected_energy_random_codewords_j']:.3f} → {1000*row['expected_energy_random_codewords_j']:.3f} mJ.",
            f"개선 버전의 조건부 전체 업데이트 RMSE {row['analytic_total_rmse']:.6g}, 디지털 U 복원 오차 {row['unlearning_component_error']:.6g}.",
            '단일 고정 업데이트의 최적화 개선이며, 같은 모델 품질이나 삭제 보장을 뜻하지 않는다.',
            '같은 조건의 기존 시간 최적화 분리 전송 32.823 mJ, 혼합 AirComp 5.059 mJ보다 여전히 비싸다.', '']
    lines += ['학습 큐 (미완료·실패 값은 에너지 우위 순위에서 제외):']
    for name, record in records.items():
        last = record['last']
        if last:
            v = record['latest_validation'] or {}
            lines.append(f"  {name}: {record['status']}, {last['round']}/100, 누적 상한 {last.get('cumulative_energy_bound_j'):.6f} J, 최근 validation loss {v.get('loss')}, accuracy {v.get('accuracy')}")
        else:
            lines.append(f"  {name}: {record['status']}")
    h, m = records['coupled_hybrid'], records['coupled_mixed_aircomp']
    if h['status'] == m['status'] == 'completed' and h['last']['round'] == m['last']['round'] == 100:
        saving = 100*(1-m['last']['cumulative_energy_bound_j']/h['last']['cumulative_energy_bound_j'])
        lines += ['', f'100라운드 완료본에서 혼합 AirComp의 누적 에너지 상한이 Hybrid보다 {saving:.2f}% 작다.',
            '단일 seed 결과이고, 최종 재학습 모델 대비 forget KL·예측 차이·MIA AUC 차이는 아직 평가 전이다.',
            '따라서 최종 삭제 품질 우열은 아직 결론 내리지 않는다.']
    lines += ['', f'오차 예산 맵 {len(regimes)}/90, 공동 최적화 비교 {len(joint)}/36, 단일 단계 VLM 예비 평가 {len(pilot)}/14 벡터 완료.',
        '공동 최적화는 Hybrid·분리 전송·혼합 AirComp 모두에 동일 적용하며 기존의 더 좋은 해도 보존한다.',
        '예비 평가는 validation/forget 각 200개, 공통 잡음 seed 3개이며 테스트셋 설정 선택을 하지 않는다.',
        '재학습 정답과의 비교가 아닌 이상적 단일 단계 업데이트와의 차이이므로 누적 학습·삭제 성능을 대체하지 못한다.',
        '', '기존 GPU 큐 → 6비트 Hybrid 후속 학습 → 최종 품질 평가 → retain-only 대조군이 순차 대기 중이다.',
        'GPU 실험을 중복 실행하지 않았다. 원본 manuscript·학습 소스·기존 결과를 보존했다.',
        '다음 점검은 전체 예비 평가와 개선 AO 비교 결과를 확인하고, 근거 있는 다음 실험을 선정한다.']
    (out/'heartbeat_0142_report_ko.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({'report': str(out/'heartbeat_0142_report_ko.txt'), 'at_kst': now,
                      'core_unchanged': hashes == expected, 'joint_rows': len(joint)}))


if __name__ == '__main__':
    import sys
    main(Path(sys.argv[1]).resolve())
