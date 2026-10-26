"""Compare cached final outputs to retraining and to coupled ideal separately."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,required=True)
    p.add_argument('--wait-for-early-gate',action='store_true')
    args=p.parse_args();root=args.project.resolve();suite=root/'results/comparisons_seed0_solver_v2';quality=root/'results/energy_quality_v1'
    report_path=root/'results/overnight_energy_v2/early_reference_gate_v1/final_reference_report.json'
    if args.wait_for_early_gate:
        gate_status=report_path.parent/'status.json'
        for _ in range(180):
            state=json.loads(gate_status.read_text()) if gate_status.exists() else {}
            if state.get('status')=='completed' and report_path.exists():break
            if state.get('status','').startswith('failed'):
                print(json.dumps({'status':'early_gate_failed_no_diagnostic','gate_status':state}));return
            time.sleep(10)
        else:print(json.dumps({'status':'timed_out_waiting_for_early_gate'}));return
    if not report_path.exists():print(json.dumps({'status':'waiting_for_early_reference_gate'}));return
    sys.path.insert(0,str(root/'tools/energy_quality_v1'));import energy_quality as eq
    report=json.loads(report_path.read_text());assert report['reference_available']
    protocol=json.loads((suite/'config.json').read_text());source=suite/'coupled_hybrid'
    expected=json.loads((suite/'suite_metadata.json').read_text())['source_sha256'];assert all(sha(root/n)==h for n,h in expected.items())
    names=['coupled_hybrid','coupled_hybrid_q6','coupled_mixed_aircomp','coupled_fixed_head','coupled_ideal','retrain_reference_ideal']
    predictions={};hashes={};report_rows={x['method']:x for x in report['methods']}
    for name in names:
        directory=suite/name;validated,status=eq.validate_completed(directory,suite,protocol,source);assert validated is not None,(name,status)
        cache=json.loads((quality/(name+'.cache.json')).read_text())
        keys={'checkpoint_sha256':sha(directory/'latest.pt'),'cohorts_sha256':sha(quality/'cohorts.json'),
              'evaluator_sha256':sha(eq.__file__),'training_source_sha256':validated[0]['source_sha256']}
        assert all(cache.get(k)==v for k,v in keys.items()),name
        with np.load(quality/(name+'.npz'),allow_pickle=False) as data:predictions[name]={k:data[k].copy() for k in data.files}
        hashes[name]={'predictions_sha256':sha(quality/(name+'.npz')),'cache_sha256':sha(quality/(name+'.cache.json'))}
    reference=predictions['retrain_reference_ideal'];ideal=predictions['coupled_ideal'];diagnosis=[]
    for name in names:
        row={'method':name,'energy_role':report_rows[name]['energy_role'],
             'energy_bound_j':report_rows[name]['energy_bound_j']}
        for group in ['validation','retained','forget']:
            a=predictions[name];assert np.array_equal(a[group+'_labels'],reference[group+'_labels'])
            assert np.array_equal(a[group+'_labels'],ideal[group+'_labels'])
            row[group+'_classification']=eq.classification(a[group],a[group+'_labels'])
            row[group+'_gap_to_retraining']=eq.reference_gap(a[group],reference[group],a[group+'_labels'])
            row[group+'_gap_to_coupled_ideal']=eq.reference_gap(a[group],ideal[group],a[group+'_labels'])
        row['forget_attack_auc']=report_rows[name]['forget_loss_attack_auc_validation']
        row['forget_attack_auc_gap_to_retraining']=report_rows[name]['forget_attack_auc_gap_to_reference_validation']
        diagnosis.append(row)
    ideal_ref=next(x for x in diagnosis if x['method']=='coupled_ideal')['forget_gap_to_retraining']['kl_reference_to_candidate']
    result={'at_kst':datetime.now().isoformat(timespec='seconds'),'status':'completed_cached_final_output_diagnosis',
            'methods':diagnosis,'coupled_ideal_forget_kl_to_retraining':ideal_ref,
            'communication_kl_over_ideal_retraining_kl':{x['method']:x['forget_gap_to_coupled_ideal']['kl_reference_to_candidate']/ideal_ref if ideal_ref>0 else None for x in diagnosis if x['energy_role']=='physical_model'},
            'observed_three_objective_frontier':report['observed_frontier'],
            'source_report_sha256':sha(report_path),'cohorts_sha256':sha(quality/'cohorts.json'),
            'cache_sha256':hashes,'script_sha256':sha(__file__),
            'limits':['KL to coupled ideal and KL to retraining are distinct comparisons and are not an additive causal decomposition.',
                      'Ideal communication has energy zero only as a reference and is excluded from physical energy ranking.',
                      'One seed and one retraining realization; no certified deletion or task-quality equivalence.',
                      'Exact observed frontier does not establish statistical superiority. Quality thresholds were not prespecified.',
                      'Retain-only control and improved burst protocol remain pending. No predictions or test sets were used to modify a queued configuration.',
                      'This script only reads verified caches; it launches no training or inference.']}
    out=root/'results/overnight_energy_v2/final_output_diagnosis_v1';out.mkdir(parents=True,exist_ok=True)
    eq.write_json(out/'summary.json',result)
    lines=['최종 재학습 기준과 이상적 링크의 출력 비교 — '+result['at_kst']+' KST','',
           '방식 | 통신 상한 J | utility val loss | forget KL(retrain||method) | forget 예측 불일치 | MIA AUC 차이']
    for x in diagnosis:
        gap=x['forget_gap_to_retraining'];lines.append(f"{x['method']} | {x['energy_bound_j']:.9f} | {x['validation_classification']['loss']:.8f} | {gap['kl_reference_to_candidate']:.9g} | {100*gap['prediction_disagreement']:.3f}% | {x['forget_attack_auc_gap_to_retraining']:.7f}")
    lines+=['', '물리적 방식과 coupled ideal 사이의 forget 출력 KL:',
            *[f"{x['method']} | {x['forget_gap_to_coupled_ideal']['kl_reference_to_candidate']:.9g}" for x in diagnosis if x['energy_role']=='physical_model'],
            '',f'coupled ideal 자체와 재학습 기준의 forget KL: {ideal_ref:.9g}',
            '두 KL을 더하거나 빼서 원인별 비중으로 해석하지 않는다. 별도 비교값의 크기로 무선 오차와 알고리즘 차이를 구분해 점검한다.',
            '작은 MIA 신호, MSE, 출력 KL만으로 삭제를 보장하지 않는다. 모든 값은 한 seed와 한 최종 재학습 실행에 대한 진단이다.',
            '실패·중단·이상적 링크 에너지 0은 물리적 에너지 우위 비교에서 제외한다. retain-only 대조와 개선 프로토콜 학습은 대기 중이다.']
    (out/'report_ko.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':main()
