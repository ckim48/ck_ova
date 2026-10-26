"""CPU-only completion/provenance and energy attribution for the Q6 run."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,required=True)
    root=p.parse_args().project.resolve();suite=root/'results/comparisons_seed0_solver_v2'
    sys.path.insert(0,str(root/'tools/energy_quality_v1'))
    import energy_quality as eq
    import torch
    protocol=json.loads((suite/'config.json').read_text());source=suite/'coupled_hybrid'
    expected=json.loads((suite/'suite_metadata.json').read_text())['source_sha256']
    assert all(sha(root/n)==h for n,h in expected.items())
    gate,status=eq.validate_completed(suite/'coupled_hybrid_q6',suite,protocol,source)
    if gate is None:
        print(json.dumps({'status':'waiting_for_q6_completion','training_status':status}));return
    schedule=json.loads((source/'schedule.json').read_text());initial=torch.load(source/'initial.pt',map_location='cpu',weights_only=True)['theta']
    methods=[];requests=[];raw={};input_hashes={};excluded=[]
    for name in ['coupled_hybrid','coupled_hybrid_q6','coupled_mixed_aircomp','coupled_fixed_head','coupled_orthogonal_equal_time','uncoupled_hybrid']:
        directory=suite/name;validated,state=eq.validate_completed(directory,suite,protocol,source)
        if validated is None:
            excluded.append({'method':name,'status':state,'energy_ranking_eligible':False});continue
        manifest,records=validated;raw[name]=records
        assert [x['round'] for x in records]==list(range(1,protocol['rounds']+1))
        assert torch.equal(initial,torch.load(directory/'initial.pt',map_location='cpu',weights_only=True)['theta'])
        final=torch.load(directory/'latest.pt',map_location='cpu',weights_only=True)
        assert final['round']==protocol['rounds']
        cumulative=expected_cumulative=0.
        for row in records:
            t=row['transmission'];assert t['status']=='feasible'
            assert t['error_bound']<=t['epsilon']*(1+1e-5)
            cumulative+=t['energy_bound_j'];expected_cumulative+=t['expected_energy_random_codewords_j']
            assert abs(cumulative-row['cumulative_energy_bound_j'])<1e-10
            assert abs(expected_cumulative-row['cumulative_expected_energy_j'])<1e-10
        request_rows=[x for x in records if x['newly_deleted']]
        assert [x['round'] for x in request_rows]==protocol['request_rounds']
        request_energy=sum(x['transmission']['energy_bound_j'] for x in request_rows)
        methods.append({'method':name,'rounds':len(records),'energy_bound_j':cumulative,
                        'expected_random_codeword_energy_j':expected_cumulative,
                        'request_round_energy_bound_j':request_energy,
                        'learning_only_energy_bound_j':cumulative-request_energy,
                        'request_rounds_delivered':len(request_rows),
                        'training_validation':records[-1]['validation']})
        for row in request_rows:
            who=[u for u,ids in enumerate(schedule[row['round']-1]['forget']) if ids]
            head=row['transmission']['head']
            requests.append({'method':name,'round':row['round'],'requesters':who,'head':head,
                             'remote_requesters':sum(u!=head for u in who),
                             'energy_bound_j':row['transmission']['energy_bound_j'],
                             'newly_deleted':row['newly_deleted']})
        input_hashes[name]={file:sha(directory/file) for file in ['manifest.json','rounds.jsonl','initial.pt','latest.pt','schedule.json']}
    baseline=raw['coupled_hybrid'];variant=raw['coupled_hybrid_q6']
    assert [r['newly_deleted'] for r in baseline]==[r['newly_deleted'] for r in variant]
    assert all(a['transmission']['energy_bound_j']==b['transmission']['energy_bound_j'] for a,b in zip(baseline[:50],variant[:50]))
    by_name={x['method']:x for x in methods};h,q,m=[by_name[k] for k in ['coupled_hybrid','coupled_hybrid_q6','coupled_mixed_aircomp']]
    result={'at_kst':datetime.now().isoformat(timespec='seconds'),'status':'q6_training_complete_final_reference_quality_pending',
            'completed_methods':methods,'excluded_methods':excluded,'request_rounds':requests,
            'q6_vs_q8_energy_bound_reduction_fraction':1-q['energy_bound_j']/h['energy_bound_j'],
            'q6_vs_q8_expected_energy_reduction_fraction':1-q['expected_random_codeword_energy_j']/h['expected_random_codeword_energy_j'],
            'mixed_vs_q6_energy_bound_reduction_fraction':1-m['energy_bound_j']/q['energy_bound_j'],
            'q6_vs_q8_request_energy_reduction_fraction':1-q['request_round_energy_bound_j']/h['request_round_energy_bound_j'],
            'first50_energy_identical':True,'input_sha256':input_hashes,'core_source_sha256':expected,'script_sha256':sha(__file__),
            'limits':['Training log validation is not the separate final retained utility cohort.',
                      'Final retained-retraining KL, disagreement and MIA gap pending; no forgetting or equal-quality claim.',
                      'Same canonical schedule/config except quantization; head choices and later trajectories differ.',
                      'One seed. Modeled communication energy excludes uncalibrated flight/local-compute/circuit details.',
                      'Source/checkpoint/log integrity was checked; physical power/SINR constraints are not independently reconstructed from logs without saved designs.',
                      'All five request updates delivered is not certified data removal. Test data not used.']}
    out=root/'results/overnight_energy_v2/precision_completion_v1';out.mkdir(parents=True,exist_ok=True)
    eq.write_json(out/'summary.json',result)
    lines=['6비트 Hybrid 100라운드 완료 — '+result['at_kst']+' KST','',
           '방식 | 누적 통신 에너지 상한 J | 요청 라운드 에너지 상한 J | 마지막 학습 검증 loss | 정확도']
    for row in methods:
        val=row['training_validation'];lines.append(f"{row['method']} | {row['energy_bound_j']:.9f} | {row['request_round_energy_bound_j']:.9f} | {val['loss']:.8f} | {100*val['accuracy']:.2f}%")
    lines+=['',f"6비트는 8비트 대비 상한 {100*result['q6_vs_q8_energy_bound_reduction_fraction']:.2f}% 감소, 랜덤 코드워드 기대 에너지 {100*result['q6_vs_q8_expected_energy_reduction_fraction']:.2f}% 감소.",
            f"Mixed는 6비트 Hybrid보다 에너지 상한이 {100*result['mixed_vs_q6_energy_bound_reduction_fraction']:.2f}% 낮다.",
            '6비트는 모든 100라운드와 5개 요청 라운드를 처리했다. 실패·중단·이상적 링크 에너지 0은 우위 비교에서 제외했다.',
            '처음 50라운드 에너지는 6/8비트가 정확히 같다. 이후 head 선택과 학습 궤적은 달라져 비트 수만의 고정 head 효과가 아니다.',
            '특히 71라운드 6비트는 요청자4를 head로 골라 원격 U 스트림 하나를 없앴다. 8비트의 head2는 비요청자였다.',
            '작은 검증 loss/정확도 차이는 한 seed의 관측이며 품질 동등성 검정이 아니다. 별도 retained utility와 최종 재학습 기준 대비 삭제 품질 평가는 대기 중이다.',
            '원래 학습 소스, 초기 theta, 스케줄, 체크포인트 최종 round와 에너지 누적 로그의 일관성을 확인했다. 로그만으로 모든 무선 제약을 독립 재구성한 검증은 아니다.']
    (out/'report_ko.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':main()
