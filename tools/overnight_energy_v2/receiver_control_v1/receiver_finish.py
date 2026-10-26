"""Freeze completed pilot reports and a checksum-protected reproducibility bundle."""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import zipfile
from receiver_pilot import sha,write


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,required=True)
    args=p.parse_args();r=args.project.resolve();f=r/'results/overnight_energy_v2';primary=f/'receiver_control_v1';second=f/'receiver_control_v1_request91';s=Path(__file__).parent
    while not (primary/'strong_feedback/completion.json').exists():time.sleep(20)
    subprocess.run([sys.executable,str(s/'receiver_audit.py'),'--project',str(r)],cwd=r,check=True)
    import numpy as np
    import torch
    import cvxpy as cp
    write(primary/'runtime.json',dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,
        cvxpy=cp.__version__,cuda=torch.version.cuda,gpu=subprocess.run(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip()))
    summary=[];ideals=json.loads((primary/'gpu_confirmation/quality.json').read_text())
    strong=json.loads((primary/'strong_feedback/quality.json').read_text())
    selections=json.loads((primary/'strong_feedback/plan.json').read_text())['choices']
    for label,case,quality in [('61',primary,[x for x in ideals if x['case']=='request61']),
                               ('91_legacy',second,json.loads((second/'gpu_quality/quality.json').read_text()))]:
        radio=json.loads((case/'radio_results.json').read_text());geom=json.loads((case/'geometry.json').read_text())
        for name in ['hybrid','oma_grid']:
            row=next(x for x in radio if x['name']==name)
            values=[x for x in quality if x['name'].startswith(name+'_noise')]
            summary.append(dict(case=label,method=name,status=row['status'],energy_j=row.get('energy_bound_j'),
                zero_extra_circuit_energy_j=row.get('zero_extra_circuit_energy_j'),frame_s=row.get('total_frame_s'),
                application_rmse_budget=geom['application_rmse_budget'],application_bound=row.get('application_bound'),
                update_space_rmse_128_draws=row.get('application_rmse_mc'),quality_draws=len(values),
                **quality_means(values)))
        if label=='61':
            identity=json.loads((primary/'gpu_confirmation/ordinary_sum_mixed_identity.json').read_text())
            values=[x for x in quality if x['name'].startswith('ordinary_sum_mixed_noise')]
            summary.append(dict(case=label,method='ordinary_sum_mixed_same_target_at_this_event',status='feasible',
                energy_j=identity['energy_bound_j'],zero_extra_circuit_energy_j=identity['energy_bound_j'],frame_s=identity['frame_s'],
                quality_draws=len(values),**quality_means(values)))
        for criterion,trial in selections[case.name].items():
            values=[x for x in strong if x['case']==case.name and x['trial']==trial['name']]
            summary.append(dict(case=label,method='paid_feedback_mixed_'+criterion,status='ex_post_budget_eligible_optimistic_head_selection',
                energy_j=trial['energy_bound_j'],zero_extra_circuit_energy_j=trial['zero_extra_circuit_energy_j'],
                frame_s=trial['total_frame_s'],sketch_bias_norm=trial['sketch_bias_norm'],trial=trial['name'],
                application_rmse_budget=geom['application_rmse_budget'],application_bound=trial['application_bound'],
                update_space_rmse_128_draws=trial['application_rmse_mc'],quality_draws=len(values),**quality_means(values)))
    write(primary/'summary.json',summary)
    fields=list(dict.fromkeys(k for row in summary for k in row))
    with (primary/'energy_quality.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(summary)
    table=[]
    for case in ['61','91_legacy']:
        h=next(x for x in summary if x['case']==case and x['method']=='hybrid')
        o=next(x for x in summary if x['case']==case and x['method']=='oma_grid')
        table.append(f"요청 {case}: Hybrid {h['energy_j']*1000:.6f} mJ / OMA {o['energy_j']*1000:.6f} mJ / Hybrid가 {(h['energy_j']/o['energy_j']-1)*100:.2f}% 높음 / 둘 다 {h['frame_s']*1000:.3f} ms")
    q91={x['name']:x for x in ideals if x['case']=='request91_legacy'}
    a=q91['canonical_sum'];b=q91['control_alpha1']
    lines=[
        '수신 후 요청별 보정 실험 — 완료 보고서',
        '완료 시각(서버 한국시간): '+datetime.now().isoformat(timespec='seconds'),
        '',
        '결론: 이번 수정만으로 Hybrid NOMA-AirComp가 최적화한 분리 전송(OMA)보다 에너지가 적어지지 않았다. 두 실제 요청의 1-step 파일럿이며 새 100-round 학습 결과가 아니다.',
        *table,
        '',
        '무선 오류 없는 보정 자체의 효과:',
        '- 요청61: 주 설정 alpha=1과 사전 지정 민감도 .25/.5 모두 보정량0. 동일한 float32 모델/예측이므로 새 분리 복원으로 얻은 응용 이득이 없다.',
        '- 기존 이력의 요청91: alpha=1에서 두 요청 가중치 약1.6198/1.5004. 각 요청 방향 진행률을 기존 약.627/.647에서1/1로 높였다. alpha=.25/.5는 보정량0.',
        f"- 요청91 validation loss {a['validation']['loss']:.10f} -> {b['validation']['loss']:.10f}; retained loss {a['retained']['loss']:.10f} -> {b['retained']['loss']:.10f}. 모두 소폭 악화.",
        f"- 요청91 최종시점 재학습 모델 대비 forget 출력 KL {a['forget']['stage_mismatched_retrain100_gap']['kl_reference_to_candidate']:.10f} -> {b['forget']['stage_mismatched_retrain100_gap']['kl_reference_to_candidate']:.10f}. 작은 개선이지만 재학습 기준의 시점이 달라 삭제 성능 입증으로 사용할 수 없다.",
        f"- 요청91 작은 표본 loss-MIA AUC 차이 {a['small_cohort_auc_absolute_gap_to_stage_mismatched_reference']:.10f} -> {b['small_cohort_auc_absolute_gap_to_stage_mismatched_reference']:.10f}. 통계적 우열이나 삭제 보장이 아니다.",
        '',
        '새 방법과 원고의 차이:',
        '- 최종 업데이트를 A+sum U에서 최소거리 QP 보정 F(A,{Q8(U)},g)로 변경했다. 요청별 Q8(U)^T d >= alpha||Q8(U)||^2와 g^T d <= g^T A를 사용한다.',
        '- g는 평가 표본과 겹치지 않는 공개/헤드 접근 가능 validation64에서 계산했다. 이런 자료 접근과 계산은 새로운 가정이다.',
        '- 모든 방식이 같은 8-bit 요청 벡터를 알고리즘 입력으로 사용한다. 이 새 목표에 대한 전송 오차를 비교해 디지털 재전송의 양자화 오차는0이다. 원고의 raw-update 목표/양자화 항과 동일한 문제라고 주장하지 않는다. 원래 raw 합산도 응용 대조군으로 평가했다.',
        '- QP 조건은 방향/1차 기울기 대리 지표다. 실제 retained loss, 요청별 삭제, 재학습 동등성을 보장하지 않는다. 방향이 개선되어도 held-out loss가 나빠진 이번 결과를 그대로 남겼다.',
        '',
        '비교 조건:',
        '- 모두10개 헤드 후보, 동일 채널/0.32W 전력/.1J 배터리/167936 symbols, 공통8-bit 코드. 같은 강한 초기화와 AO30 사용.',
        '- OMA는 학습 아날로그/언러닝 디지털 시간을 분리한다. 디지털 사용자 간에는 NOMA를 유지하며7개 시간 비율을 비교했다. 모든 사용자 TDMA나 연속 시간배분의 전역 최적해라는 뜻은 아니다.',
        '- Mixed에도16-bit 통계 스케치64/128/256, 유료 피드백, 보수적 노름/유료 노름 보고를 허용했다. 추가 전송 RF/프레임 자원을 계산했다.',
        '- 스케치 오차를 전수 헤드에서 사후 점검하고 예산을 만족한 후보 중 최소 에너지도 별도 평가했다. 이는 Mixed에 유리한 오프라인 비교이고, 실제 온라인 보장의 증거는 아니다.',
        '- 추가 회로전력 .01W는 측정값이 아닌 설계 민감도다. 이 가정으로 제안법을 유리하게 만들지 않도록 추가 회로전력0에서도 모든 헤드 중 최소 에너지를 따로 선택/평가했다.',
        '- 요청61은 QP가 비활성이라 일반 sum Mixed도 정확히 같은 결정론적 목표를 만든다. 이 경우 추가 피드백 없는 원래 Mixed를 별도 물리 비교군으로 평가했다.',
        '- 요청91의 무료 정확 가중치 Mixed는 genie 참고값으로만 남기고 실제 에너지 우위 순위에서 제외했다.',
        '- 완료된 GPU 캠페인/최종평가 이후 GPU 확인 평가를 순차 실행했다. 기존 원고, 학습 소스, 원본 결과 보존. CPU 최초 추론 중단은 GPU로 전환하기 위한 것이며 흔적을 보존했다.',
        '',
        '평가와 해석:',
        '- calibration64와 평가 표본을 분리했다. validation/retained 각200, 요청61 forget200, 요청91 현재 forget140. 테스트셋은 쓰지 않았다.',
        '- 요청61과 기존 요청91은 다른 관측 이력이다. 독립 학습 seed2개나 하나의 공통 궤적으로 해석하거나 에너지를 합쳐 누적 우위로 주장하지 않는다.',
        '- 같은3개 채널 잡음으로 응용 품질 평가,128개 잡음으로 업데이트 RMS 진단. 독립 학습 seed 반복이 아니다.',
        '- 표본이 작고 matched-stage 재학습 기준이 없다. MIA와 작은 업데이트 MSE는 삭제 보장이 아니며, 새100-round 우위를 주장하지 않는다.',
        '- CPU replay의 벡터, 요청91 GPU 보정 기울기, 공통 CPU/GPU 수치 차이의 범위를 기록했다. 모든 같은-요청 비교는 같은 벡터/동일 평가 장치를 사용했다.',
        '- 에너지는 통신 모형의 상한이다. CLIP 평가/보정 계산, 비행, 실제 코딩/헤더 구현의 하드웨어 전력은 측정하지 않았다.',
        '',
        '현재 판단:',
        '요청별 보정으로 분리 정보의 응용상 용도는 만들 수 있지만, NOMA가 에너지상 유리하다는 결론은 나오지 않았다. 이번 고정 규칙은 held-out utility도 개선하지 못했다. 이 상태에서 새100-round 학습을 추가하지 않았다. 다음 수정은 실제 validation 손실/언러닝 품질 조건과 계산비용을 포함한 제어 목표부터 검증하는 것이 필요하다.',
        '',
        '재현 파일: plan.json, geometry.json, radio_plan.json, radio_results.json, gpu_confirmation/quality.json, request91_radio 결과, strong_feedback, final_audit.json, runtime.json, tools의 격리된 신규 소스. 기존 CIFAR/CLIP 체크포인트와 원본 벡터는 서버의 명시된 경로/해시를 사용한다.',
    ]
    (primary/'report_ko.txt').write_text('\n'.join(lines)+'\n')
    archive=primary/'receiver_control_experiment.zip';items=[]
    for base in [primary,second,r/'tools/overnight_energy_v2/receiver_control_v1',r/'tools/overnight_energy_v2/receiver_control_v1_request91']:
        for path in sorted(base.rglob('*')):
            if not path.is_file() or path==archive or '__pycache__' in path.parts or path.suffix=='.lock':continue
            items.append(path)
    # Include every new prediction/diagnostic, including failed variants.
    inventory={str(path.relative_to(r)):sha(path) for path in items}
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path in items:z.write(path,str(path.relative_to(r)))
        z.writestr('SHA256.json',json.dumps(inventory,indent=2)+'\n')
    write(primary/'deliverable.json',dict(status='completed',archive=str(archive),archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,
        report_sha256=sha(primary/'report_ko.txt'),files=len(inventory),completed_at_kst=datetime.now().isoformat(timespec='seconds')))
    print('DELIVERABLE',json.dumps(json.loads((primary/'deliverable.json').read_text())),flush=True)


def quality_means(rows):
    import numpy as np
    if not rows:return {}
    result={}
    for group in ['validation','retained','forget']:
        for key in ['loss','accuracy']:
            values=[x[group][key] for x in rows]
            result[group+'_'+key+'_mean']=float(np.mean(values));result[group+'_'+key+'_noise_std']=float(np.std(values))
        for key in ['kl_reference_to_candidate','prediction_disagreement']:
            result[group+'_stage_mismatched_'+key+'_mean']=float(np.mean([x[group]['stage_mismatched_retrain100_gap'][key] for x in rows]))
    result['small_cohort_mia_auc_mean']=float(np.mean([x['small_cohort_loss_attack_auc'] for x in rows]))
    result['small_cohort_mia_reference_gap_mean']=float(np.mean([x['small_cohort_auc_absolute_gap_to_stage_mismatched_reference'] for x in rows]))
    return result


if __name__=='__main__':main()
