"""Archive completed CPU evidence and the 02:12 overnight research decision."""
import json
from pathlib import Path
from datetime import datetime
import hashlib

import numpy as np


def main(root):
    def read(name, default=None):
        path = root/name
        return json.loads(path.read_text()) if path.exists() else default
    base = 'results/overnight_energy_v2'
    regimes = read(base+'/error_budget_regimes/results.json', [])
    joint = read(base+'/joint_power_v3/results.json', [])
    comparisons = read(base+'/joint_power_v3/comparisons.json', [])
    pilot = read(base+'/single_step_quality_pilot/summary.json', [])
    jobs = read('tools/overnight_energy_v2/launched_jobs.json')
    gate = read(base+'/burst_joint_v3/legacy_adapter_validation.json', {})
    state = read(base+'/burst_joint_v3/status.json', {})
    now = datetime.now().isoformat(timespec='seconds')
    snapshot = {'at_kst':now, 'regime_cases':len(regimes), 'joint_cases':len(joint),
                'joint_comparisons':comparisons, 'single_step_pilot':pilot,
                'new_campaign_status':state, 'new_adapter_validation':gate, 'jobs':jobs}
    source = read('results/comparisons_seed0_solver_v2/suite_metadata.json')['source_sha256']
    snapshot['core_unchanged'] = all(hashlib.sha256((root/n).read_bytes()).hexdigest()==h for n,h in source.items())
    text = [f'ova_CK 밤샘 연구 중간 기록 — {now} KST', '',
        f'오차 예산 맵 {len(regimes)}/90, 공동 전력 최적화 비교 {len(joint)}/36, VLM 단일 단계 예비 평가 14/14가 완료됐다.',
        '기존 학습 소스의 모든 해시는 유지됐다. 이번 CPU 진단은 동일한 과거 91라운드 업데이트를 사용한다.', '',
        '공동 최적화 AO 결과 (통신 에너지 상한, mJ):',
        '비트 | 상대 RMS 예산 | Hybrid | 시간 배분 최적화 분리 전송 | 혼합 AirComp']
    for r in comparisons:
        text.append(f"{r['bits']:>4} | {r['relative_tolerance']:.1f} | {1000*r['hybrid']['selected_energy_bound_j']:.3f} | {1000*r['orthogonal']['selected_energy_bound_j']:.3f} | {1000*r['mixed_aircomp']['selected_energy_bound_j']:.3f}")
    text += ['', '0.6 예산에서는 Hybrid가 분리 전송보다 약 3~7% 적은 에너지 상한을 보이지만, 혼합 AirComp가 훨씬 저렴하다.',
        '기존 0.2 예산에서는 개선 AO를 사용해도 분리 전송이 Hybrid보다 저렴하다.',
        '각 비트 설정은 U 양자화 오차가 다르므로 서로 같은 삭제 품질이라고 볼 수 없다.',
        '분리 전송은 디지털 사용자 간 NOMA를 유지하며 7개 시간 비율을 탐색했다. 전역 최적 OMA/TDMA 결과가 아니다.', '',
        'VLM 예비 평가 (8비트, 기존 AO, validation/forget 각 200개, 공통 잡음 seed 3개):',
        '방식 | 에너지 상한 mJ | validation KL | forget KL (이상적 단일 단계 업데이트 대비)']
    for r in pilot:
        text.append(f"{r['method']} | {1000*r['energy_bound_j']:.3f} | {r['validation_kl_reference_to_candidate_mean']:.3g} | {r['forget_kl_reference_to_candidate_mean']:.3g}")
    text += ['모든 잡음 draw에서 validation 정확도 73.5%, forget 정확도 71.5%, 예측 불일치 0이었다.',
        '이 예비 평가에서는 혼합 AirComp의 출력 열화가 Hybrid보다 크다는 증거가 없었다.',
        '단일 단계 이상적 모델은 재학습 삭제 기준이 아니다. 작은 KL/MSE 또는 MIA만으로 삭제 보장을 주장하지 않는다.', '',
        '후속 전체 학습 실험을 별도 burst_joint_v3 버전으로 구성했다:',
        '  - 평상시 상대 RMS 예산 0.2, 삭제 업데이트가 있는 라운드에만 0.6.',
        '  - Hybrid·혼합 AirComp·시간 분리 전송 모두 같은 변경, 8비트, 100라운드, 같은 초기 모델·데이터·전력·시간·배터리 조건.',
        '  - 분리 전송은 매 라운드 7개 비율에서 최소 에너지 해 선택. 학습만 있는 라운드는 세 방식에 동일한 코드 경로 사용.',
        '  - 기존 GPU 큐, 6비트 후속 학습, 품질 평가, retain-only 대조군이 끝난 뒤 Hybrid → 혼합 AirComp → 분리 전송 순서로 실행.',
        '  - 원본 학습 루프를 재사용하고 통신 함수만 버전이 명시된 어댑터로 교체한다. 별도 소스 해시와 프로토콜을 기록한다.',
        '  - 테스트 결과로 설정·순서를 고르지 않는다. 새 학습 시작은 정오까지, 이미 시작한 학습과 최종 평가는 안전하게 완료한다.',
        '  - 실제 크기의 업데이트로 통신 어댑터가 독립 AO 진단과 일치하는지 검증한 뒤에만 GPU 학습 가능.',
        '  - 동일한 frozen cohort와 수치 평가기로 retained loss, 재학습 대비 forget KL/예측 차이, MIA AUC 차이, 처리·완료 여부와 누적 통신 에너지 평가.', '',
        '어댑터·평가 무결성 테스트 5개와 계획/평가 요약 smoke check가 통과했다.',
        f"새 캠페인 상태: {state.get('status', 'pending')}; 실제 크기 어댑터 검증: {gate.get('status', 'running_or_pending')}",
        f"단일 GPU를 중복 사용하지 않도록 대기 중인 캠페인 PID: {jobs.get('burst_joint_v3',{}).get('pid')}",
        '', '현재 증거로는 제안 Hybrid의 보편적 우위를 주장할 수 없다. 조건부 이득과 품질 손실을 함께 검증한다.']
    out = root/base
    (out/'heartbeat_0212_snapshot.json').write_text(json.dumps(snapshot,indent=2)+'\n')
    (out/'heartbeat_0212_report_ko.txt').write_text('\n'.join(text)+'\n',encoding='utf-8')
    plot(out, regimes, pilot)
    print(json.dumps({'at_kst':now, 'core_unchanged':snapshot['core_unchanged'], 'validation':gate.get('status'), 'campaign':state.get('status')}))


def plot(out, regimes, pilot):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9, 'axes.spines.top':False, 'axes.spines.right':False})
    fig, axes = plt.subplots(1,3,figsize=(13.4,4.5),gridspec_kw={'width_ratios':[1,1,1.25]})
    styles = [('hybrid','Hybrid','#1671b6','o'),('orthogonal','Phase, best grid','#df7312','s'),
              ('mixed_aircomp','Mixed AirComp','#38834c','^')]
    budgets = [.1,.2,.4,.6,.8]
    for ax,bits in zip(axes[:2],[8,6]):
        for mode,label,color,marker in styles:
            values=[]
            for relative in budgets:
                rows=[r for r in regimes if r['bits']==bits and r['relative_tolerance']==relative and r['mode']==mode and r['status']=='feasible']
                values.append(1000*min(r['energy_bound_j'] for r in rows) if rows else np.nan)
            ax.plot(budgets,values,marker=marker,label=label,color=color,linewidth=1.4,markersize=4)
        ax.set(yscale='log',xlabel='Relative RMS envelope',ylabel='Communication-energy bound (mJ)',
               title=f'{bits}-bit payload: original AO',xticks=budgets,ylim=(.12,85))
        ax.grid(True,which='major',alpha=.22)
        ax.text(.5,-.25,'Hybrid at 0.1: no feasible design found',transform=ax.transAxes,ha='center',fontsize=7.4)
    axes[0].legend(loc='lower left',fontsize=7.5,frameon=False)
    names=['hybrid_default','hybrid_loose','phase_loose','mixed_loose']; data={r['method']:r for r in pilot}
    ax=axes[2]; x=np.arange(4)
    for offset,group,label,color,marker in [(-.09,'validation','Validation','#1671b6','o'),(.09,'forget','Forget cohort','#df7312','s')]:
        mean=[data[n][group+'_kl_reference_to_candidate_mean'] for n in names]
        std=[data[n][group+'_kl_reference_to_candidate_noise_std'] for n in names]
        ax.errorbar(x+offset,mean,yerr=std,fmt=marker,color=color,label=label,capsize=3,markersize=5,linestyle='none')
    ax.set(yscale='log',ylabel='Output KL to ideal single step',title='VLM pilot: not a retraining reference',
           xticks=x,xticklabels=['Hybrid\n0.2','Hybrid\n0.6','Phase\n0.6','Mixed\n0.6'],ylim=(8e-9,1.1e-6))
    ax.grid(True,axis='y',alpha=.22);ax.legend(frameon=False,fontsize=7.5,loc='upper left')
    ax.text(.5,-.25,'Accuracy identical in all draws: val 73.5%, forget 71.5%',transform=ax.transAxes,ha='center',fontsize=7.4)
    fig.suptitle('One fixed round: energy regimes and one-step VLM fidelity',fontsize=12,y=.98)
    fig.subplots_adjust(left=.055,right=.985,top=.84,bottom=.27,wspace=.36)
    fig.text(.055,.045,'Same legacy round-91 updates/channels; 167.936 ms. Phase digital users retain NOMA. Lines guide the eye between tested budgets.\nVLM: 200 validation + 200 forget samples, 3 common Gaussian noise draws; error bars are noise SD, not independent training seeds. No forgetting guarantee.',fontsize=7.4)
    fig.savefig(out/'heartbeat_0212_summary.png',dpi=180)
    fig.savefig(out/'heartbeat_0212_summary.svg')
    plt.close(fig)


if __name__ == '__main__':
    import sys
    main(Path(sys.argv[1]).resolve())
