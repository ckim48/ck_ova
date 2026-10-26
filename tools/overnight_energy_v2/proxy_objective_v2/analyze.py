"""Postprocess this single already-running campaign; never starts training."""
from pathlib import Path
import json,time,os,sys
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'results/overnight_energy_v2/proxy_objective_v2'

def build(final):
    source=json.loads((OUT/'summary.json').read_text())
    arms={r['method']:r for r in source['methods']}
    proposed=arms['proposed'];comparisons=[]
    for key in ['aircomp_only','aircomp_oma','oma_only']:
        other=arms[key];p=proposed['joint_target'];q=other['joint_target']
        r=dict(comparator=key,proposed_qualified=p is not None,comparator_qualified=q is not None)
        if p and q:
            a=p['confirmation'];b=q['confirmation']
            r.update(energy_saving_percent=100*(1-a['energy_bound_j']/b['energy_bound_j']),
              expected_energy_saving_percent=100*(1-a['energy_expected_j']/b['energy_expected_j']),
              uplink_time_saving_percent=100*(1-a['uplink_s']/b['uplink_s']),
              proposed_confirmation_round=a['round'],comparator_confirmation_round=b['round'],
              interpretation='Positive saving favors proposed; both satisfy the same joint target, proxy costs included.')
        else:r['interpretation']='No common qualified energy-to-target ranking yet; misses are censored.'
        comparisons.append(r)
    report=dict(final=final,created_unix=time.time(),source=source,comparisons=comparisons,
      scope='Single seed0, validation development study. Small KL/MIA is not a deletion certificate. CPU/GPU runtime is not uplink duration.')
    name='final_analysis' if final else 'first40_analysis'
    (OUT/(name+'.json')).write_text(json.dumps(report,indent=2)+'\n')
    lines=['새 VLM 목적함수와 전송 방식 비교', '전체200라운드 완료' if final else '40라운드 중간 결과',
      '누적 TX 비용과 채널 점유시간에는 각 방식의 대리 모델20라운드 생성 비용이 포함됨.',
      '주 비교 네 방식 모두 동일한 목적함수/교차모달 보존 항. 별도 no-crossmodal ablation 포함.',
      '방식 | 최종라운드 | 보존mR | 보존loss | forgetKL | 누적TX상한J | 누적uplink초 | 공동목표확정라운드']
    for r in source['methods']:
        last=r.get('last',{});hit=r['joint_target']
        lines.append(f"{r['label']} | {last.get('round')} | {last.get('retained_mR')} | {last.get('retained_loss')} | {last.get('forget_KL')} | {last.get('energy_bound_j')} | {last.get('uplink_s')} | {hit['confirmation']['round'] if hit else '미달/미확정'}")
    lines+=['','제안법의 공동목표 달성 비용 비교:']
    for r in comparisons:
        if 'energy_saving_percent' in r:
            lines.append(f"{r['comparator']}: TX상한에너지 절감률 {r['energy_saving_percent']:.3f}%, 기대에너지 절감률 {r['expected_energy_saving_percent']:.3f}%, uplink시간 절감률 {r['uplink_time_saving_percent']:.3f}%. 양수만 제안법 우위.")
        else:lines.append(f"{r['comparator']}: 공통 목표에 대한 순위 확정 불가. 제안법 달성={r['proposed_qualified']}, 비교군 달성={r['comparator_qualified']}.")
    lines+=['','미완료/목표미달을 에너지0인 우수 결과로 취급하지 않는다. 단일seed의 개발 결과이며 삭제 보장이 아니다.',
      '계산/SIC/수신/다운링크/비행 비용은 사용자 정의 TX-only 지표에서 제외되며 전체 시스템 비용과 구분한다.']
    (OUT/(name+'_ko.txt')).write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
    deadline=time.time()+8*3600;early=False
    while time.time()<deadline:
        status=json.loads((OUT/'status.json').read_text())
        if status['stage']=='failed':
            (OUT/'postprocess_failure.json').write_text(json.dumps(status,indent=2));sys.exit(1)
        if not early and (OUT/'first40_complete.json').exists():build(False);early=True
        if (OUT/'finished.json').exists():build(True);print('FINAL_ANALYSIS_WRITTEN',flush=True);break
        time.sleep(20)
    else:raise TimeoutError('Campaign still pending after8h; no results fabricated.')
