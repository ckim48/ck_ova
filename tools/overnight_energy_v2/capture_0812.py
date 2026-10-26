"""Archive earlier-stage reproduction, quality-pilot repair and current queue."""
import base64
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import subprocess
import time
import zipfile

root = Path('/home/wnlab/Documents/Codex/2026-10-20/files-pasted-by-the-user-standalone/outputs/ova_CK')
tools = root/'tools/overnight_energy_v2'; results = root/'results/overnight_energy_v2'
suite = root/'results/comparisons_seed0_solver_v2'


def read(path):
    for attempt in range(10):
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            if attempt == 9:
                raise
            time.sleep(.02)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(value, indent=2)+'\n'); tmp.replace(path)


expected = read(suite/'suite_metadata.json')['source_sha256']
assert all(sha(root/n) == v for n, v in expected.items())
burst_plan = read(results/'burst_joint_v3/plan.json')
assert all(sha(root/n) == v for n, v in burst_plan['source_sha256'].items())
early = results/'public_history_early_v17/radio'
certificates = read(early/'physical_certificates.json')
assert certificates['checked_saved_radio_designs'] == 30
rows17 = read(early/'results.json'); rows15 = read(results/'public_history_fp16_v15/results.json')
now = datetime.now().isoformat(timespec='seconds')
burst_path = results/'burst_joint_v3/hybrid/rounds.jsonl'
burst_rows = [json.loads(x) for x in burst_path.read_text().splitlines() if x.strip()]
requests = [x for x in burst_rows if x['newly_deleted'] > 0]
processes = subprocess.check_output(['ps', '-u', 'wnlab', '-o', 'pid,ppid,stat,etime,args'], text=True).splitlines()
snapshot = dict(captured_at_kst=now, core_sha256=expected, frozen_burst_source_sha256=burst_plan['source_sha256'],
                early_screen=read(results/'public_history_early_v17/screen/results.json'),
                early_radio_status=read(early/'status.json'), early_radio_results=rows17,
                early_radio_summary=read(early/'summary.json'), early_physical_certificates=certificates,
                late_radio_results=rows15, late_radio_summary=read(results/'public_history_fp16_v15/summary.json'),
                failed_quality_pilot=read(results/'history_quality_v18/plan_comparison_failure.json'),
                repaired_quality_status=read(results/'history_quality_v18b/status.json'),
                fu_observer_status=read(results/'public_history_fu_v19/observer/status.json'),
                burst_status=read(results/'burst_joint_v3/status.json'), burst_completed_rounds=len(burst_rows),
                burst_request_rounds_delivered=len(requests), burst_last_round=burst_rows[-1],
                gpu_processes=subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader'], text=True),
                project_processes=[x for x in processes if str(root)+'/' in x])
if (results/'history_quality_v18b/results.json').exists():
    snapshot['quality_partial_or_completed_results'] = read(results/'history_quality_v18b/results.json')
if (results/'history_quality_v18b/summary.json').exists():
    snapshot['quality_summary'] = read(results/'history_quality_v18b/summary.json')
write(results/'heartbeat_0812_snapshot.json', snapshot)
state = read(tools/'campaign_state.json')
state.update(last_heartbeat_research_kst=now, public_history_early_v17_completed=True,
             public_history_early_v17_physical_audit_passed=True,
             public_history_early_v17_radio_saving_fraction=0.053750456286254134,
             last_completed_evidence_snapshot='results/overnight_energy_v2/heartbeat_0812_snapshot.json',
             current_gpu_method='burst_joint_v3/hybrid', current_gpu_pid=3602333,
             notes='Actual preselected round30 paid-half rank3 reproduces modest genericFL gain5.37505% vs1.79385% atlate100. Independent30design physical/codec auditpasses; rank1 loses, preserved. Extra networkcompute/headers unmeasured,39.061uJ early break-even. v18 failed beforeinference because2displaynorms differoneBLASulp, fullvectorhashes identical; originalfailure preserved. Repairedv18bCPU3616240 validates normonlywith1e-12relative tolerance, ALLupdatehashes/inputs/source/protocol still exact, dryplanspassdefault/4thread. Sevenvector CPUtaskpilot pending; no automaticGPU. v19readonlyobserver3615412 capturespublic57..60 for actualrequest61 CPUfollow-up afterv18b; no core/burst edits or extraGPU. Morningflag/noonpolicy unchanged.')
write(tools/'campaign_state.json', state)
with (tools/'campaign_plan.txt').open('a') as f:
    f.write('\nHeartbeat08:12 evidence captured '+now+' KST\n'
            'v17 earlieractualround30 baseline726.717542uJ,16bitrank3 687.656142uJ(5.37505%saving,39.061399uJ); rank1 735.615411uJ loses1.22439%, preserved. All30designs passindependentphysicalunits/half-bitreconstruction; sameabsoluteRMS0.0055862815,total167.936ms,peak.32W,battery.1J/all10heads/AO30. No FUrequestat30; genericFLonly, not Hybrid/NOMA-over-Mixed or fulltrajectory/forgetting advantage. Extra compute/basis/receiver/coding/headersmustfit39.061uJ networkbudget; nohardwareenergymeasurement.\n'
            'Qualitypilot v18 failedbeforeanyinference atstrictplancomparison because two displaynorms differoneulp withBLASthreadcount; originalplan/source/console/diff retained. Newv18b hasSAMEvectors/designs/cohorts/noise, exactallvectorhashes andallotherprotocol/sourcefields; toleranceONLYfordisplaynorms1e-12rtol/1e-15atol. Existing3recoverytests unchanged, plansverifiedbothdefaultand4threads; CPU3616240 nowruns7fixedvectors onfrozen200utility+200futureforgetprobes. Round30beforeFU: notactualforgetting, noMIA/test/GPUclaim.\n'
            'v19readonlyobserver3615412 pre-registersactualrequest61 beforepublic57..60. AftercurrentCPUquality, replayactual L+coupledU atpublictheta60 forpairedcodec/strongHybrid/Mixed/phase comparisons. No freeprivatehistory, metadata/time/RFpaid, no automaticGPU. Observeronlycurrently; inspectcompletedstates beforeauthoring/runningreplay, avoidduplicates. GPU3602333 existingburstHybrid continues, originalMixed/optimizedphase queued, all frozen sources verified. Morningflagstillfalse, noonpolicy unchanged.\n')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
for ax, rows, title in zip(axes, [rows17, rows15], ['Round30: actual burst trajectory', 'Round100: actual PUF trajectory']):
    best = [x['best'] for x in rows]
    labels = ['Baseline', '16-bit rank1', '16-bit rank3']
    overhead = [150.]*3
    data = [x['analog_data_plus_original_overhead_j']*1e6-150 for x in best]
    control = [x['coefficient_tx_energy_j']*1e6 for x in best]
    ax.bar(labels, overhead, color='#b0b0b0', label='Original fixed overhead')
    ax.bar(labels, data, bottom=overhead, color='#40778e', label='Analog data RF')
    ax.bar(labels, control, bottom=[x+150 for x in data], color='#ce8841', label='Paid coefficient RF')
    for i, row in enumerate(best):
        value = row['total_modeled_energy_j']*1e6
        ax.text(i, value+14, f'{value:.2f}', ha='center', fontsize=9)
    ax.set_ylim(0, 1000); ax.set_title(title, fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
axes[0].set_ylabel('Modeled communication energy (microjoules)')
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .93), ncol=3, frameon=False, fontsize=8)
fig.suptitle('Paid public-history codec: two fixed learning-only updates', y=.995, fontsize=12)
fig.text(.5, .018, 'Same time/error/power/battery within each fixture. Extra compute/headers unmeasured. No FU at either step; no full-trajectory or Hybrid superiority claim.', ha='center', fontsize=7)
fig.tight_layout(rect=(0, .055, 1, .85))
for ext in ['png', 'svg']:
    fig.savefig(results/f'heartbeat_0812_reproduction.{ext}', dpi=170)
plt.close(fig)

report = f'''ova_CK 밤샘 연구 — 08:12 heartbeat 후속 결과
실제 캡처: {now} KST

1. 사전 지정한 학습 초반 업데이트에서도 압축 절감 재현
공개 모델26·27·28·29가 나오기 전에 실제 burst Hybrid의 round30을 검사 대상으로 고정했다. GPU 학습을 수정하지 않는 observer가 공개29까지 기록했고, CPU에서 실제 round30 SGD minibatch와 동일한 가중치·learning rate0.0028225873으로 로컬 업데이트를 재현했다. 마지막 round100의 learning rate 약8.64e−7보다 훨씬 큰 업데이트 조건이다. 다른 방법의 private gradient나 미래 모델을 쓰지 않았다.

round30 방식 | 데이터+원래 overhead µJ | 계수 RF µJ | 총 통신 에너지 µJ | baseline 대비
원래 업데이트 | 726.717542 | 0 | 726.717542 | 기준
16비트 rank1 | 728.224365 | 7.391046 | 735.615411 | +1.22439%
16비트 rank3 | 665.483004 | 22.173138 | 687.656142 | −5.37505%

이전 실제 PUF round100에서는 rank3의 총 에너지가823.615818→808.841386µJ로1.79385% 감소했다. 초반 round30에서도 같은 공유 scale/16비트 codec 규칙으로 비용 포함 절감이 재현됐다. 서로 다른 궤적의 두 고정 업데이트 결과이며 전체 학습의 누적 절감률은 아니다. rank1의 실패도 그대로 보존했다.

2. 독립 물리 검증 완료 및 해석 한계
초반 검사에서도 모든 방식에 동일한 절대 RMS0.0055862815, 전체167.936ms, peak전력0.32W, UAV별battery0.1J, 기존fixed overhead, AO30/5시작점과10개head검색을 적용했다. rank3 계수 총432bits/864TDMA symbols의 시간과RF22.173138µJ를 모두 지불했다. 최저에너지head는 모두5다. 원래/압축 analytic RMSE는0.0041339333/0.0039270928이며 보수적 전체 RMS한도 내에 있다.
30개 저장 설계를 solver와 독립적으로 원래 h,b,c와W/J로 재구성했다. 실제 half비트 복호와 residual 재구성, payload/rate/RF비용, peak/배터리, 시간/coherence/isometric매핑, 전체 RMS/실제 벡터 MSE/단조 에너지 이력을 모두 확인해 통과했다. 유한 반복의 실행 가능 설계 검증이며 전역 최적성·실제 장비 측정이 아니다.
추가 projection·basis생성·메모리·계수 복호·헤더·유한블록오류 비용은 미측정이다. round30의 네트워크 전체 추가비용이39.061399µJ 미만이어야 순이득이 남는다. 각 UAV의 projection/residual 계산만 세어도2457720FLOPs이며 이 작업만 고려한 허용 추가비용은약15.89pJ/FLOP다. 실제 UAV 하드웨어가 이 조건을 만족한다고 단정하지 않는다.
두 라운드 모두 FU 요청이 없으므로 Hybrid/Mixed/분리 전송이 같은 아날로그 합계 문제로 환원된다. 모든 비교군에 같은 공개이력·codec을 적용할 수 있는 일반 FL 개선 후보다. NOMA 우위나 실제 언러닝 개선을 입증한 것이 아니고 작은 MSE가 삭제 보장은 아니다.

3. 실제 VLM 출력 검사를 별도 CPU 버전에서 진행
실제theta29와 round30 업데이트를 사용해 ideal한1단계와 물리적인 원래/압축rank3를 비교한다. 물리 방식별 공통 Gaussian noise3회(30/1030/2030), 총7개 vector를 출력 검사 전에 고정했다. 기존200개 utility-validation과200개 future-forget probe를 그대로 사용한다. 테스트셋 선택이나 새GPU 추론은 없고 작은 pilot으로 MIA를 추정하지 않는다. round30은 삭제 요청 전이므로 future-forget 출력 민감도는 실제 forgetting/요청처리 검증이 아니다. 최종 재학습 기준 gap은 서로 다른 학습단계의 참고값이다.
첫 v18 실행은 표시용 realized error norm 두 개의 마지막 소수점1ulp 차이를 strict plan비교가 잡아 추론 전에 중단됐다. update vector SHA256는 전부 동일했다. 원본 실패 plan/source/console/차이 기록을 보존했다. v18b는 표시용 norm만rtol1e−12/atol1e−15로 비교하며 update vector해시·입력·코드·나머지protocol은 그대로 정확히 검사한다. 변경 없는3개 복구/잡음/padding검증은 이미 통과했고, default/4thread 두 환경에서 plan검증을 확인했다.
현재CPU PID3616240, 상태: {snapshot['repaired_quality_status']['status']}. 모든 후보는 자동GPU 승격하지 않으며, 결과 완료 후 품질과 비용을 함께 판단한다.

4. 실제 FU 요청에서도 적용 가능한지 후속 관찰 시작
초반 learning-only 결과를 FU로 일반화하지 않기 위해 실제 request61을 별도 CPU 대상으로 사전에 정했다. observer PID3615412는 현재 진행 중인 burst Hybrid의 공개57·58·59·60을 읽기만 한다. 캡처 상태: {snapshot['fu_observer_status']['status']}, 최신공개{snapshot['fu_observer_status']['last_round']}.
현재 VLM CPU검사 후 공개theta60에서 실제request61의 learning/coupled unlearning 업데이트를 재현한다. Hybrid/분리 전송은L, Mixed는L+U를 같은 공개basis·codec으로 다루며 원래 방식도 비교군으로 유지한다. 같은시간·절대오차·전력·배터리 및 충분한초기화/반복, 실제계수전송비를 모든 비교군에 적용한다. 현재는 관찰만 실행 중이며 아직 request61 재현/전송 결과나 전체학습 우위를 주장하지 않는다.

5. 기존 GPU 큐와 원래 완료 비교
기존 burst Hybrid PID3602333 한 개만GPU에서 실행 중이다. 캡처 기준{len(burst_rows)}/100라운드, 도착한요청라운드{len(requests)}개 전달완료다. 전달은 삭제 보장이 아니다. Mixed와시간배분최적화분리전송은 원래 큐에서 순차 실행한다. burst의요청라운드 RMS0.6/그외0.2 조건은 모든 비교군에 동일하며 원래RMS0.2 완료실험과 섞지 않는다. original/core/burst 고정소스 해시를 모두 확인해 변경이 없었다.
원래0.2 완료실험의 이번seed에서는 Mixed가 통신에너지상한0.087379794J, 검증loss0.96706282, forgetKL0.02574322로 이세지표의 최저다. 6비트Hybrid0.189836125J, 8비트Hybrid0.213161276J다. MIA까지모두지배하거나 여러seed에서우월함을 뜻하지않는다. ideal에너지0과실패/미완료는에너지우위순위에서제외했다. 상세 retained utility/KL/예측차이/MIA/완료율은 동봉한기존canonical report에보존했다.
이상적 링크에서명시적FU와retain-only의추가forgetKL/MIA이득이뚜렷하지않았다는기존한계도그대로유지한다. manuscript와원본결과는수정하지않았다. 오전9시이후첫heartbeat에최선/한계/비교표를한번보고하고, 정오에는현황을보고한뒤heartbeat를일시중지한다. 정오이후새장시간실험은추가하지않는다.

증거: heartbeat_0812_results.zip 안의v17전체결과/30설계/관찰/원래minibatch재현/provenance, v18실패와v18b수정/현재상태, v19사전관찰계획, 현재GPU큐/동결소스해시.
'''
(results/'heartbeat_0812_report_ko.txt').write_text(report)
paths = [results/'heartbeat_0812_snapshot.json', results/'heartbeat_0812_report_ko.txt',
         results/'heartbeat_0812_reproduction.png', results/'heartbeat_0812_reproduction.svg',
         tools/'campaign_plan.txt', tools/'campaign_state.json', tools/'capture_0812.py',
         root/'results/energy_quality_v1/energy_quality_report.json', suite/'suite_metadata.json',
         results/'burst_joint_v3/status.json', results/'burst_joint_v3/plan.json', results/'burst_joint_v3/config.json', burst_path,
         results/'queue_recovery_v1/status.json', results/'public_history_fp16_v15/results.json',
         results/'public_history_fp16_v15/summary.json', results/'public_history_fp16_v15/physical_certificates.json']
for name in ['public_history_early_v17', 'history_quality_v18', 'history_quality_v18b', 'public_history_fu_v19']:
    for parent in [results, tools]:
        paths.extend(p for p in (parent/name).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.tmp')
contents = {}; omitted = []
for path in dict.fromkeys(paths):
    for attempt in range(5):
        try:
            data = path.read_bytes()
            if path.suffix == '.json':
                json.loads(data)
            if path.suffix in ['.npz', '.pt']:
                with zipfile.ZipFile(io.BytesIO(data)) as archive:
                    assert archive.testzip() is None
            contents[str(path.relative_to(root))] = data; break
        except (OSError, json.JSONDecodeError, zipfile.BadZipFile, AssertionError):
            time.sleep(.02)
    else:
        omitted.append(str(path.relative_to(root)))
manifest = dict(captured_at_kst=now, ongoing_files_are_partial=True, in_progress_files_omitted=omitted,
                files={n: dict(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data)) for n, data in contents.items()})
buffer = io.BytesIO()
with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
    for name, data in contents.items():
        archive.writestr(name, data)
    archive.writestr('capture_manifest.json', json.dumps(manifest, indent=2)+'\n')
print(base64.b64encode(buffer.getvalue()).decode())
