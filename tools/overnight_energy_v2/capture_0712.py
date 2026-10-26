"""Freeze completed retain-control, paid-codec and initialization evidence."""
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
results = root/'results/overnight_energy_v2'
tools = root/'tools/overnight_energy_v2'
suite = root/'results/comparisons_seed0_solver_v2'
quality_path = root/'results/energy_quality_v1'


def read(path):
    for attempt in range(10):
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            if attempt == 9:
                raise
            time.sleep(.02)


def write(path, obj):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(obj, indent=2)+'\n'); temporary.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


expected = read(suite/'suite_metadata.json')['source_sha256']
assert all(sha(root/n) == value for n, value in expected.items())
quality = read(quality_path/'energy_quality_report.json')
methods = {x['method']: x for x in quality['methods']}
assert methods['retain_only_ideal']['quality_status'] == 'evaluated'
audit = read(results/'retain_control_audit_v16/results.json')
certificates = read(results/'completed_transport_audit_v14/certificates.json')
assert certificates['checked_saved_feasible_candidates'] == 22
now = datetime.now().isoformat(timespec='seconds')
process_rows = subprocess.check_output(['ps', '-u', 'wnlab', '-o', 'pid,ppid,stat,etime,args'], text=True).splitlines()
project_processes = [x for x in process_rows if str(root)+'/' in x]
gpu = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader'], text=True)
burst_path = results/'burst_joint_v3/hybrid/rounds.jsonl'
burst_rows = [json.loads(x) for x in burst_path.read_text().splitlines() if x.strip()] if burst_path.exists() else []
snapshot = dict(captured_at_kst=now, source_sha256=expected,
                retained_control_status=read(results/'retain_control_status.json'),
                ideal_algorithm_controls={n: methods[n] for n in ['coupled_ideal', 'retain_only_ideal', 'retrain_reference_ideal']},
                paired_control_audit=audit, initialization_status=read(results/'initialization_v11/status.json'),
                initialization_comparisons=read(results/'initialization_v11/comparisons.json'),
                initialization_physical_certificates=certificates,
                public_history_float32_radio=read(results/'public_history_v13/radio/results.json'),
                public_history_float32_radio_summary=read(results/'public_history_v13/radio/summary.json'),
                public_history_fp16_status=read(results/'public_history_fp16_v15/status.json'),
                burst_status=read(results/'burst_joint_v3/status.json'),
                burst_completed_rounds=len(burst_rows), burst_last_round=burst_rows[-1] if burst_rows else None,
                project_processes=project_processes, gpu_processes=gpu,
                observed_frontier=quality['observed_frontier'], selection_status=quality['selection_status'])
if (results/'public_history_fp16_v15/results.json').exists():
    snapshot['public_history_fp16_partial_or_completed_results'] = read(results/'public_history_fp16_v15/results.json')
if (results/'public_history_fp16_v15/summary.json').exists():
    snapshot['public_history_fp16_summary'] = read(results/'public_history_fp16_v15/summary.json')
write(results/'heartbeat_0712_snapshot.json', snapshot)
state = read(tools/'campaign_state.json')
state.update(last_heartbeat_research_kst=now, current_gpu_method='burst_joint_v3/hybrid', current_gpu_pid=3602333,
             retain_only_ideal_completed=True, retain_only_ideal_final_quality_evaluated=True,
             retain_control_paired_audit_v16_completed=True, initialization_v11_completed=True,
             initialization_v11_physical_audit_completed=True, public_history_v13_paid_radio_completed=True,
             public_history_v13_rejected_for_gpu=True, public_history_fp16_v15_tests_passed=3,
             public_history_fp16_v15_gpu_promoted=False,
             last_completed_evidence_snapshot='results/overnight_energy_v2/heartbeat_0712_snapshot.json',
             notes='Retain-only ideal completed100 and canonical final audit; paired conditional sample bootstrap finds tiny forget-KL difference with interval crossing0, no MIA advantage, retained/validation loss worse with explicit FU. Not a seed-level equivalence or deletion result. Burst Hybrid is the one activeGPU job; Mixed/optimized-phase wait in original frozen queue. v11 best-per-architecture unchanged, one inferior6bit f0.5 phase incumbent improves2.271uJ;22saved candidates independently verified. v13 paid32bit history projections lose1.417%/0.772%, preserved/rejected. New shared-public-scale16bit coefficient CPU radio v15 continues without GPU promotion. No core/burst source edits. Morning flag unchanged; noon policy unchanged.')
write(tools/'campaign_state.json', state)
with (tools/'campaign_plan.txt').open('a') as f:
    f.write('\nHeartbeat07:12 evidence captured '+now+' KST\n'
            'Retain-only ideal completed100 and final canonical cacheaudit; validationloss0.93993418/acc73.8%, retainedloss0.92360919/acc74.9%, forgetKL0.01835310/disagreement5.3%, validationloss-MIA0.503144/gap0.000972. Coupledideal forgetKL0.01824249/disagreement5.1%, validationloss0.94428719. Pairedclass-stratified cache-only5000resamples: coupled-minus-retain forgetKL-0.000110614,95%conditionalinterval[-0.000339100,+0.000120842]; utilityloss+0.004353012,interval[+0.002865165,+0.005814664]; MIAgapdelta-0.000006,2000resampleinterval[-0.0003901,+0.0003610]. Conditional image-sample uncertainty only, not trainingseed/confirmatory/deletion guarantee. No test arrays used, no newGPU inference.\n'
            'v11 completed36cases;22saved feasible candidates pass independent physical watts/J/rate/totalerror/power/battery/time/mapping/noise checks.21selectedoldincumbents/1newincumbent;6bitphasef0.5 improves0.03354593167 to0.03354366075J(2.271uJ), but bestphasef0.125 and H/M minima unchanged for all4bit/tolerance pairs.\n'
            'v13 paid32bit publichistory projections completedall30headcases: baseline823.616uJ,rank1 835.287uJ(+1.417%),rank3 829.971uJ(+0.772%). Even before unmeasuredextraCPUenergy these fail and are not GPU promoted. v15 tests16bit coefficients with scale derivedonlyfrompastpublictheta; residual usesdecodedhalfvalue tocancelrounding. Paidtime/RF/allheads/sameoriginalabsoluteRMS, genericFLonly. CPU3602543 pendingatcaptureunlessstatuscompleted.\n'
            'Existing burst Hybrid started07:19:21 GPU3602333 afterretainaudit; existing Mixed thenoptimizedphase remainsequential. Modifiedrequest-roundtolerance0.6 vs original0.2 appliedequallytoallburstcomparators, versionsseparate. No duplicateGPU or existingcore/queuedsourcehashchanges.\n')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
selected = [('true_label_loss', 'validation', 'Validation loss difference'),
            ('kl_to_final_retrain', 'forget', 'Forget KL difference to retraining')]
for ax, (metric, cohort, title) in zip(axes, selected):
    row = next(x for x in audit['rows'] if x['metric'] == metric and x['cohort'] == cohort)
    point = row['coupled_minus_retain_only']; lower, upper = row['conditional_percentile_interval_95']
    ax.errorbar([point], [0], xerr=[[point-lower], [upper-point]], fmt='o', color='#326b87', capsize=7)
    ax.axvline(0, color='#999999', linewidth=1)
    ax.set_yticks([]); ax.set_ylim(-1, 1); ax.set_title(title)
    ax.set_xlabel('Coupled ideal minus retain-only ideal\n(lower is better)')
    ax.spines[['top', 'right', 'left']].set_visible(False)
fig.text(.5, .025, 'Fixed final models, one training seed; 1000 frozen samples/group. Intervals: paired class-stratified image bootstrap, not seed uncertainty or deletion guarantees.', ha='center', fontsize=7)
fig.tight_layout(rect=(0, .065, 1, 1))
for ext in ['png', 'svg']:
    fig.savefig(results/f'heartbeat_0712_control_audit.{ext}', dpi=170)
plt.close(fig)

lines = [
    'ova_CK 밤샘 연구 — 07:12 heartbeat 후속 결과', '실제 캡처: '+now+' KST', '',
    '1. 새로 끝난 핵심 대조군: 명시적 언러닝의 추가 이득은 아직 약함',
    '동일 seed/초기모델/데이터 일정의 retain_only_ideal 100라운드와 최종 재학습 기준 평가가 완료됐다. 삭제 요청 샘플을 이후 SGD에서 제외하되 별도의 FU 업데이트를 더하지 않는 대조군이다. 아래 세 방법은 이상적 링크 기준이므로 에너지 순위에 넣지 않는다.',
    '방법 | 검증 loss/정확도 | retained loss/정확도 | forget KL(ref||candidate)/예측 불일치 | validation loss-MIA AUC/재학습 AUC와 절댓값 차이']
for n in ['retain_only_ideal', 'coupled_ideal', 'retrain_reference_ideal']:
    x = methods[n]
    lines.append(f"{n} | {x['validation_loss']:.8f}/{100*x['validation_accuracy']:.1f}% | {x['retained_loss']:.8f}/{100*x['retained_accuracy']:.1f}% | {x['forget_kl_reference_to_candidate']:.8f}/{100*x['forget_prediction_disagreement']:.1f}% | {x['forget_loss_attack_auc_validation']:.6f}/{x['forget_attack_auc_gap_to_reference_validation']:.6f}")
lines += ['',
    '기존 최종 예측 캐시만 사용해 같은 샘플을 모델 간 짝지은 층화 부트스트랩으로 진단했다. 테스트셋 예측은 이 분석에 읽지 않았고, 추가 GPU 추론도 없다.',
    'coupled ideal − retain-only ideal: forget KL 차이 −0.000110614, 조건부 95% 구간 [−0.000339100, +0.000120842]. 검증 loss 차이 +0.004353012, 구간 [+0.002865165, +0.005814664]. MIA AUC gap 차이 −0.000006, 구간 [−0.000390100, +0.000361025].',
    '따라서 현재 forget 출력 차이와 MIA만으로 명시적 FU의 뚜렷한 추가 이득을 주장하기 어렵다. 검증/retained loss는 retain-only 쪽이 더 낮다. 이는 단일 학습 seed의 고정 모델에 대한 샘플 조건부 진단이며, 여러 seed의 불확실성·통계적 동등성·삭제 보장을 뜻하지 않는다. 삭제 샘플 노출이 대부분 1회인 기존 약한 기억 조건도 한계다.', '',
    '2. 초기화/AO 검사 완료 및 독립 물리 단위 검증',
    '36개 설정(8/6비트 × RMS 비율0.2/0.6 × Hybrid/Mixed/7개 분리 전송 시간 비율)의 CPU 검사가 완료됐다. 모든 방법에 기존 4개 시작점과 추가 convex 시작점을 적용하고 AO 상한을 30회로 늘렸다.',
    '22개 저장된 실행 가능 후보에 대해 원래 h,b,c와 W/J 단위로 에너지·SIC payload·전력·배터리·전체 RMS·잡음·시간·매핑 조건을 최적화 코드와 독립적으로 재구성해 통과했다. 나머지 실패는 그대로 보존했다. 이는 전역 최적성이나 실제 하드웨어 측정이 아니다.',
    '6비트·RMS0.2·50/50 분리 전송만 33.545932→33.543661mJ(2.271µJ)로 조금 개선됐다. 하지만 방법별 최저 에너지와 최적 시간 비율0.125는 네 조합 모두 이전 값 그대로다. 초기화 변경으로 Hybrid의 열세가 해소되지는 않았다.', '',
    '3. 공개 모델 이력 압축의 비용 포함 검사: 32비트 계수 방식 탈락',
    '실제 PUF 궤적의 round100 learning-only 업데이트를 고정했다. 모든 방법에 같은 공개 이력을 주고, 원래 전체 167.936ms에서 계수 전송 시간을 차감하고 RF 에너지·전력·배터리까지 계상했다. 각 방식 10개 head를 모두 검사했다.',
    '방식 | 데이터+기존 overhead | 추가 계수 RF | 총 에너지 | 원래 대비',
    '원래 업데이트 | 823.616µJ | 0 | 823.616µJ | 기준',
    '최근 방향1개 | 820.505µJ | 14.782µJ | 835.287µJ | +1.417%',
    '최근 방향3개 | 785.624µJ | 44.346µJ | 829.971µJ | +0.772%',
    '잔여 업데이트 norm 감소만 보고 에너지 절감을 주장하면 잘못된다. 두 방식 모두 추가 계산 에너지를 넣기 전부터 불리해서 GPU 후보로 올리지 않았다.',
    '별도 v15는 공개 theta99−theta98 norm으로 정한 공유 scale과 16비트 계수를 검사 중이다. 송신자가 복호된 계수로 residual을 다시 계산하여 양자화 반올림은 이상적 합계에서 상쇄되도록 했다. 동일 총시간·절대 RMS·전력·배터리 아래 실제 계수 RF 비용을 지불한다. 3개 codec/시간/rate 검증을 통과했다. 현재 상태: '+snapshot['public_history_fp16_status']['status']+'.',
    '이 조건은 FU 요청이 없는 한 라운드이고 모든 전송 구조가 같은 아날로그 문제로 환원된다. 일반 FL 개선 후보일 뿐 Hybrid의 Mixed 대비 우위가 아니다. 추가 계산/헤더/유한 블록 오류 비용이 미측정이므로 순수 RF 절감만으로 전체 시스템 에너지 우위를 주장할 수 없다.', '',
    '4. 완료된 기존 무선 실험의 현재 비교(원래 RMS0.2, seed0)',
    '방법 | 100라운드 통신 에너지 상한J | 검증 loss | forget KL | MIA AUC gap']
for n in ['coupled_mixed_aircomp', 'coupled_hybrid_q6', 'puf_regular_adapted_hybrid', 'coupled_hybrid', 'coupled_fixed_head']:
    x = methods[n]
    lines.append(f"{n} | {x['energy_bound_j']:.9f} | {x['validation_loss']:.8f} | {x['forget_kl_reference_to_candidate']:.8f} | {x['forget_attack_auc_gap_to_reference_validation']:.6f}")
lines += [
    '위 5개는 100/100 라운드와 5/5 요청 라운드 전달을 완료했다. 전달은 삭제 보장이 아니다. PUF adapted는 요청 UAV의 learning을 생략하고 FU 가중치도 달라 순수 고정 업데이트 무선 비교가 아니다.',
    'Mixed가 이번 seed의 에너지·검증 loss·forget KL 세 지표에서는 가장 낮다. MIA gap까지 모두 지배하거나 여러 seed에서 우수하다는 뜻은 아니다. 품질 통과 임계값을 사전에 정하지 않았으므로 공식 최종 승자로 선언하지 않는다. ideal 에너지0, 실패·미완료 결과는 에너지 순위에서 제외했다.',
    '기존 50/50 분리 전송은 91라운드에서 실패(90라운드 완료), uncoupled Hybrid는 57라운드 중단이다. 분리 전송의 다른 시간 배분까지 모두 불가능하다는 뜻은 아니다. 원본 결과와 manuscript는 보존했다.', '',
    '5. 현재 큐와 후속 판단',
    f"retain-only 최종 평가 후 07:19:21에 예약된 burst Hybrid가 시작됐다. 캡처 기준 {len(burst_rows)}/100 라운드. 현재 GPU PID3602333 하나, v15는 CPU PID3602543이다.",
    'burst는 요청 라운드의 RMS 비율을0.6으로 완화하고 기존 라운드는0.2를 쓰는 별도 문제 조건이다. Hybrid→Mixed→시간 배분 최적화 분리 전송 순으로 동일 조건에서 실행한다. 원래0.2 실험의 품질과 직접 동일하다고 취급하지 않는다. 기존 학습 및 burst 소스 해시는 변경하지 않았고 중복 학습도 없다.',
    '결과를 유리하게 만들기 위해 Mixed를 제외하거나 아직 응용 이득이 확인되지 않은 별도 U 정밀도 제약을 소급 도입하지 않는다. 실제로 개선된 CPU 후보만 검증 후 별도 GPU 큐에 고려한다. 현재 v15는 GPU 승격하지 않았다.',
    '오전9시 이후 첫 heartbeat에는 현 최선/한계/비교표를 한 번 보고한다. 정오에는 최종 현황을 보고하고 heartbeat를 일시중지한다. 정오 전에 시작된 계산은 안전하게 완료하게 두되 이후 새 장시간 실험은 추가하지 않는다.', '',
    '증거: heartbeat_0712_results.zip 내부 snapshot, retain_control_audit_v16, completed_transport_audit_v14, initialization_v11, public_history_v13/radio, public_history_fp16_v15 및 캐시 provenance를 함께 보존했다.']
report_path = results/'heartbeat_0712_report_ko.txt'
report_path.write_text('\n'.join(lines)+'\n')

paths = [results/'heartbeat_0712_snapshot.json', report_path, results/'heartbeat_0712_control_audit.png', results/'heartbeat_0712_control_audit.svg',
         tools/'campaign_plan.txt', tools/'campaign_state.json', tools/'capture_0712.py',
         tools/'completed_transport_audit_v14.py', tools/'retain_control_audit_v16.py',
         quality_path/'energy_quality_report.json', quality_path/'energy_quality.csv', quality_path/'cohorts.json',
         results/'retain_control_status.json', results/'retain_control_plan.json',
         suite/'retain_only_ideal/rounds.jsonl', suite/'retain_only_ideal/manifest.json', suite/'retain_only_ideal/latest.pt',
         results/'burst_joint_v3/status.json', results/'burst_joint_v3/plan.json', results/'burst_joint_v3/config.json',
         results/'queue_recovery_v1/status.json']
if burst_path.exists():
    paths.append(burst_path)
for n in ['coupled_ideal', 'retain_only_ideal', 'retrain_reference_ideal']:
    paths.extend([quality_path/(n+'.cache.json'), quality_path/(n+'.npz')])
for name in ['completed_transport_audit_v14', 'retain_control_audit_v16', 'initialization_v11', 'public_history_v13', 'public_history_fp16_v15']:
    for parent in [results, tools]:
        paths.extend(p for p in (parent/name).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.tmp')
contents = {}; omitted = []
for path in dict.fromkeys(paths):
    for attempt in range(5):
        try:
            data = path.read_bytes()
            if path.suffix == '.json':
                json.loads(data)
            if path.suffix == '.npz':
                with zipfile.ZipFile(io.BytesIO(data)) as archive:
                    assert archive.testzip() is None
            contents[str(path.relative_to(root))] = data; break
        except (OSError, json.JSONDecodeError, zipfile.BadZipFile, AssertionError):
            time.sleep(.02)
    else:
        omitted.append(str(path.relative_to(root)))
manifest = dict(captured_at_kst=now, ongoing_files_are_partial=True, in_progress_files_omitted=omitted,
                files={name: dict(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data)) for name, data in contents.items()})
buffer = io.BytesIO()
with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
    for name, data in contents.items():
        archive.writestr(name, data)
    archive.writestr('capture_manifest.json', json.dumps(manifest, indent=2)+'\n')
print(base64.b64encode(buffer.getvalue()).decode())
