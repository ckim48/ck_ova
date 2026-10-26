"""Freeze completed one-step task sensitivity and ongoing concurrentFU check."""
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


def write(path, obj):
    tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(obj, indent=2)+'\n'); tmp.replace(path)


expected = read(results/'burst_joint_v3/plan.json')['source_sha256']
assert all(sha(root/n) == value for n, value in expected.items())
qdir = results/'history_quality_v18b'
quality_rows = read(qdir/'results.json'); assert len(quality_rows) == 7
metrics = read(qdir/'independent_metrics.json'); assert len(metrics['certificates']) == 14
fu = results/'public_history_fu_v19'
now = datetime.now().isoformat(timespec='seconds')
burst_path = results/'burst_joint_v3/hybrid/rounds.jsonl'
burst_rows = [json.loads(x) for x in burst_path.read_text().splitlines() if x.strip()]
processes = subprocess.check_output(['ps', '-u', 'wnlab', '-o', 'pid,ppid,stat,etime,args'], text=True).splitlines()
snapshot = dict(captured_at_kst=now, frozen_burst_sources=expected,
                quality_status=read(qdir/'status.json'), quality_summary=read(qdir/'summary.json'),
                independent_quality_audit=metrics, quality_rows=quality_rows,
                fu_observer=read(fu/'observer/status.json'), fu_registered_plan=read(fu/'screen/plan.json'),
                burst_status=read(results/'burst_joint_v3/status.json'), burst_completed_rounds=len(burst_rows),
                burst_last=burst_rows[-1], burst_requests=[x for x in burst_rows if x['newly_deleted'] > 0],
                gpu_processes=subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader'], text=True),
                project_processes=[x for x in processes if str(root)+'/' in x])
if (results/'public_history_fu_v19b/waiting_status.json').exists():
    snapshot['phase_boundary_waiting_status'] = read(results/'public_history_fu_v19b/waiting_status.json')
for sub in ['screen/status.json', 'screen/results.json', 'radio/status.json', 'radio/results.json', 'radio/comparisons.json']:
    if (fu/sub).exists():
        snapshot['fu_'+sub.replace('/', '_').replace('.json', '')] = read(fu/sub)
write(results/'heartbeat_0842_snapshot.json', snapshot)
state = read(tools/'campaign_state.json')
state.update(last_heartbeat_research_kst=now, history_quality_v18b_completed=True, history_quality_v18b_independent_audit_passed=True,
             public_history_fu_v19_observer_completed=True, current_gpu_method='burst_joint_v3/hybrid', current_gpu_pid=3602333,
             last_completed_evidence_snapshot='results/overnight_energy_v2/heartbeat_0842_snapshot.json',
             notes='v18b one-stepCPUquality complete7vectors/14cohortpredictions independently audited. Atactualround30 bothvalaccuracy72.5%, rank3 val loss1.08000129 vs1.08004708, KLtoideal1.35522e-5 vs1.69556e-5; finalretrainKL slightlylarger and retainedraw/descriptive becausepreFU. Not fulltraining/equivalence/deletion. v19actualrequest61 replay-radioCPU3621851 startedaftercompletedobserver60/v18b; allH/M/7phasefractions,ranks0/1/3,epsilon0.2/0.6,paidhalfmetadata/time/RF,originalL-basedabsoluteepsilon,AO30/5starts. Twoframe/FUhead/Ucodec tests andplanpassed. v19bCPU3622815 waits3621851 then adds4 exactminimum-L_A=d compressedphase cases becausecoefficientairtime movesphysicalfractionboundary; mustmergeallincumbentsBEFOREenergycomparisons. CPU/CUDArandom/numericdifferences acknowledged, pairedtransportsuseidenticalfrozenCPUvectors. ExistingburstGPU3602333alone, othermodesqueued, frozenhashesintact. Morningflag/noonpolicy unchanged.')
write(tools/'campaign_state.json', state)
with (tools/'campaign_plan.txt').open('a') as f:
    f.write('\nHeartbeat08:42 evidence captured '+now+' KST\n'
            'v18b complete08:29:32, independentall7updatevectorSHA256and14cohortprobability/label-loss/accuracy/idealKL/finalKL checks passed08:47:33. Rank3val loss1.0800012877 vsbaseline1.0800470830, accuracy72.5both, idealstepKL1.355223e-5 vs1.695563e-5; finalretrainKL0.0453239053 vs0.0453053338 slightlyworse. Futureforgetprobe loss1.1380913845 vs1.1381661423, idealKL1.302118e-5 vs1.598034e-5, finalretrainKL0.0410754053 vs0.0410501974 slightlyworse. Allrawreferences kept; round30beforeFU so neitherfinalKLdirection provesforgetting. Threecommondrawsnottrainingseeds/equivalence. No MIA/test/newGPU.\n'
            'v19CPU3621851 actualrequest61 replay+54casepaidradio nowactive, observerpublic57..60complete. Head-ownedFUphase/no-remotepayload andno-free/double-controltime tests passed; residualcodec cancels exactidealquantization anddoesnotchangeUcodec. Allmode/rankepsilonfixedfromoriginalL, paidcontrolRF/time, sameP/battery/coherence/AO30/starts, uncompressed/rank1/phasebaselinefailuresretained. v19bCPU3622815 waitsthenadds4 exactminimum-L_A=d phaseboundary cases: positivecodecairtime makesfixedf0.125slightlytoosmall. Originalrank0f0.125alreadycoversthatboundary. Mustmergeeligibleincumbentsbeforeanyenergycomparison, neverweakencompressedphase. No taskquality/energywinner beforecomplete; no automaticGPU. ExistingburstH GPU3602333 progresses, originalM/phasequeues/hashessafe; 09morningreportflagstillfalse/noonunchanged.\n')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
noise = [30, 1030, 2030]
base = {x['noise_seed']: x for x in quality_rows if x['method'].startswith('unprojected_baseline_')}
candidate = {x['noise_seed']: x for x in quality_rows if x['method'].startswith('recent_public_span_rank3_')}
for ax, key, title in zip(axes, ['gap_to_ideal_step', 'gap_to_final_retrain_descriptive_only'],
                          ['Validation KL to ideal step', 'Validation KL to final retrain (different stage)']):
    delta = [(candidate[s]['validation'][key]['kl_reference_to_candidate']-base[s]['validation'][key]['kl_reference_to_candidate'])*1e6 for s in noise]
    ax.bar([str(s) for s in noise], delta, color=['#548567' if x < 0 else '#a5674b' for x in delta])
    ax.axhline(0, color='#999999', linewidth=1)
    for i, value in enumerate(delta):
        ax.text(i, value+(0.35 if value >= 0 else -0.2), f'{value:+.2f}', ha='center', va='bottom' if value >= 0 else 'top', fontsize=9)
    ax.set_title(title, fontsize=10); ax.set_xlabel('Common Gaussian noise draw')
    ax.set_ylabel('Rank3 minus baseline KL (x1e-6)')
    ax.spines[['top', 'right']].set_visible(False)
axes[0].set_ylim(-6, 1); axes[1].set_ylim(0, 37)
fig.suptitle('One actual round30: transport fidelity improves, final-reference KL need not', fontsize=11)
fig.text(.5, .015, '200 frozen validation samples. Three noise draws are not training seeds. Pre-FU round: final-reference KL is descriptive, no deletion or long-run equivalence claim.', ha='center', fontsize=7)
fig.tight_layout(rect=(0, .055, 1, .94))
for ext in ['png', 'svg']:
    fig.savefig(results/f'heartbeat_0842_task_sensitivity.{ext}', dpi=170)
plt.close(fig)

fu_status = snapshot.get('fu_screen_status', {}).get('status', 'not_yet_recorded')
radio_status = snapshot.get('fu_radio_status', {}).get('status', 'not_yet_started')
report = f'''ova_CK 밤샘 연구 — 08:42 heartbeat 후속 결과
실제 캡처: {now} KST

1. 실제 VLM 출력 검사 완료: 보존 정확도는 같고 전송 민감도 차이는 작음
사전에 정한 실제round30의theta29/SGD업데이트에서 ideal1단계와 원래무선/공개이력16비트rank3를 검사했다. 물리방식마다같은Gaussian noise3회(30/1030/2030), 총7개vector를 고정하고 기존200utility-validation/200future-forget표본을 CPU로평가했다. v18b는08:29:32완료했다. 7개실제updatevector해시와14개cohort예측의확률정규화·label loss·정확도·두기준KL을 별도코드로재검증해모두통과했다.

cohort/방법 | loss평균 | 정확도 | 이상적1단계대비KL | 최종재학습대비KL(학습단계다름)
validation/원래 | 1.0800470830 | 72.5% | 0.000016955634 | 0.0453053338
validation/rank3 | 1.0800012877 | 72.5% | 0.000013552231 | 0.0453239053
future-forget probe/원래 | 1.1381661423 | 72.5% | 0.000015980344 | 0.0410501974
future-forget probe/rank3 | 1.1380913845 | 72.5% | 0.000013021182 | 0.0410754053

이번표본에서는rank3가loss와이상적업데이트대비출력KL이조금낮았고정확도는같았다. 하지만최종재학습기준KL은조금더컸다. 유리한기준만남기지않고모두보존했다. round30은삭제요청전이므로future-forget probe는출력민감도검사이며실제forgetting/요청처리/삭제보장을뜻하지않는다. 서로다른학습단계의최종재학습KL도삭제효과판정으로쓰지않는다. 3회잡음은학습seed가아니고, 이작은pilot만으로장기품질동등성이나통계적우위를주장하지않는다. 새테스트셋선택/GPU추론/작은pilot의MIA추정은없었다.

2. 확인된에너지후보와남은비용한계
같은실제round30의모델링통신에너지는원래726.717542µJ→rank3 687.656142µJ로5.37505%감소했다. 계수RF22.173138µJ와864TDMA symbols의시간까지지불했다. 원래절대RMS0.0055862815, 전체167.936ms, peak0.32W/UAV battery0.1J, head10개와같은AO30/5시작점조건이다. 모든30개저장설계를독립물리단위검증했다. 별도실제PUF round100에서는1.79385%감소했다. 두고정업데이트의결과이며누적전체학습절감률은아니다.
계산/basis생성/메모리/복호/헤더/유한블록오류비용은미측정이다. round30에서는네트워크전체추가비용이39.061399µJ미만이어야순이득이남는다. 모든전송방법에같은공개이력/codec을쓸수있는일반FL개선후보이며NOMA가Mixed보다좋다는증거는아니다. 아직새장시간GPU학습으로승격하지않았다.

3. 실제언러닝요청61의동일조건비교를실행
사전등록한observer가공개57·58·59·60을08:30:28까지11개상태/읽기오류0회로기록했다. 완료된VLM CPU검사후PID3621851에서실제request61 minibatch의learning/coupled-unlearning을theta60에서재현하고54개CPU무선설정을비교하도록시작했다. 현재replay상태:{fu_status}, radio상태:{radio_status}.
검사설정은rank0/1/3 × 상대전체오차0.2/0.6 × Hybrid/Mixed/7개분리시간비율이다. 모든경우head10개를검색하고동일AO30/5시작점을준다. Hybrid/분리전송은L, Mixed는L+U를같은공개basis와동일16비트codec으로분해한다. 계수전송시간/RF를원래프레임에서차감하고원래uncompressed/rank1비교군도유지한다. 전체epsilon은모든mode/rank에대해압축전learning통계에서같이정하므로압축에따라품질한도를풀지않는다. 별도U정밀도제약도소급도입하지않는다.
추가시간이공짜가되거나두번계상되지않는지, head가FU보유자인경우원격payload가없어지는분리시간계상, projection이U의codec/오차한도를바꾸지않는지확인하는2개검증과plan검사를통과했다. CPU/CUDA수치와projected random basis실현은다를수있다. 모든전송구조는같이동결한CPU replay vector로비교하므로그차이를방식우위로쓰지않는다. 아직request61의완료결과나에너지/삭제우위를선언하지않는다.
공정성보강v19b도별도로예약했다. 계수시간을차감한뒤고정비율0.125를쓰면아날로그symbol수가mapping차원d보다조금작아질수있다. 실패는보존하되압축rank1/rank3각각에대해실제최소L_A=d(나머지는digital)인물리경계도두오차조건/10개head에서검사한다. 원래rank0의0.125는이미그경계와같다. CPU PID3622815가현재v19를기다린뒤추가4개경계설정을실행한다. 이결과까지합쳐최저에너지를비교하며분리전송을불리하게고정하지않는다. 기존v19/현재GPU/큐소스는변경하지않았다.

4. 기존GPU큐와원래완료결과
기존burst Hybrid GPU PID3602333 한개가실행중이다. 캡처기준{len(burst_rows)}/100라운드, 도착한요청{len(snapshot['burst_requests'])}개전달완료다. 나머지Mixed/시간배분최적화분리전송은같은큐에서순차실행한다. burst는요청라운드RMS비율0.6/그외0.2인별도문제조건이며모든비교군에같이적용한다. 원래RMS0.2결과와섞지않는다. 현재GPU/기존큐가검증하는모든source해시는변경이없다.
원래0.2의이번seed 완료무선실험에서는Mixed의에너지상한0.087379794J, 검증loss0.96706282, forgetKL0.02574322가이세지표의최저다. 6비트Hybrid0.189836125J/8비트Hybrid0.213161276J다. MIA까지모두지배하거나여러seed에서우월함은뜻하지않는다. ideal에너지0과실패/미완료방법은에너지우위순위에서제외했다. 기존retained loss/정확도,forget KL/예측차이,MIA AUC gap,전체완료율/요청전달/에너지표는동봉canonical report에보존했다. 전달이나작은MSE는삭제보장이아니다.
manuscript/원본결과는보존했다. 오전9시이후첫heartbeat에최선후보·한계·비교표를한번보고한다. 정오에는최종상태와진행중계산을보고한뒤heartbeat를일시중지하며새장시간실험은추가하지않는다.

증거: heartbeat_0842_results.zip의v18b전체7개vector/예측/독립재검증/코드, v19사전등록/관찰/동결실험계획/현재출력, 현GPU상태/소스해시.
'''
(results/'heartbeat_0842_report_ko.txt').write_text(report)
paths = [results/'heartbeat_0842_snapshot.json', results/'heartbeat_0842_report_ko.txt',
         results/'heartbeat_0842_task_sensitivity.png', results/'heartbeat_0842_task_sensitivity.svg',
         tools/'campaign_plan.txt', tools/'campaign_state.json', tools/'capture_0842.py', tools/'audit_history_quality_v18b.py',
         root/'results/energy_quality_v1/energy_quality_report.json', results/'burst_joint_v3/status.json',
         results/'burst_joint_v3/plan.json', results/'burst_joint_v3/config.json', burst_path, results/'queue_recovery_v1/status.json']
for name in ['history_quality_v18b', 'public_history_fu_v19', 'public_history_fu_v19b']:
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
