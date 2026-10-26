"""First post09:00 research report, with eligible results and explicit limits."""
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
suite = root/'results/comparisons_seed0_solver_v2'; canonical = root/'results/energy_quality_v1'


def read(path):
    for attempt in range(10):
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            if attempt == 9:
                raise
            time.sleep(.02)


def write(path, obj):
    tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(obj, indent=2)+'\n'); tmp.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


now = datetime.now().isoformat(timespec='seconds')
assert datetime.now().hour >= 9
state = read(tools/'campaign_state.json'); assert not state['morning_report_sent']
expected = read(results/'burst_joint_v3/plan.json')['source_sha256']
assert all(sha(root/n) == value for n, value in expected.items())
quality = read(canonical/'energy_quality_report.json'); methods = {x['method']: x for x in quality['methods']}
names = ['coupled_mixed_aircomp', 'coupled_hybrid_q6', 'puf_regular_adapted_hybrid', 'coupled_hybrid', 'coupled_fixed_head']
short = ['Mixed AirComp', 'Hybrid6', 'PUF adapted8', 'Hybrid8', 'Fixed-head Hybrid8']
full_table = []
for n in names:
    x = methods[n]; assert x['status'] == 'completed' and x['round'] == 100 and x['quality_status'] == 'evaluated'
    rows = [json.loads(line) for line in (suite/n/'rounds.jsonl').read_text().splitlines() if line.strip()]
    requests = [r for r in rows if r['newly_deleted'] > 0]
    assert len(rows) == 100 and len(requests) == 5
    full_table.append(dict(method=n, energy_upper_j=x['energy_bound_j'], validation_loss=x['validation_loss'],
                          validation_accuracy=x['validation_accuracy'], retained_loss=x['retained_loss'], retained_accuracy=x['retained_accuracy'],
                          forget_kl=x['forget_kl_reference_to_candidate'], forget_prediction_disagreement=x['forget_prediction_disagreement'],
                          mia_validation_auc=x['forget_loss_attack_auc_validation'], mia_auc_gap=x['forget_attack_auc_gap_to_reference_validation'],
                          completed_rounds=100, request_rounds_delivered=5, samples_per_group=x['validation_samples']))
burst_status = read(results/'burst_joint_v3/status.json'); burst_progress = {}
for mode in ['hybrid', 'mixed_aircomp', 'phase_grid']:
    p = results/'burst_joint_v3'/mode/'rounds.jsonl'
    if p.exists():
        rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
        burst_progress[mode] = dict(rounds=len(rows), requests_delivered=sum(x['newly_deleted'] > 0 for x in rows),
                                    last=rows[-1] if rows else None)
fu = results/'public_history_fu_v19'; boundary = results/'public_history_fu_v19b'
fu_rows = read(fu/'radio/results.json') if (fu/'radio/results.json').exists() else []
gpu = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader'], text=True)
processes = subprocess.check_output(['ps', '-u', 'wnlab', '-o', 'pid,ppid,stat,etime,args'], text=True).splitlines()
snapshot = dict(captured_at_kst=now, eligible_original_table=full_table, canonical_report_sha256=sha(canonical/'energy_quality_report.json'),
                cohorts_sha256=sha(canonical/'cohorts.json'), frozen_burst_source_sha256=expected,
                original_observed_frontier=quality['observed_frontier'], original_selection_status=quality['selection_status'],
                ideal_controls={n: methods[n] for n in ['coupled_ideal', 'retain_only_ideal', 'retrain_reference_ideal']},
                original_incomplete={n: methods[n] for n in ['coupled_orthogonal_equal_time', 'uncoupled_hybrid']},
                burst_status=burst_status, burst_progress=burst_progress, gpu_processes=gpu,
                project_processes=[x for x in processes if str(root)+'/' in x],
                fu_status=read(fu/'radio/status.json'), fu_completed_cases=len(fu_rows),
                boundary_status=read(boundary/'waiting_status.json'),
                partial_transport_certificate=read(results/'transport_certificate_v20/heartbeat0912.json'),
                retained_control_audit=read(results/'retain_control_audit_v16/results.json'),
                codec_quality_audit=read(results/'history_quality_v18b/independent_metrics.json'))
if (boundary/'status.json').exists():
    snapshot['boundary_running_or_completed_status'] = read(boundary/'status.json')
if (results/'burst_joint_v3/quality/energy_quality_report.json').exists():
    snapshot['revised_quality_report'] = read(results/'burst_joint_v3/quality/energy_quality_report.json')
write(results/'heartbeat_0912_snapshot.json', snapshot)

lines = [
    'ova_CK 밤샘 연구 — 오전 보고(09시 이후 첫 heartbeat)', '캡처: '+now+' KST', '',
    '현재 판단: 원래 조건의 완료된 이번 seed 결과에서는 Mixed AirComp가 통신 에너지·검증 loss·forget 출력 KL 세 지표의 최저다. 제안 Hybrid가 현재 최고라고 말할 근거는 없다. 6비트 Hybrid는 8비트 대비 통신 에너지를 약10.94% 줄였지만 Mixed보다 높다. MIA AUC gap은 Hybrid가 조금 더 작아 Mixed의 모든 지표 우위나 통계적 우월성을 주장하지 않는다.', '',
    '1. 원래 조건의 완료된 무선 실험 비교',
    '100rounds,10UAV,CIFAR100/CLIP ViT-B16 prompt adaptation,seed0,원래 RMS비율0.2. 아래5개는100/100round와5/5요청라운드 전달완료. 전부 같은1000개/cohort 최종평가 및 최종retained-retrain 기준이다.',
    '방법 | 통신에너지상한J | utility검증loss/정확도 | retained loss/정확도 | forget KL(ref||method)/예측차이 | validation loss-MIA AUC/재학습AUC와절댓값차이']
for label, n in zip(short, names):
    x = methods[n]
    lines.append(f"{label} | {x['energy_bound_j']:.9f} | {x['validation_loss']:.8f}/{100*x['validation_accuracy']:.1f}% | {x['retained_loss']:.8f}/{100*x['retained_accuracy']:.1f}% | {x['forget_kl_reference_to_candidate']:.8f}/{100*x['forget_prediction_disagreement']:.1f}% | {x['forget_loss_attack_auc_validation']:.6f}/{x['forget_attack_auc_gap_to_reference_validation']:.6f}")
lines += [
    '',
    'PUF adapted는 요청UAV의learning을생략하고FU가중치도달라순수고정업데이트무선비교가아니다. Fixed head는head선택효과를보는baseline이다. 사전설정된task품질통과임계값이없고학습seed도1개라공식최종승자/동등품질인증으로해석하지않는다.',
    '에너지는모델링된통신에너지상한이며실제배터리/비행/로컬학습/최적화계산을포함한UAV전체에너지가아니다. 정확한삭제나privacy보장으로해석하지않는다.', '',
    '2. 미완료·실패·이상적링크는에너지순위에서제외',
    '기존50:50시간분리전송은91round에서실패하여90round까지만완료했다(누적0.133814316J). uncoupled Hybrid는57round에서중단됐다(0.061257523J,원래supervisor/child가사라진원인은미확정). 둘다완료방법보다에너지가적다고순위에넣지않는다.',
    '50:50실패는시간배분을최적화한모든분리전송의불가능성을뜻하지않는다. CPU에서같은원래업데이트·총시간·품질조건의7개시간비율을검색하면분리전송실행가능설계가있고일부조건에서는Hybrid보다낮은에너지였다. 별도전체학습큐에서도강한분리전송baseline을유지한다.',
    'coupled ideal,retain-only ideal,최종retained-retrain ideal은알고리즘/품질기준이다. 에너지0을실제무선경쟁방법으로취급하지않는다.', '',
    '3. 언러닝자체의한계: 추가효과가아직뚜렷하지않음',
    '이상적링크에서도coupled FU의forgetKL0.01824249,retain-only0.01835310으로차이가작았다. 동일sample짝을보존한class-stratified bootstrap에서coupled−retain-only forgetKL차이−0.000110614,조건부95%구간[−0.000339100,+0.000120842]로0을포함했다. validation loss는coupled0.94428719,retain-only0.93993418로retain-only가낮았다. MIA AUC gap차이도−0.000006으로뚜렷한추가이득이없었다.',
    '이구간은고정모델의sample조건부불확실성으로여러training seed나client의존성을포함하지않는다. 삭제대상은대부분과거SGD노출1회라기억이약한조건이다. MIA가0.5부근이라는사실이나작은wireless MSE만으로삭제보장을주장하지않는다.',
    '현재목표가L/U업데이트를단순히더한합계인것이Mixed baseline이강한주요이유다. L/U를분리복원하는것자체가추가VLM손실/삭제이득을만들지는않는다. 응용이득이확인되지않은별도U정밀도제약을소급도입해Mixed/dual-analog결과를제외하지않았다.', '',
    '4. 확인한개선과검증상태',
    '6비트quantization: Hybrid100round최종평가완료,8비트대비10.94%에너지상한감소. 최종utility/forgetKL은가깝지만같다고인증하지않는다. head선택차이도포함되므로순수bit효과만분리한절감률은아니다.',
    'solver: 고정digitalpower를풀어analog/digital을공동최적화하고convex초기화/4개기존시작점/AO30을모든비교군에같이검사했다. 원래legacy고정업데이트에서일부개선은있었으나Hybrid의Mixed열세를해소하지못했다. 독립물리단위검증과실패기록을보존했다.',
    '압축: 과거공개모델이력으로rank3 residual+16비트계수를전달하는일반FL후보에서계수전송비/시간을지불하고두고정learning-only업데이트의통신에너지가1.79%(late100)/5.38%(early30)감소했다. 추가basis/계산/메모리/복호/헤더비용은미측정이다. early30에서는네트워크추가비용39.061µJ미만이어야순이득이남는다. 모든비교군이같은codec을쓸수있으므로NOMA우위가아니다.',
    'CPU출력pilot: early30의200validation/200future-forget,3개공통잡음에서정확도는모두72.5%,rank3의ideal-step출력KL은조금낮았지만최종재학습기준KL은조금더컸다. round30은요청전이라삭제효과검사가아니며전체학습의품질동등성/개선을입증하지않는다.',
    '실제요청61: 공개theta60에서동일CPU learning/coupledU를재현해Hybrid/Mixed/시간최적화분리전송에동일codec을적용한54설정검사를진행중이다. 계수시간때문에분리전송의mapping최소경계가움직이므로추가4개최소L_A=d경계도별도v19b로대기시켰다. 이결과까지합치기전부분결과로우위를선언하지않는다.',
    f"현재요청61 CPU상태:{snapshot['fu_status']['status']},완료{len(fu_rows)}/54설정.09:14독립검증시점의21개관찰설정중저장된17개실행가능최저후보가원래W/J단위의power/battery/SICpayload/시간/오차/codec검증을통과했다. 나머지는계속추가검증한다. unsaved다른head나전역최적성/실제장비/삭제효과를검증했다는뜻은아니다.", '',
    '5. 진행중인전체학습(다른조건이므로원래표와별도로판단)',
    'burst_joint_v3는요청round만RMS비율0.6,나머지는0.2로변경한별도문제다. Hybrid/Mixed/시간최적화분리전송모두같은조건을적용한다. 현재큐상태:'+burst_status['status']+', mode:'+str(burst_status.get('mode'))+'.']
for mode, progress in burst_progress.items():
    lines.append(f"{mode}: {progress['rounds']}/100round,요청전달{progress['requests_delivered']}/5.현재누적통신상한{progress['last']['cumulative_energy_bound_j']:.9f}J. 완전한최종평가전에는완료방법과의에너지우위순위에넣지않는다.")
lines += [
    'GPU학습/평가는한번에하나로유지하고현재기존큐소스해시를전부확인했다. Mixed/분리전송은순차대기한다. 새codec의고정업데이트검사와응용검증은CPU에서수행하며아직전체GPU학습후보로승격하지않았다. 원본manuscript와결과는그대로보존했다.',
    '정오까지필요한진단/검증을이어가고, 정오에는결과와실행중인계산을보고한뒤heartbeat를일시중지한다. 정오전에시작된계산은안전하게완료하게두며정오이후새장시간실험은추가하지않는다.', '',
    '증거: heartbeat_0912_results.zip의canonical최종품질표/provenance,이상적대조군bootstrap,현재burst진행과동결해시,v19/v19b상태/코드/동결update/부분물리certificate. 이전negative controls와실패결과는앞선heartbeat묶음에도보존돼있다.']
report_path = results/'morning_report_0912_ko.txt'; report_path.write_text('\n'.join(lines)+'\n')
state.update(morning_report_sent=True, morning_report_at_kst=now, morning_report_path=str(report_path.relative_to(root)),
             last_heartbeat_research_kst=now, last_completed_evidence_snapshot='results/overnight_energy_v2/heartbeat_0912_snapshot.json',
             notes='Firstpost09report saved: originalcompletedseed0 MixedlowestmodeledcommE/valloss/forgetKL; noformalallmetric/statisticalwinner, MIAHslightlylower.5physicalmethods100/100,5/5; ideal0andfailed/incompleteexcluded. FUbenefitweakvsretain-only/weakmemory. Changedburstprotocolseparatependingfinalaudit. v19partial54cases+v19b4phaseboundarymustmergebeforecomparison; independentv20certificate17savedcandidatespassedat09:14. NewcodecRFgain1.79/5.38% onlyfixedlearningsteps, CPUcost/fulltrajectory/FUbenefitunproved. NoexistingGPU/sourcechanges, morningreportflagtrue; noonreport/pause/no-new-long-jobpolicyunchanged.')
if gpu.strip():
    state['current_gpu_pid'] = int(gpu.splitlines()[0].split(',')[0])
state['current_gpu_method'] = 'burst_joint_v3/'+str(burst_status.get('mode'))+'/'+burst_status['status']
write(tools/'campaign_state.json', state)
with (tools/'campaign_plan.txt').open('a') as f:
    f.write('\nFIRST POST09 MORNING REPORT '+now+' KST savedmorning_report_0912_ko.txt; flagtrue. Originalcompleted5physicalmethods100/100 and5/5. MixedE0.087379794J/valloss0.96706282/forgetKL0.02574322 minimum3metrics; Hybrid6E0.189836125J vsHybrid8 0.213161276J, butMIAgapsHslightlysmaller, noallmetric/statistical/formalqualitywinner. Ideal0/fail/incompleteexcluded. WeakFUbeyondretain-only andlowmemorization reported; allrawnegativeresultsretained. Burstchangedrequesttolerance0.6 reportedseparatelypendingfinalcompletion/audit. v19partialplusv19bminimumphaseboundary mustmergebeforeanyrank; v20independent17savedcandidatecertificatespassed. No extraGPU/currentfrozen-sourceedits, CPUresearchcontinues, noonreport/pause/no-new-longjobs policyunchanged.\n')

paths = [report_path, results/'heartbeat_0912_snapshot.json', tools/'campaign_state.json', tools/'campaign_plan.txt',
         tools/'capture_0912_morning.py', tools/'transport_certificate_v20.py', canonical/'energy_quality_report.json',
         canonical/'cohorts.json', suite/'suite_metadata.json', results/'burst_joint_v3/status.json',
         results/'burst_joint_v3/plan.json', results/'burst_joint_v3/config.json', results/'queue_recovery_v1/status.json',
         results/'retain_control_audit_v16/results.json', results/'retain_control_audit_v16/plan.json']
for n in names+['coupled_ideal', 'retain_only_ideal', 'retrain_reference_ideal']:
    paths.extend([suite/n/'manifest.json', suite/n/'rounds.jsonl', canonical/(n+'.cache.json')])
for mode in burst_progress:
    paths.extend([results/'burst_joint_v3'/mode/'rounds.jsonl', results/'burst_joint_v3'/mode/'manifest.json'])
for name in ['public_history_fu_v19', 'public_history_fu_v19b', 'transport_certificate_v20']:
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
