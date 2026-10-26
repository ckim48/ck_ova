"""Versioned heartbeat capture; completed Hybrid audit and queued fair FU pilot."""
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
canonical = root/'results/energy_quality_v1'; burst = results/'burst_joint_v3'


def read(path):
    for attempt in range(10):
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            if attempt == 9:
                raise
            time.sleep(.02)


def write(path, value):
    tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(value, indent=2)+'\n'); tmp.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


now = datetime.now().isoformat(timespec='seconds')
state = read(tools/'campaign_state.json'); assert state['morning_report_sent']
expected = read(burst/'plan.json')['source_sha256']
assert all(sha(root/n) == value for n, value in expected.items())
comparison = read(results/'burst_quality_comparison_v21/results.json')
old, changed = [comparison[x] for x in ['original', 'changed']]
assert changed['status'] == 'completed' and changed['round'] == 100 and changed['deletion_rounds_delivered'] == 5
progress = {}
for mode in ['hybrid', 'mixed_aircomp', 'phase_grid']:
    path = burst/mode/'rounds.jsonl'
    if path.exists():
        rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
        progress[mode] = dict(rounds=len(rows), requests_delivered=sum(r['newly_deleted'] > 0 for r in rows), last=rows[-1] if rows else None)
    else:
        progress[mode] = dict(rounds=0, requests_delivered=0, last=None)
fu = results/'public_history_fu_v19'; boundary = results/'public_history_fu_v19b'
fu_rows = read(fu/'radio/results.json'); boundary_rows = read(boundary/'results.json') if (boundary/'results.json').exists() else []
gpu = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader'], text=True)
processes = subprocess.check_output(['ps', '-u', 'wnlab', '-o', 'pid,ppid,stat,etime,args'], text=True).splitlines()
snapshot = dict(captured_at_kst=now, original_report_sha256=sha(canonical/'energy_quality_report.json'),
    changed_hybrid=changed, changed_minus_original_cache_analysis=comparison,
    source_freeze_verified=expected, burst_status=read(burst/'status.json'), burst_progress=progress,
    gpu_processes=gpu, project_processes=[x for x in processes if str(root)+'/' in x],
    fu_case_count=len(fu_rows), fu_status=read(fu/'radio/status.json'), boundary_case_count=len(boundary_rows),
    boundary_status=read(boundary/('status.json' if (boundary/'status.json').exists() else 'waiting_status.json')),
    physical_certificate=read(results/'transport_certificate_v20/heartbeat0942.json'),
    queued_cpu_quality_status=read(results/'fu_quality_v22/status.json'),
    queued_cpu_quality_plan=read(results/'fu_quality_v22/plan.json'))
write(results/'heartbeat_0942_snapshot.json', snapshot)

lines = [
    'ova_CK 밤샘 연구 — 09:42 heartbeat 후속 결과', '캡처: '+now+' KST', '',
    '현재 판단: 새 조건의 Hybrid는100/100round와5/5요청라운드 전달 및 최종 평가를 완료했다. 기존Hybrid8보다 통신에너지상한은34.16%낮지만 검증loss와 최종재학습기준forgetKL은 조금 높다. 같은품질의우위나NOMA우위로해석하지않는다. 같은새조건Mixed가학습중이며, 시간배분최적화분리전송은그다음이다.', '',
    '1. 완료된 Hybrid 비교(조건변경을포함한에너지-품질절충)',
    '지표 | 원래Hybrid8 | 새burst Hybrid8',
    f"통신에너지상한J | {old['energy_bound_j']:.12f} | {changed['energy_bound_j']:.12f}",
    f"통신에너지기댓값J | {old['expected_energy_j']:.12f} | {changed['expected_energy_j']:.12f}",
    f"utility validation loss/accuracy | {old['validation_loss']:.9f}/{100*old['validation_accuracy']:.1f}% | {changed['validation_loss']:.9f}/{100*changed['validation_accuracy']:.1f}%",
    f"retained loss/accuracy | {old['retained_loss']:.9f}/{100*old['retained_accuracy']:.1f}% | {changed['retained_loss']:.9f}/{100*changed['retained_accuracy']:.1f}%",
    f"forget KL(ref||method)/prediction disagreement | {old['forget_kl_reference_to_candidate']:.9f}/{100*old['forget_prediction_disagreement']:.1f}% | {changed['forget_kl_reference_to_candidate']:.9f}/{100*changed['forget_prediction_disagreement']:.1f}%",
    f"validation loss-MIA AUC/absolute gap from retrain AUC | {old['forget_loss_attack_auc_validation']:.6f}/{old['forget_attack_auc_gap_to_reference_validation']:.6f} | {changed['forget_loss_attack_auc_validation']:.6f}/{changed['forget_attack_auc_gap_to_reference_validation']:.6f}",
    '둘다100/100round,5/5요청전달,누적uplink16.7936s. 에너지는통신모델상한/기댓값이며실제UAV전체배터리소모측정이아니다. 요청전달은삭제보장이아니다.',
    '원래실험은요청오차비율0.2/기존solver,새실험은요청라운드만0.6/공동power solver다. 모델궤적과head도달라절감률을오차완화만또는solver만의인과효과로분해하지않는다. 새조건을Hybrid/Mixed/분리전송모두동일하게적용하며원래조건의순위와합치지않는다.',
    '동일1000개검증/retained/forget sample의cache와cohort/evaluator/checkpoint provenance를검사했고추가추론이나테스트기반설정선택은하지않았다. 5000회paired class-stratified bootstrap은고정모델의image-sample조건부분석일뿐training seed/client/radio불확실성이나확증적검정이아니다.',
    '새조건−원래조건: validation loss+0.00125812(조건부95%[0.00095467,0.00157983]), retained loss+0.00090813([0.00057840,0.00124965]), forget KL+0.000280845([0.000226409,0.000336321]). 정확도가같다는이유로품질이동등하다고쓰지않는다.', '',
    '2. 사용자가물은OMA상태',
    '기존50:50시간분리전송은90round완료후요청91에서실행가능설계를찾지못했다. 모든OMA가불가능하다는뜻은아니다. 실패/미완료누적에너지는완료방법의에너지우위순위에서제외한다.',
    '원래요청91고정업데이트·오차비율0.2·8비트·총시간/전력/배터리동일조건에서는7개시간비율을최적화한분리전송32.822688mJ,Hybrid55.018777mJ,Mixed5.059153mJ였다. 고정업데이트진단이며100round누적실험결과가아니다. 이수치는분리전송을강한비교군으로유지해야한다는근거다.',
    '여기서OMA라는대화상표현은learning과unlearning의시간분리를뜻한다. digital unlearning 사용자들끼리는NOMA를쓴다. 전사용자를완전TDMA로분리한실험과동일하지않다.',
    f"현재새조건전체학습: Mixed {progress['mixed_aircomp']['rounds']}/100round; 최적시간분리전송 {progress['phase_grid']['rounds']}/100round. Hybrid만먼저완료한부분상태로새조건승자를결정하지않는다.", '',
    '3. 실제요청61에서압축개선검증을계속',
    f"공개theta60에서동일CPU learning/coupled unlearning벡터를고정했다. 현재v19 {len(fu_rows)}/54설정,추가최소analog L_A=d경계v19b {len(boundary_rows)}/4설정.54+4개를모두합친후비교한다. 계수전달로분리전송의최소mapping경계가이동하므로이경계를검사하지않고우위를선언하지않는다.",
    '계수16비트RF/시간을지불한rank0/1/3를Hybrid/Mixed/분리전송에같이적용한다. 오차기준은원래uncompressed learning벡터로고정하며요청unlearning양자화의신호와가중치를몰래바꾸지않는다. 추가basis/projection/메모리/코딩/복호/헤더에너지는미측정이다.',
    '09:52독립물리검증: 관찰된48설정중저장된실행가능최저후보39개가원래W/J단위의power/battery/SICrate/시간/오차/codec검사를통과했다. 다른미저장head,전역최적성,실제장비,응용삭제효과를검증했다는뜻은아니다.',
    '별도fu_quality_v22를사전등록하고CPU대기큐에추가했다. 전체58개와독립검증완료후에만원래오차0.2의rank0/rank3×세전송방식에서에너지로만설계를선택한다.19개one-step벡터(ideal1+6설계×공통잡음3),validation200/retained200/실제요청61삭제대상200을평가한다. 복원신호부호/head자체업데이트/패딩/SIC가정관련의미있는검사3개통과. GPU나기존학습소스는변경하지않았다.',
    'pilot은ideal-step출력KL과최종retained-retrain출력KL/예측차이를함께저장한다. 최종100round재학습모델은round61pilot과단계가달라그KL은기술적참고치이고삭제보장/품질동등성증거가아니다. 작은pilot으로MIA를새로추정하지않으며완료100round MIA는위표에유지한다.3잡음은3학습seed가아니다. 전체GPU학습으로자동승격하지않는다.', '',
    '4. 현재큐와다음판단',
    '기존GPU학습/평가는한번에하나다. 현재Mixed,그다음시간최적화분리전송,그다음정오전시작가능한기존복구큐순서다. 모든동결학습소스해시를다시검사했다. 원래manuscript/결과/실패기록을보존했다.',
    '원래조건에서완료된Mixed가현재관찰된에너지·검증loss·forgetKL최저라는오전보고판단은그대로다. MIA gap은Hybrid가조금낮고학습seed1개이므로모든지표/통계적승자를주장하지않는다. 아직Hybrid의NOMA에너지우위를입증하지못했다.',
    '정오에는진행중인계산과최종밤샘상태를보고하고heartbeat를일시중지한다. 정오이후새장시간실험은추가하지않으며이미시작한계산은안전하게완료하게둔다.', '',
    '근거는heartbeat_0942_results.zip에보존했다.이전오전전체비교표는morning_report_0912_ko.txt와heartbeat_0912_results.zip에있다.'
]
report = results/'heartbeat_0942_report_ko.txt'; report.write_text('\n'.join(lines)+'\n')
state.update(last_heartbeat_research_kst=now, last_completed_evidence_snapshot='results/overnight_energy_v2/heartbeat_0942_snapshot.json',
    current_gpu_method='burst_joint_v3/mixed_aircomp/training',
    burst_hybrid_completed=True, burst_hybrid_final_quality_evaluated=True, burst_quality_comparison_v21_completed=True,
    notes='ChangedHybrid100/5 audited Eupper0.140355664J vsoriginal0.213161276J(-34.16%),butvalidationloss/forgetKLslightlyworse; epsilon0.6andjointsolverchanged, noequalqualitycausalclaim. Mixedcurrentlytraining, optimizedphasewaits. OriginalOMAequal50failed91doesnotimplyallOMAinfeasible; legacyoptimizedphasebeatsH. v19+Bfull58fairpoolrequired; v20partial39savedcandidatespassed. v22CPU19vectors/200eachcohortpreregisteredwaitsB,3tests,energyonlyselectionandfullphysicalgatebeforerunning. NoGPU/sourcechanges;morningflagtrue;noonreportpausepolicyunchanged.')
if gpu.strip():
    state['current_gpu_pid'] = int(gpu.splitlines()[0].split(',')[0])
write(tools/'campaign_state.json', state)
with (tools/'campaign_plan.txt').open('a') as stream:
    stream.write('\nPOST09:42 '+now+' KST: Hybrid changedprotocol complete100/5 with finalaudit; Eupper0.140355664J (-34.155% vsoriginalH8),validationloss0.968945219 andforgetKL0.026242090 slightlyworse,cohortpairedcachev21saved. Differentrequestepsilon/solver, noequalquality/NOMAclaim. Mixedrunningandoptimizedphasewaits. v19partialplusv19bminimumphaseboundarymerge58beforearchitecturecomparison. v20partial39bestcandidatecertspassed09:52. v22CPUwaitpid3622815 launchedpid'+str(state['fu_quality_v22_cpu_pid'])+';19one-stepvectors/3commonnoises,validation200retained200currentDF61forget200,3recoverytests; fullphysicalgateandenergy-onlyselectionbeforeinference. NoextraGPU/nofrozen-sourcechanges, morningreportalreadysent; noonreportpause/no-new-longjobsafter12unchanged.\n')

paths = [report, results/'heartbeat_0942_snapshot.json', tools/'campaign_state.json', tools/'campaign_plan.txt',
         tools/'capture_0942.py', canonical/'energy_quality_report.json', canonical/'cohorts.json',
         burst/'plan.json', burst/'config.json', burst/'status.json', results/'queue_recovery_v1/status.json']
for mode in ['hybrid', 'mixed_aircomp', 'phase_grid']:
    for name in ['rounds.jsonl', 'manifest.json', 'variant_provenance.json']:
        p = burst/mode/name
        if p.exists(): paths.append(p)
paths.append(burst/'hybrid/latest.pt')
paths.extend(p for p in (burst/'quality').iterdir() if p.is_file() and p.suffix in ['.json', '.npz'])
for name in ['public_history_fu_v19', 'public_history_fu_v19b', 'transport_certificate_v20', 'fu_quality_v22', 'burst_quality_comparison_v21']:
    for parent in [results, tools]:
        paths.extend(p for p in (parent/name).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.tmp')
paths.append(tools/'burst_quality_comparison_v21.py')
contents = {}; omitted = []
for path in dict.fromkeys(paths):
    for attempt in range(5):
        try:
            data = path.read_bytes()
            if path.suffix == '.json': json.loads(data)
            if path.suffix in ['.npz', '.pt']:
                with zipfile.ZipFile(io.BytesIO(data)) as archive: assert archive.testzip() is None
            contents[str(path.relative_to(root))] = data; break
        except (OSError, json.JSONDecodeError, zipfile.BadZipFile, AssertionError):
            time.sleep(.02)
    else: omitted.append(str(path.relative_to(root)))
manifest = dict(captured_at_kst=now, ongoing_files_are_partial=True, in_progress_files_omitted=omitted,
    files={n: dict(sha256=hashlib.sha256(v).hexdigest(), bytes=len(v)) for n, v in contents.items()})
buffer = io.BytesIO()
with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
    for name, data in contents.items(): archive.writestr(name, data)
    archive.writestr('capture_manifest.json', json.dumps(manifest, indent=2)+'\n')
print(base64.b64encode(buffer.getvalue()).decode())
