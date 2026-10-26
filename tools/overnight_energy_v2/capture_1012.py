"""Capture completed fair FU61 grid and newly queued justified precision work."""
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
        try: return json.loads(path.read_text())
        except json.JSONDecodeError:
            if attempt == 9: raise
            time.sleep(.02)


def write(path, value):
    tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(value, indent=2)+'\n'); tmp.replace(path)


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


now = datetime.now().isoformat(timespec='seconds'); state = read(tools/'campaign_state.json'); assert state['morning_report_sent']
burst = results/'burst_joint_v3'; expected = read(burst/'plan.json')['source_sha256']
assert all(sha(root/n) == value for n, value in expected.items())
summary = read(results/'fu_radio_summary_v23/results.json'); cert = read(results/'transport_certificate_v20/fu_quality_v22.json')
assert summary['status'] == 'completed_all58_fair_radio_cases' and cert['checked_saved_candidates'] == 48
progress = {}
for mode in ['hybrid', 'mixed_aircomp', 'phase_grid']:
    path = burst/mode/'rounds.jsonl'; rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []
    progress[mode] = dict(rounds=len(rows), requests_delivered=sum(r['newly_deleted'] > 0 for r in rows), last=rows[-1] if rows else None)
gpu = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader'], text=True)
processes = subprocess.check_output(['ps', '-u', 'wnlab', '-o', 'pid,ppid,stat,etime,args'], text=True).splitlines()
pilot = results/'fu_quality_v22'; precision = results/'fu_precision_v24'; quality_audit = results/'fu_quality_audit_v25'
snapshot = dict(captured_at_kst=now, frozen_source_sha256=expected, burst_status=read(burst/'status.json'), burst_progress=progress,
    gpu_processes=gpu, project_processes=[x for x in processes if str(root)+'/' in x], completed_radio_summary=summary,
    physical_certificate_status={k: v for k, v in cert.items() if k != 'certificates'},
    task_pilot_status=read(pilot/'status.json'), completed_task_vectors=len(read(pilot/'results.json')),
    precision_status=read(precision/'status.json'), precision_checker_startup=read(precision/'checker_startup_validation.json'),
    independent_task_audit_status=read(quality_audit/'status.json'))
write(results/'heartbeat_1012_snapshot.json', snapshot)
lines = ['ova_CK 밤샘 연구 —10:12 heartbeat 후속결과', '캡처:'+now+' KST', '',
    '새로완료된작업: 실제요청61의공개theta60에서고정한learning/coupled unlearning에대한54설정+4최소시간경계,총58설정을모두완료했다. 저장된실행가능최저설계48개는독립물리계산검증을통과했다. 부분grid를가지고결론내리지않았다.', '',
    '1. 압축전같은고정업데이트의전송비교',
    '요청61,10UAV,8bit삭제신호,동일.167936s/.32W/.1J,오차기준은원래learning norm으로고정. 각설정all10heads/AO30/convex+4시작점. 분리전송은7개시간비율을비교했다. 아래는한라운드통신에너지상한이며전체100round누적실험결과가아니다.',
    '오차비율 | Hybrid mJ | 최적시간분리전송 mJ | Mixed mJ']
for relative in [.2, .6]:
    selected = {x['mode']: x for x in summary['selected'] if x['relative_tolerance'] == relative and x['rank'] == 0}
    h, p, m = [selected[x]['energy_bound_j'] for x in ['hybrid', 'phase_grid', 'mixed_aircomp']]
    lines.append(f'{relative:g} | {h*1000:.6f} | {p*1000:.6f} | {m*1000:.6f}')
lines += [
    '원래오차비율0.2에서는분리전송이Hybrid보다낮다. 별도완화조건0.6에서는Hybrid가분리전송보다약3.28%낮다. 이조건부여지는있지만Mixed는두조건모두가장낮으므로제안법의전체에너지우위를입증한것은아니다.',
    '대화상OMA는learning/unlearning의시간분리이며digital사용자들끼리는NOMA를사용한다. 전사용자를TDMA로만분리한baseline과는다르다. 충분오차bound의실행가능설계를찾았다는뜻이지원래문제의전역최적성이나응용품질/삭제보장을확인한것은아니다.', '',
    '2. 공개이력rank3압축은모든방식에같이적용한후평가',
    '오차0.2:Hybrid통신상한1.024%감소,Mixed5.094%감소,분리전송0.292%증가. 오차0.6에서는Hybrid/Mixed/분리전송모두압축후통신에너지가증가했다. 일부learning-only샘플의압축이득이FU전체에도항상성립하지않는다는음성결과를보존한다.',
    '압축계수16비트전달의RF에너지와시간은차감했으나basis/projection/메모리/코딩/복호/헤더추가에너지는미측정이다. 따라서양의통신절감률도총UAV에너지이득으로쓰지않는다. 정확한수치와추가비용손익분기점은fu_radio_summary_v23/results.json에저장했다.',
    '압축으로분리전송의최소analog mapping경계가이동한다. 추가4개L_A=d경계를포함한결과압축분리전송의가장좋은후보는이경계였다. 경계를빠뜨려비교군을약화하지않았다.', '',
    '3. 출력품질평가는계속진행중',
    f"fu_quality_v22: {snapshot['completed_task_vectors']}/19one-step벡터평가완료, 상태:{snapshot['task_pilot_status']['status']}. 미리등록한원래오차0.2/rank0·3/세전송방식/공통잡음3개/validation200·retained200·현재요청forget200을그대로사용한다. 설계는완료된58개중에너지로만선택했고추론전에물리검증을완료했다.",
    '별도v25 cache감사를대기시켰다.19개벡터를물리신호식으로독립재구성하고57개cohort출력의loss/accuracy/ideal-step KL/최종retained-retrain KL/예측차이와6개에너지최저선택을검사한다. 추가모델추론이나GPU를사용하지않는다.',
    '최종100round재학습은round61과학습단계가달라그기준KL은기술적참고치다. 세잡음은세학습seed가아니고pilot으로품질동등성/삭제보장을주장하지않는다. 작은pilot의MIA를새로추정하지않으며완료100round의MIA AUC gap은기존오전보고에유지한다.', '',
    '4. 진단결과에근거한새실험: 삭제신호비트수',
    '같은요청61벡터에서정량화오차bound만분리하면6bit는모든head에서원래오차예산보다작고4bit는일부head에서여유가있다.2bit는모든head에서이충분bound예산을초과한다. 모델출력이나테스트셋을보고설정을선택한것이아니다.',
    '따라서v24에서6/4/2bit×Hybrid/7개시간비율분리전송의24설정을등록하고CPU검사를시작했다.8bit와Mixed는완료된동일벡터비교결과를재사용한다. 모든head/강한초기화/AO30/같은절대오차·시간·전력·배터리를적용한다. 고정bit설정이므로별도adaptive block header는없고각block32bit scale metadata는포함한다.',
    '각저장설계를별도비최적화물리계산으로즉시검사한다. 시작시기존8bit Hybrid/분리전송설계를재현하고의도적전력위반2개가거부됨을확인했다. 정오이후새head최적화를시작하지않고미완료grid는순위에서제외한다. 아직전체GPU학습으로승격하지않았다.',
    '현재v24상태:'+snapshot['precision_status']['status']+', 완료설정수:'+str(snapshot['precision_status'].get('completed_cases', '?'))+'.', '',
    '5. 전체학습큐',
    f"변경조건burst Hybrid100/100·5/5요청전달및최종평가완료. Mixed{progress['mixed_aircomp']['rounds']}/100, 최적시간분리전송{progress['phase_grid']['rounds']}/100. GPU는Mixed한개만사용한다. 기존queued학습소스해시는모두그대로다.",
    'burst는요청오차0.6+공동power solver로원래조건과다르다. 완료Hybrid의통신상한0.140355664J(기존H8보다34.16%감소),validation loss0.968945219/72.8%,retained loss0.946835329/74.5%,forget KL0.026242090/예측차이5.4%,MIA AUC0.503316/재학습과gap0.001144. loss/KL은기존H8보다조금커동등품질의우위로쓰지않는다. 같은새조건의전체Mixed/분리전송이완료되기전승자를정하지않는다.',
    '원래조건완료실험에서는Mixed가현재관찰된에너지·검증loss·forgetKL최저,Hybrid의MIAgap은약간작음이라는오전보고판단이유지된다. 실패/미완료/ideal에너지0은우위순위에서제외한다. 원본manuscript와모든실패/음성결과를보존한다.',
    '다음판단: v22독립검증완료후출력변화를해석하고,v24의완료된동일품질bound비교에서근거가있으면별도CPU one-step품질평가를순차진행한다. 아직응용품질·실측총에너지·전체궤적이득은입증되지않았다. 정오에보고하고heartbeat를일시중지하며정오이후새장시간실험은추가하지않는다.'
]
report = results/'heartbeat_1012_report_ko.txt'; report.write_text('\n'.join(lines)+'\n')
state.update(last_heartbeat_research_kst=now, last_completed_evidence_snapshot='results/overnight_energy_v2/heartbeat_1012_snapshot.json',
    current_gpu_method='burst_joint_v3/mixed_aircomp/training',
    notes='FullFU61 fair58grid complete;48saveddesign certificatespassed. q8nohistory H/P/M mJ=.2:5.810614/3.530960/1.375426;.6:2.710487/2.802450/.198797. Hbeatsphase3.28%onlyloosecondition,Mstilllowest. Rank3paidcodec .2 Hsave1.024%,M5.094%,phasecost.292%;.6allcostmore,negativesretained. V22task19vectorsrunning; V25independentcache/physicalvectorauditqueued. V24actualFU61global6/4/2bit24caseCPUstartedfromquantizationfloorpreview;bothH/strongphase,sameepsilon/time/power/battery,checkerstartupknownrefs+invalidpowerpassed;Mixed/q8exactcomparatorsreused. NoextraGPU/sourcechanges;morningflagtrue;noonpausepolicyunchanged.')
if gpu.strip(): state['current_gpu_pid'] = int(gpu.splitlines()[0].split(',')[0])
write(tools/'campaign_state.json', state)
with (tools/'campaign_plan.txt').open('a') as stream:
    stream.write('\nPOST10:12 '+now+' KST: allFU61fair58grid complete;48savedphysicalcertspassed. q8original.2 H5.810614/P3.530960/M1.375426mJ,loose.6 H2.710487/P2.802450/M.198797mJ. Hbeatsphase3.28%onlyloosecondition,Mstilllowest. Rank3codec .2 Hsave1.024%,M5.094%,phasecost.292%;.6allcostmore,allnegativespreserved. V22taskpilotrunning19vectors;V25independentvector/cacheauditPID'+str(state['fu_quality_audit_v25_cpu_pid'])+'waitsV22. V24PID'+str(state['fu_precision_v24_cpu_pid'])+' registered24actualFU61lowbitcases6/4/2bit,H/strong7fractionphase,all10heads/AO30sameoriginalabsoluteepsilon/time/P/battery,32bitmetadata;2bitfloorfailsallsufficientbounds,4bitroomonlysomehead,6bitall. CodecsappliedtobothH/P;Mixedandq8comparatorsreused. Independentphysicalcheckerknown8bitH/Pandinvalidpowerstartupchecks passed. NoGPU/sourcechanges;Mixed'+str(progress['mixed_aircomp']['rounds'])+'/100andphasewaits; noonpolicyunchanged.\n')

paths = [report, results/'heartbeat_1012_snapshot.json', tools/'capture_1012.py', tools/'campaign_state.json', tools/'campaign_plan.txt',
    root/'results/energy_quality_v1/energy_quality_report.json', root/'results/energy_quality_v1/cohorts.json',
    burst/'status.json', burst/'plan.json', burst/'config.json', results/'queue_recovery_v1/status.json']
for mode in progress:
    for name in ['rounds.jsonl', 'manifest.json']:
        p = burst/mode/name
        if p.exists(): paths.append(p)
for name in ['public_history_fu_v19', 'public_history_fu_v19b', 'fu_quality_v22', 'fu_radio_summary_v23', 'fu_precision_v24', 'fu_quality_audit_v25']:
    for parent in [results, tools]:
        paths.extend(p for p in (parent/name).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.tmp')
paths.extend([tools/'fu_radio_summary_v23.py', tools/'fu_quality_audit_v25.py', results/'transport_certificate_v20/fu_quality_v22.json', tools/'transport_certificate_v20.py'])
contents = {}; omitted = []
for path in dict.fromkeys(paths):
    for attempt in range(5):
        try:
            data = path.read_bytes()
            if path.suffix == '.json': json.loads(data)
            if path.suffix in ['.npz', '.pt']:
                with zipfile.ZipFile(io.BytesIO(data)) as archive: assert archive.testzip() is None
            contents[str(path.relative_to(root))] = data; break
        except (OSError, json.JSONDecodeError, zipfile.BadZipFile, AssertionError): time.sleep(.02)
    else: omitted.append(str(path.relative_to(root)))
manifest = dict(captured_at_kst=now, ongoing_files_are_partial=True, in_progress_files_omitted=omitted,
    files={n: dict(sha256=hashlib.sha256(v).hexdigest(), bytes=len(v)) for n, v in contents.items()})
buffer = io.BytesIO()
with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
    for name, data in contents.items(): archive.writestr(name, data)
    archive.writestr('capture_manifest.json', json.dumps(manifest, indent=2)+'\n')
print(base64.b64encode(buffer.getvalue()).decode())
