"""Capture completed low-bit screen, independent FU pilot and paid-report work."""
import base64
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import subprocess
import time
import zipfile

root=Path('/home/wnlab/Documents/Codex/2026-10-20/files-pasted-by-the-user-standalone/outputs/ova_CK')
tools=root/'tools/overnight_energy_v2';results=root/'results/overnight_energy_v2';burst=results/'burst_joint_v3'


def read(path):
    for attempt in range(10):
        try:return json.loads(path.read_text())
        except json.JSONDecodeError:
            if attempt==9:raise
            time.sleep(.02)


def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


now=datetime.now().isoformat(timespec='seconds');state=read(tools/'campaign_state.json');assert state['morning_report_sent']
expected=read(burst/'plan.json')['source_sha256'];assert all(sha(root/n)==h for n,h in expected.items())
low=results/'fu_precision_v24';rows=read(low/'results.json');assert len(rows)==24
assert read(low/'status.json')['saved_feasible_candidates']==len(read(low/'certificates.json'))==15
quality=read(results/'fu_quality_audit_v25/results.json');assert quality['status']=='passed19_vectors_and57_cohort_predictions_and6_energy_only_selections'
old=read(results/'public_history_fu_v19/radio/results.json');best=[]
for bits in [4,6,8]:
    for mode in ['hybrid','phase_grid']:
        eligible=([r for r in old if r['relative_tolerance']==.2 and r['rank']==0 and r['mode']==mode and r['status']=='feasible_radio_only'] if bits==8 else
                  [r for r in rows if r['quantization_bits']==bits and r['mode']==mode and r['status']=='feasible_radio_only'])
        chosen=min(eligible,key=lambda x:x['energy_bound_j']);best.append(dict(chosen,quantization_bits=bits))
progress={}
for mode in ['hybrid','mixed_aircomp','phase_grid']:
    path=burst/mode/'rounds.jsonl';data=[json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []
    progress[mode]=dict(rounds=len(data),requests_delivered=sum(r['newly_deleted']>0 for r in data),last=data[-1] if data else None)
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True)
processes=subprocess.check_output(['ps','-u','wnlab','-o','pid,ppid,stat,etime,args'],text=True).splitlines()
q26=results/'fu_precision_quality_v26';r27=results/'certified_error_v27'
snapshot=dict(captured_at_kst=now,frozen_source_sha256=expected,burst_status=read(burst/'status.json'),burst_progress=progress,
    gpu_processes=gpu,project_processes=[x for x in processes if str(root)+'/' in x],
    completed_lowbit_best=[{k:v for k,v in x.items() if k!='heads'} for x in best],
    completed_v22_v25_task_summary=quality['summary'],completed_v22_v25_task_paired=quality['paired'],
    q26_status=read(q26/'status.json'),q26_completed_variants=len(read(q26/'results.json')),
    r27_status=read(r27/'status.json'),r27_startup=read(r27/'startup_checks.json'),r27_completed_cases=len(read(r27/'results.json')))
if (q26/'summary.json').exists():snapshot['q26_summary']=read(q26/'summary.json')
write(results/'heartbeat_1042_snapshot.json',snapshot)
lines=['ova_CK 밤샘 연구 —10:42 heartbeat 후속결과','캡처:'+now+' KST','',
    '이번완료: 실제요청61의고정CPU업데이트에서저비트24설정을완료했고실행가능최저설계15개를물리단위로검사했다.6비트는유효한통신에너지개선후보다.4비트는단순히bit수가작다는이유로좋지않았고2비트는보수적오차bound조건을통과하지못했다.','',
    '1. 같은업데이트/오차/시간/전력에서의bit비교',
    '공개theta60은별도burst Hybrid궤적에서관찰했고실제요청61의retained/forget minibatch와기존coupled update함수를CPU재현했다. CPU/CUDA수치/랜덤projection실현은다를수있지만아래전송비교군은전부동일한동결벡터를쓴다. 원래0.2전체학습궤적의결과나100round누적결과가아니다.',
    '원래learning norm으로고정한절대오차epsilon0.0023699601564,총uplink0.167936s,peak/average블록전력0.32W,각UAV통신예산0.1J,10head/AO30/convex+4초기화,고정bitquantizer와block별32bitscale metadata를모두적용했다.분리전송은7개시간비율을검색했다.',
    '고정bit | Hybrid통신상한mJ | 최적시간분리전송통신상한mJ']
for bits in [4,6,8]:
    h=next(x for x in best if x['quantization_bits']==bits and x['mode']=='hybrid');p=next(x for x in best if x['quantization_bits']==bits and x['mode']=='phase_grid')
    lines.append(f"{bits} | {1000*h['energy_bound_j']:.6f} | {1000*p['energy_bound_j']:.6f}")
lines += [
    '6bit대8bit:Hybrid8.6096%절감,분리전송15.0455%절감. 하지만Mixed8bit의동일벡터통신상한1.375426mJ보다두방법모두높다. 공개이력rank3를동일하게적용한Mixed는1.305367mJ였다. 새로운codec을제안법에만허용하지않았다.',
    '4bit의큰양자화오차가무선오차예산을잠식해더높은analog전력을요구했다. 따라서payload절감보다analog에너지증가가커졌다.2bit의최악quantization-bound floor는모든head에서이충분bound예산을초과한다. 이결과는원래physical/task문제가전역적으로불가능하다는증명이아니다.',
    '각저장된실행가능후보는quantization복원/원래W·J단위의power·battery·SICpayload·mapping·시간·noise RMS를비최적화계산으로확인했다. 초기검사에서알려진8bitHybrid/분리전송설계를재현하고의도적power위반을거부했다. 미저장head나전역최적성을검증했다는뜻은아니다.','',
    '2. 먼저시작한공개이력압축의실제요청품질pilot완료',
    'v22의19one-step벡터와57개cohort출력및6개설계선택을v25가독립검증했다. 전체58개무선설정에서에너지로만선택한rank0/rank3×Hybrid/Mixed/시간분리전송,3공통잡음,각200validation/retained/현재삭제요청sample이다.벡터를물리식으로재구성하고loss/accuracy/두reference KL/예측차이를캐시에서재계산했다.',
    '방법/rank | validation loss/정확도 | retained loss/정확도 | forget ideal-stepKL | forget final-retrainKL(단계다름,참고값) | forget final-reference예측차이']
for mode in ['hybrid','mixed_aircomp','phase_grid']:
    for rank in [0,3]:
        s={x['cohort']:x for x in quality['summary'] if x['mode']==mode and x['rank']==rank};v=s['validation'];r=s['retained'];f=s['forget']
        lines.append(f"{mode}/rank{rank} | {v['loss']:.9f}/{100*v['accuracy']:.1f}% | {r['loss']:.9f}/{100*r['accuracy']:.3f}% | {f['ideal_step_kl']:.9g} | {f['final_retrain_descriptive_only_kl']:.9f} | {100*f['final_retrain_descriptive_only_prediction_disagreement']:.3f}%")
lines += [
    '차이는작고rank3효과도방식/지표에따라다르다. 최종retained-retrain100은round61과단계가달라그KL은기술적참고치이며삭제보장/동등품질인증이아니다.3잡음은3학습seed가아니다. 요청전달/MSE/MIA0.5근방을삭제증명으로쓰지않으며이작은pilot의MIA를새로추정하지않았다.','',
    '3. 새6bit후보의별도품질검사',
    f"v26상태:{snapshot['q26_status']['status']},완료{snapshot['q26_completed_variants']}/6새벡터. Hybrid/분리전송의bit설정을완료된24개저비트설정과기존8bit중통신에너지로만고른결과둘다6bit였다. 새출력평가전에선택·벡터해시·물리certificate를등록했다. 같은theta60/동결업데이트/200씩세cohort/공통잡음61·1061·2061를유지한다.",
    '기존19개V22모델추론은재실행하지않고8bit/ideal/Mixed출력을재사용한다. 새6개벡터만CPU에서평가하고모든새출력의loss/accuracy/KL/예측차이를별도산술검사한다. 추가codec계산/circuit/header에너지는미측정이며아직전체GPU학습으로승격하지않았다.']
if 'q26_summary' in snapshot:
    lines.append('캡처시v26도완료됐으며전체6개벡터/18개cohort산술검사와6bit·8bit짝비교가동봉summary.json에있다.')
lines += ['',
    '4. 새로운에너지개선방법검사: 실제양자화오차를유료보고',
    '실제로계산한decoded-minus-original norm은최악quantizerbound보다작았다.4bit의경우기존최악bound는일부head에서만원래epsilon여유가있었으나계산된local오차norm은모든head에서여유가있었다. 이를설계에알리는추가통신비용을지불할때도이득이남는지v27로검사한다. 모델출력/테스트셋으로고른후보가아니다.',
    '각remote요청UAV가41개block오차norm을위로반올림한32bitfloat로보낸다. 추가1312bits를0.5bit/complexsymbol의2624symbol TDMA packet으로전송한다. head자체요청은local이므로RF보고없다. remote요청자1명/2명에따라mainframe에서2624/5248symbols를정확히차감하고실제packetpower/RF/각battery/coherence비용을검사한다. 추가오차정보가공짜라고가정하지않았다.',
    '서로독립된고정4/6/8bit설정×Hybrid/강한시간분리전송27설정을등록했다. bit선택은실험설정이며여러bit오차profile을공짜로모두보고하는온라인adaptive scheme으로해석하지않는다. original worstbound 프로토콜과Mixed도그대로비교후보로남긴다. Mixed는이보고가불필요하므로원래frame을사용한다.',
    '추가보고때문에분리전송analog최소mapping경계가head별로달라진다.7개시간비율에각head의정확한L_A=d경계를추가했다. 같은epsilon/time/power/battery와동결update,강한AO/초기화를양쪽에동일적용한다. 새로운control/protocol버전이며원본문제/결과를덮어쓰지않았다.',
    f"v27상태:{snapshot['r27_status']['status']},완료{snapshot['r27_completed_cases']}/27설정. 알려진8bit설계재현/전력위반거부/위반없는오차float상향반올림/head자체요청의packet차감검사가통과했다. 각저장후보를다시독립물리식으로검사한다. 부분grid로우위를선언하지않는다.",
    '오차norm계산/코딩/새header/수신처리/추가회로에너지는미측정이다. RF에너지이득이있더라도이추가비용까지잰총UAV에너지개선이라고쓰지않는다. 모든실패/부정결과와기존강한비교군을보존한다.','',
    '5. 기존전체학습큐',
    f"burst Hybrid100/100·요청5/5전달및최종품질평가완료,Mixed{progress['mixed_aircomp']['rounds']}/100·요청{progress['mixed_aircomp']['requests_delivered']}/5전달,최적시간분리전송{progress['phase_grid']['rounds']}/100. GPU는Mixed한개만사용하고기존소스해시는모두그대로다.",
    'burst는요청오차0.6와공동power solver를적용한별도조건이다. 원래조건의100round완료표와합치지않는다. 원래조건에서Mixed가관찰된에너지/검증loss/forgetKL최저라는오전보고판단은유지된다. Hybrid의MIAgap은약간작고seed1개라모든지표/통계적우승을주장하지않는다. ideal에너지0과미완료/실패는에너지우위순위에서제외한다.',
    '정오전에시작된계산은안전하게완료하게두고정오이후새장시간실험은추가하지않는다. 정오보고후heartbeat일시중지정책을유지한다. 다음판단은6bit출력검사와추가보고프로토콜의완료된전체grid결과에근거한다.'
]
report=results/'heartbeat_1042_report_ko.txt';report.write_text('\n'.join(lines)+'\n')
state.update(last_heartbeat_research_kst=now,last_completed_evidence_snapshot='results/overnight_energy_v2/heartbeat_1042_snapshot.json',
    current_gpu_method='burst_joint_v3/mixed_aircomp/training',
    notes='V24complete24lowbitFU61cases/15savedphysicalcerts:6bitH5.310343mJ(-8.61%),phase2.999709(-15.05%)vs8bit,stillaboveMixed1.375426/rank3M1.305367.4bitcostmore/2bitfailsconservativefloor. V22/V25complete19vectors57cohortmetrics6energyselectionsindependentlychecked,smallmixedqualityeffects,noequivalence/deletionclaim. V26sixnew6bitvectorsCPU,noGPU,baselinecachesreused. V27paidactualquantizationnormreports27caseCPUstarted:1312bits/remoteF,2624symbols/packet,subtractexactframe/paypowerRFbattery;4/6/8fixedbits,H/strongphase+minimumdimensionboundaries,originalepsilon. AllnegativecontrolsandMixedkept,computecircuitheadersunmeasured. ExistingGPUsourcefreezeintact,morningflagtrue,noonreportpausepolicyunchanged.')
if gpu.strip():state['current_gpu_pid']=int(gpu.splitlines()[0].split(',')[0])
write(tools/'campaign_state.json',state)
with (tools/'campaign_plan.txt').open('a') as stream:
    stream.write('\nPOST10:42 '+now+' KST: V24complete24/15savedphyscerts. SameFU61vectors6bitH5.310343mJ(-8.61%vs8bit),phase2.999709(-15.05%),stillaboveM1.375426/rank3M1.305367.4bitmoreenergy,2bitfailsconservativefloor,negativepreserved. V22/V25complete19vectors57cohortmetrics6energyselectionsindependent:smallrank3qualityeffects,noequivalence/deletionclaim. V26CPU'+str(state['fu_precision_quality_v26_cpu_pid'])+' sixnew6bitvectors,baselinecachesreused,selectedbycompletedenergygridbeforeoutputs. V27CPU'+str(state['certified_error_v27_cpu_pid'])+' started27fixed4/6/8bitactual-errornorm-reportcases:1312additionalbitsperremoteF,2624symbolsTDMApacket,head-ownedlocalnoRF,exacttime/power/batterycost;allH/strong7fractionphase+headdependentminimumL_A=d boundary,sameoriginalepsilon. Notfreeadaptivebitprofilesorquantizationinformation;originalworstbound/Mixedremaineligible,extracompute/circuit/headerunmeasured. Startupknownrefs/powerviolation/upwardrounding/headpacketcountpassed. NoGPU/sourcechanges;Mixed'+str(progress['mixed_aircomp']['rounds'])+'/100phasewaits;morningflagtrue,noonpolicyunchanged.\n')

paths=[report,results/'heartbeat_1042_snapshot.json',tools/'capture_1042.py',tools/'campaign_state.json',tools/'campaign_plan.txt',
    root/'results/energy_quality_v1/energy_quality_report.json',root/'results/energy_quality_v1/cohorts.json',burst/'plan.json',burst/'config.json',burst/'status.json',results/'queue_recovery_v1/status.json',
    results/'public_history_fu_v19/screen/plan.json',results/'public_history_fu_v19/screen/vectors.npz',results/'public_history_fu_v19/observer/round_060.pt',
    results/'public_history_fu_v19/radio/results.json',results/'public_history_fu_v19b/results.json',results/'transport_certificate_v20/fu_quality_v22.json']
for mode in progress:
    for name in ['rounds.jsonl','manifest.json']:
        path=burst/mode/name
        if path.exists():paths.append(path)
for name in ['fu_quality_v22','fu_quality_audit_v25','fu_precision_v24','fu_precision_quality_v26','certified_error_v27']:
    for parent in [results,tools]:paths.extend(p for p in (parent/name).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.tmp')
paths.extend([tools/'fu_quality_audit_v25.py',tools/'fu_precision_quality_v26.py'])
contents={};omitted=[]
for path in dict.fromkeys(paths):
    for attempt in range(5):
        try:
            data=path.read_bytes()
            if path.suffix=='.json':json.loads(data)
            if path.suffix in ['.npz','.pt']:
                with zipfile.ZipFile(io.BytesIO(data)) as archive:assert archive.testzip() is None
            contents[str(path.relative_to(root))]=data;break
        except (OSError,json.JSONDecodeError,zipfile.BadZipFile,AssertionError):time.sleep(.02)
    else:omitted.append(str(path.relative_to(root)))
manifest=dict(captured_at_kst=now,ongoing_files_are_partial=True,in_progress_files_omitted=omitted,
    files={n:dict(sha256=hashlib.sha256(v).hexdigest(),bytes=len(v)) for n,v in contents.items()})
buffer=io.BytesIO()
with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as archive:
    for name,data in contents.items():archive.writestr(name,data)
    archive.writestr('capture_manifest.json',json.dumps(manifest,indent=2)+'\n')
print(base64.b64encode(buffer.getvalue()).decode())
