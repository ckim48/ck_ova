"""Completed burstMixed, started strongphase, new jointTX/RX evidence."""
import base64
from datetime import datetime
import hashlib,io,json
from pathlib import Path
import subprocess,time,zipfile

root=Path('/home/wnlab/Documents/Codex/2026-10-20/files-pasted-by-the-user-standalone/outputs/ova_CK');tools=root/'tools/overnight_energy_v2';f=root/'results/overnight_energy_v2';b=f/'burst_joint_v3'


def read(p):
    for attempt in range(10):
        try:return json.loads(p.read_text())
        except json.JSONDecodeError:
            if attempt==9:raise
            time.sleep(.02)


def write(p,v):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


now=datetime.now().isoformat(timespec='seconds');state=read(tools/'campaign_state.json');assert state['morning_report_sent']
expected=read(b/'plan.json')['source_sha256'];assert all(sha(root/n)==h for n,h in expected.items())
comparison=read(f/'burst_comparison_v31/results.json');assert read(f/'burst_comparison_v31/certificate.json')['status'].startswith('passed6_non_test')
methods={r['method']:r for r in comparison['methods']};hybrid=methods['burst_joint_v3/hybrid'];mixed=methods['burst_joint_v3/mixed_aircomp']
head_summary=read(f/'inverse_heads_v29b/summary.json');assert head_summary['cases']==61 and head_summary['head_candidates']==610 and head_summary['certified_heads']==300
inverse=read(f/'inverse_budget_v29/summary.json');assert inverse['certified_nonincreasing']==61
q28=read(f/'certified_error_quality_v28/summary.json');q30=f/'inverse_quality_v30'
progress={}
for mode in ['hybrid','mixed_aircomp','phase_grid']:
    p=b/mode/'rounds.jsonl';rows=[json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []
    progress[mode]=dict(rounds=len(rows),requests_delivered=sum(x['newly_deleted']>0 for x in rows),last=rows[-1] if rows else None)
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True)
processes=subprocess.check_output(['ps','-u','wnlab','-o','pid,ppid,stat,etime,args'],text=True).splitlines()
snapshot=dict(captured_at_kst=now,frozen_source_sha256=expected,burst_status=read(b/'status.json'),burst_progress=progress,
    gpu_processes=gpu,project_processes=[x for x in processes if str(root)+'/' in x],completed_burst_comparison=comparison,
    paid_error_report_quality=q28['metrics'],joint_solver_fixed_head=inverse,all_head_refinement=head_summary,
    q30_status=read(q30/'status.json'),q30_completed_variants=len(read(q30/'results.json')) if (q30/'results.json').exists() else 0)
write(f/'heartbeat_1113_snapshot.json',snapshot)
lines=['ova_CK 밤샘 연구 —11:13 heartbeat 후속결과','캡처:'+now+' KST','',
    '중요한새상태: 변경조건Mixed의100round전체학습과최종품질평가가완료됐고,시간배분최적화분리전송의전체학습이11:31:21에시작됐다. 제안법의에너지개선도더찾았으나Mixed에도동일한개선을적용하면아직Mixed가가장낮다. 원래manuscript/결과/실패를보존하고모든변경은별도버전이다.','',
    '1. 같은변경조건의완료100round비교',
    '요청오차비율0.6/나머지0.2,공동power solver의동일설정/초기모델/요청일정/cohort/numeric evaluator. 아래두방법모두100/100round·5/5요청전달·누적uplink16.7936s. 같은설정이지만실제모델궤적은달라진다.',
    '방법 | 통신에너지상한J | validation loss/정확도 | retained loss/정확도 | forgetKL(ref||method)/예측차이 | validation loss-MIA AUC/재학습AUC와gap']
for label,row in [('Hybrid',hybrid),('Mixed',mixed)]:
    lines.append(f"{label} | {row['energy_bound_j']:.12f} | {row['validation_loss']:.9f}/{100*row['validation_accuracy']:.1f}% | {row['retained_loss']:.9f}/{100*row['retained_accuracy']:.1f}% | {row['forget_kl_reference_to_candidate']:.9f}/{100*row['forget_prediction_disagreement']:.1f}% | {row['forget_loss_attack_auc_validation']:.6f}/{row['forget_attack_auc_gap_to_reference_validation']:.6f}")
lines += [
    'Mixed의통신에너지상한은같은변경조건Hybrid보다46.6774%낮다. validation/retained loss와forgetKL도관찰값은Mixed가조금낮지만,validation정확도와forget예측차이는Hybrid가0.1%p좋아모든지표의우위로쓰지않는다. 사전품질동등성임계값/여러training seed가없으므로동등품질인증이나통계적전체승자를선언하지않는다.',
    '동일1000개cohort의paired class-stratified5000회bootstrap(Mixed−Hybrid):validation loss−0.000794371,조건부95%[−0.001255204,−0.000325238];retained loss−0.000781750,[−0.001176948,−0.000369792];forgetKL−0.000102887,[−0.000191837,−0.000013054]. 고정모델의image조건부분석이며training seed/client/radio불확실성을포함하지않는다는한계를유지한다.',
    '각cache의6개비테스트cohort loss/accuracy/referenceKL/예측차이와100round에너지합·5요청전달합계를독립산술검사했다. 추가추론/테스트기반설정선택은없다. 요청전달과작은MSE/MIA0.5근방은삭제보장이아니다.',
    '새Mixed는원래0.2실험Mixed0.087379794J대비14.3495%낮지만validation loss+0.001088031,retained loss+0.000624349,forgetKL+0.000395982로품질지표가일부나빠졌다. 요청오차와solver/궤적을함께바꿔오차완화만의인과효과나동등품질의우위로쓰지않는다.','',
    '2. 사용자가물은시간최적화OMA상태',
    f"phase_grid 전체학습이11:31:21에시작됐으며현재{progress['phase_grid']['rounds']}/100round·요청{progress['phase_grid']['requests_delivered']}/5전달이다. GPU는이학습하나만사용한다. 완료하지않은누적에너지는완료방법과순위를매기지않는다.",
    '대화상OMA는learning/unlearning시간분리이며digital사용자끼리는NOMA다. 전사용자TDMA와같지않다. 예전50:50은요청91에서설계를찾지못해90round완료했지만이강한7비율검색baseline의불가능성을뜻하지않는다.',
    '현재분리전송은이미정오전에시작한계산이므로정오후에도그실험과필수최종평가가안전하게끝나도록둔다. 정오이후새장시간실험을추가하지않는다.','',
    '3. 추가에너지개선: 수신역수변환으로송신·수신공동최적화',
    '고정head/codec/시간과각block RMS예산r_m에서t_m=1/s_m으로바꾸면오차제약은 sum_u|H_um*a_um−rho_um*t_m|+sqrt(d_m/(2L_A,m)) <= (r_m−sum_u delta_um)*t_m 이된다. 현재SICresidual0조건에서최소digitalpower도analog간섭의2차norm으로제거할수있어전체송신·수신동시설계가SOCP가된다. 기존AO의수신계수고정으로생기는에너지정체를줄인다.',
    '단일링크해석적최저에너지와의일치,zero수신계수의명시처리,nonzeroSIC가정의거부검사2개를통과했다. v29는기존58history/24lowbit/27paid-report전체grid중원래0.2조건의저장된61개실행가능incumbent를모두같은방식으로개선했고61개가모두독립물리검증및에너지비증가조건을통과했다. 제안법에만강한solver를적용하지않았다.',
    'v29b에서는각61개의고정block오차예산에대해모든10head,총610head를검사했다.300개의새head설계가독립검증을통과했으며61개의기존검증해도fallback으로유지했다. 나머지조건실패/해를못찾은상태도저장했고원래문제의불가능성으로해석하지않았다.',
    '이볼록성은고정된block오차배분/head/codec/시간의내부문제에한정된다. 전체RMS예산배분이나모든가능한codec/시간에대한원래문제의전역최적성을주장하지않는다. allhead검사에서도numerical실패를불가능성증명으로쓰지않는다. 추가solver계산/circuit/header에너지는미측정이다.',
    '같은실제요청61의동결CPU learning/unlearning과원래global0.2오차조건에서각방식의새통신에너지상한mJ:']
for row in head_summary['selected']:
    lines.append(f"{row['mode']}: {1000*row['refined_energy_j']:.6f}mJ, head{row['head']}, bit{row['bits']}, public-history rank{row['rank']}, 실제오차추가보고={row['reports']}.")
lines += [
    'Hybrid는처음같은벡터8bit5.810614mJ에서새3.696861mJ로약36.4%낮아졌다. bit/실제오차유료보고/공동최적화/head변경을함께포함한절감이며순수solver효과한가지만의수치가아니다. 분리전송도3.530960→2.508936mJ, Mixed도1.375426→1.029606mJ로좋아졌다. 새Mixed가여전히가장적다.',
    '모든history계수와실제오차보고의추가RF/시간/power/battery비용을포함했다. 공개이력rank3Mixed는계수16bit/remoteUAV를전달하며,오차보고Hybrid/분리전송은41block×32bits=1312bit/remote요청UAV의2624symbol packet을원래frame에서차감한다. 실제양자화오차float는위로반올림한다. 불필요한보고를Mixed에강제로부과하지않았다.',
    'CPU source theta60은별도burstHybrid의공개상태이며CUDA와CPU수치/랜덤projection실현은다를수있다. 전송방식들은같은동결벡터를쓰지만이수치를원래0.2전체학습궤적이나100round누적에너지결과와합치지않는다.','',
    '4. 완료된유료오차보고품질pilot과새후보평가',
    'v28의6개one-step벡터/18개cohort산술검사가11:27:20에완료됐다. 각200validation/retained/현재요청forget,3공통잡음이다. 기존8bit/6bit/ideal출력은재추론하지않고재사용했다. 아래는공동역수최적화전의유료보고6bit후보평균이다.',
    '방법 | validation loss/accuracy | retained loss/accuracy | forget final-retrainKL(단계다름,참고치)']
for mode in ['hybrid','phase_grid']:
    s={x['cohort']:x for x in q28['metrics'] if x['mode']==mode and x['protocol']=='paid_actual_error_report6'};v=s['validation'];r=s['retained'];ff=s['forget']
    lines.append(f"{mode} | {v['loss']:.9f}/{100*v['accuracy']:.1f}% | {r['loss']:.9f}/{100*r['accuracy']:.3f}% | {ff['final_retrain_descriptive_only_kl_reference_to_candidate']:.9f}")
lines += [
    '차이는작고지표방향이항상일치하지않는다. 최종retained-retrain100은round61과단계가달라그KL은삭제보장/품질동등성증거가아니다.3잡음은3학습seed가아니며작은pilot에서MIA를새로추정하지않았다. 전체완료100round MIA는위표에유지한다.',
    f"v30은완료된61예산×610head의검증된통신에너지최저후보를방식별로출력추론전에고정했다. Hybrid6bit+보고/Mixed rank3/분리전송4bit+보고의9개one-step벡터(3공통잡음)를같은cohort로CPU순차평가중이다.현재{snapshot['q30_completed_variants']}/9완료,상태:{snapshot['q30_status']['status']}. 기존GPU학습큐는변경하지않았다.",
    '같은오차상한이같은VLM품질을뜻하지않으므로새후보의품질검사를마치기전전체학습으로승격하거나최종우위로보고하지않는다. 정오전시작한이pilot은안전하게완료하게두되정오이후새장시간실험은추가하지않는다.','',
    '5. 보존과종료정책',
    '현재기존학습source10개해시는모두정확히유지됐다. 오전9시전체보고는이미완료했으며반복하지않았다. 원래조건완료Mixed가관찰된통신에너지·검증loss·forgetKL최저라는기존판단도유지된다. ideal에너지0·실패/미완료는에너지우위순위에서제외한다.',
    '에너지는모델링된통신상한/기댓값이고UAV비행·로컬학습·새codec·headsolver계산을포함한실측총배터리에너지가아니다. 실제삭제보장으로해석하지않는다. 정오에밤샘결과와진행중인계산을보고하고heartbeat를일시중지한다.',
    '공동최적화수식의구체적메모는joint_tx_rx_method_v29.txt,전체원시근거와동결해시는heartbeat_1113_results.zip에보존했다.'
]
report=f/'heartbeat_1113_report_ko.txt';report.write_text('\n'.join(lines)+'\n')
state.update(last_heartbeat_research_kst=now,last_completed_evidence_snapshot='results/overnight_energy_v2/heartbeat_1113_snapshot.json',
    current_gpu_method='burst_joint_v3/phase_grid/training',burst_mixed_completed=True,burst_mixed_final_quality_evaluated=True,burst_comparison_v31_completed=True,
    notes='ChangedburstMixed100/5auditedE.074841260J vsH.140355664J(-46.68%),valloss.968150849vs.968945219,forgetKL.026139203vs.026242090;Hvalacc72.8vsM72.7,predgapH5.4vsM5.5,notallmetricwinner. StrongphaseGPUstarted11:31before12. NewinverseRXfixedblockbudgetSOC:61oldincumbentsallcertifiednonincrease,all610heads300newcertified,fallbacksretained;bestCPUFU61H3.696861/Mrank3 1.029606/phase4bit2.508936mJ;nooriginalglobaloptimum,totalenergyorNOMAclaim. V28paiderrorreportqualitycomplete6/18checks;V30nineallheadconvexcandidatesCPUqualityrunning,energy-onlypreselected,nocurrentGPU/sourcechanges. Morningflagtrue,noonreportpause/no-new-longjobsafter12unchanged.')
if gpu.strip():state['current_gpu_pid']=int(gpu.splitlines()[0].split(',')[0])
write(tools/'campaign_state.json',state)
with (tools/'campaign_plan.txt').open('a') as stream:
    stream.write('\nPOST11:13 '+now+' KST: changedMixedcomplete100/5/finalaudit,E.074841260JvsH.140355664(-46.68%),loss/FKLslightlylowerbutHvalacc72.8vsM72.7andpredgap5.4vs5.5,notallmetric/equivalenceclaim. StrongphaseGPUstarted11:31:21before12;allowitsalready-startedexperimentandfinalauditfinishafter12,noextraexperiment. V29inverseRXfixedblockbudgetSOCP analytic+zeroRx/SICtests2pass,all61savedincumbentscertifiednonincrease. V29b61budgetvectorsall610heads/300newcertified,allfallbacks/failuresretained;bestCPUactualFU61H3.696861/Mrank3 1.029606/phase4bit2.508936mJ,sameoriginalglobal.2/time/P/battery/paidmetadata. Convexinnernotglobalbudgetallocation/headinfeasibilityproof;solvercomputecostunmeasured. V28paiderror6bitpilot6vectors18checkscomplete11:27. V30CPU'+str(state['inverse_quality_v30_cpu_pid'])+' nine3modevectors/3commonnoiseenergy-onlyselectedafterfull61/610physicalchecks,noextraGPU. V31cachepaired5000andindependent6cohort+2energy/requesttotalsverified. Allfrozen10sourcesintact,morningflagtrue,noonreportpausepolicyunchanged.\n')

paths=[report,f/'heartbeat_1113_snapshot.json',tools/'capture_1113.py',tools/'campaign_state.json',tools/'campaign_plan.txt',
    root/'results/energy_quality_v1/energy_quality_report.json',root/'results/energy_quality_v1/cohorts.json',b/'plan.json',b/'config.json',b/'status.json',f/'queue_recovery_v1/status.json',
    f/'public_history_fu_v19/screen/plan.json',f/'public_history_fu_v19/screen/vectors.npz',f/'public_history_fu_v19/observer/round_060.pt']
for mode in progress:
    for name in ['rounds.jsonl','manifest.json']:
        p=b/mode/name
        if p.exists():paths.append(p)
for name in ['report.json','cohorts.json','hybrid.cache.json','mixed_aircomp.cache.json','hybrid.npz','mixed_aircomp.npz']:
    p=b/'quality'/name
    if p.exists():paths.append(p)
for name in ['certified_error_v27','certified_error_quality_v28','inverse_budget_v29','inverse_heads_v29b','inverse_quality_v30','burst_comparison_v31']:
    for parent in [f,tools]:paths.extend(p for p in (parent/name).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.tmp')
paths.extend([tools/n for n in ['certified_error_quality_v28.py','inverse_heads_v29b.py','inverse_quality_v30.py','burst_comparison_v31.py','verify_burst_v31.py']])
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
manifest=dict(captured_at_kst=now,ongoing_files_are_partial=True,in_progress_files_omitted=omitted,files={n:dict(sha256=hashlib.sha256(v).hexdigest(),bytes=len(v)) for n,v in contents.items()})
buffer=io.BytesIO()
with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as archive:
    for n,data in contents.items():archive.writestr(n,data)
    archive.writestr('capture_manifest.json',json.dumps(manifest,indent=2)+'\n')
print(base64.b64encode(buffer.getvalue()).decode())
