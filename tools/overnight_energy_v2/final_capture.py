"""Close the authorized overnight window; archive evidence without starting experiments."""
import base64,hashlib,io,json,subprocess,time,zipfile
from datetime import datetime
from pathlib import Path
r=Path('/home/wnlab/Documents/Codex/2026-10-20/files-pasted-by-the-user-standalone/outputs/ova_CK');f=r/'results/overnight_energy_v2';t=r/'tools/overnight_energy_v2';b=f/'burst_joint_v3'
def read(p):
    for attempt in range(10):
        try:return json.loads(p.read_text())
        except json.JSONDecodeError:
            if attempt==9:raise
            time.sleep(.02)
def write(p,v):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
now=datetime.now().isoformat(timespec='seconds');expected=read(b/'plan.json')['source_sha256'];assert all(sha(r/n)==h for n,h in expected.items())
assert read(f/'inverse_quality_audit_v32/certificate.json')['status']=='passed_all9_physical_vectors_and27_non_test_cohort_metric_sets'
original=read(r/'results/energy_quality_v1/energy_quality_report.json');burst=read(f/'burst_comparison_v31/results.json');q30=read(f/'inverse_quality_v30/summary.json')
progress={}
for mode in ['hybrid','mixed_aircomp','phase_grid']:
    rows=[json.loads(x) for x in (b/mode/'rounds.jsonl').read_text().splitlines() if x.strip()]
    progress[mode]=dict(rounds=len(rows),requests_delivered=sum(x['newly_deleted']>0 for x in rows),last=rows[-1] if rows else None)
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True)
ps=subprocess.check_output(['ps','-u','wnlab','-o','pid,ppid,stat,etime,args'],text=True).splitlines()
snapshot=dict(captured_at_kst=now,scheduled_window_ended_kst='2026-10-21T12:00:00+09:00',report_preparation_late=True,
    automation_pause_confirmed_by_app=True,automation_pause_receipt='ova-ck mode=update status=PAUSED; local automation.toml status verified',
    frozen_source_sha256=expected,burst_status=read(b/'status.json'),progress=progress,gpu_processes=gpu,
    project_processes=[x for x in ps if str(r)+'/' in x],recovery_status=read(f/'queue_recovery_v1/status.json'),
    final_quality_pilot_status=read(f/'inverse_quality_v30/status.json'),final_quality_audit=read(f/'inverse_quality_audit_v32/certificate.json'))
write(f/'final_noon_snapshot.json',snapshot)
lines=['ova_CK 밤샘 연구 마감 보고','작업 기한: 2026-10-21 12:00 KST','실제 마감 캡처: '+now+' KST',
    '정오 마감 정리가 늦었다. 자동 확인 ova-ck는 앱에서 PAUSED로 변경하고 로컬 설정에서도 확인했다. 정오 이후 새 장시간 실험은 시작하지 않았다. 이미 시작된 OMA 학습과 그 실험의 필수 최종 평가는 안전하게 완료하도록 유지했다.','',
    '현재 결론',
    '현재의 합 업데이트 복원 문제와 실험 조건에서는 제안 Hybrid가 에너지 최저라는 근거를 얻지 못했다. 완료된 원래 조건과 변경 조건에서 Mixed AirComp의 통신 에너지가 가장 적었다. Hybrid의 통신 설계를 개선하는 구체적인 방법은 찾았으며, 같은 개선을 비교군에도 적용한 결과를 모두 보존했다.',
    'Mixed가 학습과 언러닝의 합만 전달해도 이상적인 글로벌 업데이트를 직접 구현하는 현재 정의에서는, 두 성분을 분리 복원한다는 사실만으로 VLM loss 개선이 따라오지 않는다. 실제 응용이 두 성분의 개별 신뢰도/처리 확인/후속 사용을 요구하는지 별도로 입증해야 한다. 이번 정밀도 진단에서는 그러한 필요성을 확인하지 못했다.','',
    '1. 원래 조건의 완료된 물리 링크: 요청 포함 모든 라운드 상대 RMS 0.2',
    '각 100/100 라운드, 5/5 요청 전달. 동일 1000개 validation/retained/forget cohort. 수치는 통신 에너지 상한이다.',
    '방법 | E상한/E기댓값 J | validation loss/정확도 | retained loss/정확도 | forget KL(ref||방법)/예측차이 | validation MIA AUC/재학습 AUC와의 절대차']
names=['coupled_mixed_aircomp','coupled_hybrid_q6','coupled_hybrid','puf_regular_adapted_hybrid','coupled_fixed_head']
def metric_line(row):
    return f"{row['method']} | {row['energy_bound_j']:.12f}/{row['expected_energy_j']:.12f} | {row['validation_loss']:.9f}/{100*row['validation_accuracy']:.1f}% | {row['retained_loss']:.9f}/{100*row['retained_accuracy']:.1f}% | {row['forget_kl_reference_to_candidate']:.9f}/{100*row['forget_prediction_disagreement']:.1f}% | {row['forget_loss_attack_auc_validation']:.6f}/{row['forget_attack_auc_gap_to_reference_validation']:.6f}"
for name in names:
    row=next(x for x in original['methods'] if x['method']==name);assert row['status']=='completed';lines.append(metric_line(row))
lines += ['PuF adapted는 retained 업데이트 생략/다른 언러닝 가중치 등의 알고리즘 차이가 있는 변형이며 동일 고정 업데이트의 무선 비교군이나 원 논문의 충실한 재현으로 쓰지 않는다.',
    'Mixed의 validation loss와 forget KL도 관찰값이 가장 낮다. Hybrid의 MIA AUC gap은 조금 낮아 모든 지표의 우위나 여러 학습 seed에 대한 통계적 승자로 표현하지 않는다.',
    '기존 50:50 시간 분리 OMA는 90라운드 완료 후 요청91에서 실행 가능한 설계를 찾지 못했다(E=0.1338143156J). 이는 모든 OMA가 불가능하다는 뜻이 아니다. 미완료 에너지를 완료 방법의 우위 순위에 넣지 않았다. uncoupled Hybrid는 57라운드에서 중단됐고 원인은 확정하지 못했다.','',
    '2. 변경한 에너지–품질 절충 문제의 완료 결과',
    '요청 라운드만 상대 RMS를 0.6으로 완화하고 나머지는 0.2로 유지했다. 동일 joint-power solver, 초기 모델/데이터/요청 일정과 평가 cohort를 세 방식에 공통 적용했다. 원래 0.2 결과와 동등 품질 비교로 합치지 않는다.']
for row in burst['methods']:lines.append(metric_line(row))
lines += ['위 두 방식 모두 100/100 라운드·5/5 요청 전달. Mixed의 에너지 상한은 같은 변경 조건 Hybrid보다 46.6774% 낮다. Hybrid의 validation 정확도(72.8% 대 72.7%)와 forget 예측차이(5.4% 대 5.5%)는 조금 좋으므로 지표별 차이를 유지한다.',
    '고정 모델의 같은 이미지에 대한 paired class-stratified 5000회 bootstrap(Mixed−Hybrid): validation loss −0.000794371, 조건부 95%구간 [−0.001255204,−0.000325238]; retained loss −0.000781750, [−0.001176948,−0.000369792]; forget KL −0.000102887, [−0.000191837,−0.000013054]. 학습 seed/client/radio의 불확실성을 포함한 구간은 아니다.',
    '변경 조건 Mixed는 원래 Mixed보다 에너지 14.3495% 감소했지만 validation loss +0.001088031, retained loss +0.000624349, forget KL +0.000395982였다. 사전 품질 동등성 기준이 없어 동등 품질의 에너지 우위로 주장하지 않는다.','',
    '3. OMA 진행과 남겨 둔 계산',
    f"시간배분을 최적화하는 phase_grid 학습은 11:31:21에 시작했다. 현재 {progress['phase_grid']['rounds']}/100 라운드·요청 {progress['phase_grid']['requests_delivered']}/5 전달. 완료되지 않아 누적 에너지/최종 품질 순위를 매기지 않는다.",
    '대화에서 OMA는 학습/언러닝의 시간 분리다. 디지털 사용자들은 각 구간에서 여전히 NOMA를 쓰므로 전 사용자 TDMA와 구분한다. 요청이 없는 초기 구간은 학습 전송만 있어 아직 동시 학습·언러닝 구간의 비교를 대표하지 않는다.',
    '서버 GPU 학습 PID3657921과 이를 순차 실행하는 controller3534100을 유지한다. 이 실험 종료 후 controller가 필수 최종 평가를 수행한다. 복구 supervisor3546625는 기존 burst 완료를 기다리며 정오 이후 새 학습은 시작하지 않는 제한이 있다. 자동 확인은 일시중지했으므로 후속 결과 알림은 예약하지 않았다.','',
    '4. 실제로 구현하고 검증한 Hybrid 개선',
    '기존 AO에서는 송신 계수를 최적화할 때 수신 계수를 고정한다. 새 v29는 수신 크기의 역수 t=1/s를 사용하여 고정 block 오차예산 r에서 송신·수신을 동시에 최적화한다:',
    'sum_u |H_um*a_um-rho_um*t_m| + sqrt(d_m/(2L_A,m)) <= (r_m-sum_u delta_um)*t_m.',
    '현재 SIC 잔류율0에서 최소 digital power도 간섭 norm으로 제거할 수 있어 이 내부 문제는 SOCP가 된다. 단일 링크 해석적 에너지값, 수신 크기0 처리, 잔류율0 가정 검사 두 검증을 통과했다. 전체 RMS 오차 배분/모든 codec/시간의 전역 최적해를 주장하는 결과는 아니다.',
    '기존의 실행 가능 설계 61개 모두에 같은 개선을 적용했고, 61개가 모두 독립 물리 검증 및 에너지 비증가 조건을 통과했다. 각 61개 block 예산에 대해 10개 head, 총610개를 검사해 새 후보300개를 독립 인증했다. 나머지 실패와 기존 fallback도 보존했으며 수치 실패를 불가능성 증명으로 쓰지 않았다.',
    '같은 실제 요청61, 동결 CPU learning/unlearning 벡터, 원래 global RMS 0.2, 같은 시간·전력·배터리 조건에서의 결과:',
    '방법 | 새 E상한 mJ | 처음 raw8bit E상한 mJ | 적용 설정']
raw={'hybrid':5.810613655,'mixed_aircomp':1.37542619,'phase_grid':3.530959647}
for row in q30['selected']:
    lines.append(f"{row['mode']} | {row['energy_bound_j']*1000:.6f} | {raw[row['mode']]:.6f} | head{row['head']}, bits={row['bits']}, 공개이력rank={row['rank']}, 실제오차유료보고={row['reports']}")
lines += ['Hybrid는 약36.4% 개선했지만 Mixed가 여전히 최저다. 이 절감은 양자화/실제오차 보고/공동 최적화/head 변경을 합친 결과이며, 100라운드 누적 에너지나 순수 solver 효과 하나로 해석하지 않는다.',
    '공개 이력 계수와 실제 양자화 오차 보고는 공짜로 가정하지 않았다. 계수16bit/원격 UAV와 41block×32bit/원격 요청 UAV의 보고, RF 전력/시간/배터리를 계산하고 원래 frame에서 해당 airtime을 차감했다. 실제 오차 보고값은 위로 반올림했다. 추가 codec/solver 계산·header·회로 에너지는 실측하지 않았다.','',
    '5. 새 후보 품질 평가: 실제 요청61 한 단계, 3개 공통 잡음, cohort별200개',
    '완료된 전체61예산×610head 검사에서 방식별 검증 에너지 최저 후보를 모델 출력 추론 전에 고정했다. 이후 9개 후보 출력·27개 cohort 지표를 평가했고, 별도 v32에서 물리 출력벡터 9개 및 loss/정확도/KL/예측차이 27세트를 재계산해 모두 통과했다.',
    '방법 | validation loss | retained loss | 현재 forget의 최종재학습 KL(단계가 달라 참고치) | ideal 한 단계 대비 forget KL']
for mode in ['hybrid','mixed_aircomp','phase_grid']:
    m={x['cohort']:x for x in q30['metrics'] if x['mode']==mode};lines.append(f"{mode} | {m['validation']['loss']:.9f} | {m['retained']['loss']:.9f} | {m['forget']['final_retrain_descriptive_only_kl_reference_to_candidate']:.9f} | {m['forget']['ideal_step_kl_reference_to_candidate']:.9g}")
lines += ['세 방식의 validation 정확도는72.0%, retained 평균72.333%, forget71.0%였다. 공통 잡음3개는 학습 seed3개가 아니며 한 단계의 작은 차이가 전체 학습의 품질 동등성/삭제 보장을 입증하지 않는다. 이 작은 pilot에서 MIA를 새로 추정하지 않았으며, 완료100라운드의 MIA와 구분했다.',
    'Hybrid 새 후보의 retained loss와 최종재학습 forget KL은 공동 최적화 전 유료보고6bit 후보보다 조금 높았다. Mixed도 일부 지표가 악화했다. 오차상한이 같은 것만으로 응용 품질이 같다고 보고하지 않는다.','',
    '6. 연구 해석과 다음에 검증할 주장',
    '논문은 현재 증거로 “NOMA이므로 OMA보다 에너지가 항상 적고 Mixed와 비슷한 에너지로 더 좋은 VLM loss”를 주장할 수 없다. 간섭/SIC/디지털 payload/최적 시간배분과 요구 품질에 따라 순위가 달라진다. 개선된 에너지 설계 자체는 구체적인 기여 후보이나 현재 최저 에너지 비교군을 정직하게 포함해야 한다.',
    '현재 합만 필요한 문제를 유지한다면 Mixed를 포함한 전송방식 선택과 응용 품질 제약 아래 에너지 최소화가 자연스럽다. Hybrid의 분리 복원이 꼭 필요한 문제로 바꾸려면 개별 성분을 요구하는 실제 사용처와 평가 지표를 먼저 제시하고 모든 비교군에 같은 요구를 적용해야 한다. 이번 실험에서 그 추가 응용 이득을 입증하지 못했다.',
    '공개 이력 저랭크 압축, 2/4/6/8bit, 여러 초기값/AO 수렴, 시간배분, 실제오차 유료보고, 송수신 공동 최적화, 모든 head를 점검했다. 듀얼 아날로그/재언러닝 가중치/guard/클리핑/기타 손실 개선 후보의 부정 결과도 이전 capsule과 서버 버전 폴더에 보존했다. 새 GPU 후보 승격은 하지 않았다.',
    '언러닝 자체도 아직 강한 근거가 부족하다. 대부분의 삭제 샘플이 이전 SGD에서 한 번 정도만 쓰인 조건이며, 이상적 링크의 언러닝 대 retained-only 제어 비교도 forget KL 차이의 조건부 구간이0을 포함했다. MIA0.5 근방, 작은 MSE, 요청 전달은 삭제 보장이 아니다.',
    '총 에너지는 무선 통신의 모델 상한/기댓값이다. UAV 비행·로컬 학습·새 codec·head solver를 포함한 실측 총 배터리 에너지로 확대 해석하지 않는다. ideal 링크의 에너지0, 실패/중단/미완료는 에너지 우위 순위에서 제외했다. 테스트셋으로 설정을 고르지 않았다.','',
    '보존과 재현',
    '원본 manuscript와 결과는 변경하지 않았다. 기존 GPU 큐의 동결 source10개 SHA256은 모두 유지됐다. 새 코드는 tools/overnight_energy_v2, 결과는 results/overnight_energy_v2의 별도 버전에 있다.',
    'final_noon_results.zip은 이번 최종 후보/캐시/물리 설계/검증 코드, 이전 원래 조건 결과, 진행 중 프로세스 및 동결 해시를 포함한다. 이전 각 heartbeat zip은 실패/중간 결과와 더 오래된 버전의 추가 근거를 유지한다. 모든 파일의 SHA256을 capture_manifest.json에 기록하고 로컬에서 재확인한다. 자격증명은 저장하지 않았다.',
    '수식 메모: joint_tx_rx_method_v29.txt. 자동 확인은 PAUSED이며 서버의 이미 시작한 phase_grid 학습/최종 평가만 계속된다.'
]
report=f/'final_noon_report_ko.txt';report.write_text('\n'.join(lines)+'\n')
state=read(t/'campaign_state.json');state.update(overnight_window_closed=True,closed_at_kst=now,automation_status='PAUSED',new_long_experiments_after_noon=False,
    final_report='results/overnight_energy_v2/final_noon_report_ko.txt',final_snapshot='results/overnight_energy_v2/final_noon_snapshot.json',current_gpu_method='burst_joint_v3/phase_grid/training',
    inverse_quality_v30_complete=True,inverse_quality_v32_audited=True,notes='Window closed; report preparation late. App automation paused and local TOML verified. No new long jobs after12. Existing11:31phase training+essential final audit continue. New reciprocal-RX fixed-budget convex inner problem improves all61 seeds,610heads300certified;H3.696861mJ butM1.029606 stilllower. All9 physical vectors/27 taskmetricsets independently verified. No taskqualityequivalence/deletion/totalUAVenergy claim. Completedoriginalandchanged fulltaskMixed lowest observed communication energy. Sources intact; no automaticpromotion.')
write(t/'campaign_state.json',state)
with (t/'campaign_plan.txt').open('a') as s:s.write('\nCLOSE '+now+' KST: scheduled noon window closed, report preparation late; automation PAUSED confirmed app/localTOML. No new long experiments after12. Existing phase PID3657921 and controller3534100 continue safely through final audit; recovery supervisor3546625 retains noon start cutoff. V30 complete9/27 at11:48 and independent v32 cached9vectors/27metricsets passed. Final report and snapshot saved; no GPU promotion.\n')
paths=[report,f/'final_noon_snapshot.json',t/'campaign_state.json',t/'campaign_plan.txt',t/'final_capture.py',t/'verify_inverse_v32.py',t/'capture_1113.py',b/'plan.json',b/'config.json',b/'status.json',f/'queue_recovery_v1/status.json',f/'public_history_fu_v19/observer/round_060.pt']
for mode in progress:
    paths.extend(p for p in (b/mode).glob('*') if p.suffix in ['.json','.jsonl'])
paths.extend(p for p in (b/'quality').glob('*') if p.suffix in ['.json','.npz'])
for name in ['public_history_fu_v19/screen','public_history_fu_v19/radio','public_history_fu_v19b','fu_quality_v22','fu_precision_v24','certified_error_v27','certified_error_quality_v28','inverse_budget_v29','inverse_heads_v29b','inverse_quality_v30','inverse_quality_audit_v32','burst_comparison_v31']:
    for parent in [f,t]:paths.extend(p for p in (parent/name).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ['.tmp','.pt'])
for name in ['energy_quality_v1','energy_quality_radio_v1','comparisons_seed0_solver_v2']:
    paths.extend(p for p in (r/'results'/name).rglob('*') if p.is_file() and p.suffix in ['.json','.jsonl'])
for name in ['certified_error_quality_v28.py','inverse_heads_v29b.py','inverse_quality_v30.py','burst_comparison_v31.py','verify_burst_v31.py']:paths.append(t/name)
paths.append(r/'tools/energy_quality_v1/launched_jobs.json')
paths.append(r/'results/energy_quality_v1/retrain_reference_ideal.npz')
paths.extend(r/n for n in expected)
contents={};omitted=[]
for path in dict.fromkeys(paths):
    for attempt in range(5):
        try:
            data=path.read_bytes()
            if path.suffix=='.json':json.loads(data)
            if path.suffix in ['.npz','.pt']:
                with zipfile.ZipFile(io.BytesIO(data)) as z:assert z.testzip() is None
            contents[str(path.relative_to(r))]=data;break
        except (OSError,json.JSONDecodeError,zipfile.BadZipFile,AssertionError):time.sleep(.02)
    else:omitted.append(str(path.relative_to(r)))
manifest=dict(captured_at_kst=now,incomplete_phase_files_are_partial=True,omitted_files=omitted,files={n:dict(sha256=hashlib.sha256(data).hexdigest(),bytes=len(data)) for n,data in contents.items()})
buffer=io.BytesIO()
with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as z:
    for n,data in contents.items():z.writestr(n,data)
    z.writestr('capture_manifest.json',json.dumps(manifest,indent=2)+'\n')
print(base64.b64encode(buffer.getvalue()).decode())
