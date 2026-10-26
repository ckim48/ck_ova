"""Freeze the first paid-metadata radio improvement and its limits."""
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


def write(path, value):
    tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(value, indent=2)+'\n'); tmp.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


expected = read(suite/'suite_metadata.json')['source_sha256']
assert all(sha(root/n) == value for n, value in expected.items())
v15 = results/'public_history_fp16_v15'
certificates = read(v15/'physical_certificates.json')
assert certificates['checked_saved_radio_designs'] == 30
now = datetime.now().isoformat(timespec='seconds')
rows15 = read(v15/'results.json'); rows13 = read(results/'public_history_v13/radio/results.json')
quality = read(root/'results/energy_quality_v1/energy_quality_report.json')
processes = subprocess.check_output(['ps', '-u', 'wnlab', '-o', 'pid,ppid,stat,etime,args'], text=True).splitlines()
gpu = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader'], text=True)
burst_path = results/'burst_joint_v3/hybrid/rounds.jsonl'
burst_rows = [json.loads(x) for x in burst_path.read_text().splitlines() if x.strip()]
snapshot = dict(captured_at_kst=now, source_sha256=expected, public_history_fp16_status=read(v15/'status.json'),
                public_history_fp16_results=rows15, public_history_fp16_summary=read(v15/'summary.json'),
                public_history_fp16_physical_certificates=certificates, public_history_float32_results=rows13,
                early_replay_registered_plan=read(results/'public_history_early_v17/screen/plan.json'),
                early_observer_status=read(results/'public_history_early_v17/observer/status.json'),
                early_replay_status=read(results/'public_history_early_v17/screen/status.json'),
                burst_status=read(results/'burst_joint_v3/status.json'), burst_completed_rounds=len(burst_rows),
                burst_last_round=burst_rows[-1], gpu_processes=gpu,
                project_processes=[x for x in processes if str(root)+'/' in x],
                observed_original_frontier=quality['observed_frontier'], original_selection_status=quality['selection_status'])
write(results/'heartbeat_0742_snapshot.json', snapshot)
state = read(tools/'campaign_state.json')
state.update(last_heartbeat_research_kst=now, current_gpu_method='burst_joint_v3/hybrid', current_gpu_pid=3602333,
             public_history_fp16_v15_completed=True, public_history_fp16_v15_physical_audit_passed=True,
             public_history_fp16_v15_radio_saving_fraction=0.017938499956994276,
             public_history_fp16_v15_gpu_promoted=False, public_history_early_v17_gpu_promoted=False,
             last_completed_evidence_snapshot='results/overnight_energy_v2/heartbeat_0742_snapshot.json',
             notes='v15 paid16bit rank3 modest RF+fixedOH gain1.79385% in actual learning-only round100, independent30design physical/halfcodec audits pass. Extra network compute/headers must staybelow14.774uJ; not measured or task/FU/NOMA-superiority. Rank1 and original32bit failures preserved. v17 pre-registered actual earlierround30 beforepublic26..29occur, readonlyobserver3608159/replay-radioCPU3608882 wait theta29; samecodec/params/allhead budgets, no test/validation choice. No GPU promotion. Existing burstHybridGPU3602333 only; Mixed/optimizedphase wait; core/queued sources unchanged. Morningflag/noonpolicy unchanged.')
write(tools/'campaign_state.json', state)
with (tools/'campaign_plan.txt').open('a') as f:
    f.write('\nHeartbeat07:42 evidence captured '+now+' KST\n'
            'v15 complete07:30:04; independent physical/codec audit07:45:57 passesall30designs. Baseline823.615818uJ; half16rank1 828.208353uJ (loses0.5576%); half16rank3 808.841386uJ (saves1.79385%,14.774432uJ). Rank3 residualdata+fixedOH786.668248uJ pluspaidcoefficientRF22.173138uJ. Same167936symbols, originalabsoluteRMS1.790535297e-6/power.32W/battery.1J, all10heads, finiteAO30. Besthead5 forall. Codecactual432bits/head,864TDMAcontrolsymbols. Quantizedcoefficientroundingcancelswithresidual, actualnetworkidealreconstructionchecked.\n'
            'This is a modest genericFL RF+fixedOH saving only. Additionalnetworkcompute/basis/receiver/coding/memory/headersmustfit14.774uJ; projection/residual alone listed2457720FLOPs=>6.011pJ/FLOP break-even beforeothercosts, nohardwaremeasurement. No FUrequest, no VLM task/forgetting improvement or Hybrid superiority, no GPU promotion. Keep32bitrank1/3and16bitrank1negativecontrols.\n'
            'v17 pre-registered earlieractualround30 atpublic26..29, observer3608159 sourceburstHybridGPU3602333, replay-radioCPU3608882. Sourceobservedbefore26, samehalfcodec/publicscale/allheadtime/power/battery/absoluteRMS, no validation/test outputs used. Wait for completedpublic29 then replayactualSGDminibatches and paidradio; no duplicatedGPU or queuedsource edits. Morning09reportflagfalse, noonstop-new-longjobs/pausepolicyunchanged.\n')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plot_rows = [rows15[0]['best'], rows13[1]['best'], rows13[2]['best'], rows15[1]['best'], rows15[2]['best']]
labels = ['Baseline', '32-bit\nrank1', '32-bit\nrank3', '16-bit\nrank1', '16-bit\nrank3']
overhead = 150.
data_energy = [(x['analog_data_plus_original_overhead_j']*1e6-overhead) for x in plot_rows]
control = [x['coefficient_tx_energy_j']*1e6 for x in plot_rows]
total = [x['total_modeled_energy_j']*1e6 for x in plot_rows]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.1), gridspec_kw={'width_ratios': [1.5, 1]})
axes[0].bar(labels, [overhead]*5, label='Original fixed overhead', color='#b5b5b5')
axes[0].bar(labels, data_energy, bottom=[overhead]*5, label='Analog data RF', color='#40778e')
axes[0].bar(labels, control, bottom=[overhead+x for x in data_energy], label='Coefficient RF', color='#ce8841')
for x, y in enumerate(total):
    axes[0].text(x, y+12, f'{y:.1f}', ha='center', fontsize=9)
axes[0].set_ylim(0, 1000); axes[0].set_ylabel('Modeled communication energy (microjoules)')
axes[0].set_title('Actual learning-only round100, all paid metadata')
axes[0].legend(fontsize=8, frameon=False, loc='upper right')
delta = [x-total[0] for x in total[1:]]
axes[1].bar(labels[1:], delta, color=['#a65d44' if x > 0 else '#548567' for x in delta])
axes[1].axhline(0, color='#999999', linewidth=1)
for i, x in enumerate(delta):
    axes[1].text(i, x+(1 if x >= 0 else -1), f'{x:+.2f}', ha='center', va='bottom' if x >= 0 else 'top', fontsize=9)
axes[1].set_ylim(-20, 20); axes[1].set_ylabel('Energy difference from baseline (microjoules)')
axes[1].set_title('16-bit rank3: 1.79% RF + fixed-overhead gain')
for ax in axes:
    ax.spines[['top', 'right']].set_visible(False)
fig.text(.5, .015, 'Same total time and absolute RMS/power/battery constraints. Extra compute and finite-block headers/errors unmeasured. One fixed update, not full training, FU or Hybrid superiority.', ha='center', fontsize=7)
fig.tight_layout(rect=(0, .04, 1, 1))
for ext in ['png', 'svg']:
    fig.savefig(results/f'heartbeat_0742_paid_codec.{ext}', dpi=170)
plt.close(fig)

report = f'''ova_CK 밤샘 연구 — 07:42 heartbeat 후속 결과
실제 캡처: {now} KST

1. 16비트 계수 압축에서 작은 통신 에너지 개선을 확인
실제 PUF 궤적의 learning-only round100 업데이트를 고정했다. 공개 모델 이력의 최근3개 방향으로 각 UAV의 로컬 learning 업데이트를 분해하고, 16비트 계수와 아날로그 residual을 전달한다. 계수 scale은 모든 UAV가 이미 아는 theta99−theta98 norm으로 결정한다. 송신 residual을 복호된 계수 기준으로 계산하므로 계수 반올림은 이상적인 모델 업데이트 합계에서 상쇄된다. 무선 residual 오차까지 사라지는 것은 아니다.

방식 | 데이터+원래 overhead µJ | 계수 RF µJ | 합계 µJ | baseline 대비
원래 업데이트 | 823.615818 | 0 | 823.615818 | 기준
32비트 rank1 | 820.504947 | 14.782092 | 835.287039 | +1.4171%
32비트 rank3 | 785.624348 | 44.346275 | 829.970623 | +0.7716%
16비트 rank1 | 820.817307 | 7.391046 | 828.208353 | +0.5576%
16비트 rank3 | 786.668248 | 22.173138 | 808.841386 | −1.79385%

16비트 rank3는 추가 계수 RF 비용을 모두 포함하고도 14.774432µJ 적게 들었다. 기존 실패한 32비트 방식과 16비트 rank1도 삭제하지 않고 함께 보존했다. 각 방식 10개 head를 모두 검사했고 최저 에너지 head는 모두5였다. 계수16비트 rank3의 원격9개 UAV payload는 총432bits이고, TDMA864symbols를 원래167936symbols에서 차감했다. 같은 절대 RMS1.790535297e−6, 전체167.936ms, peak0.32W, UAV별 battery0.1J, fixed overhead, AO30/5개 시작점을 적용했다.

2. 독립 검증과 실제 의미
30개 저장 설계 전부를 최적화 코드를 호출하지 않고 원래 h,b,c 및 W/J 단위로 다시 검증했다. 실제 half 비트 복호·residual 재구성, 계수 전달률·RF 에너지, peak전력, UAV별 배터리, 전체 시간·coherence·매핑, 전체 RMS와 실제 벡터 기반 평균제곱 오차, 단조 에너지 이력을 확인했다. 모든 설계가 통과했다. 유한 반복의 실행 가능 해이며 전역 최적성이나 실제 장비 측정은 아니다.

이 1.79%는 무선 RF와 기존 고정 통신 overhead의 합계에 대한 한 라운드 결과다. 추가 projection·basis생성·메모리·계수 복호·헤더·유한 블록 오류 비용은 미측정이다. 네트워크 전체의 추가 비용이14.774µJ 미만이어야 순이득이 남는다. 각 UAV의 projection/residual만 세어도 합계 약2457720 FLOPs이며 이 작업만으로 계산한 허용 추가 비용은 약6.01pJ/FLOP다. basis 생성과 수신기 비용을 넣으면 허용 여유가 더 줄어든다. 하드웨어가 이 조건을 만족한다고 단정하지 않는다.

이 라운드는 FU 요청이 없으므로 Hybrid/Mixed/분리 전송이 동일한 아날로그 합계 문제로 환원된다. 모든 비교군이 같은 공개 이력과 압축법을 사용할 수 있다. 따라서 일반 FL 전송의 작은 개선 후보이며, NOMA의 우위나 VLM 언러닝 향상을 보인 결과가 아니다. 아직 추가 GPU 학습으로 승격하지 않았다.

3. 학습 초반 업데이트로 재현 검사 시작
마지막 라운드는 learning rate가 매우 작으므로 한 라운드 절감을 일반화하지 않기 위해, 실제 진행 중인 burst Hybrid의 round30을 사전에 고정했다. 공개26·27·28·29 모델이 생기기 전에 관찰 계획을 등록했다. GPU 학습을 건드리지 않는 observer PID3608159가 공개29까지 기록하고, CPU PID3608882가 실제 round30 minibatch 업데이트를 theta29에서 재현한 뒤 동일16비트 rank1/rank3와 원래 업데이트를 비교한다.
사전에 고정된 같은 codec, 공유 scale 규칙, 전체 시간·절대 RMS0.2·전력·배터리·head검색·AO 조건을 사용한다. 이후 모델이나 다른 방법의 private gradient를 이용하지 않고 validation/test 출력으로 라운드·rank·codec을 선택하지 않는다. 요청이 없는 일반 FL 검사이며 자동 GPU 승격은 없다.
캡처 기준 observer: {snapshot['early_observer_status']['status']}, 최신 공개 라운드 {snapshot['early_observer_status']['last_round']}.
CPU replay: {snapshot['early_replay_status']['status']}.

4. 기존 학습 큐와 원래 비교 결과
현재 GPU는 예약된 burst Hybrid PID3602333 한 개이며, 캡처 기준 {len(burst_rows)}/100라운드다. Mixed와 시간 배분 최적화 분리 전송은 같은 큐에서 순차 실행한다. burst는 FU 요청 라운드의 RMS 비율을0.6으로 바꾸고 나머지는0.2를 유지하는 별도 조건이다. 모든 burst 비교군에 같은 조건을 적용하며 원래0.2 결과와 섞지 않는다.
기존 원래0.2 조건의 완료된 무선 학습에서는 이번 seed의 Mixed가 통신 에너지 상한0.087379794J, 검증 loss0.96706282, forget KL0.02574322로 이 세 지표에서 가장 낮다. 6비트 Hybrid는0.189836125J, 8비트 Hybrid는0.213161276J다. MIA까지 모두 지배하거나 여러 seed에서 우월함을 뜻하지 않는다. ideal 에너지0과 실패·미완료 방법은 에너지 우위 순위에서 제외했다.
직전 완료한 이상적 링크의 retain-only 대조군과 coupled FU 비교에서는 forget KL 차이가 작고 paired 조건부 샘플 구간이0을 포함했다. 명시적 FU의 추가 이득·삭제 보장은 아직 입증되지 않았다. 기존 최종 캐시/품질 결과는 재계산하지 않았고 manuscript·원본 결과·현재 GPU/기존 큐 소스 해시는 보존했다.

오전9시 이후 첫 heartbeat에는 최선 후보/한계/비교표를 한 번 보고한다. 정오에는 결과와 실행 중인 계산을 보고하고 heartbeat를 일시중지한다. 정오 이후 새 장시간 실험은 추가하지 않는다.
증거: heartbeat_0742_results.zip 안의 snapshot, v15전체결과/30개설계/물리검증/codec원본, v13비용포함음성대조군, v17사전계획/공개상태/현재큐와 소스해시.
'''
(results/'heartbeat_0742_report_ko.txt').write_text(report)

paths = [results/'heartbeat_0742_snapshot.json', results/'heartbeat_0742_report_ko.txt',
         results/'heartbeat_0742_paid_codec.png', results/'heartbeat_0742_paid_codec.svg',
         tools/'campaign_plan.txt', tools/'campaign_state.json', tools/'capture_0742.py',
         root/'results/energy_quality_v1/energy_quality_report.json', suite/'suite_metadata.json',
         results/'burst_joint_v3/status.json', results/'burst_joint_v3/plan.json', results/'burst_joint_v3/config.json', burst_path,
         results/'queue_recovery_v1/status.json']
for name in ['public_history_fp16_v15', 'public_history_early_v17']:
    for parent in [results, tools]:
        paths.extend(p for p in (parent/name).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.tmp')
paths.extend(p for p in (results/'public_history_v13/radio').rglob('*') if p.is_file() and p.suffix != '.tmp')
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
                files={n: dict(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data)) for n, data in contents.items()})
buffer = io.BytesIO()
with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
    for name, data in contents.items():
        archive.writestr(name, data)
    archive.writestr('capture_manifest.json', json.dumps(manifest, indent=2)+'\n')
print(base64.b64encode(buffer.getvalue()).decode())
