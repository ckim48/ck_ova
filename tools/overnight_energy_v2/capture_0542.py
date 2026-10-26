"""Capture completed CPU screens without changing queued training sources."""
import base64
import hashlib
import io
import json
from datetime import datetime
from pathlib import Path
import subprocess
import zipfile

root = Path('/home/wnlab/Documents/Codex/2026-10-20/files-pasted-by-the-user-standalone/outputs/ova_CK')
tools = root / 'tools/overnight_energy_v2'
results = root / 'results/overnight_energy_v2'
def read(path):
    return json.loads(path.read_text())
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')

names = ['history_weighted_v6', 'history_guard_v7', 'global_fusion_v8']
assert all(read(results / n / 'status.json')['status'].startswith('completed') for n in names)
assert read(results / 'history_guard_v7/radio/status.json')['cases'] == 18
expected = {
    'run_clip_federated.py':'0d1dfbc5803d5ca37fc430ed0d5d41214d232b94c52aea15f1ec7074474e1a61',
    'clip_model.py':'715c37f57eb89d9a5c53e7fedae45b672f484961c9677b76214f0bcdcf320334',
    'transport.py':'585ca82d9631a36c4807255166a9cf50d4bf59237cb28de784fe7f65a78df91c',
    'wireless.py':'0d981861984500a14890ff5391d1d826068fb879cedf76039f4403b8729a8606',
}
assert all(sha(root / f) == h for f, h in expected.items())
now = datetime.now().isoformat(timespec='seconds')
puf_dir = root / 'results/comparisons_seed0_solver_v2/puf_regular_adapted_hybrid'
puf_rows = [json.loads(x) for x in (puf_dir/'rounds.jsonl').read_text().splitlines() if x.strip()]
snapshot = {
    'captured_at_kst':now,
    'heartbeat_at_kst':'2026-10-21T05:42:55',
    'core_sha256':expected,
    'new_cpu_status':{n:read(results/n/'status.json') for n in names},
    'new_cpu_results':read(results/'global_fusion_v8/results.json'),
    'guard_radio_status':read(results/'history_guard_v7/radio/status.json'),
    'guard_radio_summary':[{k:v for k,v in x.items() if k!='heads'} for x in read(results/'history_guard_v7/radio/results.json')],
    'puf_completed_rounds':len(puf_rows),
    'puf_last_round':puf_rows[-1],
    'process_snapshot':subprocess.run(['ps','-p','3546625,3546626,3520581,3534100,3575038','-o','pid,ppid,stat,etime,args'],capture_output=True,text=True).stdout,
    'gpu_compute_processes':subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],capture_output=True,text=True).stdout,
    'decision':'No GPU promotion from v6/v7/v8: none jointly improves validation loss and retraining-reference forget KL in the fixed one-step screen. Preserve all negative results. Leave existing single-GPU queue intact.',
    'limits':['One legacy state, fixed200 validation/200forget; not a full trajectory or statistical comparison.',
              'Final-reference cached probabilities GPU; new one-step inference CPU.',
              'v7 radio cases fail the task screen and cannot be ranked as quality-qualified energy winners.',
              'v8 changes additive fusion; any future architecture claim needs strong separated dual-analog and phase baselines.',
              'No test data select coefficients. Local inference costs uncalibrated. No deletion certification.']
}
write(results/'heartbeat_0542_snapshot.json',snapshot)
state_path=tools/'campaign_state.json'
state=read(state_path)
state.update(last_heartbeat_research_kst=now,
    notes='Early reference audit finished and recovery resumed. PUF3575038 is the single active GPU job; original retain3520581 and burst3534100 remain intentionally held until recovery releases them. v6/v7/v8 CPU screens complete; no joint task-quality improvement and no GPU promotion. Do not duplicate completed CPU or cached reference inference.',
    history_guard_v7_radio_completed=True, global_fusion_v8_cpu_completed=True,
    global_fusion_v8_rejected_for_gpu=True, global_fusion_v8_gpu_promoted=False,
    last_cpu_screen_snapshot='results/overnight_energy_v2/heartbeat_0542_snapshot.json')
write(state_path,state)
with (tools/'campaign_plan.txt').open('a') as f:
    f.write('\nHeartbeat05:42 completion recorded '+now+' KST\n'
        'v7 radio finished all18 prespecified cases. At0.2 only phase fractions0.125/0.25 found feasible designs(56.248/68.401mJ); H/M no feasible design found. At0.6 H38.400mJ, Mixed10.995mJ, bestphase0.125 39.875mJ. These are RADIO STRESS ONLY because v7 failed the global task screen. Do not use failures or weaker task quality to claim an energy win.\n'
        'v8 all3 ideal-fusion variants completed: localhistory val1.01357585/forgetKL0.02511637, globalhistory1.01373722/0.02510962, globaloriginalU1.01367416/0.02505517. Original additive ideal step1.01383826/0.02498938; validation gains trade against worse forget KL. No joint improvement, no GPU promotion. Clip-then-halfspace is not an exact intersection projection. Separate streams are needed for this proposed global rule, but that alone does not justify digital NOMA: include a strong dual-analog separated comparator before any architecture claim.\n'
        'All v6/v7/v8 sources, actual update vectors, negative results and fixed protocols preserved. No test tuning, no further coefficients selected from these validation outputs. Current PUF/retain/burst queue and core hashes unchanged. Wait for full-trajectory controls before additional long candidate training.\n')

paths=[results/'heartbeat_0542_snapshot.json',tools/'campaign_state.json',tools/'campaign_plan.txt',tools/'capture_0542.py',
       root/'results/energy_quality_v1/energy_quality_report.json',root/'results/energy_quality_v1/energy_quality.csv',
       root/'results/energy_quality_v1/cohorts.json',puf_dir/'rounds.jsonl',puf_dir/'manifest.json']
for n in names:
    paths.extend(x for x in (results/n).rglob('*') if x.is_file() and '__pycache__' not in x.parts)
    paths.extend(x for x in (tools/n).rglob('*') if x.is_file() and '__pycache__' not in x.parts)
paths=list(dict.fromkeys(paths))
assert all(p.exists() for p in paths)
manifest={'captured_at_kst':now,'files':{str(p.relative_to(root)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in paths}}
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as archive:
    for p in paths:archive.write(p,str(p.relative_to(root)))
    archive.writestr('capture_manifest.json',json.dumps(manifest,indent=2)+'\n')
print(base64.b64encode(buf.getvalue()).decode())
