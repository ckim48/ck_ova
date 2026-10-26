from pathlib import Path
import os,json,time,signal,fcntl,hashlib
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'results/overnight_energy_v2/iq_focus_20261023'
CAM=ROOT/'results/overnight_energy_v2/iq_parallel_v1_1'
PID=158721;START='882231362'
def write(name,obj):
 p=OUT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(obj,indent=2));t.replace(p)
def identity():
 try:
  p=Path('/proc')/str(PID);s=(p/'stat').read_text().split(') ',1)[1].split();cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()
  return dict(pid=PID,start=s[19],state=s[0],cmd=cmd)
 except FileNotFoundError:return None
def matches(x):return x is not None and x['start']==START and 'iq_parallel_v1_1/run.py' in x['cmd']
lock=(OUT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (OUT/'finished.json').exists()
write('status.json',dict(unix=time.time(),stage='waiting_iq_project_final_result',target=identity()))
while True:
 ident=identity()
 if not matches(ident):
  write('finished.json',dict(unix=time.time(),stage='target_ended_or_changed_without_controller_signal',observed=ident));break
 f=CAM/'iq_project/manifest.json'
 try:v=json.loads(f.read_text()) if f.exists() else None
 except (ValueError,FileNotFoundError):v=None
 ready=v and v.get('status')=='completed_200FL_and_requests' and v.get('completed_FL_rounds')==200 and v.get('requests',{}).get('procedure_completed')==45 and 'final_joint_quality_pass' in v
 if ready:
  assert (CAM/'iq_project/mia.json').exists() and (CAM/'iq_project/request_records.json').exists()
  ident=identity();assert matches(ident)
  controls={a:dict(started=(CAM/a/'started.json').exists(),completed=(CAM/a/'manifest.json').exists()) for a in ['packed_sum','separate_packed_project']}
  record=dict(unix=time.time(),stage='user_requested_stop_after_proposed_candidate',target=ident,signal='SIGTERM',completed_candidate='iq_project',manifest_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),cancelled_controls=controls,reason='User requested freezing existing comparison algorithms and focusing exclusively on proposed-method improvement; all completed and failed results preserved.')
  write('stop_decision.json',record);os.kill(PID,signal.SIGTERM)
  for _ in range(30):
   if not matches(identity()):break
   time.sleep(.5)
  record['target_after_signal']=identity();record['stage']='stopped_remaining_controls' if not matches(identity()) else 'stop_signal_sent_target_still_present'
  write('finished.json',record);break
 time.sleep(1)
