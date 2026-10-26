"""Wait for the existing serial experiment, then resume two existing checkpoints."""
from pathlib import Path
import copy,hashlib,json,os,subprocess,time,sys
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
OLD=ROOT/'results/overnight_energy_v2/deletion_window_screen_v1'
OUT=ROOT/'results/overnight_energy_v2/energy_to_target_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def status(**k):
 t=OUT/'queue_status.tmp';t.write_text(json.dumps(dict(updated_unix=time.time(),**k),indent=2));t.replace(OUT/'queue_status.json')

def main():
 import fcntl
 OUT.mkdir(exist_ok=False)
 lock=(OUT/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 while True:
  q=json.loads((OLD/'queue_status.json').read_text())
  if q['status']=='screen_completed_no_automatic_extension':break
  if q['status'].startswith('blocked'):status(status='blocked_prior_queue',detail=q);return
  status(status='waiting_for_existing_serial_queue',prior_status=q['status'],prior_method=q.get('method'));time.sleep(30)
 methods=['proposed','aircomp_only']
 for method in methods:
  assert json.loads((OLD/method/'manifest.json').read_text())['status']=='screen_completed'
 old=json.loads((OLD/'plan.json').read_text());first=json.loads((OLD/'records.json').read_text())[0]
 historical=json.loads((ROOT/'results/overnight_energy_v2/burst_joint_v3/hybrid/schedule.json').read_text())
 records=[]
 for original in historical[65:80]:
  rec=copy.deepcopy(original)
  for u in range(10):
   # Restore deletions from the historical future: the new scenario has one burst at 61 only.
   future=set().union(*(set(x['forget'][u]) for x in historical[61:rec['round']]))
   retained=(set(rec['retained'][u])|future)-set(first['forget'][u])
   rec['retained'][u]=sorted(retained);rec['forget'][u]=[]
   rec['batches'][u]=np.random.default_rng(rec['batch_seeds'][u]).choice(rec['retained'][u],min(old['config']['batch_size'],len(retained)),replace=False).astype(int).tolist()
   assert not set(rec['batches'][u]).intersection(first['forget'][u])
  rec['deleted']=list(first['deleted']);records.append(rec)
 assert [r['round'] for r in records]==list(range(66,81))
 (OUT/'records.json').write_text(json.dumps(records))
 old['methods']=methods;old['records_sha256']=sha(OUT/'records.json')
 old['prefix_folders']={m:str((OLD/m).relative_to(ROOT)) for m in methods}
 old['warm_checkpoints']={m:str((OLD/m/'checkpoints/round_065.pt').relative_to(ROOT)) for m in methods}
 old['warm_checkpoint_sha256']={m:sha(ROOT/p) for m,p in old['warm_checkpoints'].items()}
 old['prefix_metrics_sha256']={m:sha(OLD/m/'rounds.jsonl') for m in methods}
 old['runner_sha256']={p.name:sha(p) for p in HERE.glob('*.py')}
 old['scope']='Twenty post-request rounds (61-80), including the reused five-round screen. Common historical round-60 start; one K=5 request burst at 61; subsequent historical deletion events disabled consistently for both methods. Costs exclude the common prehistory. Utility comparisons do not establish equal forgetting quality.'
 old['relative_loss_improvements']=[0,.001,.002,.005,.01,.02,.05]
 old['relative_accuracy_improvements']=[0,.001,.002,.005,.01,.02,.05]
 old['fixed_forget_probe_ids']=np.random.default_rng(61).permutation(sorted(set().union(*map(set,first['forget']))))[:200].astype(int).tolist()
 old['plateau_criterion']='Across ten consecutive observed rounds, absolute difference of the two five-round mean losses <=0.001, loss range <=0.005, and mean accuracy difference <=0.002; descriptive plateau only.'
 old['assumed_aircomp_definition']='Learning and unlearning superimposed in a single aggregate (the existing aircomp_only implementation); separate two-phase AirComp is not represented.'
 old.pop('warm_checkpoint',None)
 (OUT/'plan.json').write_text(json.dumps(old,indent=2)+'\n')
 env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
 for method in methods:
  while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
   status(status='waiting_for_gpu',method=method);time.sleep(30)
  with (OUT/f'{method}.log').open('w') as log:
   proc=subprocess.Popen([sys.executable,'-u',str(HERE/'run_continuation.py'),'--plan',str(OUT/'plan.json'),'--method',method],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
   status(status='running',method=method,pid=proc.pid);rc=proc.wait()
  with (OUT/'report.log').open('a') as log:subprocess.run([sys.executable,str(HERE/'report_targets.py'),str(OUT)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
  print(json.dumps(dict(method=method,returncode=rc)),flush=True)
 status(status='completed_budget_no_automatic_extension')

if __name__=='__main__':main()
