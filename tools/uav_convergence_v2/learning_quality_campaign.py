"""Run registered candidate diagnostics under a shared exclusive GPU lock."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import fcntl
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import learning_quality_v3 as study

LOCK=study.ROOT/'results/overnight_energy_v2/association_objective_gpu.lock'


def control():
 import paper_uav_charts as old
 rows=[]
 for ds in study.infra.DATASETS:
  new=study.OUT/'control/cases/baseline'/(ds.lower()+'_seed0')/study.PROPOSED;prior=old.source('baseline',ds,study.PROPOSED)
  assert study.read(new/'outcome.json')['status']=='completed'
  assert study.read(new/'manifest.json')['evaluation']==study.read(prior/'manifest.json')['evaluation']
  tx=[study.read(p) for p in sorted((new/'transactions').glob('*.json'))];before=[study.read(p) for p in sorted((prior/'transactions').glob('*.json'))]
  assert len(tx)==len(before)
  for a,b in zip(tx,before):
   assert a['kind']==b['kind'] and a['FU_request_ids']==b['FU_request_ids']
   for k in ['start_s','end_s']:assert math.isclose(a[k],b[k],rel_tol=0,abs_tol=1e-10)
   assert a['radio']['energy_bound_j']==b['radio']['energy_bound_j']
  rows.append(dict(dataset=ds,exact_quality=True,identical_schedule_physical_time_and_energy=True))
 result=dict(passed=True,rows=rows);study.write(study.OUT/'control_check.json',result);return result


def child(variant,case):
 assert int(os.environ['OVA_LEARNING_PARENT'])==os.getppid()
 original=fcntl.flock
 def lock(handle,mode):
  if getattr(handle,'name',None)==str(LOCK):assert mode==fcntl.LOCK_EX;return
  return original(handle,mode)
 fcntl.flock=lock;study.run(variant,case)


def run(variants,cases):
 study.register();control()
 identifier='campaign_'+str(int(time.time()));items=[(v,c) for v in variants for c in cases]
 study.write(study.OUT/(identifier+'.json'),dict(items=items,source=str(Path(__file__).relative_to(study.ROOT)),sha256=study.sha(Path(__file__)),concurrency=3))
 def launch(v,c):
  log=study.OUT/'logs'/f'{v}_{c}.log';log.parent.mkdir(exist_ok=True)
  with log.open('a') as f:
   result=subprocess.run([sys.executable,str(Path(__file__)),'child','--variant',v,'--case',c],cwd=study.ROOT,
    env=dict(os.environ,OVA_LEARNING_PARENT=str(os.getpid())),stdout=f,stderr=subprocess.STDOUT)
  return dict(variant=v,case=c,returncode=result.returncode)
 rows=[]
 with LOCK.open('a') as gpu:
  fcntl.flock(gpu,fcntl.LOCK_EX)
  with ThreadPoolExecutor(max_workers=3) as pool:
   futures=[pool.submit(launch,v,c) for v,c in items]
   for f in as_completed(futures):r=f.result();rows.append(r);print(r,flush=True)
 study.write(study.OUT/(identifier+'_finished.json'),dict(rows=rows,all_completed=all(r['returncode']==0 for r in rows)))
 assert all(r['returncode']==0 for r in rows),rows


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['control','run','child']);p.add_argument('--variants',nargs='+');p.add_argument('--cases',nargs='+',default=['baseline']);p.add_argument('--variant');p.add_argument('--case');a=p.parse_args()
 if a.action=='control':print(control())
 elif a.action=='child':child(a.variant,a.case)
 else:run(a.variants,a.cases)
