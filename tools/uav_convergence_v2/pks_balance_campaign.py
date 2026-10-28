"""Run the predeclared P-KS repair candidates under one GPU reservation."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
import pks_balance_v1 as study

LOCK=study.ROOT/'results/overnight_energy_v2/association_objective_gpu.lock'


def child(a):
 assert int(os.environ['OVA_PKS_PARENT'])==os.getppid()
 original=fcntl.flock
 def lock(handle,mode):
  if getattr(handle,'name',None)==str(LOCK):assert mode==fcntl.LOCK_EX;return
  return original(handle,mode)
 fcntl.flock=lock
 if a.phase=='baseline':study.baseline(a.variant)
 else:study.continued(a.variant,a.dataset)


def run():
 study.register();study.unit()
 job=study.OUT/('campaign_'+str(int(time.time()))+'.json')
 variants=[v for v in study.VARIANTS if v!='control']
 study.write(job,dict(variants=variants,source=str(Path(__file__).relative_to(study.ROOT)),sha256=study.sha(Path(__file__)),concurrency=3))
 def launch(variant,phase,ds=None):
  path=study.OUT/'logs'/f'{variant}_{phase}_{ds or "both"}.log';path.parent.mkdir(exist_ok=True)
  command=[sys.executable,str(Path(__file__)),'child','--variant',variant,'--phase',phase]
  if ds:command+=['--dataset',ds]
  with path.open('a') as log:
   result=subprocess.run(command,cwd=study.ROOT,env=dict(os.environ,OVA_PKS_PARENT=str(os.getpid())),stdout=log,stderr=subprocess.STDOUT)
  row=dict(variant=variant,phase=phase,dataset=ds,returncode=result.returncode);print(row,flush=True);return row
 def pipeline(v):
  rows=[]
  if 'reuse' not in study.VARIANTS[v]:
   rows.append(launch(v,'baseline'))
   if rows[-1]['returncode']:return rows
  for ds in study.q.infra.DATASETS:rows.append(launch(v,'continue',ds))
  return rows
 with LOCK.open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  control=launch('control','baseline');assert control['returncode']==0
  assert study.read(study.OUT/'control_check.json')['passed']
  rows=[control]
  with ThreadPoolExecutor(max_workers=3) as pool:
   for future in as_completed([pool.submit(pipeline,v) for v in variants]):rows+=future.result()
 study.write(job.with_name(job.stem+'_finished.json'),dict(rows=rows,all_completed=all(r['returncode']==0 for r in rows)))
 assert all(r['returncode']==0 for r in rows),rows


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['run','child']);p.add_argument('--variant',choices=study.VARIANTS);p.add_argument('--phase',choices=['baseline','continue']);p.add_argument('--dataset',choices=study.q.infra.DATASETS);a=p.parse_args()
 if a.action=='child':child(a)
 else:run()
