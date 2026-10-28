"""Run frozen original or corrected learning arms, preserving every attempt."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
import learning_quality_v3 as study
import learning_quality_campaign as original


def main(a):
 if a.corrected:
  import learning_selection_fix_v2
  learning_selection_fix_v2.install()
 if a.action=='child':
  original.child(a.variant,a.case);return
 study.register()
 assert study.read(study.ROOT/'results/learning_quality_v3_20261028/control_check.json')['passed']
 cases=[c['id'] for c in study.cases() if c['id']!='baseline'] if a.all_sweeps else a.cases
 items=[(v,c) for v in a.variants for c in cases];stamp='campaign_'+str(int(time.time()))
 study.write(study.OUT/(stamp+'.json'),dict(items=items,source=str(Path(__file__).relative_to(study.ROOT)),sha256=study.sha(Path(__file__)),concurrency=3,corrected=a.corrected))
 def launch(v,c):
  logfile=study.OUT/'logs'/f'{v}_{c}.log';logfile.parent.mkdir(exist_ok=True)
  command=[sys.executable,str(Path(__file__)),'child','--variant',v,'--case',c]
  if a.corrected:command.append('--corrected')
  with logfile.open('a') as out:
   result=subprocess.run(command,cwd=study.ROOT,env=dict(os.environ,OVA_LEARNING_PARENT=str(os.getpid())),stdout=out,stderr=subprocess.STDOUT)
  return dict(variant=v,case=c,returncode=result.returncode)
 rows=[]
 with original.LOCK.open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  with ThreadPoolExecutor(max_workers=3) as pool:
   for future in as_completed([pool.submit(launch,v,c) for v,c in items]):
    row=future.result();rows.append(row);print(row,flush=True)
 study.write(study.OUT/(stamp+'_finished.json'),dict(rows=rows,all_completed=all(x['returncode']==0 for x in rows)))
 assert all(x['returncode']==0 for x in rows),rows


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['run','child']);p.add_argument('--corrected',action='store_true');p.add_argument('--all-sweeps',action='store_true');p.add_argument('--variants',nargs='+');p.add_argument('--cases',nargs='+',default=['baseline']);p.add_argument('--variant');p.add_argument('--case');main(p.parse_args())
