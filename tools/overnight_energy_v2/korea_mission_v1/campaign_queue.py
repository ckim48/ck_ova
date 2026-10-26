"""Authorized mission only, after the previously registered three-seed queue."""
from pathlib import Path
import os,json,time,subprocess,fcntl,traceback
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];OUT=ROOT/'results/overnight_energy_v2/korea_mission_v1'
def write(name,d):
 p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2));tmp.replace(p)
def identity(pid):
 p=Path('/proc')/str(pid)
 try:return dict(pid=pid,start_ticks=p.joinpath('stat').read_text().split(') ',1)[1].split()[19],argv=p.joinpath('cmdline').read_bytes().replace(b'\0',b' ').decode())
 except FileNotFoundError:return None
def main():
 own=(OUT/'queue.lock').open('a');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (OUT/'queue_started.json').exists()
 write('queue_started.json',dict(unix=time.time(),identity=identity(os.getpid())))
 proto=json.loads((OUT/'protocol.json').read_text());dep=proto['dependency'];folder=ROOT/'results/overnight_energy_v2'/dep['folder']
 while not (folder/'finished.json').exists():
  now=identity(dep['pid']);valid=now and now['start_ticks']==dep['start_ticks'] and dep['argv_suffix'] in now['argv']
  if not valid or (folder/'failure.json').exists():raise RuntimeError('Three-seed dependency stopped/failed; no automatic restart')
  write('status.json',dict(stage='waiting_for_three_seed',unix=time.time(),dependency=now));time.sleep(30)
 done=[];fail=[]
 for job in proto['runs']:
  name=f"{job['dataset'].lower()}_seed{job['seed']}";dest=OUT/name;dest.mkdir(parents=True,exist_ok=True)
  assert not (dest/'started.json').exists() and not (dest/'launched.json').exists()
  log=(HERE/(name+'.log')).open('a')
  proc=subprocess.Popen([str(ROOT/'.venv/bin/python'),str(HERE/'campaign.py'),'--dataset',job['dataset'],'--seed',str(job['seed'])],cwd=str(ROOT),stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONUNBUFFERED='1'))
  write(name+'/launched.json',dict(unix=time.time(),identity=identity(proc.pid)));write('status.json',dict(stage='running',current=name,child=identity(proc.pid),finished=done,failures=fail,unix=time.time()))
  rc=proc.wait();log.close()
  if rc or not (dest/'finished.json').exists():fail.append(dict(run=name,returncode=rc,status='failed_no_automatic_retry'))
  done.append(name);write('status.json',dict(stage='between_runs',finished=done,failures=fail,unix=time.time()))
 write('finished.json',dict(status='all_registered_missions_finished_or_failed',runs=done,failures=fail,unix=time.time()));write('status.json',dict(stage='completed',finished=done,failures=fail,unix=time.time()))
if __name__=='__main__':
 try:main()
 except Exception as exc:
  write('failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),unix=time.time()));write('status.json',dict(stage='failed',error=repr(exc),unix=time.time()));raise
