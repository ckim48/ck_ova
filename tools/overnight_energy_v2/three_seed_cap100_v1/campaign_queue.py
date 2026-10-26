"""Only the four newly authorized repeats; no automatic restart or sweep."""
from pathlib import Path
import json, os, time, subprocess, fcntl, traceback
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
OUT=ROOT/'results/overnight_energy_v2/three_seed_cap100_v1'

def write(name,d):
    p=OUT/name;temp=p.with_suffix('.tmp');temp.write_text(json.dumps(d,indent=2));temp.replace(p)

def identity(pid):
    p=Path('/proc')/str(pid)
    if not p.exists():return None
    try:return dict(pid=pid,start_ticks=p.joinpath('stat').read_text().split(') ',1)[1].split()[19],argv=p.joinpath('cmdline').read_bytes().replace(b'\0',b' ').decode())
    except FileNotFoundError:return None

def main():
    own=(OUT/'queue.lock').open('a');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'queue_started.json').exists(),'Duplicate queue'
    write('queue_started.json',dict(unix=time.time(),identity=identity(os.getpid())))
    proto=json.loads((OUT/'protocol.json').read_text());dep=proto['dependency']
    dependency=ROOT/'results/overnight_energy_v2/rsicd_cap100_v1'
    while not (dependency/'finished.json').exists():
        now=identity(dep['pid'])
        valid=now and now['start_ticks']==dep['start_ticks'] and dep['argv_suffix'] in now['argv']
        if not valid or (dependency/'failure.json').exists():
            raise RuntimeError('RSICD seed0 dependency failed/stopped; no repeated experiments started.')
        write('status.json',dict(stage='waiting_for_RSICD_seed0',unix=time.time(),dependency=now))
        time.sleep(30)
    done=[]
    for job in proto['runs']:
        name=f"{job['dataset'].lower()}_seed{job['seed']}";dest=OUT/name
        assert not (dest/'started.json').exists() and not (dest/'launched.json').exists()
        log=(HERE/(name+'.log')).open('a')
        proc=subprocess.Popen([str(ROOT/'.venv/bin/python'),str(HERE/'repeat.py'),'--dataset',job['dataset'],'--seed',str(job['seed'])],cwd=str(ROOT),stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONUNBUFFERED='1'))
        write(name+'/launched.json',dict(unix=time.time(),identity=identity(proc.pid)))
        write('status.json',dict(stage='running',current=name,child=identity(proc.pid),finished=done,unix=time.time()))
        rc=proc.wait();log.close()
        if rc!=0 or not (dest/'finished.json').exists():
            raise RuntimeError(f'{name} exited {rc}; preserve output and require diagnosis, no automatic retry.')
        done.append(name)
        write('status.json',dict(stage='between_runs',finished=done,unix=time.time()))
    write('finished.json',dict(status='all_registered_repeats_finished',finished=done,unix=time.time()))
    write('status.json',dict(stage='completed',finished=done,unix=time.time()))

if __name__=='__main__':
    try:main()
    except Exception as exc:
        write('failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),unix=time.time()))
        write('status.json',dict(stage='failed',error=repr(exc),unix=time.time()))
        raise
