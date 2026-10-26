"""Run exactly two proposed pilots after the original mission queue terminates."""
from pathlib import Path
import fcntl,json,os,subprocess,time,traceback
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OUT=ROOT/'results/overnight_energy_v2/korea_joint_seed0_pilot_v2'


def write(name,obj):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(p)


def identity(pid):
    p=Path('/proc')/str(pid)
    try:return dict(pid=pid,start_ticks=p.joinpath('stat').read_text().split(') ',1)[1].split()[19],argv=p.joinpath('cmdline').read_bytes().replace(b'\0',b' ').decode())
    except FileNotFoundError:return None


def main():
    lock=(OUT/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'queue_started.json').exists(),'No automatic duplicate/retry'
    write('queue_started.json',dict(unix=time.time(),identity=identity(os.getpid())))
    protocol=json.loads((OUT/'protocol.json').read_text());dep=protocol['dependency']
    parent=ROOT/'results/overnight_energy_v2'/dep['folder']
    while not (parent/'finished.json').exists():
        now=identity(dep['pid'])
        assert now and now['start_ticks']==dep['start_ticks'] and dep['argv_suffix'] in now['argv'],'Dependency stopped; no restart'
        assert not (parent/'failure.json').exists(),'Dependency queue failed'
        write('status.json',dict(stage='waiting_for_original_mission',dependency=now,unix=time.time()))
        time.sleep(30)
    done=[];fail=[]
    for job in protocol['runs']:
        assert job['seed']==0 and job['method']=='proposed_fu_guard_cliperase'
        name=job['dataset'].lower()+'_seed0';dest=OUT/name
        assert not (dest/'started.json').exists() and not (dest/'launched.json').exists()
        with (HERE/(name+'.log')).open('a') as log:
            p=subprocess.Popen([str(ROOT/'.venv/bin/python'),str(HERE/'campaign.py'),'--dataset',job['dataset']],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONUNBUFFERED='1'))
            child=identity(p.pid)
            write(name+'/launched.json',dict(unix=time.time(),identity=child))
            write('status.json',dict(stage='running',current=name,child=child,finished=done,failures=fail,unix=time.time()))
            rc=p.wait()
        if rc or not (dest/'finished.json').exists():fail.append(dict(run=name,returncode=rc,status='failed_no_automatic_retry'))
        done.append(name)
    write('finished.json',dict(status='two_proposed_pilots_terminal',runs=done,failures=fail,unix=time.time()))
    write('status.json',dict(stage='completed',finished=done,failures=fail,unix=time.time()))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        write('failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),unix=time.time()))
        raise
