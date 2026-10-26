"""Wait for the identified existing campaign, then run exactly one IQ follow-up."""
from pathlib import Path
import os,json,time,fcntl,subprocess,sys,hashlib
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
OUT=ROOT/'results/overnight_energy_v2/iq_parallel_v1_1'
SOURCE=ROOT/'results/overnight_energy_v2/streaming_requests_v1'
def write(name,obj):
    p=OUT/name;tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(p)
def identity(pid):
    try:
        p=Path('/proc')/str(pid);s=p.joinpath('stat').read_text().split(') ',1)[1].split()
        return dict(pid=pid,start=s[19],state=s[0],cmd=p.joinpath('cmdline').read_bytes().replace(b'\0',b' ').decode())
    except FileNotFoundError:return None
def main():
    own=(OUT/'queue.lock').open('a');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'queue_registered.json').exists(),'Duplicate queue'
    spec=json.loads((OUT/'protocol.json').read_text())
    for name,digest in spec['source_hashes'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    write('queue_registered.json',dict(unix=time.time(),identity=identity(os.getpid()),predecessor=spec['predecessor'],scheduled_arms=list(spec['arms'])))
    while True:
        prior=identity(134207)
        same=prior is not None and prior['start']=='881579733'
        if same:
            assert 'streaming_requests_v1/run.py' in prior['cmd']
            write('queue_status.json',dict(unix=time.time(),stage='waiting_existing_streaming_campaign',predecessor=prior))
        elif not (SOURCE/'finished.json').exists():
            write('queue_status.json',dict(unix=time.time(),stage='blocked_predecessor_ended_without_finished_marker'))
            return
        else:break
        time.sleep(30)
    # GPU/launch locks and source hashes are checked again by run.py.
    write('queue_status.json',dict(unix=time.time(),stage='starting_IQ_campaign'))
    with (OUT/'run.log').open('ab') as log:
        child=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u',str(HERE/'run.py')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        write('queue_status.json',dict(unix=time.time(),stage='running_IQ_campaign',child=identity(child.pid)))
        rc=child.wait()
    write('queue_status.json',dict(unix=time.time(),stage='completed' if rc==0 else 'failed',returncode=rc))
if __name__=='__main__':main()
