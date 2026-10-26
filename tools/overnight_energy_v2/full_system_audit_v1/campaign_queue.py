"""Serial GPU work; no method/result is silently substituted for a source gap."""
from pathlib import Path
import subprocess,json,os,time,hashlib,fcntl
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'results/overnight_energy_v2/full_system_baselines_v1'


def write(name,value):
    p=OUT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(value,indent=2)+'\n');t.replace(p)


def main():
    claim=(OUT/'queue.lock').open('a');fcntl.flock(claim,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'queue_started.json').exists(),'Duplicate queue'
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')}
    write('queue_started.json',dict(pid=os.getpid(),unix=time.time(),source_hashes=hashes))
    steps=[('fedwiper_payload_audit',[str(HERE/'fedwiper_payload_audit.py')],False),
           ('fedu_hvp_probe',[str(HERE/'fedu_probe.py')],False),
           ('fedmemprune_200',['tools/overnight_energy_v2/full_system_baselines_v1/run_memprune.py','--stop','200','--resume'],True),
           ('matched_reference',[str(HERE/'matched_reference.py')],True)]
    for stage,args,required in steps:
        write('queue_status.json',dict(stage=stage,status='running',unix=time.time(),pid=os.getpid()))
        with (OUT/(stage+'.log')).open('ab') as log:
            process=subprocess.Popen(['.venv/bin/python','-u',*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
            write('queue_status.json',dict(stage=stage,status='running',unix=time.time(),pid=os.getpid(),child_pid=process.pid))
            code=process.wait()
        write(stage+'_exit.json',dict(exit_code=code,unix=time.time()))
        if code and required:
            write('queue_status.json',dict(stage=stage,status='failed',exit_code=code,unix=time.time()));return
        if stage=='fedmemprune_200':
            end=OUT/'fedmemprune_rho040/finished.json'
            if not end.exists() or json.loads(end.read_text()).get('rounds')!=200:
                write('queue_status.json',dict(stage=stage,status='incomplete',reason='Full candidate did not finish; no unnecessary matched reference launched',unix=time.time()));return
    write('queue_status.json',dict(stage='available_work_completed',status='completed',unix=time.time(),
        pending=['FedU scalar IAF objective implementation fidelity','FedWiper retrieval/packetization reproduction'],full_four_method_comparison_complete=False))


if __name__=='__main__':main()
