"""Recover missing original comparisons without interrupting active q6 work."""
import argparse
from datetime import datetime,timedelta,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(pid):
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()
        return None if fields[0]=='Z' else fields[19]
    except FileNotFoundError:
        return None


def wait_for_pid(pid):
    initial=identity(pid)
    while initial is not None and identity(pid)==initial:
        time.sleep(30)


def plan(root,wait_pid,waiter_pid,burst_pid):
    suite=root/'results/comparisons_seed0_solver_v2'
    expected=json.loads((suite/'suite_metadata.json').read_text())['source_sha256']
    if any(digest(root/n)!=h for n,h in expected.items()):raise ValueError('Original training sources changed.')
    return {'variant':'queue_recovery_v1','source_sha256':expected,
        'config':json.loads((suite/'config.json').read_text()),
        'wait_for_active_q6_and_initial_audit_pid':wait_pid,'hold_idle_retain_waiter_pid':waiter_pid,
        'existing_burst_campaign_pid':burst_pid,'no_new_training_after_kst':'2026-10-21T12:00:00+09:00',
        'essential_jobs':[['retrain_reference','ideal','retrain_reference_ideal'],
                          ['coupled','ideal','coupled_ideal'],
                          ['puf_regular_adapted','hybrid','puf_regular_adapted_hybrid']],
        'deferred_job':['uncoupled','hybrid','uncoupled_hybrid_restart_v1'],
        'reason':'Original queue and uncoupled child disappeared after round57 with no Python traceback/final manifest update. No cause is established. Preserve interrupted output; complete reference and missing comparisons after active q6/audit. Defer the fresh uncoupled restart until the already-approved improved-method campaign finishes.',
        'integrity':'Original training files unchanged. No overwriting or resume from incomplete state. Fresh uncoupled restart uses a separate output. Reuse unchanged numeric evaluator and verified prediction caches. No concurrent GPU process. Both idle downstream controllers are temporarily held to prevent a scheduling race, then released after essential recovery.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,required=True)
    parser.add_argument('--wait-pid',type=int,required=True)
    parser.add_argument('--waiter-pid',type=int,required=True)
    parser.add_argument('--burst-pid',type=int,required=True)
    parser.add_argument('--plan-only',action='store_true')
    args=parser.parse_args();root=args.project.resolve();suite=root/'results/comparisons_seed0_solver_v2'
    output=root/'results/overnight_energy_v2/queue_recovery_v1';output.mkdir(parents=True,exist_ok=True)
    script_dir=Path(__file__).resolve().parent
    recipe=plan(root,args.wait_pid,args.waiter_pid,args.burst_pid)
    recipe['recovery_source_sha256']={p.name:digest(p) for p in [Path(__file__),script_dir/'audit_recovered.py',script_dir/'release_waiter.py']}
    path=output/'plan.json'
    if path.exists() and json.loads(path.read_text())!=recipe:raise ValueError('Different recovery plan exists.')
    path.write_text(json.dumps(recipe,indent=2)+'\n')
    if args.plan_only:print(json.dumps(recipe,indent=2));return
    held_identities={pid:identity(pid) for pid in [args.waiter_pid,args.burst_pid]};outcomes={};released=False
    deadline=datetime(2026,10,21,12,tzinfo=timezone(timedelta(hours=9)))
    env=dict(os.environ,OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    def status(value,**extra):
        p=output/'status.json';temp=p.with_suffix('.partial.json')
        temp.write_text(json.dumps(dict(status=value,pid=os.getpid(),at_kst=datetime.now().isoformat(),outcomes=outcomes,**extra),indent=2)+'\n');temp.replace(p)
    def release_waiter():
        nonlocal released
        for pid,expected in held_identities.items():
            if expected is not None and identity(pid)==expected:
                try:os.kill(pid,signal.SIGCONT)
                except ProcessLookupError:pass
        released=True
    def assert_sources():
        if any(digest(root/n)!=h for n,h in recipe['source_sha256'].items()):raise ValueError('Frozen training source changed.')
    def gpu_commands():
        result=[]
        for p in Path('/proc').glob('[0-9]*/cmdline'):
            try:command=p.read_bytes().replace(b'\0',b' ').decode(errors='replace')
            except (OSError,ProcessLookupError):continue
            if str(root)+'/' in command and any(n in command for n in ['run_clip_federated.py','energy_quality.py','run_variant.py','audit_variants.py','audit_recovered.py']):result.append(p.parent.name)
        return result
    def wait_gpu_free():
        while True:
            active=gpu_commands()
            if not active:return
            status('waiting_for_existing_gpu_process',active_pids=active);time.sleep(30)
    def complete(directory,algorithm,transport):
        manifest=directory/'manifest.json';rounds=directory/'rounds.jsonl'
        if not manifest.exists() or not rounds.exists():return False
        m=json.loads(manifest.read_text());records=[json.loads(s) for s in rounds.read_text().splitlines() if s.strip()]
        return (m['status']=='completed' and m['algorithm']==algorithm and m['transport']==transport
            and m['source_sha256']==recipe['source_sha256'] and m['config']==recipe['config']
            and [x['round'] for x in records]==list(range(1,recipe['config']['rounds']+1))
            and digest(directory/'schedule.json')==digest(suite/'coupled_hybrid/schedule.json'))
    def train(job):
        algorithm,transport,name=job;directory=suite/name
        wait_gpu_free();assert_sources()
        if datetime.now(timezone(timedelta(hours=9)))>=deadline:
            outcomes[name]='not_started_noon_cutoff';return False
        if directory.exists():
            outcomes[name]='completed_existing' if complete(directory,algorithm,transport) else 'existing_incomplete_not_overwritten'
            return True
        directory.mkdir();status('training_missing_comparison',method=name)
        with (directory/'console.log').open('x') as log:
            run=subprocess.run([sys.executable,str(root/'run_clip_federated.py'),'--device','cuda',
                '--config',str(suite/'config.json'),'--algorithm',algorithm,'--transport',transport,
                '--output',str(directory)],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
        outcomes[name]='completed' if run.returncode==0 and complete(directory,algorithm,transport) else 'failed_or_invalid_exit_'+str(run.returncode)
        status('comparison_finished',method=name);return True
    def audit(include_restart=False):
        wait_gpu_free();assert_sources();status('evaluating_with_final_reference')
        command=[sys.executable,str(script_dir/'audit_recovered.py'),'--project',str(root)]
        if include_restart:command.append('--include-restart')
        with (output/'quality.log').open('a') as log:
            run=subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
        if run.returncode:raise RuntimeError('Recovered quality evaluation failed: '+str(run.returncode))
        subprocess.run([sys.executable,str(root/'run_comparisons.py'),'--summary-only'],cwd=root,env=env,check=True)
    try:
        with (suite/'queue.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            status('waiting_for_active_q6_and_initial_audit',wait_pid=args.wait_pid)
            wait_for_pid(args.wait_pid)
            for job in recipe['essential_jobs']:
                if not train(job):break
            audit()
            release_waiter()
            status('essential_recovery_done_waiting_existing_burst',burst_pid=args.burst_pid)
            wait_for_pid(args.burst_pid)
            if train(recipe['deferred_job']):audit(include_restart=True)
            status('completed',downstream_waiter_released=True)
    except Exception as error:
        status('recovery_failed',error=str(error));raise
    finally:
        if not released:release_waiter()


if __name__=='__main__':
    main()
