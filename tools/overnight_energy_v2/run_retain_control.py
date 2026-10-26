"""Serialize an empirical no-unlearning-update control after the current audit."""
import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def pid_identity(pid):
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()
        return None if fields[0]=='Z' else fields[19]
    except FileNotFoundError:
        return None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,required=True)
    parser.add_argument('--wait-pid',type=int)
    parser.add_argument('--plan-only',action='store_true')
    parser.add_argument('--summary-only',action='store_true')
    args=parser.parse_args();root=args.project.resolve()
    sys.path[:0]=[str(root),str(root/'tools/energy_quality_v1')]
    import energy_quality as eq
    suite=root/'results/comparisons_seed0_solver_v2'
    output=root/'results/energy_quality_v1'
    status_path=root/'results/overnight_energy_v2/retain_control_status.json'
    plan={"role":"retained-data-only algorithmic control, ideal channel; never an energy competitor",
        "algorithm":"retain_only","transport":"ideal","seed":0,
        "unchanged_protocol":str(suite/'config.json'),
        "reason":"Deleted cohorts mostly had one prior SGD exposure. Compare explicit forgetting updates to removal from future minibatches, rather than treating chance-level membership attack as proof.",
        "wait_pid":args.wait_pid,"no_new_training_after_kst":"2026-10-21T12:00:00+09:00",
        "wrapper_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (status_path.parent/'retain_control_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    if args.plan_only:
        print(json.dumps(plan,indent=2));return
    eq.METHODS=eq.METHODS+('retain_only_ideal',)
    evaluate_args=argparse.Namespace(project=root,suite=suite,output=output,device='cuda',
        cpu_threads=4,samples=1000,batch_size=32,validation_loss_max=None,forget_kl_max=None,attack_gap_max=None)
    if args.summary_only:
        eq.summarize(evaluate_args);return
    def status(value,**extra):
        status_path.write_text(json.dumps(dict(status=value,pid=os.getpid(),**extra),indent=2)+'\n')
    status('waiting_for_existing_quality_workflow',wait_pid=args.wait_pid)
    if args.wait_pid:
        identity=pid_identity(args.wait_pid)
        while identity is not None and pid_identity(args.wait_pid)==identity:
            time.sleep(30)
    # Also protect against an orphaned training/evaluation child.
    while True:
        processes=subprocess.check_output(['ps','-u',str(os.getuid()),'-o','args='],text=True).splitlines()
        active=[x for x in processes if '/ova_CK/' in x and
                ('run_clip_federated.py' in x or ('energy_quality.py' in x and '--summary-only' not in x))]
        if not active:break
        status('waiting_for_other_project_gpu_job',active_commands=active)
        time.sleep(30)
    now=datetime.now(timezone(timedelta(hours=9)))
    deadline=datetime(2026,10,21,12,tzinfo=timezone(timedelta(hours=9)))
    if now>=deadline:
        status('not_started_overnight_window_ended');return
    expected=json.loads((suite/'suite_metadata.json').read_text())['source_sha256']
    if any(eq.digest(root/name)!=sha for name,sha in expected.items()):
        status('not_started_training_source_changed');raise RuntimeError('Training source hashes changed.')
    destination=suite/'retain_only_ideal'
    training_exit_code=0
    if destination.exists():
        valid,reason=eq.validate_completed(destination,suite,json.loads((suite/'config.json').read_text()),suite/'coupled_hybrid')
        if valid is None:
            status('existing_incomplete_not_overwritten',reason=reason);return
    else:
        status('training')
        destination.mkdir()
        with (destination/'console.log').open('x') as log:
            result=subprocess.run([sys.executable,str(root/'run_clip_federated.py'),'--device','cuda',
                '--config',str(suite/'config.json'),'--algorithm','retain_only','--transport','ideal',
                '--output',str(destination)],cwd=root,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            training_exit_code=result.returncode
            status('training_failed',exit_code=result.returncode)
    status('evaluating')
    try:
        eq.collect(evaluate_args)
    except Exception as error:
        status('audit_failed',error=str(error),control_training_exit_code=training_exit_code)
        raise
    status('completed' if training_exit_code==0 else 'control_failed_audit_completed',
           control_training_exit_code=training_exit_code)


if __name__=='__main__':
    main()
