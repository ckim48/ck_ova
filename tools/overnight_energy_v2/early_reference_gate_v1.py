"""Insert cached final-reference evaluation after the current GPU training."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def identity(pid):
    try:
        f=Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()
        return None if f[0]=='Z' else f[19]
    except FileNotFoundError:return None


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,required=True)
    args=parser.parse_args();root=args.project.resolve()
    out=root/'results/overnight_energy_v2/early_reference_gate_v1'
    plan=json.loads((out/'plan.json').read_text());parent=plan['parent_pid']
    def status(value,**extra):
        temp=out/'status.partial.json';temp.write_text(json.dumps(dict(status=value,pid=os.getpid(),at_kst=datetime.now().isoformat(timespec='seconds'),**extra),indent=2)+'\n');temp.replace(out/'status.json')
    def resume():
        if identity(parent)==plan['parent_identity']:
            try:os.kill(parent,signal.SIGCONT)
            except ProcessLookupError:pass
    try:
        status('waiting_for_launcher_activation')
        for _ in range(60):
            if (out/'activation.json').exists():break
            time.sleep(.5)
        else:raise RuntimeError('Launcher did not activate; leaving original queue running.')
        assert identity(parent)==plan['parent_identity']
        assert Path(f'/proc/{parent}/stat').read_text().split(') ',1)[1].split()[0]=='T'
        status('waiting_for_current_gpu_child',child_pid=plan['child_pid'])
        while identity(plan['child_pid'])==plan['child_identity']:time.sleep(3)
        suite=root/'results/comparisons_seed0_solver_v2'
        for name in ['retrain_reference_ideal','coupled_ideal']:
            m=json.loads((suite/name/'manifest.json').read_text())
            assert m['status']=='completed',name
        for name,expected in plan['source_sha256'].items():assert digest(root/name)==expected,name
        # With the only launcher held, no training or evaluation may remain.
        for p in Path('/proc').glob('[0-9]*/cmdline'):
            try:cmd=p.read_bytes().replace(b'\0',b' ').decode(errors='replace')
            except (OSError,ProcessLookupError):continue
            if str(root)+'/' in cmd and any(x in cmd for x in ['run_clip_federated.py','run_variant.py','energy_quality.py','audit_recovered.py','audit_variants.py']):
                raise RuntimeError('Unexpected active project computation: '+p.parent.name)
        status('evaluating_final_reference_with_existing_prediction_caches')
        env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
        with (out/'quality.log').open('x') as log:
            run=subprocess.run([sys.executable,str(root/'tools/overnight_energy_v2/queue_recovery_v1/audit_recovered.py'),'--project',str(root)],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=900)
        if run.returncode:raise RuntimeError('Early audit exited '+str(run.returncode))
        report=root/'results/energy_quality_v1/energy_quality_report.json'
        assert json.loads(report.read_text())['reference_available']
        (out/'final_reference_report.json').write_bytes(report.read_bytes())
        status('completed',report=str(out/'final_reference_report.json'),original_parent_resumed=True)
    except Exception as e:
        status('failed_original_queue_will_resume',error=str(e));raise
    finally:resume()


if __name__=='__main__':main()
