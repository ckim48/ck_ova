"""Serial GPU queue; every completed or failed attempt is preserved."""
import fcntl,json,os,subprocess,sys,time,zipfile
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
OUT=ROOT/'results/overnight_energy_v2/deletion_window_screen_v1'


def status(**kw):
    p=OUT/'queue_status.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(dict(updated_unix=time.time(),**kw),indent=2)+'\n');t.replace(p)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'gpu_queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=json.loads((OUT/'plan.json').read_text());env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
    checks=json.loads((ROOT/'results/overnight_energy_v2/convergence_energy_delay_v1/validation.json').read_text());assert checks['status']=='passed'
    for method in plan['methods']:
        target=OUT/method
        if target.exists():
            status(status='blocked_existing_output',method=method);raise FileExistsError(target)
        while True:
            gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout.strip()
            if not gpu:break
            status(status='waiting_for_gpu',method=method,existing_gpu_pids=gpu);time.sleep(30)
        with (OUT/f'{method}.log').open('w') as log:
            proc=subprocess.Popen([sys.executable,'-u',str(HERE/'run_screen.py'),'--plan',str(OUT/'plan.json'),'--method',method],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            status(status='running',method=method,pid=proc.pid)
            rc=proc.wait()
        manifest=json.loads((target/'manifest.json').read_text()) if (target/'manifest.json').exists() else {'status':'failed_before_manifest'}
        with (OUT/'summary.log').open('a') as log:
            subprocess.run([sys.executable,str(HERE/'summarize.py'),str(OUT)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        print(json.dumps(dict(method=method,exit_code=rc,status=manifest['status'])),flush=True)
    status(status='screen_completed_no_automatic_extension')
    with zipfile.ZipFile(OUT/'results_package.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in OUT.rglob('*'):
            if p.is_file() and p.suffix in ('.json','.jsonl','.csv','.png','.pdf') and 'checkpoints' not in p.parts:z.write(p,str(p.relative_to(OUT)))
        for p in HERE.glob('*.py'):z.write(p,'source/'+p.name)


if __name__=='__main__':main()
