"""Run three independent immutable scenarios under one exclusive GPU-group lock.

Only loading/execution concurrency changes. Seeds, model functions, simulated
clocks and physical costs remain untouched; GPU wall time is not a metric.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
import paper_uav_batch as batch
study=batch.study
LOCK=study.ROOT/'results/overnight_energy_v2/association_objective_gpu.lock'


def child(case):
    assert int(os.environ['OVA_PAPER_GROUP_PID'])==os.getppid()
    original=fcntl.flock
    def controlled_lock(handle,mode):
        if getattr(handle,'name',None)==str(LOCK):
            assert mode==fcntl.LOCK_EX
            # The direct parent owns this same global lock throughout all children.
            return
        return original(handle,mode)
    fcntl.flock=controlled_lock
    batch.run_case(case)


def run():
    study.register()
    manifest=dict(unix=time.time(),source=str(Path(__file__).relative_to(study.ROOT)),sha256=study.sha(Path(__file__)),
        batch_source=str(Path(batch.__file__).relative_to(study.ROOT)),batch_sha256=study.sha(Path(batch.__file__)),max_concurrent_cases=3,
        reason='GPU memory preflight:346MiB used of10240MiB,8% utilization with one process. Three independent cases share one parent-held exclusive GPU lock; fresh process per case, same per-process seeds and simulated MAC/radio clocks. No GPU wall-time metric used.')
    study.write(study.OUT/'execution_parallel.json',manifest)
    pending=[]
    for c in study.cases():
        paths=[study.OUT/'cases'/c['id']/(ds.lower()+'_seed0')/m/'outcome.json' for ds in study.DATASETS for m in study.METHODS]
        if all(p.exists() for p in paths):assert all(study.read(p)['status']=='completed' for p in paths)
        else:pending.append(c['id'])
    def launch(cid):
        log=study.OUT/'case_logs'/f'{cid}.log';log.parent.mkdir(exist_ok=True)
        env=dict(os.environ,OVA_PAPER_GROUP_PID=str(os.getpid()))
        with log.open('a') as h:subprocess.run([sys.executable,str(Path(__file__)),'case','--case',cid],cwd=study.ROOT,env=env,stdout=h,stderr=subprocess.STDOUT,check=True)
        return cid
    with LOCK.open('a') as gpu:
        fcntl.flock(gpu,fcntl.LOCK_EX)
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures={pool.submit(launch,c):c for c in pending}
            for f in as_completed(futures):print(dict(completed_case=f.result()),flush=True)
    result=study.verify();study.write(study.OUT/'finished.json',dict(unix=time.time(),**result));study.state(stage='completed')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['case','run']);p.add_argument('--case');a=p.parse_args()
    child(a.case) if a.action=='case' else run()
