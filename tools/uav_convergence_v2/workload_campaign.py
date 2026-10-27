"""Audited zero-change control, then all declared workload conditions."""
import argparse
import collections
from concurrent.futures import ThreadPoolExecutor,as_completed
import fcntl
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import workload_sweeps_v3 as study
import paper_uav_charts as prior

LOCK=study.ROOT/'results/overnight_energy_v2/association_objective_gpu.lock'


def control_check():
    rows=[]
    for ds in study.DATASETS:
        for method in study.METHODS:
            arm=study.OUT/'cases/control'/(ds.lower()+'_seed0')/method;old=prior.source('baseline',ds,method)
            assert study.read(arm/'outcome.json')['status']=='completed'
            a=study.read(arm/'manifest.json')['evaluation'];b=study.read(old/'manifest.json')['evaluation']
            assert a==b,(ds,method,'quality mismatch')
            tx=[study.read(p) for p in sorted((arm/'transactions').glob('*.json'))]
            original=[study.read(p) for p in sorted((old/'transactions').glob('*.json'))]
            assert len(tx)==len(original)
            for x,y in zip(tx,original):
                assert x['kind']==y['kind'] and x['FU_request_ids']==y['FU_request_ids']
                for key in ['start_s','end_s']:assert math.isclose(x[key],y[key],abs_tol=1e-10)
                assert x['radio']['energy_bound_j']==y['radio']['energy_bound_j']
            for name in ['plan.json','requests.json','initial20/final.pt','feature_cache/features.pt','reference200/final.pt']:
                assert study.sha(arm.parent/name)==study.sha(old.parent/name),(ds,method,name)
            rows.append(dict(dataset=ds,method=method,final_metrics_exact=True,transaction_order_times_and_RF_identical=True,transactions=len(tx)))
    r=dict(passed=True,arms=6,rows=rows);study.write(study.OUT/'control_verification.json',r);return r


def configure_validation(case):
    study.TOTAL=case['value'] if case['axis']=='fl_rounds' else 200;study.ADAPT=study.TOTAL-20;study.SITE_FL=study.ADAPT//5
    study.LOSS=case['value'] if case['axis']=='packet_loss' else 0.
    _,base=study.lib.infra.prior.load_base()
    import event_energy
    study.PHYSICAL=event_energy.energy
    study.PHYSICAL.PARAMETERS=study.read(study.OUT/'cases'/case['id']/'settings.json')['energy_model']


def verify():
    study.register();control_check();rows=[];hashes={}
    expected_old=study.read(study.lib.adaptive.OUT/'protocol.json')['input_sha256']
    for name,digest in expected_old.items():assert study.sha(study.ROOT/name)==digest
    for case in study.cases():
        configure_validation(case)
        for ds in study.DATASETS:
            dd=study.OUT/'cases'/case['id']/(ds.lower()+'_seed0')
            for m in study.METHODS:
                arm=dd/m;o=study.read(arm/'outcome.json');assert o['status']!='failed',o
                record=dict(case=case['id'],dataset=ds,method=m,status=o['status'],reused=study.reused(case['id'],m))
                if o['status']=='completed':
                    if not record['reused']:
                        check,_,tx=study.validate(arm,study.read(dd/'requests.json'),study.read(dd.parent/'mission_spec.json'),study.read(dd/'plan.json')['config'],.1)
                        assert check==study.read(arm/'verification.json');record.update(check)
                    else:
                        check=study.read(arm/'verification.json');assert check['passed'] and check['FL_rounds']==200 and check['FU_steps']==225
                    refs=prior.source('baseline',ds,m).parent
                    for name in ['feature_cache/features.pt','initial20/final.pt','reference200/final.pt']:
                        f=dd/name
                        # Request reuse preflight does not create these links for an empty method list.
                        if not f.exists():study.lib.adaptive.link(refs/name,f)
                        actual=study.sha(f);assert actual==study.sha(refs/name)
                        hashes[str(f.relative_to(study.ROOT))]=actual
                    assert {i for r in study.read(dd/'requests.json') for i in r['images']}==set(study.read(refs/'plan.json')['forgotten_images'])
                else:assert 'feasible' in o['status']
                rows.append(record)
    assert len(rows)==78
    r=dict(passed=True,arms=len(rows),new_training_arms=70,reused_arms=8,rows=rows,outcomes=dict(collections.Counter(x['status'] for x in rows)),input_sha256=hashes)
    study.write(study.OUT/'verification.json',r);study.write(study.OUT/'finished.json',dict(unix=time.time(),**r));return r


def child(cid):
    assert int(os.environ['OVA_WORKLOAD_GROUP_PID'])==os.getppid()
    original=fcntl.flock
    def lock(handle,mode):
        if getattr(handle,'name',None)==str(LOCK):assert mode==fcntl.LOCK_EX;return
        return original(handle,mode)
    fcntl.flock=lock;study.run_case(cid)


def run():
    study.register();control_check()
    study.write(study.OUT/'execution.json',dict(unix=time.time(),source=str(Path(__file__).relative_to(study.ROOT)),sha256=study.sha(Path(__file__)),
        concurrency=3,policy='Parent holds exclusive GPU lock; three independent condition processes, each uses <=8 solver workers on32 available CPUs. Same source,seed and simulated clocks; GPU walltime not used for energy.'))
    pending=[]
    for c in study.cases():
        if c['id']=='control':continue
        paths=[study.OUT/'cases'/c['id']/(ds.lower()+'_seed0')/m/'outcome.json' for ds in study.DATASETS for m in study.METHODS]
        if all(p.exists() for p in paths):assert all(study.read(p)['status']!='failed' for p in paths)
        else:pending.append(c['id'])
    def launch(cid):
        log=study.OUT/'case_logs'/f'{cid}.log';log.parent.mkdir(exist_ok=True)
        env=dict(os.environ,OVA_WORKLOAD_GROUP_PID=str(os.getpid()))
        with log.open('a') as f:subprocess.run([sys.executable,str(Path(__file__)),'case','--case',cid],cwd=study.ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        return cid
    with LOCK.open('a') as gpu:
        fcntl.flock(gpu,fcntl.LOCK_EX)
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures={pool.submit(launch,c):c for c in pending}
            for f in as_completed(futures):print(dict(completed_case=f.result()),flush=True)
    result=verify();study.state(stage='completed',outcomes=result['outcomes']);print(dict(passed=True,outcomes=result['outcomes']),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['case','run','control','verify']);p.add_argument('--case');a=p.parse_args()
    if a.action=='case':child(a.case)
    elif a.action=='run':run()
    elif a.action=='control':print(control_check())
    else:print(verify())
