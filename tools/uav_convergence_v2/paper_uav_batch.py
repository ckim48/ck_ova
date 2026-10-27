"""Execute immutable paper arms sharing dataset loading, without changing training."""
import argparse
from pathlib import Path
import subprocess
import sys
import time
import paper_uav_benchmarks as study


def run_case(cid):
    study.register();case=next(c for c in study.cases() if c['id']==cid)
    study.CURRENT_CASE=case;study.BATCH=int(case['value']) if case['axis']=='batch' else 16
    lib=study.lib;lib.BATCH=study.BATCH;lib.ACTIVE='fedavg_clip_gaft';lib.infra.extra.Data=lib.infra.VideoData
    lib.adaptive.link(lib.adaptive.SOURCE/'prepared/airscape_seed0',study.OUT/'prepared/airscape_seed0')
    _,base=lib.infra.prior.load_base()
    import worker,accounting,event_energy
    lib.infra.ORIGINAL_INSTRUMENTED=accounting.instrumented_engine;accounting.instrumented_engine=lib.infra.controlled_engine
    event_energy.method_energy=study.energy_ledger
    def maker(b,deployment,radios,cap):
        study.BASE=b;study.DEPLOYMENT=deployment;return study.train_arm
    run=base.repeat.clone_function(worker.run,[
        ("opt=base.base.ARMS['proposed_fu_guard_cliperase' if method=='heuristic_ova' else method]",'opt=dict(method=method,scope="paper-core transfer; see protocol.json")',1)],
        dict(OUT=study.OUT,METHODS=list(study.METHODS),EXTRA_DATASETS=['AirScape'],make_arm=maker,validate=study.validate,state=study.state))
    run(case,study.DATASETS,list(study.METHODS))
    statuses={ds+':'+m:study.read(study.OUT/'cases'/cid/(ds.lower()+'_seed0')/m/'outcome.json')['status'] for ds in study.DATASETS for m in study.METHODS}
    assert set(statuses.values())=={'completed'},statuses
    print(dict(case=cid,statuses=statuses),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['run','case']);p.add_argument('--case');a=p.parse_args()
    if a.action=='case':run_case(a.case);return
    study.register()
    study.write(study.OUT/'execution_batching.json',dict(unix=time.time(),source=str(Path(__file__).relative_to(study.ROOT)),sha256=study.sha(Path(__file__)),
        reason='Reuse each dataset engine load across both independent arms; fresh theta/queue/caches and work-counter deltas per arm, original seeds and source unchanged. Completed arms skipped. No training algorithm or outcome filter changes.'))
    for c in study.cases():
        paths=[study.OUT/'cases'/c['id']/(ds.lower()+'_seed0')/m/'outcome.json' for ds in study.DATASETS for m in study.METHODS]
        if all(p.exists() for p in paths):
            assert all(study.read(p)['status']=='completed' for p in paths);continue
        subprocess.run([sys.executable,str(Path(__file__)),'case','--case',c['id']],cwd=study.ROOT,check=True)
    result=study.verify();study.write(study.OUT/'finished.json',dict(unix=time.time(),**result));study.state(stage='completed')


if __name__=='__main__':main()
