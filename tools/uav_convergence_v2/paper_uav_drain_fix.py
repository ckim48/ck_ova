"""Complete already-started asynchronous jobs before ending active service.

A source-frozen correction for five arms rejected by the original terminal-time
assertion. Preserve original failure records and prove every transaction/job is
unchanged. Only the final 0.128-0.246ms compute drain extends hover/service time.
"""
import argparse
import copy
import inspect
import linecache
from pathlib import Path
import subprocess
import sys
import time
import paper_uav_benchmarks as study

ORIGINAL=study.OUT
OUT=study.ROOT/'results/paper_uav_drain_fix_v1_20261028'
METHOD='uav_afl_ota_gaft'


def register():
    path=OUT/'protocol.json'
    if path.exists():
        p=study.read(path)
        assert study.sha(Path(__file__))==p['correction_source_sha256']
        for name,digest in p['source_sha256'].items():assert study.sha(study.ROOT/name)==digest
        return p
    base=study.register();affected=[]
    outcomes=list((ORIGINAL/'cases').glob('*/*/*/outcome.json'));assert len(outcomes)==84
    for f in outcomes:
        o=study.read(f)
        if o['status']=='failed':
            assert f.parent.name==METHOD
            failure=study.read(f.parent/'failure.json');assert "all(j['ready_s']<=t+1e-8 for j in all_jobs)" in failure['traceback']
            tx=[study.read(p) for p in sorted((f.parent/'transactions').glob('*.json'))]
            jobs=[study.read(p) for p in (f.parent/'jobs').glob('*.json')]
            assert tx[-1]['completed_FL_rounds']==200
            latest=max(j['ready_s'] for j in jobs);missing=latest-tx[-1]['end_s'];assert 0<missing<.001
            affected.append(dict(case=o['case'],dataset=o['dataset'],method=METHOD,missing_terminal_wait_s=missing,original_outcome_sha256=study.sha(f),source=str(f.parent.relative_to(study.ROOT))))
        else:assert o['status']=='completed'
    assert len(affected)==5
    p=dict(registered_unix=time.time(),correction_source_sha256=study.sha(Path(__file__)),source_sha256=base['source_sha256'],affected=affected,
        change='After last required update/broadcast, wait for every already-started local compute job to finish, charge that waiting interval as hover. No gradient/objective/radio/schedule decisions before final drain change.',
        preservation='Original five failed arms remain untouched. Verify all transmitted records, work ledgers and local-job records against corrected reruns. Remaining79completed arms unaffected.',
        expected_change='0.128ms CapERA or0.24576ms AirScape active completion delay; corresponding10-UAV hover energy. Final learned weights/metrics unchanged from stored last transaction evaluation.')
    study.write(path,p);return p


def install():
    source=inspect.getsource(study.train_arm)
    old="    assert fl==180 and all(j['ready_s']<=t+1e-8 for j in all_jobs)"
    new="    terminal=max([t,*[j['ready_s'] for j in all_jobs]])\n    if terminal>t:clock.wait(t,terminal);t=terminal\n"+old
    assert source.count(old)==1;source=source.replace(old,new).replace('def train_arm(', 'def drain_fixed_train_arm(',1)
    filename='/tmp/paper_uav_terminal_drain.py';linecache.cache[filename]=(len(source),None,source.splitlines(True),filename)
    exec(compile(source,filename,'exec'),study.__dict__)
    study.train_arm=study.drain_fixed_train_arm


def run_case(cid):
    protocol=register();datasets=[r['dataset'] for r in protocol['affected'] if r['case']==cid]
    study.OUT=OUT;study.register=lambda:protocol
    case=next(c for c in study.cases() if c['id']==cid);study.CURRENT_CASE=case;study.BATCH=int(case['value'])
    lib=study.lib;lib.BATCH=study.BATCH;lib.ACTIVE='fedavg_clip_gaft';lib.infra.extra.Data=lib.infra.VideoData
    lib.adaptive.link(lib.adaptive.SOURCE/'prepared/airscape_seed0',OUT/'prepared/airscape_seed0')
    _,base=lib.infra.prior.load_base();install()
    import worker,accounting,event_energy
    lib.infra.ORIGINAL_INSTRUMENTED=accounting.instrumented_engine;accounting.instrumented_engine=lib.infra.controlled_engine
    event_energy.method_energy=study.energy_ledger
    def maker(b,deployment,radios,cap):study.BASE=b;study.DEPLOYMENT=deployment;return study.train_arm
    run=base.repeat.clone_function(worker.run,[
        ("opt=base.base.ARMS['proposed_fu_guard_cliperase' if method=='heuristic_ova' else method]",'opt=dict(method=method,scope="paper-core transfer; terminal drain correction")',1)],
        dict(OUT=OUT,METHODS=[METHOD],EXTRA_DATASETS=['AirScape'],make_arm=maker,validate=study.validate,state=study.state))
    run(case,datasets,[METHOD])
    for ds in datasets:assert study.read(OUT/'cases'/cid/(ds.lower()+'_seed0')/METHOD/'outcome.json')['status']=='completed'
    print(dict(corrected_case=cid,datasets=datasets),flush=True)


def verify():
    p=register();rows=[]
    for r in p['affected']:
        old=study.ROOT/r['source'];new=OUT/'cases'/r['case']/(r['dataset'].lower()+'_seed0')/METHOD
        assert study.sha(old/'outcome.json')==r['original_outcome_sha256']
        o=study.read(new/'outcome.json');assert o['status']=='completed'
        count={}
        for folder in ['transactions','work_attempts','jobs']:
            files=sorted((old/folder).glob('*.json'));assert len(files)==len(list((new/folder).glob('*.json')))
            for f in files:assert study.sha(f)==study.sha(new/folder/f.name),(folder,f.name)
            count[folder]=len(files)
        last=study.read(sorted((old/'transactions').glob('*.json'))[-1]);final=study.read(new/'service_completion.json')
        assert last['evaluation']==final['evaluation']
        assert abs(final['mission_clock_s']-last['end_s']-r['missing_terminal_wait_s'])<1e-10
        rows.append(dict(**r,corrected_source=str(new.relative_to(study.ROOT)),bit_identical_records=count,final_metrics_identical=True))
    checked=dict(passed=True,corrected_arms=5,rows=rows,correction_source_sha256=study.sha(Path(__file__)))
    study.write(OUT/'verification.json',checked);study.write(OUT/'finished.json',checked)
    # Final release view explicitly identifies source replacements; old outcomes stay failed.
    cases=[];mapping={(r['case'],r['dataset'],METHOD):r for r in rows}
    for c in study.cases():
        for ds in study.DATASETS:
            for method in study.METHODS:
                key=(c['id'],ds,method);origin=OUT if key in mapping else ORIGINAL
                ad=origin/'cases'/c['id']/(ds.lower()+'_seed0')/method
                o=study.read(ad/'outcome.json');assert o['status']=='completed'
                v=study.read(ad/'verification.json');assert v['passed']
                cases.append(dict(case=c['id'],dataset=ds,method=method,source=str(ad.relative_to(study.ROOT)),corrected_terminal_drain=key in mapping,**v))
    release=dict(passed=True,completed_arms=84,original_completed_arms=79,corrected_arms=5,original_failed_outcomes_preserved=True,rows=cases,
        correction_protocol=str((OUT/'protocol.json').relative_to(study.ROOT)),correction_verification=str((OUT/'verification.json').relative_to(study.ROOT)))
    study.write(ORIGINAL/'verification.json',release);study.write(ORIGINAL/'finished.json',dict(unix=time.time(),**release));study.state(stage='completed_with_terminal_drain_correction')
    return checked


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['run','case','verify']);p.add_argument('--case');a=p.parse_args()
    if a.action=='case':run_case(a.case);return
    if a.action=='verify':print(verify());return
    protocol=register()
    for cid in dict.fromkeys(r['case'] for r in protocol['affected']):subprocess.run([sys.executable,str(Path(__file__)),'case','--case',cid],cwd=study.ROOT,check=True)
    print(verify())


if __name__=='__main__':main()
