"""Resolve only the documented regional-completion failure; preserve all attempts."""
import argparse
import collections
from pathlib import Path
import subprocess
import sys
import time
import workload_sweeps_v3 as study
import workload_campaign as original
import workload_departure_fix as fix
import workload_correction_audit as correction_audit
import paper_uav_charts as prior


def corrected(case,ds,m):return (case,ds,m)==(fix.CASE,fix.DATASET,fix.METHOD)


def verify():
    study.register();original.control_check();correction=correction_audit.verify();rows=[];hashes={};cache={};original_statuses=[]
    def digest(p):
        key=str(p.resolve())
        if key not in cache:cache[key]=study.sha(p)
        hashes[str(p.relative_to(study.ROOT))]=cache[key];return cache[key]
    expected=study.read(study.lib.adaptive.OUT/'protocol.json')['input_sha256']
    for name,digest_expected in expected.items():assert digest(study.ROOT/name)==digest_expected
    for case in study.cases():
        original.configure_validation(case)
        for ds in study.DATASETS:
            dd=study.OUT/'cases'/case['id']/(ds.lower()+'_seed0')
            for m in study.METHODS:
                previous=dd/m;old=study.read(previous/'outcome.json');original_statuses.append(old['status'])
                arm=fix.OUT/'cases'/case['id']/dd.name/m if corrected(case['id'],ds,m) else previous
                o=study.read(arm/'outcome.json');assert o['status']=='completed',o
                if corrected(case['id'],ds,m):assert old['status']=='failed'
                else:assert old['status']=='completed'
                record=dict(case=case['id'],dataset=ds,method=m,status=o['status'],original_status=old['status'],
                    source=str(arm.relative_to(study.ROOT)),reused=study.reused(case['id'],m),regional_completion_correction=arm!=previous)
                if not record['reused']:
                    check,tl,tx=study.validate(arm,study.read(dd/'requests.json'),study.read(dd.parent/'mission_spec.json'),study.read(dd/'plan.json')['config'],.1)
                    assert check==study.read(arm/'verification.json');record.update(check)
                else:
                    check=study.read(arm/'verification.json');assert check['passed'] and check['FL_rounds']==200 and check['FU_steps']==225
                refs=prior.source('baseline',ds,m).parent
                for name in ['feature_cache/features.pt','initial20/final.pt','reference200/final.pt']:
                    assert digest(dd/name)==digest(refs/name)
                assert {i for r in study.read(dd/'requests.json') for i in r['images']}==set(study.read(refs/'plan.json')['forgotten_images'])
                rows.append(record)
    assert len(rows)==78 and collections.Counter(original_statuses)=={'completed':77,'failed':1}
    result=dict(passed=True,arms=78,new_completed_training_arms=70,reused_arms=8,training_attempts=71,
        original_outcomes=dict(collections.Counter(original_statuses)),outcomes={'completed':78},rows=rows,input_sha256=hashes,
        correction=correction,source_sha256={str(Path(__file__).relative_to(study.ROOT)):study.sha(Path(__file__))})
    study.write(study.OUT/'verification.json',result);study.write(study.OUT/'finished.json',dict(unix=time.time(),**result))
    print(dict(passed=True,arms=78,outcomes=result['outcomes'],original_outcomes=result['original_outcomes']),flush=True);return result


def run():
    # The original source remains frozen. It runs every queued condition before
    # returning the documented failed Future. No failed results are overwritten.
    result=subprocess.run([sys.executable,str(Path(original.__file__)),'run'],cwd=study.ROOT)
    if result.returncode:
        failed=[]
        for c in study.cases():
            for ds in study.DATASETS:
                for m in study.METHODS:
                    p=study.OUT/'cases'/c['id']/(ds.lower()+'_seed0')/m/'outcome.json'
                    o=study.read(p)
                    if o['status']=='failed':failed.append((c['id'],ds,m))
        assert failed==[(fix.CASE,fix.DATASET,fix.METHOD)],failed
    subprocess.run([sys.executable,str(Path(correction_audit.__file__)),'run'],cwd=study.ROOT,check=True)
    return verify()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['run','verify']);a=p.parse_args()
    run() if a.action=='run' else verify()
