"""Complete late regional FU before departure at a truncated FL budget.

Original100-round AirScape Proposed attempt reached its third site's16 FL
updates with one FU cycle still pending. Complete that work without adding FL.
Already-completed arms satisfy the same regional-completion rule unchanged.
"""
import argparse
import math
from pathlib import Path
import numpy as np
import workload_sweeps_v3 as base

ORIGINAL=base.OUT
OUT=base.ROOT/'results/workload_departure_fix_v1_20261028'
CASE='fl_rounds_100';DATASET='AirScape';METHOD=base.lib.PROPOSED
MAKE=base.proposed_arm


def register():
    path=OUT/'protocol.json'
    if path.exists():
        p=base.read(path)
        for name,digest in p['source_sha256'].items():assert base.sha(base.ROOT/name)==digest
        return p
    original=base.read(ORIGINAL/'protocol.json');sources=dict(original['source_sha256'])
    sources[str(Path(__file__).relative_to(base.ROOT))]=base.sha(Path(__file__))
    old=ORIGINAL/'cases'/CASE/(DATASET.lower()+'_seed0')/METHOD
    failure=base.read(old/'failure.json');assert 'close_site' in failure['traceback'] and 'AssertionError' in failure['error']
    assert len(list((old/'transactions').glob('*.json')))==48
    p=dict(registered_unix=__import__('time').time(),affected=[dict(case=CASE,dataset=DATASET,method=METHOD)],
        reason='Original FL-budget truncation reaches site2 quota16 with4 of5 FU cycles for its late release. Preserve failure and complete remaining regional FU before departure, without extra FL. Identical completion-before-departure rule for all methods; every other completed arm already satisfied it.',
        change='At a site FL quota, if regional FU remains, stay at that site and execute FU-only transactions until complete, then depart. All additional work,radio,broadcast,hover and time charged. No data,objective,seed,radio or quality threshold changes. Subsequent transaction seeds follow their actual shifted sequence.',
        expected='80 mission FL updates,225 FU cycles,45 requests. First48 transactions identical to preserved failed prefix. One added FU-only transaction expected from declared queue state; not based on quality/rank.',
        source_sha256=sources,original_failure_sha256=base.sha(old/'failure.json'))
    base.write(path,p);return p


def site_for(clock,q,fl):
    site=min(fl//base.SITE_FL,4)
    if clock.sites:
        current=clock.sites[-1]['site']
        pending=[r for r in q.items.values() if r['mission_site']==current and r['steps']<r['required_steps']]
        if fl==(current+1)*base.SITE_FL and pending:
            assert all(r['arrival_s'] is not None for r in pending)
            return current
    return site


def decide(clock,q,t,fl):
    site=clock.sites[-1]['site']
    if fl==(site+1)*base.SITE_FL:
        pending=q.pending(t)
        if pending:
            assert all(r['mission_site']==site for r in pending)
            return 'fu',[r['id'] for r in pending]
    return q.decision(t,fl)


def maker(*args):
    original=MAKE(*args)
    return base.cached_clone(original,[('site=min(fl//SITE_FL,4)','site=site_for(clock,q,fl)',1),
        ('kind,ids=q.decision(t,fl)','kind,ids=decide(clock,q,t,fl)',1)],dict(site_for=site_for,decide=decide))


def run():
    register();base.OUT=OUT;base.DATASETS=[DATASET];base.register=register;base.proposed_arm=maker
    base.run_case(CASE,'proposed');verify()


def verify():
    register();old=ORIGINAL/'cases'/CASE/(DATASET.lower()+'_seed0')/METHOD
    arm=OUT/'cases'/CASE/(DATASET.lower()+'_seed0')/METHOD
    assert base.read(arm/'outcome.json')['status']=='completed'
    previous=[base.read(p) for p in sorted((old/'transactions').glob('*.json'))]
    tx=[base.read(p) for p in sorted((arm/'transactions').glob('*.json'))]
    assert previous==tx[:len(previous)]
    assert len(tx)==81 and sum(t['kind']=='fu' for t in tx)==1
    extra=tx[48];assert extra['site']==2 and extra['kind']=='fu' and len(extra['FU_request_ids'])==5
    for name in ['plan.json','requests.json','initial20/final.pt','feature_cache/features.pt','reference200/final.pt']:
        assert base.sha(arm.parent/name)==base.sha(old.parent/name),name
    v=base.read(arm/'verification.json');assert v['passed'] and v['FL_rounds']==100 and v['FU_steps']==225
    result=dict(passed=True,affected=register()['affected'],original_failure_preserved=True,
        identical_failed_prefix_transactions=len(previous),additional_regional_FU_transactions=1,
        completed_FL_rounds=100,FU_cycles=225,extra_transaction_duration_s=extra['timing']['duration_s'],
        correction_source_sha256=base.sha(Path(__file__)))
    base.write(OUT/'verification.json',result);base.write(OUT/'finished.json',result);print(result,flush=True);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['run','verify']);a=p.parse_args()
    run() if a.action=='run' else verify()
