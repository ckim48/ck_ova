"""Validate departure correction, excluding measured host optimizer wall time.

That diagnostic is not part of the simulated energy or delay. All remaining
prefix fields, including every learned/radio value and simulated time, must match.
"""
import argparse
from pathlib import Path
import workload_departure_fix as fix

base=fix.base


def strip_wall(value):
    if isinstance(value,dict):return {k:strip_wall(v) for k,v in value.items() if k!='optimizer_wall_s'}
    if isinstance(value,list):return [strip_wall(v) for v in value]
    return value


def verify():
    fix.register();old=fix.ORIGINAL/'cases'/fix.CASE/(fix.DATASET.lower()+'_seed0')/fix.METHOD
    arm=fix.OUT/'cases'/fix.CASE/(fix.DATASET.lower()+'_seed0')/fix.METHOD
    assert base.read(arm/'outcome.json')['status']=='completed'
    previous=[base.read(p) for p in sorted((old/'transactions').glob('*.json'))]
    tx=[base.read(p) for p in sorted((arm/'transactions').glob('*.json'))]
    assert strip_wall(previous)==strip_wall(tx[:len(previous)])
    assert len(tx)==81 and sum(t['kind']=='fu' for t in tx)==1
    extra=tx[48];assert extra['site']==2 and extra['kind']=='fu' and len(extra['FU_request_ids'])==5
    for name in ['plan.json','requests.json','initial20/final.pt','feature_cache/features.pt','reference200/final.pt']:
        assert base.sha(arm.parent/name)==base.sha(old.parent/name),name
    v=base.read(arm/'verification.json');assert v['passed'] and v['FL_rounds']==100 and v['FU_steps']==225
    result=dict(passed=True,affected=fix.register()['affected'],original_failure_preserved=True,
        identical_failed_prefix_transactions=len(previous),prefix_excluded_fields=['optimizer_wall_s'],
        exclusion_reason='Measured host optimizer runtime is nondeterministic and excluded from modeled UAV energy/delay; every other transaction field is exactly identical.',
        additional_regional_FU_transactions=1,completed_FL_rounds=100,FU_cycles=225,
        extra_transaction_duration_s=extra['timing']['duration_s'],correction_source_sha256=base.sha(Path(fix.__file__)),
        verification_source_sha256={str(Path(__file__).relative_to(base.ROOT)):base.sha(Path(__file__))})
    base.write(fix.OUT/'verification.json',result);base.write(fix.OUT/'finished.json',result);print(result,flush=True);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['run','verify']);a=p.parse_args()
    if a.action=='run':fix.verify=verify;fix.run()
    else:verify()
