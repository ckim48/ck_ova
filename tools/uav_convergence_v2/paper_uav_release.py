"""Reproduce the complete published campaign, preserving documented failed arms."""
import argparse
from pathlib import Path
import subprocess
import sys
import paper_uav_benchmarks as study


def command(script,*args,check=True):
    return subprocess.run([sys.executable,str(Path(__file__).with_name(script)),*args],cwd=study.ROOT,check=check)


def run():
    command('proposed_resource_sweep.py','run');command('proposed_resource_timing.py','run')
    for case in study.cases():
        cid=case['id'];arms=[study.OUT/'cases'/cid/(ds.lower()+'_seed0')/m for ds in study.DATASETS for m in study.METHODS]
        if not all((a/'outcome.json').exists() for a in arms):command('paper_uav_batch.py','case','--case',cid,check=False)
        for a in arms:
            o=study.read(a/'outcome.json')
            if o['status']=='completed':continue
            # Only the preserved terminal-drain assertion is an expected original rejection.
            assert o['status']=='failed' and a.name=='uav_afl_ota_gaft'
            assert "all(j['ready_s']<=t+1e-8 for j in all_jobs)" in study.read(a/'failure.json')['traceback']
            last=study.read(sorted((a/'transactions').glob('*.json'))[-1]);assert last['completed_FL_rounds']==200
    command('paper_uav_drain_fix.py','run')
    finish()


def finish():
    command('paper_uav_audit.py');command('paper_uav_charts.py');command('paper_uav_artifact_check.py')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['run','finish']);a=p.parse_args()
    run() if a.action=='run' else finish()
