"""Disclosed moderate-stress follow-up selected for feasibility, not model ranking."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import combined_iq_diagnostic_v1 as base

ROOT,read,write,sha=base.ROOT,base.read,base.write,base.sha
OUT=ROOT/'results/combined_iq_diagnostic_v2_20261030'
CASES={'fu_moderate':dict(ratio=.25,channel_gain_db=0.,modes=['combined','iq'])}
PARENT=base.OUT


def register():
    p=OUT/'protocol.json'
    if p.exists():
        result=read(p);assert result['source_sha256']==sha(Path(__file__));return result
    result=dict(registered_unix=time.time(),source_sha256=sha(Path(__file__)),
        parent_source_sha256=sha(Path(base.__file__)),cases=CASES,
        parent_protocol=read(PARENT/'protocol.json'),
        reason='Ratio0.1 had no feasible I/Q design at the first FU transaction on either dataset. The registered frozen-update grid showed ratio0.25 feasible for all five regional snapshots of both datasets at0dB. Run this same shared moderate setting end-to-end for BOTH methods. Selected using feasibility, before its mR/KL outcomes; adaptive development, not independent confirmation.',
        fixed='Only FU relative tolerance ratio changes from the original balanced case; keep total MSE/C9 budgets, learning, hardware, geometry, seed and workload unchanged.',
        all_prior_failures_preserved=True,no_required_winning_outcome=True)
    write(p,result);return result


def install():
    base.OUT=OUT;base.CASES=CASES;base.register=register


def queue():
    register();(OUT/'logs').mkdir(exist_ok=True);rows=[]
    for mode in ['combined','iq']:
        if (OUT/'fu_moderate'/mode/'finished.json').exists():continue
        with (OUT/'logs'/('fu_moderate_'+mode+'.log')).open('w') as f:
            result=subprocess.run([sys.executable,__file__,'run','--method',mode],stdout=f,stderr=subprocess.STDOUT,
                env=dict(os.environ,MPLCONFIGDIR='/tmp/ova-mpl'))
        rows.append(dict(mode=mode,returncode=result.returncode));write(OUT/'queue_status.json',rows);print(rows[-1],flush=True)
    write(OUT/'finished.json',dict(unix=time.time(),runs=rows))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['run','queue']);p.add_argument('--method',choices=['combined','iq']);a=p.parse_args()
    if a.action=='queue':queue()
    else:install();base.run('fu_moderate',a.method)
