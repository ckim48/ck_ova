"""Fresh training for CPU/downlink sensitivities, so scheduler changes are real."""
import argparse
from pathlib import Path
import subprocess
import sys
import time
import literature_baselines as runner
ROOT=runner.ROOT
OUT=ROOT/'results/proposed_resource_timing_v1_20261028'
read,write,sha=runner.read,runner.write,runner.sha

def cases():
    return [dict(id=runner.case_id(a,v),axis=a,value=v) for a in ['downlink','frequency'] for v in runner.AXES[a]['values'] if v!=runner.AXES[a]['base']]

def register():
    p=OUT/'protocol.json'
    if p.exists():
        obj=read(p)
        for k,v in obj['source_sha256'].items():assert sha(ROOT/k)==v,k
        return obj
    obj=runner.adaptive.register();source=dict(obj['source_sha256'])
    for f in [Path(__file__),Path(runner.__file__)]:source[str(f.relative_to(ROOT))]=sha(f)
    obj=dict(registered_unix=time.time(),cases=cases(),method=runner.PROPOSED,datasets=runner.DATASETS,new_training_arms=16,
      reason='Fresh reruns instead of timing replay because CPU/downlink changes can alter actual request scheduling. No rescaled old convergence curve.',
      source_sha256=source,input_sha256=dict(obj['input_sha256']))
    write(p,obj);return obj

def verify():
    rows=[]
    for c in cases():
        for ds in runner.DATASETS:
            ad=OUT/'cases'/c['id']/(ds.lower()+'_seed0')/runner.PROPOSED;o=read(ad/'outcome.json')
            assert o['status']=='completed',o
            v=read(ad/'verification.json');assert v['passed'] and v['FL_rounds']==200 and v['FU_steps']==225
            rows.append(dict(case=c['id'],dataset=ds,**v))
    obj=dict(passed=True,completed_arms=len(rows),rows=rows);write(OUT/'verification.json',obj);return obj

def main():
    runner.OUT=OUT;runner.register=register;runner.cases=cases
    p=argparse.ArgumentParser();p.add_argument('action',choices=['run','case','verify']);p.add_argument('--case');a=p.parse_args()
    if a.action=='case':runner.run_case(a.case,runner.PROPOSED);return
    if a.action=='verify':print(verify());return
    register()
    for c in cases():subprocess.run([sys.executable,str(Path(__file__)),'case','--case',c['id']],cwd=ROOT,check=True)
    result=verify();write(OUT/'finished.json',dict(unix=time.time(),**result));runner.state(stage='completed')

if __name__=='__main__':main()
