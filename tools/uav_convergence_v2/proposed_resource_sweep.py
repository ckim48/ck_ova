"""Independent Proposed sweep while paper baselines are reviewed."""
import argparse
import collections
from pathlib import Path
import subprocess
import sys
import time
import literature_baselines as runner

ROOT=runner.ROOT
OUT=ROOT/'results/proposed_resource_sweep_v1_20261028'
read,write,sha=runner.read,runner.write,runner.sha


def register():
 path=OUT/'protocol.json'
 if path.exists():
  p=read(path)
  for group in ['source_sha256','input_sha256']:
   for name,d in p[group].items():assert sha(ROOT/name)==d,name
  return p
 old=runner.adaptive.register();sources=dict(old['source_sha256'])
 for p in [Path(__file__),Path(runner.__file__)]:sources[str(p.relative_to(ROOT))]=sha(p)
 p=dict(registered_unix=time.time(),method=runner.PROPOSED,datasets=runner.DATASETS,
  axes=runner.AXES,training_cases=runner.cases(),new_training_runs=20,reused_training_arms=6,
  definition='Existing adaptive Proposed, unchanged, with one-factor bandwidth, FU-cycle and FL-batch-cap sweeps. Downlink/frequency use separately audited scheduler/energy replay.',
  scope='No external comparison baseline is claimed or run by this driver. literature_baselines.py supplies only the tested sampler/runner plumbing here; its FT/GA/GAFT proposal is not the final main comparison.',
  accounting='Existing active FL/FU communication+compute+hover; excludes travel, collection, preflight. Same MAC accounting and physical limits as previous adaptive version.',
  outcome_policy='All predeclared values and outcomes retained; no outcome-based change or rank selection. Fixed warm20/data/reference and sampling seed0.',
  source_sha256=sources,input_sha256=dict(old['input_sha256']))
 write(path,p);return p


def verify():
 rows=[]
 for case in runner.cases():
  for ds in runner.DATASETS:
   arm=OUT/'cases'/case['id']/(ds.lower()+'_seed0')/runner.PROPOSED
   o=read(arm/'outcome.json');assert o['status']!='failed',o
   row=dict(case=case['id'],dataset=ds,status=o['status'],reused=runner.reused(case['id'],runner.PROPOSED))
   if o['status']=='completed':
    expected=45*(case['value'] if case['axis']=='fu_steps' else 5)
    v=read(arm/'verification.json');assert v['FL_rounds']==200 and v['FU_steps']==expected
    row.update(FL_rounds=200,FU_cycles=expected,physical_validation=v['passed'])
   rows.append(row)
 result=dict(passed=True,rows=rows,outcomes=dict(collections.Counter(r['status'] for r in rows)))
 write(OUT/'verification.json',result);return result


def main():
 runner.OUT=OUT;runner.register=register
 p=argparse.ArgumentParser();p.add_argument('action',choices=['run','case','verify']);p.add_argument('--case',default='baseline');a=p.parse_args()
 if a.action=='case':runner.run_case(a.case,runner.PROPOSED);return
 if a.action=='verify':print(verify());return
 register()
 for c in runner.cases():subprocess.run([sys.executable,str(Path(__file__)),'case','--case',c['id']],cwd=ROOT,check=True)
 result=verify();write(OUT/'finished.json',dict(unix=time.time(),**result));runner.state(stage='completed');print(result,flush=True)


if __name__=='__main__':main()
