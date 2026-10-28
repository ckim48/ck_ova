"""Verify plots, reference conventions, exact400 checkpoints and provenance."""
from pathlib import Path
import math
import numpy as np
from PIL import Image
import pks_report as report

read,write,sha=report.read,report.write,report.sha


def main():
 report.study.register()
 dest=report.DEST;manifest=read(dest/'data/manifest.json');rows=read(dest/'data/metrics.json')
 strict=read(dest/'data/strict400.json');budget=read(dest/'data/budget400.json');series=read(dest/'data/plot_series.json')
 assert manifest['complete'] and len(rows)==40 and len(strict)==4 and len(manifest['figures'])==14
 assert set(r['method'] for r in rows)==set(report.METHODS)
 for r in rows:
  assert r['policy']==report.evaluation.POLICY[r['method']]
  assert r['common_mean_pass']==(r['common_KL']<=.01) and r['common_all_requests_pass']==(r['common_worst_KL']<=.01)
  assert r['matched_mean_pass']==(r['matched_KL']<=.01) and r['matched_all_requests_pass']==(r['matched_worst_KL']<=.01)
  if r['stage']=='terminal':assert 400<r['active_s']<403
 by={(r['dataset'],r['method'],r['stage']):r for r in rows+strict}
 for s in series:
  assert len(s['x'])==len(s['y'])
  if s.get('bar'):
   r=by[(s['dataset'],s['method'],s['stage'])];assert s['y']==[r[s['metric']]]
  else:
   assert s['drawstyle']=='steps-post'
   assert all(a<=b for a,b in zip(s['x'],s['x'][1:]))
   assert s['x'][-1]<=400 and s['display_x'][-1]==400 and s['display_y'][-1]==s['y'][-1]
   r=next(r for r in budget if r['dataset']==s['dataset'] and r['method']==s['method'])
   assert s['y'][-1]==r['retained_mR_percent' if s['metric']=='mR' else 'retained_loss']
 for r in strict:
  assert r['active_s']<=400 and r['FL_round']==612
  b=next(b for b in budget if b['dataset']==r['dataset'] and b['method']==r['method'])
  assert r['retained_mR_percent']==b['retained_mR_percent'] and r['retained_loss']==b['retained_loss']
  assert r['active_s']==b['checkpoint_s']
 for f in manifest['figures']:
  precision=[]
  for panel in f['axes']:
   item={}
   for dim in ['x','y']:
    a=panel[dim];t=a['ticks'];assert a['limits']==[t[0],t[-1]]
    assert np.allclose(np.diff(t),t[1]-t[0])
    digits=[len(x.split('.')[-1]) if '.' in x else 0 for x in a['labels']]
    assert len(set(digits))==1;item[dim]=digits[0]
   precision.append(item)
  assert precision[0]==precision[1]
  for item in f['files']:
   p=dest/item['file'];assert sha(p)==item['sha256']
   if p.suffix=='.png':
    image=Image.open(p);assert image.width>1800 and image.height>800
 for file,digest in manifest['source_sha256'].items():assert sha(report.ROOT/file)==digest,file
 audit=read(dest/'data/audit.json');assert audit['passed'] and audit['zero_change_control']['passed']
 assert len(audit['rows'])==20 and len(audit['gradient_checks'])==6
 for ds in report.study.q.infra.DATASETS:
  x=read(report.strict_budget.OUT/(ds.lower()+'_seed0')/'finished.json')
  assert x['passed'] and x['unchanged_PKS_replay_exact'] and x['references_reproduced_exactly']
 result=dict(passed=True,figures=14,series=len(series),endpoints=40,strict_budget_endpoints=4,
  checks=['All10 methods retain both common and policy-matched reference evaluations',
   'Original P-KS warmup/workload and continuation reproduce exactly; no result overwritten',
   'Both gradient passes and their computation are present in actual execution records',
   'All bounded gradients respect the declared local training-only norm constraint',
   'Only eligible retained data enter selection; common physical clocks and energies audited',
   'Strict400 KL, mR and loss use the identical saved model; no mixed-endpoint claim',
   'All figure values, regular endpoint ticks, precision and source/artifact hashes verified'],
  source_sha256={str(Path(__file__).relative_to(report.ROOT)):sha(Path(__file__))})
 write(dest/'data/artifact_checks.json',result);print(result)


if __name__=='__main__':main()
