"""Check measured plot values, saved artifacts and causal sampling records."""
import math
from pathlib import Path
import numpy as np
from PIL import Image
import learning_quality_report as report
import learning_time_budget as continuation

ROOT=report.ROOT
read,write,sha=report.read,report.write,report.sha


def sampling():
 rows=[]
 for ds in report.quality.infra.DATASETS:
  reference=report.variant_source(ds,'control')
  for variant in report.quality.VARIANTS:
   arm=report.variant_source(ds,variant)
   for name in ['plan.json','requests.json','initial20/final.pt','reference200/final.pt']:
    assert sha(arm.parent/name)==sha(reference.parent/name),(ds,variant,name)
   spec=report.quality.VARIANTS[variant];calls=0
   if spec['selection']:
    requests=read(arm/'request_records.json');plan=read(arm.parent/'plan.json')
    for file in (arm/'work_attempts').glob('*.json'):
     work=read(file);records=work['compute']['FL_sample_selection']
     assert sum(work['compute']['device_macs'])>=sum(work['compute']['base_device_macs'])
     for r in records:
      user=r['user'];deleted={i for q in requests if q['arrival_s']<=work['mission_time_s']+1e-9 for i in q['images']}
      pool=set(plan['partitions'][user])-deleted
      assert set(r['selected'])<=pool and len(r['selected'])==min(spec['selection'],len(pool))
      assert len(set(r['selected']))==len(r['selected']) and set(r['exploration'])<=set(r['selected'])
      assert r['candidates']==len(pool);calls+=1
    assert calls==1800
   rows.append(dict(dataset=ds,variant=variant,shared_inputs=True,causal_selection_calls=calls))
 result=dict(passed=True,arms=len(rows),rows=rows,checks=['Identical plans, requests, initial and reference checkpoints','Selection records contain only locally retained eligible samples at execution time','All declared FL selections executed and their extra operations charged'])
 write(ROOT/'final_final_charts/learning_quality/data/audit.json',result)


def bundle(name,figures,rows_count):
 dest=ROOT/'final_final_charts'/name;manifest=read(dest/'data/manifest.json');rows=read(dest/'data/metrics.json');series=read(dest/'data/plot_series.json')
 assert manifest['complete'] and len(manifest['figures'])==figures and len(rows)==rows_count
 by={(r.get('case'),r['dataset'],r['method']):r for r in rows}
 for s in series:
  assert len(s['x'])==len(s['y']) and all(a<=b for a,b in zip(s['x'],s['x'][1:]))
  if 'cases' in s:
   expected=[by[(c,s['dataset'],s['method'])][s['metric']]*s['scale'] for c in s['cases']]
   assert np.allclose(s['y'],expected,atol=1e-10,rtol=0)
  elif name!='same_time':
   key={'mR':'retained_mR_percent','loss':'retained_loss','KL':'forget_KL'}[s['metric']];row=by[('baseline',s['dataset'],s['method'])]
   assert math.isclose(s['y'][-1],row[key],abs_tol=1e-12)
   if '_time_' in s['figure']:assert math.isclose(s['x'][-1],row['delay_s'],abs_tol=1e-8)
   else:assert s['x'][-1]==200
  else:
   if s['metric'] in ['mR','loss']:
    r=next(r for r in rows if r['dataset']==s['dataset'] and r['method']==s['method'] and r['budget_s']==400)
    assert s['x'][-1]==r['quality_checkpoint_s']
    assert s['y'][-1]==r['retained_mR_percent' if s['metric']=='mR' else 'retained_loss']
   assert s['drawstyle']=='steps-post' and s['display_x'][-1]==400 and s['display_y'][-1]==s['y'][-1]
 for f in manifest['figures']:
  precisions=[]
  assert len(f['axes'])==2
  for panel in f['axes']:
   precision={}
   for d in ['x','y']:
    axis=panel[d];ticks=axis['ticks'];assert np.allclose(np.diff(ticks),ticks[1]-ticks[0])
    assert axis['limits']==[ticks[0],ticks[-1]]
    digits=[len(s.split('.')[-1]) if '.' in s else 0 for s in axis['labels']];assert len(set(digits))==1;precision[d]=digits[0]
   precisions.append(precision)
  assert precisions[0]==precisions[1]
  for item in f['files']:
   path=dest/item['file'];assert sha(path)==item['sha256']
   if path.suffix=='.png':
    image=Image.open(path);assert image.width>1800 and image.height>800
 for name,digest in manifest['source_sha256'].items():assert sha(ROOT/name)==digest,name
 result=dict(passed=True,figures=figures,rows=rows_count,series=len(series),checks=['Plotted metrics match their measured tables','Endpoint ticks are regular and panel precision agrees','All registered figure and provenance hashes match','No smoothed, rescaled or invented quality trajectories'])
 write(dest/'data/artifact_checks.json',result);print(dict(bundle=dest.name,**result),flush=True)


def main():
 report.quality.register();report.correction.register();continuation.register()
 sampling()
 bundle('four_methods',23,168);bundle('learning_quality',6,20);bundle('same_time',4,60)


if __name__=='__main__':main()
