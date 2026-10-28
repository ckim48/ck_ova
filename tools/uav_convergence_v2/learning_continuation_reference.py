"""Reporting-only round-matched retained reference for final continuations.

Never feeds candidate training. Extends the same common reference200 using
uniform retained sampling and the declared continuation LR, with exact sums.
"""
import argparse
import fcntl
from pathlib import Path
import time
import numpy as np
import learning_time_budget as study

OUT=study.ROOT/'results/learning_continuation_reference_v1_20261028'
read,write,sha=study.read,study.write,study.sha


def run(ds):
 directory=OUT/(ds.lower()+'_seed0')
 if (directory/'finished.json').exists():return
 targets={m:read(study.OUT/(ds.lower()+'_seed0')/m/'finished.json')['extra_FL'] for m in study.METHODS}
 sources=dict(study.register()['source_sha256']);sources[str(Path(__file__).relative_to(study.ROOT))]=sha(Path(__file__))
 protocol=dict(unix=time.time(),source_sha256=sources,targets=targets,
  scope='Reporting-only extension of common retained reference200 to each actual final FL count. Uniform retained batches, sample-weighted exact sums and shared constant0.0035 LR. This adjusts round count; it is not a method-specific retraining oracle for the selective/equal-client learners, nor a deletion certificate. No reference output feeds candidate learning.')
 write(directory/'protocol.json',protocol)
 q=study.quality;q.infra.lib.infra.extra.Data=q.infra.lib.infra.VideoData
 event,base=q.infra.lib.infra.prior.load_base();base.repeat.clone_function=q.infra.cached_clone
 import torch
 from prepare_extra import native_engine,native_base,Data
 from train import update_theta
 assert torch.cuda.is_available();torch.set_num_threads(1)
 with (study.ROOT/'results/overnight_energy_v2/association_objective_gpu.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  initial=study.source(ds,'current');plan=read(initial.parent/'plan.json')
  data=Data(ds) if ds=='AirScape' else event.recover.Data(ds)
  if ds=='AirScape':base=native_base(base,data.caption_count)
  Engine=native_engine(base,data.caption_count) if ds=='AirScape' else base.Engine
  payload=torch.load(initial.parent/'feature_cache/features.pt',map_location='cpu',weights_only=False)
  engine=Engine(data,plan,payload);theta=base.core.load_theta(initial.parent/'reference200/final.pt',engine.device)
  queue=base.RoundQueue(read(initial.parent/'requests.json'));queue.items={r['id']:r for r in read(initial/'request_records.json')}
  mission=read(initial/'mission_timeline.json')['service_end_s'];pools=base.clip.stream.retained_pools(plan,queue,mission)
  total=sum(map(len,pools));deleted=set(plan['forgotten_images']);assert not any(set(p)&deleted for p in pools)
  rows=[]
  for extra in range(1,max(targets.values())+1):
   k=200+extra;updates=[]
   for u,pool in enumerate(pools):
    batch=np.random.default_rng(730000+k*10+u).choice(pool,min(16,len(pool)),replace=False).astype(int).tolist()
    g,_=engine.gradient(theta,batch,k);updates.append((-study.LR*len(pool)/total*g[:-1]).cpu().numpy())
   theta=update_theta(theta,np.sum(updates,axis=0))
   if extra in targets.values():
    reference,prob=engine.evaluate(theta,None);base.core.save_theta(directory/f'reference_{k}.pt',theta,k)
    for m,n in targets.items():
     if n!=extra:continue
     arm=study.OUT/(ds.lower()+'_seed0')/m;candidate=base.core.load_theta(arm/'final.pt',engine.device)
     value=base.clip.stream.evaluation(engine,candidate,prob,queue,mission+1000)
     finished=read(arm/'finished.json')
     row=dict(dataset=ds,method=m,label=study.METHODS[m],FL_round=k,active_s=finished['active_end_s'],
      retained_mR_percent=100*value['retained']['mR'],retained_loss=value['retained']['loss'],
      matched_common_reference_KL=value['forget']['KL_retrain_to_candidate'],worst_request_KL=value['worst_request_KL'],
      reference_mR_percent=100*reference['retained']['mR'],reference_loss=reference['retained']['loss'],
      candidate_sha256=sha(arm/'final.pt'),reference_sha256=sha(directory/f'reference_{k}.pt'))
     rows.append(row)
  assert len(rows)==len(targets)
  write(directory/'finished.json',dict(passed=True,reporting_only=True,rows=rows));print(dict(dataset=ds,rows=rows),flush=True)


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=study.quality.infra.DATASETS,required=True);run(p.parse_args().dataset)
