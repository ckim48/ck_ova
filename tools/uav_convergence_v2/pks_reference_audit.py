"""Separate the effect of a different FL sampler from unlearning discrepancy.

Evaluation-only references. Original common-reference scores are preserved.
No reference parameters or predictions are made available to candidate training.
"""
import argparse
import fcntl
from pathlib import Path
import time
import numpy as np
import learning_quality_report as report
import learning_time_budget as continuation

q=report.quality
ROOT=q.ROOT;OUT=ROOT/'results/pks_reference_audit_v1_20261028'
read,write,sha=q.read,q.write,q.sha


def run(ds):
 directory=OUT/(ds.lower()+'_seed0')
 if (directory/'finished.json').exists():return
 assert not (directory/'started.json').exists(),'Preserve incomplete audit'
 sources=dict(q.register()['source_sha256']);sources[str(Path(__file__).relative_to(ROOT))]=sha(Path(__file__))
 protocol=dict(registered_unix=time.time(),source_sha256=sources,
  purpose='Reporting-only diagnosis of training-policy mismatch; never substitutes original reported KL or feeds candidate learning.',
  variants=['original_reference_reproduction','uniform_sampler','selective_sampler'],
  warmup='All counterfactual references start from theta0 and execute the original retained reference records for rounds1..20 with exact aggregation, not the noisy proxy20.',
  matching='From21 onward use the same seed730000+k*10+user, local retained pools excluding the entire45-request union from the outset, sample weights, LR schedule and uniform/priority sampler as the respective FL policy. No FU gradients because deleted samples never entered training. Reset priority cache at round200 as in the deployed continuation. Exact communication is the retraining-oracle convention.',
  limitations='Single seed and common data split. Original stored reference uses its own prespecified batch records; new references additionally match the mission FL sampler. All reference scores reported alongside the old common-reference values, not renamed as a repaired model.')
 write(directory/'protocol.json',protocol)
 q.infra.lib.infra.extra.Data=q.infra.lib.infra.VideoData
 event,base=q.infra.lib.infra.prior.load_base();base.repeat.clone_function=q.infra.cached_clone
 import torch,accounting
 from prepare_extra import native_engine,native_base,Data
 from train import update_theta
 assert torch.cuda.is_available();torch.set_num_threads(1)
 q.infra.lib.infra.ORIGINAL_INSTRUMENTED=accounting.instrumented_engine
 with (ROOT/'results/overnight_energy_v2/association_objective_gpu.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);write(directory/'started.json',dict(unix=time.time()))
  initial=report.variant_source(ds,'gate_select16');plan=read(initial.parent/'plan.json')
  records=read((initial.parent/'reference200').resolve().parent/'reference_records.json')
  if ds=='AirScape':base=native_base(base,1)
  pool=[sorted(set(part)-set(plan['forgotten_images'])) for part in plan['partitions']];total=sum(map(len,pool))
  queue=base.RoundQueue(read(initial.parent/'requests.json'));queue.items={r['id']:r for r in read(initial/'request_records.json')}
  mission=read(initial/'mission_timeline.json')['service_end_s']
  n=read(continuation.OUT/(ds.lower()+'_seed0')/'gate_select16/finished.json')['extra_FL']
  candidates={200:initial/'final.pt',200+n:continuation.OUT/(ds.lower()+'_seed0')/'gate_select16/final.pt'}
  rows=[]
  for policy in protocol['variants']:
   q.CONFIG=q.VARIANTS['control'];data=Data(ds) if ds=='AirScape' else event.recover.Data(ds)
   Engine=native_engine(base,data.caption_count) if ds=='AirScape' else base.Engine
   payload=torch.load(initial.parent/'feature_cache/features.pt',map_location='cpu',weights_only=False)
   engine=q.engine_wrapper(Engine)(data,plan,payload);theta=engine.theta0.clone()
   horizon=200 if policy=='original_reference_reproduction' else 200+n
   for k in range(1,horizon+1):
    if policy=='original_reference_reproduction' or k<=20:
     engine.plan['learning_rate']=base.core.lr_at(k);l,_,_=engine.updates(theta,records[k-1],k,'retrain',[[] for _ in range(10)])
    else:
     if k==201:engine.selection_losses={};engine.selection_records=[]
     q.CONFIG=q.VARIANTS['select16' if policy=='selective_sampler' else 'control']
     lr=base.core.lr_at(k) if k<=200 else continuation.LR;l=np.zeros((10,len(theta)-1))
     for user,part in enumerate(pool):
      batch=q.select_batch(engine,theta,part,k,user) if policy=='selective_sampler' else np.random.default_rng(730000+k*10+user).choice(part,min(16,len(part)),replace=False).astype(int).tolist()
      assert not set(batch)&set(plan['forgotten_images'])
      g,_=engine.gradient(theta,batch,k);l[user]=(-lr*len(part)/total*g[:-1]).cpu().numpy()
    theta=update_theta(theta,l.sum(0))
    if k in candidates:
     reference,prob=engine.evaluate(theta,None)
     candidate=base.core.load_theta(candidates[k],engine.device);e=base.clip.stream.evaluation(engine,candidate,prob,queue,mission+1000)
     row=dict(dataset=ds,reference_policy=policy,FL_round=k,retained_mR_percent=100*e['retained']['mR'],retained_loss=e['retained']['loss'],
      reference_mR_percent=100*reference['retained']['mR'],reference_loss=reference['retained']['loss'],forget_KL=e['forget']['KL_retrain_to_candidate'],worst_request_KL=e['worst_request_KL'])
     if policy=='original_reference_reproduction':
      original=base.core.load_theta(initial.parent/'reference200/final.pt',engine.device)
      row['reference_max_parameter_difference']=float((theta-original).abs().max());assert torch.equal(theta,original),row
     base.core.save_theta(directory/policy/f'round_{k}.pt',theta,k);rows.append(row)
     print(row,flush=True)
   del engine,data,payload;torch.cuda.empty_cache()
  write(directory/'finished.json',dict(passed=True,original_reference_preserved=True,reporting_only=True,rows=rows))


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=q.infra.DATASETS,required=True);run(p.parse_args().dataset)
