"""Common and learning-policy-matched retraining references for every method.

This is evaluation only. Compare unchanged old models as well as real repairs;
never use a different reference to label an unchanged model as a new algorithm.
"""
import argparse
import fcntl
from pathlib import Path
import time
import numpy as np
import pks_balance_v1 as study
import learning_time_budget as old_time
import learning_quality_report as old_report

ROOT=study.ROOT;OUT=ROOT/'results/pks_evaluation_v1_20261028'
read,write,sha=study.read,study.write,study.sha
METHODS={'current':'Proposed','pks':'P-KS','iscc':'ISCC-P2','afl':'AFL-OTA','salun':'F-SalUn',
 **{v:s['label'] for v,s in study.VARIANTS.items() if v!='control'}}
POLICY={'current':'uniform','pks':'priority','iscc':'equal','afl':'equal','salun':'equal',
 'mix25_gate':'mix25','mix50_gate':'mix50','mix50_adaptive':'mix50',
 'pk_uniform':'priority_then_uniform','pls_uniform':'priority_then_uniform'}


def sources(ds,m):
 if m in study.VARIANTS:return study.initial_source(ds,m),study.CONT/(ds.lower()+'_seed0')/m
 name='gate_select16' if m=='pks' else m
 return old_time.source(ds,name),old_time.OUT/(ds.lower()+'_seed0')/name


def run(ds):
 directory=OUT/(ds.lower()+'_seed0')
 if (directory/'finished.json').exists():return
 assert not (directory/'started.json').exists(),'Preserve incomplete evaluation'
 candidates={};input_sha={}
 for m in METHODS:
  initial,terminal=sources(ds,m);finished=read(terminal/'finished.json');assert finished['passed']
  for stage,path,k in [('round200',initial/'final.pt',200),('terminal',terminal/'final.pt',200+finished['extra_FL'])]:
   input_sha[str(path.relative_to(ROOT))]=sha(path)
   candidates[(m,stage)]=dict(method=m,stage=stage,path=path,FL_round=k,
    active_s=sum(r['service_duration_s'] for r in read(initial/'mission_timeline.json')['sites']) if stage=='round200' else finished['active_end_s'])
 sources_digest=dict(study.register()['source_sha256']);sources_digest[str(Path(__file__).relative_to(ROOT))]=sha(Path(__file__))
 protocol=dict(registered_unix=time.time(),methods=METHODS,learning_policy=POLICY,source_sha256=sources_digest,input_sha256=input_sha,
  common_reference='Preserve the exact original reference200; extend it uniformly with the declared0.0035 LR to each final FL count, exactly as the previous report.',
  matched_reference='Start from theta0, replay original retained reference records1..20, then apply each method FL sampling, weighting and step-size rule on the full retained pool with the entire forget union excluded. Exact aggregation, no FU updates because forgotten samples never entered training. Priority caches reset after200, matching the actual continuation. Equal-client references for all three external transfers; no privileged P-KS-only reference change.',
  comparison='Both common and matched KL are published side by side for every model. This explicitly changes an evaluation counterfactual, not the trained candidate. The new bounded-gradient arms and phase changes are actual training interventions and remain separately named.',
  scope='Original reporting-only warmup reference is exact, whereas actual warm20 is shared noisy training including forgotten data. This is the same retrain convention as before. Single-seed development; no deletion certificate or distributional indistinguishability guarantee. Terminal models are first completed updates beyond400s; actual timestamps are retained.')
 write(directory/'protocol.json',protocol)
 q=study.q;q.infra.lib.infra.extra.Data=q.infra.lib.infra.VideoData
 event,base=q.infra.lib.infra.prior.load_base();base.repeat.clone_function=q.infra.cached_clone
 import torch,accounting
 from prepare_extra import native_engine,native_base,Data
 from train import update_theta
 assert torch.cuda.is_available();torch.set_num_threads(1)
 q.infra.lib.infra.ORIGINAL_INSTRUMENTED=accounting.instrumented_engine
 with (ROOT/'results/overnight_energy_v2/association_objective_gpu.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);write(directory/'started.json',dict(unix=time.time()))
  initial=old_time.source(ds,'current');plan=read(initial.parent/'plan.json')
  records=read((initial.parent/'reference200').resolve().parent/'reference_records.json')
  if ds=='AirScape':base=native_base(base,1)
  pool=[sorted(set(part)-set(plan['forgotten_images'])) for part in plan['partitions']];total=sum(map(len,pool))
  queue=base.RoundQueue(read(initial.parent/'requests.json'));queue.items={r['id']:r for r in read(initial/'request_records.json')};mission=read(initial/'mission_timeline.json')['service_end_s']
  rows={key:dict(dataset=ds,method=m,label=METHODS[m],stage=stage,FL_round=item['FL_round'],active_s=item['active_s'],policy=POLICY[m]) for key,item in candidates.items() for m,stage in [key]}
  horizons={policy:max(item['FL_round'] for (m,s),item in candidates.items() if POLICY[m]==policy) for policy in set(POLICY.values())}
  horizons={'common':max(x['FL_round'] for x in candidates.values()),**horizons}
  for policy,horizon in horizons.items():
   data=Data(ds) if ds=='AirScape' else event.recover.Data(ds)
   Engine=native_engine(base,data.caption_count) if ds=='AirScape' else base.Engine
   payload=torch.load(initial.parent/'feature_cache/features.pt',map_location='cpu',weights_only=False)
   study.initialize_variant('pk_uniform');engine=study.engine_wrapper(Engine)(data,plan,payload)
   theta=base.core.load_theta(initial.parent/'reference200/final.pt',engine.device) if policy=='common' else engine.theta0.clone()
   start=200 if policy=='common' else 1
   for k in range(start,horizon+1):
    if policy=='common' and k==200:pass
    elif k<=20:
     engine.plan['learning_rate']=base.core.lr_at(k);l,_,_=engine.updates(theta,records[k-1],k,'retrain',[[] for _ in range(10)]);theta=update_theta(theta,l.sum(0))
    else:
     if k==201:engine.selection_losses={};engine.selection_records=[];engine.pending_priority={}
     variant='control' if policy in ['priority','priority_then_uniform'] and (k<=200 or policy=='priority') else 'mix25_gate' if policy=='mix25' else 'mix50_gate' if policy=='mix50' else 'pk_uniform'
     study.initialize_variant(variant)
     lr=base.core.lr_at(k) if k<=200 else old_time.LR;l=np.zeros((10,len(theta)-1))
     for user,part in enumerate(pool):
      batch=study.choose_batch(engine,theta,part,k,user,np.random.default_rng(730000+k*10+user))
      assert not set(batch)&set(plan['forgotten_images'])
      g,_=engine.gradient(theta,batch,k);weight=1/10 if policy=='equal' else len(part)/total
      l[user]=(-lr*weight*g[:-1]).cpu().numpy()
     theta=update_theta(theta,l.sum(0))
    relevant=[(key,item) for key,item in candidates.items() if item['FL_round']==k and (policy=='common' or POLICY[key[0]]==policy)]
    if not relevant:continue
    reference,prob=engine.evaluate(theta,None);file=directory/'references'/policy/f'{k}.pt';base.core.save_theta(file,theta,k)
    if policy in ['uniform','priority']:
     old_policy='uniform_sampler' if policy=='uniform' else 'selective_sampler'
     checked=ROOT/'results/pks_reference_audit_v1_20261028'/(ds.lower()+'_seed0')/old_policy/f'round_{k}.pt'
     if checked.exists():assert torch.equal(theta,base.core.load_theta(checked,engine.device)),(ds,policy,k)
    for key,item in relevant:
     candidate=base.core.load_theta(item['path'],engine.device);e=base.clip.stream.evaluation(engine,candidate,prob,queue,mission+1000)
     row=rows[key];prefix='common' if policy=='common' else 'matched'
     if 'retained_mR_percent' in row:assert row['retained_mR_percent']==100*e['retained']['mR'] and row['retained_loss']==e['retained']['loss']
     row.update(retained_mR_percent=100*e['retained']['mR'],retained_loss=e['retained']['loss'])
     row[prefix+'_KL']=e['forget']['KL_retrain_to_candidate'];row[prefix+'_worst_KL']=e['worst_request_KL']
     row[prefix+'_reference_mR_percent']=100*reference['retained']['mR'];row[prefix+'_reference_sha256']=sha(file)
    if k%100==0 or k==horizon:print(dict(dataset=ds,policy=policy,round=k),flush=True)
   del engine,data,payload;torch.cuda.empty_cache()
  for row in rows.values():
   assert all(key in row for key in ['common_KL','matched_KL','common_worst_KL','matched_worst_KL'])
   row.update(common_mean_pass=row['common_KL']<=.01,common_all_requests_pass=row['common_worst_KL']<=.01,
    matched_mean_pass=row['matched_KL']<=.01,matched_all_requests_pass=row['matched_worst_KL']<=.01)
  result=dict(passed=True,reporting_only=True,all_methods_have_both_references=True,rows=list(rows.values()))
  write(directory/'finished.json',result);print(result,flush=True)


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=study.q.infra.DATASETS,required=True);run(p.parse_args().dataset)
