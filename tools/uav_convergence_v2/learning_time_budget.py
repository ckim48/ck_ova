"""Spend saved service time on retained-only FL after the common workload.

This is a continuation experiment, not a new exogenous online-arrival benchmark.
All45 deletion requests must first finish. Same additional FL learning-rate rule
for every method. Compare last completed models at common active-time budgets.
"""
import argparse
import concurrent.futures
import copy
import fcntl
import math
import multiprocessing
import os
from pathlib import Path
import time
import numpy as np
import learning_quality_v3 as quality
import paper_uav_charts as prior
import paper_uav_radio as radio

ROOT=quality.ROOT;OUT=ROOT/'results/learning_time_budget_v1_20261028'
read,write,sha=quality.read,quality.write,quality.sha
METHODS={'current':'Current Proposed','gate_select16':'KL gate + FL selection','iscc':'ISCC-P2','afl':'AFL-OTA','salun':'F-SalUn'}
BUDGETS=[150.,200.,250.,300.,350.,400.]
LR=.0035


def source(ds,m):
 if m in ['current','gate_select16','salun']:
  v='control' if m=='current' else m;name=quality.DIGITAL if m=='salun' else quality.PROPOSED
  origin=ROOT/'results/learning_selection_fix_v2_20261028' if m=='gate_select16' else quality.OUT
  return origin/v/'cases/baseline'/(ds.lower()+'_seed0')/name
 return prior.source('baseline',ds,'iscc_p2_gaft' if m=='iscc' else 'uav_afl_ota_gaft')


def register():
 path=OUT/'protocol.json'
 if path.exists():
  p=read(path)
  for n,d in p['source_sha256'].items():assert sha(ROOT/n)==d,n
  return p
 sources=dict(quality.register()['source_sha256'])
 for name in ['learning_selection_fix.py','learning_selection_fix_v2.py']:
  file=Path(__file__).with_name(name);sources[str(file.relative_to(ROOT))]=sha(file)
 sources[str(Path(__file__).relative_to(ROOT))]=sha(Path(__file__))
 p=dict(registered_unix=time.time(),methods=METHODS,datasets=quality.infra.DATASETS,budgets_s=BUDGETS,extra_FL_learning_rate=LR,
  policy='Continue FL after200 rounds and all45 requests/225FU cycles complete, using only the retained training pool, until the first completed update beyond400 active-service seconds. At each budget, report the last actually completed checkpoint; no interpolation or invented extra updates. The original200-round cosine schedule ends at zero, so every method explicitly starts the same constant0.0035 continuation schedule.',
  scope='Same five-region original route and round-triggered deletion workload. Continuation stays in final region before return; unchanged final-site channels. This measures usefulness of saved active service time after processing a fixed workload. It is not an exogenous-time arrival/load-capacity study, and does not mean extra FU cycles necessarily improve mR.',
  metrics='Retained mR/loss after completed updates; cumulative energy to those actual completion times. Fixed reference200 KL is diagnostic only after round200, not equivalence to a round-matched retraining oracle. Fixed-time energy comparisons omit in-flight partial transactions and are not claimed; use completed time-to-target energy.',
  costs='All adapter work, selection overhead, radio uplink/broadcast/circuit and active hover charged. No travel/collection/preflight energy. Continuation selection caches start empty consistently at the new phase. No extra deletion data is trained on.',
  validity='Single-seed developmental continuation experiment. No validation/reference input into training or stopping except prescribed time budget. All outcomes retained.',source_sha256=sources)
 write(path,p);return p


def run(ds):
 register();quality.infra.lib.infra.extra.Data=quality.infra.lib.infra.VideoData
 event,base=quality.infra.lib.infra.prior.load_base();base.repeat.clone_function=quality.infra.cached_clone
 import torch
 import accounting,event_energy,heuristic
 from prepare_extra import native_engine,native_base,Data
 from train import update_theta
 assert torch.cuda.is_available();torch.set_num_threads(1)
 if ds=='AirScape':base=native_base(base,1)
 lock=(ROOT/'results/overnight_energy_v2/association_objective_gpu.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
 physical=event_energy.energy
 quality.infra.lib.infra.ORIGINAL_INSTRUMENTED=accounting.instrumented_engine
 p=base.base.parent.common.base;p.CAP=.1
 p.POOL=concurrent.futures.ProcessPoolExecutor(max_workers=8,mp_context=multiprocessing.get_context('spawn'),initializer=quality.infra.lib.infra.pool_init,initargs=(.1,))
 try:
  for m in METHODS:
   ad=OUT/(ds.lower()+'_seed0')/m
   if (ad/'finished.json').exists():continue
   assert not (ad/'started.json').exists(),'Preserve incomplete continuation'
   initial=source(ds,m);assert read(initial/'outcome.json')['status']=='completed'
   settings=read(initial.parents[1]/'settings.json');physical.PARAMETERS=settings['energy_model'];quality.infra.PHYSICAL=physical;quality.infra.LOSS=0.
   physical.communication=quality.infra.communication
   plan=read(initial.parent/'plan.json');config=plan['config'];geo=read(initial.parents[1]/'mission_spec.json');deployment=read(initial.parents[1]/'deployment.json')
   channels=np.load(ROOT/'results/iot_uav_deployment_v3_global_iot/channels_by_site.npy')[4]
   quality.CONFIG=quality.VARIANTS['gate_select16' if m=='gate_select16' else 'control'];quality.ACTIVE=m
   data=Data(ds) if ds=='AirScape' else event.recover.Data(ds)
   Engine=native_engine(base,data.caption_count) if ds=='AirScape' else base.Engine
   payload=torch.load(initial.parent/'feature_cache/features.pt',map_location='cpu',weights_only=False)
   engine=quality.engine_wrapper(Engine)(data,plan,payload)
   theta=base.core.load_theta(initial/'final.pt',engine.device);warm=base.core.load_theta(initial.parent/'initial20/final.pt',engine.device);engine.set_teacher(warm)
   ref=base.core.load_theta(initial.parent/'reference200/final.pt',engine.device);_,refprob=engine.evaluate(ref,None)
   q=base.RoundQueue(read(initial.parent/'requests.json'));q.items={r['id']:r for r in read(initial/'request_records.json')}
   timeline=read(initial/'mission_timeline.json');mission=timeline['service_end_s'];active=sum(x['service_duration_s'] for x in timeline['sites'])
   assert q.metrics(mission)['procedure_completed']==45
   # Recompute the starting active ledger, including older records with broader total fields.
   physical_p=physical.PARAMETERS;start_compute=start_comm=0.
   for f in (initial/'work_attempts').glob('*.json'):start_compute+=float(physical.compute_cost(read(f)['compute']['device_macs'])[0].sum())
   for f in (initial/'transactions').glob('*.json'):
    r=read(f)['radio'];start_comm+=quality.infra.communication(r)[0];start_compute+=physical_p['communication']['head_processing_w']*r['uplink_s']
   energy=start_compute+start_comm+10*float(physical.propulsion_power(0))*active
   pools=base.clip.stream.retained_pools(plan,q,mission);deleted=set(plan['forgotten_images']);assert not any(set(x)&deleted for x in pools)
   write(ad/'started.json',dict(unix=time.time(),method=m,source=str(initial.relative_to(ROOT)),source_checkpoint_sha256=sha(initial/'final.pt'),active_start_s=active,energy_start_j=energy))
   evaluations=[dict(extra_FL=0,FL_round=200,active_s=active,energy_j=energy,evaluation=read(initial/'service_completion.json')['evaluation'])]
   extra=0;cached={};current_cost=np.zeros(10);job_no=0;all_jobs=[]
   def jobs(users,at):
    nonlocal job_no,current_cost
    engine.start_work();made=[]
    for u in users:
     k=201+extra;pool=pools[u];batch=np.random.default_rng(730000+k*10+u).choice(pool,min(16,len(pool)),replace=False).astype(int).tolist()
     g,_=engine.gradient(theta,batch,k);made.append((u,g[:-1].cpu().numpy().copy(),batch))
    cost=engine.finish_work();macs=np.asarray(cost['device_macs']);current_cost+=macs;_,seconds=physical.compute_cost(macs)
    for u,g,batch in made:
     job_no+=1;j=dict(job=job_no,user=u,version=extra,start_s=at,ready_s=at+float(seconds[u]),MACs=float(macs[u]),samples=batch)
     cached[u]=dict(gradient=g,**j);all_jobs.append(j);write(ad/'jobs'/f'{job_no:05d}.json',j)
   while active<max(BUDGETS) and extra<1000:
    start=active;current_cost=np.zeros(10);selection=[];engine_work=None
    if m in ['current','gate_select16']:
     engine.start_work();updates=np.zeros((10,len(theta)-1));total=sum(map(len,pools));k=201+extra
     for u,pool in enumerate(pools):
      batch=quality.select_batch(engine,theta,pool,k,u) if m=='gate_select16' else np.random.default_rng(730000+k*10+u).choice(pool,min(16,len(pool)),replace=False).astype(int).tolist()
      assert not set(batch)&deleted;g,_=engine.gradient(theta,batch,k);updates[u]=-LR*len(pool)/total*g[:-1].cpu().numpy()
     engine_work=engine.finish_work();current_cost+=np.asarray(engine_work['device_macs']);_,ct=physical.compute_cost(current_cost)
     vector,rm=base.base.parent.guard.analog(updates,np.zeros_like(updates),channels,config,900000+extra)
     assert vector is not None,rm
     compute_upload=float(ct.max())+rm['uplink_s'];rm['broadcast_required']=bool(np.any(vector))
    elif m=='afl':
     if not cached:jobs(list(range(10)),active)
     ready=max(active,min(j['ready_s'] for j in cached.values()));selection=[u for u,j in cached.items() if j['ready_s']<=ready+1e-12]
     updates=np.zeros((10,len(theta)-1));versions={u:cached[u]['version'] for u in selection}
     selected_jobs=[cached[u]['job'] for u in selection]
     for u in selection:updates[u]=-LR*cached[u]['gradient']/len(selection)
     vector,rm=radio.ota(updates,channels,config,deployment['sites'][4]['head_zero_based'],900000+extra)
     rm.update(selected_job_ids=selected_jobs,staleness={u:extra-v for u,v in versions.items()})
     compute_upload=ready-active+rm['uplink_s']
     for u in selection:del cached[u]
    else:
     jobs(list(range(10)),active);_,ct=physical.compute_cost(current_cost)
     updates=np.stack([-LR*cached[u]['gradient']/10 for u in range(10)])
     vector,rm=radio.digital(updates,channels,config,deployment['sites'][4]['head_zero_based'],ct);compute_upload=rm['compute_upload_finish_s'];cached.clear()
    quality.infra.attach_arq(rm,extra);comm,down=quality.infra.communication(rm);active+=compute_upload+down;extra+=1
    theta=update_theta(theta,vector)
    if m=='afl' and active<max(BUDGETS):jobs(selection,active)
    comp=float(physical.compute_cost(current_cost)[0].sum())+physical_p['communication']['head_processing_w']*rm['uplink_s']
    hover=10*float(physical.propulsion_power(0))*(active-start);energy+=comp+comm+hover
    point=dict(extra_FL=extra,FL_round=200+extra,active_s=active,energy_j=energy)
    record=dict(**point,start_s=start,radio=rm,compute_j=comp,communication_j=comm,hover_j=hover,device_macs=current_cost.tolist(),
     selection=[] if engine_work is None else engine_work['FL_sample_selection'],timing=dict(compute_upload_s=compute_upload,downlink_s=down,duration_s=active-start))
    write(ad/'transactions'/f'{extra:04d}.json',record)
    # Every completed model is evaluated, so common-time comparisons do not depend on sparse checkpoints.
    ev=base.clip.stream.evaluation(engine,theta,refprob,q,mission+active-evaluations[0]['active_s'])
    evaluations.append(dict(**point,evaluation=ev));write(ad/'evaluations'/f'{extra:04d}.json',evaluations[-1])
   assert extra<1000 and active>=max(BUDGETS)
   base.core.save_theta(ad/'final.pt',theta,200+extra);write(ad/'curve.json',evaluations)
   write(ad/'finished.json',dict(passed=True,extra_FL=extra,active_end_s=active,energy_end_j=energy,deleted_samples_never_trained=True,
    model_at_budget={str(int(b)):next((x for x in reversed(evaluations) if x['active_s']<=b),None) for b in BUDGETS}))
   print(dict(dataset=ds,method=m,extra_FL=extra,active_s=active),flush=True)
   del engine,data,payload,refprob;torch.cuda.empty_cache()
 finally:p.close()


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=quality.infra.DATASETS,required=True);a=p.parse_args();run(a.dataset)
