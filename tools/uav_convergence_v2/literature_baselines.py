"""Literature-derived CLIP FU baselines and five resource/workload sensitivities.

FT/GA/GAFT objective families follow the SLUG authors' CLIP comparison code.
These are federated adapter adaptations, not reproductions of SLUG or FedCLIP.
"""
import argparse
import collections
import copy
import inspect
import linecache
import math
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import time
from types import SimpleNamespace
import numpy as np
import adaptive_fu as adaptive
from adaptive_fu import ROOT, read, write, sha
import fu_joint_optimization as infra

OUT=ROOT/'results/literature_baselines_v1_20261028'
PROPOSED=adaptive.NEW
EXTERNAL={'fedavg_clip_ft':'FA-FT','fedavg_clip_ga':'FA-GA','fedavg_clip_gaft':'FA-GAF'}
METHODS={PROPOSED:'Proposed',**EXTERNAL}
DATASETS=adaptive.DATASETS
AXES={
 'bandwidth':dict(label='Uplink bandwidth (MHz)',values=[1,2,3,4,5],base=1,kind='training'),
 'downlink':dict(label='Downlink rate (Mbit/s)',values=[.25,.5,1,2,4],base=1,kind='trace replay'),
 'frequency':dict(label='CPU frequency (GHz)',values=[.75,1,1.5,2,3],base=1.5,kind='trace replay'),
 'fu_steps':dict(label='FU cycles per request',values=[1,3,5,7,9],base=5,kind='training'),
 'batch':dict(label='FL batch-size cap',values=[8,16,32,64,128],base=16,kind='training')}
ACTIVE=PROPOSED
BATCH=16


def case_id(axis,value):
 return 'baseline' if value==AXES[axis]['base'] else axis+'_'+str(value).replace('.','p')


def cases():
 return [dict(id='baseline',axis=None,value=None)]+[dict(id=case_id(axis,v),axis=axis,value=v) for axis,a in AXES.items() for v in a['values'] if v!=a['base'] and a['kind']=='training']


def state(dest=None,**kw):write((Path(dest) if dest else OUT)/'status.json',dict(unix=time.time(),pid=os.getpid(),**kw))


def configure_case(case,*args):
 import common
 return common.configure(case,*args)


def baseline_gradient(engine,theta,fids,rids,step,teacher):
 import torch
 import torch.nn.functional as F
 lr=engine.literature_lr;value=theta.detach().requires_grad_(True)
 forget=retain=None;extra=0
 if ACTIVE!='fedavg_clip_ft':
  cids=[c for i in fids for c in engine.data.caption_ids[i]]
  owner=torch.tensor([j for j,i in enumerate(fids) for _ in engine.data.caption_ids[i]],device=theta.device)
  im,tx,_=engine.features(value,fids,cids)
  # Ascent on (1-cos) is descent on cos; all native positive captions included.
  forget=(im[owner]*tx).sum(-1).mean()
  gf=torch.autograd.grad(forget,value)[0].detach();gf[-1]=0
  extra+=len(cids)*(3*im.shape[1]+4)
 else:gf=torch.zeros_like(theta)
 if ACTIVE!='fedavg_clip_ga':
  # GAFT performs forget SGD followed by retain SGD, then communicates net delta.
  current=(theta-lr*gf).detach() if ACTIVE=='fedavg_clip_gaft' else theta.detach()
  variable=current.requires_grad_(True);rcids=engine.data.training_caption_ids(rids,21+step)
  ri,rt,scale=engine.features(variable,rids,rcids)
  logits=scale.exp()*(ri@rt.T);target=torch.arange(len(rids),device=theta.device)
  retain=.5*(F.cross_entropy(logits,target)+F.cross_entropy(logits.T,target))
  gr=torch.autograd.grad(retain,variable)[0].detach();gr[-1]=0
  extra+=len(rids)*len(rcids)*(3*ri.shape[1]+16)
 else:gr=torch.zeros_like(theta)
 g=gf+gr;g[-1]=0;assert torch.isfinite(g).all()
 infra.charge(engine,fids,int(extra))
 return g,dict(FU_variant=ACTIVE,forget_cosine=None if forget is None else float(forget.detach()),
  retained_CE=None if retain is None else float(retain.detach()),sequential_GA_then_FT=ACTIVE=='fedavg_clip_gaft',
  optimizer_substeps=2 if ACTIVE=='fedavg_clip_gaft' else 1,FU_gradient_norm=float(g.norm()),
  extra_loss_MAC_equivalents=int(extra),images=len(fids),retained_images=rids,
  clean_proxy_used=False,request_teacher_used=False,adaptive_retention_used=False,
  objective_source='SLUG official CLIP comparison FT/GA/GAFT families; federated rank10 SGD adaptation')


def vectors(base,objective):
 if ACTIVE==PROPOSED:
  adaptive.ACTIVE_METHOD=PROPOSED
  original=adaptive.vectors(base,objective)
 else:
  original=base.control.synchronized_vectors('cliperase')
 source=textwrap.dedent(inspect.getsource(original))
 old='batch=np.random.default_rng(730000+k*10+user).choice(pool,min(16,len(pool)),replace=False)'
 new='batch=np.random.default_rng(730000+k*10+user).choice(pool,min(FL_BATCH,len(pool)),replace=False)'
 assert source.count(old)==1;source=source.replace(old,new)
 if ACTIVE!=PROPOSED:
  source=source.replace('if rid not in teachers:teachers[rid]=core.Teacher(engine,theta)','engine.literature_lr=core.lr_at(21+fl)')
  source=source.replace('request_gradient(engine,theta,fids,rids,step,teachers[rid])','request_gradient(engine,theta,fids,rids,step,None)')
 namespace=dict(original.__globals__,FL_BATCH=BATCH)
 if ACTIVE!=PROPOSED:namespace['request_gradient']=baseline_gradient
 filename='/tmp/literature_vectors_'+__import__('hashlib').sha256(source.encode()).hexdigest()+'.py'
 linecache.cache[filename]=(len(source),None,source.splitlines(True),filename)
 exec(compile(source,filename,'exec'),namespace)
 return namespace[original.__name__]


def make_arm(base,deployment,radios,cap):
 import event_arm
 control=SimpleNamespace(**base.control.__dict__);control.synchronized_vectors=lambda objective:vectors(base,objective)
 adapted=SimpleNamespace(**dict(base.__dict__,control=control))
 edits=[
  ("radio=base.parent.guard if name=='proposed_fu_guard_cliperase' else base.SequentialRadio if name=='sequential_digital_periodic' else base.parent.common","radio=Radios[name]",1),
  ("if name!='sequential_digital_periodic':assert rm['energy_bound_j']<=.1*(1+2e-6)","if name not in EXTERNAL:assert rm['energy_bound_j']<=NETWORK_CAP*(1+2e-6)",1),
  ("start=t;timing=clock.transaction_timing(compute,rm);end=t+timing['duration_s']", "rm['broadcast_required']=bool(np.any(vector != 0))\n  start=t;timing=clock.transaction_timing(compute,rm);end=t+timing['duration_s']",1)]
 maker=base.repeat.clone_function(event_arm.make_arm,[
  ('return base.repeat.clone_function','edits.extend(EXTRA_EDITS)\n    return base.repeat.clone_function',1),
  ('deployment=deployment,state=state))','deployment=deployment,state=state,Radios=RADIOS,NETWORK_CAP=CAP,EXTERNAL=EXTERNAL))',1)],
  dict(EXTRA_EDITS=edits,RADIOS=radios,CAP=cap,EXTERNAL=list(EXTERNAL)))
 return maker(adapted,deployment,state)


def options(base,method):
 if method==PROPOSED:return dict(base.base.ARMS[adaptive.PROPOSED],adaptive_controller=adaptive.CONTROLLER)
 return dict(base.base.ARMS['sequential_digital_periodic'],objective='vlm_literature_baseline',
  literature_method=EXTERNAL[method],FL='FedAvg with one local SGD update',transport='Serial Q6 OFDMA; existing digital resource solver',
  adaptation='Frozen RemoteCLIP, trainable rank10 adapters, SGD and common FL cosine LR, native captions; not original full-model AdamW or centralized epochs.',
  FU='retain contrastive SGD' if method.endswith('_ft') else 'cosine-forget SGD' if method.endswith('_ga') else 'cosine-forget SGD followed by retained contrastive SGD')


def reused(case,method):
 return method==PROPOSED and case in ['baseline','bandwidth_2','bandwidth_4']


def register():
 path=OUT/'protocol.json'
 if path.exists():
  p=read(path)
  for group in ['source_sha256','input_sha256']:
   for name,d in p[group].items():assert sha(ROOT/name)==d,name
  return p
 previous=adaptive.register()
 sources=dict(previous['source_sha256']);sources[str(Path(__file__).relative_to(ROOT))]=sha(Path(__file__))
 inputs=dict(previous['input_sha256'])
 for ds in DATASETS:
  inputs[str((adaptive.OUT/'cases/baseline'/(ds.lower()+'_seed0')/PROPOSED/'outcome.json').relative_to(ROOT))]=sha(adaptive.OUT/'cases/baseline'/(ds.lower()+'_seed0')/PROPOSED/'outcome.json')
 evidence=read(Path('/tmp/ova_literature_baselines/sources.json'))
 p=dict(registered_unix=time.time(),methods=METHODS,datasets=DATASETS,axes=AXES,training_cases=cases(),sampling_seed=0,
  actual_training_runs=98,reused_training_arms=6,replayed_resource_conditions=64,
  objective_families='FT: retained CE; GA: minimize positive-pair cosine, equivalent to ascent on1-cos; GAFT: one local GA then one FT step per cycle. FA prefix denotes FedAvg one local SGD step.',
  adaptation='CLIP baseline objective families from SLUG ICML2025 official code, adapted to federated rank10 RemoteCLIP video retrieval, native captions, fixed warm20 and common SGD cosine schedule. Not faithful full-model AdamW reproduction, not SLUG or FedCLIP itself. GAFT has2 optimizer substeps per cycle and both are charged; all methods use same cycles, not equal optimizer substeps.',
  transport='External baselines use serial Q6 OFDMA and the existing digital head/resource solver. No Proposed AirComp/FU guard, adaptive loss, proxy or CM. Digital soft network cap retained with every overrun logged; per-user power/energy/deadline/error constraints unchanged. Full-system comparison, not isolated learning causality.',
  timing_replay='Downlink and CPU-frequency sweeps recompute recorded MAC/radio costs and replay the actual scheduler, requiring exact transaction-kind/request/FL ordering. No new learning claim. Any schedule mismatch must trigger a fresh training run, not use incorrect old quality values.',
  accounting='Same active FL/FU communication+compute+hover scope. Executed adapter passes and additional objective arithmetic charged. Shared FL contrastive scalar operations retain existing model limitation. Resource-optimizer CPU time is not hardware-calibrated.',
  fairness='All curves, infeasible points and utility/forgetting tradeoffs retained. No rank-based parameter selection or weakened baseline. Same data, feature cache, warm20, requests,200FL. FU sweep changes prescribed cycles for every method equally; batch sweep changes FL cap for every method equally. Single development seed, not statistical superiority.',
  context_sources=[dict(title='SLUG ICML2025',url='https://proceedings.mlr.press/v267/cai25f.html'),
   dict(title='CLIPErase ACL2025',url='https://aclanthology.org/2025.acl-long.1469/'),
   dict(title='FedCLIP IEEE Data Engineering Bulletin2023',url='https://arxiv.org/abs/2302.13485'),
   dict(title='UAV-Enabled Asynchronous Federated Learning TWC2025',url='https://doi.org/10.1109/TWC.2024.3520501'),
   dict(title='CLIP-AVC IEEE GRSL2025',url='https://doi.org/10.1109/LGRS.2025.3532987')],
  official_baseline_code=evidence,source_sha256=sources,input_sha256=inputs,
  publication='New final_final_charts. Raw data, logs and checkpoints remain local; publish code, plots, plot arrays and summaries only.')
 write(path,p);return p


def run_case(cid,method):
 global ACTIVE,BATCH
 register();case=next(c for c in cases() if c['id']==cid);ACTIVE=method;BATCH=case['value'] if case['axis']=='batch' else 16
 adaptive.ACTIVE_METHOD=PROPOSED if method==PROPOSED else adaptive.PROPOSED
 infra.OUT=OUT;adaptive.TASKROOT=OUT
 adaptive.link(adaptive.SOURCE/'prepared/airscape_seed0',OUT/'prepared/airscape_seed0')
 _,base=infra.prior.load_base()
 import worker,heuristic,accounting,event_energy,final_broadcast_fix as fix
 infra.extra.Data=infra.VideoData;heuristic.pool_init=infra.pool_init
 infra.ORIGINAL_INSTRUMENTED=accounting.instrumented_engine;accounting.instrumented_engine=infra.controlled_engine
 fix.ENERGY=event_energy.energy;fix.ORIGINAL_COMMUNICATION=fix.ENERGY.communication;fix.ENERGY.communication=fix.communication
 valid=base.repeat.clone_function(worker.validate,[("if ad.name!='sequential_digital_periodic':",'if ad.name not in EXTERNAL:',1)],dict(EXTERNAL=list(EXTERNAL)))
 run=base.repeat.clone_function(worker.run,[
  ('arm=make_arm(base,deployment,radios,cap);results=[]','radios[PROPOSED]=adaptive_radio(base)\n    for m in EXTERNAL:radios[m]=overrun_radio.build(base)\n    arm=make_arm(base,deployment,radios,cap);results=[]',1),
  ("opt=base.base.ARMS['proposed_fu_guard_cliperase' if method=='heuristic_ova' else method]",'opt=options(base,method)',1)],
  dict(OUT=OUT,EXTRA_DATASETS=['AirScape'],PROPOSED=PROPOSED,EXTERNAL=list(EXTERNAL),adaptive_radio=adaptive.radio,
   make_arm=make_arm,validate=valid,state=state,options=options,configure=configure_case))
 if reused(cid,method):
  run(case,DATASETS,[],preflight=True)
  for ds in DATASETS:
   src=adaptive.OUT/'cases'/cid/(ds.lower()+'_seed0');dd=OUT/'cases'/cid/src.name
   for p in src.iterdir():
    if p.name not in ['status.json']:adaptive.link(p,dd/p.name)
  return
 run(case,DATASETS,[method])
 statuses={ds:read(OUT/'cases'/cid/(ds.lower()+'_seed0')/method/'outcome.json')['status'] for ds in DATASETS}
 assert 'failed' not in statuses.values(),statuses
 print(dict(case=cid,method=method,statuses=statuses),flush=True)


def verify():
 register();rows=[]
 for case in cases():
  for ds in DATASETS:
   for method in METHODS:
    arm=OUT/'cases'/case['id']/(ds.lower()+'_seed0')/method
    outcome=read(arm/'outcome.json');row=dict(case=case['id'],dataset=ds,method=method,status=outcome['status'],reused=reused(case['id'],method))
    assert outcome['status']!='failed',outcome
    if outcome['status']=='completed':
     steps=case['value'] if case['axis']=='fu_steps' else 5
     v=read(arm/'verification.json');assert v['FL_rounds']==200 and v['FU_steps']==45*steps
     work=[read(p) for p in sorted((arm/'work_attempts').glob('*.json'))];local=[r for w in work for r in w['local']]
     assert len(local)==45*steps
     if method in EXTERNAL:
      assert all(not r['clean_proxy_used'] and not r['request_teacher_used'] and not r['adaptive_retention_used'] for r in local)
      assert all(w['kind'] in ['fl','fu'] for w in work)
     row.update(verified_FU_cycles=len(local),transactions=len(work),physical_validation=v['passed'])
    rows.append(row)
 result=dict(passed=True,rows=rows,outcomes=dict(collections.Counter(r['status'] for r in rows)))
 write(OUT/'verification.json',result);return result


def unit():
 import torch
 import torch.nn.functional as F
 global ACTIVE
 torch.manual_seed(28);images=F.normalize(torch.randn(10,4,dtype=torch.double),dim=-1);texts=F.normalize(torch.randn(20,4,dtype=torch.double),dim=-1)
 theta=torch.cat([.01*torch.randn(16,dtype=torch.double),torch.tensor([1.],dtype=torch.double)])
 class Data:
  caption_ids={i:[2*i,2*i+1] for i in range(10)}
  def training_caption_ids(self,ids,k):return [2*i for i in ids]
 class Engine:
  data=Data();literature_lr=.0035;work_started=None
  def features(self,v,ids,cids):
   matrix=torch.eye(4,dtype=torch.double)+v[:16].reshape(4,4)
   return F.normalize(images[ids]@matrix,dim=-1),F.normalize(texts[cids]@matrix,dim=-1),v[-1]
 engine=Engine();grad={}
 for method in EXTERNAL:
  ACTIVE=method;grad[method],_=baseline_gradient(engine,theta,[0,1],[2,3,4],0,None)
  assert grad[method][-1]==0 and torch.isfinite(grad[method]).all()
 # Net GAFT update equals explicit two-step execution, not same-point loss summation.
 ACTIVE='fedavg_clip_ft';mid=theta-engine.literature_lr*grad['fedavg_clip_ga']
 second,_=baseline_gradient(engine,mid,[0,1],[2,3,4],0,None)
 assert torch.allclose(grad['fedavg_clip_gaft'],grad['fedavg_clip_ga']+second,rtol=1e-10,atol=1e-12)
 _,base=infra.prior.load_base()
 for method in METHODS:
  ACTIVE=method;fn=vectors(base,'cliperase');base.repeat.clone_function(fn,[('np.random.default_rng(','seeded_rng(',2)],{'seeded_rng':lambda k:base.repeat.seed_rng(k,0)})
 write(OUT/'unit_checks.json',dict(passed=True,GAFT_is_two_sequential_local_steps=True,temperature_frozen=True,sampler_clone_valid=True))
 print('unit checks passed')


def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['unit','case','run','verify']);p.add_argument('--case',default='baseline');p.add_argument('--method',default=PROPOSED,choices=list(METHODS));a=p.parse_args()
 if a.action=='unit':unit();return
 if a.action=='case':run_case(a.case,a.method);return
 if a.action=='verify':print(verify());return
 register()
 for case in cases():
  for method in METHODS:
   subprocess.run([sys.executable,str(Path(__file__)),'case','--case',case['id'],'--method',method],cwd=ROOT,check=True)
 result=verify();write(OUT/'finished.json',dict(unix=time.time(),**result));state(stage='completed',outcomes=result['outcomes']);print(result,flush=True)


if __name__=='__main__':main()
