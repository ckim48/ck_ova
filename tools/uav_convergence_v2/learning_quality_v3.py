"""Training-only FL selection, FU diagnostics and a federated SalUn transfer.

Every setting is retained. No validation/reference predictions choose gradients,
weights or samples. New policies remain development candidates until evaluated.
"""
import argparse
import copy
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import numpy as np
import workload_sweeps_v3 as infra
import workload_charts as previous

ROOT=infra.ROOT;OUT=ROOT/'results/learning_quality_v3_20261028'
PROPOSED=infra.lib.PROPOSED;DIGITAL='iscc_p2_gaft'
VARIANTS={
 'control':dict(label='Current Proposed',objective='adaptive',selection=0,equal=False,exact=False),
 'exact':dict(label='Exact aggregation diagnostic',objective='adaptive',selection=0,equal=False,exact=True),
 'equal':dict(label='Equal-client FL diagnostic',objective='adaptive',selection=0,equal=True,exact=False),
 'gaft':dict(label='Joint GAFT diagnostic',objective='gaft',selection=0,equal=False,exact=False),
 'retain1':dict(label='Retention floor1',objective='retain1',selection=0,equal=False,exact=False),
 'gate':dict(label='Proxy-KL gate',objective='gate',selection=0,equal=False,exact=False),
 'select16':dict(label='Loss-priority FL16',objective='adaptive',selection=16,equal=False,exact=False),
 'gate_select16':dict(label='KL gate + FL16 selection',objective='gate',selection=16,equal=False,exact=False),
 'gate_select8':dict(label='KL gate + FL8 selection',objective='gate',selection=8,equal=False,exact=False),
 'salun':dict(label='F-SalUn',objective='salun',selection=0,equal=True,exact=False),
}
ACTIVE=None;CONFIG=None;BASE=None
ORIGINAL_VECTORS=infra.lib.vectors
ORIGINAL_ARM=infra.proposed_arm
ORIGINAL_PAPER=infra.paper_arm
ORIGINAL_ENGINE=infra.lib.infra.controlled_engine
read,write,sha=infra.read,infra.write,infra.sha


def cases():
 return [dict(id='baseline',axis=None,value=None)]+[dict(id=previous.cid(a,v),axis=a,value=v) for a,s in previous.AXES.items() for v in s['values'] if v!=s['base']]


def register():
 path=OUT/'protocol.json'
 if path.exists():
  p=read(path)
  for n,d in p['source_sha256'].items():assert sha(ROOT/n)==d,n
  return p
 inherited=read(infra.OUT/'protocol.json')
 sources=dict(inherited['source_sha256']);sources[str(Path(__file__).relative_to(ROOT))]=sha(Path(__file__))
 p=dict(registered_unix=time.time(),variants=VARIANTS,datasets=infra.DATASETS,seed=0,source_sha256=sources,
  fixed='Same cached features, warm20, requests, reporting reference200, route, channel and complete FL/FU workloads. Per-scenario interventions inherited unchanged. Control must reproduce prior final quality and physical clocks. All candidates and failures retained; development experiments, not independent held-out confirmation.',
  selection='Causal per-sample symmetric contrastive training CE cached after actual FL gradient calls. Half batch uniform exploration; remainder highest cached CE among current retained local pool, randomized deterministic tie order. Unseen samples start at median known loss. Purge absent pool samples. Same16 cap or explicit8 cap. No validation,reference,teacher,or future-request input. Extra scoring matrix arithmetic and selection CPU scalar work charged.',
  retain1='Current adaptive FU except retention coefficient fixed1, compared with actual original controller median~0.5. Training-only diagnostic, no test tuning.',
  gate='Objective3*relu(forget_proxy_retrieval_KL-0.01) + retained_CE +3*request_teacher_retrieval_KL. Stop pushing training proxy KL below the declared tolerance; retain coefficient1. Threshold is a surrogate on local training batches, not the reported reference KL or deletion guarantee. Existing known-forget-cohort proxy20 assumption retained and disclosed.',
  gaft='Same Proposed joint transport and weighting; change only FU to existing GA then FT. No proxy/teacher/controller. Isolates adopted FU objective at fixed network design family.',
  equal='Same adaptive objective/transport; FL uses1/10 instead of local retained-pool-size weights. FU weighting unchanged.',
  exact='Replace applied noisy receiver update with exact FL+FU sum. Keep designed physical cost. Diagnostic only, not realizable baseline.',
  salun=dict(label='F-SalUn',paper='https://proceedings.iclr.cc/paper_files/paper/2024/hash/ec4d2e436794d1bf55ca83f5ebb31887-Abstract-Conference.html',
   full_paper='https://proceedings.iclr.cc/paper_files/paper/2024/file/ec4d2e436794d1bf55ca83f5ebb31887-Paper-Conference.pdf',
   preserved='ICLR2024 SalUn Alg.1: forget-loss gradient saliency at request start, median hard mask, random-label forgetting plus retained CE, masked SGD updates for5 prescribed cycles. Mask fixed per request; no teacher/proxy.',
   adapted='Federated cached VLM rank10 adapter setting. Classification labels become one-to-one video/text retrieval assignments: random nonzero cyclic permutation yields wrong caption matches. Native one-caption-per-video sampler retained. Frozen backbone/temperature; mask across20480 adapter coordinates. Equal-client FedSGD and ISCC-P2 FP32 FDMA; no claim SalUn itself is a UAV-FL paper or complete full-model reproduction. Five local iterations per request, same225 FU cycles. Dense full-vector wire format shared with digital baseline; no unimplemented sparse encoding benefit.',
   loss='Symmetric CE on relabeled forget image/text batch plus symmetric CE on retained batch, coefficient1 each. Saliency uses original positive pairs, mask |g|>=median(|g|). Charge initial saliency backward, all update passes and mask/random-label arithmetic.'),
  publication='Code, aggregate metrics, plot arrays and charts only. Raw samples, checkpoints and transaction logs local.')
 write(path,p);return p


def select_batch(engine,theta,pool,k,user):
 pool=list(map(int,pool));cache=engine.selection_losses.setdefault(user,{})
 for i in set(cache)-set(pool):del cache[i]
 rng=np.random.default_rng(730000+k*10+user);size=min(CONFIG['selection'],len(pool));explore=min(size,max(1,size//2))
 random=list(map(int,rng.choice(pool,explore,replace=False)));remaining=[i for i in rng.permutation(pool).tolist() if i not in random]
 default=float(np.median(list(cache.values()))) if cache else math.log(max(2,size))
 ordered=sorted(remaining,key=lambda i:cache.get(i,default),reverse=True);chosen=random+ordered[:size-explore]
 assert len(set(chosen))==size and set(chosen)<=set(pool)
 engine.selection_records.append(dict(user=user,round=k,candidates=len(pool),selected=chosen,exploration=random,cached_scores=len(cache)))
 infra.lib.infra.charge(engine,[pool[0]],int(12*len(pool)+len(pool)*math.ceil(math.log2(max(2,len(pool))))))
 return chosen


def engine_wrapper(parent):
 wrapped=ORIGINAL_ENGINE(parent)
 class Engine(wrapped):
  def __init__(self,*a,**kw):
   super().__init__(*a,**kw);self.selection_losses={};self.selection_records=[];self.salun_masks={}
  def gradient(self,theta,ids,k,kind='ce'):
   if not CONFIG['selection'] or kind!='ce':return super().gradient(theta,ids,k,kind)
   import torch
   import torch.nn.functional as F
   if len(ids)<2:return torch.zeros_like(theta),0.
   cids=self.data.training_caption_ids(ids,k);variable=theta.detach().requires_grad_(True)
   im,tx,scale=self.features(variable,ids,cids);logits=scale.exp()*(im@tx.T);target=torch.arange(len(ids),device=theta.device)
   per=.5*(F.cross_entropy(logits,target,reduction='none')+F.cross_entropy(logits.T,target,reduction='none'))
   value=per.mean();g=torch.autograd.grad(value,variable)[0].detach();g[-1]=0
   user=self.sample_owner[int(ids[0])];assert all(self.sample_owner[int(i)]==user for i in ids)
   cache=self.selection_losses.setdefault(user,{})
   for i,v in zip(ids,per.detach().cpu().tolist()):cache[int(i)]=float(v)
   # Additional elementwise per-example accounting; matrix loss is the existing FL operation.
   infra.lib.infra.charge(self,ids,8*len(ids)+4*len(ids)**2)
   return g,float(value.detach())
  def finish_work(self):
   value=super().finish_work();value['FL_sample_selection']=self.selection_records[:];self.selection_records=[];return value
 return Engine


def salun_gradient(engine,theta,fids,rids,step,teacher):
 import torch
 import torch.nn.functional as F
 key=tuple(map(int,fids));fids=list(key);n=len(fids);assert n>=2
 def loss(variable,ids,cids,shift=0):
  im,tx,scale=engine.features(variable,ids,cids);logits=scale.exp()*(im@tx.T);target=torch.arange(len(ids),device=theta.device)
  return .5*(F.cross_entropy(logits,(target+shift)%len(ids))+F.cross_entropy(logits.T,(target-shift)%len(ids)))
 cids=engine.data.training_caption_ids(fids,21+step);extra=0
 if key not in engine.salun_masks:
  assert step==0
  v=theta.detach().requires_grad_(True);sal=loss(v,fids,cids);g=torch.autograd.grad(sal,v)[0].detach();g[-1]=0
  threshold=torch.median(g[:-1].abs());mask=(g.abs()>=threshold).to(g.dtype);mask[-1]=0
  engine.salun_masks[key]=(mask,float(threshold));extra+=n*n*(3*512+16)+theta.numel()*math.ceil(math.log2(theta.numel()))
 mask,threshold=engine.salun_masks[key]
 rng=np.random.default_rng(991000+17*sum(fids)+step);shift=int(rng.integers(1,n))
 v=theta.detach().requires_grad_(True);forget=loss(v,fids,cids,shift)
 rcids=engine.data.training_caption_ids(rids,21+step);retained=loss(v,rids,rcids)
 g=torch.autograd.grad(forget+retained,v)[0].detach()*mask;g[-1]=0
 extra+=(n*n+len(rids)**2)*(3*512+16)+3*theta.numel()+8*n
 infra.lib.infra.charge(engine,fids,int(extra))
 return g,dict(FU_variant='F-SalUn',random_label_shift=shift,mask_fraction=float(mask[:-1].mean()),
  mask_sha256=hashlib.sha256(mask[:-1].cpu().numpy().tobytes()).hexdigest(),saliency_threshold=threshold,
  mask_created=step==0,forget_random_label_CE=float(forget.detach()),retained_CE=float(retained.detach()),
  clean_proxy_used=False,request_teacher_used=False,adaptive_retention_used=False,FU_gradient_norm=float(g.norm()),
  extra_loss_MAC_equivalents=int(extra),images=n,retained_images=rids,objective_source='SalUn ICLR2024 Alg.1 federated video/text retrieval adaptation')


def vectors(base,objective):
 old_active=infra.lib.ACTIVE
 if CONFIG['objective'] in ['gaft','salun']:
  infra.lib.ACTIVE='fedavg_clip_gaft'
  original=ORIGINAL_VECTORS(base,objective)
  if CONFIG['objective']=='salun':original=base.repeat.clone_function(original,[],dict(request_gradient=salun_gradient))
 else:original=ORIGINAL_VECTORS(base,objective)
 infra.lib.ACTIVE=old_active
 edits=[]
 if CONFIG['equal']:
  edits.append(('len(pool)/total','1/10',1))
 if CONFIG['selection']:
  old='batch=np.random.default_rng(730000+k*10+user).choice(pool,min(FL_BATCH,len(pool)),replace=False).astype(int).tolist()'
  edits.append((old,'batch=select_batch(engine,theta,pool,k,user)',1))
 return base.repeat.clone_function(original,edits,dict(select_batch=select_batch))


def choose_site(clock,q,fl):
 s=min(fl//infra.SITE_FL,4)
 if clock.sites:
  prev=clock.sites[-1]['site'];pending=[r for r in q.items.values() if r['mission_site']==prev and r['steps']<r['required_steps']]
  if fl==(prev+1)*infra.SITE_FL and pending:return prev
 return s


def choose_kind(clock,q,t,fl):
 site=clock.sites[-1]['site']
 if fl==(site+1)*infra.SITE_FL and q.pending(t):
  pending=q.pending(t);assert all(r['mission_site']==site for r in pending)
  return 'fu',[r['id'] for r in pending]
 return q.decision(t,fl)


def make_arm(base,deployment,radios,cap):
 global BASE
 BASE=base
 if CONFIG['objective'] in ['gaft','salun']:radios[PROPOSED]=base.base.parent.guard
 if CONFIG['exact']:
  old=radios[PROPOSED]
  class Exact:
   @staticmethod
   def analog(l,u,*args):
    received,meta=old.analog(l,u,*args)
    if received is not None:meta['exact_diagnostic']=True;received=(l+u).sum(0)
    return received,meta
  radios[PROPOSED]=Exact
 original=ORIGINAL_ARM(base,deployment,radios,cap)
 return infra.cached_clone(original,[('site=min(fl//SITE_FL,4)','site=choose_site(clock,q,fl)',1),('kind,ids=q.decision(t,fl)','kind,ids=choose_kind(clock,q,t,fl)',1)],dict(choose_site=choose_site,choose_kind=choose_kind))


def options(base,method):return dict(base.base.ARMS[infra.lib.adaptive.PROPOSED],method=method,learning_variant=ACTIVE,configuration=CONFIG,scope='Registered learning-quality diagnostic or SalUn paper-core transfer')


def run(variant,case):
 global ACTIVE,CONFIG
 register();ACTIVE=variant;CONFIG=VARIANTS[variant]
 infra.OUT=OUT/variant;infra.register=register;infra.cases=cases
 infra.lib.infra.controlled_engine=engine_wrapper;infra.lib.vectors=vectors;infra.proposed_arm=make_arm
 infra.lib.options=options
 if CONFIG['objective']=='retain1':infra.lib.adaptive.CONTROLLER.update(initial=1.,lower=1.,upper=1.)
 if CONFIG['objective']=='gate':
  _,b=infra.lib.infra.prior.load_base()
  infra.lib.adaptive.request_gradient=infra.cached_clone(infra.lib.adaptive.request_gradient,[
   ('weight, smooth, signal = feedback(old_weight,old_ema,dr,df)','weight, smooth, signal = 1., 0., 0.',1),
   ('3*fk+3*ck','3*torch.relu(fk-.01)+3*ck',1)],{})
  # Log the actual gated objective rather than the ungated diagnostic sum.
  infra.lib.adaptive.request_gradient=infra.cached_clone(infra.lib.adaptive.request_gradient,[
   ('objective=float((3*fk+weight*rm+3*ck).detach())','objective=float((3*torch.relu(fk-.01)+weight*rm+3*ck).detach())',1)],{})
 if CONFIG['objective']=='gate':
  old_radio=infra.lib.adaptive.radio
  def gated_radio(b):
   old=old_radio(b)
   class Radio:
    @staticmethod
    def analog(*args):
     result,meta=old.analog(*args)
     for item in infra.lib.adaptive.ACTIVE_ENGINE.adaptive_pending.values():
      row=item['row'];row['objective']=3*max(0.,row['forget_loss']-.01)+row['RM_weight']*row['RM']+3*row['CM'];row['proxy_KL_gate']=.01;row['proxy_KL_active']=row['forget_loss']>.01
     return result,meta
   return Radio
  infra.lib.adaptive.radio=gated_radio
 if variant=='salun':infra.lib.baseline_gradient=salun_gradient
 method=DIGITAL if variant=='salun' else PROPOSED
 # Keep the fixed paper digital transport for SalUn; Proposed diagnostics remain joint.
 original_run=infra.run_case
 if variant=='salun':
  source=inspect.getsource(original_run)
  fn=infra.cached_clone(original_run,[("new=[m for m in chosen if not reused(cid,m)]","new=[m for m in chosen if not reused(cid,m)]\n    new=[DIGITAL]",1),
   ("chosen=list(METHODS) if group=='all' else [lib.PROPOSED] if group=='proposed' else list(paper.METHODS)","chosen=[DIGITAL]",1)],dict(DIGITAL=DIGITAL,reused=lambda c,m:False),live_globals=infra.__dict__)
 else:
  infra.reused=lambda c,m:False;fn=original_run
 fn(case,'proposed')
 for ds in infra.DATASETS:
  arm=infra.OUT/'cases'/case/(ds.lower()+'_seed0')/method
  assert read(arm/'outcome.json')['status']=='completed'
 print(dict(variant=variant,case=case,completed=True),flush=True)


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--variant',choices=list(VARIANTS),required=True);p.add_argument('--case',default='baseline');a=p.parse_args();run(a.variant,a.case)
