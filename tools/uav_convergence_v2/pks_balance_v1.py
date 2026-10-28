"""Bound priority-sampling drift using a local uniform-gradient anchor.

Training-only adaptation; reporting references never enter updates. Preserve all
predeclared candidates and the original P-KS, including unfavorable outcomes.
"""
import argparse
from pathlib import Path
import time
import numpy as np
import learning_quality_v3 as q
import learning_quality_report as prior
import learning_time_budget as time_study

ROOT=q.ROOT;OUT=ROOT/'results/pks_balance_v1_20261028';CONT=OUT/'continuation'
read,write,sha=q.read,q.write,q.sha
VARIANTS={
 'control':dict(label='P-KS control',objective='gate',selection=16,equal=False,exact=False,rho=None),
 'mix25_gate':dict(label='P-KB25',objective='gate',selection=16,equal=False,exact=False,rho=.25),
 'mix50_gate':dict(label='P-KB50',objective='gate',selection=16,equal=False,exact=False,rho=.5),
 'mix50_adaptive':dict(label='P-AB50',objective='adaptive',selection=16,equal=False,exact=False,rho=.5),
 'pk_uniform':dict(label='P-KS then uniform FL',objective='gate',selection=0,equal=False,exact=False,rho=None,reuse='gate_select16'),
 'pls_uniform':dict(label='P-LS then uniform FL',objective='adaptive',selection=0,equal=False,exact=False,rho=None,reuse='select16'),
}
ACTIVE=None
PARENT_ENGINE=q.engine_wrapper
BASE_PROTOCOL=q.read(q.OUT/'protocol.json')
OLD_SOURCES={(ds,v):prior.variant_source(ds,v) for ds in q.infra.DATASETS for v in ['gate_select16','select16']}
priority_with_rng=q.infra.cached_clone(q.select_batch,[
 ('def select_batch(engine,theta,pool,k,user):','def select_batch(engine,theta,pool,k,user,rng):',1),
 ('rng=np.random.default_rng(730000+k*10+user);','',1),
 ("CONFIG['selection']","q.CONFIG['selection']",1)],dict(q=q))


def register():
 path=OUT/'protocol.json'
 if path.exists():
  p=read(path)
  for file,digest in p['source_sha256'].items():assert sha(ROOT/file)==digest,file
  return p
 sources=dict(BASE_PROTOCOL['source_sha256'])
 for file in [Path(__file__),Path(time_study.__file__)]:sources[str(file.relative_to(ROOT))]=sha(file)
 result=dict(registered_unix=time.time(),variants=VARIANTS,datasets=q.infra.DATASETS,seed=0,source_sha256=sources,
  purpose='Retain a useful priority-sampling contribution while bounding its deviation from uniform retained FL. All constants identical across datasets; no reporting-reference or validation input into training.',
  rule='For each local FL update, compute g_U on the original uniform16 batch and g_H on the original P-KS priority16 batch selected from past training CE. d=g_H-g_U; beta=min(1,rho*norm(g_U)/norm(d)); use g_U+beta*d. Hence norm(g-g_U)<=rho*norm(g_U), up to numerical tolerance. rho in the predeclared small grid0.25,0.5. This local gradient bound is not a bound on final KL.',
  costs='Execute and charge both gradient batches, original priority selection arithmetic, and norm/blend operations. Same20480 payload coordinates and original joint resource/head optimizer. Charge actual compute,radio,broadcast and hover time; no borrowed reference model.',
  FU='Keep original P-KS proxy-KL gate for two variants; one0.5 variant restores the original adaptive FU coefficient to isolate the gate. No additional FU cycles:45 requests x5.',
  controls='Fresh control must reproduce original P-KS200 final metrics, checkpoint and modeled physical costs. Two continuation-only interventions reuse the actual P-KS or P-LS200 checkpoint and change subsequent sampling to uniform. No retrospective rescaling.',
  continuation='Same previous rule: after200FL and225FU, continue retained FL with constant0.0035 at the last site until the first completed update beyond400 active seconds. New candidates retain their same sampler/gradient-bound rule throughout. Compare completed checkpoints under the shared time budget.',
  evaluation='Keep old common reference200 and round-matched common reference scores. Sampling-matched references are evaluation-only diagnostics and never rename an unchanged model as a repaired one. Single-seed exploratory development; no guaranteed rank or held-out-superiority claim.',
  criteria='Report both mean and worst-request KL, mR, loss, energy and time for every arm. The existing0.01 threshold is evaluated unchanged. No acceptance filter discards losing candidates.',
  background=[dict(title='SLUG ICML2025: gradient alignment for targeted VLM unlearning',url='https://proceedings.mlr.press/v267/cai25f.html'),
   dict(title='CLIPErase ACL2025: forgetting, retention and consistency',url='https://aclanthology.org/2025.acl-long.1469/')],
  attribution='The bounded uniform/priority gradient combination is our adaptation, not a reproduction of SLUG or CLIPErase and not a transferred theoretical deletion guarantee.')
 write(path,result);return result


def initialize_variant(name):
 global ACTIVE
 ACTIVE=name;q.CONFIG=VARIANTS[name];q.ACTIVE=name


def choose_batch(engine,theta,pool,k,user,rng):
 spec=VARIANTS[ACTIVE]
 if not spec['selection']:return rng.choice(pool,min(16,len(pool)),replace=False).astype(int).tolist()
 if spec['rho'] is None:return priority_with_rng(engine,theta,pool,k,user,rng)
 uniform=rng.choice(pool,min(16,len(pool)),replace=False).astype(int).tolist()
 priority=priority_with_rng(engine,theta,pool,k,user,np.random.default_rng(730000+k*10+user))
 engine.pending_priority[(user,k)]=(uniform,priority)
 return uniform


def engine_wrapper(parent):
 wrapped=PARENT_ENGINE(parent)
 class Engine(wrapped):
  def __init__(self,*args,**kwargs):
   super().__init__(*args,**kwargs);self.pending_priority={};self.blend_records=[]
  def gradient(self,theta,ids,k,kind='ce'):
   user=self.sample_owner[int(ids[0])];pending=self.pending_priority.pop((user,k),None) if kind=='ce' else None
   if pending is None:return super().gradient(theta,ids,k,kind)
   uniform,priority=pending;assert list(ids)==uniform
   gu,lu=super().gradient(theta,uniform,k,kind);gh,lh=super().gradient(theta,priority,k,kind)
   delta=gh-gu;gn=float(gu.norm());dn=float(delta.norm());rho=VARIANTS[ACTIVE]['rho']
   beta=min(1.,rho*gn/max(dn,1e-30));gradient=gu+beta*delta;gradient[-1]=0
   relative=float((gradient-gu).norm())/max(gn,1e-30)
   assert relative<=rho*(1+1e-5)+1e-8
   q.infra.lib.infra.charge(self,uniform,int(8*len(theta)+16))
   self.blend_records.append(dict(user=user,round=k,uniform=uniform,priority=priority,rho=rho,beta=beta,
    uniform_loss=lu,priority_loss=lh,uniform_gradient_norm=gn,priority_difference_norm=dn,relative_gradient_difference=relative))
   return gradient,(1-beta)*lu+beta*lh
  def finish_work(self):
   value=super().finish_work();value['FL_gradient_blend']=self.blend_records[:];self.blend_records=[];return value
 return Engine


def vectors(base,objective):
 fn=q.ORIGINAL_VECTORS(base,objective)
 old='batch=np.random.default_rng(730000+k*10+user).choice(pool,min(FL_BATCH,len(pool)),replace=False).astype(int).tolist()'
 return q.infra.cached_clone(fn,[(old,'batch=choose_batch(engine,theta,pool,k,user,np.random.default_rng(730000+k*10+user))',1)],dict(choose_batch=choose_batch))


def initial_source(ds,m):
 if 'reuse' in VARIANTS[m]:return OLD_SOURCES[(ds,VARIANTS[m]['reuse'])]
 return OUT/'training'/m/'cases/baseline'/(ds.lower()+'_seed0')/q.PROPOSED


def baseline(name):
 assert 'reuse' not in VARIANTS[name];register();initialize_variant(name)
 q.OUT=OUT/'training';q.register=register;q.VARIANTS=VARIANTS;q.engine_wrapper=engine_wrapper;q.vectors=vectors
 q.run(name,'baseline')
 if name=='control':
  rows=[]
  for ds in q.infra.DATASETS:
   new=initial_source(ds,name);old=OLD_SOURCES[(ds,'gate_select16')]
   assert read(new/'service_completion.json')['evaluation']==read(old/'service_completion.json')['evaluation']
   import torch
   assert torch.equal(torch.load(new/'final.pt',weights_only=True)['theta'],torch.load(old/'final.pt',weights_only=True)['theta'])
   for a,b in zip(sorted((new/'transactions').glob('*.json')),sorted((old/'transactions').glob('*.json'))):
    x,y=read(a),read(b)
    assert x['kind']==y['kind'] and x['FU_request_ids']==y['FU_request_ids']
    assert x['start_s']==y['start_s'] and x['end_s']==y['end_s'] and x['radio']['energy_bound_j']==y['radio']['energy_bound_j']
   rows.append(dict(dataset=ds,identical_checkpoint_metrics_and_physical_clock=True))
  write(OUT/'control_check.json',dict(passed=True,rows=rows))


def continued(name,ds):
 register();initialize_variant(name);q.engine_wrapper=engine_wrapper
 # Extend the frozen executable without changing its recorded source.
 fn=q.infra.cached_clone(time_study.run,[
  ("quality.CONFIG=quality.VARIANTS['gate_select16' if m=='gate_select16' else 'control'];quality.ACTIVE=m","initialize_variant(m)",1),
  ("if m in ['current','gate_select16']:","if m in METHODS:",1),
  ("batch=quality.select_batch(engine,theta,pool,k,u) if m=='gate_select16' else np.random.default_rng(730000+k*10+u).choice(pool,min(16,len(pool)),replace=False).astype(int).tolist()",
   "batch=choose_batch(engine,theta,pool,k,u,np.random.default_rng(730000+k*10+u))",1),
  ("selection=[] if engine_work is None else engine_work['FL_sample_selection'],timing=", "gradient_blend=[] if engine_work is None else engine_work['FL_gradient_blend'],selection=[] if engine_work is None else engine_work['FL_sample_selection'],timing=",1),
  ('theta=update_theta(theta,vector)','theta=update_theta(theta,vector)\n    base.core.save_theta(ad/\'checkpoints\'/f\'{200+extra:04d}.pt\',theta,200+extra)',1)],
  dict(OUT=CONT,METHODS={name:VARIANTS[name]['label']},source=initial_source,register=register,
   initialize_variant=initialize_variant,choose_batch=choose_batch))
 fn(ds)


def unit():
 rng=np.random.default_rng(17)
 for rho in [.25,.5]:
  for _ in range(100):
   gu=rng.normal(size=64);gh=rng.normal(size=64)*3;d=gh-gu;beta=min(1.,rho*np.linalg.norm(gu)/np.linalg.norm(d));g=gu+beta*d
   assert np.linalg.norm(g-gu)<=rho*np.linalg.norm(gu)*(1+1e-12)
   assert np.dot(g,gu)>0
 result=dict(passed=True,gradient_deviation_bound=True,uniform_descent_alignment_for_rho_below_one=True)
 write(OUT/'unit_checks.json',result);return result


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['baseline','continue','unit']);p.add_argument('--variant',choices=VARIANTS);p.add_argument('--dataset',choices=q.infra.DATASETS);a=p.parse_args()
 if a.action=='unit':print(unit())
 elif a.action=='baseline':baseline(a.variant)
 else:continued(a.variant,a.dataset)
