from common import *
import argparse,fcntl,os,traceback
import numpy as np
import torch
import torch.nn.functional as F
from clip_model import DeepPromptCLIP
from dataset import RetrievalData,prepare
from objectives import contrastive,relation_kl,self_check

class Engine:
 def __init__(self,data,plan):
  self.data=data;self.plan=plan;self.device=torch.device('cuda');torch.set_num_threads(4)
  torch.manual_seed(0);torch.cuda.manual_seed_all(0)
  self.model=DeepPromptCLIP.from_local_checkpoint(ROOT/'checkpoints/clip-vit-base-patch16',
    prompt_layers=8,prompt_tokens=2,checkpoint_layers=True).to(self.device).eval()
  self.theta0=self.model.initial_vector.detach().clone()
  self.teacher=None;self.teacher_images={};self.teacher_texts={}
 def features(self,theta,ids,cids):
  v,t,s=self.model.parts(theta)
  images=F.normalize(self.model.image_features(self.data.pixels(ids,self.device),v),dim=-1)
  texts=F.normalize(self.model.text_features(**self.data.texts(cids,self.device),text=t),dim=-1)
  return images,texts,s
 def set_teacher(self,theta):
  self.teacher=theta.clone().detach();self.teacher_images={};self.teacher_texts={}
 @torch.no_grad()
 def teacher_features(self,ids,cids):
  v,t,_=self.model.parts(self.teacher)
  missing=[i for i in ids if i not in self.teacher_images]
  if missing:
   im=F.normalize(self.model.image_features(self.data.pixels(missing,self.device),v),dim=-1)
   self.teacher_images.update({i:f.cpu() for i,f in zip(missing,im)})
  missing=[i for i in cids if i not in self.teacher_texts]
  if missing:
   tx=F.normalize(self.model.text_features(**self.data.texts(missing,self.device),text=t),dim=-1)
   self.teacher_texts.update({i:f.cpu() for i,f in zip(missing,tx)})
  return torch.stack([self.teacher_images[i] for i in ids]).to(self.device),torch.stack([self.teacher_texts[i] for i in cids]).to(self.device)
 def gradient(self,theta,ids,k,kind='ce'):
  if len(ids)<2:return torch.zeros_like(theta),0.
  cids=self.data.training_caption_ids(ids,k)
  old=self.teacher_features(ids,cids) if kind=='relation' else None
  variable=theta.detach().requires_grad_(True)
  im,tx,s=self.features(variable,ids,cids)
  if kind=='relation':value=relation_kl(im,tx,*old,s)
  else:
   ix=torch.tensor(ids,device=self.device);value=contrastive(im,tx,s,ix,ix)
  grad=torch.autograd.grad(value,variable)[0].detach()
  # Shared scale fixed for all arms so loss cannot fall merely by changing temperature.
  grad[-1]=0
  assert torch.isfinite(grad).all()
  return grad,float(value.detach())
 def updates(self,theta,record,k,kind,forget_pools):
  n=10;d=len(theta);learning=np.zeros((n,d));unlearning=np.zeros((n,d))
  count=sum(map(len,record['retained']));nf=sum(map(len,forget_pools));stats=[]
  for user,ids in enumerate(record['batches']):
   g,loss=self.gradient(theta,ids,k)
   learning[user]=(-self.plan['learning_rate']*len(record['retained'][user])/count*g).cpu().numpy()
   f=forget_pools[user]
   if f and k in self.plan['unlearning_rounds']:
    batch=np.random.default_rng(9000+k*10+user).choice(f,min(16,len(f)),replace=False).astype(int).tolist()
    fg,floss=self.gradient(theta,batch,k)
    update=self.plan['forget_weight']*fg
    reg=0.
    if kind=='proposed':
     rg,reg=self.gradient(theta,ids,k,'relation')
     update=update-self.plan['crossmodal_weight']*rg
    unlearning[user]=(self.plan['learning_rate']*len(f)/nf*update).cpu().numpy()
   stats.append(dict(user=user,retained_loss=loss))
  # Frozen logit scale coordinate is never transmitted/noised.
  learning[:,-1]=0;unlearning[:,-1]=0
  return learning[:,:-1],unlearning[:,:-1],stats
 @torch.no_grad()
 def score(self,theta,ids):
  v,t,s=self.model.parts(theta);ims=[];txs=[];cids=[c for i in ids for c in self.data.caption_ids[i]]
  for j in range(0,len(ids),16):ims.append(F.normalize(self.model.image_features(self.data.pixels(ids[j:j+16],self.device),v),dim=-1).cpu())
  for j in range(0,len(cids),64):txs.append(F.normalize(self.model.text_features(**self.data.texts(cids[j:j+64],self.device),text=t),dim=-1).cpu())
  logits=float(s.exp())*torch.cat(ims)@torch.cat(txs).T
  return logits
 @torch.no_grad()
 def evaluate(self,theta,reference=None,test=False):
  ids=self.plan['official_test_images'] if test else self.plan['gallery_images']
  logits=self.score(theta,ids);logi=F.log_softmax(logits,dim=1);logt=F.log_softmax(logits.T,dim=1)
  groups={'test':ids} if test else {'retained':self.plan['validation_images'],'forget':self.plan['forget_probe']}
  owner=torch.arange(len(ids)).repeat_interleave(5);rows={}
  for name,group in groups.items():
   indices=torch.tensor([ids.index(i) for i in group]);ci=(indices[:,None]*5+torch.arange(5)).flatten()
   ri=logits[indices].argsort(dim=1,descending=True)
   rt=logits.T[ci].argsort(dim=1,descending=True)
   metrics={}
   for k in [1,5,10]:
    metrics[f'i2t_R{k}']=float((owner[ri[:,:k]]==indices[:,None]).any(1).float().mean())
    metrics[f't2i_R{k}']=float((rt[:,:k]==owner[ci,None]).any(1).float().mean())
   metrics['mR']=sum(metrics.values())/6
   image_loss=-torch.logsumexp(logi[indices[:,None],indices[:,None]*5+torch.arange(5)],1)
   text_loss=-logt[ci,owner[ci]]
   metrics['loss']=float((image_loss.mean()+text_loss.mean())/2)
   if reference is not None:
    oldi,oldt=reference
    ki=(oldi[indices].exp()*(oldi[indices]-logi[indices])).sum(1).mean()
    kt=(oldt[ci].exp()*(oldt[ci]-logt[ci])).sum(1).mean()
    metrics['KL_retrain_to_candidate']=float(torch.clamp((ki+kt)/2,min=0))
    metrics['i2t_top1_disagreement']=float((oldi[indices].argmax(1)!=logi[indices].argmax(1)).float().mean())
    metrics['t2i_top1_disagreement']=float((oldt[ci].argmax(1)!=logt[ci].argmax(1)).float().mean())
   rows[name]=metrics
  return rows,(logi,logt)
 @torch.no_grad()
 def mia_scores(self,theta):
  members=self.plan['mia_member_images'];nonmembers=self.plan['mia_nonmember_images'];ids=members+nonmembers
  logits=self.score(theta,ids);li=F.log_softmax(logits,1);lt=F.log_softmax(logits.T,1)
  idx=torch.arange(len(ids));cap=idx[:,None]*5+torch.arange(5)
  scores=(torch.logsumexp(li[idx[:,None],cap],1)+lt[cap,idx[:,None]].mean(1))/2
  a=scores[:len(members)];b=scores[len(members):]
  auc=float(((a[:,None]>b).float()+.5*(a[:,None]==b).float()).mean())
  return dict(loss_score_auc=auc,orientation_free_auc=max(auc,1-auc),members=len(members),nonmembers=len(nonmembers),scope='Descriptive fixed negative retrieval-loss attack, no learned attack/calibrated deletion guarantee'),scores.numpy()

def update_theta(theta,vector):
 result=theta.clone();result[:-1]+=torch.as_tensor(vector,device=theta.device,dtype=theta.dtype)
 assert torch.isfinite(result).all();return result.detach()

def trajectory(engine,plan,records,method,forget_pools,reference=None,config=None,stop=40):
 from radio import communicate
 dest=OUT/method;dest.mkdir(exist_ok=False);theta=engine.theta0.clone();rows=[]
 cumulative_e=cumulative_expected=cumulative_t=0.;channels=np.load(OUT/'channels.npy');start=time.time()
 initial,probs=engine.evaluate(theta,reference);write(dest/'initial.json',initial)
 for rec in records[:stop]:
  k=rec['round'];t0=time.time()
  status(stage=method,round=k,rounds=stop,pid=os.getpid(),elapsed_stage_s=time.time()-start)
  learning,unlearning,local=engine.updates(theta,rec,k,method,forget_pools)
  if method in ['ideal_warm','retrain']:
   vector=(learning+unlearning).sum(0);radio=dict(status='ideal_reference',energy_bound_j=None,uplink_s=None)
  else:
   mode='hybrid' if method in ['proposed','hybrid_no_crossmodal'] else method
   vector,radio=communicate(learning,unlearning,channels,config,mode,seed=100000+k)
   if vector is None:
    write(dest/'manifest.json',dict(status='incomplete_no_feasible_design_found',failed_round=k,completed_rounds=k-1,radio=radio))
    return None,None
   cumulative_e+=radio['energy_bound_j'];cumulative_expected+=radio['expected_energy_j'];cumulative_t+=radio['uplink_s']
  theta=update_theta(theta,vector)
  if k==20:engine.set_teacher(theta)
  row=dict(method=method,round=k,radio=radio,cumulative_tx_energy_bound_j=cumulative_e,
   cumulative_tx_energy_expected_j=cumulative_expected,cumulative_uplink_s=cumulative_t,
   request_transmission_steps_completed=max(0,min(k,25)-20),request_algorithm_completed=k>=25,
   logical_requests=5 if k>=21 else 0,local=local)
  if k in plan['eval_rounds'] or k==stop:
   row['evaluation'],probs=engine.evaluate(theta,reference)
   torch.save(dict(theta=theta.cpu(),round=k),dest/f'round_{k:03d}.pt')
   np.savez_compressed(dest/f'probabilities_{k:03d}.npz',log_image=probs[0].numpy(),log_text=probs[1].numpy())
  row['wall_s']=time.time()-t0;rows.append(row)
  with (dest/'rounds.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
  print(json.dumps(dict(method=method,round=k,wall_s=row['wall_s'],evaluation=row.get('evaluation')),allow_nan=False),flush=True)
 write(dest/'manifest.json',dict(status='completed',rounds=stop,wall_s=time.time()-start,
  cumulative_tx_energy_bound_j=cumulative_e if method not in ['ideal_warm','retrain'] else None,
  cumulative_uplink_s=cumulative_t if method not in ['ideal_warm','retrain'] else None))
 return theta,probs

def smoke(engine,plan,records):
 checks=self_check();ids=records[0]['batches'][0]
 start=time.time();grad,loss=engine.gradient(engine.theta0,ids,1)
 assert len(ids)>=2 and float(grad[:-1].norm())>0
 eps=1e-5;trial=engine.theta0-eps*grad
 _,loss2=engine.gradient(trial,ids,1)
 assert loss2<loss+1e-5
 checks.update(batch_size=len(ids),gradient_norm=float(grad.norm()),loss_before=loss,loss_after_small_descent=loss2,
   full_batch_gradient_wall_s=time.time()-start,peak_gpu_memory_gb=torch.cuda.max_memory_allocated()/1e9,
   image_caption_groups_disjoint=True,test_images_not_trained=True,gpu=torch.cuda.get_device_name())
 write(OUT/'self_checks.json',checks)
 return checks

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--smoke-only',action='store_true');args=parser.parse_args()
 OUT.mkdir(parents=True,exist_ok=True)
 lock=(ROOT/'results/overnight_energy_v2/association_objective_gpu.lock').open('a')
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert torch.cuda.is_available();assert not (OUT/'run_started.json').exists(),'Refuse duplicate campaign.'
 if not (OUT/'plan.json').exists():data,plan,records,refrecords=prepare()
 else:
  data=RetrievalData();plan=json.loads((OUT/'plan.json').read_text());records=json.loads((OUT/'records.json').read_text());refrecords=json.loads((OUT/'retrain_records.json').read_text())
 for n,h in plan['sources'].items():assert sha(ROOT/n)==h,n
 for n,h in plan['runner_hashes'].items():assert sha(HERE/n)==h,n
 engine=Engine(data,plan);checks=smoke(engine,plan,records);print(json.dumps(checks),flush=True)
 if args.smoke_only:status(stage='smoke_passed',pid=os.getpid());return
 write(OUT/'run_started.json',dict(pid=os.getpid(),started_unix=time.time(),plan_sha256=sha(OUT/'plan.json')))
 empty=[[] for _ in range(10)]
 old=ROOT/'results/overnight_energy_v2/retrieval_fourway_v1'
 if (old/'ideal_warm/round_020.pt').exists():
  import shutil
  oldplan=json.loads((old/'plan.json').read_text())
  assert oldplan['annotation_sha256']==plan['annotation_sha256'] and oldplan['partitions']==plan['partitions']
  assert oldplan['sources']==plan['sources'] and oldplan['learning_rate']==plan['learning_rate']
  assert json.loads((old/'records.json').read_text())==records
  shutil.copytree(old/'ideal_warm',OUT/'ideal_warm')
  warm=torch.load(old/'ideal_warm/round_020.pt',map_location='cpu',weights_only=True)['theta'].to(engine.device)
  write(OUT/'warm_reuse.json',dict(source=str(old/'ideal_warm/round_020.pt'),sha256=sha(old/'ideal_warm/round_020.pt'),scope='Identical ideal-link preparatory warmup only; every ranked arm still starts at round1.'))
 else: warm,_=trajectory(engine,plan,records,'ideal_warm',empty,stop=20)
 engine.set_teacher(warm)
 learning,unlearning,_=engine.updates(warm,records[20],21,'hybrid_no_crossmodal',records[20]['forget'])
 np.savez_compressed(OUT/'fixed_updates.npz',learning=learning,unlearning=unlearning,channels=np.load(OUT/'channels.npy'))
 from radio import communicate
 preflight=[];selected=None
 for multiplier in [16,32,64]:
  c=dict(plan['config'],symbols_per_coordinate=multiplier);current=[]
  for method in ['hybrid','aircomp_only','aircomp_oma','oma_only']:
   status(stage='physical_preflight',multiplier=multiplier,method=method,pid=os.getpid())
   _,r=communicate(learning,unlearning,np.load(OUT/'channels.npy'),c,method,100021)
   entry=dict(multiplier=multiplier,method=method,**r);preflight.append(entry);current.append(r['status']=='feasible')
   write(OUT/'physical_preflight.json',preflight)
  if all(current):selected=c;break
 if selected is None:
  status(stage='blocked_no_common_feasible_deadline',pid=os.getpid(),note='No full comparison launched; physical CPU evidence available. No global infeasibility claim.');return
 write(OUT/'selected_radio_config.json',selected)
 ref,refprobs=trajectory(engine,plan,refrecords,'retrain',empty,stop=40)
 mia,_=engine.mia_scores(ref);write(OUT/'retrain/mia.json',mia)
 completed=[]
 for method in plan['methods']:
  theta,_=trajectory(engine,plan,records,method,records[20]['forget'],reference=refprobs,config=selected,stop=40)
  completed.append(dict(method=method,completed=theta is not None))
  if theta is not None:
   m,scores=engine.mia_scores(theta);m['auc_gap_vs_retrain']=abs(m['loss_score_auc']-mia['loss_score_auc']);write(OUT/method/'mia.json',m)
   np.save(OUT/method/'mia_scores.npy',scores)
  from report import build_report
  build_report()
 # Test evaluations only after all preregistered trajectories; never used to select settings.
 for method in ['retrain']+plan['methods']:
  path=OUT/method/'round_040.pt'
  if not path.exists():continue
  theta=torch.load(path,map_location='cpu',weights_only=True)['theta'].to(engine.device)
  metrics,_=engine.evaluate(theta,test=True);write(OUT/method/'final_test.json',metrics)
 from report import build_report
 build_report();status(stage='completed',methods=completed,pid=os.getpid())

if __name__=='__main__':
 try:main()
 except Exception as e:
  status(stage='failed',error=repr(e),pid=os.getpid());traceback.print_exc();raise

