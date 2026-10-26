"""Eight preregistered arms: one frozen UAV-demonstrated backbone, two datasets.
Existing algorithms/data stay read-only. No automatic rerun or result tuning.
"""
from pathlib import Path
import argparse,copy,fcntl,hashlib,importlib.util,json,os,subprocess,sys,time,traceback
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
OUT=ROOT/'results/overnight_energy_v2/uav_single_model_two_dataset_v1'
OLD=ROOT/'results/overnight_energy_v2/korea_mission_v1'
SOURCE=ROOT/'results/overnight_energy_v2/korea_joint_lr_sync_seed0_v5'
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
control=load('uav_frozen_controls',HERE.parent/'korea_round_lr_sync_rsitmd_seed0_v6/campaign.py')
pilot,old,base,core,clip,repeat=control.pilot,control.old,control.base,control.core,control.clip,control.repeat
RoundQueue=pilot.RoundQueue;SerialRoundQueue=control.SerialRoundQueue
import numpy as np
sys.path.insert(0,str(HERE))
from feature_engine import Engine,prepare_features
METHODS=list(base.ARMS)
write,sha=core.write,core.sha
def read(p):return json.loads(Path(p).read_text())
def state(dest=OUT,**kw):old.state(dest,**kw)

def adapted_arm():
 from types import SimpleNamespace
 synced=SimpleNamespace(**clip.__dict__)
 synced.vectors=control.synchronized_vectors('cliperase')
 synced.ORIGINAL_VECTORS=control.synchronized_vectors('proxy')
 return repeat.clone_function(old.run_arm,[
  ("q=clip.stream.Queue(requests,opt['parallel'],opt['period'])","q=RoundQueue(requests) if opt['parallel'] else SerialRoundQueue(requests,False,opt['period'])",1),
  ("site,endwindow=mission.window_at(t,geo)\n  if site is None:horizon_before(endwindow);t=endwindow;continue\n  kind,ids=mission.decide(q,t,fl,geo)",
   "site=min(fl//45,3);endwindow=geo['stops'][site]['end_s']\n  t=max(t,geo['stops'][site]['start_s'])\n  q.sync(fl,t,occupied)\n  kind,ids=q.decision(t,fl)",1),
  ("l,u,local=vecfn(engine,theta,plan,q,t,fl,kind,ids,proxy,tentative,opt['crossmodal'])",
   "engine.start_work()\n  l,u,local=vecfn(engine,theta,plan,q,t,fl,kind,ids,proxy,tentative,opt['crossmodal'])\n  compute=engine.finish_work()\n  assert np.isfinite(l).all() and np.isfinite(u).all()\n  if kind=='joint':assert np.linalg.norm(l)>0 and np.linalg.norm(u)>0\n  if kind=='fu':assert np.linalg.norm(l)==0 and np.linalg.norm(u)>0\n  if kind=='fl':assert np.linalg.norm(u)==0",1),
  ("FL_gradient_calls=10 if kind in ['fl','joint'] else 0,new_teacher_requests=",
   "compute=compute,FL_vector_norm=float(np.linalg.norm(l)),FU_vector_norm=float(np.linalg.norm(u)),FL_learning_rate=core.lr_at(21+fl),FU_learning_rate=core.lr_at(21+fl) if ids else None,\n    FL_gradient_calls=10 if kind in ['fl','joint'] else 0,new_teacher_requests=",1),
  ("if end>endwindow+1e-9:","if end>endwindow+1e-9:\n   write(ad/f'work_attempts/{attempt:04d}.json',dict(audit,status='computed_but_window_exceeded'))\n   raise RuntimeError('Fixed site window exceeded; no retiming or retry')",1),
  ("audit.update(status='transmitted',transaction=seq)","audit.update(status='transmitted',transaction=seq,FL_receiver_apply_s=end if kind in ['fl','joint'] else None,FU_receiver_apply_s=end if ids else None,FL_round_after=20+fl)",1),
  ("mission_clock_s=at,","mission_clock_s=at,clock_policy='round_triggered_two_dataset',model='frozen_CLIP_B32_residual_rank10',",1)
 ],{'RoundQueue':RoundQueue,'SerialRoundQueue':SerialRoundQueue,'clip':synced})

def register():
 assert not (OUT/'protocol.json').exists(),'Already registered; do not replace'
 model=read(OUT/'model_provenance.json');config=read(control.OUT/'protocol.json')['config']
 hashes={}
 for name,h in read(control.OUT/'protocol.json')['source_hashes'].items():
  if name.endswith('.py'):assert sha(ROOT/name)==h,name;hashes[name]=h
 for module in list(sys.modules.values()):
  file=getattr(module,'__file__',None)
  if not file:continue
  p=Path(file).resolve()
  if p.suffix=='.py' and p.is_file() and p.is_relative_to(ROOT) and not p.is_relative_to(ROOT/'.venv'):
   hashes[str(p.relative_to(ROOT))]=sha(p)
 for p in [HERE.parent/'korea_round_lr_sync_rsitmd_seed0_v6/campaign.py',HERE.parent/'korea_round_lr_sync_rsitmd_seed0_v6/vectors_cliperase.py',HERE.parent/'korea_round_lr_sync_rsitmd_seed0_v6/vectors_proxy.py',HERE.parent/'korea_round_lr_sync_rsitmd_seed0_v6/comparison_schedule.py']:
  hashes[str(p.relative_to(ROOT))]=sha(p)
 checks={}
 geo=read(OLD/'mission_spec.json')
 for dataset,src in [('RSITMD','retrieval_long200_v3'),('RSICD','rsicd_cap100_v1')]:
  prior=SOURCE/f'{dataset.lower()}_seed0';original=read(prior/'plan.json');req=read(prior/'requests.json')
  records_path=ROOT/f'results/overnight_energy_v2/{src}/records.json'
  refs_path=ROOT/f'results/overnight_energy_v2/{src}/retrain_records.json'
  records=read(records_path);refs=read(refs_path)
  keys=['dataset','seed','training_images','validation_images','official_test_images','unused_official_validation_images','annotation_sha256','partitions','forgotten_images','forget_probe','mia_member_images','mia_nonmember_images','gallery_images','batch_size','learning_rate','forget_weight','crossmodal_weight','unlearning_rounds']
  plan={k:copy.deepcopy(original[k]) for k in keys if k in original}
  plan.update(version=HERE.name,dataset=dataset,seed=0,config=config,rounds=200,initial_learning_rounds=20,eval_rounds=list(range(0,201,5)),model='CLIP ViT-B/32 frozen + 2 residual rank10 adapters',trainable_coordinates=20480,methods=METHODS,
   optimizer='SGD, no momentum',FL_rate='core.lr_at(k)',FU_rate='core.lr_at(21+completed_adaptation_FL)',reference='Reporting only',scope='Whole-system comparison: method objectives differ. Synthetic UAV mission and remote-sensing image-caption replay.')
  assert len(plan['gallery_images'])==(814 if dataset=='RSITMD' else 920)
  assert len(req)==45 and all(r['required_steps']==5 and r['arrival_s'] is None and r['deadline_s'] is None for r in req)
  assert sorted(set(r['release_before_total_FL_round'] for r in req))==[26,42,59,75,96,120,141,165,186]
  forgotten=set(plan['forgotten_images']);assert forgotten=={i for r in req for i in r['images']}
  assert forgotten<=set(plan['gallery_images'])
  assert len(records)==len(refs)==200
  assert sorted(sum(plan['partitions'],[]))==sorted(sum(records[0]['retained'],[]))
  assert set(sum(plan['partitions'],[])).isdisjoint(plan['validation_images']+plan['official_test_images'])
  assert all(set(b).isdisjoint(forgotten) for r in refs for b in r['batches'])
  seen=[set() for _ in range(10)]
  for r in records[:20]:
   for u,b in enumerate(r['batches']):seen[u].update(b)
  assert all(set(r['images'])<=seen[r['user']] for r in req)
  proxy_records=core.proxy_schedule(records,forgotten)
  dest=OUT/f'{dataset.lower()}_seed0'
  for n,value in [('plan',plan),('requests',req),('initial_records',records[:20]),('proxy_records',proxy_records),('reference_records',refs)]:write(dest/(n+'.json'),value)
  checks[dataset]=dict(joint=pilot.replay(req,geo),serial=control.replay(req,geo,dict(fl=.02048,fu=.02048)),gallery_coverage=True,all_requests_previously_seen=True,test_disjoint=True)
  for p in [prior/'plan.json',prior/'requests.json',records_path,refs_path]:hashes[str(p.relative_to(ROOT))]=sha(p)
  for p in dest.glob('*.json'):hashes[str(p.relative_to(ROOT))]=sha(p)
 for p in [OLD/'mission_spec.json',OLD/'channels_by_site.npy',OUT/'model_provenance.json']:
  hashes[str(p.relative_to(ROOT))]=sha(p)
 for p in HERE.glob('*.py'):hashes[str(p.relative_to(ROOT))]=sha(p)
 adapted_arm()
 write(OUT/'cpu_preflight.json',dict(passed=True,cases=checks))
 write(OUT/'protocol.json',dict(version=HERE.name,registered_unix=time.time(),authorization='2026-10-25 user: start the one-UAV-model/two-dataset experiment; recommended RSITMD/RSICD retrieval configuration accepted. seed0, four methods each, eight arms once.',
  datasets=['RSITMD','RSICD'],seed=0,methods=base.ARMS,source_hashes=hashes,model=model,
  architecture=dict(backbone='CLIP ViT-B/32',feature_dim=512,adapter='z + W_up GELU(W_down z), no biases',bottleneck=10,per_modality_coordinates=10240,total_trainable_coordinates=20480,initialization='seed0; W_down normal std1/sqrt512; W_up zeros; identity residual initially',logit_scale='Frozen checkpoint scalar; never transmitted'),
  config=config,effective_radio=dict(digital_bits=6,OFDMA_tones=128,bandwidth_Hz=1000000,TX_cap_J=.10,max_UAV_power_W=.32,serial_period_s=.1024,note='Frozen certified transport overrides legacy config quantization for actual digital paths; verify emitted transaction metadata. Prefix has no FU payload.'),
  rounds=dict(initial20=20,adaptation=180,total_FL=200,proxy=20,reference=200),FU_requests=45,FU_steps_per_request=5,
  training='Fresh new-model initial20/proxy20/reference200 for each dataset. Same initial vector/prefix/reference shared by all four methods; old B16 checkpoints/results never used as new results.',
  schedule='Same round releases for both datasets:26,42,59,75,96,120,141,165,186. Four map sites with45adaptiveFL each. Shared cosine FU rule. Serial FU pausesFL; proposed joint advancesFL. Numerical step rates may differ. Last FL200 zero-update preserved. No30s deadline comparison.',
  cache='Only frozen deterministic pre-adapter embeddings; no encoder backward. Cold extraction/cache operations and retained/deleted sample lineage are recorded. Charge per independent deployment; report evaluation-only costs separately. Physical cache erasure not implemented or claimed.',
  accounting='Record server wall time, feature/adapter operations and memory. RTX3080 timings are not UAV timings. Total UAV energy remains component-based with unknown residual; do not treat missing calibration as0. Reference/preflight separate from training. Uplink-only mission clock excludes compute/decoder/control/downlink and cannot establish actual in-flight completion.',
  frozen_policy='No tuning, added seed, automatic retry, best-checkpoint selection, request removal or suppressed raw values. Failed independent arms are preserved and remaining registered arms continue.',
  reference_policy='Reference/test/validation never used to select rate, settings, stop or checkpoint. Report final200 and raw curves, KL/disagreement/MIA, procedure counts, energy/time.',
  provenance='Backbone informed by GRaD-Nav++ onboard CLIP-B32 use; this residual-adapter retrieval configuration is our adaptation, not its navigation policy or proof of onboard training. Actual OSM Daejeon coordinates, synthetic mission, existing RSITMD/RSICD imagery; ©OpenStreetMap contributors/ODbL.',
  literature=['https://arxiv.org/html/2506.14009v2','https://github.com/Qianzhong-Chen/grad_nav','https://huggingface.co/openai/clip-vit-base-patch32']))
 print('REGISTERED_8_ARMS_CPU_PASSED',flush=True)

def verify(model_files=False):
 p=read(OUT/'protocol.json')
 for name,h in p['source_hashes'].items():assert sha(ROOT/name)==h,name
 if model_files:
  for name,meta in p['model']['files'].items():assert sha(ROOT/p['model']['checkpoint']/name)==meta['sha256'],name
 return p

def pure_trajectory(engine,plan,records,ch,kind,dest):
 from train import update_theta
 import radio as legacy
 folder=dest/kind;assert not (folder/'started.json').exists();write(folder/'started.json',dict(unix=time.time()))
 theta=engine.theta0.clone();energy=airtime=0.;rows=[];total_seconds=0.
 for rec in records:
  k=rec['round'];state(dest,stage=kind,round=k,total=len(records));state(stage=kind,dataset=plan['dataset'],round=k,total=len(records))
  engine.plan['learning_rate']=core.lr_at(k) if kind=='reference200' else .0035
  engine.start_work();l,_,_=engine.updates(theta,rec,k,'retrain',[[] for _ in range(10)]);work=engine.finish_work();total_seconds+=work['server_seconds']
  if kind=='reference200':vec=l.sum(0);rm=dict(status='ideal_reference',energy_bound_j=None,uplink_s=None)
  else:
   vec,rm=legacy.communicate(l,np.zeros_like(l),ch,plan['config'],'hybrid',(240000 if kind=='proxy20' else 100000)+k)
   assert vec is not None and rm['status']=='feasible',rm
   assert rm['energy_bound_j']<=.10*(1+2e-6)
   energy+=rm['energy_bound_j'];airtime+=rm['uplink_s']
  theta=update_theta(theta,vec);row=dict(round=k,radio=rm,compute=work)
  if k%5==0:
   row['evaluation'],_=engine.evaluate(theta,None);core.save_theta(folder/'checkpoint.pt',theta,k)
  rows.append(row);write(folder/f'rounds/{k:03d}.json',row)
  write(folder/'progress.json',dict(round=k,total=len(records),server_compute_seconds=total_seconds))
 core.save_theta(folder/'final.pt',theta,len(records));write(folder/'rounds.json',rows)
 write(folder/'finished.json',dict(status='completed',rounds=len(records),TX_energy_j=energy if kind!='reference200' else None,uplink_s=airtime if kind!='reference200' else None,server_compute_seconds=total_seconds))
 return theta,dict(energy_j=energy,uplink_s=airtime,server_compute_seconds=total_seconds)

def validate_arm(ad,opt):
 manifest=read(ad/'manifest.json');assert manifest['status']=='completed_200FL_and_requests',manifest['status']
 tx=[read(p) for p in sorted((ad/'transactions').glob('*.json'))]
 work=[read(p) for p in sorted((ad/'work_attempts').glob('*.json'))]
 counts={k:sum(t['kind']==k for t in tx) for k in ['joint','fl','fu']}
 assert counts==(dict(joint=45,fl=135,fu=0) if opt['parallel'] else dict(joint=0,fl=180,fu=45))
 assert manifest['completed_FL_rounds']==200
 requests=read(ad/'request_records.json');assert len(requests)==45 and all(r['steps']==5 for r in requests)
 assert sum(len(t['FU_request_ids']) for t in tx)==225
 for a in work:
  assert a['status']=='transmitted'
  if a['kind']=='joint':assert a['FL_vector_norm']>0 and a['FU_vector_norm']>0 and a['FL_receiver_apply_s']==a['FU_receiver_apply_s']
  if a['kind']=='fu':assert a['FL_round']==a['FL_round_after'] and a['FL_vector_norm']==0 and a['FU_vector_norm']>0
  if a['FU_request_ids']:assert a['FU_learning_rate']==core.lr_at(a['FL_round']+1)>0
 write(ad/'transmission_verification.json',dict(passed=True,counts=counts,request_steps=225,all45_requests_5steps=True,actual_positive_uplink_transactions=sum(t['radio']['uplink_s']>0 for t in tx),logical_updates=len(tx)))
 write(ad/'compute_summary.json',dict(adaptation_server_seconds=sum(a['compute']['server_seconds'] for a in work),work_attempts=len(work),backbone_backward_calls=0,backbone_forward_calls_in_adaptation=0,total_UAV_energy_j=None,scope='Adapter/teacher gradient preparation server wall time. Cold extraction and initial/proxy costs separate; unknown target-device calibration.'))

def run_dataset(dataset,protocol):
 import torch
 from dataset import RetrievalData
 dest=OUT/f'{dataset.lower()}_seed0';assert not (dest/'started.json').exists()
 write(dest/'started.json',dict(unix=time.time(),identity=base.identity(os.getpid())))
 plan=read(dest/'plan.json');requests=read(dest/'requests.json');geo=read(OLD/'mission_spec.json');channels=np.load(OLD/'channels_by_site.npy')
 data=(RetrievalData if dataset=='RSITMD' else base.RetrievalData)()
 status=lambda **x:(state(dest,**x),state(dataset=dataset,**x))
 payload,cache=prepare_features(data,plan,ROOT/protocol['model']['checkpoint'],dest/'feature_cache',write,status)
 engine=Engine(data,plan,payload);batch=read(dest/'initial_records.json')[0]['batches'][0]
 engine.start_work();gradient,loss=engine.gradient(engine.theta0,batch,1);work=engine.finish_work()
 assert torch.isfinite(gradient).all() and gradient[:-1].numel()==20480
 assert float(gradient[:10240].norm())>0 and float(gradient[10240:20480].norm())>0 and gradient[-1]==0
 _,next_loss=engine.gradient(engine.theta0-1e-5*gradient,batch,1)
 assert next_loss<=loss+1e-6
 write(dest/'actual_model_preflight.json',dict(passed=True,initial_loss=loss,small_descent_loss=next_loss,visual_gradient_norm=float(gradient[:10240].norm()),text_gradient_norm=float(gradient[10240:20480].norm()),compute=work,no_encoder_backward=True,scope='Initial training batch numerical check only; not selection by validation/reference.'))
 warm,warmcost=pure_trajectory(engine,plan,read(dest/'initial_records.json'),channels[0],'initial20',dest)
 proxy,proxycost=pure_trajectory(engine,plan,read(dest/'proxy_records.json'),channels[0],'proxy20',dest)
 prefix={k:warmcost[k]+proxycost[k] for k in warmcost};prefix.update(charged_per_independent_arm=True,feature_extraction_cost_separate='feature_cache/manifest.json')
 write(dest/'common_prefix_cost.json',prefix);engine.set_teacher(warm)
 # Actual full-dimensional FL and both FU objectives before ranked arms.
 q=RoundQueue(requests);first=min(r['release_after_adaptation_FL'] for r in requests);q.sync(first,0.,0.);kind,ids=q.decision(0.,first)
 engine.start_work();l,u,meta=control.synchronized_vectors('cliperase')(engine,warm,plan,q,0.,first,kind,ids,core.Teacher(engine,proxy),{},True)
 l2,old_u,old_meta=control.synchronized_vectors('proxy')(engine,warm,plan,q,0.,first,kind,ids,core.Teacher(engine,proxy),{},True);work=engine.finish_work()
 assert np.array_equal(l,l2) and np.linalg.norm(l)>0 and np.linalg.norm(u)>0 and np.linalg.norm(old_u)>0
 for vector in [l,u,old_u]:assert np.linalg.norm(vector[:,:10240])>0 and np.linalg.norm(vector[:,10240:])>0
 previous=base.OUT;base.OUT=dest
 try:
  for site,ch in enumerate(channels):base.physical_checks(l,u,old_u,ch,plan['config'],f'actual_site{site}_preflight.json')
 finally:base.OUT=previous
 write(dest/'actual_FL_FU_preflight.json',dict(passed=True,compute=work,FL_norm=float(np.linalg.norm(l)),cliperase_FU_norm=float(np.linalg.norm(u)),proxy_FU_norm=float(np.linalg.norm(old_u)),cliperase=meta,proxy=old_meta))
 ref,_=pure_trajectory(engine,plan,read(dest/'reference_records.json'),channels[0],'reference200',dest)
 refmetrics,refprob=engine.evaluate(ref,None);refmia,_=engine.mia_scores(ref);threshold=.95*refmetrics['retained']['mR']
 write(dest/'reference_metrics.json',dict(metrics=refmetrics,retained_mR_threshold=threshold,reporting_only=True));write(dest/'reference_mia.json',refmia)
 fullq=RoundQueue(requests);fullq.sync(180,0.,0.);ev=clip.stream.evaluation(engine,warm,refprob,fullq,0.)
 assert len(ev['per_request'])==45
 write(dest/'initial_evaluation_preflight.json',dict(passed=True,evaluation=ev))
 write(dest/'common_checkpoint_hashes.json',{k:sha(dest/k/'final.pt') for k in ['initial20','proxy20','reference200']})
 results=[]
 for method in METHODS:
  verify();status(stage='running_arm',method=method);started=time.time();ad=dest/method
  try:
   adapted_arm()(engine,plan,requests,geo,channels,method,base.ARMS[method],prefix,warm,proxy,refprob,threshold,dest,0)
   validate_arm(ad,base.ARMS[method]);results.append(dict(method=method,status='completed'))
  except Exception as e:
   write(ad/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),unix=time.time()))
   results.append(dict(method=method,status='failed',error=repr(e)))
  write(ad/'runtime_seconds.json',dict(elapsed_server_wall_seconds=time.time()-started,scope='Simulator execution; not UAV deployment latency.'))
 write(dest/'finished.json',dict(status='all_four_arms_terminal',results=results,unix=time.time(),no_retries=True))
 del engine,data,payload,refprob
 import gc;gc.collect();torch.cuda.empty_cache()
 return results

def main():
 protocol=verify(model_files=True)
 own=(OUT/'campaign.lock').open('a');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (OUT/'started.json').exists(),'Duplicate/retry forbidden'
 gpu=(ROOT/'results/overnight_energy_v2/association_objective_gpu.lock').open('a');fcntl.flock(gpu,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU occupied'
 write(OUT/'started.json',dict(unix=time.time(),identity=base.identity(os.getpid()),protocol_sha256=sha(OUT/'protocol.json')))
 results=[]
 for dataset in ['RSITMD','RSICD']:
  try:results.append(dict(dataset=dataset,arms=run_dataset(dataset,protocol)))
  except Exception as e:
   write(OUT/f'{dataset.lower()}_seed0/failure.json',dict(error=repr(e),traceback=traceback.format_exc(),unix=time.time()))
   results.append(dict(dataset=dataset,status='failed_before_all_arms',error=repr(e)))
 write(OUT/'finished.json',dict(status='all_registered_datasets_terminal',results=results,unix=time.time(),no_retries=True));state(stage='terminal',results=results)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--register',action='store_true');parser.add_argument('--verify',action='store_true');args=parser.parse_args()
 try:
  if args.register:register()
  elif args.verify:verify(model_files=True);print('ALL_INPUTS_MODEL_SOURCES_VERIFIED',flush=True)
  else:main()
 except Exception as e:
  write(OUT/('registration_failure.json' if args.register else 'failure.json'),dict(error=repr(e),traceback=traceback.format_exc(),unix=time.time()));raise
 finally:base.parent.common.base.close()
