"""Preregistered geographic mission; imports frozen algorithms read-only."""
from pathlib import Path
import os,sys,json,time,fcntl,subprocess,traceback,importlib.util,hashlib,copy
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
spec=importlib.util.spec_from_file_location('mission_seed_base',ROOT/'tools/overnight_energy_v2/three_seed_cap100_v1/repeat.py')
repeat=importlib.util.module_from_spec(spec);spec.loader.exec_module(repeat)
sys.path.insert(0,str(HERE))
import mission
import numpy as np
base=repeat.base;core=base.core;clip=base.clip
OUT=ROOT/'results/overnight_energy_v2/korea_mission_v1'
RUNS=[(d,s) for s in [0,1,2] for d in ['RSITMD','RSICD']]

def write(path,obj):core.write(path,obj)
def state(dest,**obj):write(dest/'status.json',dict(unix=time.time(),pid=os.getpid(),**obj))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def source_for(dataset,seed):
 if seed:return ROOT/'results/overnight_energy_v2/three_seed_cap100_v1'/f'{dataset.lower()}_seed{seed}'
 return ROOT/'results/overnight_energy_v2'/('retrieval_long200_v3' if dataset=='RSITMD' else 'rsicd_cap100_v1')
def request_path(dataset,seed):
 if dataset=='RSITMD' and seed==0:return ROOT/'results/overnight_energy_v2/fourway_cap100_v1/requests.json'
 return source_for(dataset,seed)/'requests.json'

def guard_checks(l,u,ch,c,path):
 rows={}
 for kind,a,b in [('FL',l,np.zeros_like(l)),('FU',np.zeros_like(l),u)]:
  v,m=base.parent.guard.analog(a,b,ch,c,500000)
  assert v is not None and m['status']=='feasible',(kind,m)
  assert np.isfinite(v).all() and m['conditional_rmse']<=m['epsilon']*(1+3e-5)+1e-9
  assert m['energy_bound_j']<=.1*(1+2e-6) and m['max_power_w']<=c['power_limit_w']*(1+2e-6)
  rows[kind]=m
 write(path,dict(passed=True,cases=rows))

def register():
 OUT.mkdir(parents=True,exist_ok=True)
 if (OUT/'protocol.json').exists():verify();return
 geo=mission.prepare();ch=np.load(OUT/'channels_by_site.npy')
 req=mission.requests_on_mission(json.loads(request_path('RSITMD',0).read_text()),geo)
 schedule={}
 for name,opt in base.ARMS.items():
  for slow in [False,True]:
   duration=dict(fl=.6 if slow else .02048,fu=.4 if slow else .02048,joint=.8 if slow else .02048)
   schedule[name+('_stress' if slow else '')]=mission.replay(req,geo,clip.stream.Queue,opt['parallel'],opt['period'],duration)
 # No GPU or final-reference evaluation is used for this physical audit.
 config=json.loads((base.parent.OUT/'preflight.json').read_text())['config']
 z=np.load(clip.OUT/'fixed_training_updates.npz');physical={}
 oldout=base.OUT;base.OUT=OUT
 for site,links in enumerate(ch):
  base.physical_checks(z['FL'],z['FU'],z['old_FU'],links,config,f'cpu_site{site}.json')
  guard_checks(z['FL'],z['FU'],links,config,OUT/f'cpu_site{site}_guard.json')
  physical[str(site)]=sha(OUT/f'cpu_site{site}.json')
 base.OUT=oldout
 hashes=dict(json.loads((repeat.OUT/'protocol.json').read_text())['source_hashes'])
 for p in HERE.glob('*.py'):hashes[str(p.relative_to(ROOT))]=sha(p)
 for p in [OUT/'mission_spec.json',OUT/'channels_by_site.npy',mission.MAP/'osm_snapshot.json']:
  hashes[str(p.relative_to(ROOT))]=sha(p)
 write(OUT/'cpu_preflight.json',dict(passed=True,scheduler_replays=schedule,physical_checks=physical,
   scope='Fixed real20480-coordinate training updates; new geographic channels. No learned performance result.',
   actual_each_seed='Additional fixed-initial-model update check before any ranked mission arm.'))
 write(OUT/'protocol.json',dict(version='korea_mission_v1',registered_unix=time.time(),runs=[dict(dataset=d,seed=s) for d,s in RUNS],
   authorization='User requested a Korean-map UAV swarm mission after the current three-seed experiments, all result CSVs and graphs, on2026-10-24.',
   dependency=dict(folder='three_seed_cap100_v1',pid=249494,start_ticks='884722863',argv_suffix='three_seed_cap100_v1/campaign_queue.py'),
   methods=base.ARMS,labels=base.LABELS,config=config,
   common='10UAV,200FL including20 initial,45requests x5steps; existing objectives/radio solvers unchanged; same route/releases/requests and paired training/noise seeds across all four methods. DigitalQ6,128tones,1MHz,TXcap0.10J.',
   geographic_change='OSM coordinates and moving formation determine four channel matrices. Causal inspection FL opportunities and deletion bursts use the common mission-service clock. Hover-only uplinks; deferred work drains after return.',
   model_data='FrozenCLIP ViT-B/16 visual/text prompts; RSITMD/RSICD image-caption replay, not Korean geotagged mission imagery. No new data-arrival pool; each inspection releases a training-work opportunity over the fixed available retained pool.',
   losses='Proposed and sequential digital retain CLIPErase-inspired loss. Serial AirComp and AirComp+OFDMA retain their existing safe-proxy loss. Full-system comparison, not isolated radio causality.',
   reused='Matching dataset/seed initial20,proxy20,final-retained-reference200 checkpoints reused because premission data and learning are unchanged. Initial/proxy physical costs charged to every independent arm. Record checkpoint hashes. Reference only evaluates, never tunes or teaches.',
   evaluation='Raw mean and sampleSD over seeds0/1/2, no smoothing. Final and common mission horizon, deadline completion, retainedmR/loss, mean/worstKL,disagreement,MIA. Target computed perseed before aggregation; failures visible.',
   timing_energy='Pure uplink separately from modeled mission-service clock excluding local compute/recovery/control/downlink. TXbound supplementary; totalenergy calibration unavailable. Flight/hover energy unknown, distance/time recorded. No empirical deployment or deletion guarantee.',
   nonclaims='This is a synthetic mission on real map geometry, not a flight authorization/obstacle-aware operational plan. Performance or NOMA superiority is not assumed. Underload can reduce parallel advantage.',
   source_hashes=hashes))
 write(OUT/'queue_registered.json',dict(status='registered',unix=time.time(),runs=[dict(dataset=d,seed=s) for d,s in RUNS],after='three_seed_cap100_v1'))

def verify():
 p=json.loads((OUT/'protocol.json').read_text())
 for f,h in p['source_hashes'].items():assert sha(ROOT/f)==h,f
 return p

def run_arm(engine,plan,requests,geo,channels,name,opt,prefix,warm,proxy_theta,refprob,threshold,dest,seed):
 from train import update_theta
 import torch
 ad=dest/name;ad.mkdir(parents=True,exist_ok=True);assert not (ad/'started.json').exists()
 write(ad/'started.json',dict(unix=time.time(),policy=opt))
 theta=warm.clone();q=clip.stream.Queue(requests,opt['parallel'],opt['period']);teachers={};proxy=core.Teacher(engine,proxy_theta)
 vecfn=clip.vectors if opt['objective']=='cliperase_inspired' else clip.ORIGINAL_VECTORS
 vecfn=repeat.clone_function(vecfn,[('np.random.default_rng(','seeded_rng(',2)],{'seeded_rng':lambda k:repeat.seed_rng(k,seed*10000000)})
 radio=base.parent.guard if name=='proposed_fu_guard_cliperase' else base.SequentialRadio if name=='sequential_digital_periodic' else base.parent.common
 t=energy=occupied=0.;fl=seq=attempt=0;last_fl=-1;horizon=False;target=None;hits=[]
 def point(at,ev=None):
  return dict(sequence=seq,completed_FL_rounds=20+fl,mission_clock_s=at,
   adaptation_elapsed_service_s=at,adaptation_uplink_occupied_s=occupied,adaptation_tx_energy_bound_j=energy,
   cumulative_elapsed_service_s=prefix['uplink_s']+at,cumulative_uplink_occupied_s=prefix['uplink_s']+occupied,
   cumulative_tx_energy_bound_j=prefix['energy_j']+energy,requests=q.metrics(at),evaluation=ev,joint_target=target)
 def horizon_before(newtime):
  nonlocal horizon
  if not horizon and newtime>=geo['mission_end_s']-1e-9:
   ev=clip.stream.evaluation(engine,theta,refprob,q,geo['mission_end_s'])
   write(ad/'common_horizon.json',point(geo['mission_end_s'],ev));horizon=True
 for _ in range(10000):
  if fl==180 and q.metrics(t)['procedure_completed']==45:break
  site,endwindow=mission.window_at(t,geo)
  if site is None:horizon_before(endwindow);t=endwindow;continue
  kind,ids=mission.decide(q,t,fl,geo)
  if kind=='wait':horizon_before(ids);t=ids;continue
  if kind=='done':raise RuntimeError('Scheduler ended before prescribed work completion')
  state(dest,stage='mission_training',method=name,transaction=seq+1,FL_round=20+fl,mission_time_s=t,site=site,requests=q.metrics(t))
  attempt+=1;tentative=dict(teachers)
  l,u,local=vecfn(engine,theta,plan,q,t,fl,kind,ids,proxy,tentative,opt['crossmodal'])
  vector,rm=(radio.digital_fu(u,channels[site],plan['config']) if kind=='fu' and opt['digital'] else radio.analog(l,u,channels[site],plan['config'],500000+seq+seed*10000000))
  audit=dict(attempt=attempt,kind=kind,FU_request_ids=ids,site=site,mission_time_s=t,FL_round=20+fl,local=local,
    FL_gradient_calls=10 if kind in ['fl','joint'] else 0,new_teacher_requests=[i for i in tentative if i not in teachers],
    status='prepared',radio=rm,energy_scope='Work counts only; per-device calibration pending. Unsent attempts still consumed computation.')
  if vector is None:
   audit['status']='radio_infeasible';write(ad/f'work_attempts/{attempt:04d}.json',audit)
   write(ad/'request_records.json',list(q.items.values()))
   write(ad/'manifest.json',dict(status='incomplete_no_feasible_radio_design',kind=kind,radio=rm,**point(t)));return
  assert rm['conditional_rmse']<=rm['epsilon']*(1+3e-5)+1e-9
  assert rm['energy_bound_j']<=.1*(1+2e-6)
  assert rm['max_power_w']<=plan['config']['power_limit_w']*(1+2e-6)
  start=t;end=t+rm['uplink_s']
  if end>endwindow+1e-9:
   audit['status']='computed_but_deferred_hover_window';write(ad/f'work_attempts/{attempt:04d}.json',audit)
   nxt=mission.next_window(t,geo);horizon_before(nxt);t=nxt;continue
  # Commit only a fully transmitted update and its first-service teachers.
  teachers=tentative;theta=update_theta(theta,vector)
  if kind in ['fl','joint']:fl+=1
  q.commit(ids,start,end);t=end;seq+=1;energy+=rm['energy_bound_j'];occupied+=rm['uplink_s']
  audit.update(status='transmitted',transaction=seq);write(ad/f'work_attempts/{attempt:04d}.json',audit)
  due=(fl!=last_fl and fl%5==0) or any(q.items[i]['completed_s']==end for i in ids)
  ev=None
  if due:
   ev=clip.stream.evaluation(engine,theta,refprob,q,t);last_fl=fl
   if fl%5==0:
    ok=q.metrics(t)['procedure_completed']==45 and ev['retained']['mR']>=threshold and ev['forget']['KL_retrain_to_candidate']<=.01 and ev['worst_request_KL']<=.01
    # Distinct FL observations only; a burst of FU-only evaluations is not3FL checks.
    if not hits or hits[-1]['fl_round']!=20+fl:
     hits=hits+[dict(fl_round=20+fl,time_s=t)] if ok else []
    elif not ok:hits=[]
    if len(hits)>=3 and target is None:target=dict(confirmation_FL_round=20+fl,mission_clock_s=t,uplink_s=prefix['uplink_s']+occupied,energy_j=prefix['energy_j']+energy,first_pass=hits[0])
   write(ad/f'evaluations/{seq:04d}.json',point(t,ev));core.save_theta(ad/'checkpoint.pt',theta,20+fl)
  row=dict(**point(t,ev),kind=kind,FU_request_ids=ids,radio=rm,local=local,site=site,start_s=start,end_s=end)
  write(ad/f'transactions/{seq:04d}.json',row);write(ad/'latest.json',row)
 else:raise RuntimeError('Mission transaction/wait safety limit')
 assert fl==180 and q.metrics(t)['procedure_completed']==45
 final=clip.stream.evaluation(engine,theta,refprob,q,t)
 if not horizon:write(ad/'common_horizon.json',dict(**point(geo['mission_end_s'],final),note='Workload completed before common mission horizon; no extra learning.'))
 mia,_=engine.mia_scores(theta);refmia=json.loads((dest/'reference_mia.json').read_text());mia['auc_gap_vs_retrain']=abs(mia['loss_score_auc']-refmia['loss_score_auc'])
 write(ad/'mia.json',mia);core.save_theta(ad/'final.pt',theta,20+fl)
 write(ad/'request_records.json',list(q.items.values()))
 write(ad/'manifest.json',dict(status='completed_200FL_and_requests',**point(t,final)))

def run(dataset,seed):
 protocol=verify();dest=OUT/f'{dataset.lower()}_seed{seed}';dest.mkdir(parents=True,exist_ok=True)
 own=(dest/'launch.lock').open('a');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (dest/'started.json').exists(),'No automatic duplicate run'
 gpu=(ROOT/'results/overnight_energy_v2/association_objective_gpu.lock').open('a');state(dest,stage='waiting_for_GPU_lock');fcntl.flock(gpu,fcntl.LOCK_EX)
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU occupied without commonlock'
 source=source_for(dataset,seed);plan=json.loads((source/'plan.json').read_text());plan['seed']=seed;plan['config']=protocol['config']
 geo=json.loads((OUT/'mission_spec.json').read_text());req=mission.requests_on_mission(json.loads(request_path(dataset,seed).read_text()),geo)
 warmfile=source/'proposed/round_020.pt';proxyfile=source/'proxy20/final.pt';reffile=source/'reference200/final.pt';costfile=source/'common_prefix_cost.json'
 if dataset=='RSITMD' and seed==0:
  proxyfile=ROOT/'results/overnight_energy_v2/proxy_objective_v2/proxies/analog/round_020.pt'
  reffile=source/'retrain/round_200.pt';costfile=ROOT/'results/overnight_energy_v2/fourway_cap100_v1/common_prefix_cost.json'
 for p in [warmfile,proxyfile,reffile,costfile]:assert p.exists(),p
 write(dest/'reused_checkpoints.json',dict(files={str(p.relative_to(ROOT)):sha(p) for p in [warmfile,proxyfile,reffile,costfile]},policy=protocol['reused']))
 write(dest/'plan.json',plan);write(dest/'requests.json',req);write(dest/'started.json',dict(unix=time.time(),identity=base.identity(os.getpid())))
 import train
 from dataset import RetrievalData as RSITMDData
 init=repeat.clone_function(train.Engine.__init__,[('torch.manual_seed(0)',"torch.manual_seed(plan['seed'])",1),('torch.cuda.manual_seed_all(0)',"torch.cuda.manual_seed_all(plan['seed'])",1)])
 Engine=type('MissionEngine',(train.Engine,),{'__init__':init})
 engine=Engine((RSITMDData if dataset=='RSITMD' else base.RetrievalData)(),dict(plan))
 warm=core.load_theta(warmfile,engine.device);proxy=core.load_theta(proxyfile,engine.device);ref=core.load_theta(reffile,engine.device)
 refmetrics,refprob=engine.evaluate(ref,None);refmia,_=engine.mia_scores(ref);threshold=.95*refmetrics['retained']['mR']
 write(dest/'reference_mia.json',refmia);write(dest/'reference_metrics.json',dict(metrics=refmetrics,retained_mR_threshold=threshold,reporting_only=True))
 prefix=json.loads(costfile.read_text());write(dest/'common_prefix_cost.json',prefix);channels=np.load(OUT/'channels_by_site.npy')
 # Validation of actual seed updates uses only initial model and local training data.
 # All four sites are tested; failures are preserved rather than tuning geometry.
 state(dest,stage='actual_initial_update_preflight')
 synthetic=copy.deepcopy(req)
 for r in synthetic:r['arrival_s']=max(0,r['arrival_s']-5.)
 q=clip.stream.Queue(synthetic,True,0);kind,ids=q.decision(0.,0);teacher=core.Teacher(engine,proxy)
 rng={'seeded_rng':lambda k:repeat.seed_rng(k,seed*10000000)}
 fn=repeat.clone_function(clip.vectors,[('np.random.default_rng(','seeded_rng(',2)],rng)
 oldfn=repeat.clone_function(clip.ORIGINAL_VECTORS,[('np.random.default_rng(','seeded_rng(',2)],rng)
 l,u,_=fn(engine,warm,plan,q,0.,0,kind,ids,teacher,{},True)
 l2,old,_=oldfn(engine,warm,plan,q,0.,0,kind,ids,teacher,{},True);assert np.allclose(l,l2,rtol=1e-6,atol=1e-8)
 oldout=base.OUT;base.OUT=dest
 for site,ch in enumerate(channels):
  base.physical_checks(l,u,old,ch,plan['config'],f'actual_site{site}_preflight.json')
  guard_checks(l,u,ch,plan['config'],dest/f'actual_site{site}_guard.json')
 base.OUT=oldout
 done=[];fail=[]
 for name,opt in base.ARMS.items():
  verify();state(dest,stage='running',method=name,finished=done);started=time.time()
  try:
   run_arm(engine,plan,req,geo,channels,name,opt,prefix,warm,proxy,refprob,threshold,dest,seed)
   st=json.loads((dest/name/'manifest.json').read_text())['status']
   if st!='completed_200FL_and_requests':fail.append(dict(method=name,status=st))
  except Exception as exc:
   fail.append(dict(method=name,error=repr(exc)));write(dest/name/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc()))
  write(dest/name/'runtime_seconds.json',dict(started_unix=started,finished_unix=time.time(),elapsed_server_wall_seconds=time.time()-started,scope='Simulator runtime only'))
  done.append(name)
 write(dest/'finished.json',dict(status='all_registered_arms_finished_or_failed',failures=fail,unix=time.time()));state(dest,stage='completed',failures=fail)

if __name__=='__main__':
 dest=OUT
 try:
  if '--register' in sys.argv:register();print('MISSION_REGISTRATION_PASSED',flush=True)
  else:
   import argparse
   p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['RSITMD','RSICD'],required=True);p.add_argument('--seed',type=int,choices=[0,1,2],required=True);a=p.parse_args()
   dest=OUT/f'{a.dataset.lower()}_seed{a.seed}';run(a.dataset,a.seed)
 except Exception as exc:
  write(dest/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),unix=time.time()));state(dest,stage='failed',error=repr(exc));raise
 finally:base.parent.common.base.close()
