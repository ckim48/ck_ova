"""Second dataset only. Frozen RSITMD algorithms and outputs are read-only."""
from pathlib import Path
import os,sys,json,time,re,hashlib,fcntl,subprocess,traceback,importlib.util
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
OUT=ROOT/'results/overnight_energy_v2/rsicd_cap100_v1'
PARENT=ROOT/'tools/overnight_energy_v2/fourway_cap100_v1'
sys.path.insert(0,str(PARENT))
import prepare_campaign as parent
import runtime,metrics
import numpy as np
clip=parent.clip;core=parent.core
sys.path.insert(0,str(HERE))
from dataset_rsicd import RetrievalData
from data_schedule import dirichlet_partition,make_schedule
ARMS={k:dict(parent.ARMS[k]) for k in ['proposed_fu_guard_cliperase','separate_aircomp_periodic','aircomp_ofdma_periodic']}
ARMS['sequential_digital_periodic']=dict(parallel=False,crossmodal=True,period=5,digital=False,main=True,objective='cliperase_inspired')
LABELS={k:parent.LABELS[k] for k in ARMS if k in parent.LABELS}
LABELS['sequential_digital_periodic']='Digital FL + periodic FU (OFDMA)'
def write(name,d):core.write(OUT/name,d)
def status(**d):write('status.json',dict(unix=time.time(),pid=os.getpid(),**d))
def identity(pid):
 p=Path('/proc')/str(pid);return dict(pid=pid,start_ticks=p.joinpath('stat').read_text().split(') ',1)[1].split()[19],argv=p.joinpath('cmdline').read_bytes().replace(b'\0',b' ').decode())
def verify():
 d=json.loads((OUT/'protocol.json').read_text())
 for n,h in d['source_hashes'].items():assert core.sha(ROOT/n)==h,n
 return d

class SequentialRadio:
 @staticmethod
 def analog(l,u,channels,config,seed):
  assert not (np.any(l) and np.any(u))
  phase='FU' if np.any(u) else 'FL'
  vector,meta=parent.digital.analog(l,u,channels,config,seed)
  meta.update(physical_mode='digital_sequential_ofdma',phase=phase,
    packet='Q6('+phase+' only); one payload per transmitting UAV',FL_FU_temporally_overlapped=False)
  return vector,meta

def prepare():
 OUT.mkdir(parents=True,exist_ok=True)
 if (OUT/'protocol.json').exists():
  verify();return json.loads((OUT/'plan.json').read_text()),json.loads((OUT/'requests.json').read_text())
 root=ROOT/'data/rsicd_original_v1'
 dl=json.loads((root/'download_status.json').read_text());assert dl['stage']=='completed'
 items=json.loads((root/'dataset_rsicd.json').read_text())['images'];items.sort(key=lambda x:int(x['imgid']))
 assert [x['imgid'] for x in items]==list(range(len(items)))
 assert all(len(x['sentences'])==5 and all(y['raw'].strip() for y in x['sentences']) for x in items)
 train=np.array([x['imgid'] for x in items if x['split']=='train'])
 order=np.random.default_rng(0).permutation(train);validation=sorted(map(int,order[:256]));train=order[256:]
 test=[x['imgid'] for x in items if x['split']=='test'];reserved=[x['imgid'] for x in items if x['split']=='val']
 labels=np.array([re.sub(r'[_\d]+$','',Path(x['filename']).stem) for x in items])
 partitions=[train[x] for x in dirichlet_partition(labels[train],10,.1,0,16)]
 records=make_schedule(partitions,rounds=200,initial_rounds=20,requests=(21,),initial_fraction=1.,
   deletion_fraction=.5,requesting_users=5,batch_size=16,seed=0)
 cohort=records[20]['forget'];forgotten=sorted(set().union(*map(set,cohort)))
 requests=clip.stream.make_requests(cohort)
 assert len(requests)==45 and all(len(r['images'])>=2 for r in requests)
 seen=[set() for _ in range(10)]
 for rec in records[:20]:
  for u,batch in enumerate(rec['batches']):seen[u].update(batch)
 assert all(set(cohort[u])<=seen[u] for u in range(10))
 assert set(forgotten).isdisjoint(validation+test+reserved)
 refrecords=[]
 for rec in records:
  pools=[sorted(set(x)-set(forgotten)) for x in rec['retained']]
  batches=[np.random.default_rng(s).choice(p,min(16,len(p)),replace=False).astype(int).tolist() for p,s in zip(pools,rec['batch_seeds'])]
  refrecords.append(dict(round=rec['round'],retained=pools,batches=batches))
 config=json.loads((parent.OUT/'preflight.json').read_text())['config']
 members=sorted(map(int,np.random.default_rng(0).choice(forgotten,128,replace=False)))
 plan=dict(version='rsicd_cap100_v1',dataset='Original RSICD real image-caption pairs',seed=0,
   training_images=len(train),validation_images=validation,official_test_images=test,unused_official_validation_images=reserved,
   annotation_sha256=core.sha(root/'dataset_rsicd.json'),partitions=[list(map(int,x)) for x in partitions],
   forgotten_images=forgotten,forget_probe=forgotten,mia_member_images=members,mia_nonmember_images=validation[:128],
   gallery_images=sorted(validation+forgotten),config=config,rounds=200,initial_learning_rounds=20,
   batch_size=16,prompt_layers=8,prompt_tokens=2,learning_rate=.0035,forget_weight=1.,crossmodal_weight=1.,
   unlearning_rounds=[21,22,23,24,25],eval_rounds=list(range(0,201,5)))
 write('plan.json',plan);write('records.json',records);write('retrain_records.json',refrecords)
 write('requests.json',requests)
 channels=np.load(ROOT/'results/overnight_energy_v2/retrieval_long200_v3/channels.npy');np.save(OUT/'channels.npy',channels)
 hashes=dict(json.loads((parent.OUT/'protocol.json').read_text())['source_hashes'])
 hashes[str((parent.OUT/'protocol.json').relative_to(ROOT))]=core.sha(parent.OUT/'protocol.json')
 for p in HERE.glob('*.py'):hashes[str(p.relative_to(ROOT))]=core.sha(p)
 hashes[str((root/'dataset_rsicd.json').relative_to(ROOT))]=core.sha(root/'dataset_rsicd.json')
 for name in ['plan.json','records.json','retrain_records.json','requests.json','channels.npy']:
  hashes[str((OUT/name).relative_to(ROOT))]=core.sha(OUT/name)
 protocol=dict(version='rsicd_cap100_v1',created_unix=time.time(),arms=ARMS,labels=LABELS,source_hashes=hashes,
   authorization='2026-10-23 user requested another dataset and 1x2 figures; RSICD selected as the stated default after an optional dataset question.',
   source='https://github.com/201528014227051/RSICD_optimal',data_download=dl,
   changed='Dataset only: original RSICD; all four system algorithms and transmission settings frozen from the completed RSITMD comparison.',
   split='Same policy: seed0 holdout256 from official train; remaining official train for FL. Official val/test unused for configuration or training.',
   workload='200 FL=20 common initial+180 adaptation; same seed0 Dirichlet alpha.1,10UAV,50% of seen data of5 users;45requests,9bursts,5FUsteps/request. Dataset-specific deleted count='+str(len(forgotten)),
   model='Frozen pretrained CLIP ViT-B/16; 2 visual/text prompts in first8blocks;20480 transmitted coordinates; logit scale fixed.',
   objective_policy='Proposed and sequential digital: CLIPErase-inspired. Serial analog and hybrid: existing safe-proxy objective. Full-system comparison.',
   cost='All four pay dataset-specific physical initial20 and safe-proxy20. Every actual adaptation/FU upload charged. TX cap0.10J; compute totals pending hardware calibration.',
   prefix='Rebuild initial20 and safe-proxy20 on RSICD with the inherited legacy analog prefix kernel; never reuse RSITMD-trained checkpoints.',
   reference='200-round ideal final retained-set retrain on RSICD from pretrained prompts; evaluation only, never an FU teacher.',
   target='0.95 times dataset-specific final retrain validation mR, same reporting rule; reference/test do not choose settings. Three scheduled observations, raw values.',
   plots='Display-only centered three-point mean for mR/loss across all arms, endpoints preserved, faint raw curves. No smoothing of final numbers/targets/costs.',
   execution='Sequential initial20,proxy20,training-only preflight,reference200,then four200-total-FL systems under shared GPU lock. No old campaign restart.',
   limitations='Single seed; different dataset-specific gallery sizes and deleted counts; fixed channels; no total-energy calibration; no deletion guarantee.')
 write('protocol.json',protocol)
 write('preparation_checks.json',dict(passed=True,train=len(train),validation=256,forgotten=len(forgotten),
   requests=len(requests),partition_sizes=list(map(len,partitions)),official_test_unused=True,
   official_validation_unused=True,cohort_previously_seen=True,training_validation_disjoint=True,
   future_request_times_not_used_in_proxy=True,reference_not_teacher=True))
 return plan,requests

def physical_checks(l,u,old,ch,c,label):
 cases=[('proposed',parent.guard.analog,l,u),('separate_FL',parent.common.analog,l,np.zeros_like(l)),
   ('separate_FU',parent.common.analog,np.zeros_like(l),old),
   ('hybrid_FU',lambda a,b,ch,c,s:parent.common.digital_fu(b,ch,c),np.zeros_like(l),old),
   ('digital_FL',SequentialRadio.analog,l,np.zeros_like(l)),('digital_FU',SequentialRadio.analog,np.zeros_like(l),u)]
 results={}
 for n,fn,a,b in cases:
  v,m=fn(a,b,ch,c,500000)
  assert v is not None and m['status']=='feasible',(n,m)
  assert np.isfinite(v).all() and m['conditional_rmse']<=m['epsilon']*(1+3e-5)+1e-9
  assert m['energy_bound_j']<=.1*(1+2e-6) and m['max_power_w']<=c['power_limit_w']*(1+2e-6)
  assert m['max_user_energy_j']<=c['available_energy_j']*(1+2e-6) and m['uplink_s']<=c['round_uplink_deadline_s']+1e-10
  results[n]=m
 write(label,dict(passed=True,dimension=l.shape[1],records=results))

def cpu_preflight(plan,requests):
 checks=parent.common.base.old.ofdma.checks();assert checks['passed']
 assert clip.stream.self_check()
 z=np.load(clip.OUT/'fixed_training_updates.npz')
 physical_checks(z['FL'],z['FU'],z['old_FU'],z['channels'],plan['config'],'cpu_preflight.json')
 write('queue_registered.json',dict(unix=time.time(),stages=['initial20','proxy20','actual_update_preflight','reference200']+list(ARMS),
   status='registered',GPU_lock='results/overnight_energy_v2/association_objective_gpu.lock'))

def pure_trajectory(engine,plan,records,ch,kind):
 from train import update_theta
 import radio as legacy
 theta=engine.theta0.clone();energy=airtime=0.;rows=[]
 for rec in records:
  k=rec['round'];status(stage=kind,round=k,total=len(records))
  engine.plan['learning_rate']=core.lr_at(k) if kind=='reference200' else .0035
  l,_,_=engine.updates(theta,rec,k,'retrain',[[] for _ in range(10)])
  if kind=='reference200':vec=l.sum(0);rm=dict(status='ideal_reference',energy_bound_j=None,uplink_s=None)
  else:
   seed=(240000 if kind=='proxy20' else 100000)+k
   vec,rm=legacy.communicate(l,np.zeros_like(l),ch,plan['config'],'hybrid',seed)
   assert vec is not None and rm['status']=='feasible',rm
   assert rm['energy_bound_j']<=.10*(1+2e-6)
   energy+=rm['energy_bound_j'];airtime+=rm['uplink_s']
  theta=update_theta(theta,vec);rows.append(dict(round=k,radio=rm))
  write(kind+'/progress.json',dict(round=k,total=len(records)))
  if k%5==0:core.save_theta(OUT/kind/'checkpoint.pt',theta,k)
 core.save_theta(OUT/kind/'final.pt',theta,len(records))
 write(kind+'/rounds.json',rows)
 write(kind+'/finished.json',dict(status='completed',rounds=len(records),TX_energy_j=energy if kind!='reference200' else None,uplink_s=airtime if kind!='reference200' else None))
 return theta,dict(energy_j=energy,uplink_s=airtime)

def main():
 plan,requests=prepare()
 if '--prepare' in sys.argv:cpu_preflight(plan,requests);print('CPU_PREFLIGHT_PASSED',flush=True);return
 verify();assert json.loads((OUT/'cpu_preflight.json').read_text())['passed']
 own=(OUT/'launch.lock').open('a');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (OUT/'started.json').exists(),'Refuse duplicate campaign'
 gpu=(ROOT/'results/overnight_energy_v2/association_objective_gpu.lock').open('a')
 status(stage='waiting_for_GPU_lock');fcntl.flock(gpu,fcntl.LOCK_EX)
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU occupied by an unlocked process'
 write('started.json',dict(unix=time.time(),identity=identity(os.getpid())))
 from train import Engine
 engine=Engine(RetrievalData(),dict(plan));ch=np.load(OUT/'channels.npy')
 records=json.loads((OUT/'records.json').read_text());refs=json.loads((OUT/'retrain_records.json').read_text())
 warm,warmcost=pure_trajectory(engine,plan,records[:20],ch,'initial20')
 core.save_theta(OUT/'proposed/round_020.pt',warm,20)
 proxy_records=core.proxy_schedule(records,set(plan['forgotten_images']))
 write('proxy_records.json',proxy_records)
 proxy,proxycost=pure_trajectory(engine,plan,proxy_records,ch,'proxy20')
 prefix={k:warmcost[k]+proxycost[k] for k in warmcost};write('common_prefix_cost.json',prefix)
 q=clip.stream.Queue(requests,True,0);kind,ids=q.decision(0.,0)
 teacher=core.Teacher(engine,proxy)
 l,u,_=clip.vectors(engine,warm,plan,q,0.,0,kind,ids,teacher,{},True)
 l2,old,_=clip.ORIGINAL_VECTORS(engine,warm,plan,q,0.,0,kind,ids,teacher,{},True)
 assert np.allclose(l,l2,rtol=1e-6,atol=1e-8)
 np.savez_compressed(OUT/'fixed_training_updates.npz',FL=l,FU=u,old_FU=old,channels=ch)
 physical_checks(l,u,old,ch,plan['config'],'actual_update_preflight.json')
 ref,_=pure_trajectory(engine,plan,refs,ch,'reference200')
 ev,prob=engine.evaluate(ref,None);mia,_=engine.mia_scores(ref);threshold=.95*ev['retained']['mR']
 write('reference_metrics.json',dict(metrics=ev,retained_mR_threshold=threshold,reporting_only=True))
 write('reference_mia.json',mia)
 runtime.OUT=OUT;runtime.write=write;runtime.ARMS=ARMS;core.SOURCE=OUT
 metrics.OUT=OUT;metrics.ARMS=ARMS;metrics.LABELS=LABELS;metrics.THRESHOLD=threshold;metrics.write=write
 done=[];fail=[]
 for name,opt in ARMS.items():
  verify()
  runtime.vectors=clip.vectors if opt['objective']=='cliperase_inspired' else clip.ORIGINAL_VECTORS
  runtime.radio=parent.guard if name=='proposed_fu_guard_cliperase' else SequentialRadio if name=='sequential_digital_periodic' else parent.common
  write('queue_status.json',dict(current=name,finished=done,pending=[n for n in ARMS if n not in done and n!=name],failures=fail))
  t=time.time()
  try:
   runtime.run_arm(engine,plan,requests,proxy,prob,threshold,ch,name,opt,prefix)
   result=json.loads((OUT/name/'manifest.json').read_text())
   if result['status']!='completed_200FL_and_requests':fail.append(dict(method=name,status=result['status']))
  except Exception as exc:
   fail.append(dict(method=name,error=repr(exc)));write(name+'/failure.json',dict(error=repr(exc),traceback=traceback.format_exc()))
  write(name+'/runtime_seconds.json',dict(started_unix=t,finished_unix=time.time(),elapsed_server_wall_seconds=time.time()-t,scope='Server simulation runtime; not UAV end-to-end time.'))
  done.append(name);metrics.report()
 write('queue_status.json',dict(current=None,finished=done,pending=[],failures=fail))
 write('finished.json',dict(unix=time.time(),status='all_registered_arms_finished_or_failed',failures=fail))
 status(stage='completed',failures=fail)

if __name__=='__main__':
 try:main()
 except Exception as exc:
  status(stage='failed',error=repr(exc));write('failure.json',dict(error=repr(exc),traceback=traceback.format_exc()));raise
 finally:parent.common.base.close()

