from common import *
import argparse,os,fcntl,traceback,shutil,math
import numpy as np

def lr_at(k):
 return .0035 if k<=40 else .0035*.5*(1+math.cos(math.pi*(k-40)/160))

def prepare():
 from data_schedule import make_schedule
 old=json.loads((SOURCE/'plan.json').read_text())
 for n,h in old['sources'].items():assert sha(ROOT/n)==h,n
 for n,h in old['runner_hashes'].items():assert sha(FROZEN/n)==h,n
 OUT.mkdir(parents=True,exist_ok=True)
 if (OUT/'plan.json').exists():return json.loads((OUT/'plan.json').read_text())
 records=make_schedule([np.asarray(x) for x in old['partitions']],rounds=200,initial_rounds=20,requests=(21,),
  initial_fraction=1.,deletion_fraction=.5,requesting_users=5,batch_size=16,seed=0)
 assert records[:40]==json.loads((SOURCE/'records.json').read_text())
 forgotten=set(old['forgotten_images']);refrecords=[]
 for rec in records:
  pools=[sorted(set(x)-forgotten) for x in rec['retained']]
  batches=[np.random.default_rng(s).choice(p,min(16,len(p)),replace=False).astype(int).tolist() for p,s in zip(pools,rec['batch_seeds'])]
  refrecords.append(dict(round=rec['round'],retained=pools,batches=batches))
 assert refrecords[:40]==json.loads((SOURCE/'retrain_records.json').read_text())
 assert all(not (forgotten&set(batch)) for r in records[21:] for batch in r['batches'])
 assert all(not (forgotten&set(batch)) for r in refrecords for batch in r['batches'])
 c=json.loads((SOURCE/'selected_radio_config.json').read_text())
 assert c['symbols_per_coordinate']==64
 c['round_uplink_deadline_s']=20480*64/c['symbol_rate_hz']
 plan=dict(old,version='retrieval_long200_v3',created_unix=time.time(),rounds=200,config=c,
  eval_rounds=sorted(set(range(5,201,5))|{21}),
  learning_rate=.0035,learning_rate_schedule='Constant0.0035 through round40; cosine decay to0 over rounds41..200, identical for all methods and retrain.',
  method_labels=dict(proposed='NOMA+AirComp',aircomp_only='AirComp+AirComp (single aggregate)',aircomp_oma='OMA+AirComp',oma_only='OMA-only',hybrid_no_crossmodal='NOMA+AirComp without cross-modal term'),
  main_methods=['proposed','aircomp_only','aircomp_oma','oma_only'],ablation_methods=['hybrid_no_crossmodal'],
  methods=['proposed','aircomp_only','aircomp_oma','oma_only','hybrid_no_crossmodal'],
  transmission='Proposed and paired ablation use partial overlap: analog2d inside digital62d; common maximum frame64d. Other schemes retain v2 definitions. No receiver-specific controller.',
  timing=dict(common_frame_s=1.31072,partial_overlap_active_request_s=1.26976,analog_only_s=.04096),
  prefix_reuse='Exact matching v2 prefixes only: proposed/no-crossmodal branch from common physical round20; unchanged baselines and retrain resume from40. All prefix transmission costs are included. Old KL values are recomputed against the new final200-round retrain reference.',
  quality_thresholds=dict(retained_mR_fraction_of_final_retrain=.95,forget_bidirectional_KL_max=.01,consecutive_evaluations=3,first_eligible_round=25),
  empirical_plateau=dict(span_rounds=30,evaluations=7,retained_mR_range_max=.005,relative_loss_range_max=.01,minimum_round=60,interpretation='Observed utility plateau, not successful forgetting or mathematical convergence.'),
  stopping='Fixed200 rounds for every feasible method; no winner-dependent budget extension. Targets and plateau are measured from the complete trajectory, not used to tune the test set.',
  test_policy='v3 never reads v2 final_test results and never uses official test to pick configuration. Existing frozen v2 may evaluate its own test. v3 final200 checkpoints are evaluated only after all registered arms end.',
  scope='Single seed0 controlled long-horizon study; identical data/request schedule, Q8 target, channel, power and error policy. One request burst at21, five deletion steps21..25; no later requests or arrivals. More rounds do not ensure AirComp-only is worse.',
  frozen_v2_plan_sha256=sha(SOURCE/'plan.json'),frozen_engine_hashes=old['runner_hashes'],
  overlap_solver_sha256=sha(ROOT/'tools/overnight_energy_v2/partial_overlap_tx_v1/overlap_solver.py'),
  runner_hashes={p.name:sha(p) for p in HERE.glob('*.py')})
 write(OUT/'plan.json',plan);write(OUT/'records.json',records);write(OUT/'retrain_records.json',refrecords)
 shutil.copy2(SOURCE/'channels.npy',OUT/'channels.npy')
 write(OUT/'preparation_checks.json',dict(first40_records_exact=True,retrain_first40_exact=True,forgotten_data_excluded=True,
  learning_rate_start=lr_at(1),learning_rate_40=lr_at(40),learning_rate_200=lr_at(200),common_frame_s=c['round_uplink_deadline_s']))
 return plan

def read_rows(dest):
 p=dest/'prefix_rounds.json';rows=json.loads(p.read_text()) if p.exists() else []
 for f in sorted((dest/'rounds').glob('*.json')):rows.append(json.loads(f.read_text()))
 if rows:assert [r['round'] for r in rows]==list(range(1,rows[-1]['round']+1))
 return rows

def remap_evaluation(evaluation,prob,reference,plan):
 if reference is None:return evaluation
 gallery=plan['gallery_images'];oldi,oldt=[x.numpy() for x in reference]
 ni,nt=prob['log_image'],prob['log_text']
 for name,ids in [('retained',plan['validation_images']),('forget',plan['forget_probe'])]:
  ii=np.array([gallery.index(i) for i in ids]);ci=(ii[:,None]*5+np.arange(5)).reshape(-1)
  e=evaluation[name]
  ki=(np.exp(oldi[ii])*(oldi[ii]-ni[ii])).sum(1).mean()
  kt=(np.exp(oldt[ci])*(oldt[ci]-nt[ci])).sum(1).mean()
  e['KL_retrain_to_candidate']=max(0.,float((ki+kt)/2))
  e['i2t_top1_disagreement']=float(np.mean(oldi[ii].argmax(1)!=ni[ii].argmax(1)))
  e['t2i_top1_disagreement']=float(np.mean(oldt[ci].argmax(1)!=nt[ci].argmax(1)))
 return evaluation

def import_prefix(method,plan,reference):
 dest=OUT/method
 if (dest/'prefix_import.json').exists():return
 src=SOURCE/('proposed' if method in ['proposed','hybrid_no_crossmodal'] else method)
 limit=20 if method in ['proposed','hybrid_no_crossmodal'] else 40
 rows=[json.loads(x) for x in (src/'rounds.jsonl').read_text().splitlines()]
 rows=[r for r in rows if r['round']<=limit]
 if not rows or rows[-1]['round']!=limit or not (src/f'round_{limit:03d}.pt').exists():
  if method not in ['proposed','hybrid_no_crossmodal','retrain']:
   manifest=json.loads((src/'manifest.json').read_text())
   if manifest['status'].startswith('incomplete'):
    for r in rows:
     if 'evaluation' in r:r['evaluation']=remap_evaluation(r['evaluation'],np.load(src/f"probabilities_{r['round']:03d}.npz"),reference,plan)
    write(dest/'manifest.json',dict(manifest,source='unchanged v2 baseline',prefix_source=str(src)))
    write(dest/'prefix_rounds.json',rows);write(dest/'prefix_import.json',dict(status='inherited_incomplete'));return
  raise RuntimeError('Required completed prefix is unavailable: '+str(src))
 for r in rows:
  r['method']=method;r['learning_rate']=.0035;r['prefix_source']='retrieval_fourway_v2'
  if 'evaluation' in r:
   p=src/f"probabilities_{r['round']:03d}.npz"
   r['evaluation']=remap_evaluation(r['evaluation'],np.load(p),reference,plan)
   shutil.copy2(p,dest/p.name)
 for k in sorted(set([20,limit])):
  f=src/f'round_{k:03d}.pt'
  if f.exists():shutil.copy2(f,dest/f.name)
 write(dest/'prefix_rounds.json',rows)
 write(dest/'prefix_import.json',dict(status='imported',source=str(src),rounds=limit,checkpoint_sha256=sha(src/f'round_{limit:03d}.pt'),all_prefix_costs_included=True))

def trajectory(engine,plan,records,method,forget,reference=None):
 import torch
 from train import update_theta
 from radio_partial import communicate
 from report_long import build_report
 dest=OUT/method;dest.mkdir(exist_ok=True);(dest/'rounds').mkdir(exist_ok=True)
 manifest=dest/'manifest.json'
 if manifest.exists():
  m=json.loads(manifest.read_text())
  if m['status'].startswith('incomplete'):return None,None
 import_prefix(method,plan,reference)
 if manifest.exists() and json.loads(manifest.read_text())['status'].startswith('incomplete'):return None,None
 rows=read_rows(dest);last=rows[-1];k0=last['round']
 theta=torch.load(dest/f'round_{k0:03d}.pt',map_location='cpu',weights_only=True)['theta'].to(engine.device)
 teacher=torch.load(dest/'round_020.pt',map_location='cpu',weights_only=True)['theta'].to(engine.device)
 engine.set_teacher(teacher)
 e=last['cumulative_tx_energy_bound_j'];ee=last['cumulative_tx_energy_expected_j'];tt=last['cumulative_uplink_s']
 channels=np.load(OUT/'channels.npy');start=time.time();probs=None
 for rec in records[k0:]:
  k=rec['round'];t0=time.time();engine.plan['learning_rate']=lr_at(k)
  status(stage=method,round=k,rounds=200,pid=os.getpid(),elapsed_stage_s=time.time()-start)
  learning,unlearning,local=engine.updates(theta,rec,k,method,forget)
  if method=='retrain':vector=(learning+unlearning).sum(0);radio=dict(status='ideal_reference',energy_bound_j=None,uplink_s=None)
  else:
   mode='hybrid' if method in ['proposed','hybrid_no_crossmodal'] else method
   vector,radio=communicate(learning,unlearning,channels,plan['config'],mode,100000+k)
   if vector is None:
    write(manifest,dict(status='incomplete_no_feasible_design_found',failed_round=k,completed_rounds=k-1,radio=radio))
    build_report();return None,None
   e+=radio['energy_bound_j'];ee+=radio['expected_energy_j'];tt+=radio['uplink_s']
  theta=update_theta(theta,vector)
  row=dict(method=method,round=k,learning_rate=lr_at(k),radio=radio,cumulative_tx_energy_bound_j=e,
   cumulative_tx_energy_expected_j=ee,cumulative_uplink_s=tt,request_algorithm_completed=k>=25,
   request_transmission_steps_completed=max(0,min(k,25)-20),logical_requests=5 if k>=21 else 0,local=local)
  if k in plan['eval_rounds']:
   row['evaluation'],probs=engine.evaluate(theta,reference)
   np.savez_compressed(dest/f'probabilities_{k:03d}.npz',log_image=probs[0].numpy(),log_text=probs[1].numpy())
  row['wall_s']=time.time()-t0
  temp=dest/'checkpoint.tmp';torch.save(dict(theta=theta.cpu(),round=k),temp);temp.replace(dest/f'round_{k:03d}.pt')
  write(dest/'rounds'/f'{k:03d}.json',row)
  print(json.dumps(dict(method=method,round=k,wall_s=row['wall_s'],energy_j=e,uplink_s=tt,evaluation=row.get('evaluation'))),flush=True)
  if k in plan['eval_rounds']:build_report()
 if probs is None:_,probs=engine.evaluate(theta,reference)
 write(manifest,dict(status='completed',rounds=200,cumulative_tx_energy_bound_j=e if method!='retrain' else None,cumulative_uplink_s=tt if method!='retrain' else None))
 build_report();return theta,probs

def main():
 p=argparse.ArgumentParser();p.add_argument('--prepare-only',action='store_true');p.add_argument('--cpu-check',action='store_true');args=p.parse_args()
 plan=prepare()
 if args.cpu_check:
  from radio_partial import communicate
  z=np.load(SOURCE/'fixed_updates.npz')
  vector,r=communicate(z['learning'],z['unlearning'],z['channels'],plan['config'],'hybrid',100021)
  assert vector is not None and r['error_bound']<=r['epsilon']*(1+2e-5)
  assert abs(r['uplink_s']-1.26976)<1e-10
  write(OUT/'transport_check.json',r);print(json.dumps(r));return
 if args.prepare_only:status(stage='prepared',rounds=200);return
 queue_lock=(OUT/'queue.lock').open('a')
 try:fcntl.flock(queue_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:print('Long200 queue already running.');return
 status(stage='waiting_for_existing_gpu_queue',pid=os.getpid(),rounds=200)
 lock=(ROOT/'results/overnight_energy_v2/association_objective_gpu.lock').open('a')
 fcntl.flock(lock,fcntl.LOCK_EX)
 if (OUT/'status.json').exists() and (OUT/'finished.json').exists():return
 dep=json.loads((SOURCE/'status.json').read_text())
 if dep['stage']!='completed':
  status(stage='blocked_dependency_not_completed',dependency=dep,pid=os.getpid());return
 for n,h in plan['sources'].items():assert sha(ROOT/n)==h,n
 for n,h in plan['frozen_engine_hashes'].items():assert sha(FROZEN/n)==h,n
 for n,h in plan['runner_hashes'].items():assert sha(HERE/n)==h,n
 assert sha(ROOT/'tools/overnight_energy_v2/partial_overlap_tx_v1/overlap_solver.py')==plan['overlap_solver_sha256']
 import torch
 from train import Engine
 from dataset import RetrievalData
 from report_long import build_report
 assert torch.cuda.is_available()
 write(OUT/'started.json',dict(pid=os.getpid(),unix=time.time(),plan_sha256=sha(OUT/'plan.json')))
 engine=Engine(RetrievalData(),dict(plan));empty=[[] for _ in range(10)]
 records=json.loads((OUT/'records.json').read_text());refs=json.loads((OUT/'retrain_records.json').read_text())
 ref,probs=trajectory(engine,plan,refs,'retrain',empty)
 assert ref is not None
 mia,_=engine.mia_scores(ref);write(OUT/'retrain/mia.json',mia)
 for method in plan['methods']:
  theta,_=trajectory(engine,plan,records,method,records[20]['forget'],probs)
  if theta is not None:
   m,s=engine.mia_scores(theta);m['auc_gap_vs_retrain']=abs(m['loss_score_auc']-mia['loss_score_auc']);write(OUT/method/'mia.json',m);np.save(OUT/method/'mia_scores.npy',s)
  build_report()
 for method in ['retrain']+plan['methods']:
  checkpoint=OUT/method/'round_200.pt'
  if not checkpoint.exists():continue
  theta=torch.load(checkpoint,map_location='cpu',weights_only=True)['theta'].to(engine.device)
  result,_=engine.evaluate(theta,test=True);write(OUT/method/'final_test.json',result)
 build_report();write(OUT/'finished.json',dict(unix=time.time()));status(stage='completed',pid=os.getpid(),rounds=200)

if __name__=='__main__':
 try:main()
 except Exception as e:status(stage='failed',error=repr(e),pid=os.getpid());traceback.print_exc();raise
