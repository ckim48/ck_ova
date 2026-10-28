"""Compare P-KS and its25% repair at the actual last model within400 seconds."""
import fcntl
from pathlib import Path
import time
import numpy as np
import pks_balance_v1 as study
import pks_evaluation_v1 as references
import learning_time_budget as old

ROOT=study.ROOT;OUT=ROOT/'results/pks_budget400_check_v1_20261028'
read,write,sha=study.read,study.write,study.sha


def run(ds):
 directory=OUT/(ds.lower()+'_seed0')
 if (directory/'finished.json').exists():return
 assert not (directory/'started.json').exists(),'Preserve incomplete check'
 q=study.q;q.infra.lib.infra.extra.Data=q.infra.lib.infra.VideoData
 event,base=q.infra.lib.infra.prior.load_base();base.repeat.clone_function=q.infra.cached_clone
 import torch,accounting
 from prepare_extra import native_engine,native_base,Data
 from train import update_theta
 assert torch.cuda.is_available();torch.set_num_threads(1)
 q.infra.lib.infra.ORIGINAL_INSTRUMENTED=accounting.instrumented_engine
 old_arm=old.OUT/(ds.lower()+'_seed0')/'gate_select16';control=study.CONT/(ds.lower()+'_seed0')/'control'
 assert torch.equal(torch.load(control/'final.pt',weights_only=True)['theta'],torch.load(old_arm/'final.pt',weights_only=True)['theta'])
 before=read(old_arm/'curve.json');after=read(control/'curve.json');assert len(before)==len(after)
 for a,b in zip(before,after):
  for key in ['FL_round','active_s','energy_j','evaluation']:assert a[key]==b[key],(ds,key,a['FL_round'])
 paths={'pks':control,'mix25_gate':study.CONT/(ds.lower()+'_seed0')/'mix25_gate'}
 points={m:next(r for r in reversed(read(path/'curve.json')) if r['active_s']<=400) for m,path in paths.items()}
 sources=dict(study.register()['source_sha256']);sources[str(Path(__file__).relative_to(ROOT))]=sha(Path(__file__))
 write(directory/'protocol.json',dict(registered_unix=time.time(),source_sha256=sources,
  scope='Exact unchanged P-KS continuation replay recovers previously unsaved pre400 checkpoints. Verify every original observed metric, clock, energy and final parameter tensor first. Evaluate P-KS and P-KB25 at their last completed update within400s using both existing common and policy-matched references. No training or outcome selection beyond the already prescribed400s budget.',
  points={m:dict(round=r['FL_round'],active_s=r['active_s'],checkpoint_sha256=sha(paths[m]/'checkpoints'/f"{r['FL_round']:04d}.pt")) for m,r in points.items()}))
 with (ROOT/'results/overnight_energy_v2/association_objective_gpu.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);write(directory/'started.json',dict(unix=time.time()))
  initial=study.OLD_SOURCES[(ds,'gate_select16')];plan=read(initial.parent/'plan.json')
  if ds=='AirScape':base=native_base(base,1)
  pools=[sorted(set(part)-set(plan['forgotten_images'])) for part in plan['partitions']];total=sum(map(len,pools))
  queue=base.RoundQueue(read(initial.parent/'requests.json'));queue.items={r['id']:r for r in read(initial/'request_records.json')};mission=read(initial/'mission_timeline.json')['service_end_s']
  rows=[]
  for m,policy,variant in [('pks','priority','control'),('mix25_gate','mix25','mix25_gate')]:
   study.initialize_variant(variant);data=Data(ds) if ds=='AirScape' else event.recover.Data(ds)
   Engine=native_engine(base,data.caption_count) if ds=='AirScape' else base.Engine
   payload=torch.load(initial.parent/'feature_cache/features.pt',map_location='cpu',weights_only=False);engine=study.engine_wrapper(Engine)(data,plan,payload)
   rd=references.OUT/(ds.lower()+'_seed0')/'references';theta=base.core.load_theta(rd/policy/'200.pt',engine.device)
   k_target=points[m]['FL_round'];target=None
   for k in range(201,k_target+2):
    updates=np.zeros((10,len(theta)-1))
    for u,pool in enumerate(pools):
     batch=study.choose_batch(engine,theta,pool,k,u,np.random.default_rng(730000+k*10+u));assert not set(batch)&set(plan['forgotten_images'])
     g,_=engine.gradient(theta,batch,k);updates[u]=(-old.LR*len(pool)/total*g[:-1]).cpu().numpy()
    theta=update_theta(theta,updates.sum(0))
    if k==k_target:target=theta.clone()
   checked=rd/policy/f'{k_target+1}.pt';assert checked.exists() and torch.equal(theta,base.core.load_theta(checked,engine.device))
   candidate=base.core.load_theta(paths[m]/'checkpoints'/f'{k_target:04d}.pt',engine.device)
   common=base.core.load_theta(rd/'common'/f'{k_target}.pt',engine.device)
   row=dict(dataset=ds,method=m,label='P-KS' if m=='pks' else 'P-KB25',stage='budget400',budget_s=400,
    FL_round=k_target,active_s=points[m]['active_s'],total_energy_kJ=points[m]['energy_j']/1000)
   for name,ref in [('common',common),('matched',target)]:
    _,prob=engine.evaluate(ref,None);e=base.clip.stream.evaluation(engine,candidate,prob,queue,mission+1000)
    assert e['retained']==points[m]['evaluation']['retained']
    row.update(retained_mR_percent=100*e['retained']['mR'],retained_loss=e['retained']['loss'])
    row[name+'_KL']=e['forget']['KL_retrain_to_candidate'];row[name+'_worst_KL']=e['worst_request_KL']
   rows.append(row);del engine,data,payload;torch.cuda.empty_cache()
  result=dict(passed=True,unchanged_PKS_replay_exact=True,references_reproduced_exactly=True,rows=rows)
  write(directory/'finished.json',result);print(result,flush=True)


if __name__=='__main__':
 for dataset in study.q.infra.DATASETS:run(dataset)
