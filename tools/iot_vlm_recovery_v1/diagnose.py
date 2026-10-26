"""Diagnose failed transport using saved common checkpoints, without tuning."""
from pathlib import Path
import sys,json,importlib.util,time
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('registered_iot',ROOT/'tools/iot_vlm_v3_global_iot/experiment.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
original=module.original;base=original.baseline_module()
import numpy as np
import torch
p=module.OUT/'rsitmd_seed0'
plan=original.read(p/'plan.json');requests=original.read(p/'requests.json')
data=original.Data('RSITMD')
payload=torch.load(p/'feature_cache/features.pt',map_location='cpu',weights_only=False)
engine=base.Engine(data,plan,payload)
warm=base.core.load_theta(p/'initial20/final.pt',engine.device)
proxy=base.core.load_theta(p/'proxy20/final.pt',engine.device);engine.set_teacher(warm)
q=base.RoundQueue(requests);first=min(r['release_after_adaptation_FL'] for r in requests)
q.sync(first,0.,0.);kind,ids=q.decision(0.,first)
l,u,meta=base.control.synchronized_vectors('cliperase')(engine,warm,plan,q,0.,first,kind,ids,base.core.Teacher(engine,proxy),{},True)
_,old_u,_=base.control.synchronized_vectors('proxy')(engine,warm,plan,q,0.,first,kind,ids,base.core.Teacher(engine,proxy),{},True)
radio=base.base.parent.digital
channels=np.load(module.GEOMETRY/'channels_by_site.npy')
out=ROOT/'results/iot_uav_recovery_diagnostics';out.mkdir(exist_ok=True)
np.savez(out/'preflight_vectors.npz',FL=l,FU=u,old_FU=old_u,channels=channels)
results={}
for name,vec in [('FL',l),('FU',u)]:
 eps=radio.guard.budget(vec if name=='FL' else np.zeros_like(vec),vec if name=='FU' else np.zeros_like(vec),plan['config'])[-1]
 _,blocks,quant,delta,payload,_,dims,_=radio.base.old.make_inputs(np.zeros_like(vec),vec,dict(plan['config'],quantization_bits=6,ofdma_subcarriers=128))
 bounds=[float(np.linalg.norm(delta[np.arange(10)!=h].sum(0))) for h in range(10)]
 results[name]=dict(epsilon=eps,Q6_bounds_by_head=bounds,all_heads_fail_quantization=all(v>eps for v in bounds),payload_bits=int(payload.sum()),counts=list(radio.base.COUNTS))
print(json.dumps(results,indent=2),flush=True)
original.write(out/'quantization_diagnosis.json',results)
for site,ch in enumerate(channels):
 v,m=base.base.SequentialRadio.analog(l,np.zeros_like(l),ch,plan['config'],500000)
 original.write(out/f'digital_FL_site{site}.json',m)
 print('site',site,m,flush=True)
base.base.parent.common.base.close()
