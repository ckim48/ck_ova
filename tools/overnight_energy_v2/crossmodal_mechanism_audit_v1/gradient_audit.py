import os,sys,json,time,hashlib,fcntl,subprocess
from pathlib import Path
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
sys.dont_write_bytecode=True
ROOT=Path('/home/wnlab/Documents/Codex/2026-10-20/files-pasted-by-the-user-standalone/outputs/ova_CK')
SRC=ROOT/'results/overnight_energy_v2/retrieval_long200_v3'
OUT=ROOT/'results/overnight_energy_v2/crossmodal_mechanism_audit_v1'
OUT.mkdir(parents=True,exist_ok=True)
lock=(ROOT/'results/overnight_energy_v2/association_objective_gpu.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
if active: raise RuntimeError('GPU is occupied; no duplicate diagnostic started')
sys.path.insert(0,str(ROOT/'tools/overnight_energy_v2/retrieval_long200_v3'))
import common
from train import Engine
from dataset import RetrievalData
import torch,numpy as np
plan=json.loads((SRC/'plan.json').read_text())
records=json.loads((SRC/'records.json').read_text())
hashes={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for folder in ['retrieval_fourway_v2','retrieval_long200_v3'] for f in (ROOT/'tools/overnight_energy_v2'/folder).glob('*.py')}
engine=Engine(RetrievalData(),dict(plan))
teacher=torch.load(SRC/'proposed/round_020.pt',map_location='cpu',weights_only=True)['theta'].to(engine.device)
engine.set_teacher(teacher)
forget=records[20]['forget'];nf=sum(map(len,forget));allrows=[]
for k in range(21,26):
 theta=torch.load(SRC/'proposed'/f'round_{k-1:03d}.pt',map_location='cpu',weights_only=True)['theta'].to(engine.device)
 rec=records[k-1];fg_sum=torch.zeros_like(theta);rg_sum=torch.zeros_like(theta);users=[]
 for u,ids in enumerate(rec['batches']):
  f=forget[u]
  if not f:continue
  batch=np.random.default_rng(9000+k*10+u).choice(f,min(16,len(f)),replace=False).astype(int).tolist()
  fg,floss=engine.gradient(theta,batch,k)
  rg,rloss=engine.gradient(theta,ids,k,'relation')
  fn=float(fg.norm());rn=float(rg.norm());w=len(f)/nf
  fg_sum+=w*plan['forget_weight']*fg
  rg_sum+=w*plan['crossmodal_weight']*rg
  users.append(dict(user=u,forget_loss=floss,relation_kl=rloss,forget_grad_norm=fn,relation_grad_norm=rn,
    correction_to_forget_grad_ratio=plan['crossmodal_weight']*rn/max(plan['forget_weight']*fn,1e-30),
    cosine=float(torch.dot(fg,rg))/(fn*rn) if fn*rn>1e-20 else None))
 fn=float(fg_sum.norm());rn=float(rg_sum.norm())
 row=dict(round=k,users=users,weighted_forget_grad_norm=fn,weighted_relation_grad_norm=rn,
          weighted_correction_ratio=rn/max(fn,1e-30),
          ideal_relation_update_norm=plan['learning_rate']*rn,
          weighted_cosine=float(torch.dot(fg_sum,rg_sum))/(fn*rn) if fn*rn>1e-20 else None)
 allrows.append(row)
 print(json.dumps({q:v for q,v in row.items() if q!='users'}),flush=True)
 (OUT/'progress.json').write_text(json.dumps(allrows,indent=2))
assert all(hashlib.sha256(Path(f).read_bytes()).hexdigest()==h for f,h in hashes.items())
result=dict(scope='Read-only diagnostic of fixed saved proposed checkpoints at request rounds; no training or hyperparameter change; no test-set selection.',
            note='Gradient norms and directions describe this trajectory; they do not establish causal utility gains.',
            teacher_round=20,unlearning_rounds=list(range(21,26)),source_hashes_unchanged=True,rows=allrows)
(OUT/'gradient_audit.json').write_text(json.dumps(result,indent=2))
print('AUDIT_COMPLETED',flush=True)
