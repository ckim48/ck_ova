"""CPU all-coordinate audit; preserve sources; no GPU, training or test tuning."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import json,ast,time,hashlib,fcntl
import numpy as np
ROOT=Path.cwd();OUT=ROOT/'results/overnight_energy_v2/all_block_radio_audit_v2'
OUT.mkdir(parents=True,exist_ok=True)
lock=(OUT/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (OUT/'started.json').exists(),'Refuse duplicate'
def write(n,x):
 p=OUT/n;t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False));t.replace(p)
source=ROOT/'tools/overnight_energy_v2/minimum_latency_v2/run.py'
tree=ast.parse(source.read_text());nodes=[]
for n in tree.body:
 if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='protocol' for t in n.targets):break
 nodes.append(n)
original=ast.unparse(ast.Module(body=nodes,type_ignores=[]))
base_ns={'__name__':'base','__file__':str(source)};exec(compile(original,str(source),'exec'),base_ns)
variant=original
old='n = cp.norm(cp.hstack([1, self.H @ self.a]))'
assert variant.count(old)==1
variant=variant.replace(old,"self.mix = np.divide(vec, self.rho[:, None], out=np.zeros_like(vec), where=self.rho[:, None] > 0).T * self.H[None, :]\n        n = cp.norm(cp.hstack([1, self.mix @ self.a]))")
assert variant.count('(self.H @ a) ** 2')==2
variant=variant.replace('(self.H @ a) ** 2','float(np.sum((self.mix @ a) ** 2))')
ns={'__name__':'oracle','__file__':str(source)};exec(compile(variant,'isolated_exact_interference','exec'),ns)
from wireless import quantize
path=ROOT/'results/overnight_energy_v2/retrieval_fourway_v2/fixed_updates.npz'
z=np.load(path,allow_pickle=False);V=z['learning'];U=z['unlearning'];channels=z['channels']
D=128;N=V.shape[1]//D;assert N*D==V.shape[1] and V.shape==U.shape==(10,20480)
assert np.array_equal(V[:,:D],ns['learning']) and np.array_equal(U[:,:D],ns['unlearning'])
assert np.array_equal(channels,ns['base'])
head=ns['head'];ids=np.delete(np.arange(10),head);H=channels[ids,head]*10**.5
W=ns['W'];P=ns['P'];TS=[800,2048];BITS=[6,8];LAS=[x for x in ns['LAS'] if x<max(TS)];ETAS=[0.]+ns['ETAS']
def encode(x,b):
 q,d,p=quantize(x[:,None,:],b,32);return q[:,0,:],d[:,0],p[:,0]
def best(x):return min(x,key=lambda a:a['energy_j']) if x else None
def compact(x):
 keep=['energy_j','expected_energy_j','peak_power_w','total_error_bound','actual_expected_aggregate_rmse','analog_symbols','digital_tail_symbols','fu_overlap_fraction','radio_error_bound']
 return {k:v for k,v in x.items() if k in keep} if x else None
proto=dict(scope='All 160 sequential 128-coordinate blocks of one stored training update; fixed CSI head7, +10dB. Not learning trajectories or a globally optimized allocation.',
 source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),backend_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
 input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),head=head,blocks=N,dimension=V.shape[1],block_dimension=D,
 fu_bits=BITS,block_deadlines=TS,bandwidth_hz=W,power_bound_w=P,
 quality='Each block RMS allowance is 0.2 sum of all 10 local learning norms; total allowance root-sum-square across disjoint blocks. Identical allowance and canonical digital payload for all physical designs.',
 time='All blocks sequential; full-vector uplink is sum of assigned block durations. No local/SIC time.',
 partial='Same original analog length and payload-fraction grid below2048; zero overlap separately disclosed.',
 oracle='Extra diagnostic only: exact analog interference requires free local directions. Oracle is not a deployable candidate or a fair information-matched winner.',
 limitations=['Single stored training update and geometry; no statistical generalization.','Head fixed from channel before update inspection.','Uniform per-block deadlines and preset RMS budgets; no inter-block resource allocation.','Original conservative TX and RMS, ideal coding/SIC/CSI; metadata feedback and receiver costs omitted.','No claim about VLM performance from radio MSE.'])
write('protocol.json',proto);write('started.json',dict(pid=os.getpid(),unix=time.time()))
# Preset redistribution based only on current training-update quantizer bounds.
# All methods and both FU bit depths receive the same redistributed budget.
blocks_v=V.reshape(10,N,D);blocks_u=U.reshape(10,N,D)
base_eps=.2*np.linalg.norm(blocks_v,axis=2).sum(0)
global_eps=float(np.linalg.norm(base_eps))
_,qv6,_=quantize(blocks_v,6,32);_,qu6,_=quantize(blocks_u,6,32)
reserved=(qv6[ids]+qu6[ids]).sum(0)
assert np.linalg.norm(reserved)<global_eps
allocated=reserved+(global_eps-np.linalg.norm(reserved))*base_eps/global_eps
assert np.linalg.norm(allocated)<=global_eps*(1+1e-12)
proto['quality']='Common Q6 FL+FU quantizer bound per block plus remaining global RMS allowance proportional to original learning norms. Same allocation for every scheme and Q6/Q8; global norm no larger than v1. No validation/reference tuning.'
proto['quality_allocation']={'original_global_budget':global_eps,'allocated_global_budget':float(np.linalg.norm(allocated)),'quantizer_reservation_norm':float(np.linalg.norm(reserved))}
proto['limitations'].append('Preset feasible RMS redistribution, not globally optimized quality/time allocation; compare with v1 as whole-vector policies only.')
write('protocol.json',proto)
t0=time.time();rows=[];block_stats=[];verified=0
for block in range(N):
 sl=slice(block*D,(block+1)*D);v=V[:,sl];u=U[:,sl];eps=float(allocated[block])
 assert eps>0
 rho=np.linalg.norm(v[ids],axis=1)
 ratio=float(np.linalg.norm(v[ids].sum(0))**2/(rho.sum()**2))
 block_stats.append(dict(block=block,epsilon=eps,coherent_ratio_aligned=ratio,learning_norm_sum=float(rho.sum())))
 for space in [ns,base_ns]:
  space['codec_bound']=0.;space['codec_bias']=np.zeros(D)
 rawengine=base_ns['ConvexOverlap'](H,(v+u)[ids],eps)
 raw={T:rawengine.solve(T,np.zeros(len(ids))) for T in TS}
 for bits in BITS:
  qu,du,bu=encode(u,bits);ql,dl,bl=encode(v,bits);qb=float(du[ids].sum());qe=float(dl[ids].sum()+qb)
  for space in [ns,base_ns]:
   space['codec_bound']=qb;space['codec_bias']=(qu[ids]-u[ids]).sum(0)
  eng=base_ns['ConvexOverlap'](H,v[ids],eps-qb) if qb<eps else None
  oracle=ns['ConvexOverlap'](H,v[ids],eps-qb) if qb<eps else None
  states=[]
  if eng:
   for la in LAS:
    for eta in ETAS:
     bo=np.floor(bu[ids]*eta+.5);a=eng.solve(la,bo)
     if a:states.append((la,eta,bo,a))
  for T in TS:
   methods=dict(full_noma=eng.solve(T,bu[ids]) if eng else None,oracle_full_noma=oracle.solve(T,bu[ids]) if oracle else None,airair_raw=raw[T])
   partial=[];separate=[];ofdma=[]
   for la,eta,bo,a in states:
    if la>=T:continue
    tail=base_ns['noma_tail'](H,bu[ids]-bo,T-la)
    if tail:
     value=dict(a,energy_j=a['energy_j']+tail['energy_j'],
       expected_energy_j=a['expected_energy_j']+tail['energy_j'],peak_power_w=max(a['peak_power_w'],tail['peak_power_w']),
       digital_tail_symbols=T-la,fu_overlap_fraction=float(bo.sum()/bu[ids].sum()))
     (separate if eta==0 else partial).append(value)
    if eta==0:
     d=base_ns['old'].ofdma(bu[ids],H,T-la)
     if d:ofdma.append(dict(a,energy_j=a['energy_j']+d['energy'],expected_energy_j=a['expected_energy_j']+d['energy'],peak_power_w=max(a['peak_power_w'],float(d['power'].max())),digital_tail_symbols=T-la))
   methods.update(partial_noma=best(partial),separate_noma=best(separate),air_ofdma=best(ofdma))
   d=base_ns['old'].ofdma((bu+bl)[ids],H,T) if qe<=eps else None
   methods['ofdma_only']=dict(energy_j=d['energy'],expected_energy_j=d['energy'],peak_power_w=float(d['power'].max()),total_error_bound=qe) if d else None
   choices=[dict(a,mode=k) for k,a in methods.items() if a and k in ['full_noma','partial_noma','separate_noma']]
   methods['adaptive_noma']=best(choices)
   for name,x in methods.items():
    if not x:continue
    assert x['peak_power_w']<=P*(1+4e-6)
    assert x['total_error_bound']<=eps*(1+4e-6)
    assert x.get('actual_expected_aggregate_rmse',0)<=x['total_error_bound']*(1+4e-6)
    verified+=1
   a=methods['full_noma'];b=methods['oracle_full_noma']
   if a:assert b and b['energy_j']<=a['energy_j']*(1+2e-5)
   rows.append(dict(block=block,bits=bits,deadline_symbols=T,rms_budget=eps,fu_quantization_bound=qb,
    methods={k:compact(x) for k,x in methods.items()},adaptive_mode=methods['adaptive_noma'].get('mode') if methods['adaptive_noma'] else None))
 if (block+1)%20==0:
  write('results.json',rows);print(json.dumps(dict(completed_blocks=block+1,total=N)),flush=True)
summary=[]
for bits in BITS:
 for T in TS:
  g=[r for r in rows if r['bits']==bits and r['deadline_symbols']==T];assert len(g)==N
  methods={}
  for name in g[0]['methods']:
   ok=[r['methods'][name] for r in g if r['methods'][name]]
   complete=len(ok)==N
   methods[name]=dict(complete=complete,feasible_blocks=len(ok),required_blocks=N,
    total_energy_j=sum(x['energy_j'] for x in ok) if complete else None,
    total_uplink_s=N*T/W if complete else None,
    partial_feasible_block_energy_j=sum(x['energy_j'] for x in ok),
    max_block_average_power_bound_w=max((x['peak_power_w'] for x in ok),default=None),
    aggregate_RMS_bound=float(np.linalg.norm([x['total_error_bound'] for x in ok])) if complete else None)
  modes={k:sum(r['adaptive_mode']==k for r in g) for k in ['full_noma','partial_noma','separate_noma',None]}
  summary.append(dict(bits=bits,block_deadline_symbols=T,full_vector_deadline_s=N*T/W,methods=methods,adaptive_block_modes=modes))
ratios=np.array([x['coherent_ratio_aligned'] for x in block_stats])
checks=dict(blocks=N,cases=len(rows),returned_designs_verified=verified,first_block_matches_previous_input=True,
 original_backend_unchanged=hashlib.sha256(source.read_bytes()).hexdigest()==proto['backend_sha256'],
 global_RMS_budget=float(np.linalg.norm([x['epsilon'] for x in block_stats])),original_global_RMS_budget=global_eps,global_budget_not_relaxed=bool(np.linalg.norm(allocated)<=global_eps*(1+1e-12)))
assert checks['original_backend_unchanged'] and len(rows)==640
report=dict(protocol=proto,checks=checks,summary=summary,
 interference_ratio=dict(first=float(ratios[0]),min=float(ratios.min()),q10=float(np.quantile(ratios,.1)),median=float(np.median(ratios)),q90=float(np.quantile(ratios,.9)),max=float(ratios.max())),
 note='Never rank partial-feasible energy totals; any missing block makes the full-vector method incomplete. Adaptive selects a mode per block. Oracle requires unavailable free information.')
write('block_statistics.json',block_stats);write('report.json',report);write('finished.json',dict(cases=len(rows),runtime_s=time.time()-t0,checks=checks))
print('COMPLETE',time.time()-t0,flush=True)

