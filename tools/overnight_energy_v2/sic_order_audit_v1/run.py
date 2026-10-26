"""Exhaust all five-user SIC orders for the same robust full-overlap model; CPU."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import ast,json,time,hashlib,fcntl,itertools
import numpy as np
ROOT=Path.cwd();OUT=ROOT/'results/overnight_energy_v2/sic_order_audit_v1';OUT.mkdir(parents=True,exist_ok=True)
lock=(OUT/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (OUT/'started.json').exists(),'Refuse duplicate'
def write(name,obj):
 p=OUT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(obj,indent=2,allow_nan=False));t.replace(p)
src=ROOT/'tools/overnight_energy_v2/minimum_latency_v2/run.py';source=src.read_text();nodes=[]
for n in ast.parse(source).body:
 if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='protocol' for t in n.targets):break
 nodes.append(n)
ns={'__name__':'backend','__file__':str(src)};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(src),'exec'),ns)
from wireless import quantize
def ordered_coefficients(bits,H,L,order):
 co=np.zeros(len(H));later=0.
 assert set(order)==set(np.flatnonzero(bits>0))
 for i in reversed(order):
  x=np.log(2)*bits[i]/L
  if x>300:return None
  co[i]=np.expm1(x)*(1+later);later+=co[i]
 return co
# Clone only in this private interpreter; no frozen source mutation.
cls=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='ConvexOverlap')
adapted=ast.get_source_segment(source,cls).replace('class ConvexOverlap:','class OrderedOverlap:')
oldkey='key=(int(L),tuple(np.asarray(bits,dtype=int)))'
assert oldkey in adapted;adapted=adapted.replace(oldkey,'key=(int(L),tuple(np.asarray(bits,dtype=int)),tuple(self.order))')
oldcall='co=coefficients(bits,self.H,L)';assert oldcall in adapted
adapted=adapted.replace(oldcall,'co=ordered_coefficients(bits,self.H,L,self.order)')
oldorder="qrx=self.H**2*po;order=np.flatnonzero(bits>0);order=order[np.argsort(-self.H[order],kind='stable')]"
assert oldorder in adapted;adapted=adapted.replace(oldorder,'qrx=self.H**2*po;order=np.asarray(self.order,dtype=int)')
ns['ordered_coefficients']=ordered_coefficients;exec(compile(adapted,'private_ordered_backend','exec'),ns)
W=ns['W'];P=ns['P'];D=128;N=160
inp=ROOT/'results/overnight_energy_v2/retrieval_fourway_v2/fixed_updates.npz';z=np.load(inp,allow_pickle=False)
V=z['learning'].reshape(10,N,D);U=z['unlearning'].reshape(10,N,D);head=ns['head'];ids=np.delete(np.arange(10),head);H=z['channels'][ids,head]*10**.5
base_eps=.2*np.linalg.norm(V,axis=2).sum(0);global_eps=float(np.linalg.norm(base_eps))
_,dv6,_=quantize(V,6,32);_,du6,_=quantize(U,6,32);reserve=(dv6[ids]+du6[ids]).sum(0)
allocated=reserve+(global_eps-np.linalg.norm(reserve))*base_eps/global_eps
active=tuple(int(i) for i in np.flatnonzero(np.linalg.norm(U[ids],axis=(1,2))))
orders=list(itertools.permutations(active));baseline_order=tuple(sorted(active,key=lambda i:-H[i]));assert len(orders)==120
previous=json.loads((ROOT/'results/overnight_energy_v2/all_block_radio_audit_v2/report.json').read_text())
protocol=dict(scope='Fixed-update CPU audit; no training or general NOMA superiority claim.',head=head,channel_gain_db=10,bits=[6,8],block_symbols=[800,2048],blocks=N,block_dimension=D,power_w=P,
 active_fu_uavs=ids[list(active)].tolist(),permutations=len(orders),fixed_order_uavs=ids[list(baseline_order)].tolist(),global_RMS_budget=global_eps,allocated_RMS=float(np.linalg.norm(allocated)),
 source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),backend_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),input_sha256=hashlib.sha256(inp.read_bytes()).hexdigest(),
 model='Perfect SIC, same triangle analog interference/RMS and robust waveform energy, ideal coding/CSI, same native codec and head. Exact convex reciprocal-receiver solve per order.',
 pruning='Necessary analog sum amplitude M>=sum(rho)*sqrt(D/(2L))/radio_eps. Hence digital power>=c/H^2*(1+M_min^2). Reject orders if this exceeds any user power, or if sum digital energy lower bound exceeds the current feasible full energy.',
 caveats=['Adaptive SIC requires communicating or agreeing the order; signalling omitted only for this optimistic diagnostic, never a new implementable winner.',
 'Fixed lengths and head/common block RMS. Exhaustive only over the 120 orders at these fixed conditions; numerical solver failures distinguished from analytic pruning.',
 'Existing partial/separate NOMA are feasible reference designs only; no claim of jointly order-optimized partial NOMA.',
 'No reference/test metrics used. Feasible/incomplete costs separated. No new GPU promotion without codec and learning validation.'],
 source_context=dict(title='Attention-based SIC Ordering and Power Allocation for Non-orthogonal Multiple Access Networks',venue='IEEE TMC 24(2), 2025; online 2024',doi='10.1109/TMC.2024.3470828',author_manuscript='https://arxiv.org/html/2407.14740v1',scope='Sections II/III motivate dynamic order and specify uplink SINR. Their weighted proportional-fair DRL ASOPA is not reproduced by this energy-minimizing exhaustive diagnostic.'))
write('protocol.json',protocol);write('started.json',dict(pid=os.getpid(),unix=time.time()));t0=time.time()
# Independent recurrence/rate checks and adjacent-swap theorem for pure digital energy.
rng=np.random.default_rng(1017);synthetic=0
for _ in range(200):
 h=np.exp(rng.uniform(-1,3,5));b=rng.integers(1,1000,5);L=int(rng.integers(1000,5000));oo=tuple(np.argsort(-h));costs=[]
 for order in itertools.permutations(range(5)):
  co=ordered_coefficients(b,h,L,order);costs.append(float(np.sum(co/h**2)))
  for j,i in enumerate(order):assert abs(L*np.log2(1+co[i]/(1+sum(co[k] for k in order[j+1:])))-b[i])<1e-7
 assert np.sum(ordered_coefficients(b,h,L,oo)/h**2)<=min(costs)*(1+1e-10)
 synthetic+=1
# M lower bound tested on randomly feasible aligned/reduced-gain designs.
for _ in range(2000):
 A=rng.uniform(1.01,20);noise=rng.uniform(.01,2);zz=noise*rng.uniform(1,5);M=A*zz+rng.uniform(-1,1)*(zz-noise)
 assert M>=A*noise-1e-10
rows=[];returned=0;pruned_power=0;pruned_energy=0;solver_unresolved=0;backend_matches=0
for block in range(N):
 v=V[:,block];u=U[:,block];eps=float(allocated[block])
 for bits in [6,8]:
  qu,du,payload=quantize(u[:,None,:],bits,32);payload=payload[ids,0];qb=float(du[ids,0].sum());radio_eps=eps-qb;assert radio_eps>0
  ns['codec_bound']=qb;ns['codec_bias']=(qu[ids,0]-u[ids]).sum(0)
  engine=ns['OrderedOverlap'](H,v[ids],radio_eps);engine.order=baseline_order
  original=ns['ConvexOverlap'](H,v[ids],radio_eps)
  for L in [800,2048]:
   engine.order=baseline_order
   fixed=engine.solve(L,payload);ref=original.solve(L,payload)
   assert (fixed is None)==(ref is None)
   if fixed:assert abs(fixed['energy_j']-ref['energy_j'])<=1e-7*max(ref['energy_j'],1e-12)
   backend_matches+=1;best=fixed;best_order=baseline_order if fixed else None;feasible=0;analytical_pruned=0;unresolved=0
   rho=np.linalg.norm(v[ids],axis=1);M_min=float(rho.sum()*np.sqrt(D/(2*L))/radio_eps)
   for order in orders:
    co=ordered_coefficients(payload,H,L,order);pmin=co/H**2*(1+M_min*M_min)
    if pmin.max()>P*(1+4e-6):pruned_power+=1;analytical_pruned+=1;continue
    if best and L/W*pmin.sum()>best['energy_j']*(1+1e-8):pruned_energy+=1;analytical_pruned+=1;continue
    engine.order=order;x=engine.solve(L,payload)
    if x is None:
     # Solver-declared infeasible/failed are not a proof of global infeasibility.
     solver_unresolved+=1;unresolved+=1;continue
    returned+=1;feasible+=1
    assert x['total_error_bound']<=eps*(1+4e-6) and x['peak_power_w']<=P*(1+4e-6)
    # Re-evaluate each user's exact SIC rate against the chosen order.
    q=H**2*x['po'];noise=1+(H@x['a'])**2
    for j,i in enumerate(order):assert L*np.log2(1+q[i]/(noise+sum(q[k] for k in order[j+1:])))-payload[i]>=-1e-5
    if best is None or x['energy_j']<best['energy_j']*(1-1e-9):best=x;best_order=order
   def compact(x):
    if x is None:return None
    return {k:x[k] for k in ['energy_j','peak_power_w','total_error_bound','solver_status','min_rate_margin_bits']}
   if fixed:assert best and best['energy_j']<=fixed['energy_j']*(1+1e-8)
   rows.append(dict(block=block,bits=bits,L=L,fixed=compact(fixed),best=compact(best),best_order_uavs=ids[list(best_order)].tolist() if best_order else None,
    feasible_returned_orders=feasible,analytically_pruned_orders=analytical_pruned,solver_unresolved_orders=unresolved))
 if (block+1)%20==0:
  write('status.json',dict(blocks=block+1,runtime_s=time.time()-t0));print(json.dumps(dict(blocks=block+1,runtime_s=time.time()-t0)),flush=True)
write('block_results.json',rows);summary=[]
for bits in [6,8]:
 for L in [800,2048]:
  rr=[r for r in rows if r['bits']==bits and r['L']==L];fixed=[r['fixed'] for r in rr if r['fixed']];best=[r['best'] for r in rr if r['best']]
  old=next(x for x in previous['summary'] if x['bits']==bits and x['block_deadline_symbols']==L)['methods']
  fE=sum(x['energy_j'] for x in fixed) if len(fixed)==N else None;bE=sum(x['energy_j'] for x in best) if len(best)==N else None
  if fE is not None:assert abs(fE-old['full_noma']['total_energy_j'])<=1e-5*fE
  summary.append(dict(bits=bits,uplink_s=N*L/W,fixed_feasible_blocks=len(fixed),best_feasible_blocks=len(best),fixed_energy_j=fE,order_optimized_energy_j=bE,
   energy_saving_pct=100*(1-bE/fE) if fE and bE else None,changed_order_blocks=sum(r['best_order_uavs'] is not None and r['best_order_uavs']!=ids[list(baseline_order)].tolist() for r in rr),
   reference_feasible_costs={k:old[k].get('total_energy_j') for k in ['air_ofdma','partial_noma','separate_noma','airair_raw']},solver_unresolved_orders=sum(r['solver_unresolved_orders'] for r in rr)))
checks=dict(synthetic_scenarios=synthetic,synthetic_permutation_rate_checks=synthetic*120,amplitude_bound_cases=2000,unchanged_backend_cases=backend_matches,returned_designs=returned,
 analytic_power_prunes=pruned_power,analytic_cost_prunes=pruned_energy,solver_unresolved_orders=solver_unresolved,backend_unchanged=hashlib.sha256(src.read_bytes()).hexdigest()==protocol['backend_sha256'])
assert checks['backend_unchanged'];report=dict(protocol=protocol,checks=checks,summary=summary)
write('report.json',report);write('finished.json',dict(runtime_s=time.time()-t0,checks=checks));print('COMPLETE',time.time()-t0,flush=True)
