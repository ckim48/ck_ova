"""CPU comparison of genuine TDMA and OFDMA under matched radio assumptions."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import ast,json,time,hashlib,fcntl
import numpy as np
from scipy.optimize import brentq
from scipy.special import lambertw
import cvxpy as cp
ROOT=Path.cwd();OUT=ROOT/'results/overnight_energy_v2/orthogonal_access_audit_v1';OUT.mkdir(parents=True,exist_ok=True)
lock=(OUT/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (OUT/'started.json').exists(),'Refuse duplicate'
def write(n,x):
 p=OUT/n;t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False));t.replace(p)
source=ROOT/'tools/overnight_energy_v2/minimum_latency_v2/run.py';nodes=[]
for node in ast.parse(source.read_text()).body:
 if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='protocol' for t in node.targets):break
 nodes.append(node)
ns={'__name__':'backend','__file__':str(source)};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),ns)
from wireless import quantize
W=ns['W'];P=ns['P'];D=128;N=160
path=ROOT/'results/overnight_energy_v2/retrieval_fourway_v2/fixed_updates.npz';data=np.load(path,allow_pickle=False)
V=data['learning'].reshape(10,N,D);U=data['unlearning'].reshape(10,N,D);head=ns['head'];ids=np.delete(np.arange(10),head);H=data['channels'][ids,head]*10**.5
base_eps=.2*np.linalg.norm(V,axis=2).sum(0);global_eps=float(np.linalg.norm(base_eps))
_,qv6,_=quantize(V,6,32);_,qu6,_=quantize(U,6,32);reserved=(qv6[ids]+qu6[ids]).sum(0)
allocated=reserved+(global_eps-np.linalg.norm(reserved))*base_eps/global_eps
LAS=[128,160,192,256,384,512,768,1024,1536];TS=[800,2048]
proto=dict(scope='Pure CPU fixed training-update audit, not an end-to-end OFDMA learning experiment.',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 backend_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),head=head,channel_gain_db=10,
 blocks=N,block_dimension=D,bits=[6,8],uniform_total_block_symbols=TS,analog_length_candidates=LAS,power_limit_w=P,global_RMS=global_eps,allocated_RMS=float(np.linalg.norm(allocated)),
 definitions=[
 'Use f_i as time fraction for TDMA, bandwidth fraction for OFDMA; sum(f)<=1. Let q_i denote block-average TX power for TDMA and full-duration user power for OFDMA.',
 'Both ideal flat-channel rate constraints are B_i<=L f_i log2(1+H_i^2 q_i/f_i), and energy=(L/W)sum(q_i). TDMA active-slot power<=P gives q_i<=P f_i; OFDMA total user power<=P gives q_i<=P.',
 'Every TDMA feasible point embeds in OFDMA with equal rate/energy. OFDMA may improve feasibility/energy through its less restrictive total-power constraint, not because the rate formula is changed arbitrarily.',
 'Minimize sum q_i(f_i)=f_i/H_i^2*(exp(a_i/f_i)-1), a_i=ln2 B_i/L. This is convex. Inactive power limits give equal marginal derivatives. The principal Lambert W branch solves f_i=a_i/[1+W0((lambda H_i^2-1)/e)]. Clamp to the power-induced lower fraction and solve sum f=1 by scalar bisection.',
 'TDMA lower fraction=B_i/[L log2(1+H_i^2 P)]. OFDMA lower fraction is the root q_i(f)=P. Exact convex solution within numerical tolerance, continuous fractions only.',
 'For AirComp+access, analog optimization and RMS budget are identical and analog/digital durations are separated. Best design is selected over the same finite analog-duration grid. Native FL/FU quantization and raw RMS budget are preserved.'
 ],limitations=['Single stored update,head7,+10dB,ideal CSI/SIC/coding; no VLM accuracy or forgetting conclusion.',
 'Continuous OFDMA bandwidth allocation; no discrete tones, guard band, pilots, FEC or receiver circuit energy. Existing128-tone implementation is separate.',
 'The old completed proxy_objective_v2 OMA training is TDMA. This CPU result cannot relabel or reconstruct its changed OFDMA learning trajectory.',
 'Main analog coefficients are optimized per block, not a globally optimal joint head/quality/time allocation. Failed whole-vector designs excluded from cost ranking.'])
write('protocol.json',proto);write('started.json',dict(pid=os.getpid(),unix=time.time()));t0=time.time()
def allocation(bits,h,L,mode):
 bits=np.asarray(bits,float);active=np.flatnonzero(bits>0);full=np.zeros(len(bits))
 if not len(active):return dict(energy_j=0.,average_power=full,fractions=full.copy(),active_power_max=0.,capacity_margin=0.,kkt_error=0.)
 if L<=0:return None
 b=bits[active];g=h[active]**2;a=np.log(2)*b/L
 def power(f):return f/g*np.expm1(np.minimum(a/f,700))
 if np.any(power(np.ones(len(b)))>P*(1+1e-12)):return None
 if mode=='tdma':lo=b/(L*np.log2(1+g*P))
 else:lo=np.array([brentq(lambda f:f/gg*np.expm1(min(aa/f,700))-P,1e-10,1.,xtol=1e-14) for aa,gg in zip(a,g)])
 if lo.sum()>1+1e-10:return None
 def derivative(f):
  x=a/f;return (1+(x-1)*np.exp(x))/g
 if len(b)==1:f=np.ones(1);lam=0.
 elif abs(lo.sum()-1)<1e-11:f=lo/lo.sum();lam=float(derivative(lo).max())
 else:
  def fractions(lam):
   if lam==0:return np.ones(len(b))
   den=1+lambertw((lam*g-1)/np.e).real
   return np.maximum(lo,np.minimum(1,a/np.maximum(den,1e-15)))
  upper=float(derivative(lo).max())*(1+1e-9)
  lam=brentq(lambda z:float(fractions(z).sum()-1),0.,upper,xtol=1e-14,rtol=1e-13)
  f=fractions(lam)
 q=power(f);cap=L*f*np.log2(1+g*q/f);limit=P*f if mode=='tdma' else np.full(len(b),P)
 assert abs(f.sum()-1)<2e-8 and np.all(q<=limit*(1+2e-7)) and np.all(cap>=b-1e-6)
 active_power=q/f if mode=='tdma' else q
 free=f>lo+1e-7;kkt=float(np.max(np.abs(derivative(f)[free]-lam))) if free.any() and len(b)>1 else 0.
 assert kkt<1e-5*max(1.,lam)
 full[active]=q;ff=np.zeros(len(bits));ff[active]=f
 return dict(energy_j=L/W*float(q.sum()),average_power=full,fractions=ff,active_power_max=float(active_power.max()),capacity_margin=float((cap-b).min()),kkt_error=kkt)
def compact(x):
 if x is None:return None
 return {k:v for k,v in x.items() if not isinstance(v,np.ndarray)}
# Independent conic solver and existing OFDMA solver comparison on preset synthetic inputs.
rng=np.random.default_rng(947);solver_checks=[]
for i in range(30):
 h=np.exp(rng.uniform(.3,3.,5));bits=rng.integers(32,400,5).astype(float);L=[512,1024,2048][i%3]
 for mode in ['tdma','ofdma']:
  z=allocation(bits,h,L,mode);f=cp.Variable(5,nonneg=True);q=cp.Variable(5,nonneg=True)
  constraints=[cp.sum(f)<=1,cp.rel_entr(f,f+cp.multiply(h*h,q))+np.log(2)*bits/L<=0,q<=P*f if mode=='tdma' else q<=P]
  pr=cp.Problem(cp.Minimize(cp.sum(q)),constraints);pr.solve(solver='CLARABEL',tol_gap_abs=1e-10,tol_feas=1e-10,tol_gap_rel=1e-10,max_iter=200)
  if z is None:assert pr.status in ['infeasible','infeasible_inaccurate']
  else:
   assert pr.status in ['optimal','optimal_inaccurate'];rel=abs(z['energy_j']-pr.value*L/W)/max(z['energy_j'],1e-12);assert rel<3e-5
  solver_checks.append(dict(case=i,mode=mode,kkt_feasible=z is not None,conic_status=pr.status))
 td=allocation(bits,h,L,'tdma');of=allocation(bits,h,L,'ofdma')
 if td:assert of and of['energy_j']<=td['energy_j']*(1+1e-7)
 old=ns['old'].ofdma(bits,h,L)
 if old:assert of and of['energy_j']<=old['energy']*(1+3e-5)
write('solver_checks.json',solver_checks)
rows=[];verified=0;equal=0;embedded=0
for block in range(N):
 v=V[:,block];u=U[:,block];eps=float(allocated[block])
 for bits in [6,8]:
  qu,du,pu=quantize(u[:,None,:],bits,32);ql,dl,pl=quantize(v[:,None,:],bits,32)
  qb=float(du[ids,0].sum());qe=qb+float(dl[ids,0].sum());payload=pu[ids,0]
  ns['codec_bound']=qb;ns['codec_bias']=(qu[ids,0]-u[ids]).sum(0);engine=ns['ConvexOverlap'](H,v[ids],eps-qb)
  analog={L:engine.solve(L,np.zeros(len(ids))) for L in LAS}
  for total in TS:
   choices={'air_tdma':[],'air_ofdma':[]}
   for L,z in analog.items():
    if L>=total or z is None:continue
    td=allocation(payload,H,total-L,'tdma');of=allocation(payload,H,total-L,'ofdma')
    if td:
     assert of and of['energy_j']<=td['energy_j']*(1+1e-7);embedded+=1;equal+=int(abs(of['energy_j']-td['energy_j'])<1e-8*max(td['energy_j'],1e-12))
    for mode,x in [('tdma',td),('ofdma',of)]:
     if x:
      verified+=1;choices['air_'+mode].append(dict(energy_j=z['energy_j']+x['energy_j'],analog_symbols=L,digital_symbols=total-L,
       max_power_w=max(z['peak_power_w'],x['active_power_max']),error_bound=z['total_error_bound'],digital_energy_j=x['energy_j']))
   methods={k:min(x,key=lambda y:y['energy_j']) if x else None for k,x in choices.items()}
   for mode in ['tdma','ofdma']:
    x=allocation(payload+pl[ids,0],H,total,mode) if qe<=eps else None
    methods[mode+'_only']=dict(energy_j=x['energy_j'],max_power_w=x['active_power_max'],error_bound=qe) if x else None
   for a,b in [('air_tdma','air_ofdma'),('tdma_only','ofdma_only')]:
    if methods[a]:assert methods[b] and methods[b]['energy_j']<=methods[a]['energy_j']*(1+1e-7)
   for x in methods.values():
    if x:assert x['max_power_w']<=P*(1+4e-6) and x['error_bound']<=eps*(1+4e-6)
   rows.append(dict(block=block,bits=bits,total_block_symbols=total,methods=methods))
 if (block+1)%40==0:print(json.dumps(dict(blocks=block+1,elapsed_s=time.time()-t0)),flush=True)
write('block_results.json',rows);summary=[]
old=json.loads((ROOT/'results/overnight_energy_v2/all_block_radio_audit_v2/report.json').read_text())
for bits in [6,8]:
 for total in TS:
  group=[r for r in rows if r['bits']==bits and r['total_block_symbols']==total];methods={}
  for method in group[0]['methods']:
   designs=[r['methods'][method] for r in group if r['methods'][method] is not None];complete=len(designs)==N
   methods[method]=dict(complete=complete,feasible_blocks=len(designs),energy_j=sum(x['energy_j'] for x in designs) if complete else None,
    allocated_uplink_s=N*total/W if complete else None)
  previous=next(x for x in old['summary'] if x['bits']==bits and x['block_deadline_symbols']==total)['methods']['air_ofdma']
  if previous['complete']:assert methods['air_ofdma']['complete'] and methods['air_ofdma']['energy_j']<=previous['total_energy_j']*(1+4e-5)
  summary.append(dict(bits=bits,full_vector_deadline_s=N*total/W,methods=methods,previous_OFDMA_energy_j=previous.get('total_energy_j')))
checks=dict(independent_conic_checks=len(solver_checks),returned_digital_tail_designs=verified,TDMA_points_embedded_in_OFDMA=embedded,
 exact_equal_tail_costs=equal,whole_block_cases=len(rows),backend_unchanged=hashlib.sha256(source.read_bytes()).hexdigest()==proto['backend_sha256'])
assert checks['backend_unchanged'];report=dict(protocol=proto,checks=checks,summary=summary)
write('report.json',report);write('finished.json',dict(runtime_s=time.time()-t0,checks=checks));print('COMPLETE',time.time()-t0,flush=True)
