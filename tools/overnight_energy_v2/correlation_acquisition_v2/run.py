"""CPU deterministic projection certificate + acquisition cost, frozen updates only."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import ast,json,time,hashlib,fcntl
from pathlib import Path
import numpy as np
ROOT=Path.cwd()
OUT=ROOT/'results/overnight_energy_v2/correlation_acquisition_v2'
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
variant=variant.replace(old,"self.mix = certificate_projection.T * self.H[None, :]\n        self.rem = certificate_residual * self.H\n        self.qerr = certificate_delta * self.H\n        self.raw_mix = np.divide(vec, self.rho[:, None], out=np.zeros_like(vec), where=self.rho[:, None] > 0).T * self.H[None, :]\n        n = cp.norm(cp.hstack([1, cp.norm(self.mix @ self.a) + self.qerr @ self.a, self.rem @ self.a]))")
assert variant.count('(self.H @ a) ** 2')==2
variant=variant.replace('(self.H @ a) ** 2','float((np.linalg.norm(self.mix @ a) + self.qerr @ a) ** 2 + (self.rem @ a) ** 2)')
variant=variant.replace('result = dict(','''exact_i = float(np.sum((self.raw_mix @ a) ** 2))
        cert_i = float((np.linalg.norm(self.mix @ a) + self.qerr @ a) ** 2 + (self.rem @ a) ** 2)
        assert exact_i <= cert_i * (1 + 1e-9) + 1e-12
        result = dict(exact_interference=exact_i, certificate_interference=cert_i, ''')
ns={'__name__':'cert','__file__':str(source)};exec(compile(variant,'isolated_projection_certificate','exec'),ns)
from wireless import quantize
V=ns['learning'];U=ns['unlearning'];D=ns['D'];W=ns['W'];P=ns['P'];eps=ns['total_eps']
head=ns['head'];ids=np.delete(np.arange(len(V)),head)
rho=np.linalg.norm(V[ids],axis=1)
unit=np.divide(V[ids],rho[:,None],out=np.zeros_like(V[ids]),where=rho[:,None]>0)
rng=np.random.default_rng(647)
Q,_=np.linalg.qr(rng.normal(size=(D,D)))
assert np.max(np.abs(Q.T@Q-np.eye(D)))<1e-12
RANKS=[8,16,32,64,128];DB=[0,10];BITS=[6,8];DEADLINES=[800,2048,8192]
def encode(x,b):
 q,d,p=quantize(x[:,None,:],b,32)
 return q[:,0,:],d[:,0],p[:,0]
def compact(x):
 return {k:v for k,v in x.items() if k not in ['a','po','bias','power','fractions','pt','bandwidth_fractions']} if x else None
def best(a):return min(a,key=lambda x:x['energy_j']) if a else None
def checked_ofdma(bits,H,L):
 d=ns['old'].ofdma(bits,H,L)
 if d is None:return None
 active=bits>0;f=np.asarray(d['fractions']);pp=d['power'][active]
 assert abs(f.sum()-1)<1e-6
 assert pp.max()<=P*(1+1e-6)
 assert np.min(L*f*np.log2(1+H[active]**2*pp/f)-bits[active])>=-2e-5
 assert abs(d['energy']-L/W*d['power'].sum())<1e-12
 return d
protocol=dict(scope='Fixed-head deterministic projection certificate with OFDMA acquisition cost. CPU single frozen block, not VLM performance or deployable validated PHY.',
 head=head,head_policy='Original maximum mean log channel head, fixed before inspecting updates; same head for every scheme.',
 projection='Data-independent orthonormal Gaussian basis QR seed647; leading ranks; no rank selected from test or retrain data.',
 ranks=RANKS,projection_bits=8,fu_bits=BITS,gain_db=DB,deadline_symbols=DEADLINES,
 certificate='Pythagorean bound: (||sum w q_projection|| + sum w quantizer_delta)^2 + (sum w upward_float32 residual_norm)^2. Projection and its complement are orthogonal.',
 packet='Native Q8 projected unit direction +32-bit residual-norm certificate +32 hypothetical framing bits; native quantizer includes scale metadata. Norm rho and CSI assumed available in original model.',
 acquisition='Dedicated OFDMA before data; optimized continuous bandwidth; power <=0.32W; same total deadline includes acquisition and data.',
 cost='Conservative RF TX bound, pure uplink. No local/receiver/SIC time. Controller feedback and hardware/FEC cost omitted, so this is an optimistic screen.',
 no_free_reuse='Where acquisition suffices for the same RMS budget, also evaluate skipping analog FL and sending only FU by NOMA/OFDMA.',
 original_backend_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
 source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 input_sha256=hashlib.sha256((ROOT/'results/overnight_energy_v2/short_regime_probe_v1/fixed_input_block.npz').read_bytes()).hexdigest())
write('protocol.json',protocol);write('started.json',dict(pid=os.getpid(),unix=time.time()))
prior=json.loads((ROOT/'results/overnight_energy_v2/head_energy_deadline_v1/results.json').read_text())
baselines=[x for x in prior if x['head']==head and x['deadline_symbols'] in DEADLINES]
assert len(baselines)==12
t0=time.time();rows=[];audits=[];verified=0;cert_checks=0
# A zero-dimensional sketch with a unit residual must recover the original norm-triangle engine.
for space in [ns,base_ns]:
 space['codec_bound']=0.;space['codec_bias']=np.zeros(D)
ns['certificate_projection']=np.zeros((len(ids),1))
ns['certificate_residual']=np.ones(len(ids));ns['certificate_delta']=np.zeros(len(ids))
Htest=ns['base'][ids,head]*10**.5
_,_,btest=encode(U,6)
atest=ns['ConvexOverlap'](Htest,V[ids],eps).solve(2048,btest[ids])
btest=base_ns['ConvexOverlap'](Htest,V[ids],eps).solve(2048,btest[ids])
assert atest is not None and btest is not None
assert abs(atest['energy_j']/btest['energy_j']-1)<2e-5

for rank in RANKS:
 projected=unit@Q[:,:rank]
 qp,delta,payload=encode(projected,8)
 residual=np.linalg.norm(unit-projected@Q[:,:rank].T,axis=1)
 rem32=np.nextafter(residual.astype(np.float32),np.float32(np.inf)).astype(float)
 assert np.all(rem32>=residual)
 remainder=rem32+delta
 payload=payload+64
 # Pythagorean/quantization certificate; use arbitrary nonnegative amplitude weights.
 weights=rng.lognormal(size=(1000,len(ids)))
 actual=np.linalg.norm(weights@unit,axis=1)
 bound=np.sqrt((np.linalg.norm(weights@qp,axis=1)+weights@delta)**2+(weights@rem32)**2)
 assert np.all(actual<=bound*(1+1e-12)+1e-12)
 cert_checks+=len(weights)
 audit=dict(rank=rank,bits_per_remote=payload.tolist(),total_acquisition_bits=int(payload.sum()),
     mean_residual_norm=float(residual.mean()),mean_quantizer_delta=float(delta.mean()),
     fl_reconstruction_error_bound=float(np.hypot(rho@delta,rho@rem32)),
     actual_fl_reconstruction_error=float(np.linalg.norm((rho[:,None]*(qp@Q[:,:rank].T-unit)).sum(0))),
     certificate_random_checks=1000)
 audits.append(audit)
 for db in DB:
  H=ns['base'][ids,head]*10**(db/20)
  for bits in BITS:
   qu,du,bu=encode(U,bits);qb=float(du[ids].sum())
   ns['codec_bound']=qb;ns['codec_bias']=(qu[ids]-U[ids]).sum(0)
   ns['certificate_projection']=qp;ns['certificate_residual']=rem32;ns['certificate_delta']=delta
   eng=ns['ConvexOverlap'](H,V[ids],eps-qb)
   bypass_q=float(np.hypot(rho@delta,rho@rem32)+qb)
   for T in DEADLINES:
    acq_lengths=sorted(set([64,128,256,384,512,768,1024,1536,2048,3072,4096,6144]+[int(T*f) for f in [.1,.25,.5,.75]]))
    candidates=[];bypasses=[]
    for Lmeta in acq_lengths:
     if Lmeta>=T:continue
     acq=checked_ofdma(payload,H,Lmeta)
     if acq is None:continue
     Ldata=T-Lmeta
     d=eng.solve(Ldata,bu[ids])
     if d is not None:
      cert_energy=d['energy_j'];value=compact(d)
      value.update(energy_j=acq['energy']+cert_energy,acquisition_energy_j=acq['energy'],
       data_energy_j=cert_energy,acquisition_symbols=Lmeta,data_symbols=Ldata,
       total_symbols=T,block_average_power_bound_w=max(d['peak_power_w'],float(acq['power'].max())))
      assert value['block_average_power_bound_w']<=P*(1+4e-6)
      assert value['total_error_bound']<=eps*(1+4e-6)
      verified+=1;candidates.append(value)
     if bypass_q<=eps:
      dt=ns['noma_tail'](H,bu[ids],Ldata)
      of=checked_ofdma(bu[ids],H,Ldata)
      for mode,tail in [('noma',dt),('ofdma',of)]:
       if tail:
        e=tail['energy_j'] if mode=='noma' else tail['energy']
        peak=tail['peak_power_w'] if mode=='noma' else float(tail['power'].max())
        bypasses.append(dict(energy_j=acq['energy']+e,acquisition_energy_j=acq['energy'],data_energy_j=e,
          acquisition_symbols=Lmeta,data_symbols=Ldata,total_symbols=T,
          block_average_power_bound_w=max(float(acq['power'].max()),peak),
          total_error_bound=bypass_q,mode='reuse_projection_no_analog_'+mode))
        verified+=1
    baseline=next(x for x in baselines if x['db']==db and x['bits']==bits and x['deadline_symbols']==T)
    rows.append(dict(rank=rank,db=db,bits=bits,deadline_symbols=T,head=head,
       total_acquisition_bits=int(payload.sum()),projection_then_full_noma=best(candidates),
       reuse_projection_no_analog=best(bypasses),projection_reconstruction_bound=bypass_q,
       baseline_methods={k:compact(v) for k,v in baseline['methods'].items()}))
 write('results.json',rows)
 print(json.dumps(dict(rank=rank,completed=len(rows))),flush=True)
summary=[]
for db in DB:
 for bits in BITS:
  for T in DEADLINES:
   g=[r for r in rows if r['db']==db and r['bits']==bits and r['deadline_symbols']==T]
   chosen=best([dict(r['projection_then_full_noma'],rank=r['rank']) for r in g if r['projection_then_full_noma']])
   reuse=best([dict(r['reuse_projection_no_analog'],rank=r['rank']) for r in g if r['reuse_projection_no_analog']])
   summary.append(dict(db=db,bits=bits,deadline_symbols=T,projection_then_full_noma=chosen,
     reuse_projection_no_analog=reuse,baseline_methods=g[0]['baseline_methods']))
checks=dict(zero_information_reduces_to_original=True,pythagorean_certificate=True,returned_designs_verified=verified,certificate_weight_tests=cert_checks,orthonormal_projection=True,
 source_unchanged=hashlib.sha256(source.read_bytes()).hexdigest()==protocol['original_backend_sha256'],
 acquisition_energy_and_time_included=True,analog_skip_competitor_included=True)
assert checks['source_unchanged'] and len(rows)==60
write('certificate_audit.json',audits);write('report.json',dict(protocol=protocol,checks=checks,cases=summary,certificate_audit=audits))
write('finished.json',dict(cases=len(rows),runtime_s=time.time()-t0,checks=checks))
print('COMPLETE',time.time()-t0,flush=True)

