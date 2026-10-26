"""CPU-only diagnostic: quantization, energy budget and overlap frontiers.
Fixed raw updates, common fidelity, power and resources; not a VLM experiment.
"""
from pathlib import Path
import os, ast, json, time, hashlib, fcntl
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np

ROOT=Path.cwd()
OUT=ROOT/'results/overnight_energy_v2/sic_sensitivity_v2'
OUT.mkdir(exist_ok=True,parents=True)
lock=(OUT/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (OUT/'started.json').exists(), 'Refuse duplicate campaign'
def write(name,x):
    p=OUT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False));t.replace(p)

source=ROOT/'tools/overnight_energy_v2/minimum_latency_v2/run.py'
tree=ast.parse(source.read_text());prefix=[]
for node in tree.body:
    if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='protocol' for t in node.targets):break
    prefix.append(node)
assert any(isinstance(n,ast.FunctionDef) and n.name=='choose_best' for n in prefix)
ns={'__name__':'isolated_latency_backend','__file__':str(source)}
exec(compile(ast.Module(body=prefix,type_ignores=[]),str(source),'exec'),ns)
ns['OUT']=OUT
V=ns['learning'];U=ns['unlearning'];ids=ns['ids'];eps=ns['total_eps']
W=ns['W'];P=ns['P'];D=ns['D'];BS=ns['bsearch'];best=ns['choose_best']
CVX=ns['ConvexOverlap'];tail=ns['noma_tail'];MAXL=ns['MAXL']
QUANTS=[6,8];DB=[0,10];CAPS=[.0003,None]
protocol=dict(sic_residual_power_fractions=[0.,1e-6,1e-4,1e-3,1e-2],
    residual_assumption='Hypothetical sensitivity, not measured SIC quality. Conservative coherent residual RMS as in manuscript. OFDMA and raw AirAir have no SIC penalty.',
    scope='one frozen real 128-coordinate block; not VLM performance or convergence',
    quantization_bits=QUANTS,channel_gain_db=DB,network_energy_caps_j=CAPS,
    quantizer_error_policy='Native float32 scale and quantizer supplied conservative error bounds. v3 used measured per-user error norms.',
    raw_update_rms_budget=eps,power_per_user_w=P,bandwidth_hz=W,head=ns['head'],
    users=10,remote_requests=int(np.count_nonzero(np.linalg.norm(U[ids],axis=1))),
    fairness='Identical raw update target, RMS budget and power cap. Digital methods share canonical Qb FU. Strong AirComp+AirComp sends the raw FL+FU sum without quantization.',
    resource_scope='RF TX conservative bound; uplink occupation only. Residual SIC sensitivity, ideal CSI/coding.',
    optimization='Joint convex receiver/transmitter for fixed overlap. Restricted analog duration/payload split grid; optimized continuous OFDMA bandwidth.',
    adaptive='Minimum among full, nonzero partial and zero-overlap NOMA. Choosing zero overlap is allowed and disclosed.',
    limitations=['One head and geometry, hypothetical gain shifts. No deletion guarantee from update error.',
                 'Power-only removes the chosen network energy cap, not the physical per-user power limit.',
                 'AirComp+AirComp recovers only the sum; no artificial failure penalty.',
                 'nominal_rmse_without_sic_residual excludes unknown residual statistics and is not actual RMSE at positive chi.',
                 'Payload split assumes shared existing metadata, no extra fragment headers.',
                 'Offline fixed-update geometry. Update-statistic signaling and feedback costs are not included.',
                 'Failure of this conservative bound is not proof that true MSE violates the target.'],
    backend_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    input_sha256=hashlib.sha256((ROOT/'results/overnight_energy_v2/short_regime_probe_v1/fixed_input_block.npz').read_bytes()).hexdigest())
write('protocol.json',protocol);write('started.json',dict(pid=os.getpid(),unix=time.time()))
t0=time.time();rows=[];audits=[]
def encode(x,b):
    from wireless import quantize
    q,delta,payload=quantize(x[:,None,:],b,32)
    q=q[:,0,:];delta=delta[:,0];payload=payload[:,0]
    actual=np.linalg.norm(q-x,axis=1)
    assert np.all(actual<=delta+1e-12), 'native quantizer bound violated'
    return q,payload,delta


# Residual-SIC extension; original backend is read-only.
cp=ns['cp']
CHI=0.
CHIS=[0.,1e-6,1e-4,1e-3,1e-2]
def sic_coefficients(bits,H,L,chi):
    bits=np.asarray(bits)
    co=np.zeros(len(H))
    order=np.flatnonzero(bits>0)
    order=order[np.argsort(-H[order],kind='stable')]
    if not len(order):return co
    if L<=0 or np.max(np.log(2)*bits[order]/L)>300:return None
    gamma=np.expm1(np.log(2)*bits[order]/L)
    F=gamma[:,None]*(np.triu(np.ones((len(order),len(order))),1)+chi*np.tril(np.ones((len(order),len(order))),-1))
    if max(abs(np.linalg.eigvals(F)),default=0)>=1-1e-12:return None
    try:required=np.linalg.solve(np.eye(len(order))-F,gamma)
    except np.linalg.LinAlgError:return None
    if np.min(required)<0 or not np.all(np.isfinite(required)):return None
    scale=np.linalg.norm(required,np.inf)*(1+np.linalg.norm(F,np.inf))+np.linalg.norm(gamma,np.inf)
    assert np.linalg.norm(required-F@required-gamma,np.inf)<=1e-10*scale
    co[order]=required
    return co

class ResidualOverlap(ns['ConvexOverlap']):
    def __init__(self,H,vec,eps):
        super().__init__(H,vec,eps)
        self.chi=CHI
        residual=cp.sum(cp.multiply(np.sqrt(self.chi)*self.H,self.t))
        self.constraints[-1]=cp.norm1(cp.multiply(self.H,self.a)-self.rho/self.eps*self.zz)+self.noise+residual<=self.zz
        self.problem=cp.Problem(cp.Minimize(cp.sum_squares(self.a+self.t)),self.constraints)
        assert self.problem.is_dcp()
    def solve(self,L,bits):
        key=(int(L),tuple(np.asarray(bits,dtype=int)))
        if key in self.cache:return self.cache[key]
        self.cache[key]=None
        if L<D or L>MAXL:return None
        co=sic_coefficients(bits,self.H,L,self.chi)
        if co is None or np.max(co/self.H**2)>P*(1+1e-9):return None
        self.alpha.value=np.sqrt(co)/self.H;self.noise.value=np.sqrt(D/(2*L))
        try:self.problem.solve(solver='CLARABEL',max_iter=300,tol_gap_abs=1e-10,tol_gap_rel=1e-10,tol_feas=1e-10)
        except cp.error.SolverError:return None
        if self.problem.status not in ('optimal','optimal_inaccurate') or self.a.value is None:return None
        if self.zz.value is None or float(self.zz.value)<=0:return None
        a=np.maximum(self.a.value,0);rx=self.eps/float(self.zz.value)
        po=co/self.H**2*(1+(self.H@a)**2);power=(a+np.sqrt(po))**2
        residual=float(rx*np.sqrt(self.chi)*np.sum(self.H*np.sqrt(po)))
        err=float(np.abs(rx*self.H*a-self.rho).sum()+rx*np.sqrt(D/(2*L))+residual)
        if err>self.eps*(1+3e-6) or power.max()>P*(1+3e-6):return None
        gain=np.divide(rx*self.H*a,self.rho,out=np.zeros_like(a),where=self.rho>0)
        bias=((gain-1)[:,None]*self.vec).sum(0)
        nominal_rmse=float(np.sqrt(np.dot(bias+ns['codec_bias'],bias+ns['codec_bias'])+rx*rx*D/(2*L)))
        qrx=self.H**2*po;order=np.flatnonzero(bits>0)
        order=order[np.argsort(-self.H[order],kind='stable')]
        margins=[L*np.log2(1+qrx[i]/(1+(self.H@a)**2+qrx[order[j+1:]].sum()+self.chi*qrx[order[:j]].sum()))-bits[i] for j,i in enumerate(order)]
        if min(margins,default=0)<-1e-5:return None
        result=dict(energy_j=float(L/W*power.sum()),expected_energy_j=float(L/W*np.sum(a*a+po)),
            peak_power_w=float(power.max()),radio_error_bound=err,total_error_bound=err+ns['codec_bound'],
            nominal_rmse_without_sic_residual=nominal_rmse,residual_rms_bound=residual,
            rx=rx,analog_symbols=int(L),min_rate_margin_bits=float(min(margins,default=0)),
            solver_status=self.problem.status,a=a,po=po,bias=bias)
        self.cache[key]=result
        return result

def residual_tail(H,bits,L):
    co=sic_coefficients(bits,H,L,CHI)
    if co is None:return None
    p=co/H**2
    if p.max()>P*(1+2e-7):return None
    order=np.flatnonzero(bits>0);order=order[np.argsort(-H[order],kind='stable')]
    margins=[L*np.log2(1+co[i]/(1+co[order[j+1:]].sum()+CHI*co[order[:j]].sum()))-bits[i] for j,i in enumerate(order)]
    if min(margins,default=0)<-1e-5:return None
    return dict(energy_j=float(L/W*p.sum()),peak_power_w=float(p.max()),min_rate_margin_bits=float(min(margins,default=0)))

CVX=ResidualOverlap;tail=residual_tail

# Independent two-user analytic solution and perfect-SIC reduction checks.
for chi in CHIS:
    two=sic_coefficients(np.array([32.,64.]),np.array([2.,1.]),128,chi)
    g=np.expm1(np.log(2)*np.array([32.,64.])/128)
    q1=g[0]*(1+g[1])/(1-chi*g[0]*g[1]);q2=g[1]*(1+chi*q1)
    assert np.allclose(two,[q1,q2],rtol=1e-12,atol=1e-12)
assert sic_coefficients(np.array([128.,128.]),np.array([2.,1.]),128,1.) is None
for L in [128,1024,10000]:
    probe=np.where(np.linalg.norm(U[ids],axis=1)>0,800.,0.)
    assert np.allclose(sic_coefficients(probe,ns['base'][ids,ns['head']],L,0),ns['coefficients'](probe,ns['base'][ids,ns['head']],L),rtol=1e-10,atol=1e-10)

for CHI in CHIS:
    for db in DB:
        H=ns['base'][ids,ns['head']]*10**(db/20)
        # Raw analog baseline is independent of FU quantizer precision.
        ns['codec_bound']=0.;ns['codec_bias']=np.zeros(D)
        aa=CVX(H,(V+U)[ids],eps)
        aa_cases={cap:BS(lambda L:aa.solve(L,np.zeros(len(ids))),float('inf') if cap is None else cap,D) for cap in CAPS}
        for bits in QUANTS:
            q,b,delta=encode(U,bits);ql,bl,delta_l=encode(V,bits)
            qb=float(delta[ids].sum())
            measured=float(np.linalg.norm(q[ids]-U[ids],axis=1).sum())
            ns['codec_bound']=qb;ns['codec_bias']=(q[ids]-U[ids]).sum(0)
            qe=qb+float(delta_l[ids].sum())
            audits.append(dict(chi=CHI,db=db,bits=bits,quantization_triangle_bound=qb,rms_budget=eps,
                               measured_triangle_error=measured,native_to_measured_ratio=qb/measured if measured else None,
                               budget_fraction=qb/eps,full_digital_bound=qe,quantizer_fits=qb<eps))
            states=[];engine=CVX(H,V[ids],eps-qb) if qb<eps else None
            if engine:
                for la in ns['LAS']:
                    for eta in [0.]+ns['ETAS']:
                        bo=np.floor(b[ids]*eta+.5);a=engine.solve(la,bo)
                        if a:states.append((la,eta,bo,a))
            for cap in CAPS:
                budget=float('inf') if cap is None else cap
                full=BS(lambda L:engine.solve(L,b[ids]),budget,D) if engine else None
                partial=[];separate=[];ofdma=[]
                for la,eta,bo,a in states:
                    if a['energy_j']>=budget:continue
                    bt=b[ids]-bo
                    def f_noma(L):
                        d=tail(H,bt,L-la)
                        if d is None:return None
                        return dict(a,energy_j=a['energy_j']+d['energy_j'],
                            expected_energy_j=a['expected_energy_j']+d['energy_j'],
                            peak_power_w=max(a['peak_power_w'],d['peak_power_w']),
                            min_rate_margin_bits=min(a['min_rate_margin_bits'],d['min_rate_margin_bits']),
                            digital_tail_symbols=L-la,fu_overlap_fraction=float(bo.sum()/b[ids].sum()))
                    x=BS(f_noma,budget,la+1)
                    if x:(separate if eta==0 else partial).append(x)
                    if eta==0:
                        def f_oma(L):
                            d=ns['old'].ofdma(b[ids],H,L-la)
                            if d is None:return None
                            return dict(a,energy_j=a['energy_j']+d['energy'],
                                expected_energy_j=a['expected_energy_j']+d['energy'],
                                peak_power_w=max(a['peak_power_w'],float(d['power'].max())),
                                digital_tail_symbols=L-la)
                        x=BS(f_oma,budget,la+1)
                        if x:ofdma.append(x)
                def f_digital(L):
                    if qe>eps:return None
                    d=ns['old'].ofdma((b+bl)[ids],H,L)
                    if d is None:return None
                    return dict(energy_j=d['energy'],expected_energy_j=d['energy'],
                        peak_power_w=float(d['power'].max()),total_error_bound=qe)
                methods=dict(full_noma=full,partial_noma=best(partial),air_ofdma=best(ofdma),
                    separate_noma=best(separate),airair_raw=aa_cases[cap],
                    ofdma_only=BS(f_digital,budget,1))
                choices=[dict(v,chosen_mode=k) for k,v in methods.items() if k in ['full_noma','partial_noma','separate_noma'] and v]
                methods['adaptive_noma']=best(choices)
                for v in methods.values():
                    if not v:continue
                    assert v['energy_j']<=budget*(1+1e-6)
                    assert v['peak_power_w']<=P*(1+4e-6)
                    assert v['total_error_bound']<=eps*(1+4e-6)
                row=dict(chi=CHI,db=db,bits=bits,cap_j=cap,methods={k:ns['compact'](v) if v else None for k,v in methods.items()})
                rows.append(row);write('results.json',rows)
                print(json.dumps(dict(chi=CHI,db=db,bits=bits,cap_j=cap,latency_ms={k:v['uplink_ms'] if v else None for k,v in methods.items()})),flush=True)
write('codec_audit.json',audits)

old_rows=json.loads((ROOT/'results/overnight_energy_v2/deadline_frontier_v4/results.json').read_text())
zero_checks=[]
for r in rows:
    if r['chi']!=0:continue
    previous=next(v for v in old_rows if all(v[k]==r[k] for k in ['db','bits','cap_j']))
    for method,value in r['methods'].items():
        old_value=previous['methods'][method]
        assert (value is None)==(old_value is None),(method,r['db'],r['bits'])
        if value:
            assert abs(value['symbols']-old_value['symbols'])<=1
            zero_checks.append(dict(method=method,db=r['db'],bits=r['bits'],cap_j=r['cap_j'],symbol_difference=value['symbols']-old_value['symbols']))
write('independent_checks.json',dict(two_user_closed_form=True,perfect_sic_coefficients=True,
    spectral_radius_rejection=True,zero_sic_v4_designs=len(zero_checks),zero_sic_symbol_differences=zero_checks,
    statement='Returned designs checked against power, energy, conservative RMS and SIC rate constraints. Not global optima or measured hardware results.'))

write('finished.json',dict(cases=len(rows),runtime_s=time.time()-t0,
    checks='All returned designs satisfy raw RMS, individual power and chosen energy constraints',
    original_backend_unchanged=hashlib.sha256(source.read_bytes()).hexdigest()==protocol['backend_sha256']))


