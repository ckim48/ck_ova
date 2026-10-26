"""Minimum found uplink duration at fixed total TX energy and update fidelity.
CPU only, frozen real updates, new isolated version. Not a learning experiment.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,json,time,hashlib,importlib.util,math,warnings
import numpy as np
import cvxpy as cp
warnings.filterwarnings('ignore',category=UserWarning,module='cvxpy')
ROOT=Path.cwd()
OLD=ROOT/'tools/overnight_energy_v2/short_regime_probe_v1/run.py'
spec=importlib.util.spec_from_file_location('old_probe',OLD)
old=importlib.util.module_from_spec(spec)
sys.path[:0]=[str(ROOT/'tools/overnight_energy_v2/retrieval_fourway_v2'),str(ROOT/'tools/overnight_energy_v2/initialization_v11'),str(ROOT)]
spec.loader.exec_module(old)
OUT=ROOT/'results/overnight_energy_v2/minimum_latency_v2'
OUT.mkdir(parents=True,exist_ok=True)
W=1e6;P=.32;D=128;MAXL=10000
CAPS=[.0001,.0003,.001]
GDB=[-10,0,10]
LAS=[128,160,192,256,384,512,768,1024,1536,2048,4096,8192]
ETAS=[.0025,.005,.01,.025,.05,.1,.25,.5,.75,.9]
t0=time.time()
def write(n,x):
    (OUT/n).write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def compact(x):
    return {k:v for k,v in x.items() if k not in ('a','po','pt','bits_overlap','bits_tail','bias')}
z=np.load(ROOT/'results/overnight_energy_v2/short_regime_probe_v1/fixed_input_block.npz')
learning=z['learning'];unlearning=z['unlearning'];base=z['channels']
head=int(np.argmax([np.mean(np.log(np.delete(base[:,v],v))) for v in range(10)]))
ids=np.delete(np.arange(10),head)
ql,bl=old.encode(learning,128);qu,bu=old.encode(unlearning,128)
codec_bound=float(np.linalg.norm(qu[ids]-unlearning[ids],axis=1).sum())
total_eps=.2*float(np.linalg.norm(learning,axis=1).sum())
radio_eps=total_eps-codec_bound
assert radio_eps>0
codec_bias=(qu[ids]-unlearning[ids]).sum(0)
def coefficients(bits,H,L):
    if L<=0:return np.zeros(len(H)) if not np.any(bits) else None
    co=np.zeros(len(H));later=0.
    order=np.flatnonzero(bits>0);order=order[np.argsort(-H[order],kind='stable')]
    for i in order[::-1]:
        x=np.log(2)*bits[i]/L
        if x>300:return None
        co[i]=np.expm1(x)*(1+later);later+=co[i]
    return co

class ConvexOverlap:
    """Exact change of variables z=eps/rx for fixed length/overlap bits.
    sum |H*a - rho*z/eps| + sqrt(D/(2L)) <= z is convex.
    This eliminates receiver-grid initialization for perfect SIC, one block.
    """
    def __init__(self,H,vec,eps):
        self.H=np.asarray(H);self.vec=vec;self.rho=np.linalg.norm(vec,axis=1);self.eps=eps
        self.a=cp.Variable(len(H),nonneg=True);self.t=cp.Variable(len(H),nonneg=True)
        self.zz=cp.Variable(nonneg=True);self.alpha=cp.Parameter(len(H),nonneg=True)
        self.noise=cp.Parameter(nonneg=True)
        n=cp.norm(cp.hstack([1,self.H@self.a]))
        self.constraints=[self.a+self.t<=np.sqrt(P),
            cp.multiply(self.rho==0,self.a)==0,
            cp.multiply(self.alpha,n)<=self.t,
            cp.norm1(cp.multiply(self.H,self.a)-self.rho/self.eps*self.zz)+self.noise<=self.zz]
        self.problem=cp.Problem(cp.Minimize(cp.sum_squares(self.a+self.t)),self.constraints)
        assert self.problem.is_dcp()
        self.cache={}
    def solve(self,L,bits):
        key=(int(L),tuple(np.asarray(bits,dtype=int)))
        if key in self.cache:return self.cache[key]
        self.cache[key]=None
        if L<D or L>MAXL:return None
        co=coefficients(bits,self.H,L)
        if co is None or np.max(co/self.H**2)>P*(1+1e-9):return None
        self.alpha.value=np.sqrt(co)/self.H;self.noise.value=np.sqrt(D/(2*L))
        try:self.problem.solve(solver='CLARABEL',max_iter=300,tol_gap_abs=1e-10,tol_gap_rel=1e-10,tol_feas=1e-10)
        except cp.error.SolverError:return None
        if self.problem.status not in ('optimal','optimal_inaccurate') or self.a.value is None:return None
        a=np.maximum(self.a.value,0);rx=self.eps/float(self.zz.value)
        po=co/self.H**2*(1+(self.H@a)**2)
        power=(a+np.sqrt(po))**2
        err=float(np.abs(rx*self.H*a-self.rho).sum()+rx*np.sqrt(D/(2*L)))
        if err>self.eps*(1+3e-6) or power.max()>P*(1+3e-6):return None
        gain=np.divide(rx*self.H*a,self.rho,out=np.zeros_like(a),where=self.rho>0)
        bias=((gain-1)[:,None]*self.vec).sum(0)
        rmse=float(np.sqrt(np.dot(bias+codec_bias,bias+codec_bias)+rx*rx*D/(2*L)))
        qrx=self.H**2*po;order=np.flatnonzero(bits>0);order=order[np.argsort(-self.H[order],kind='stable')]
        margins=[L*np.log2(1+qrx[i]/(1+(self.H@a)**2+qrx[order[j+1:]].sum()))-bits[i] for j,i in enumerate(order)]
        if min(margins,default=0)<-1e-5:return None
        result=dict(energy_j=float(L/W*power.sum()),expected_energy_j=float(L/W*np.sum(a*a+po)),
            peak_power_w=float(power.max()),radio_error_bound=err,total_error_bound=err+codec_bound,
            actual_expected_aggregate_rmse=rmse,rx=rx,analog_symbols=int(L),
            min_rate_margin_bits=float(min(margins,default=0)),solver_status=self.problem.status,
            a=a,po=po,bias=bias)
        self.cache[key]=result
        return result

def bsearch(fun,cap,lo,hi=MAXL):
    """First feasible integer symbol length; model feasibility monotone in tail length."""
    if lo>hi:return None
    def good(x):return x is not None and x['energy_j']<=cap*(1+1e-7)
    top=fun(hi)
    if not good(top):return None
    bottom=fun(lo)
    if good(bottom):return dict(bottom,symbols=int(lo),uplink_ms=lo/W*1000)
    while hi-lo>1:
        mid=(hi+lo)//2
        if good(fun(mid)):hi=mid
        else:lo=mid
    x=fun(hi)
    return dict(x,symbols=int(hi),uplink_ms=hi/W*1000,previous_symbol_feasible=False)

def noma_tail(H,bits,L):
    co=coefficients(bits,H,L)
    if co is None:return None
    p=co/H**2
    if p.max()>P*(1+2e-7):return None
    return dict(energy_j=float(L/W*p.sum()),peak_power_w=float(p.max()),pt=p)
def choose_best(candidates):
    return min(candidates,key=lambda x:(x['symbols'],x['energy_j'])) if candidates else None

protocol=dict(version='minimum_latency_v2',scope='Frozen 128-coordinate real-update block; minimum found transmission duration, not time to VLM learning/unlearning target.',
    users=10,head=head,remote_fu_requests=int(np.count_nonzero(bu[ids])),bandwidth_hz=W,power_limit_w=P,
    total_network_energy_caps_j=CAPS,channel_gain_shifts_db=GDB,max_symbols=MAXL,
    time_resolution_us=1,analog_symbol_grid=LAS,payload_fraction_grid=ETAS,
    full_sic='Perfect SIC, ideal CSI/coding, fixed descending-channel order.',
    receiver='Joint convex reciprocal-receiver formulation for one block; no AO initialization.',
    ofdma='Continuous optimized bandwidth, analog duration grid; same power and total-energy cap.',
    codec='No FU sparsification; Q8/32bit scale, 1056 bits per active FU UAV, same canonical FU for all schemes.',
    total_raw_update_rms_budget=total_eps,codec_triangle_bound=codec_bound,radio_rms_budget=radio_eps,
    quality='All methods satisfy same conservative aggregate RMS bound against the unquantized ideal update. Digital methods also decode each canonical FU payload. Neither is a VLM deletion-quality guarantee.',
    airair='One simultaneous FL+canonical-FU aggregate. Individually unidentifiable, but not marked an unlearning failure.',
    costs='RF TX upper bound, uplink occupation only. No compute, receiver/SIC, downlink or flight costs.',
    input_sha256=hashlib.sha256((ROOT/'results/overnight_energy_v2/short_regime_probe_v1/fixed_input_block.npz').read_bytes()).hexdigest(),
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    limitations=['One channel geometry and one update block; scenario gains are hypothetical.',
      'Partial/OFDMA time and split grids are restricted searches, not global minimum-latency certificates.',
      'Full overlap duration search uses integer symbols and conic numerical tolerances.',
      'Existing training trajectories cannot be retimed as new validated training trajectories.'])
write('protocol.json',protocol)
active=np.flatnonzero(bu[ids])
write('codec_audit.json',dict(stack_relative_l2=float(np.linalg.norm(qu[ids]-unlearning[ids])/np.linalg.norm(unlearning[ids])),
    per_request_relative_l2=[float(np.linalg.norm(qu[ids[i]]-unlearning[ids[i]])/np.linalg.norm(unlearning[ids[i]])) for i in active],
    digital_fl_fu_total_quantization_bound=float(np.linalg.norm(ql[ids]-learning[ids],axis=1).sum()+codec_bound)))

# Analytical one-user check and cross-check against old receiver-grid results.
testH=np.array([2.]);testV=np.array([[1.]+[0.]*(D-1)])
test=ConvexOverlap(testH,testV,.5).solve(1024,np.array([0]))
assert test is not None and test['total_error_bound']-codec_bound<=.5*(1+1e-6)
assert test['energy_j']>0
baseline=ConvexOverlap(base[ids,head]*10**.5,learning[ids],total_eps).solve(1024,bu[ids])
previous=json.loads((ROOT/'results/overnight_energy_v2/short_regime_probe_v1/tight_refinement.json').read_text())
prev=next(x for x in previous if x['symbols']==1024)['full_noma']['energy_j']
assert baseline is not None and baseline['energy_j']<=prev*(1+2e-5)
write('checks.json',dict(passed=True,convexity=True,one_user_feasibility=True,
    joint_receiver_energy_j=baseline['energy_j'],previous_grid_energy_j=prev,joint_not_worse=True))
rows=[];designs=[];errors=[]
for db in GDB:
    H=base[ids,head]*10**(db/20)
    engine=ConvexOverlap(H,learning[ids],radio_eps)
    aa_engine=ConvexOverlap(H,(learning+qu)[ids],radio_eps)
    # Precompute all overlap states once; tails change analytically with duration.
    states=[]
    for la in LAS:
        for eta in [0.]+ETAS:
            bo=np.floor(bu[ids]*eta+.5)
            a=engine.solve(la,bo)
            if a is not None:states.append((la,eta,bo,a))
    for cap in CAPS:
        full=bsearch(lambda L:engine.solve(L,bu[ids]),cap,D)
        airair=bsearch(lambda L:aa_engine.solve(L,np.zeros(len(ids))),cap,D)
        oma=[];partial=[];sequential=[]
        for la,eta,bo,a in states:
            if a['energy_j']>=cap or la>=MAXL:continue
            bt=bu[ids]-bo
            def eval_noma(L):
                d=noma_tail(H,bt,L-la)
                if d is None:return None
                return dict(a,energy_j=a['energy_j']+d['energy_j'],
                    expected_energy_j=a['expected_energy_j']+d['energy_j'],
                    peak_power_w=max(a['peak_power_w'],d['peak_power_w']),
                    digital_tail_symbols=int(L-la),fu_overlap_fraction=float(bo.sum()/bu[ids].sum()),pt=d['pt'],
                    bits_overlap=bo,bits_tail=bt)
            design=bsearch(eval_noma,cap,la+1)
            if design:
                (sequential if eta==0 else partial).append(design)
            if eta==0:
                def eval_oma(L):
                    d=old.ofdma(bu[ids],H,L-la)
                    if d is None:return None
                    return dict(a,energy_j=a['energy_j']+d['energy'],
                        expected_energy_j=a['expected_energy_j']+d['energy'],
                        peak_power_w=max(a['peak_power_w'],float(d['power'].max())),
                        digital_tail_symbols=int(L-la),bandwidth_fractions=d['fractions'],
                        pt=d['power'],bits_overlap=bo,bits_tail=bu[ids])
                design=bsearch(eval_oma,cap,la+1)
                if design:oma.append(design)
        def eval_digital(L):
            d=old.ofdma((bu+bl)[ids],H,L)
            if d is None:return None
            qe=float(np.linalg.norm(ql[ids]-learning[ids],axis=1).sum()+codec_bound)
            if qe>total_eps:return None
            return dict(energy_j=d['energy'],expected_energy_j=d['energy'],
                peak_power_w=float(d['power'].max()),total_error_bound=qe,
                bandwidth_fractions=d['fractions'])
        digital=bsearch(eval_digital,cap,1)
        result=dict(channel_gain_db=db,energy_cap_j=cap,
            full_noma=full,partial_noma=choose_best(partial),air_ofdma=choose_best(oma),
            separate_noma=choose_best(sequential),airair=airair,ofdma_only=digital)
        public={k:compact(v) if isinstance(v,dict) else v for k,v in result.items()}
        if full and public['air_ofdma']:public['full_noma_time_saving_vs_ofdma_percent']=100*(1-full['symbols']/public['air_ofdma']['symbols'])
        if public['partial_noma'] and public['air_ofdma']:public['partial_time_saving_vs_ofdma_percent']=100*(1-public['partial_noma']['symbols']/public['air_ofdma']['symbols'])
        for k,v in result.items():
            if not isinstance(v,dict):continue
            assert v['energy_j']<=cap*(1+1e-6)
            assert v['peak_power_w']<=P*(1+4e-6)
            assert v['total_error_bound']<=total_eps*(1+4e-6)
        rows.append(public);write('results.json',rows)
        print(json.dumps(dict(db=db,cap_mJ=cap*1000,durations_ms={k:round(v['uplink_ms'],3) if isinstance(v,dict) else None for k,v in result.items() if k not in ['channel_gain_db','energy_cap_j']})),flush=True)
        designs.append(result)
write('status.json',dict(stage='complete',cases=len(rows),runtime_s=time.time()-t0))
print('COMPLETE',time.time()-t0,flush=True)
