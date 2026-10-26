"""Short, CPU-only, fixed real-update link diagnostic. No learning claim."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json,time,hashlib,warnings
from pathlib import Path
import numpy as np
from scipy.optimize import brentq,minimize
warnings.filterwarnings('ignore',category=UserWarning,module='cvxpy')
ROOT=Path.cwd()
sys.path[:0]=[str(ROOT/'tools/overnight_energy_v2/retrieval_fourway_v2'),str(ROOT/'tools/overnight_energy_v2/initialization_v11'),str(ROOT)]
from fast_solver import optimize_head
OUT=ROOT/'results/overnight_energy_v2/short_regime_probe_v1'
OUT.mkdir(parents=True,exist_ok=True)
W=1e6;P=.32;D=128

def write(name,data):
    (OUT/name).write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')

def ofdma(bits,h,L):
    """Continuous bandwidth allocation, including per-user total power cap.
    More permissive than the existing 128-tone implementation; independent audit.
    """
    ids=np.flatnonzero(bits>0)
    if not len(ids):return dict(energy=0.,power=np.zeros(len(bits)),fractions=[])
    if L<=0:return None
    b=bits[ids];g=h[ids]**2
    def powers(f):
        with np.errstate(over='ignore',invalid='ignore'):
            return f/g*np.expm1(np.minimum(np.log(2)*b/(L*f),700))
    if np.any(powers(np.ones(len(ids)))>P):return None
    low=np.array([brentq(lambda f: f/gg*np.expm1(min(np.log(2)*bb/(L*f),700))-P,1e-6,1.) for bb,gg in zip(b,g)])
    if low.sum()>1+1e-9:return None
    initial=low+(1-low.sum())*b/b.sum()
    def objective(f):return float(powers(f).sum())
    def jac(f):
        a=np.log(2)*b/(L*f)
        return (np.exp(a)*(1-a)-1)/g
    if len(ids)==1:f=np.ones(1)
    else:
        r=minimize(objective,initial,jac=jac,bounds=[(x,1.) for x in low],
            constraints=[dict(type='eq',fun=lambda f:f.sum()-1,jac=lambda f:np.ones_like(f))],
            method='SLSQP',options=dict(maxiter=200,ftol=1e-12))
        f=r.x
        if not r.success:return None
    p=powers(f);cap=L*f*np.log2(1+g*p/f)
    if f.sum()>1+1e-7 or np.any(p>P*(1+1e-7)) or np.any(cap<b-1e-6):return None
    full=np.zeros(len(bits));full[ids]=p
    return dict(energy=float(L/W*p.sum()),power=full,fractions=f.tolist())

def encode(x,k):
    q=np.zeros_like(x);bits=np.zeros(len(x))
    for i,v in enumerate(x):
        if not np.any(v):continue
        ids=np.argsort(np.abs(v),kind='stable')[-k:]
        scale=np.max(np.abs(v[ids]))/127
        q[i,ids]=np.rint(v[ids]/scale)*scale
        bits[i]=32+k*8+(k*7 if k<D else 0)
    return q,bits

def compact(d,vec,h,head,L,eps):
    users=d.users;a=d.analog_rms[:,0];rx=float(d.receiver[0]);rho=np.linalg.norm(vec[users],axis=1)
    gains=np.divide(rx*h[users,head]*a,rho,out=np.zeros_like(a),where=rho>0)
    bias=((gains-1)[:,None]*vec[users]).sum(0)
    rmse=float(np.sqrt(bias@bias+rx**2*D/(2*L)))
    return dict(energy_j=float(d.energy_j),expected_energy_j=float((L/W*(d.analog_rms**2+d.digital_sqrt_power**2)).sum()),
        error_bound=float(d.error_bound),expected_rmse=rmse,rmse_over_budget=rmse/eps,
        peak_power_bound_w=float(np.max((d.analog_rms+d.digital_sqrt_power)**2)))

def main():
    start=time.time();source=ROOT/'results/overnight_energy_v2/retrieval_fourway_v2/fixed_updates.npz'
    z=np.load(source);l=z['learning'][:,:D].copy();u=z['unlearning'][:,:D].copy();channels=z['channels'].copy()
    # Select once by channel geometry, before outcomes; same head for every method.
    scores=[np.mean(np.log(np.delete(channels[:,v],v))) for v in range(len(channels))]
    head=int(np.argmax(scores));users=np.delete(np.arange(len(l)),head)
    rho=np.linalg.norm(l,axis=1)[:,None];eps=.2*float(rho.sum())
    config=dict(symbol_rate_hz=W,power_limit_w=P,available_energy_j=.5,coherence_time_s=.01,
        round_uplink_deadline_s=.01,max_ao_iterations=6,receive_initializations=2)
    gains_db=[-10,0,10];lengths=[128,160,256,512,2048,8192];ks=[8,32,128]
    proto=dict(scope='First 128 coordinates of frozen real VLM updates; one representative block, not an entire round or convergence experiment.',
        input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        head=head,head_selection='Maximum geometric mean normalized channel magnitude, same fixed head across methods.',
        users=len(l),active_requests=np.flatnonzero(np.linalg.norm(u,axis=1)>0).tolist(),remote_requests=int(np.count_nonzero(np.linalg.norm(u[users],axis=1))),
        coordinate_block=[0,128],channel_gain_db=gains_db,symbol_lengths=lengths,topk=ks,quantization_bits=8,index_bits_per_nonzero=7,
        epsilon=eps,config=config,ideal_SIC=True,
        noma='Existing robust full-overlap SOCP/AO, descending-channel SIC; minimum-power decoding. Failure means no feasible design found, not proven infeasibility.',
        oma='AirComp then continuous-bandwidth optimized OFDMA; analog time grid includes dimension minimum and endpoint. Grid optimum only, not full global optimality.',
        airair='One simultaneous aggregate of identical FL + compressed canonical FU. Cannot recover individual requests; reported separately, never assumed unsuccessful unlearning.',
        digital='Continuous OFDMA-only with common Q8 FL and identical compressed FU; quantization error checked against same epsilon.',
        costs='TX only. Expected hybrid energy also shown; robust waveform cross-term upper bound used for optimization/peak audit.',
        limitations=['Compression may change FU quality; its actual vector distortion is reported, not assumed free.',
        'No VLM accuracy/KL/convergence or deletion certificate measured.',
        'Ideal coding/CSI/SIC, static flat channels; 10 dB changes are hypothetical link conditions.',
        'Continuous OFDMA relaxation gives a stronger comparator than integer subcarriers.'])
    write('protocol.json',proto)
    # Meaningful independent checks of Shannon bandwidth allocation and codec.
    one=ofdma(np.array([128.]),np.array([2.]),256)
    assert np.isclose(one['power'][0],(np.sqrt(2)-1)/4,rtol=1e-7)
    two=ofdma(np.array([32.,32.]),np.ones(2),256)
    assert np.allclose(two['fractions'],[.5,.5],atol=1e-6)
    zero=ofdma(np.zeros(2),np.ones(2),128);assert zero['energy']==0
    qfull,bfull=encode(u,128);assert np.all(bfull[np.linalg.norm(u,axis=1)>0]==1056)
    write('checks.json',dict(passed=True,single_user_shannon=True,symmetric_bandwidth=True,zero_payload=True,full_codec_payload=True))
    rows=[];zeros=np.zeros_like(rho);analog_cache={};sum_cache={}
    def solve(h,r,b,L):
        return optimize_head(head,h,r,zeros,b[:,None],np.array([D]),np.array([float(L)]),eps,config,0.,'hybrid')
    for db in gains_db:
        h=channels*10**(db/20)
        for k in ks:
            q,bits=encode(u,k);mixed=l+q
            dist=float(np.linalg.norm(q[users]-u[users])/np.linalg.norm(u[users]))
            for L in lengths:
                row=dict(channel_gain_db=db,topk=k,symbols=L,deadline_ms=L/W*1000,
                    payload_bits_per_request=int(bits.max()),fu_codec_relative_l2_error=dist)
                dn=solve(h,rho,bits,L)
                row['noma']=compact(dn,l,h,head,L,eps) if dn else None
                if dn:
                    active=users[bits[users]>0];order=active[np.argsort(-h[active,head],kind='stable')]
                    a=dn.analog_rms[:,0];p=dn.digital_sqrt_power[:,0]**2
                    qrx=h[users,head]**2*p;ia=float((h[users,head]@a)**2);margin=[]
                    for j,v in enumerate(order):
                        ui=int(np.flatnonzero(users==v)[0]);later=[int(np.flatnonzero(users==w)[0]) for w in order[j+1:]]
                        margin.append(float(L*np.log2(1+qrx[ui]/(1+ia+qrx[later].sum()))-bits[v]))
                    assert min(margin,default=0)>=-1e-5
                    row['noma']['min_rate_margin_bits']=min(margin,default=0)
                key=(db,k,L)
                da=solve(h,np.linalg.norm(mixed,axis=1)[:,None],np.zeros(len(l)),L)
                row['airair']=compact(da,mixed,h,head,L,eps) if da else None
                best=None
                # Full bandwidth during each phase; digital users share optimized frequencies.
                grid=np.unique(np.r_[D,np.linspace(D,max(D,L-1),13),[.25*L,.5*L,.75*L]])
                for la in grid:
                    if la<D or la>=L:continue
                    dg=ofdma(bits[users],h[users,head],L-la)
                    if dg is None:continue
                    key=(db,round(float(la),8))
                    if key not in analog_cache:analog_cache[key]=solve(h,rho,np.zeros(len(l)),la)
                    ad=analog_cache[key]
                    if ad is None:continue
                    candidate=compact(ad,l,h,head,la,eps)
                    candidate.update(analog_symbols=float(la),digital_symbols=float(L-la),analog_energy_j=candidate['energy_j'],digital_energy_j=dg['energy'],bandwidth_fractions=dg['fractions'])
                    candidate['energy_j']+=dg['energy'];candidate['expected_energy_j']=candidate['energy_j']
                    candidate['peak_power_bound_w']=max(candidate['peak_power_bound_w'],float(dg['power'].max()))
                    if best is None or candidate['energy_j']<best['energy_j']:best=candidate
                row['air_ofdma']=best
                ql,bl=encode(l,128);de=ofdma((bl+bits)[users],h[users,head],L)
                qe=float(np.linalg.norm((ql[users]-l[users]).sum(0)))
                row['ofdma_only']=(dict(energy_j=de['energy'],expected_energy_j=de['energy'],expected_rmse=qe,rmse_over_budget=qe/eps) if de is not None and qe<=eps else None)
                if dn and best:row['noma_to_oma_ratio']=row['noma']['energy_j']/best['energy_j']
                rows.append(row);write('rows.json',rows)
                print(json.dumps({key:row.get(key) for key in ['channel_gain_db','topk','deadline_ms','noma_to_oma_ratio']},allow_nan=False),flush=True)
    both=[x for x in rows if x.get('noma_to_oma_ratio') is not None]
    summary=dict(cases=len(rows),runtime_s=time.time()-start,head=head,
        feasible={m:sum(x[m] is not None for x in rows) for m in ['noma','air_ofdma','airair','ofdma_only']},
        paired_cases=len(both),noma_energy_wins=sum(x['noma_to_oma_ratio']<.99 for x in both),
        noma_only_found=sum(x['noma'] is not None and x['air_ofdma'] is None for x in rows),
        noma_vs_oma_best=min(both,key=lambda x:x['noma_to_oma_ratio']) if both else None,
        caution='A deadline-feasibility advantage is not an energy win over a failed baseline. No learning/deletion quality claim.')
    write('summary.json',summary);print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
