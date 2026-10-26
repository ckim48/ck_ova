"""Five labels, four distinct controlled arms; no invented proposed advantage.

Delay is minimized on a declared common frame grid under a network TX cap.
The proposed and conventional joint AirComp definitions are presently identical.
"""
from pathlib import Path
import os,sys,time,concurrent.futures,multiprocessing
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tools/overnight_energy_v2/adaptive_ofdma_v2'))
import adaptive_radio as old
import numpy as np
import cvxpy as cp
from types import SimpleNamespace
COUNTS=tuple(range(1,17))+(20,24,32,40,48,56,64)
SPLITS=(.1,.25,.5,.75,.9)
CAP=.05
POOL=None

def close():
    global POOL
    if POOL is not None:POOL.shutdown(wait=True);POOL=None

def exact_aligned(head,channels,norms,dims,counts,noise_budget,c,digital_e=None):
    """Convex minimum-energy channel inversion, jointly across all analog phases.

    x=L/(W*r^2), E_i=sum(rho_i^2/h_i^2*x),
    variance=sum(d/(2*W*x)); independent zero-mean noise from separate phases.
    Optimize scaled x to keep the conic problem well conditioned.
    """
    users=np.delete(np.arange(len(channels)),head);h=channels[users,head,None]
    W=c['symbol_rate_hz'];M=len(dims);coeff=np.concatenate([v[users]**2/h**2 for v in norms],axis=1)
    lengths=np.concatenate([n*dims for n in counts]);dd=np.tile(dims,len(counts));v=dd/(2*W)
    active=np.any(coeff>0,axis=0);di=np.zeros(len(users)) if digital_e is None else digital_e
    if noise_budget<=0 or np.any(di>c['available_energy_j']):return None
    scale=max(float(np.sqrt(coeff[:,active].sum(0)*v[active]).sum()/noise_budget**2),1e-30)
    x0=np.ones(len(v));x0[active]=scale*np.sqrt(v[active]/np.maximum(coeff[:,active].sum(0),1e-30))
    x=cp.Variable(int(active.sum()),pos=True);ec=coeff[:,active]*x0[active]
    var=v[active]/x0[active]/noise_budget**2
    energy=ec@x
    peak=c['power_limit_w']*lengths[active]/W
    problem=cp.Problem(cp.Minimize(cp.sum(energy)*1000),[
        cp.sum(cp.multiply(var,cp.inv_pos(x)))<=1,
        cp.multiply(ec,cp.reshape(x,(1,int(active.sum())),order='C'))<=peak,
        energy+di<=c['available_energy_j'],cp.sum(energy+di)<=CAP])
    try:problem.solve(solver='CLARABEL',max_iter=150,tol_gap_abs=1e-9,tol_feas=1e-9)
    except cp.error.SolverError:return None
    if problem.status not in ['optimal','optimal_inaccurate'] or x.value is None:return None
    xx=np.zeros(len(v));xx[active]=x0[active]*np.asarray(x.value)
    if np.any(xx[active]<=0):return None
    rx=np.zeros(len(v));rx[active]=np.sqrt(lengths[active]/(W*xx[active]))
    amplitudes=np.zeros_like(coeff);amplitudes[:,active]=np.sqrt(coeff[:,active])/rx[active]
    eu=(lengths/W*amplitudes**2).sum(1);variance=float((v[active]/xx[active]).sum())
    if amplitudes.max()**2>c['power_limit_w']*(1+2e-6) or (eu+di).max()>c['available_energy_j']*(1+2e-6) or (eu+di).sum()>CAP*(1+2e-6) or variance>noise_budget**2*(1+2e-5):return None
    result=[]
    for j,n in enumerate(counts):
        sl=slice(j*M,(j+1)*M)
        result.append(SimpleNamespace(analog_rms=amplitudes[:,sl],receiver=rx[sl],
            energy_j=float((n*dims/W*amplitudes[:,sl]**2).sum()),
            error_bound=float(np.sqrt((v[sl][active[sl]]/xx[sl][active[sl]]).sum()))))
    return result

def design(head,channels,l,u,q,delta,payload,rho,dims,eps,c,mode,frame_count):
    users=np.delete(np.arange(len(channels)),head);h=channels[users,head]
    W=c['symbol_rate_hz'];has_u=bool(np.any(u[users]))
    # A head's own FU requires no uplink, but is still applied at recovery.
    if mode=='conventional_joint':mode='joint'
    if not has_u:mode='joint'
    norms={'joint':np.linalg.norm(l+u,axis=-1),'fl':rho,'fu':np.linalg.norm(u,axis=-1)}
    def pack(n,parts,digital=None,quant=0.,detail=None):
        eu=np.zeros(len(users));var=0.
        for name,a,nphase in parts:
            eu+=(nphase*dims/W*a.analog_rms**2).sum(1);var+=a.error_bound**2
        err=float(np.sqrt(var+quant**2))
        if digital is not None:eu+=digital['energy_by_user']
        if eu.sum()>CAP*(1+1e-7) or eu.max()>c['available_energy_j']*(1+1e-7) or err>eps*(1+3e-5):return None
        return dict(head=head,users=users,parts=parts,digital=digital,energy_j=float(eu.sum()),
            max_user_energy_j=float(eu.max()),error_bound=float(err),quantization_error_bound=float(quant),
            frame_count=n,uplink_s=float(n*dims.sum()/W),detail=detail or {})
    qb=float(np.linalg.norm(delta[users].sum(0))) if mode=='ofdma' else 0.
    if qb>=eps:return None
    for n in [frame_count]:
        if n*dims.sum()/W>c['round_uplink_deadline_s']+1e-12 or np.max(n*dims/W)>c['coherence_time_s']:continue
        candidates=[]
        if mode=='joint':
            a=exact_aligned(head,channels,[norms['joint']],dims,[n],eps,c)
            if a is not None:candidates.append(pack(n,[('joint',a[0],n)]))
        else:
            for na in range(1,n):
                nu=n-na
                if mode=='separate':
                    ab=exact_aligned(head,channels,[norms['fl'],norms['fu']],dims,[na,nu],eps,c)
                    if ab is not None:candidates.append(pack(n,[('fl',ab[0],na),('fu',ab[1],nu)],detail=dict(error_allocation='joint_convex_variance')))
                elif mode=='ofdma':
                    dg=old.ofdma.digital(payload[users],h,nu*dims,c)
                    if dg is None or dg['energy_by_user'].sum()>CAP:continue
                    a=exact_aligned(head,channels,[norms['fl']],dims,[na],np.sqrt(eps**2-qb**2),c,dg['energy_by_user'])
                    if a is None:continue
                    candidates.append(pack(n,[('fl',a[0],na)],dg,qb,dict(digital_count=nu,access='integer_subcarrier_OFDMA')))
                else:raise ValueError(mode)
        good=[x for x in candidates if x is not None]
        if good:return min(good,key=lambda x:x['energy_j'])
    return None

def worker(args):
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):return design(*args)

def communicate(learning,unlearning,channels,c,mode,seed):
    global POOL
    start=time.time();d=learning.shape[1]
    c=dict(c,quantization_bits=6,ofdma_subcarriers=128)
    l,u,q,delta,payload,rho,dims,total=old.make_inputs(learning,unlearning,c)
    eps=max(c['absolute_tolerance_floor'],c['relative_tolerance']*float(np.linalg.norm(rho.sum(0))))
    if POOL is None:POOL=concurrent.futures.ProcessPoolExecutor(max_workers=8,mp_context=multiprocessing.get_context('spawn'))
    good=[];ds=[]
    for count in COUNTS:
        futures=[POOL.submit(worker,(h,channels,l,u,q,delta,payload,rho,dims,eps,c,mode,count)) for h in range(len(channels))]
        ds=[f.result() for f in futures];good=[x for x in ds if x is not None]
        if good:break
    if not good:return None,dict(status='no_feasible_design_found',mode=mode,epsilon=eps,network_tx_cap_j=CAP,optimizer_wall_s=time.time()-start)
    best=min(good,key=lambda x:(x['uplink_s'],x['energy_j']));head=best['head'];users=best['users']
    rng=np.random.default_rng(seed);mean=l[head]+u[head];variance=0.;norms={'joint':np.linalg.norm(l+u,axis=-1),'fl':rho,'fu':np.linalg.norm(u,axis=-1)}
    parts={'joint':l+u,'fl':l,'fu':u};received=mean.copy();maxpower=0.
    phase_meta=[]
    for name,a,n in best['parts']:
        gain=a.receiver*channels[users,head,None]*a.analog_rms/np.where(norms[name][users]>0,norms[name][users],1.)
        mu=(gain[:,:,None]*parts[name][users]).sum(0);mean+=mu;received+=mu
        for m in range(len(dims)):
            sd=a.receiver[m]/np.sqrt(2*n*dims[m]);received[m]+=rng.normal(0,sd,received.shape[1]);variance+=dims[m]*sd**2
        maxpower=max(maxpower,float((a.analog_rms**2).max()))
        phase_meta.append(dict(stream=name,count=n,energy_j=a.energy_j,error_bound=a.error_bound))
    dg=best['digital']
    if dg is not None:
        received+=q[users].sum(0);mean+=q[users].sum(0);maxpower=max(maxpower,dg['max_power_w'])
    # A head with no remote FU uses the joint path, including its local FU once.
    raw=(l+u).sum(0);rmse=float(np.sqrt(np.square(mean-raw).sum()+variance))
    assert rmse<=best['error_bound']*(1+1e-4)+1e-10
    assert maxpower<=c['power_limit_w']*(1+1e-6)
    meta=dict(status='feasible',physical_mode=mode,head=int(head),head_candidates=len(ds),head_feasible=len(good),
        energy_bound_j=best['energy_j'],expected_energy_j=best['energy_j'],uplink_s=best['uplink_s'],
        frame_count=best['frame_count'],network_tx_cap_j=CAP,max_user_energy_j=best['max_user_energy_j'],max_power_w=maxpower,
        error_bound=best['error_bound'],epsilon=eps,conditional_rmse=rmse,quantization_error_bound=best['quantization_error_bound'],
        error_target='raw unquantized FL+FU sum',fu_bits=6 if dg is not None else None,phases=phase_meta,
        optimizer_wall_s=time.time()-start,requests_in_update=int(np.any(unlearning,axis=1).sum()),
        realized_error_to_raw=float(np.linalg.norm(received-raw)),**best['detail'])
    if dg is not None:meta.update(tones=128,subcarrier_counts=dg['subcarrier_counts'].tolist(),minimum_rate_margin_bits=dg['minimum_rate_margin'])
    vector=received.reshape(-1)[:d];assert np.isfinite(vector).all()
    return vector,meta
