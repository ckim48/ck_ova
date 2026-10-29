"""I/Q aligned AirComp, with the manuscript's three-block BCD and SOCP.

Channels are noise-normalized, as in the existing simulator (sigma^2=1).
Curvature inputs are empirical training-only surrogates, not certified bounds.
"""
import time
import numpy as np
import cvxpy as cp

COUNTS=tuple(range(1,17))+(20,24,32,40,48,56,64)
CAP=.1
PA=.35
HOVER=10*(79.86+88.63)


def inputs(l,u,c,ell):
    tasks=np.stack([l,u]).reshape(2,len(l),-1,c['block_dimension'])
    rho=np.linalg.norm(tasks,axis=-1)
    bounds=np.linalg.norm(rho.sum(1),axis=1)
    eps=np.maximum(c['absolute_tolerance_floor'],c['relative_tolerance']*bounds)
    eps[bounds==0]=0
    weights=np.asarray(ell,float)
    weights=weights/np.maximum(weights.mean(1,keepdims=True),1e-30)
    # Unspecified manuscript delta: half the distortion of uniform C8 allocation.
    # Pin before observing outcomes. Normalized 2*delta = .5*sum(epsilon^2).
    targets=np.full(len(weights),.5*float(np.sum(eps**2)))
    return rho,eps,weights,targets


def coefficients(head,channels,rho,eps,c):
    a,m=rho.shape[0],rho.shape[2];dm=c['block_dimension'];W=c['symbol_rate_hz']
    remote=np.arange(len(channels))!=head
    h=np.abs(channels[:,head]);h[head]=1
    if np.any((h[remote]==0)&np.any(rho[:,remote]>0,axis=(0,2))):return None
    A=rho**2/np.maximum(h[None,:,None]**2,1e-300);A[:,head]=0
    active=A.sum(1)>0
    # x = x_scale*y; y is well scaled near one for a uniform MSE allocation.
    x_scale=np.ones((a,m))
    for t in range(a):
        if eps[t]>0:x_scale[t]=dm/(2*W)*max(1,int(active[t].sum()))/eps[t]**2
    E=A*x_scale[:,None,:]
    V=np.where(active,dm/(2*W)/x_scale,0.)
    return E,V,active,x_scale


def measure(head,n,y,channels,rho,eps,c,weights,targets):
    info=coefficients(head,channels,rho,eps,c)
    if info is None:return None
    E,V,active,scale=info
    if np.any(y[active]<=0):return None
    energies=(E*y[:,None,:]).sum(0)
    variances=np.divide(V,y,out=np.zeros_like(V),where=active)
    mse=variances.sum(1);dist=weights@variances.sum(0)
    powers=energies*c['symbol_rate_hz']/(n*c['block_dimension'])
    total=energies.sum();user=energies.sum(1)
    good=(np.all(mse<=eps**2*(1+2e-6)+1e-30) and np.all(dist<=targets*(1+2e-6)+1e-30)
          and powers.max(initial=0)<=c['power_limit_w']*(1+1e-6)
          and user.max(initial=0)<=c['available_energy_j']*(1+1e-6) and total<=CAP*(1+1e-6))
    if not good:return None
    d=rho.shape[2]*c['block_dimension'];up=n*d/c['symbol_rate_hz'] if active.any() else 0.
    return dict(head=int(head),n=int(n),y=y.copy(),mse=mse,distortion=dist,
                energy=energies,power=powers,variance=variances,active=active,x=scale*y,
                objective=float(HOVER*up+total/PA),uplink_s=up)


def transceiver(head,n,channels,rho,eps,c,weights,targets):
    info=coefficients(head,channels,rho,eps,c)
    if info is None:return None
    E,V,active,scale=info
    ids=np.flatnonzero(active.ravel());m=rho.shape[2];U=len(channels)
    if not len(ids):return measure(head,n,np.ones_like(V),channels,rho,eps,c,weights,targets)
    y=cp.Variable(len(ids),pos=True);z=cp.Variable(len(ids),nonneg=True)
    # Rotated-cone representation y*z >= 1; z upper bounds 1/y.
    constraints=[cp.SOC(y+z,cp.vstack([2*np.ones(len(ids)),y-z]),axis=0)]
    Ef=E.transpose(1,0,2).reshape(U,-1)[:,ids]
    Vf=V.ravel()[ids]
    for a in range(2):
        mask=ids//m==a
        if mask.any():constraints.append((Vf[mask]/eps[a]**2)@z[mask]<=1.)
    for w,target in zip(weights,targets):
        if target>0:constraints.append((w[ids%m]*Vf/target)@z<=1.)
    # Each block's power includes BOTH simultaneous I/Q streams.
    # Sparse selector maps active task-block columns to physical blocks.
    import scipy.sparse as sp
    selector=sp.coo_matrix((np.ones(len(ids)),(np.arange(len(ids)),ids%m)),shape=(len(ids),m)).tocsr()
    block_energy=cp.multiply(Ef,y[None,:])@selector
    peak=n*c['block_dimension']*c['power_limit_w']/c['symbol_rate_hz']
    constraints += [block_energy/peak<=1, (Ef@y)/c['available_energy_j']<=1,cp.sum(Ef@y)/CAP<=1]
    cost=Ef.sum(0)/PA;normal=max(float(cost.sum()),1e-30)
    problem=cp.Problem(cp.Minimize(cost@y/normal),constraints)
    try:problem.solve(solver='CLARABEL',max_iter=150,tol_gap_abs=1e-9,tol_gap_rel=1e-9,tol_feas=1e-9)
    except cp.error.SolverError:return None
    if y.value is None or problem.status not in ['optimal','optimal_inaccurate']:return None
    value=np.ones_like(V);value.ravel()[ids]=np.asarray(y.value)
    return measure(head,n,value,channels,rho,eps,c,weights,targets)


def design(l,u,channels,c,ell):
    rho,eps,weights,targets=inputs(l,u,c,ell)
    ns=[n for n in COUNTS if n*l.shape[1]/c['symbol_rate_hz']<=c['round_uplink_deadline_s']+1e-12
        and n*c['block_dimension']/c['symbol_rate_hz']<=c['coherence_time_s']+1e-12]
    if not ns:return None,dict(reason='No admissible symbol expansion')
    # Deterministic channel-inversion energy proxy orders feasible initialization.
    score=[]
    for head in range(len(channels)):
        E,V,active,_=coefficients(head,channels,rho,eps,c)
        score.append(float(E.sum()))
    solves=0;best=None
    for head in np.argsort(score,kind='stable'):
        best=transceiver(int(head),max(ns),channels,rho,eps,c,weights,targets);solves+=1
        if best is not None:break
    if best is None:return None,dict(reason='No feasible initialization at maximum expansion',solves=solves)
    history=[best['objective']]
    for _ in range(c['max_ao_iterations']):
        old=best
        new=transceiver(best['head'],best['n'],channels,rho,eps,c,weights,targets);solves+=1
        if new is not None and new['objective']<=best['objective']*(1+1e-9):best=new
        for n in ns:
            candidate=measure(best['head'],n,best['y'],channels,rho,eps,c,weights,targets)
            if candidate is not None:
                best=candidate;break
        # Keep x fixed, not scaled y: head-dependent active sets change x_scale.
        fixed_x=best['x'].copy()
        for head in range(len(channels)):
            *_,scale=coefficients(head,channels,rho,eps,c)
            candidate=measure(head,best['n'],fixed_x/scale,channels,rho,eps,c,weights,targets)
            if candidate is not None and candidate['objective']<best['objective']-1e-12:best=candidate
        history.append(best['objective'])
        assert history[-1]<=history[-2]*(1+1e-8)
        if abs(old['objective']-best['objective'])<=1e-8*max(old['objective'],1e-12):break
    return best,dict(solves=solves,BCD_objectives=history,component_epsilon=eps.tolist(),
                     normalized_curvature=weights.tolist(),normalized_distortion_limits=targets.tolist())


def analog(l,u,channels,c,seed,ell):
    start=time.time();best,audit=design(l,u,channels,c,ell)
    if best is None:return None,dict(status='no_feasible_design_found',**audit,optimizer_wall_s=time.time()-start)
    rng=np.random.default_rng(seed);dm=c['block_dimension'];raw=(l+u).sum(0)
    noise=rng.normal(size=(2,len(raw)//dm,dm))*np.sqrt(best['variance'][:,:,None]/dm)
    received=raw+noise.sum(0).reshape(-1)
    mse=best['mse'];energy=best['energy'];power=best['power'];eps=np.asarray(audit['component_epsilon'])
    meta=dict(status='feasible',physical_mode='manuscript_parallel_IQ',head=best['head'],
        frame_count=best['n'],uplink_s=best['uplink_s'],energy_bound_j=float(energy.sum()),expected_energy_j=float(energy.sum()),
        max_power_w=float(power.max()),max_user_energy_j=float(energy.sum(1).max()),network_tx_cap_j=CAP,
        conditional_rmse=float(np.sqrt(mse.sum())),epsilon=float(np.linalg.norm(eps)),
        realized_error_to_raw=float(np.linalg.norm(received-raw)),component_mse=mse.tolist(),
        component_realized_error=np.linalg.norm(noise.reshape(2,-1),axis=1).tolist(),
        distortion=best['distortion'].tolist(),block_variance=best['variance'].tolist(),
        user_block_power_w=power.tolist(),user_RF_energy_j=energy.sum(1).tolist(),
        transceiver_x=best['x'].tolist(),exact_alignment=True,optimizer_wall_s=time.time()-start,
        coefficient_convention='Noise-normalized channel; c=1/sqrt(W*x); b=sqrt(W*x)/h; orthonormal repetition; independent I/Q noise.',
        **audit)
    return received,meta
