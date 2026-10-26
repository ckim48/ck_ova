"""Real-update I/Q aggregation; shared RF power, explicit real dimensions.

Channels are noise-normalized magnitudes from the frozen experiment. Perfect
phase inversion is assumed. All arms may pack two real coordinates per complex
symbol. No digital/NOMA/SIC or extra antennas are introduced.
"""
import os
for _k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[_k]='1'
import time
import numpy as np
import cvxpy as cp

COUNTS=tuple(range(1,33))+(40,48,64,80,96,112,128)
CAP=.05

def controller(l,u,project):
    l=np.asarray(l,dtype=float);u=np.asarray(u,dtype=float)
    ll=float(l@l);uu=float(u@u);dot=float(l@u)
    coefficient=min(0.,dot)/uu if project and uu>1e-24 else 0.
    adjusted=l-coefficient*u
    return adjusted+u,dict(project=bool(project),conflict=bool(dot<0 and ll>1e-24 and uu>1e-24),
        cosine=dot/np.sqrt(ll*uu) if ll*uu>1e-48 else None,
        FL_norm=np.sqrt(ll),FU_norm=np.sqrt(uu),projection_coefficient=coefficient,
        removed_FL_norm=float(np.linalg.norm(adjusted-l)),
        adjusted_FL_dot_FU=float(adjusted@u),sum_norm=float(np.linalg.norm(l+u)))

def _blocks(a,c):
    d=c['block_dimension'];assert d%2==0
    pad=(-a.shape[1])%d
    return np.pad(a,((0,0),(0,pad))).reshape(a.shape[0],-1,d)

def _solve(head,h,norms,dims,lengths,budgets,c,simultaneous):
    """x_b=1/(W*c_b^2). E_ib=||v_ib||^2*x_b/(W-normalized h_i^2).

    Here h is already normalized by noise std, so E=norm^2/h^2*x;
    noise variance=d/(2*W*x). I and Q powers ADD in each symbol interval.
    """
    users=np.delete(np.arange(len(h)),head);W=c['symbol_rate_hz'];M=len(dims)
    hh=h[users,head,None]
    if np.any(hh<=0):return None
    coeff=np.stack([n[users]**2/hh**2 for n in norms]) # branch,user,block
    branches=len(norms);active=np.any(coeff>0,axis=1)
    ids=np.flatnonzero(active.ravel())
    if not len(ids):return dict(head=head,users=users,c=np.zeros((branches,M)),
        energy_by_user=np.zeros(len(users)),variance=np.zeros(branches),max_power=0.)
    v=np.broadcast_to(dims/(2*W),(branches,M))
    x0=np.ones((branches,M))
    for b in range(branches):
        a=active[b]
        if not a.any():continue
        if budgets[b]<=0:return None
        total=coeff[b].sum(0)
        scale=max(float(np.sqrt(total[a]*v[b,a]).sum()/budgets[b]**2),1e-30)
        x0[b,a]=scale*np.sqrt(v[b,a]/np.maximum(total[a],1e-300))
    var=cp.Variable(len(ids),pos=True)
    # Linear selector avoids unconstrained inactive variables and permits silent blocks.
    select=np.zeros((branches*M,len(ids)));select[ids,np.arange(len(ids))]=1.
    xx=cp.reshape(select@cp.multiply(x0.ravel()[ids],var),(branches,M),order='C')
    energy=[cp.multiply(coeff[b],cp.reshape(xx[b],(1,M),order='C')) for b in range(branches)]
    eu=sum(cp.sum(e,axis=1) for e in energy)
    constraints=[eu<=c['available_energy_j'],cp.sum(eu)<=CAP]
    for b in range(branches):
        own=(ids//M)==b
        if own.any():
            constraints.append(cp.sum(cp.multiply((v.ravel()[ids]/x0.ravel()[ids])[own]/budgets[b]**2,cp.inv_pos(var[own])))<=1.)
    if simultaneous:
        constraints.append(sum(energy)<=c['power_limit_w']*lengths[0][None,:]/W)
    else:
        for b in range(branches):constraints.append(energy[b]<=c['power_limit_w']*lengths[b][None,:]/W)
    problem=cp.Problem(cp.Minimize(1000*cp.sum(eu)),constraints)
    try:problem.solve(solver='CLARABEL',max_iter=150,tol_gap_abs=1e-9,tol_feas=1e-9)
    except cp.error.SolverError:return None
    if problem.status not in ['optimal','optimal_inaccurate'] or var.value is None:return None
    x=np.zeros((branches,M));x.ravel()[ids]=x0.ravel()[ids]*np.asarray(var.value)
    if np.any(x[active]<=0):return None
    e=coeff*x[:,None,:];user_e=e.sum((0,2))
    vv=np.array([(v[b,active[b]]/x[b,active[b]]).sum() for b in range(branches)])
    powers=e.sum(0)*W/lengths[0] if simultaneous else np.stack([e[b]*W/lengths[b] for b in range(branches)])
    if user_e.sum()>CAP*(1+2e-6) or user_e.max()>c['available_energy_j']*(1+2e-6):return None
    if powers.max()>c['power_limit_w']*(1+2e-6) or np.any(vv>np.square(budgets)*(1+2e-5)+1e-20):return None
    scale=np.zeros_like(x);scale[active]=np.sqrt(1/(W*x[active]))
    return dict(head=head,users=users,c=scale,energy_by_user=user_e,variance=vv,max_power=float(powers.max()))

def communicate(l,u,h,c,mode,project,seed,return_design=False):
    start=time.time();d=l.shape[1];lb,ub=_blocks(l,c),_blocks(u,c)
    dims=np.full(lb.shape[1],lb.shape[2]);W=c['symbol_rate_hz']
    nl,nu=np.linalg.norm(lb,axis=-1),np.linalg.norm(ub,axis=-1)
    el=max(c['absolute_tolerance_floor'],c['relative_tolerance']*np.linalg.norm(nl.sum(0)))
    eu=max(c['absolute_tolerance_floor'],c['relative_tolerance']*np.linalg.norm(nu.sum(0)))
    total_budget=float(np.hypot(el if np.any(l) else 0.,eu if np.any(u) else 0.))
    clean,truth=controller(l.sum(0),u.sum(0),project)
    if not np.any(l) and not np.any(u):
        meta=dict(status='feasible',access='silent',energy_bound_j=0.,uplink_s=0.,conditional_rmse=0.,epsilon=total_budget,
                  max_power_w=0.,diagnostic_true=truth,diagnostic_received=truth)
        return (clean,meta,None) if return_design else (clean,meta)
    if mode=='packed_sum':
        branches=[lb+ub];norms=[np.linalg.norm(branches[0],axis=-1)];budgets=[total_budget];names=['sum']
    else:
        branches=[];norms=[];budgets=[];names=[]
        for name,b,n,budget in [('FL',lb,nl,el),('FU',ub,nu,eu)]:
            if np.any(b):branches.append(b);norms.append(n);budgets.append(budget);names.append(name)
    best=None;tried=0
    for count in COUNTS:
        if count<2 and len(branches)==2:continue
        duration=float(count*(dims/2).sum()/W)
        if duration>c['round_uplink_deadline_s']+1e-12:continue
        layouts=([(count,)] if len(branches)==1 else
                 [(count,count)] if mode=='iq' else [(a,count-a) for a in range(1,count)])
        candidates=[]
        for layout in layouts:
            lengths=[n*dims/2 for n in layout]
            # Two real aggregates require d complex symbols; one aggregate needs d/2.
            if mode=='iq' and len(branches)==2 and any(np.any(t<dims) for t in lengths):continue
            if any(np.max(t/W)>c['coherence_time_s']+1e-12 for t in lengths):continue
            for head in range(len(h)):
                z=_solve(head,h,norms,dims,lengths,budgets,c,mode=='iq' and len(branches)==2);tried+=1
                if z is not None:
                    z.update(count=count,lengths=lengths,names=names,duration=duration)
                    candidates.append(z)
        if candidates:
            best=min(candidates,key=lambda z:float(z['energy_by_user'].sum()));break
    if best is None:
        meta=dict(status='no_feasible_design_found',access=mode,epsilon=total_budget,optimizer_wall_s=time.time()-start)
        return (None,meta,None) if return_design else (None,meta)
    rng=np.random.default_rng(seed);est={};noises=[]
    for b,name in enumerate(names):
        mean=branches[b].sum(0)
        noise=rng.normal(size=mean.shape)*best['c'][b,:,None]/np.sqrt(2.)
        est[name]=(mean+noise).ravel()[:d];noises.append(noise)
    if mode=='packed_sum':vector=est['sum'];received=None
    else:vector,received=controller(est.get('FL',np.zeros(d)),est.get('FU',np.zeros(d)),project)
    var=float(best['variance'].sum());energy=float(best['energy_by_user'].sum())
    meta=dict(status='feasible',access=mode,head=int(best['head']),head_candidates=len(h),designs_checked=tried,
        complex_symbols_per_base_block=[int(a[0]) for a in best['lengths']],half_frame_count=best['count'],
        simultaneous_branches=bool(mode=='iq' and len(branches)==2),real_coordinate_packing=True,
        uplink_s=best['duration'],energy_bound_j=energy,expected_energy_j=energy,
        max_user_energy_j=float(best['energy_by_user'].max()),max_power_w=best['max_power'],network_tx_cap_j=CAP,
        conditional_rmse=np.sqrt(var),epsilon=total_budget,branch_budgets=dict(zip(names,map(float,budgets))),
        branch_noise_variance=dict(zip(names,map(float,best['variance']))),
        error_target='Before head control: stacked FL/FU aggregates, or raw aggregate for packed_sum. No claim this bounds nonlinear post-controller error.',
        diagnostic_true=truth,diagnostic_received=received,head_control='FU-priority halfspace projection' if project else 'plain sum',
        realized_postcontrol_error=float(np.linalg.norm(vector-clean)),
        ideal_control_change=float(np.linalg.norm(clean-(l+u).sum(0))),
        optimizer_wall_s=time.time()-start,phase_error_radians=0.,
        hypothesis='First-order heuristic for the composite FU update, not a deletion guarantee or per-request guarantee.')
    assert np.isfinite(vector).all()
    return (vector,meta,best) if return_design else (vector,meta)
