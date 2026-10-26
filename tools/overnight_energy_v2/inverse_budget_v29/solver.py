"""Joint TX/RX SOCP for fixed head, codec, timing and block RMS budgets.

For zero SIC residual, minimum digital amplitude is kappa*||(1,sum H*a)||
in Hybrid and kappa in phase separation. With t=1/s the block bound is
sum|H*a-rho*t|+sqrt(d/(2L)) <= (r-delta_sum)*t.
Fixed r makes this convex. No claim of globally optimizing r or the head.
"""
import numpy as np
import cvxpy as cp


def refine(seed, payload, config):
    if float(seed['chi']) != 0.: raise ValueError('This isolated solver is for zero SIC residual only.')
    h=seed['normalized_channels'];rho=seed['rho'];delta=seed['quantization_bound'];na=seed['symbols'];nd=seed['digital_symbols']
    olda=seed['analog_rms'];olds=seed['receiver'];noise=np.sqrt(seed['dimensions']/(2*na))
    budget=np.abs(olds*h*olda-rho).sum(axis=0)+olds*noise+delta.sum(axis=0)
    coefficient=np.zeros_like(rho)
    for m in range(len(na)):
        order=np.flatnonzero(payload[:,m]>0);order=order[np.argsort(-h[order,m],kind='stable')]
        accumulator=0.
        for u in order[::-1]:
            gamma=np.expm1(np.log(2)*payload[u,m]/nd[m]);lam=gamma*(1+accumulator)
            coefficient[u,m]=np.sqrt(lam)/h[u,m];accumulator+=lam
    shape=rho.shape;a=cp.Variable(shape,nonneg=True);z=cp.Variable(len(na),nonneg=True)
    active=olds>1e-20;inverse_scale=np.divide(1.,olds,out=np.ones_like(olds),where=active)
    inverse=cp.multiply(inverse_scale,z);constraints=[]
    if np.any(active):
        mismatch=cp.sum(cp.abs(cp.multiply(h[:,active],a[:,active])-cp.multiply(rho[:,active],inverse[active])),axis=0)
        constraints.append(mismatch+noise[active] <= cp.multiply(budget[active]-delta[:,active].sum(axis=0),inverse[active]))
    if np.any(~active):constraints += [a[:,~active]==0,z[~active]==1]
    if np.any(rho==0):constraints.append(a[rho==0]==0)
    phase=str(seed['mode'])=='orthogonal'
    if phase:
        digital=cp.Constant(coefficient)
    elif np.any(coefficient>0):
        norm=cp.Variable(len(na),nonneg=True)
        constraints.append(cp.norm(cp.vstack([np.ones(len(na)),cp.sum(cp.multiply(h,a),axis=0)]),axis=0)<=norm)
        digital=cp.multiply(coefficient,norm)
    else:digital=cp.Constant(np.zeros(shape))
    da=na/config['symbol_rate_hz'];dd=nd/config['symbol_rate_hz']
    if phase:
        energy=cp.multiply(da,cp.square(a))+cp.multiply(dd,cp.square(digital))
        constraints += [a<=np.sqrt(config['power_limit_w']),digital<=np.sqrt(config['power_limit_w'])]
    else:
        energy=cp.multiply(da,cp.square(a+digital));constraints.append(a+digital<=np.sqrt(config['power_limit_w']))
    extra=seed.get('control_energy',np.zeros(shape[0]))
    constraints.append(cp.sum(energy,axis=1)+extra+config['base_overhead_j']<=config['available_energy_j'])
    a.value=olda;z.value=np.ones(len(na))
    problem=cp.Problem(cp.Minimize(1e6*cp.sum(energy)),constraints);assert problem.is_dcp()
    problem.solve(solver='CLARABEL',max_iter=300,tol_gap_abs=1e-8,tol_gap_rel=1e-8,tol_feas=1e-9)
    if problem.status not in ['optimal','optimal_inaccurate']:raise ValueError('SOCP '+str(problem.status))
    anew=np.maximum(a.value,0.);anew[rho==0]=0;anew[:,~active]=0
    t=inverse_scale*np.maximum(z.value,1e-300);snew=np.divide(1.,t,out=np.zeros_like(t),where=active)
    x=np.ones(len(na)) if phase else np.hypot(1.,np.sum(h*anew,axis=0))
    pnew=coefficient*x
    newbudget=np.abs(snew*h*anew-rho).sum(axis=0)+snew*noise+delta.sum(axis=0)
    assert np.all(newbudget<=budget+np.maximum(1e-12,np.abs(budget)*1e-5))
    result={k:np.asarray(v).copy() for k,v in seed.items()}
    result.update(analog_rms=anew,digital_sqrt_power=pnew,receiver=snew,error_bound=np.asarray(np.linalg.norm(newbudget)))
    return result,dict(status=problem.status,problem_is_dcp=problem.is_dcp(),solver_objective_j=problem.value/1e6,
        original_block_budget=budget.tolist(),refined_block_error_bound=newbudget.tolist(),iterations=problem.solver_stats.num_iters,
        limitation='Global convex inner problem only for fixed head/codec/timing and these block budgets. Outer global RMS allocation and head are not optimized globally; numerical designs require independent checks.')
