"""Two separately recovered analog streams over a common fixed time budget.

Independent real Gaussian receiver noises combine in quadrature for the final
SUM. Unknown deterministic mismatch correlations use a triangle bound. This is
not a bound for the changed nonlinear v8 fusion. No digital payload or SIC.
"""
from dataclasses import dataclass
import warnings
import cvxpy as cp
import numpy as np


@dataclass
class Design:
    head: int
    users: np.ndarray
    analog_rms: np.ndarray  # stream,user,block
    receiver: np.ndarray  # stream,block; physical |c|*sigma*sqrt(N)
    rho: np.ndarray
    channels: np.ndarray
    symbols: np.ndarray
    dimensions: np.ndarray
    energy_j: float
    error_bound: float
    history: list


def allocate_symbols(total, dims, rho, head, fraction):
    if not 0 < fraction < 1:
        raise ValueError('A two-stream phase fraction must lie in (0,1).')
    users = np.delete(np.arange(rho.shape[1]), head)
    active = np.any(rho[:, users] > 0, axis=1)
    n = np.zeros((2, len(total)), dtype=int)
    both = active.all(axis=0)
    n[0, both] = np.floor(total[both] * fraction).astype(int)
    n[1, both] = total[both] - n[0, both]
    for s in [0, 1]:
        n[s, active[s] & ~active[1-s]] = total[active[s] & ~active[1-s]]
    if np.any((n > 0) & (n < dims[None, :])):
        raise ValueError('Each active isometric analog mapping requires N>=dimension.')
    assert np.all(n.sum(axis=0) <= total)
    return n


def bounds(a, s, h, rho, dims, symbols):
    mismatch = np.abs(s[:, None, :] * h[None, :, :] * a - rho).sum(axis=(0, 1))
    noise = np.sqrt(np.divide(dims[None, :], 2*symbols,
                             out=np.zeros_like(symbols, dtype=float), where=symbols>0))
    return mismatch + np.linalg.norm(s * noise, axis=0)


def optimize(head, channel_matrix, rho_all, dims, symbols, epsilon, config):
    users = np.delete(np.arange(len(channel_matrix)), head)
    h = np.broadcast_to(channel_matrix[users, head, None], (len(users),len(dims)))
    rho = rho_all[:, users]
    duration = symbols/config['symbol_rate_hz']
    overhead = np.full(len(channel_matrix),config['base_overhead_j'])
    overhead[head] += config['head_overhead_j']
    if epsilon <= 0 or np.any(overhead>config['available_energy_j']):return None
    if duration.sum()>config['round_uplink_deadline_s']*(1+1e-10):return None
    if np.any(duration>config['coherence_time_s']) or np.any((symbols>0)&(symbols<dims)):return None
    active = symbols>0
    if np.any((~active[:,None,:])&(rho>0)):return None
    noise = np.sqrt(np.divide(dims[None,:],2*symbols,out=np.zeros_like(duration),where=active))
    cap = np.sqrt(config['power_limit_w'])
    shape = (2*len(users),len(dims))
    flat_rho = rho.reshape(shape)
    flat_h = np.vstack([h,h])
    flat_duration = np.vstack([np.broadcast_to(duration[i],h.shape) for i in [0,1]])
    flat_zero = (flat_rho==0) | np.vstack([np.broadcast_to(~active[i],h.shape) for i in [0,1]])

    def energy(a):
        return (a*a*duration[:,None,:]).sum(axis=(0,2))

    def feasible(a,s):
        if not np.all(np.isfinite(a)) or not np.all(np.isfinite(s)):return False
        if np.min(a)<-1e-9 or np.min(s)<-1e-9:return False
        if np.max(a)>cap*(1+1e-6):return False
        if np.any(np.abs(a.reshape(shape)[flat_zero])>1e-7):return False
        if np.any(np.abs(s[~active])>1e-9):return False
        if np.any(energy(a)+overhead[users]>config['available_energy_j']*(1+1e-6)):return False
        return np.linalg.norm(bounds(a,s,h,rho,dims,symbols))<=epsilon*(1+1e-5)

    def error_expression(a,s,scaled=False):
        pieces=[]
        for m in range(len(dims)):
            terms=[]
            for stream in [0,1]:
                sl=s[stream,m]
                rows=slice(stream*len(users),(stream+1)*len(users))
                if scaled:
                    terms.append(cp.norm1(cp.multiply(h[:,m],a[rows,m])-rho[stream,:,m]/epsilon))
                else:
                    terms.append(cp.norm1(cp.multiply(sl*h[:,m],a[rows,m])-rho[stream,:,m])/epsilon)
            noises=cp.hstack([s[t,m]*noise[t,m] for t in [0,1]])
            pieces.append(sum(terms)+(cp.norm(noises) if scaled else cp.norm(noises)/epsilon))
        return cp.norm(cp.hstack(pieces))

    def solve(problem):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',UserWarning)
                problem.solve(solver='CLARABEL',max_iter=200)
        except cp.error.SolverError:return False
        return problem.status in ('optimal','optimal_inaccurate')

    def amplitude_step(receiver):
        a=cp.Variable(shape,nonneg=True)
        energies=cp.multiply(flat_duration,cp.square(a))
        peruser=cp.sum(energies[:len(users)]+energies[len(users):],axis=1)
        constraints=[a<=cap,cp.multiply(flat_zero,a)==0,
                     peruser+overhead[users]<=config['available_energy_j'],
                     error_expression(a,receiver)<=1]
        objective=cp.sum(energies)/max(duration.sum(),1e-12)
        if not solve(cp.Problem(cp.Minimize(objective),constraints)) or a.value is None:return None
        av=np.maximum(a.value,0).reshape(rho.shape)
        return av if feasible(av,receiver) else None

    def receive_step(a):
        s=cp.Variable((2,len(dims)),nonneg=True)
        constraints=[cp.multiply(~active,s)==0]
        # Here amplitudes are fixed, so the receiver error is convex.
        problem=cp.Problem(cp.Minimize(error_expression(cp.Constant(a.reshape(shape)),s)),constraints)
        if not solve(problem) or s.value is None:return None
        sv=np.maximum(s.value,0)
        return sv if feasible(a,sv) else None

    # Convex minimum-error initialization in scaled variables t=s*a/epsilon,
    # r=s/epsilon, with physical peak constraints. Battery is checked on recovery.
    t=cp.Variable(shape,nonneg=True);sc=cp.Variable((2,len(dims)),nonneg=True)
    sc_rows=cp.vstack([cp.vstack([sc[i,:]]*len(users)) for i in [0,1]])
    initial=cp.Problem(cp.Minimize(error_expression(t,sc,scaled=True)),
                       [t<=cap*sc_rows,cp.multiply(flat_zero,t)==0,cp.multiply(~active,sc)==0])
    starts=[]
    if solve(initial) and t.value is not None and sc.value is not None:
        sr=np.maximum(sc.value,0)*epsilon
        starts.append(sr)
    reference=np.max(np.divide(rho,h[None,:,:]*cap*.3,
                               out=np.zeros_like(rho),where=h[None,:,:]>0),axis=1)
    reference=np.where(active,np.maximum(reference,1e-12),0)
    starts += [reference*f for f in [.5,1,2,4]]
    best=None
    for s in starts:
        a=amplitude_step(s)
        if a is None:continue
        history=[float(energy(a).sum()+overhead.sum())]
        for _ in range(config.get('ao_iterations',12)):
            sn=receive_step(a)
            if sn is None:break
            an=amplitude_step(sn)
            if an is None:break
            value=float(energy(an).sum()+overhead.sum())
            if value>history[-1]+1e-9:break
            old=history[-1];a,s=an,sn;history.append(value)
            if old-value<=max(1e-10,old*1e-5):break
        value=float(energy(a).sum()+overhead.sum())
        if feasible(a,s) and (best is None or value<best.energy_j):
            best=Design(head,users,a,s,rho,h,symbols.copy(),dims.copy(),value,
                        float(np.linalg.norm(bounds(a,s,h,rho,dims,symbols))),history)
    return best


def exact_statistics(design, blocked, original_dimension):
    """Actual sum-error MSE and separate-stream errors, no full updates in solver."""
    a,s,h,rho,n=design.analog_rms,design.receiver,design.channels,design.rho,design.symbols
    coeff=np.divide(s[:,None,:]*h[None,:,:]*a,rho,out=np.zeros_like(rho),where=rho>0)
    mismatch=((coeff-1)[...,None]*blocked[:,design.users]).sum(axis=1)
    scalar_variance=np.divide(s*s,2*n,out=np.zeros_like(s),where=n>0)
    remaining=original_dimension;variances=[]
    for m in range(len(design.dimensions)):
        valid=min(int(design.dimensions[m]),remaining);remaining-=valid
        variances.append(valid*scalar_variance[:,m])
    noise=np.stack(variances,axis=1)
    mse=float(np.square(mismatch.sum(axis=0).reshape(-1)[:original_dimension]).sum()+noise.sum())
    stream_rmse=[float(np.sqrt(np.square(mismatch[t].reshape(-1)[:original_dimension]).sum()+noise[t].sum())) for t in [0,1]]
    return dict(analytic_sum_rmse=float(np.sqrt(mse)),stream_rms_errors=stream_rmse,
                no_digital_payload=True,random_realized_noise_not_sampled=True)
