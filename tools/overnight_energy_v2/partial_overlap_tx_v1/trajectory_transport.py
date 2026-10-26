"""User-requested partial overlap and uplink-only costs for learned trajectories."""
from pathlib import Path
import importlib.util
import time
import numpy as np
from phase_split_grid import make_inputs
from receiver_control import control, sensitivity_bound
from overlap_solver import optimize_partial

ROOT=Path(__file__).resolve().parents[3]

def load_file(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

legacy=load_file('frozen_screen_transport',ROOT/'tools/overnight_energy_v2/deletion_window_screen_v1/trajectory_transport.py')
oma_solve=legacy.oma_solve
frontier=load_file('frozen_strong_oma',ROOT/'tools/overnight_energy_v2/deadline_frontier_v1/run_frontier.py')

def communicate(learning,unlearning,channels,config,method,seed,gradient=None):
    start=time.perf_counter();c=dict(config,base_overhead_j=0.,head_overhead_j=0.)
    if method=='oma_only':
        return legacy.communicate(learning,unlearning,channels,c,method,seed,gradient)
    l,u,decoded,delta,payload,rho,dims,total=make_inputs(learning,unlearning,c)
    n,dimension=learning.shape;active=np.flatnonzero(np.any(unlearning!=0,axis=1))
    canonical=decoded.reshape(n,-1)[:,:dimension]
    aggregate=learning.sum(axis=0);target=aggregate+canonical.sum(axis=0)
    epsilon=max(c['absolute_tolerance_floor'],float(c['relative_tolerance']*np.linalg.norm(rho.sum(axis=0))))
    ctrl=len(active)>0 and method!='aircomp_only';lip=1.
    if ctrl:
        target,_=control(aggregate,canonical[active],gradient,1.)
        lip=float(sensitivity_bound(canonical[active],gradient))
    eps=epsilon/lip;la=np.floor(total*.125).astype(int);ld=total-la
    lb=l.copy();rb=rho.copy();bits=payload.copy()
    if method=='aircomp_only':
        lb=l+decoded;rb=np.linalg.norm(lb,axis=-1);bits=np.zeros_like(payload)
    analog_only=not np.any(bits>0)
    if analog_only:ld=la.copy()
    frame=total if ctrl and method=='aircomp_oma' else ld
    duration=float(frame.sum()/c['symbol_rate_hz'])
    assert duration<=c['round_uplink_deadline_s']+1e-9
    assert np.all(frame/c['symbol_rate_hz']<=c['coherence_time_s']+1e-9)
    best=None;diagnostics=[]
    for head in range(n):
        if ctrl and method=='aircomp_oma':
            d=frontier.strong_oma(head,channels,rb,bits,dims,total,eps,c)
            if d is not None:
                d=dict(d,tx_energy_bound_j=float(d['energy_j']),tx_energy_expected_j=float(d['energy_j']),analog_symbols=d['symbols'])
        else:
            d=optimize_partial(head,channels,rb,bits,dims,la,ld,eps,c,
                               split_grid=(1.,) if analog_only else (.025,.05,.1,1/7,.25))
        diagnostics.append(dict(head=head,status='feasible' if d is not None else 'no_feasible_design_found'))
        if d is not None and (best is None or d['tx_energy_bound_j']<best['tx_energy_bound_j']):best=d
    if best is None:
        return None,dict(status='no_feasible_design_found',head_diagnostics=diagnostics,uplink_budget_s=duration,
                         epsilon=epsilon,optimizer_wall_s=time.perf_counter()-start,required_requests=int(len(active)),individual_requests_delivered=0)
    head=best['head'];users=best['users'];received=lb[head].copy();rng=np.random.default_rng(seed)
    for m in range(len(dims)):
        gain=best['receiver'][m]*channels[users,head]*best['analog'][:,m]/np.where(rb[users,m]>0,rb[users,m],1)
        received[m]+=(gain[:,None]*lb[users,m]).sum(axis=0)
        received[m]+=rng.normal(0,best['receiver'][m]/np.sqrt(2*best['analog_symbols'][m]),received.shape[1])
    delivered=received.reshape(-1)[:dimension];info=None
    if ctrl:delivered,info=control(delivered,canonical[active],gradient,1.)
    # With no digital payload, unlearning is already included in lb for AirComp-only.
    stats=dict(status='feasible',energy_bound_j=best['tx_energy_bound_j'],expected_energy_j=best['tx_energy_expected_j'],
               uplink_s=duration,total_delay_s=duration,epsilon=epsilon,controller_lipschitz=lip,error_bound=best['error_bound'],
               head=int(head),application_error_bound=best['error_bound']*lip,
               realized_update_error=float(np.linalg.norm(delivered-target)),head_diagnostics=diagnostics,controller=info,
               optimizer_wall_s=time.perf_counter()-start,required_requests=int(len(active)),
               individual_requests_delivered=int(len(active)) if ctrl else 0,task_eligible=bool(not len(active) or ctrl),
               metric_scope='uplink transmit energy and occupied time only; reconstruction and computation excluded',
               analog_time_s=float(best['analog_symbols'].sum()/c['symbol_rate_hz']),
               payload_overlap_fraction=best.get('payload_overlap_fraction'),
               overlap_active=bool(ctrl and method!='aircomp_oma'))
    return delivered,stats
