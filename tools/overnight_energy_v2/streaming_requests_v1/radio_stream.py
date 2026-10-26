"""Common component-noise budget; pure-FU OFDMA has no empty analog phase."""
from pathlib import Path
import sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'parallel_aircomp_v1'))
import radio_parallel as base
import numpy as np

def norm_bound(v,c):
    d=c['block_dimension'];p=np.pad(v,((0,0),(0,(-v.shape[1])%d)))
    return float(np.linalg.norm(np.linalg.norm(p.reshape(len(v),-1,d),axis=-1).sum(0)))

def silence(v,c):
    return np.zeros(v.shape[1]),dict(status='feasible',access='zero_update_silence',
        energy_bound_j=0.,expected_energy_j=0.,uplink_s=0.,conditional_rmse=0.,
        epsilon=c['absolute_tolerance_floor'],max_power_w=0.,max_user_energy_j=0.,head=None)

def analog(l,u,channels,c,seed):
    if not np.any(l) and not np.any(u):return silence(l,c)
    active=np.flatnonzero(np.any(l+u,axis=1))
    if len(active)<=1:
        _,m=silence(l,c);m.update(access='local_aggregate_no_uplink',head=int(active[0]) if len(active) else None)
        return (l+u).sum(0),m
    nl,nu=norm_bound(l,c),norm_bound(u,c)
    eps=max(c['absolute_tolerance_floor'],c['relative_tolerance']*np.hypot(nl,nu))
    if not np.any(l):l,u=u,np.zeros_like(u);denom=nu
    else:denom=nl
    config=dict(c,relative_tolerance=eps/denom)
    v,m=base.communicate(l,u,channels,config,'joint',seed)
    m['component_budget_rule']='epsilon^2 = (rel*FL_blocknorm_bound)^2 + (rel*FU_blocknorm_bound)^2, with numerical floor'
    m['FL_norm_bound']=nl;m['FU_norm_bound']=nu
    return v,m

def digital_fu(u,channels,c):
    if not np.any(u):return silence(u,c)
    active=np.flatnonzero(np.any(u,axis=1))
    if len(active)==1:
        _,m=silence(u,c);m.update(access='head_local_no_uplink',head=int(active[0]))
        return u.sum(0),m
    start=time.time();config=dict(c,quantization_bits=6,ofdma_subcarriers=128)
    _,blocks,q,delta,payload,_,dims,_=base.old.make_inputs(np.zeros_like(u),u,config)
    eps=max(c['absolute_tolerance_floor'],c['relative_tolerance']*norm_bound(u,c))
    best=None
    for n in base.COUNTS:
        if np.max(n*dims/config['symbol_rate_hz'])>config['coherence_time_s']+1e-12:continue
        found=[]
        for head in range(len(channels)):
            users=np.delete(np.arange(len(channels)),head)
            qb=float(np.linalg.norm(delta[users].sum(0)))
            if qb>eps:continue
            dg=base.old.ofdma.digital(payload[users],channels[users,head],n*dims,config)
            if dg is None:continue
            energy=float(dg['energy_by_user'].sum())
            if energy>base.CAP*(1+1e-7) or dg['energy_by_user'].max()>config['available_energy_j']*(1+1e-7):continue
            found.append((energy,head,users,dg,qb))
        if found:
            best=min(found,key=lambda x:x[0]);break
    if best is None:return None,dict(status='no_feasible_design_found',access='integer_subcarrier_OFDMA',epsilon=eps)
    energy,head,users,dg,qb=best
    vector=(blocks[head]+q[users].sum(0)).reshape(-1)[:u.shape[1]]
    error=float(np.linalg.norm(vector-u.sum(0)))
    assert error<=qb+1e-9 and dg['max_power_w']<=config['power_limit_w']*(1+1e-6)
    assert dg['minimum_rate_margin']>=-1e-6
    meta=dict(status='feasible',access='integer_subcarrier_OFDMA',fu_bits=6,tones=128,
        head=head,head_candidates=len(channels),frame_count=n,energy_bound_j=energy,expected_energy_j=energy,
        uplink_s=float(n*dims.sum()/config['symbol_rate_hz']),conditional_rmse=error,epsilon=eps,
        quantization_error_bound=qb,max_power_w=dg['max_power_w'],max_user_energy_j=float(dg['energy_by_user'].max()),
        subcarrier_counts=dg['subcarrier_counts'].tolist(),minimum_rate_margin_bits=dg['minimum_rate_margin'],
        network_tx_cap_j=base.CAP,optimizer_wall_s=time.time()-start,
        component_budget_rule='FU-only: rel*FU_blocknorm_bound; same epsilon as pure analog FU; no forced analog slot')
    return vector,meta
