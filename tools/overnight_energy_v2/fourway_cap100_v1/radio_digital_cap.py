"""All-digital concurrent FL+FU via actual integer-subcarrier OFDMA.

Each UAV quantizes its local FL+FU sum once. This is the sufficient statistic
required by the same global update and avoids artificially doubled payload.
All transmitting UAVs occupy disjoint tones in the same block interval.
"""
from pathlib import Path
import sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(Path(__file__).resolve().parent))
import radio_guard_cap as guard
base=guard.base

def analog(l,u,channels,c,seed):
    """Scheduler-compatible interface; this implementation is wholly digital."""
    start=time.time();combined=l+u;active=np.flatnonzero(np.any(combined,axis=1))
    nl,nu,old,eps=guard.budget(l,u,c)
    if len(active)<=1:
        _,m=guard.original.silence(l,c)
        m.update(access='local_aggregate_no_uplink',head=int(active[0]) if len(active) else None,
                 epsilon=eps,physical_mode='digital_parallel_ofdma_sum')
        return combined.sum(0),m
    config=dict(c,quantization_bits=6,ofdma_subcarriers=128)
    _,blocks,q,delta,payload,_,dims,_=base.old.make_inputs(np.zeros_like(combined),combined,config)
    best=None
    for n in base.COUNTS:
        lengths=n*dims
        if lengths.sum()/c['symbol_rate_hz']>c['round_uplink_deadline_s']+1e-12:continue
        if np.max(lengths/c['symbol_rate_hz'])>c['coherence_time_s']+1e-12:continue
        found=[]
        for head in range(len(channels)):
            users=np.delete(np.arange(len(channels)),head)
            qb=float(np.linalg.norm(delta[users].sum(0)))
            if qb>eps:continue
            dg=base.old.ofdma.digital(payload[users],channels[users,head],lengths,config)
            if dg is None:continue
            energy=float(dg['energy_by_user'].sum())
            if energy>base.CAP*(1+1e-7):continue
            found.append((energy,head,users,dg,qb))
        if found:
            best=min(found,key=lambda v:v[0]);break
    if best is None:
        return None,dict(status='no_feasible_design_found',physical_mode='digital_parallel_ofdma_sum',epsilon=eps,
                         network_tx_cap_j=base.CAP,optimizer_wall_s=time.time()-start)
    energy,head,users,dg,qb=best
    vector=(blocks[head]+q[users].sum(0)).reshape(-1)[:l.shape[1]]
    raw=combined.sum(0);error=float(np.linalg.norm(vector-raw))
    assert error<=qb+1e-9 and qb<=eps+1e-9
    assert dg['max_power_w']<=c['power_limit_w']*(1+1e-6)
    assert dg['minimum_rate_margin']>=-1e-6
    duration=float(n*dims.sum()/c['symbol_rate_hz'])
    meta=dict(status='feasible',access='integer_subcarrier_OFDMA',physical_mode='digital_parallel_ofdma_sum',
      packet='Q6(local FL+FU sum), one payload per UAV per block',quantization_bits=6,tones=128,
      component_updates_individually_recovered=False,UAV_updates_individually_decoded=True,
      head=int(head),head_candidates=len(channels),frame_count=int(n),energy_bound_j=energy,expected_energy_j=energy,
      uplink_s=duration,conditional_rmse=error,error_bound=qb,epsilon=eps,quantization_error_bound=qb,
      realized_error_to_raw=error,max_power_w=dg['max_power_w'],max_user_energy_j=float(dg['energy_by_user'].max()),
      network_tx_cap_j=base.CAP,subcarrier_counts=dg['subcarrier_counts'].tolist(),minimum_rate_margin_bits=dg['minimum_rate_margin'],
      total_payload_bits=int(payload[users].sum()),error_target='raw unquantized FL+FU sum',
      FL_norm_bound=nl,FU_norm_bound=nu,original_epsilon=old,protected_epsilon=eps,
      requests_in_update=int(np.any(u,axis=1).sum()),optimizer_wall_s=time.time()-start,
      phases=[dict(stream='digital_local_sum',count=int(n),energy_j=energy,error_bound=qb)],
      radio_assumptions='Existing ideal coding/CSI; bandwidth-scaled noise; shared total128tones; no extra band for FU; same zero CP/control-overhead convention')
    return vector,meta
