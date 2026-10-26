"""Versioned OFDMA replacement transport with matching legacy learning targets."""
from pathlib import Path
import os,sys,time,concurrent.futures,multiprocessing
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='1'
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'tools/overnight_energy_v2/retrieval_fourway_v2'),str(ROOT),
 str(ROOT/'tools/overnight_energy_v2'),str(ROOT/'tools/overnight_energy_v2/initialization_v11'),
 str(ROOT/'tools/overnight_energy_v2/receiver_control_v1'),str(Path(__file__).resolve().parent)]
import numpy as np
import radio as old
import ofdma
from phase_split_grid import make_inputs
from wireless_init_v11 import quantize
POOL=None


def worker(args):
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):return ofdma.head_design(*args)


def designs(channels,rho,payload,dims,total,eps,c,mode):
    global POOL
    if POOL is None:POOL=concurrent.futures.ProcessPoolExecutor(max_workers=8,mp_context=multiprocessing.get_context('spawn'))
    jobs=[POOL.submit(worker,(h,channels,rho,payload,dims,total,eps,c,mode)) for h in range(len(channels))]
    return [j.result() for j in jobs]


def close():
    global POOL
    if POOL is not None:POOL.shutdown(wait=True);POOL=None


def communicate(learning,unlearning,channels,c,method,seed):
    if method not in ['aircomp_ofdma','ofdma_only']:return old.communicate(learning,unlearning,channels,c,method,seed)
    if method=='aircomp_ofdma' and not np.any(unlearning):
        v,r=old.communicate(learning,unlearning,channels,c,'hybrid',seed)
        r['access']='AirComp: no digital payload';return v,r
    start=time.perf_counter();dimension=learning.shape[1]
    l,u,decoded,delta,payload,rho,dims,total=make_inputs(learning,unlearning,c)
    eps=max(c['absolute_tolerance_floor'],c['relative_tolerance']*float(np.linalg.norm(rho.sum(0))))
    assert total.sum()/c['symbol_rate_hz']<=c['round_uplink_deadline_s']+1e-9
    assert np.max(total/c['symbol_rate_hz'])<=c['coherence_time_s']+1e-9
    candidates=[]
    if method=='aircomp_ofdma':
        for d in designs(channels,rho,payload,dims,total,eps,c,method):
            if d is None:continue
            a=d['analog'];head=d['head'];users=d['users'];h=channels[users,head]
            rx=a.receiver;la=d['symbols_analog'];rng=np.random.default_rng(seed)
            gain=rx*h[:,None]*a.analog_rms/np.where(rho[users]>0,rho[users],1)
            received=l[head]+(gain[:,:,None]*l[users]).sum(0)
            for m in range(len(dims)):received[m]+=rng.normal(0,rx[m]/np.sqrt(2*la[m]),l.shape[-1])
            v=(received+decoded.sum(0)).reshape(-1)[:dimension]
            candidates.append((v,d,a.error_bound,None))
    else:
        remaining=set(range(len(channels)))
        # Same bit-depth policy as the previous digital baseline: first feasible
        # precision per head; no favorable codec granted to a single scheme.
        for bits in [4,6,8,10,12,16]:
            ql,dl,pl=quantize(l,bits,c['scale_metadata_bits'])
            for d in designs(channels,rho,pl+payload,dims,total,eps,c,method):
                if d is None or d['head'] not in remaining:continue
                err=float(np.linalg.norm(dl[d['users']].sum(0)))
                if err>eps:continue
                v=(l[d['head']]+ql[d['users']].sum(0)+decoded.sum(0)).reshape(-1)[:dimension]
                candidates.append((v,d,err,bits));remaining.remove(d['head'])
            if not remaining:break
    if not candidates:return None,dict(status='no_feasible_design_found',access='OFDMA',optimizer_wall_s=time.perf_counter()-start)
    v,d,error,bits=min(candidates,key=lambda x:x[1]['energy_j']);dg=d['digital']
    r=dict(status='feasible',access='integer_subcarrier_OFDMA',physical_mode=method,
        head=int(d['head']),energy_bound_j=d['energy_j'],expected_energy_j=d['energy_j'],
        uplink_s=float(total.sum()/c['symbol_rate_hz']),error_bound=float(error),epsilon=eps,
        rate_margin_bits=dg['minimum_rate_margin'],ofdma_subcarriers=dg['tones'],
        bandwidth_fractions=dg['bandwidth_fractions'].tolist(),subcarrier_counts=dg['subcarrier_counts'].tolist(),
        maximum_digital_power_w=dg['max_power_w'],learning_bits=bits,
        analog_uplink_s=float(d.get('symbols_analog',np.array([0])).sum()/c['symbol_rate_hz']),
        digital_uplink_s=float(d['symbols_digital'].sum()/c['symbol_rate_hz']),
        requests_in_update=int(np.any(unlearning!=0,axis=1).sum()),individual_request_vectors_recovered=True,
        realized_error_to_canonical=float(np.linalg.norm(v-(l+decoded).sum(0).reshape(-1)[:dimension])),
        canonical_quantization_error=float(np.linalg.norm((decoded-u).sum(0))),
        optimizer_wall_s=time.perf_counter()-start)
    assert error<=eps*(1+3e-5) and np.isfinite(v).all()
    return v,r
