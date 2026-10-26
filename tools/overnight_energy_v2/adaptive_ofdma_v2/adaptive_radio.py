"""Versioned adaptive NOMA / actual integer OFDMA, common raw-update RMS budget.

Independent ideal codes carry integer payloads in the overlap and tail. A zero
overlap payload is a permitted boundary and is reported, not called full NOMA.
AirComp+AirComp sends the unquantized sum. All digital FU uses the same Q6 codec.
"""
from pathlib import Path
import os,sys,time,importlib.util,concurrent.futures,multiprocessing
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'tools/overnight_energy_v2/ofdma_v1'),str(ROOT/'tools/overnight_energy_v2/retrieval_fourway_v2'),str(ROOT),str(ROOT/'tools/overnight_energy_v2'),str(ROOT/'tools/overnight_energy_v2/initialization_v11'),str(ROOT/'tools/overnight_energy_v2/receiver_control_v1')]
import numpy as np
import ofdma
from fast_solver import optimize_head
from phase_split_grid import make_inputs
from wireless_init_v11 import quantize
spec=importlib.util.spec_from_file_location('adaptive_overlap_audit',ROOT/'tools/overnight_energy_v2/partial_overlap_tx_v1/overlap_solver.py')
ov=importlib.util.module_from_spec(spec);spec.loader.exec_module(ov)
POOL=None
SPLITS=(0.,.05,.25,1.)
ANALOG_COUNTS=(1,2,8,32)

def close():
 global POOL
 if POOL is not None:POOL.shutdown(wait=True);POOL=None

def analog_design(head,h,rho,payload,dims,length,eps,c):
 return optimize_head(head,h,rho,np.zeros_like(rho),payload,dims,length,eps,c,0.,'hybrid',digital_symbols=length)

def worker(args):
 from threadpoolctl import threadpool_limits
 with threadpool_limits(limits=1):return design(*args)

def design(head,channels,l,u,q,delta,payload,rho,dims,total,eps,c,mode):
 users=np.delete(np.arange(len(channels)),head);h=channels[users,head];H=h[:,None];W=c['symbol_rate_hz'];best=None
 quant=float(np.linalg.norm(delta[users].sum(0))) if mode!='aircomp_aircomp' else 0.
 radioeps=eps-quant
 if radioeps<=0:return None
 def pack(a,length,energy,expected,metadata,ql=None,quant_override=None):
  qb=quant if quant_override is None else quant_override
  err=0. if a is None else float(a.error_bound)
  return dict(head=int(head),users=users,analog=a,analog_symbols=length,energy_j=float(energy),expected_j=float(expected),
   raw_error_bound=err+qb,radio_error_bound=err,quantization_error_bound=qb,metadata=metadata,quantized_learning=ql)
 if mode=='aircomp_aircomp' or not np.any(payload) and mode!='ofdma_only':
  lb=l+u if mode=='aircomp_aircomp' else l;rb=np.linalg.norm(lb,axis=-1);la=2*dims
  a=analog_design(head,channels,rb,np.zeros_like(payload),dims,la,eps,c)
  if a is None:return None
  return pack(a,la,a.energy_j,a.energy_j,dict(access='analog_sum',uplink_s=float(la.sum()/W)),quant_override=0.)
 if mode=='ofdma_only':
  for bits in [4,6,8,10,12,16]:
   ql,dl,pl=quantize(l,bits,c['scale_metadata_bits']);qb=float(np.linalg.norm((dl[users]+delta[users]).sum(0)))
   if qb>eps:continue
   dg=ofdma.digital(pl[users]+payload[users],h,total,c)
   if dg is None:continue
   meta=dict(access='integer_subcarrier_OFDMA',uplink_s=float(total.sum()/W),tones=dg['tones'],learning_bits=bits,
    counts=dg['subcarrier_counts'].tolist(),bandwidth_fractions=dg['bandwidth_fractions'].tolist(),powers=dg['powers'].tolist(),
    minimum_rate_margin_bits=dg['minimum_rate_margin'],max_power_w=dg['max_power_w'])
   return pack(None,np.zeros_like(total),dg['energy_by_user'].sum(),dg['energy_by_user'].sum(),meta,ql,qb)
  return None
 if mode=='aircomp_ofdma':
  for count in ofdma.phase_grid(total,dims):
   la=count*dims;lt=total-la;dg=ofdma.digital(payload[users],h,lt,c)
   if dg is None:continue
   a=analog_design(head,channels,rho,np.zeros_like(payload),dims,la,radioeps,c)
   if a is None:continue
   energy=(la/W*a.analog_rms**2).sum(1)+dg['energy_by_user']
   if np.max(energy)>c['available_energy_j']*(1+1e-6):continue
   meta=dict(access='integer_subcarrier_OFDMA',uplink_s=float(total.sum()/W),tones=dg['tones'],
    analog_symbols=la.tolist(),digital_symbols=lt.tolist(),counts=dg['subcarrier_counts'].tolist(),
    bandwidth_fractions=dg['bandwidth_fractions'].tolist(),powers=dg['powers'].tolist(),
    minimum_rate_margin_bits=dg['minimum_rate_margin'],max_power_w=dg['max_power_w'])
   d=pack(a,la,energy.sum(),energy.sum(),meta)
   if best is None or d['energy_j']<best['energy_j']:best=d
  return best
 assert mode=='adaptive_noma',mode
 candidates=0;feasible=0
 for count in sorted(set([x for x in ANALOG_COUNTS if np.all(x*dims<total)]+[int(total[0]//dims[0])])):
  la=count*dims;lt=total-la
  for eta in SPLITS:
   if np.any(lt==0) and eta!=1:continue
   bo=np.ceil(eta*payload).astype(float);bt=payload-bo
   co=ov.received_coefficients(bo[users],la,h);ct=ov.received_coefficients(bt[users],lt,h)
   if co is None or ct is None:continue
   po_lower=co/(H*H);pt=ct/(H*H)
   if np.max(po_lower)>c['power_limit_w'] or np.max(pt)>c['power_limit_w']:continue
   lower=float((la/W*po_lower+lt/W*pt).sum())
   if best is not None and lower>=best['energy_j']:continue
   candidates+=1;a=analog_design(head,channels,rho,bo,dims,la,radioeps,c)
   if a is None:continue
   fraction=np.divide(bo[users],payload[users],out=np.zeros_like(bo[users]),where=payload[users]>0)
   au=ov.audit(a.analog_rms,a.digital_sqrt_power**2,pt,a.receiver,h,rho[users],payload[users],la,total,fraction,dims,radioeps,c)
   if not au['passed']:continue
   feasible+=1;overlap_bits=int(bo[users].sum());totalbits=int(payload[users].sum())
   meta=dict(access='adaptive_partial_NOMA',uplink_s=float(total.sum()/W),analog_symbols=la.tolist(),
    tail_symbols=lt.tolist(),requested_overlap_payload_fraction=eta,overlap_payload_bits=overlap_bits,
    digital_payload_bits=totalbits,actual_overlap_payload_fraction=overlap_bits/max(1,totalbits),
    selected_boundary='separate' if overlap_bits==0 else ('full' if np.all(lt==0) else 'partial'),audit=au)
   d=pack(a,la,au['tx_energy_bound_j'],au['tx_energy_expected_j'],meta)
   if best is None or d['energy_j']<best['energy_j']:best=d
 if best is not None:best['metadata'].update(solved_candidates=candidates,feasible_candidates=feasible)
 return best

def communicate(learning,unlearning,channels,c,mode,seed):
 global POOL
 started=time.time();dimension=learning.shape[1];c=dict(c,quantization_bits=6,ofdma_subcarriers=128)
 assert mode in ['adaptive_noma','aircomp_aircomp','aircomp_ofdma','ofdma_only']
 assert c['sic_residual_fraction']==0
 l,u,q,delta,payload,rho,dims,total=make_inputs(learning,unlearning,c)
 eps=max(c['absolute_tolerance_floor'],c['relative_tolerance']*float(np.linalg.norm(rho.sum(0))))
 assert total.sum()/c['symbol_rate_hz']<=c['round_uplink_deadline_s']+1e-12
 assert np.max(total/c['symbol_rate_hz'])<=c['coherence_time_s']
 # This also guarantees a combined analog+tail per-user battery bound.
 assert total.sum()/c['symbol_rate_hz']*c['power_limit_w']<=c['available_energy_j']
 if POOL is None:POOL=concurrent.futures.ProcessPoolExecutor(max_workers=8,mp_context=multiprocessing.get_context('spawn'))
 jobs=[POOL.submit(worker,(head,channels,l,u,q,delta,payload,rho,dims,total,eps,c,mode)) for head in range(len(channels))]
 designs=[j.result() for j in jobs];valid=[d for d in designs if d is not None]
 if not valid:return None,dict(status='no_feasible_design_found',physical_mode=mode,epsilon=eps,optimizer_wall_s=time.time()-started)
 d=min(valid,key=lambda x:x['energy_j']);head=d['head'];users=d['users'];a=d['analog'];la=d['analog_symbols']
 if mode=='ofdma_only':received=l[head]+d['quantized_learning'][users].sum(0)+u[head]+q[users].sum(0)
 else:
  lb=l+u if mode=='aircomp_aircomp' else l;rb=np.linalg.norm(lb,axis=-1)
  gain=a.receiver*channels[users,head,None]*a.analog_rms/np.where(rb[users]>0,rb[users],1.)
  received=lb[head]+(gain[:,:,None]*lb[users]).sum(0);rng=np.random.default_rng(seed)
  for m in range(len(dims)):received[m]+=rng.normal(0,a.receiver[m]/np.sqrt(2*la[m]),received.shape[1])
  if mode!='aircomp_aircomp':received+=u[head]+q[users].sum(0)
 vector=received.reshape(-1)[:dimension];raw=(l+u).sum(0).reshape(-1)[:dimension]
 r=dict(status='feasible',physical_mode=mode,head=head,energy_bound_j=d['energy_j'],expected_energy_j=d['expected_j'],
  error_bound=d['raw_error_bound'],radio_error_bound=d['radio_error_bound'],
  quantization_error_bound=d['quantization_error_bound'],epsilon=eps,error_target='unquantized learning plus unlearning sum',
  fu_bits=6 if mode!='aircomp_aircomp' else None,fu_payload_bits=int(payload[users].sum()) if mode!='aircomp_aircomp' else 0,
  own_head_fu_unquantized=True,requests_in_update=int(np.any(unlearning,axis=1).sum()),
  individual_request_vectors_recovered=mode!='aircomp_aircomp',realized_error_to_raw=float(np.linalg.norm(vector-raw)),
  optimizer_wall_s=time.time()-started,head_candidates=len(designs),head_feasible=len(valid),**d['metadata'])
 assert r['error_bound']<=eps*(1+3e-5) and np.isfinite(vector).all()
 assert r['energy_bound_j']>=0
 return vector,r
