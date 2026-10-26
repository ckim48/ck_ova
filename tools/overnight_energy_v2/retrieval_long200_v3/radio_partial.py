"""Partial-overlap Hybrid, with unchanged canonical target and independent audit."""
from common import *
import numpy as np
from radio import communicate as frozen_communicate
from phase_split_grid import make_inputs
from fast_solver import optimize_head
import importlib.util
_path=ROOT/'tools/overnight_energy_v2/partial_overlap_tx_v1/overlap_solver.py'
_spec=importlib.util.spec_from_file_location('long200_overlap_audit',_path)
overlap=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(overlap)
GRID=(.025,.05,.1,1/7,.25)

def communicate(learning,unlearning,channels,c,method,seed):
 if method!='hybrid' or not np.any(unlearning):
  return frozen_communicate(learning,unlearning,channels,c,method,seed)
 start=time.perf_counter();dimension=learning.shape[1]
 l,u,decoded,delta,payload,rho,dims,total=make_inputs(learning,unlearning,c)
 eps=max(c['absolute_tolerance_floor'],c['relative_tolerance']*float(np.linalg.norm(rho.sum(0))))
 la=2*dims;ld=total-la;lt=ld-la;W=c['symbol_rate_hz']
 assert np.all(lt>0) and np.all(ld/W<=c['coherence_time_s'])
 assert ld.sum()/W<=c['round_uplink_deadline_s']
 # Then each combined per-user battery constraint follows from peak power.
 assert ld.sum()/W*c['power_limit_w']<=c['available_energy_j']
 best=None;diagnostics=[]
 for head in range(len(channels)):
  users=np.delete(np.arange(len(channels)),head);h=channels[users,head];H=h[:,None]
  head_best=None
  for eta in GRID:
   pt=overlap.received_coefficients((1-eta)*payload[users],lt,h)/(H*H)
   if np.max(pt)>c['power_limit_w']*(1+1e-9):continue
   d=optimize_head(head,channels,rho,np.zeros_like(delta),eta*payload,dims,la,eps,c,0.,'hybrid',digital_symbols=la)
   if d is None:continue
   audit=overlap.audit(d.analog_rms,d.digital_sqrt_power**2,pt,d.receiver,h,rho[users],payload[users],la,ld,eta,dims,eps,c)
   if not audit['passed']:continue
   if head_best is None or audit['tx_energy_bound_j']<head_best:head_best=audit['tx_energy_bound_j']
   if best is None or audit['tx_energy_bound_j']<best[0]['tx_energy_bound_j']:best=(audit,d,eta)
  diagnostics.append(dict(head=head,status='feasible' if head_best is not None else 'no_feasible_design_found',energy_bound_j=head_best))
 if best is None:return None,dict(status='no_feasible_design_found',physical_mode='partial_overlap',head_diagnostics=diagnostics,optimizer_wall_s=time.perf_counter()-start)
 au,d,eta=best;users=d.users;H=channels[users,d.head,None]
 gain=d.receiver*H*d.analog_rms/np.where(rho[users]>0,rho[users],1.)
 received=l[d.head]+(gain[:,:,None]*l[users]).sum(0)
 rng=np.random.default_rng(seed)
 for m in range(len(dims)):received[m]+=rng.normal(0,d.receiver[m]/np.sqrt(2*la[m]),received.shape[1])
 delivered=(received+decoded.sum(0)).reshape(-1)[:dimension]
 target=(l+decoded).sum(0).reshape(-1)[:dimension]
 r=dict(status='feasible',physical_mode='partial_overlap',energy_bound_j=au['tx_energy_bound_j'],expected_energy_j=au['tx_energy_expected_j'],
  uplink_s=float(ld.sum()/W),head=int(d.head),error_bound=au['error_bound'],epsilon=eps,payload_overlap_fraction=eta,
  rate_margin_bits=au['minimum_payload_margin_bits'],head_diagnostics=diagnostics,phase_energy=au,
  realized_error_to_canonical=float(np.linalg.norm(delivered-target)),canonical_quantization_error=float(np.linalg.norm((decoded-u).sum(0))),
  quantization_error_bound=float(np.linalg.norm(delta.sum(0))),requests_in_update=int(np.any(unlearning!=0,axis=1).sum()),
  individual_request_vectors_recovered=True,optimizer_wall_s=time.perf_counter()-start)
 assert np.isfinite(delivered).all() and r['energy_bound_j']>=0 and r['error_bound']<=eps*(1+2e-5)
 return delivered,r
