from common import *
import importlib.util
import numpy as np
from phase_split_grid import make_inputs
from wireless_init_v11 import optimize_head,quantize
from transport import recover,energy_accounting

def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
legacy=module('retrieval_frozen_oma',ROOT/'tools/overnight_energy_v2/deletion_window_screen_v1/trajectory_transport.py')
# run_frontier imports a generic trajectory_transport; supply its unchanged OMA helper.
sys.modules['trajectory_transport']=legacy
frontier=module('retrieval_frozen_frontier',ROOT/'tools/overnight_energy_v2/deadline_frontier_v1/run_frontier.py')

def communicate(learning,unlearning,channels,c,method,seed):
 started=time.perf_counter();n,dimension=learning.shape
 l,u,decoded,delta,payload,rho,dims,total=make_inputs(learning,unlearning,c)
 target=(l+decoded).sum(0).reshape(-1)[:dimension]
 eps=max(c['absolute_tolerance_floor'],c['relative_tolerance']*float(np.linalg.norm(rho.sum(0))))
 is_digital=bool(np.any(payload))
 full_s=float(total.sum()/c['symbol_rate_hz'])
 assert full_s<=c['round_uplink_deadline_s']+1e-9
 assert np.max(total/c['symbol_rate_hz'])<=c['coherence_time_s']+1e-9
 best=None;diagnostics=[]
 for head in range(n):
  users=np.delete(np.arange(n),head);candidate=None;h=channels[users,head]
  if method=='oma_only':
   for bits in [4,6,8,10,12,16]:
    ql,dl,pl=quantize(l,bits,c['scale_metadata_bits'])
    err=float(np.linalg.norm(dl[users].sum(0)))
    if err>eps:continue
    required=pl[users]+payload[users]
    minimum=(required/np.log2(1+h[:,None]**2*c['power_limit_w'])).sum(0)
    if np.any(minimum>total):continue
    d=legacy.oma_solve(required,h,total,c,np.zeros(len(users)))
    if d is None:continue
    delivered=(l[head]+ql[users].sum(0)+decoded.sum(0)).reshape(-1)[:dimension]
    energy=float(d['energy_by_user'].sum())
    candidate=dict(energy_bound_j=energy,expected_energy_j=energy,uplink_s=full_s,
      error_bound=err,head=head,delivered=delivered,learning_bits=bits,rate_margin_bits=d['minimum_rate_margin'])
    break
  elif method=='aircomp_oma' and is_digital:
   d=frontier.strong_oma(head,channels,rho,payload,dims,total,eps,c)
   if d is not None:
    received=l[head].copy();rng=np.random.default_rng(seed)
    for m in range(len(dims)):
     gain=d['receiver'][m]*h*d['analog'][:,m]/np.where(rho[users,m]>0,rho[users,m],1)
     received[m]+=(gain[:,None]*l[users,m]).sum(0)
     received[m]+=rng.normal(0,d['receiver'][m]/np.sqrt(2*d['symbols'][m]),len(received[m]))
    delivered=(received+decoded.sum(0)).reshape(-1)[:dimension]
    candidate=dict(energy_bound_j=d['energy_j'],expected_energy_j=d['energy_j'],uplink_s=full_s,
      error_bound=d['error_bound'],head=head,delivered=delivered,analog_fraction=d['fraction'],rate_margin_bits=d['minimum_rate_margin'])
  else:
   lb=l.copy();rb=rho;bits=payload
   if method=='aircomp_only':lb=l+decoded;rb=np.linalg.norm(lb,axis=-1);bits=np.zeros_like(payload)
   symbols=total if np.any(bits) else 2*dims
   d=optimize_head(head,channels,rb,np.zeros_like(delta),bits,dims,symbols,eps,c,0.,'hybrid',digital_symbols=symbols)
   if d is not None:
    # Independent resource/error/rate audit, separate from solver status.
    audit=frontier.audit_hybrid(d,bits,eps,c)
    decoded_for_recovery=decoded if method!='aircomp_only' else np.zeros_like(decoded)
    delivered=recover(d,lb,decoded_for_recovery,decoded_for_recovery,np.random.default_rng(seed))[:dimension]
    e=energy_accounting(d,c)
    candidate=dict(energy_bound_j=float(d.energy_j),expected_energy_j=e['expected_energy_random_codewords_j'],
     uplink_s=float(symbols.sum()/c['symbol_rate_hz']),error_bound=float(d.error_bound),head=head,delivered=delivered,
     rate_margin_bits=audit['minimum_payload_margin_bits'] if np.any(bits) else None)
  diagnostics.append(dict(head=head,status='feasible' if candidate else 'no_feasible_design_found'))
  if candidate is not None and (best is None or candidate['energy_bound_j']<best['energy_bound_j']):best=candidate
 if best is None:return None,dict(status='no_feasible_design_found',head_diagnostics=diagnostics,optimizer_wall_s=time.perf_counter()-started,deadline_s=full_s)
 delivered=best.pop('delivered');best.update(status='feasible',epsilon=eps,head_diagnostics=diagnostics,
  optimizer_wall_s=time.perf_counter()-started,realized_error_to_canonical=float(np.linalg.norm(delivered-target)),
  canonical_quantization_error=float(np.linalg.norm((decoded-u).sum(0))),
  quantization_error_bound=float(np.linalg.norm(delta.sum(0))),requests_in_update=int(np.any(unlearning!=0,axis=1).sum()),
  individual_request_vectors_recovered=method!='aircomp_only')
 assert best['error_bound']<=eps*(1+3e-5)
 assert best['energy_bound_j']>=-1e-10 and np.all(np.isfinite(delivered))
 return delivered,best

