"""Fixed identical updates: feasible energy/deadline designs and physical lower bounds."""
from pathlib import Path
import hashlib,itertools,json,time,sys
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path[:0]=[str(ROOT/'tools/overnight_energy_v2/convergence_energy_delay_v1'),str(ROOT),
 str(ROOT/'tools/overnight_energy_v2'),str(ROOT/'tools/overnight_energy_v2/initialization_v11'),
 str(ROOT/'tools/overnight_energy_v2/receiver_control_v1')]
from trajectory_transport import oma_solve
from phase_split_grid import make_inputs
from wireless_init_v11 import optimize_head,receive_optimum
from receiver_control import sensitivity_bound
from transport import energy_accounting
OUT=ROOT/'results/overnight_energy_v2/deadline_frontier_v1'
CASES=[(5,12),(8,16),(5,8),(5,10),(8,12),(8,14),(5,14),(5,16),(8,19)]

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,x):
 t=path.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(path)

def lower_bounds(h,bits,dims,c):
 rows=[];P=c['power_limit_w'];W=c['symbol_rate_hz']
 for head in range(len(h)):
  users=np.delete(np.arange(len(h)),head);H=h[users,head]
  digital=(bits[users]/np.log2(1+H[:,None]**2*P)).sum(axis=0)
  req=dims+digital
  active=[i for i in users if np.any(bits[i]>0)]
  mac=dims.astype(float).copy()
  for mask in range(1,1<<len(active)):
   ids=[active[j] for j in range(len(active)) if mask>>j&1]
   mac=np.maximum(mac,bits[ids].sum(axis=0)/np.log2(1+P*np.sum(h[ids,head]**2)))
  rows.append(dict(head=head,separated_symbols_by_block=req.tolist(),hybrid_MAC_symbols_by_block=mac.tolist(),
   separated_total_time_lower_bound_s=float(req.sum()/W),
   separated_uniform_frame_lower_bound_s=float(np.max(req/dims)*dims.sum()/W),
   hybrid_MAC_uniform_frame_lower_bound_s=float(np.max(mac/dims)*dims.sum()/W)))
 return rows

def audit_hybrid(d,bits,eps,c):
 a=d.analog_rms;t=d.digital_sqrt_power;H=d.normalized_channels
 power=(a+t)**2
 min_margin=float('inf')
 for m in range(len(d.dimensions)):
  active=np.flatnonzero(bits[d.users,m]>0);active=active[np.argsort(-H[active,m],kind='stable')]
  q=H[:,m]**2*t[:,m]**2;interference=float((H[:,m]@a[:,m])**2)
  for j,u in enumerate(active):
   den=1+interference+q[active[j+1:]].sum()+d.chi*q[active[:j]].sum()
   capacity=d.digital_symbols[m]*np.log2(1+q[u]/den)
   min_margin=min(min_margin,float(capacity-bits[d.users[u],m]))
 noise=np.sqrt(d.dimensions/(2*d.symbols))+(np.sqrt(d.chi)*H*t).sum(axis=0)
 err=float(np.linalg.norm(np.abs(d.receiver*H*a-d.rho).sum(axis=0)+d.receiver*noise))
 e=(d.symbols/c['symbol_rate_hz']*power).sum(axis=1)+c['base_overhead_j']
 assert np.max(power)<=c['power_limit_w']*(1+1e-6)
 assert np.max(e)<=c['available_energy_j']*(1+1e-6)
 assert c['base_overhead_j']+c['head_overhead_j']<=c['available_energy_j']
 assert min_margin>=-2e-3 and err<=eps*(1+1e-5)
 assert np.all(np.isfinite(power)) and np.all(d.symbols>=d.dimensions)
 return dict(maximum_total_transmit_power_w=float(np.max(power)),maximum_client_energy_j=float(np.max(e)),
  minimum_payload_margin_bits=min_margin,analog_rms_bound=err,audit='passed')

def strong_oma(head,h,rho,payload,dims,total,eps,c):
 users=np.delete(np.arange(len(h)),head);H=h[users,head]
 digital=(payload[users]/np.log2(1+H[:,None]**2*c['power_limit_w'])).sum(axis=0)
 lo=float(np.max(dims/total));hi=float(np.min(1-digital/total))
 if hi<lo-1e-10:return None
 # Extend the previous grid with the mapping limit and feasibility-adapted fractions.
 fractions=sorted(set([lo]+[x for x in (.125,.25,.375,.5,.625,.75,.875) if lo<=x<=hi]
  +np.linspace(lo,max(lo,hi-1e-7),9).tolist()))
 best=None
 for f in fractions:
  la=np.maximum(dims,np.floor(total*f).astype(int));ld=total-la
  if np.any(ld<=0) or np.any(digital>ld*(1+1e-10)):continue
  ref=np.maximum(np.max(rho[users]/(H[:,None]*np.sqrt(c['power_limit_w'])*.3),axis=0),1e-10)
  for mult in np.geomspace(.3,6,c['receive_initializations']):
   rx=ref*mult;old=float('inf')
   for iteration in range(c['max_ao_iterations']+1):
    x=oma_solve(payload[users],H,ld,c,np.full(len(users),c['base_overhead_j']),rho[users],la,rx,eps)
    if x is None:break
    value=float(x['energy_by_user'].sum())
    if value>old+max(1e-10,old*1e-6):break
    if best is None or value<best['energy_j']:
     best=dict(x,energy_j=value,receiver=rx.copy(),users=users,head=head,symbols=la.copy(),digital_symbols=ld.copy(),fraction=float(f))
    if np.isfinite(old) and old-value<=max(1e-10,old*1e-5):break
    old=value
    rx=np.asarray([receive_optimum(H,x['analog'][:,m],rho[users,m],np.sqrt(dims[m]/(2*la[m]))) for m in range(len(dims))])
 return best

def summarize(rows):
 import csv
 with (OUT/'frontier.csv').open('w',newline='') as f:
  fields=['K','symbols_per_coordinate','deadline_s','method','status','energy_bound_j','expected_energy_j','head','application_rmse_budget','application_error_bound','optimizer_wall_s']
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,2,figsize=(11,4))
 for ax,K in zip(axes,[5,8]):
  for method in ['hybrid','aircomp_oma']:
   r=sorted([x for x in rows if x['K']==K and x['method']==method and x['status']=='feasible'],key=lambda x:x['deadline_s'])
   if r:ax.plot([x['deadline_s']*1000 for x in r],[x['energy_bound_j']*1000 for x in r],'o-',label=method)
  r=[x for x in rows if x['K']==K]
  if r:
   lb=r[0]['best_separated_time_lower_bound_s']*1000
   ax.axvline(lb,color='grey',linestyle='--',label='OMA necessary minimum time')
  ax.set_title(f'{K} simultaneous requests');ax.set_xlabel('Allowed uplink time (ms)');ax.set_ylabel('Communication energy bound (mJ)');ax.grid(alpha=.2)
  if ax.get_legend_handles_labels()[0]:ax.legend(fontsize=8)
 fig.suptitle('Identical fixed updates; feasible designs and necessary capacity limits')
 fig.tight_layout();fig.savefig(OUT/'energy_deadline_frontier.png',dpi=180);fig.savefig(OUT/'energy_deadline_frontier.pdf');plt.close(fig)

def main():
 OUT.mkdir(exist_ok=False)
 source=ROOT/'results/overnight_energy_v2/convergence_energy_delay_v1/plan.json'
 prior=json.loads(source.read_text());cfg=prior['config']
 for name,expected in prior['source_sha256'].items():assert sha(ROOT/name)==expected
 transport=ROOT/'tools/overnight_energy_v2/convergence_energy_delay_v1/trajectory_transport.py'
 assert sha(transport)==prior['runner_sha256']['trajectory_transport.py']
 plan=dict(config=cfg,cases=CASES,source_sha256=prior['source_sha256'],transport_sha256=sha(transport),runner_sha256=sha(Path(__file__)),
  scope='CPU fixed identical crossmodal updates for both access schemes. No model training or test-set tuning. Canonical Q8 requests, same application error budget and controller. Learning ablation is a separate GPU experiment.',
  qualifications=['Feasible designs are achievable candidates, not global minimum-energy proofs.','No feasible design found is distinct from a necessary-capacity exclusion.',
  'Capacity exclusions assume the fixed payload codec, given channels, equal bandwidth, per-user peak power, and the S^H S = I analog mapping; allow arbitrary OMA user time allocation.',
  'OMA analog/digital fraction grid is strengthened beyond the previous seven-point grid.',
  'Expected energy reaccounts the robust-bound-optimal design; it is not reoptimized under average-power constraints.',
  'Noise and residual error bounds are conservative. Ideal coding and perfect SIC are assumed, as in the frozen experiment.'])
 write(OUT/'plan.json',plan);rows=[]
 for K,spc in CASES:
  c=dict(cfg,symbols_per_coordinate=spc)
  f=ROOT/f'results/overnight_energy_v2/crossmodal_ablation_v1/prepared/crossmodal/k{K}/prepared.npz'
  with np.load(f) as z:learning,unlearning,h,g=[z[x].copy() for x in ['learning','unlearning','channels','gradient']]
  l,u,decoded,delta,payload,rho,dims,total=make_inputs(learning,unlearning,c)
  deadline=float(total.sum()/c['symbol_rate_hz']);c['round_uplink_deadline_s']=deadline+1e-9
  assert np.all(total/c['symbol_rate_hz']<=c['coherence_time_s'])
  active=np.flatnonzero(np.any(unlearning!=0,axis=1));canonical=decoded.reshape(len(h),-1)[:,:learning.shape[1]]
  lip=float(sensitivity_bound(canonical[active],g));epsilon=float(c['relative_tolerance']*np.linalg.norm(rho.sum(axis=0)));eps=epsilon/lip
  lb=lower_bounds(h,payload,dims,c)
  write(OUT/f'capacity_bounds_k{K}.json',dict(input_sha256=sha(f),bounds=lb,scope=plan['qualifications'][2]))
  for method in ['hybrid','aircomp_oma']:
   start=time.perf_counter();diagnostics=[];best=None;aud=None
   write(OUT/'progress.json',dict(status='running',K=K,symbols_per_coordinate=spc,method=method,completed_cases=len(rows)))
   for head in range(len(h)):
    lower=np.array(lb[head]['hybrid_MAC_symbols_by_block' if method=='hybrid' else 'separated_symbols_by_block'])
    if np.any(lower>total*(1+1e-10)):
     diagnostics.append(dict(head=head,status='excluded_by_necessary_capacity_bound'));continue
    if method=='hybrid':
     d=optimize_head(head,h,rho,np.zeros_like(delta),payload,dims,total,eps,c,c['sic_residual_fraction'],'hybrid',digital_symbols=total)
     if d is not None:
      a=audit_hybrid(d,payload,eps,c)
      if best is None or d.energy_j<best.energy_j:best=d;aud=a
    else:
     d=strong_oma(head,h,rho,payload,dims,total,eps,c)
     if d is not None and (best is None or d['energy_j']<best['energy_j']):best=d
    diagnostics.append(dict(head=head,status='feasible' if d is not None else 'no_feasible_design_found'))
   row=dict(K=K,symbols_per_coordinate=spc,deadline_s=deadline,method=method,
    application_rmse_budget=epsilon,controller_lipschitz=lip,input_sha256=sha(f),head_diagnostics=diagnostics,
    best_separated_time_lower_bound_s=min(x['separated_total_time_lower_bound_s'] for x in lb))
   if best is None:
    row['status']='excluded_by_necessary_capacity_bound' if all(x['status'].startswith('excluded') for x in diagnostics) else 'no_feasible_design_found'
   elif method=='hybrid':
    e=energy_accounting(best,c)
    row.update(status='feasible',energy_bound_j=best.energy_j,expected_energy_j=e['expected_energy_random_codewords_j'],head=int(best.head),application_error_bound=best.error_bound*lip,audit=aud)
    np.savez_compressed(OUT/f'k{K}_spc{spc}_hybrid_design.npz',analog=best.analog_rms,digital=best.digital_sqrt_power,receiver=best.receiver,users=best.users,channels=best.normalized_channels,symbols=total,rho=best.rho,payload=payload[best.users],dims=dims)
   else:
    e=best['energy_j']+len(h)*c['base_overhead_j']+c['head_overhead_j']
    row.update(status='feasible',energy_bound_j=e,expected_energy_j=e,head=best['head'],application_error_bound=best['error_bound']*lip,analog_fraction=best['fraction'],minimum_rate_margin=best['minimum_rate_margin'])
    np.savez_compressed(OUT/f'k{K}_spc{spc}_oma_design.npz',**{k:v for k,v in best.items() if isinstance(v,np.ndarray)})
   row['optimizer_wall_s']=time.perf_counter()-start;rows.append(row)
   write(OUT/'results.json',rows);summarize(rows)
   print(json.dumps({k:row.get(k) for k in ['K','symbols_per_coordinate','deadline_s','method','status','energy_bound_j','expected_energy_j','head','optimizer_wall_s']}),flush=True)
 write(OUT/'progress.json',dict(status='completed',completed_cases=len(rows)))

if __name__=='__main__':main()
