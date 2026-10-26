"""Vectorized equivalent perfect-SIC full-overlap SOCP, independently audited.

Eliminate least digital power using (I-F)^-1 gamma. Fixed-receiver analog
and digital-amplitude variables still optimize the original robust energy.
No physical constraint or initialization family is removed.
"""
import numpy as np
import cvxpy as cp
from wireless_init_v11 import receive_optimum,Design

def optimize_head(head,channels,rho_all,delta_all,payload_all,dims,symbols,epsilon,c,chi,mode='hybrid',digital_symbols=None):
 assert chi==0 and mode=='hybrid' and np.all(delta_all==0)
 assert digital_symbols is None or np.array_equal(symbols,digital_symbols)
 users=np.delete(np.arange(len(channels)),head);h=channels[users,head];H=h[:,None]
 rho=rho_all[users];bits=payload_all[users];L=np.asarray(symbols,float);dims=np.asarray(dims,float)
 if np.any(h<=0) or np.any(L<dims) or np.max(L/c['symbol_rate_hz'])>c['coherence_time_s'] or L.sum()/c['symbol_rate_hz']>c['round_uplink_deadline_s']:return None
 coefficient=np.zeros_like(bits,dtype=float)
 for m in range(len(L)):
  active=np.flatnonzero(bits[:,m]>0);order=active[np.argsort(-h[active],kind='stable')];later=0.
  for u in order[::-1]:
   coefficient[u,m]=np.expm1(np.log(2)*bits[u,m]/L[m])*(1+later);later+=coefficient[u,m]
 alpha=np.sqrt(coefficient)/H;duration=L/c['symbol_rate_hz'];noise=np.sqrt(dims/(2*L))
 def digital(a):return alpha*np.sqrt(1+(H*a).sum(0)**2)
 def audit(a,t,rx):
  if not all(np.all(np.isfinite(x)) for x in [a,t,rx]):return False
  error=np.linalg.norm(np.abs(rx*H*a-rho).sum(0)+rx*noise)
  return (min(a.min(),t.min(),rx.min())>=-1e-10 and np.max((a+t)**2)<=c['power_limit_w']*(1+1e-6)
    and np.max((duration*(a+t)**2).sum(1))<=c['available_energy_j']*(1+1e-6) and error<=epsilon*(1+1e-5))
 def step(rx):
  a=cp.Variable(rho.shape,nonneg=True);t=cp.Variable(rho.shape,nonneg=True)
  I=cp.sum(cp.multiply(H,a),axis=0);N=cp.norm(cp.vstack([np.ones(len(L)),I]),axis=0)
  constraints=[a+t<=np.sqrt(c['power_limit_w']),cp.multiply(rho==0,a)==0,cp.multiply(bits==0,t)==0]
  constraints += [cp.multiply(alpha[u],N)<=t[u] for u in range(len(users))]
  error=cp.sum(cp.abs(cp.multiply(rx*H,a)-rho),axis=0)+rx*noise
  energy=cp.sum(cp.multiply(duration,cp.square(a+t)),axis=1)
  constraints += [cp.norm(error/epsilon)<=1,energy<=c['available_energy_j']]
  problem=cp.Problem(cp.Minimize(cp.sum(energy)/duration.sum()),constraints)
  try:problem.solve(solver='CLARABEL',max_iter=150)
  except cp.error.SolverError:return None
  if problem.status not in ['optimal','optimal_inaccurate'] or a.value is None:return None
  av=np.maximum(a.value,0);tv=digital(av)
  return (av,tv,rx) if audit(av,tv,rx) else None
 def init():
  y=cp.Variable(rho.shape,nonneg=True);z=cp.Variable(rho.shape,nonneg=True);r=cp.Variable(len(L),nonneg=True)
  rr=cp.vstack([r]*len(users));I=cp.sum(cp.multiply(H,y),axis=0);N=cp.norm(cp.vstack([r,I]),axis=0)
  constraints=[y+z<=np.sqrt(c['power_limit_w'])*rr,cp.multiply(rho==0,y)==0,cp.multiply(bits==0,z)==0]
  constraints += [cp.multiply(alpha[u],N)<=z[u] for u in range(len(users))]
  error=cp.sum(cp.abs(cp.multiply(H,y)-rho/epsilon),axis=0)+cp.multiply(r,noise)
  problem=cp.Problem(cp.Minimize(cp.norm(error)),constraints)
  try:problem.solve(solver='CLARABEL',max_iter=300)
  except cp.error.SolverError:return None
  if problem.status not in ['optimal','optimal_inaccurate'] or y.value is None:return None
  rv=np.maximum(r.value,0);av=np.maximum(y.value,0)/np.maximum(rv,1e-15);rx=epsilon*rv;tv=digital(av)
  return (av,tv,rx) if audit(av,tv,rx) else None
 def improve(initial):
  a,t,rx=initial;history=[float((duration*(a+t)**2).sum())]
  best=(a,t,rx)
  for _ in range(c['max_ao_iterations']):
   newrx=np.array([receive_optimum(h,a[:,m],rho[:,m],noise[m]) for m in range(len(L))])
   candidate=step(newrx)
   if candidate is None:break
   aa,tt,rr=candidate;value=float((duration*(aa+tt)**2).sum())
   if value>history[-1]+max(1e-12,history[-1]*1e-6):break
   old=history[-1];a,t,rx=best=candidate;history.append(value)
   if old-value<=max(1e-12,old*1e-5):break
  a,t,rx=best
  return Design(head,users,a,t,rx,history[-1],float(np.linalg.norm(np.abs(rx*H*a-rho).sum(0)+rx*noise)),history,
     H*np.ones((1,len(L))),rho,np.zeros_like(rho),L,dims,0.,'hybrid',L)
 ref=np.maximum(np.max(rho/np.maximum(H*np.sqrt(c['power_limit_w'])*.3,1e-15),axis=0),1e-10)
 best=None
 starts=[step(ref*x) for x in np.geomspace(.3,6,c['receive_initializations'])]+[init()]
 for initial in starts:
  if initial is None:continue
  candidate=improve(initial)
  if best is None or candidate.energy_j<best.energy_j:best=candidate
 return best
