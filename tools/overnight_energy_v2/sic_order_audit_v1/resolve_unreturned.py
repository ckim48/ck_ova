"""Resolve only unreturned SIC candidates; preserve main run and frozen sources."""
from pathlib import Path
import ast,json,hashlib,time
parent=Path('tools/overnight_energy_v2/sic_order_audit_v1/run.py')
dest=Path('results/overnight_energy_v2/sic_order_audit_v1/unreturned_resolution.json')
assert not dest.exists()
source=parent.read_text();nodes=[]
for n in ast.parse(source).body:
 if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='protocol' for t in n.targets):break
 segment=ast.get_source_segment(source,n)
 if isinstance(n,ast.Assert) and 'started.json' in segment:continue
 if 'fcntl.flock' in segment:continue
 if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='lock' for t in n.targets):continue
 nodes.append(n)
e={'__name__':'private_resolution','__file__':str(parent)};exec(compile(ast.Module(body=nodes,type_ignores=[]),'private_parent_prefix','exec'),e)
np=e['np'];cp=e['ns']['cp'];H=e['H'];ids=e['ids'];P=e['P'];W=e['W'];ns=e['ns'];t0=time.time();rows=[]
targets=[x for x in json.loads((dest.parent/'block_results.json').read_text()) if x['solver_unresolved_orders']]
for target in targets:
 block=target['block'];bits=target['bits'];L=target['L'];v=e['V'][:,block];u=e['U'][:,block];eps=float(e['allocated'][block])
 qu,du,payload=e['quantize'](u[:,None,:],bits,32);payload=payload[ids,0];qb=float(du[ids,0].sum());re=eps-qb
 ns['codec_bound']=qb;ns['codec_bias']=(qu[ids,0]-u[ids]).sum(0)
 engine=ns['OrderedOverlap'](H,v[ids],re);rho=np.linalg.norm(v[ids],axis=1);assert rho.sum()>re
 m=float(rho.sum()*np.sqrt(128/(2*L))/re);best=target['best']['energy_j'];count=0
 for order in e['orders']:
  co=e['ordered_coefficients'](payload,H,L,order);pmin=co/H**2*(1+m*m)
  if pmin.max()>P*(1+4e-6) or L/W*pmin.sum()>best*(1+1e-8):continue
  engine.order=order;x=engine.solve(L,payload)
  if x is not None:continue
  count+=1;original_status=engine.problem.status
  c=co/H**2;A=float(H@H);C=float(c.sum());S=float(H@np.sqrt(c));j=float(np.min(np.sqrt(c)/H))
  lower=L/W*max(m*m/A+(1+m*m)*C+2*m*np.sqrt(1+m*m)*j,(m+np.sqrt(1+m*m)*S)**2/A)
  row=dict(block=block,order_uavs=ids[list(order)].tolist(),original_status=original_status,energy_floor_j=lower,best_feasible_energy_j=best)
  if lower>best*(1+1e-8):row['resolution']='analytical_energy_bound_cannot_improve'
  else:
   engine.alpha.value=np.sqrt(co)/H;engine.noise.value=np.sqrt(128/(2*L))
   try:
    engine.problem.solve(solver='SCS',eps=1e-8,max_iters=100000,normalize=True)
    row['second_solver_status']=engine.problem.status
    if engine.problem.status in ['infeasible','infeasible_inaccurate']:
     row['resolution']='both_solvers_infeasible' if original_status in ['infeasible','infeasible_inaccurate'] else 'SCS_infeasible'
    elif engine.problem.status in ['optimal','optimal_inaccurate'] and engine.a.value is not None and engine.zz.value>0:
     a=np.maximum(engine.a.value,0);rx=re/float(engine.zz.value);p=co/H**2*(1+(H@a)**2);pow=(a+np.sqrt(p))**2
     error=float(np.abs(rx*H*a-rho).sum()+rx*np.sqrt(128/(2*L)));energy=float(L/W*pow.sum())
     valid=bool(error<=re*(1+4e-6) and pow.max()<=P*(1+4e-6))
     row.update(checked_feasible=valid,energy_j=energy,max_power_w=float(pow.max()),radio_error_ratio=error/re)
     row['resolution']='valid_SCS_design' if valid else 'still_unresolved_numerical_violation'
    else:row['resolution']='still_unresolved_solver_status'
   except cp.error.SolverError as exc:row['resolution']='still_unresolved_solver_error';row['error_type']=type(exc).__name__
  rows.append(row)
 assert count==target['solver_unresolved_orders'],(block,count,target['solver_unresolved_orders'])
out=dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),cases=rows,runtime_s=time.time()-t0,
 note='Secondary solver agreement is numerical evidence, not an exact infeasibility certificate. Original raw run is retained unchanged.')
dest.write_text(json.dumps(out,indent=2,allow_nan=False));print('RESOLVE_BEGIN'+json.dumps(out)+'RESOLVE_END')
