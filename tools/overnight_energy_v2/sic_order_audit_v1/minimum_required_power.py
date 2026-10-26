"""Conditioning check: minimize required per-user power for unresolved orders."""
from pathlib import Path
import ast,json,hashlib,time
parent=Path('tools/overnight_energy_v2/sic_order_audit_v1/run.py');source=parent.read_text();nodes=[]
for n in ast.parse(source).body:
 if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='protocol' for t in n.targets):break
 s=ast.get_source_segment(source,n)
 if isinstance(n,ast.Assert) and 'started.json' in s:continue
 if 'fcntl.flock' in s:continue
 if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='lock' for t in n.targets):continue
 nodes.append(n)
e={'__name__':'private_peak','__file__':str(parent)};exec(compile(ast.Module(body=nodes,type_ignores=[]),'private_parent_prefix','exec'),e)
np=e['np'];cp=e['ns']['cp'];H=e['H'];ids=e['ids'];P=e['P'];W=e['W'];base=Path('results/overnight_energy_v2/sic_order_audit_v1');dest=base/'minimum_required_power.json';assert not dest.exists()
targets=json.loads((base/'unreturned_resolution.json').read_text())['cases'];out=[]
for r in targets:
 if not r['resolution'].startswith('still_unresolved'):continue
 block=r['block'];v=e['V'][:,block];u=e['U'][:,block];eps=float(e['allocated'][block]);L=800
 qu,du,payload=e['quantize'](u[:,None,:],6,32);payload=payload[ids,0];qb=float(du[ids,0].sum());re=eps-qb;rho=np.linalg.norm(v[ids],axis=1)
 order=tuple(int(np.flatnonzero(ids==i)[0]) for i in r['order_uavs']);co=e['ordered_coefficients'](payload,H,L,order)
 a=cp.Variable(9,nonneg=True);t=cp.Variable(9,nonneg=True);z=cp.Variable(nonneg=True);tau=cp.Variable(nonneg=True)
 constraints=[a+t<=tau,cp.multiply(np.sqrt(co)/H,cp.norm(cp.hstack([1,H@a])))<=t,
 cp.norm1(cp.multiply(H,a)-rho/re*z)+np.sqrt(128/(2*L))<=z,cp.multiply(rho==0,a)==0]
 pr=cp.Problem(cp.Minimize(tau),constraints);sol=[]
 for solver in ['CLARABEL','SCS']:
  kwargs=dict(tol_gap_abs=1e-11,tol_gap_rel=1e-11,tol_feas=1e-11,max_iter=500) if solver=='CLARABEL' else dict(eps=1e-9,max_iters=200000)
  pr.solve(solver=solver,**kwargs)
  assert pr.status in ['optimal','optimal_inaccurate'] and a.value is not None and z.value>0
  av=np.maximum(a.value,0);rx=re/float(z.value);p=co/H**2*(1+(H@av)**2);power=(av+np.sqrt(p))**2
  err=float(np.abs(rx*H*av-rho).sum()+rx*np.sqrt(128/(2*L)))
  assert err<=re*(1+1e-6)
  sol.append(dict(solver=solver,status=pr.status,minimum_required_power_w=float(tau.value**2),checked_design_max_power_w=float(power.max()),radio_error_ratio=err/re))
 assert abs(sol[0]['minimum_required_power_w']-sol[1]['minimum_required_power_w'])<1e-6
 out.append(dict(block=block,order_uavs=r['order_uavs'],limit_w=P,solutions=sol,both_minima_above_limit=all(x['minimum_required_power_w']>P+1e-6 for x in sol)))
report=dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),cases=out,note='Common per-user block-average power bound minimized after removing fixed cap. Convex numerical cross-check; not an exact symbolic infeasibility proof. Original data retained.')
dest.write_text(json.dumps(report,indent=2));print('PEAK_BEGIN'+json.dumps(report)+'PEAK_END')
