import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='1'
import importlib.util,pathlib,json,time
import numpy as np,cvxpy as cp
from scipy.optimize import minimize_scalar
p=pathlib.Path(__file__).with_name('run.py');s=importlib.util.spec_from_file_location('probe',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
OUT=m.OUT;proto=json.loads((OUT/'protocol.json').read_text());rows=json.loads((OUT/'rows.json').read_text())
z=np.load('results/overnight_energy_v2/retrieval_fourway_v2/fixed_updates.npz')
l=z['learning'][:,:128];u=z['unlearning'][:,:128];base=z['channels'];head=proto['head'];ids=np.delete(np.arange(10),head);eps=proto['epsilon'];start=time.time()

def grid_solve(h,vec,bits,L):
    H=h[ids,head];rho=np.linalg.norm(vec[ids],axis=1);b=bits[ids]
    coeff=np.zeros(len(ids));later=0.
    order=np.flatnonzero(b>0);order=order[np.argsort(-H[order],kind='stable')]
    for i in order[::-1]:coeff[i]=np.expm1(np.log(2)*b[i]/L)*(1+later);later+=coeff[i]
    if np.any(coeff/H**2>m.P):return None
    alpha=np.sqrt(coeff)/H;noise=np.sqrt(128/(2*L));scale=cp.Parameter(nonneg=True)
    a=cp.Variable(len(ids),nonneg=True);t=cp.Variable(len(ids),nonneg=True)
    N=cp.norm(cp.hstack([1,H@a]))
    constraints=[a+t<=np.sqrt(m.P),cp.multiply(rho==0,a)==0,cp.multiply(b==0,t)==0,
        cp.multiply(alpha,N)<=t,cp.sum(cp.abs(cp.multiply(scale*H,a)-rho/eps))+scale*noise<=1]
    pr=cp.Problem(cp.Minimize(cp.sum_squares(a+t)),constraints);best=None
    def evaluate(r):
        nonlocal best
        scale.value=float(r)
        try:pr.solve(solver='CLARABEL',max_iter=200,tol_gap_abs=1e-10,tol_feas=1e-10,tol_gap_rel=1e-10)
        except cp.error.SolverError:return 1e6
        if pr.status not in ['optimal','optimal_inaccurate'] or a.value is None:return 1e6
        av=np.maximum(a.value,0);tv=alpha*np.sqrt(1+(H@av)**2);rx=eps*r
        error=float(np.abs(rx*H*av-rho).sum()+rx*noise)
        if error>eps*(1+1e-6) or np.max((av+tv)**2)>m.P*(1+1e-6):return 1e6
        value=float(L/m.W*np.sum((av+tv)**2));gain=np.divide(rx*H*av,rho,out=np.zeros_like(av),where=rho>0)
        bias=((gain-1)[:,None]*vec[ids]).sum(0);rmse=float(np.sqrt(bias@bias+rx**2*128/(2*L)))
        candidate=dict(energy_j=value,expected_energy_j=float(L/m.W*np.sum(av**2+tv**2)),error_bound=error,
            expected_rmse=rmse,rmse_over_budget=rmse/eps,peak_power_bound_w=float(np.max((av+tv)**2)),receiver=rx,occupied_ms=L/m.W*1000)
        if best is None or value<best['energy_j']:best=candidate
        return value
    upper=1/noise;grid=upper*np.r_[np.geomspace(1e-4,.02,8),np.linspace(.03,.999999,45)]
    values=[evaluate(r) for r in grid];j=int(np.argmin(values))
    if values[j]<1e6:
        minimize_scalar(evaluate,bounds=(grid[max(0,j-1)],grid[min(len(grid)-1,j+1)]),method='bounded',options=dict(xatol=1e-8,maxiter=40))
    return best

# Refine all cases at the favorable (+10dB) channel; preserve the original 54-case sweep.
cache={};out=[]
for row in rows:
    L=row['symbols'];k=row['topk'];db=row['channel_gain_db'];h=base*10**(db/20);q,bits=m.encode(u,k)
    n=grid_solve(h,l,bits,L);aa=grid_solve(h,l+q,np.zeros(10),L)
    best=None
    for la in np.unique(np.r_[128,np.linspace(128,max(128,L-1),17)]):
        if la>=L:continue
        digital=m.ofdma(bits[ids],h[ids,head],L-la)
        if digital is None:continue
        key=(db,float(la))
        if key not in cache:cache[key]=grid_solve(h,l,np.zeros(10),la)
        a=cache[key]
        if a is None:continue
        c=dict(a,energy_j=a['energy_j']+digital['energy'],expected_energy_j=a['energy_j']+digital['energy'],analog_symbols=float(la),digital_symbols=float(L-la),occupied_ms=L/m.W*1000,peak_power_bound_w=max(a['peak_power_bound_w'],float(digital['power'].max())))
        if best is None or c['energy_j']<best['energy_j']:best=c
    out.append(dict(channel_gain_db=db,topk=k,symbols=L,deadline_ms=L/m.W*1000,fu_codec_relative_l2_error=row['fu_codec_relative_l2_error'],noma=n,airair=aa,air_ofdma=best,ofdma_only=row['ofdma_only']))
    m.write('polished_all_rows.json',out);print(k,L,{n:round(v['energy_j']*1e6,3) if v else None for n,v in [('noma',n),('airair',aa),('oma',best)]},flush=True)
# A deadline permits finishing early. Take the best feasible earlier design for every method.
for r in out:
    for mode in ['noma','airair','air_ofdma','ofdma_only']:
        available=[x[mode] for x in out if x['channel_gain_db']==r['channel_gain_db'] and x['topk']==r['topk'] and x['symbols']<=r['symbols'] and x[mode]]
        if available:r[mode]=min(available,key=lambda x:x['energy_j'])
    if r['noma'] and r['air_ofdma']:r['noma_to_oma_ratio']=r['noma']['energy_j']/r['air_ofdma']['energy_j']
m.write('polished_all_deadline_envelope.json',out)
paired=[r for r in out if r.get('noma_to_oma_ratio') is not None]
m.write('polish_all_summary.json',dict(runtime_s=time.time()-start,scope='All 54 cases; scalar receiver grid and local refinement for BOTH analog/NOMA; continuous OFDMA and 17-point time grid; early finish allowed.',
    noma_wins=sum(r['noma_to_oma_ratio']<.99 for r in paired),paired=len(paired),best=min(paired,key=lambda x:x['noma_to_oma_ratio']) if paired else None))
print('COMPLETE',time.time()-start,flush=True)
