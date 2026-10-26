"""Run the unchanged frozen solver independently across candidate heads."""
from pathlib import Path
import os,sys,hashlib,json,concurrent.futures,multiprocessing,collections
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
FROZEN=ROOT/'tools/overnight_energy_v2/retrieval_fourway_v2'
sys.path[:0]=[str(FROZEN),str(ROOT/'tools/overnight_energy_v2/initialization_v11')]
POOL=None
CACHE=collections.OrderedDict()

def worker(args,kwargs):
    from threadpoolctl import threadpool_limits
    from fast_solver import optimize_head
    with threadpool_limits(limits=1):
        return optimize_head(*args,**kwargs)

def parallel(head,*args,**kwargs):
    global POOL
    if POOL is None:
        POOL=concurrent.futures.ProcessPoolExecutor(max_workers=8,mp_context=multiprocessing.get_context('spawn'))
    digest=hashlib.sha256()
    for item in args:
        if isinstance(item,np.ndarray):
            digest.update(str((item.shape,item.dtype)).encode());digest.update(item.tobytes())
        else:digest.update(json.dumps(item,sort_keys=True).encode())
    for k,v in sorted(kwargs.items()):
        digest.update(k.encode());digest.update(v.tobytes() if isinstance(v,np.ndarray) else json.dumps(v,sort_keys=True).encode())
    key=digest.hexdigest()
    if key not in CACHE:
        CACHE[key]=[POOL.submit(worker,(h,*args),kwargs) for h in range(len(args[0]))]
        if len(CACHE)>8:CACHE.popitem(last=False)
    return CACHE[key][head].result()

def close():
    global POOL
    if POOL is not None:POOL.shutdown(wait=True);POOL=None

def check():
    import time
    from fast_solver import optimize_head
    from phase_split_grid import make_inputs
    src=ROOT/'results/overnight_energy_v2/retrieval_long200_v3'
    old=ROOT/'results/overnight_energy_v2/retrieval_fourway_v2'
    c=json.loads((src/'plan.json').read_text())['config']
    z=np.load(old/'fixed_updates.npz');results=[]
    for digital in [False,True]:
        l,u,decoded,delta,payload,rho,dims,total=make_inputs(z['learning'],z['unlearning'] if digital else np.zeros_like(z['unlearning']),c)
        eps=max(c['absolute_tolerance_floor'],c['relative_tolerance']*float(np.linalg.norm(rho.sum(0))))
        symbols=total if digital else 2*dims
        args=(z['channels'],rho,np.zeros_like(delta),payload,dims,symbols,eps,c,0.,'hybrid')
        kwargs=dict(digital_symbols=symbols)
        t=time.time();a=optimize_head(0,*args,**kwargs);serial=time.time()-t
        t=time.time();b=parallel(0,*args,**kwargs)
        designs=[parallel(h,*args,**kwargs) for h in range(10)];wall=time.time()-t
        assert (a is None)==(b is None)
        row=dict(digital=digital,serial_one_head_s=serial,parallel_ten_heads_s=wall,
                 feasible_heads=sum(d is not None for d in designs))
        if a is not None:
            er=abs(a.energy_j-b.energy_j)/max(a.energy_j,1e-12)
            rr=abs(a.error_bound-b.error_bound)/eps
            assert er<1e-5 and rr<1e-5
            assert np.allclose(a.analog_rms,b.analog_rms,rtol=1e-5,atol=1e-9)
            row.update(relative_energy_difference=er,normalized_error_difference=rr)
        results.append(row)
    return dict(passed=True,workers=8,unchanged_solver=True,checks=results)

if __name__=='__main__':
    sys.path[:0]=[str(ROOT),str(ROOT/'tools/overnight_energy_v2'),str(ROOT/'tools/overnight_energy_v2/receiver_control_v1')]
    try:
        value=check();p=ROOT/'results/overnight_energy_v2/proxy_objective_v2';p.mkdir(exist_ok=True)
        (p/'parallel_checks.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value),flush=True)
    finally:close()
