"""CPU analytical cost-floor audit for the existing conservative radio model."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import ast,json,time,hashlib,fcntl
import numpy as np
ROOT=Path.cwd();OUT=ROOT/'results/overnight_energy_v2/full_overlap_cost_bound_v1'
OUT.mkdir(parents=True,exist_ok=True)
lock=(OUT/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (OUT/'started.json').exists(),'Refuse duplicate'
def write(n,x):
    p=OUT/n;t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False));t.replace(p)
source=ROOT/'tools/overnight_energy_v2/minimum_latency_v2/run.py'
nodes=[]
for node in ast.parse(source.read_text()).body:
    if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='protocol' for t in node.targets):break
    nodes.append(node)
ns={'__name__':'backend','__file__':str(source)}
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),ns)
from wireless import quantize
path=ROOT/'results/overnight_energy_v2/retrieval_fourway_v2/fixed_updates.npz'
z=np.load(path,allow_pickle=False);V=z['learning'];U=z['unlearning'];channels=z['channels']
D=128;N=160;assert V.shape==U.shape==(10,N*D)
head=ns['head'];ids=np.delete(np.arange(10),head);H=channels[ids,head]*10**.5
W=ns['W'];P=ns['P'];BITS=[6,8];Q=32
reference_path=ROOT/'results/overnight_energy_v2/block_time_allocation_v2/report.json'
reference=json.loads(reference_path.read_text());TS=reference['protocol']['block_length_grid'];BUDGETS=reference['protocol']['total_symbol_budgets']
dp_source=ROOT/'tools/overnight_energy_v2/block_time_allocation_v2/run.py'
fun=next(n for n in ast.parse(dp_source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='optimize')
exec(compile(ast.Module(body=[fun],type_ignores=[]),'saved_DP_function','exec'),globals())
bv=V.reshape(10,N,D);bu=U.reshape(10,N,D);base_eps=.2*np.linalg.norm(bv,axis=2).sum(0);global_eps=float(np.linalg.norm(base_eps))
_,qv6,_=quantize(bv,6,32);_,qu6,_=quantize(bu,6,32)
reserved=(qv6[ids]+qu6[ids]).sum(0);allocated=reserved+(global_eps-np.linalg.norm(reserved))*base_eps/global_eps
assert np.linalg.norm(allocated)<=global_eps
def floor_power(m,c,h):
    # c_i is the minimum digital power with noise1 and no analog interference.
    # M=h.a >= m; t=sqrt(c)*sqrt(1+M^2). Each following bound grows with M.
    digital=(1+m*m)*float(c.sum());analog=m*m/float(h@h)
    cauchy=(m+np.sqrt(1+m*m)*float(h@np.sqrt(c)))**2/float(h@h)
    cross=2*m*np.sqrt(1+m*m)*float(np.min(np.sqrt(c)/h))
    return max(digital+analog+cross,cauchy)
# Random check of the necessary-noise threshold and both power inequalities.
rng=np.random.default_rng(817);worst_slack=1e10
for i in range(5000):
    h=np.exp(rng.uniform(-1,1,9));c=np.exp(rng.uniform(-4,2,9));c[rng.random(9)<.4]=0
    m=float(np.exp(rng.uniform(-3,2)));M=m*(1+float(rng.random()))
    a=rng.random(9);a*=M/(h@a);t=np.sqrt(c)*np.sqrt(1+M*M)
    lower=floor_power(m,c,h);actual=float(np.sum((a+t)**2))
    assert lower<=actual*(1+1e-12);worst_slack=min(worst_slack,actual-lower)
    R=float(np.exp(rng.uniform(-3,2)));eps=R*float(rng.uniform(.01,.99));n=float(np.exp(rng.uniform(-2,1)))
    lower_m=R*n/eps;below=lower_m*(1-float(rng.uniform(.001,.9)))
    minimum_surrogate=R*min(1,n/below)
    assert minimum_surrogate>eps
proto=dict(scope='Analytical lower bounds inside the existing norm-triangle RMS/interference and robust TX-energy model; not a physical impossibility claim or training experiment.',
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),backend_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    candidate_report_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest(),head=head,channel_gain_db=10,blocks=N,dimension=N*D,
    bits=BITS,uniform_block_lengths=[800,2048],variable_length_grid=TS,total_symbol_budgets=BUDGETS,
    common_global_RMS_budget=global_eps,allocated_RMS=float(np.linalg.norm(allocated)),power_bound_w=P,
    derivation=[
      'Let R=sum(remote learning norms), e=block RMS budget - FU quantization bound, n=sqrt(D/(2L)), M=sum(H_i a_i), receive magnitude r>=0.',
      'The radio bound sum|r H_i a_i-rho_i|+r n is at least |r M-R|+r n. Its infimum over r>=0 is R*min(1,n/M).',
      'When 0<e<R, every feasible full-overlap design therefore has M>=m=R*n/e and analog interference M^2>=m^2. If e>=R use m=0.',
      'With c_i=co_i/H_i^2 from the ideal fixed-order SIC recursion, p_i>=(1+M^2)c_i. Thus digital energy is at least (L/W)(1+m^2)sum(c_i).',
      'Let A=sum(H_i^2), C=sum(c_i), S=sum(H_i sqrt(c_i)), j=min(sqrt(c_i)/H_i). Robust TX power >= max[m^2/A+(1+m^2)C+2m sqrt(1+m^2)j, (m+sqrt(1+m^2)S)^2/A]. Multiply byL/W for energy.',
      'All terms increase with M>=m. Ignoring individual power caps only relaxes this lower bound. Comparing it with a verified feasible comparator gives a conditional domination certificate for this model/head/quality/length grid.',
      'The dynamic program minimizes the lower-bound costs under the same discrete total-time budget. Even physically infeasible candidates are allowed in this relaxed calculation, which can only lower the bound.'
    ],limitations=[
      'RMS and analog-interference norm bounds can be loose. A certificate for this conservative model is not a proof that real correlated waveforms cannot do better.',
      'Fixed head7, prescribed common quality allocation, single stored training update and +10dB geometry. Not a certificate over other heads or quality allocations.',
      'Variable-time certificate only covers the stated length grid, not arbitrary continuous durations.',
      'Perfect SIC/CSI and ideal coding, no feedback/receiver/FEC costs, pure uplink energy/time. No VLM accuracy or deletion claim.',
      'AirComp+AirComp only provides aggregate information and is reported as such; no failure is inferred from this.'
    ])
write('protocol.json',proto);write('started.json',dict(pid=os.getpid(),unix=time.time()))
t0=time.time();rows=[];solutions_checked=0;threshold_valid=0
for block in range(N):
    v=bv[:,block,:];u=bu[:,block,:];eps=float(allocated[block]);rho=np.linalg.norm(v[ids],axis=1);R=float(rho.sum())
    for bits in BITS:
        qu,du,payu=quantize(u[:,None,:],bits,32);qu=qu[:,0,:];du=du[:,0];payu=payu[:,0];qb=float(du[ids].sum());radio_eps=eps-qb
        assert radio_eps>0
        ns['codec_bound']=qb;ns['codec_bias']=(qu[ids]-u[ids]).sum(0)
        engine=ns['ConvexOverlap'](H,v[ids],radio_eps)
        for L in TS:
            m=R*np.sqrt(D/(2*L))/radio_eps if radio_eps<R else 0.
            co=ns['coefficients'](payu[ids],H,L);assert co is not None
            c=co/H**2;no_analog=L/W*float(c.sum());lower=L/W*floor_power(m,c,H)
            row=dict(block=block,bits=bits,symbols=L,remote_learning_norm_sum=R,radio_error_budget=radio_eps,
                minimum_analog_interference_to_noise=m*m,digital_power_amplification_floor=1+m*m,
                digital_without_analog_energy_j=no_analog,digital_energy_floor_j=no_analog*(1+m*m),full_robust_energy_floor_j=lower)
            if L in [800,2048]:
                x=engine.solve(L,payu[ids]);row['full_feasible_design_found']=x is not None
                if x:
                    actual_I=float((H@x['a'])**2)
                    actual_digital=L/W*float(x['po'].sum())
                    assert lower<=x['energy_j']*(1+4e-6)
                    assert actual_I>=m*m*(1-4e-6)
                    assert abs(actual_digital-no_analog*(1+actual_I))<=1e-10*max(1,actual_digital)
                    assert x['total_error_bound']<=eps*(1+4e-6) and x['peak_power_w']<=P*(1+4e-6)
                    row.update(actual_full_energy_j=x['energy_j'],actual_interference_to_noise=actual_I,actual_digital_energy_j=actual_digital,
                        lower_bound_fraction_of_full=lower/x['energy_j'],digital_fraction_of_robust_full=actual_digital/x['energy_j'])
                    solutions_checked+=1
                threshold_valid+=int(radio_eps<R)
            rows.append(row)
    if (block+1)%40==0:print(json.dumps(dict(blocks=block+1,elapsed_s=time.time()-t0)),flush=True)
write('block_bounds.json',rows)
lookup={(x['bits'],x['block'],x['symbols']):x for x in rows}
summary=[]
for bits in BITS:
    for budget in BUDGETS:
        T=budget//N;g=[lookup[bits,i,T] for i in range(N)]
        lo=sum(x['full_robust_energy_floor_j'] for x in g);digital=sum(x['digital_energy_floor_j'] for x in g)
        no_analog=sum(x['digital_without_analog_energy_j'] for x in g);found=[x for x in g if x['full_feasible_design_found']]
        costs=np.array([[lookup[bits,i,L]['full_robust_energy_floor_j'] for L in TS] for i in range(N)])
        dp=optimize(costs,TS,budget);assert dp is not None
        ref=next(x for x in reference['summary'] if x['bits']==bits and x['total_symbol_budget']==budget)
        feasible=ref['methods']['full_noma']['energy_j'] if ref['methods']['full_noma']['complete'] else None
        if feasible is not None:assert dp['energy_j']<=feasible*(1+4e-6)
        comparisons={}
        for method in ['air_ofdma','partial_noma','separate_noma','adaptive_noma','airair_raw']:
            x=ref['methods'][method]
            comparisons[method]=dict(feasible=x['complete'],energy_j=x.get('energy_j'),
                full_lower_bound_exceeds_comparator=bool(x['complete'] and dp['energy_j']>x['energy_j']*(1+1e-5)),
                lower_bound_over_comparator=dp['energy_j']/x['energy_j'] if x['complete'] else None)
        stats=np.array([x['digital_power_amplification_floor'] for x in g])
        summary.append(dict(bits=bits,total_uplink_deadline_s=budget/W,uniform_full_complete=len(found)==N,
            uniform_full_energy_j=sum(x['actual_full_energy_j'] for x in found) if len(found)==N else None,
            uniform_full_lower_bound_j=lo,uniform_digital_lower_bound_j=digital,uniform_digital_without_analog_j=no_analog,
            unavoidable_extra_digital_energy_j=digital-no_analog,
            uniform_actual_digital_j=sum(x['actual_digital_energy_j'] for x in found) if len(found)==N else None,
            minimum_interference_amplification_stats=dict(min=float(stats.min()),median=float(np.median(stats)),max=float(stats.max())),
            variable_time_full_lower_bound_j=dp['energy_j'],variable_time_full_found_energy_j=feasible,comparators=comparisons))
checks=dict(random_inequality_and_scalar_threshold_cases=5000,worst_random_power_slack=worst_slack,
    full_designs_checked=solutions_checked,uniform_threshold_cases=threshold_valid,lower_bound_candidates=len(rows),
    original_backend_unchanged=hashlib.sha256(source.read_bytes()).hexdigest()==proto['backend_sha256'],saved_DP_source_sha256=hashlib.sha256(dp_source.read_bytes()).hexdigest(),
    stored_comparator_constraints_previously_verified=True)
assert checks['original_backend_unchanged'] and len(rows)==N*len(BITS)*len(TS)
report=dict(protocol=proto,checks=checks,summary=summary)
write('report.json',report);write('finished.json',dict(runtime_s=time.time()-t0,checks=checks));print('COMPLETE',time.time()-t0,flush=True)
