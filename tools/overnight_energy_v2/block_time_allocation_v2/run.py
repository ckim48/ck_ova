"""CPU fixed-update radio audit: common quality, variable block durations."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import ast,json,time,hashlib,fcntl,itertools
import numpy as np
ROOT=Path.cwd();OUT=ROOT/'results/overnight_energy_v2/block_time_allocation_v2'
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
assert np.array_equal(V[:,:D],ns['learning']) and np.array_equal(U[:,:D],ns['unlearning'])
head=ns['head'];ids=np.delete(np.arange(10),head);H=channels[ids,head]*10**.5
W=ns['W'];P=ns['P'];BITS=[6,8]
TS=[128,160,192,256,384,512,640,736,768,800,832,864,1024,1536,1984,2016,2048,2080,2112,3072,4096,8192]
LAS=[x for x in ns['LAS'] if x<max(TS)];ETAS=[0.]+ns['ETAS']
Q=32;BUDGETS=[N*800,N*2048]
NAMES=['full_noma','partial_noma','separate_noma','adaptive_noma','air_ofdma','airair_raw','ofdma_only']
bv=V.reshape(10,N,D);bu=U.reshape(10,N,D)
base_eps=.2*np.linalg.norm(bv,axis=2).sum(0);global_eps=float(np.linalg.norm(base_eps))
_,qv6,_=quantize(bv,6,32);_,qu6,_=quantize(bu,6,32)
reserved=(qv6[ids]+qu6[ids]).sum(0)
allocated=reserved+(global_eps-np.linalg.norm(reserved))*base_eps/global_eps
assert np.linalg.norm(allocated)<=global_eps and np.all(allocated>reserved)
proto=dict(scope='All 20480 coordinates of one fixed training update. Common v2 RMS allocation; optimize duration of each sequential block using a discrete dynamic program. Not a training experiment.',
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),backend_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    head=head,channel_gain_db=10,blocks=N,block_dimension=D,fu_bits=BITS,bandwidth_hz=W,power_bound_w=P,
    block_length_grid=TS,analog_length_grid=LAS,payload_fraction_grid=ETAS,total_symbol_budgets=BUDGETS,dp_unit_symbols=Q,
    global_RMS_budget=global_eps,common_allocated_global_RMS=float(np.linalg.norm(allocated)),
    quality='Same v2 preset RMS allocation for every method and Q6/Q8, based on current training vectors and native Q6 quantizer bounds. No test/reference data.',
    physical='Conservative TX energy and block-average power; perfect SIC/CSI and ideal coding; fixed descending-channel SIC order; same native quantization bits and scale.',
    limitations=['Single stored update/geometry, fixed CSI-selected head, no learning/deletion conclusion.',
      'DP exact only on supplied per-block candidates; partial splits and OFDMA bandwidth are numerical searches, not a continuous global optimum.',
      'Total symbol budget is an upper bound, unused time is allowed. All blocks are sequential and each <=8192 symbols, under the inherited 10000-symbol cap.',
      'No per-user round battery cap, receiver/SIC/feedback/FEC/flight energy. Uplink only excludes local and decoding time.',
      'Raw AirComp+AirComp recovers only the aggregate; digital methods recover request payloads. Neither capability alone proves deletion.'])
write('protocol.json',proto);write('started.json',dict(pid=os.getpid(),unix=time.time()))
def compact(x):
    if x is None:return None
    keys=['energy_j','expected_energy_j','peak_power_w','total_error_bound','actual_expected_aggregate_rmse','analog_symbols','digital_tail_symbols','fu_overlap_fraction','radio_error_bound','mode']
    return {k:v for k,v in x.items() if k in keys}
def best(xs):return min(xs,key=lambda x:x['energy_j']) if xs else None
ofdma_cache={};tail_cache={}
def ofdma(bits,L):
    key=(tuple(np.asarray(bits,dtype=int)),int(L))
    if key not in ofdma_cache:ofdma_cache[key]=ns['old'].ofdma(bits,H,L)
    return ofdma_cache[key]
def tail(bits,L):
    key=(tuple(np.asarray(bits,dtype=int)),int(L))
    if key not in tail_cache:tail_cache[key]=ns['noma_tail'](H,bits,L)
    return tail_cache[key]
def optimize(costs,lengths,budget):
    """Multiple-choice knapsack, exact total time states with <= budget terminal."""
    assert np.all(np.asarray(lengths)%Q==0) and budget%Q==0
    units=np.asarray(lengths,dtype=int)//Q;cap=budget//Q
    dp=np.full(cap+1,np.inf);dp[0]=0.
    choices=np.full((len(costs),cap+1),-1,dtype=np.int16)
    for i,row in enumerate(costs):
        nxt=np.full(cap+1,np.inf)
        for j,e in enumerate(row):
            if not np.isfinite(e):continue
            u=units[j]
            if u>cap:continue
            val=dp[:cap+1-u]+e;better=val<nxt[u:]
            nxt[u:][better]=val[better];choices[i,u:][better]=j
        dp=nxt
    state=int(np.argmin(dp))
    if not np.isfinite(dp[state]):return None
    energy=float(dp[state]);used=state*Q;selected=[]
    for i in range(len(costs)-1,-1,-1):
        j=int(choices[i,state]);assert j>=0
        selected.append(j);state-=int(units[j])
    assert state==0
    selected.reverse()
    assert abs(sum(costs[i,j] for i,j in enumerate(selected))-energy)<=1e-12
    assert sum(lengths[j] for j in selected)==used<=budget
    return dict(energy_j=energy,used_symbols=used,selected=selected)
# Verify DP against exhaustive enumeration, including missing choices and slack.
testcost=np.array([[3.,2.,1.],[np.inf,1.,.8],[2.,.9,.4]])
testlength=[32,64,96]
for cap in [96,128,192,256,288]:
    candidates=[(sum(testcost[i,j] for i,j in enumerate(js)),sum(testlength[j] for j in js)) for js in itertools.product(range(3),repeat=3) if sum(testlength[j] for j in js)<=cap]
    expected=min((e for e,t in candidates),default=np.inf);actual=optimize(testcost,testlength,cap)
    assert (actual is None and not np.isfinite(expected)) or (actual is not None and abs(actual['energy_j']-expected)<1e-12)
t0=time.time();rows=[];verified=0
for block in range(N):
    sl=slice(block*D,(block+1)*D);v=V[:,sl];u=U[:,sl];eps=float(allocated[block])
    ns['codec_bound']=0.;ns['codec_bias']=np.zeros(D)
    rawengine=ns['ConvexOverlap'](H,(v+u)[ids],eps)
    raw={T:rawengine.solve(T,np.zeros(len(ids))) for T in TS}
    for bits in BITS:
        qu,du,payu=quantize(u[:,None,:],bits,32);ql,dl,payl=quantize(v[:,None,:],bits,32)
        qu=qu[:,0,:];du=du[:,0];dl=dl[:,0];payu=payu[:,0];payl=payl[:,0]
        qb=float(du[ids].sum());qe=float(qb+dl[ids].sum())
        ns['codec_bound']=qb;ns['codec_bias']=(qu[ids]-u[ids]).sum(0)
        eng=ns['ConvexOverlap'](H,v[ids],eps-qb) if qb<eps else None
        states=[]
        if eng:
            for la in LAS:
                for eta in ETAS:
                    bo=np.floor(payu[ids]*eta+.5);a=eng.solve(la,bo)
                    if a:states.append((la,eta,bo,a))
        for T in TS:
            methods=dict(full_noma=eng.solve(T,payu[ids]) if eng else None,airair_raw=raw[T])
            partial=[];separate=[];orthogonal=[]
            for la,eta,bo,a in states:
                if la>=T:continue
                dt=tail(payu[ids]-bo,T-la)
                if dt:
                    value=dict(a,energy_j=a['energy_j']+dt['energy_j'],expected_energy_j=a['expected_energy_j']+dt['energy_j'],
                        peak_power_w=max(a['peak_power_w'],dt['peak_power_w']),digital_tail_symbols=T-la,fu_overlap_fraction=float(bo.sum()/payu[ids].sum()))
                    (separate if eta==0 else partial).append(value)
                if eta==0:
                    dd=ofdma(payu[ids],T-la)
                    if dd:orthogonal.append(dict(a,energy_j=a['energy_j']+dd['energy'],expected_energy_j=a['expected_energy_j']+dd['energy'],peak_power_w=max(a['peak_power_w'],float(dd['power'].max())),digital_tail_symbols=T-la))
            methods.update(partial_noma=best(partial),separate_noma=best(separate),air_ofdma=best(orthogonal))
            dd=ofdma((payu+payl)[ids],T) if qe<=eps else None
            methods['ofdma_only']=dict(energy_j=dd['energy'],expected_energy_j=dd['energy'],peak_power_w=float(dd['power'].max()),total_error_bound=qe) if dd else None
            methods['adaptive_noma']=best([dict(x,mode=k) for k,x in methods.items() if x and k in ['full_noma','partial_noma','separate_noma']])
            for name,x in methods.items():
                if x is None:continue
                assert x['peak_power_w']<=P*(1+4e-6)
                assert x['total_error_bound']<=eps*(1+4e-6)
                assert x.get('actual_expected_aggregate_rmse',0)<=x['total_error_bound']*(1+4e-6)
                verified+=1
            rows.append(dict(block=block,bits=bits,symbols=T,methods={k:compact(methods[k]) for k in NAMES}))
    if (block+1)%20==0:
        write('status.json',dict(stage='block_candidates',completed_blocks=block+1,required=N,unix=time.time()))
        print(json.dumps(dict(completed_blocks=block+1,elapsed_s=time.time()-t0)),flush=True)
write('block_candidates.json',rows)
lookup={(r['bits'],r['block'],r['symbols']):r['methods'] for r in rows}
prior=json.loads((ROOT/'results/overnight_energy_v2/all_block_radio_audit_v2/results.json').read_text())
crosschecks=0
for row in prior:
    now=lookup[row['bits'],row['block'],row['deadline_symbols']]
    for method in NAMES:
        old=row['methods'][method];new=now[method]
        if old:
            assert new and new['energy_j']<=old['energy_j']*(1+3e-5)+1e-12
            crosschecks+=1
summary=[]
for bits in BITS:
    for budget in BUDGETS:
        methods={}
        for method in NAMES:
            costs=np.array([[lookup[bits,i,T][method]['energy_j'] if lookup[bits,i,T][method] else np.inf for T in TS] for i in range(N)])
            design=optimize(costs,TS,budget)
            uniform=[lookup[bits,i,budget//N][method] for i in range(N)]
            uniform_e=sum(x['energy_j'] for x in uniform) if all(uniform) else None
            if not design:
                assert uniform_e is None
                methods[method]=dict(complete=False,energy_j=None,uniform_energy_j=uniform_e)
                continue
            choices=[lookup[bits,i,TS[j]][method] for i,j in enumerate(design['selected'])]
            rms=float(np.linalg.norm([x['total_error_bound'] for x in choices]));assert rms<=global_eps*(1+4e-6)
            if uniform_e is not None:assert design['energy_j']<=uniform_e*(1+1e-10)
            hist={str(T):sum(TS[j]==T for j in design['selected']) for T in TS}
            modes={k:sum(x.get('mode')==k for x in choices) for k in ['full_noma','partial_noma','separate_noma']} if method=='adaptive_noma' else None
            methods[method]=dict(complete=True,energy_j=design['energy_j'],uplink_s=design['used_symbols']/W,
                used_symbols=design['used_symbols'],uniform_energy_j=uniform_e,allocated_RMS=rms,
                saving_vs_uniform_percent=100*(1-design['energy_j']/uniform_e) if uniform_e is not None else None,
                max_block_average_power_bound_w=max(x['peak_power_w'] for x in choices),block_length_histogram=hist,adaptive_modes=modes,
                selected_block_symbols=[TS[j] for j in design['selected']])
        summary.append(dict(bits=bits,total_symbol_budget=budget,uplink_deadline_s=budget/W,methods=methods))
for bits in BITS:
    first,last=[r for r in summary if r['bits']==bits]
    for method in NAMES:
        a=first['methods'][method];b=last['methods'][method]
        if a['complete']:assert b['complete'] and b['energy_j']<=a['energy_j']*(1+1e-10)
checks=dict(candidate_cases=len(rows),verified_designs=verified,prior_feasible_comparisons=crosschecks,dp_exhaustive_small_cases=5,
    original_backend_unchanged=hashlib.sha256(source.read_bytes()).hexdigest()==proto['backend_sha256'],all_complete_solutions_global_RMS_checked=True,
    no_quality_relaxation=True,first128_input_matches=True,uniform_and_deadline_monotonicity=True)
assert checks['original_backend_unchanged'] and len(rows)==N*len(BITS)*len(TS)
report=dict(protocol=proto,checks=checks,summary=summary,note='Completed designs only can be ranked. These are communication bounds on a single fixed update; no deletion-quality or convergence claim.')
write('report.json',report);write('finished.json',dict(runtime_s=time.time()-t0,checks=checks));print('COMPLETE',time.time()-t0,flush=True)
