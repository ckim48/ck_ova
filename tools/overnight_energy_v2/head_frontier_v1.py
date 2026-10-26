"""CPU-only diagnostic: quantization, energy budget and overlap frontiers.
Fixed raw updates, common fidelity, power and resources; not a VLM experiment.
"""
from pathlib import Path
import os, ast, json, time, hashlib, fcntl
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np

ROOT=Path.cwd()
OUT=ROOT/'results/overnight_energy_v2/head_frontier_v1'
OUT.mkdir(exist_ok=True,parents=True)
lock=(OUT/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (OUT/'started.json').exists(), 'Refuse duplicate campaign'
def write(name,x):
    p=OUT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False));t.replace(p)

source=ROOT/'tools/overnight_energy_v2/minimum_latency_v2/run.py'
tree=ast.parse(source.read_text());prefix=[]
for node in tree.body:
    if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='protocol' for t in node.targets):break
    prefix.append(node)
assert any(isinstance(n,ast.FunctionDef) and n.name=='choose_best' for n in prefix)
ns={'__name__':'isolated_latency_backend','__file__':str(source)}
exec(compile(ast.Module(body=prefix,type_ignores=[]),str(source),'exec'),ns)
ns['OUT']=OUT
V=ns['learning'];U=ns['unlearning'];ids=ns['ids'];eps=ns['total_eps']
W=ns['W'];P=ns['P'];D=ns['D'];BS=ns['bsearch'];best=ns['choose_best']
CVX=ns['ConvexOverlap'];tail=ns['noma_tail'];MAXL=ns['MAXL']
QUANTS=[6,8];DB=[0,10];CAPS=[.0003,None]
protocol=dict(scope='one frozen real 128-coordinate block; not VLM performance or convergence',
    quantization_bits=QUANTS,channel_gain_db=DB,network_energy_caps_j=CAPS,
    quantizer_error_policy='Native float32 scale and quantizer supplied conservative error bounds. v3 used measured per-user error norms.',
    raw_update_rms_budget=eps,power_per_user_w=P,bandwidth_hz=W,heads=list(range(10)),previous_fixed_head=ns['head'],
    users=10,total_requests=int(np.count_nonzero(np.linalg.norm(U,axis=1))),
    fairness='Identical raw update target, RMS budget and power cap. Digital methods share canonical Qb FU. Strong AirComp+AirComp sends the raw FL+FU sum without quantization.',
    resource_scope='RF TX conservative bound; uplink occupation only. Ideal SIC/CSI/coding.',
    optimization='Joint convex receiver/transmitter for fixed overlap. Restricted analog duration/payload split grid; optimized continuous OFDMA bandwidth.',
    adaptive='Minimum among full, nonzero partial and zero-overlap NOMA. Choosing zero overlap is allowed and disclosed.',
    limitations=['All ten candidate heads in one geometry, hypothetical gain shifts. No deletion guarantee from update error.',
                 'Each method chooses its own best head with identical constraints. Head local updates are exact and not uploaded.',
                 'Head-specific receiver, feedback and control overhead omitted: pure uplink RF/time screen only.',
                 'Power-only removes the chosen network energy cap, not the physical per-user power limit.',
                 'AirComp+AirComp recovers only the sum; no artificial failure penalty.',
                 'Offline fixed-update geometry. Update-statistic signaling and feedback costs are not included.',
                 'Failure of this conservative bound is not proof that true MSE violates the target.'],
    backend_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    input_sha256=hashlib.sha256((ROOT/'results/overnight_energy_v2/short_regime_probe_v1/fixed_input_block.npz').read_bytes()).hexdigest())
write('protocol.json',protocol);write('started.json',dict(pid=os.getpid(),unix=time.time()))
t0=time.time();rows=[];audits=[]
def encode(x,b):
    from wireless import quantize
    q,delta,payload=quantize(x[:,None,:],b,32)
    q=q[:,0,:];delta=delta[:,0];payload=payload[:,0]
    actual=np.linalg.norm(q-x,axis=1)
    assert np.all(actual<=delta+1e-12), 'native quantizer bound violated'
    return q,payload,delta

for head in range(10):
    ids=np.delete(np.arange(10),head)
    for db in DB:
        H=ns['base'][ids,head]*10**(db/20)
        # Raw analog baseline is independent of FU quantizer precision.
        ns['codec_bound']=0.;ns['codec_bias']=np.zeros(D)
        aa=CVX(H,(V+U)[ids],eps)
        aa_cases={cap:BS(lambda L:aa.solve(L,np.zeros(len(ids))),float('inf') if cap is None else cap,D) for cap in CAPS}
        for bits in QUANTS:
            q,b,delta=encode(U,bits);ql,bl,delta_l=encode(V,bits)
            qb=float(delta[ids].sum())
            measured=float(np.linalg.norm(q[ids]-U[ids],axis=1).sum())
            ns['codec_bound']=qb;ns['codec_bias']=(q[ids]-U[ids]).sum(0)
            qe=qb+float(delta_l[ids].sum())
            audits.append(dict(head=head,db=db,bits=bits,quantization_triangle_bound=qb,rms_budget=eps,
                               measured_triangle_error=measured,native_to_measured_ratio=qb/measured if measured else None,
                               budget_fraction=qb/eps,full_digital_bound=qe,quantizer_fits=qb<eps))
            states=[];engine=CVX(H,V[ids],eps-qb) if qb<eps else None
            if engine:
                for la in ns['LAS']:
                    for eta in [0.]+ns['ETAS']:
                        bo=np.floor(b[ids]*eta+.5);a=engine.solve(la,bo)
                        if a:states.append((la,eta,bo,a))
            for cap in CAPS:
                budget=float('inf') if cap is None else cap
                full=BS(lambda L:engine.solve(L,b[ids]),budget,D) if engine else None
                partial=[];separate=[];ofdma=[]
                for la,eta,bo,a in states:
                    if a['energy_j']>=budget:continue
                    bt=b[ids]-bo
                    def f_noma(L):
                        d=tail(H,bt,L-la)
                        if d is None:return None
                        return dict(a,energy_j=a['energy_j']+d['energy_j'],
                            expected_energy_j=a['expected_energy_j']+d['energy_j'],
                            peak_power_w=max(a['peak_power_w'],d['peak_power_w']),
                            digital_tail_symbols=L-la,fu_overlap_fraction=float(bo.sum()/b[ids].sum()))
                    x=BS(f_noma,budget,la+1)
                    if x:(separate if eta==0 else partial).append(x)
                    if eta==0:
                        def f_oma(L):
                            d=ns['old'].ofdma(b[ids],H,L-la)
                            if d is None:return None
                            return dict(a,energy_j=a['energy_j']+d['energy'],
                                expected_energy_j=a['expected_energy_j']+d['energy'],
                                peak_power_w=max(a['peak_power_w'],float(d['power'].max())),
                                digital_tail_symbols=L-la)
                        x=BS(f_oma,budget,la+1)
                        if x:ofdma.append(x)
                def f_digital(L):
                    if qe>eps:return None
                    d=ns['old'].ofdma((b+bl)[ids],H,L)
                    if d is None:return None
                    return dict(energy_j=d['energy'],expected_energy_j=d['energy'],
                        peak_power_w=float(d['power'].max()),total_error_bound=qe)
                methods=dict(full_noma=full,partial_noma=best(partial),air_ofdma=best(ofdma),
                    separate_noma=best(separate),airair_raw=aa_cases[cap],
                    ofdma_only=BS(f_digital,budget,1))
                choices=[dict(v,chosen_mode=k) for k,v in methods.items() if k in ['full_noma','partial_noma','separate_noma'] and v]
                methods['adaptive_noma']=best(choices)
                for v in methods.values():
                    if not v:continue
                    assert v['energy_j']<=budget*(1+1e-6)
                    assert v['peak_power_w']<=P*(1+4e-6)
                    assert v['total_error_bound']<=eps*(1+4e-6)
                row=dict(head=head,head_has_request=bool(np.linalg.norm(U[head])>0),remote_requests=int(np.count_nonzero(b[ids])),remote_fu_payload_bits=int(b[ids].sum()),db=db,bits=bits,cap_j=cap,methods={k:ns['compact'](v) if v else None for k,v in methods.items()})
                rows.append(row);write('results.json',rows)
                print(json.dumps(dict(head=head,db=db,bits=bits,cap_j=cap,completed=len(rows))),flush=True)
write('codec_audit.json',audits)

# Both quality budget and all raw updates are fixed across the head search.
assert len(rows)==80
fixed=json.loads((ROOT/'results/overnight_energy_v2/deadline_frontier_v4/results.json').read_text())
reduction=[];summary=[];verified=0
for db in DB:
    for bits in QUANTS:
        for cap in CAPS:
            group=[r for r in rows if r['db']==db and r['bits']==bits and r['cap_j']==cap]
            assert len(group)==10
            ref=next(r for r in fixed if r['db']==db and r['bits']==bits and r['cap_j']==cap)
            oldhead=next(r for r in group if r['head']==protocol['previous_fixed_head'])
            for name,value in oldhead['methods'].items():
                ov=ref['methods'][name]
                assert (value is None)==(ov is None)
                if value:
                    assert abs(value['symbols']-ov['symbols'])<=1
                    reduction.append(dict(db=db,bits=bits,cap_j=cap,method=name,symbol_difference=value['symbols']-ov['symbols']))
            selected={}
            for method in group[0]['methods']:
                candidates=[]
                for row in group:
                    value=row['methods'][method]
                    if value:
                        assert value['peak_power_w']<=P*(1+4e-6)
                        assert value['total_error_bound']<=eps*(1+4e-6)
                        assert value['energy_j']<=(float('inf') if cap is None else cap)*(1+1e-6)
                        assert value.get('actual_expected_aggregate_rmse',0)<=value['total_error_bound']*(1+4e-6)
                        verified+=1
                        candidates.append(dict(value,head=row['head'],head_has_request=row['head_has_request'],
                            remote_requests=row['remote_requests'],remote_fu_payload_bits=row['remote_fu_payload_bits']))
                winner=best(candidates)
                selected[method]=winner
                ov=ref['methods'][method]
                if ov:
                    assert winner is not None and winner['symbols']<=ov['symbols']+1
            summary.append(dict(db=db,bits=bits,cap_j=cap,methods=selected))
report=dict(scope=protocol['scope'],head_selected_cases=summary,
    checks=dict(fixed_head_reduction=reduction,returned_designs_verified=verified,head_selection_not_worse=True),
    limitations=protocol['limitations'])
write('aggregate_report.json',report)
write('finished.json',dict(cases=len(rows),runtime_s=time.time()-t0,
    checks='All returned designs satisfy raw RMS, individual power and chosen energy constraints; fixed-head reduction verified',
    original_backend_unchanged=hashlib.sha256(source.read_bytes()).hexdigest()==protocol['backend_sha256']))
