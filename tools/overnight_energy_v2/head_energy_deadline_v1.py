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
OUT=ROOT/'results/overnight_energy_v2/head_energy_deadline_v1'
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
QUANTS=[6,8];DB=[0,10];DEADLINES=[512,640,800,1024,2048,8192]
protocol=dict(scope='minimum found TX bound at common allowed uplink duration; one frozen 128-coordinate block, not VLM performance',
    quantization_bits=QUANTS,channel_gain_db=DB,deadline_symbols=DEADLINES,
    quantizer_error_policy='Native float32 scale and quantizer supplied conservative error bounds. v3 used measured per-user error norms.',
    raw_update_rms_budget=eps,power_per_user_w=P,bandwidth_hz=W,heads=list(range(10)),previous_fixed_head=ns['head'],
    users=10,total_requests=int(np.count_nonzero(np.linalg.norm(U,axis=1))),
    fairness='Identical raw update target, RMS budget and power cap. Digital methods share canonical Qb FU. Strong AirComp+AirComp sends the raw FL+FU sum without quantization.',
    resource_scope='RF TX conservative bound; uplink occupation only. Ideal SIC/CSI/coding.',
    optimization='Joint convex receiver/transmitter for fixed overlap. Restricted analog duration/payload split grid; optimized continuous OFDMA bandwidth.',
    adaptive='Lowest energy bound among full, nonzero partial and zero-overlap NOMA at same deadline; zero overlap disclosed.',
    limitations=['All ten candidate heads in one geometry, hypothetical gain shifts. No deletion guarantee from update error.',
                 'Each method chooses its own best head with identical constraints. Head local updates are exact and not uploaded.',
                 'Head-specific receiver, feedback and control overhead omitted: pure uplink RF/time screen only.',
                 'No network energy cap in this energy minimization; same physical per-user 0.32 W peak bound.',
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


def energy_best(xs):
    return min(xs,key=lambda v:v['energy_j']) if xs else None

for head in range(10):
    ids=np.delete(np.arange(10),head)
    for db in DB:
        H=ns['base'][ids,head]*10**(db/20)
        ns['codec_bound']=0.;ns['codec_bias']=np.zeros(D)
        aa=CVX(H,(V+U)[ids],eps)
        raw={T:aa.solve(T,np.zeros(len(ids))) for T in DEADLINES}
        for bits in QUANTS:
            q,b,delta=encode(U,bits);ql,bl,dl=encode(V,bits)
            qb=float(delta[ids].sum());qe=qb+float(dl[ids].sum())
            ns['codec_bound']=qb;ns['codec_bias']=(q[ids]-U[ids]).sum(0)
            audits.append(dict(head=head,db=db,bits=bits,quantization_bound=qb,rms_budget=eps))
            eng=CVX(H,V[ids],eps-qb) if qb<eps else None
            states=[]
            if eng:
                for la in ns['LAS']:
                    for eta in [0.]+ns['ETAS']:
                        bo=np.floor(b[ids]*eta+.5);a=eng.solve(la,bo)
                        if a:states.append((la,eta,bo,a))
            for T in DEADLINES:
                methods=dict(full_noma=eng.solve(T,b[ids]) if eng else None,airair_raw=raw[T])
                partial=[];separate=[];ofdma=[]
                for la,eta,bo,a in states:
                    if la>=T:continue
                    d=tail(H,b[ids]-bo,T-la)
                    if d:
                        value=dict(a,energy_j=a['energy_j']+d['energy_j'],
                            expected_energy_j=a['expected_energy_j']+d['energy_j'],
                            peak_power_w=max(a['peak_power_w'],d['peak_power_w']),
                            digital_tail_symbols=T-la,fu_overlap_fraction=float(bo.sum()/b[ids].sum()))
                        (separate if eta==0 else partial).append(value)
                    if eta==0:
                        d=ns['old'].ofdma(b[ids],H,T-la)
                        if d:
                            ofdma.append(dict(a,energy_j=a['energy_j']+d['energy'],
                                expected_energy_j=a['expected_energy_j']+d['energy'],
                                peak_power_w=max(a['peak_power_w'],float(d['power'].max())),
                                digital_tail_symbols=T-la))
                methods.update(partial_noma=energy_best(partial),separate_noma=energy_best(separate),air_ofdma=energy_best(ofdma))
                digital=ns['old'].ofdma((b+bl)[ids],H,T) if qe<=eps else None
                methods['ofdma_only']=dict(energy_j=digital['energy'],expected_energy_j=digital['energy'],
                    peak_power_w=float(digital['power'].max()),total_error_bound=qe) if digital else None
                choices=[dict(v,chosen_mode=k) for k,v in methods.items() if k in ['full_noma','partial_noma','separate_noma'] and v]
                methods['adaptive_noma']=energy_best(choices)
                for v in methods.values():
                    if not v:continue
                    v['symbols']=T;v['uplink_ms']=T/W*1000
                    assert v['peak_power_w']<=P*(1+4e-6)
                    assert v['total_error_bound']<=eps*(1+4e-6)
                    assert v.get('actual_expected_aggregate_rmse',0)<=v['total_error_bound']*(1+4e-6)
                rows.append(dict(head=head,head_has_request=bool(np.linalg.norm(U[head])>0),
                    remote_requests=int(np.count_nonzero(b[ids])),remote_fu_payload_bits=int(b[ids].sum()),
                    db=db,bits=bits,deadline_symbols=T,methods={k:ns['compact'](v) if v else None for k,v in methods.items()}))
            write('results.json',rows)
            print(json.dumps(dict(head=head,db=db,bits=bits,completed=len(rows))),flush=True)
write('codec_audit.json',audits)
summary=[]
for db in DB:
    for bits in QUANTS:
        for T in DEADLINES:
            group=[r for r in rows if r['db']==db and r['bits']==bits and r['deadline_symbols']==T]
            assert len(group)==10
            chosen={}
            for method in group[0]['methods']:
                candidates=[dict(row['methods'][method],head=row['head'],head_has_request=row['head_has_request'],
                    remote_requests=row['remote_requests']) for row in group if row['methods'][method]]
                chosen[method]=energy_best(candidates)
            summary.append(dict(db=db,bits=bits,deadline_symbols=T,methods=chosen))
assert len(rows)==240 and len(summary)==24
verified=sum(v is not None for r in rows for v in r['methods'].values())
# Duration increase cannot raise the true minimum RF energy. Record numerical/grid violations, not a global certificate.
monotonic=[]
for db in DB:
    for bits in QUANTS:
        g=sorted([r for r in summary if r['db']==db and r['bits']==bits],key=lambda r:r['deadline_symbols'])
        for method in g[0]['methods']:
            last=None
            for r in g:
                value=r['methods'][method]
                if last and value and value['energy_j']>last['energy_j']*(1+1e-5):
                    monotonic.append(dict(db=db,bits=bits,method=method,deadline=r['deadline_symbols'],previous=last['energy_j'],energy=value['energy_j']))
                if value:last=value
checks=dict(verified_designs=verified,monotonic_grid_violations=monotonic,head_count=10,
    backend_unchanged=hashlib.sha256(source.read_bytes()).hexdigest()==protocol['backend_sha256'])
assert checks['backend_unchanged']
write('aggregate_report.json',dict(scope=protocol['scope'],head_selected_cases=summary,checks=checks,limitations=protocol['limitations']))
write('finished.json',dict(cases=len(rows),runtime_s=time.time()-t0,checks=checks))
