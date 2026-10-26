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
OUT=ROOT/'results/overnight_energy_v2/interference_oracle_v1'
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
original_tree=ast.Module(body=prefix,type_ignores=[])
original_ns={'__name__':'unchanged_latency_backend','__file__':str(source)}
exec(compile(original_tree,str(source),'exec'),original_ns)
# Isolated explicit model change; original file and training sources are untouched.
variant=ast.unparse(original_tree)
oldn='n = cp.norm(cp.hstack([1, self.H @ self.a]))'
assert variant.count(oldn)==1
variant=variant.replace(oldn,"self.mix = np.divide(vec, self.rho[:, None], out=np.zeros_like(vec), where=self.rho[:, None] > 0).T * self.H[None, :]\n        n = cp.norm(cp.hstack([1, self.mix @ self.a]))")
oldpower='(self.H @ a) ** 2'
assert variant.count(oldpower)==2,variant.count(oldpower)
variant=variant.replace(oldpower,'float(np.sum((self.mix @ a) ** 2))')
result_marker='result = dict('
assert variant.count(result_marker)==1
variant=variant.replace(result_marker,"assert float(np.sum((self.mix @ a) ** 2)) <= float((self.H @ a) ** 2) * (1 + 1e-10) + 1e-12\n        result = dict(analog_interference_exact_noise_units=float(np.sum((self.mix @ a) ** 2)), analog_interference_triangle_noise_units=float((self.H @ a) ** 2), ")
exec(compile(variant,'isolated_exact_interference_variant','exec'),ns)
ns['OUT']=OUT
V=ns['learning'];U=ns['unlearning'];ids=ns['ids'];eps=ns['total_eps']
W=ns['W'];P=ns['P'];D=ns['D'];BS=ns['bsearch'];best=ns['choose_best']
CVX=ns['ConvexOverlap'];tail=ns['noma_tail'];MAXL=ns['MAXL']
QUANTS=[6,8];DB=[0,10];DEADLINES=[512,640,800,1024,2048,8192]
protocol=dict(scope='Oracle exact-average analog interference using all raw normalized update directions; minimum found TX bound at common deadline; not a deployable scheme, FEC validation, or VLM performance',
    quantization_bits=QUANTS,channel_gain_db=DB,deadline_symbols=DEADLINES,
    quantizer_error_policy='Native float32 scale and quantizer supplied conservative error bounds. v3 used measured per-user error norms.',
    raw_update_rms_budget=eps,power_per_user_w=P,bandwidth_hz=W,heads=list(range(10)),previous_fixed_head=ns['head'],
    users=10,total_requests=int(np.count_nonzero(np.linalg.norm(U,axis=1))),
    fairness='Identical raw update target, RMS budget and power cap. Digital methods share canonical Qb FU. Strong AirComp+AirComp sends the raw FL+FU sum without quantization.',
    resource_scope='Same conservative RF waveform TX bound, pure uplink. Only analog interference term changed from norm-triangle bound to exact frozen-vector norm. Ideal SIC/CSI/coding and free oracle statistics.',
    optimization='Joint convex receiver/transmitter for fixed overlap. Restricted analog duration/payload split grid; optimized continuous OFDMA bandwidth.',
    adaptive='Lowest energy bound among full, nonzero partial and zero-overlap NOMA at same deadline; zero overlap disclosed.',
    limitations=['All ten candidate heads in one geometry, hypothetical gain shifts. No deletion guarantee from update error.',
                 'Each method chooses its own best head with identical constraints. Head local updates are exact and not uploaded.',
                 'Head-specific receiver, feedback and control overhead omitted: pure uplink RF/time screen only.',
                 'No network energy cap in this energy minimization; same physical per-user 0.32 W peak bound.',
                 'AirComp+AirComp recovers only the sum; no artificial failure penalty.',
                 'All normalized local analog update directions are available only to this offline oracle. Head acquisition, signaling, estimation error and feedback costs are NOT included.',
                 'This exact average interference in an ideal coding formula is not coded packet validation; waveform covariance and finite-block reliability remain unresolved.',
                 'All methods get the same oracle information. No digital interference exists in the raw-sum or separated analog phase, so those baselines are unchanged.',
                 'Objective remains the conservative TX energy bound; the previous expected-energy objective is not combined here.',
                 'Failure of this conservative bound is not proof that true MSE violates the target.'],
    backend_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    isolated_variant_sha256=hashlib.sha256(variant.encode()).hexdigest(),
    input_sha256=hashlib.sha256((ROOT/'results/overnight_energy_v2/short_regime_probe_v1/fixed_input_block.npz').read_bytes()).hexdigest())
write('protocol.json',protocol);write('started.json',dict(pid=os.getpid(),unix=time.time()))
t0=time.time();rows=[];audits=[]
# Collinear updates recover the original coherent bound exactly; opposite
# directions show why a sum can have much less energy than the norm bound.
vtest=np.zeros((2,D));vtest[:,0]=1
for space in [ns,original_ns]:
    space['codec_bound']=0.;space['codec_bias']=np.zeros(D)
a=CVX(np.array([20.,16.]),vtest,.5).solve(800,np.array([512.,512.]))
b=original_ns['ConvexOverlap'](np.array([20.,16.]),vtest,.5).solve(800,np.array([512.,512.]))
assert a is not None and b is not None
assert abs(a['energy_j']/b['energy_j']-1)<1e-6
assert abs(a['analog_interference_exact_noise_units']/a['analog_interference_triangle_noise_units']-1)<1e-10
vtest[1,0]=-1
mix=vtest.T
assert np.linalg.norm(mix@np.ones(2))==0 and np.sum(np.ones(2))**2==4

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
priorpath=ROOT/'results/overnight_energy_v2/head_energy_deadline_v1/aggregate_report.json'
prior=json.loads(priorpath.read_text())['head_selected_cases']
comparisons=[];same_model_reductions=0;violations=[]
for new in summary:
    old=next(r for r in prior if all(r[k]==new[k] for k in ['db','bits','deadline_symbols']))
    for method,a in old['methods'].items():
        b=new['methods'][method]
        if a is None:continue
        if b is None:
            violations.append(dict(case={k:new[k] for k in ['db','bits','deadline_symbols']},method=method,reason='lost old feasible candidate'))
            continue
        relative=b['energy_j']/a['energy_j']-1
        if relative>2e-5:violations.append(dict(case={k:new[k] for k in ['db','bits','deadline_symbols']},method=method,relative=relative))
        if method in ['airair_raw','air_ofdma','separate_noma','ofdma_only']:
            assert abs(relative)<2e-5,(method,relative)
            same_model_reductions+=1
        comparisons.append(dict(db=new['db'],bits=new['bits'],deadline_symbols=new['deadline_symbols'],method=method,
            previous_bound_j=a['energy_j'],oracle_bound_j=b['energy_j'],relative_bound_improvement=-relative,
            previous_head=a['head'],new_head=b['head']))
checks.update(collinear_reduces_to_original=True,opposite_direction_identity=True,
    same_model_reductions=same_model_reductions,compared_previous_candidates=len(comparisons),
    no_worse_than_old_feasible_solutions=len(violations)==0,violations=violations)
write('model_comparison.json',comparisons);write('validation.json',checks)
assert not violations,violations
assert not monotonic,monotonic
write('aggregate_report.json',dict(scope=protocol['scope'],head_selected_cases=summary,checks=checks,limitations=protocol['limitations']))
write('finished.json',dict(cases=len(rows),runtime_s=time.time()-t0,checks=checks))
