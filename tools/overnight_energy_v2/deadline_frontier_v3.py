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
OUT=ROOT/'results/overnight_energy_v2/deadline_frontier_v3'
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
QUANTS=[4,6,8];DB=[0,10];CAPS=[.0001,.0003,.001,.003,None]
protocol=dict(scope='one frozen real 128-coordinate block; not VLM performance or convergence',
    quantization_bits=QUANTS,channel_gain_db=DB,network_energy_caps_j=CAPS,
    raw_update_rms_budget=eps,power_per_user_w=P,bandwidth_hz=W,head=ns['head'],
    users=10,remote_requests=int(np.count_nonzero(np.linalg.norm(U[ids],axis=1))),
    fairness='Identical raw update target, RMS budget and power cap. Digital methods share canonical Qb FU. Strong AirComp+AirComp sends the raw FL+FU sum without quantization.',
    resource_scope='RF TX conservative bound; uplink occupation only. Ideal SIC/CSI/coding.',
    optimization='Joint convex receiver/transmitter for fixed overlap. Restricted analog duration/payload split grid; optimized continuous OFDMA bandwidth.',
    adaptive='Minimum among full, nonzero partial and zero-overlap NOMA. Choosing zero overlap is allowed and disclosed.',
    limitations=['One head and geometry, hypothetical gain shifts. No deletion guarantee from update error.',
                 'Power-only removes the chosen network energy cap, not the physical per-user power limit.',
                 'AirComp+AirComp recovers only the sum; no artificial failure penalty.'],
    backend_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    input_sha256=hashlib.sha256((ROOT/'results/overnight_energy_v2/short_regime_probe_v1/fixed_input_block.npz').read_bytes()).hexdigest())
write('protocol.json',protocol);write('started.json',dict(pid=os.getpid(),unix=time.time()))
t0=time.time();rows=[];audits=[]
def encode(x,b):
    result=np.zeros_like(x);bits=np.zeros(len(x));levels=2**(b-1)-1
    for i,v in enumerate(x):
        if not np.any(v):continue
        scale=np.max(np.abs(v))/levels
        result[i]=np.rint(v/scale)*scale;bits[i]=32+D*b
    return result,bits

for db in DB:
    H=ns['base'][ids,ns['head']]*10**(db/20)
    # Raw analog baseline is independent of FU quantizer precision.
    ns['codec_bound']=0.;ns['codec_bias']=np.zeros(D)
    aa=CVX(H,(V+U)[ids],eps)
    aa_cases={cap:BS(lambda L:aa.solve(L,np.zeros(len(ids))),float('inf') if cap is None else cap,D) for cap in CAPS}
    for bits in QUANTS:
        q,b=encode(U,bits);ql,bl=encode(V,bits)
        qb=float(np.linalg.norm(q[ids]-U[ids],axis=1).sum())
        ns['codec_bound']=qb;ns['codec_bias']=(q[ids]-U[ids]).sum(0)
        qe=qb+float(np.linalg.norm(ql[ids]-V[ids],axis=1).sum())
        audits.append(dict(db=db,bits=bits,quantization_triangle_bound=qb,rms_budget=eps,
                           full_digital_bound=qe,quantizer_fits=qb<eps))
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
            row=dict(db=db,bits=bits,cap_j=cap,methods={k:ns['compact'](v) if v else None for k,v in methods.items()})
            rows.append(row);write('results.json',rows)
            print(json.dumps(dict(db=db,bits=bits,cap_j=cap,latency_ms={k:v['uplink_ms'] if v else None for k,v in methods.items()})),flush=True)
write('codec_audit.json',audits)
write('finished.json',dict(cases=len(rows),runtime_s=time.time()-t0,
    checks='All returned designs satisfy raw RMS, individual power and chosen energy constraints',
    original_backend_unchanged=hashlib.sha256(source.read_bytes()).hexdigest()==protocol['backend_sha256']))
