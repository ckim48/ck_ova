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
OUT=ROOT/'results/overnight_energy_v2/packet_overhead_v1'
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
QUANTS=[6,8];DB=[10];CAPS=[.0003];HEADERS=[0,32,128];FIXED_T=800
protocol=dict(extra_header_bits_per_nonempty_packet=HEADERS,header_assumption='Hypothetical framing/CRC/control sensitivity; no standard or hardware calibration. All digital schemes pay once per nonempty packet; partial FU pays for both nonempty fragments.',fixed_energy_comparison_symbols=FIXED_T,scope='one frozen real 128-coordinate block; not VLM performance or convergence',
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
                 'Head receiver energy and downlink/control signaling omitted; explicit digital framing bits counted in uplink.',
                 'Finite-blocklength coding and minimum decodable packet size remain unmodeled. Header values are sensitivity choices, not measurements.',
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


def e_best(xs):
    return min(xs,key=lambda x:x['energy_j']) if xs else None

energy_rows=[]
for head in range(10):
    ids=np.delete(np.arange(10),head)
    for db in DB:
        H=ns['base'][ids,head]*10**(db/20)
        ns['codec_bound']=0.;ns['codec_bias']=np.zeros(D)
        aa=CVX(H,(V+U)[ids],eps)
        raw_time=BS(lambda L:aa.solve(L,np.zeros(len(ids))),CAPS[0],D)
        raw_energy=aa.solve(FIXED_T,np.zeros(len(ids)))
        for bits in QUANTS:
            q,nb,delta=encode(U,bits);ql,bl,dl=encode(V,bits)
            qb=float(delta[ids].sum());qe=qb+float(dl[ids].sum())
            ns['codec_bound']=qb;ns['codec_bias']=(q[ids]-U[ids]).sum(0)
            eng=CVX(H,V[ids],eps-qb) if qb<eps else None
            for header in HEADERS:
                b=nb+header*(nb>0)
                dbits=nb+bl;dbits=dbits+header*(dbits>0)
                states=[]
                if eng:
                    for la in ns['LAS']:
                        for eta in [0.]+ns['ETAS']:
                            native_overlap=np.floor(nb[ids]*eta+.5)
                            native_tail=nb[ids]-native_overlap
                            bo=native_overlap+header*(native_overlap>0)
                            bt=native_tail+header*(native_tail>0)
                            assert np.all(bo+bt==nb[ids]+header*((native_overlap>0).astype(int)+(native_tail>0).astype(int)))
                            a=eng.solve(la,bo)
                            if a:states.append((la,eta,bo,bt,float(native_overlap.sum()/nb[ids].sum()),a))
                partial=[];separate=[];ofdma=[];ep=[];es=[];eo=[]
                for la,eta,bo,bt,frac,a in states:
                    def f_noma(L):
                        if L<=la:return None
                        d=tail(H,bt,L-la)
                        if d is None:return None
                        return dict(a,energy_j=a['energy_j']+d['energy_j'],
                            expected_energy_j=a['expected_energy_j']+d['energy_j'],
                            peak_power_w=max(a['peak_power_w'],d['peak_power_w']),
                            digital_tail_symbols=L-la,fu_overlap_fraction=frac,
                            framing_bits=int(header*((bo>0).sum()+(bt>0).sum())))
                    x=BS(f_noma,CAPS[0],la+1)
                    if x:(separate if eta==0 else partial).append(x)
                    x=f_noma(FIXED_T)
                    if x:(es if eta==0 else ep).append(x)
                    if eta==0:
                        def f_oma(L):
                            if L<=la:return None
                            d=ns['old'].ofdma(b[ids],H,L-la)
                            if d is None:return None
                            return dict(a,energy_j=a['energy_j']+d['energy'],
                                expected_energy_j=a['expected_energy_j']+d['energy'],
                                peak_power_w=max(a['peak_power_w'],float(d['power'].max())),
                                digital_tail_symbols=L-la,framing_bits=int(header*(b[ids]>0).sum()))
                        x=BS(f_oma,CAPS[0],la+1)
                        if x:ofdma.append(x)
                        x=f_oma(FIXED_T)
                        if x:eo.append(x)
                def f_digital(L):
                    if qe>eps:return None
                    d=ns['old'].ofdma(dbits[ids],H,L)
                    if d is None:return None
                    return dict(energy_j=d['energy'],expected_energy_j=d['energy'],
                        peak_power_w=float(d['power'].max()),total_error_bound=qe,
                        framing_bits=int(header*(dbits[ids]>0).sum()))
                tm=dict(full_noma=BS(lambda L:eng.solve(L,b[ids]),CAPS[0],D) if eng else None,
                    partial_noma=best(partial),separate_noma=best(separate),air_ofdma=best(ofdma),
                    airair_raw=raw_time,ofdma_only=BS(f_digital,CAPS[0],1))
                em=dict(full_noma=eng.solve(FIXED_T,b[ids]) if eng else None,
                    partial_noma=e_best(ep),separate_noma=e_best(es),air_ofdma=e_best(eo),
                    airair_raw=raw_energy,ofdma_only=f_digital(FIXED_T))
                for methods,pick in [(tm,best),(em,e_best)]:
                    methods['adaptive_noma']=pick([dict(v,chosen_mode=k) for k,v in methods.items() if k in ['full_noma','partial_noma','separate_noma'] and v])
                    for v in methods.values():
                        if not v:continue
                        assert v['peak_power_w']<=P*(1+4e-6)
                        assert v['total_error_bound']<=eps*(1+4e-6)
                        assert v.get('actual_expected_aggregate_rmse',0)<=v['total_error_bound']*(1+4e-6)
                for v in tm.values():
                    if v:assert v['energy_j']<=CAPS[0]*(1+1e-6)
                info=dict(head=head,db=db,bits=bits,extra_header_bits=header,remote_requests=int(np.count_nonzero(nb[ids])),remote_base_fu_bits=int(nb[ids].sum()))
                rows.append(dict(info,cap_j=CAPS[0],methods={k:ns['compact'](v) if v else None for k,v in tm.items()}))
                energy_rows.append(dict(info,deadline_symbols=FIXED_T,methods={k:ns['compact'](v) if v else None for k,v in em.items()}))
            write('results.json',rows);write('energy_results.json',energy_rows)
            print(json.dumps(dict(head=head,bits=bits,completed=len(rows))),flush=True)

assert len(rows)==60 and len(energy_rows)==60
def summarize(data,kind):
    out=[]
    for bits in QUANTS:
        for header in HEADERS:
            group=[r for r in data if r['bits']==bits and r['extra_header_bits']==header]
            methods={}
            for name in group[0]['methods']:
                values=[dict(r['methods'][name],head=r['head'],remote_requests=r['remote_requests']) for r in group if r['methods'][name]]
                methods[name]=(best if kind=='latency' else e_best)(values)
            out.append(dict(db=DB[0],bits=bits,extra_header_bits=header,methods=methods))
    return out

ts=summarize(rows,'latency');ee=summarize(energy_rows,'energy')
checked=0
old_t=json.loads((ROOT/'results/overnight_energy_v2/head_frontier_v1/results.json').read_text())
old_e=json.loads((ROOT/'results/overnight_energy_v2/head_energy_deadline_v1/results.json').read_text())
for data,old,kind in [(rows,old_t,'latency'),(energy_rows,old_e,'energy')]:
    for r in data:
        if r['extra_header_bits']!=0:continue
        ref=next(x for x in old if x['head']==r['head'] and x['bits']==r['bits'] and x['db']==r['db'] and (x.get('cap_j')==CAPS[0] if kind=='latency' else x['deadline_symbols']==FIXED_T))
        for name,v in r['methods'].items():
            ov=ref['methods'][name]
            assert (v is None)==(ov is None),(kind,name,r['head'],r['bits'])
            if v:
                if kind=='latency':assert abs(v['symbols']-ov['symbols'])<=1
                else:assert abs(v['energy_j']-ov['energy_j'])<=max(1e-10,abs(ov['energy_j'])*1e-5)
                checked+=1
for data,kind in [(ts,'latency'),(ee,'energy')]:
    for bits in QUANTS:
        seq=sorted([r for r in data if r['bits']==bits],key=lambda r:r['extra_header_bits'])
        for name in seq[0]['methods']:
            previous=None
            for r in seq:
                v=r['methods'][name]
                if previous and v:
                    assert (v['symbols']>=previous['symbols']-1 if kind=='latency' else v['energy_j']>=previous['energy_j']*(1-1e-5))
                previous=v
checks=dict(zero_header_reduction_designs=checked,header_bit_conservation=True,header_monotonicity=True,
    returned_latency_designs=sum(v is not None for r in rows for v in r['methods'].values()),
    returned_energy_designs=sum(v is not None for r in energy_rows for v in r['methods'].values()),
    original_backend_unchanged=hashlib.sha256(source.read_bytes()).hexdigest()==protocol['backend_sha256'])
assert checks['original_backend_unchanged']
write('aggregate_report.json',dict(latency=ts,energy_at_800us=ee,checks=checks,limitations=protocol['limitations']))
write('finished.json',dict(latency_cases=len(rows),energy_cases=len(energy_rows),runtime_s=time.time()-t0,checks=checks))
