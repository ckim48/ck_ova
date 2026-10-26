"""CPU energy identity check on frozen designs; no PHY or learning claim."""
from pathlib import Path
import ast, os, json, hashlib, time, fcntl
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np

ROOT=Path.cwd()
OUT=ROOT/'results/overnight_energy_v2/waveform_energy_audit_v1'
OUT.mkdir(parents=True,exist_ok=True)
lock=(OUT/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (OUT/'started.json').exists(), 'Do not repeat completed/in-progress diagnostic'
def write(name,obj):
    p=OUT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(obj,indent=2,allow_nan=False));t.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
source=ROOT/'tools/overnight_energy_v2/minimum_latency_v2/run.py'
prior=ROOT/'results/overnight_energy_v2/head_energy_deadline_v1/aggregate_report.json'
inputpath=ROOT/'results/overnight_energy_v2/short_regime_probe_v1/fixed_input_block.npz'
hashes={str(p.relative_to(ROOT)):sha(p) for p in [source,prior,inputpath]}
write('started.json',dict(unix=time.time(),source_sha256=sha(__file__),inputs_sha256=hashes))

old=json.loads(prior.read_text())
decomposition=[];reversals=[];checked=0
for row in old['head_selected_cases']:
    methods={}
    for name,v in row['methods'].items():
        if v is None:methods[name]=None;continue
        bound=v['energy_j'];mean=v['expected_energy_j'];gap=bound-mean
        assert mean>0 and gap>=-1e-12
        if name in ['airair_raw','air_ofdma','separate_noma','ofdma_only']:
            assert abs(gap)<1e-12
        methods[name]=dict(head=v['head'],tx_bound_j=bound,conditional_mean_j=mean,
            conservative_cross_term_j=max(0.,gap),cross_fraction_of_bound=max(0.,gap)/bound)
        checked+=1
    setting={k:row[k] for k in ['db','bits','deadline_symbols']}
    decomposition.append(dict(setting,methods=methods))
    full=methods.get('full_noma');ofdm=methods.get('air_ofdma')
    if full and ofdm and (full['tx_bound_j']-ofdm['tx_bound_j'])*(full['conditional_mean_j']-ofdm['conditional_mean_j'])<0:
        reversals.append(dict(setting,full_noma=full,air_ofdma=ofdm))

tree=ast.parse(source.read_text());prefix=[]
for node in tree.body:
    if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='protocol' for t in node.targets):break
    prefix.append(node)
ns={'__name__':'isolated_waveform_backend','__file__':str(source)}
exec(compile(ast.Module(body=prefix,type_ignores=[]),str(source),'exec'),ns)
ns['OUT']=OUT
from wireless import quantize
V=ns['learning'];U=ns['unlearning'];W=ns['W'];D=ns['D'];eps=ns['total_eps']
L=800;HEAD=3;DB=10;N=20000;SEED=20261023
ids=np.delete(np.arange(len(V)),HEAD)
# Real semiunitary DCT columns spread the analog vector across the L symbols.
t=np.arange(L)[:,None];j=np.arange(D)[None,:]
S=np.sqrt(2./L)*np.cos(np.pi*(t+.5)*j/L);S[:,0]/=np.sqrt(2.)
assert np.max(np.abs(S.T@S-np.eye(D)))<1e-12
results=[]
for bits in [6,8]:
    q,delta,payload=quantize(U[:,None,:],bits,32)
    q=q[:,0,:];delta=delta[:,0];payload=payload[:,0]
    ns['codec_bound']=float(delta[ids].sum())
    ns['codec_bias']=(q[ids]-U[ids]).sum(0)
    H=ns['base'][ids,HEAD]*10**(DB/20)
    design=ns['ConvexOverlap'](H,V[ids],eps-ns['codec_bound']).solve(L,payload[ids])
    assert design is not None
    original=next(r for r in old['head_selected_cases'] if r['db']==DB and r['bits']==bits and r['deadline_symbols']==L)['methods']['full_noma']
    assert original['head']==HEAD
    assert abs(design['energy_j']/original['energy_j']-1)<1e-6
    a=design['a'];p=design['po'];rho=np.linalg.norm(V[ids],axis=1)
    # A has squared norm L*a^2 per UAV; digital QPSK has norm squared L.
    coeff=np.divide(a*np.sqrt(L),rho,out=np.zeros_like(a),where=rho>0)
    A=(coeff[:,None]*V[ids])@S.T
    assert np.allclose(np.sum(A*A,axis=1),L*a*a,rtol=1e-10,atol=1e-18)
    per_mean=L/W*(a*a+p);per_bound=L/W*(a+np.sqrt(p))**2
    expected=float(per_mean.sum());bound=float(per_bound.sum())
    gap=2*L/W*float(np.sum(a*np.sqrt(p)))
    assert abs(bound-expected-gap)<1e-15
    assert abs(bound-design['energy_j'])<1e-15
    # Energy variance under independent zero-mean constant-modulus QPSK.
    variance=2/W**2*float(np.sum(p*np.sum(A*A,axis=1)))
    rng=np.random.default_rng(SEED+bits);values=[];max_ratio=0.;direct_error=0.
    for start in range(0,N,256):
        count=min(256,N-start)
        signs=rng.integers(0,2,size=(count,len(ids),L),dtype=np.int8)*2-1
        real=signs/np.sqrt(2.)
        cross=2/W*np.sqrt(p)[None,:]*np.einsum('bnl,nl->bn',real,A,optimize=True)
        per=per_mean[None,:]+cross
        assert np.all(per>=-1e-15) and np.all(per<=per_bound[None,:]*(1+1e-12)+1e-15)
        max_ratio=max(max_ratio,float(np.max(per/per_bound[None,:])))
        values.extend(per.sum(axis=1).tolist())
        if start==0:
            imag=(rng.integers(0,2,size=(16,len(ids),L),dtype=np.int8)*2-1)/np.sqrt(2.)
            wave=A[None,:,:]+np.sqrt(p)[None,:,None]*(real[:16]+1j*imag)
            direct=np.sum(np.abs(wave)**2,axis=2)/W
            direct_error=float(np.max(np.abs(direct-per[:16])))
            assert direct_error<1e-15
    values=np.asarray(values);mean=float(values.mean());samplevar=float(values.var(ddof=1))
    se=np.sqrt(variance/N);zscore=abs(mean-expected)/se
    assert zscore<6 and abs(samplevar/variance-1)<.06
    ofdm=next(r for r in old['head_selected_cases'] if r['db']==DB and r['bits']==bits and r['deadline_symbols']==L)['methods']['air_ofdma']
    results.append(dict(bits=bits,head=HEAD,db=DB,symbols=L,trials=N,tx_bound_j=bound,
        conditional_mean_j=expected,analog_component_j=float(L/W*np.sum(a*a)),
        digital_component_j=float(L/W*np.sum(p)),conservative_cross_term_j=gap,
        simulated_mean_j=mean,mean_standard_error_j=float(se),mean_zscore=float(zscore),
        theoretical_variance_j2=variance,simulated_variance_j2=samplevar,
        energy_quantiles_j=[float(x) for x in np.quantile(values,[.025,.5,.975])],
        max_actual_to_robust_per_user_bound=max_ratio,direct_norm_error_j=direct_error,
        full_bound_to_ofdma_ratio=bound/ofdm['energy_j'],full_mean_to_ofdma_ratio=expected/ofdm['energy_j']))

checks=dict(all_existing_designs_checked=checked,gram_orthonormality=True,
    zero_overlap_cross_terms_zero=True,bound_decomposition=True,direct_waveform_norm=True,
    conditional_mean_and_variance=True,all_sampled_energies_below_robust_bound=True,
    original_inputs_unchanged=all(sha(ROOT/p)==h for p,h in hashes.items()))
assert checks['original_inputs_unchanged']
report=dict(scope='Frozen robust-energy designs, algebraic and Monte Carlo TX waveform-energy audit; not a new optimized scheme or VLM experiment.',
    model='Independent zero-mean constant-modulus digital symbols, real semiunitary analog mapping; fixed analog update and powers. Conditional mean excludes the norm-triangle cross penalty.',
    formulas=dict(bound='L/W sum (a+sqrt(p))^2',conditional_mean='L/W sum (a^2+p)',
        conditional_variance='2 L/W^2 sum a^2 p'),
    inputs_sha256=hashes,source_sha256=sha(__file__),head_selected_decomposition=decomposition,
    full_vs_ofdma_sign_reversals=reversals,monte_carlo=results,checks=checks,
    limitations=[
        'Codes, packet contents, FEC and SIC are not simulated; random symbols verify energy identities only, not payload decodability or achievable rate.',
        'These are designs selected and optimized under the robust upper bound, not minima under a new expected-energy or chance-constrained objective.',
        'A zero expected cross term is conditional on the symbol ensemble; a particular payload waveform need not have zero cross term.',
        'Actual hardware, PA efficiency, receiver processing, circuit, feedback, local computation and flight energy are not measured.',
        'The original robust block-average power and error constraints are retained; no samplewise power guarantee is inferred from the mean alone.',
        'One frozen training-update block and channel geometry. No VLM quality, deletion guarantee, target-round cost, or overall algorithm victory follows.',
        'All original results are preserved, including Q6 disadvantage and raw-sum AirComp energy advantage.'
    ])
write('report.json',report);write('finished.json',dict(unix=time.time(),checks=checks))
print(json.dumps(dict(checks=checks,sign_reversals=len(reversals),monte_carlo=results)),flush=True)
