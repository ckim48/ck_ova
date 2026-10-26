"""CPU cost screen of noisy shared analog norm probes; no training changes."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import ast,json,time,hashlib,fcntl
import numpy as np
from scipy.stats import chi2
ROOT=Path.cwd();OUT=ROOT/'results/overnight_energy_v2/aggregate_norm_probe_v2'
OUT.mkdir(parents=True,exist_ok=True)
lock=(OUT/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (OUT/'started.json').exists(),'Refuse duplicate'
def write(n,x):
 p=OUT/n;t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False,default=lambda q:q.item() if isinstance(q,np.generic) else q));t.replace(p)
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
head=ns['head'];ids=np.delete(np.arange(10),head);H=channels[ids,head]*10**.5;W=ns['W'];P=ns['P']
bv=V.reshape(10,N,D);bu=U.reshape(10,N,D)
base_eps=.2*np.linalg.norm(bv,axis=2).sum(0);global_eps=float(np.linalg.norm(base_eps))
_,qv6,_=quantize(bv,6,32);_,qu6,_=quantize(bu,6,32)
reserved=(qv6[ids]+qu6[ids]).sum(0);allocated=reserved+(global_eps-np.linalg.norm(reserved))*base_eps/global_eps
assert np.linalg.norm(allocated)<=global_eps
RANKS=[16,32,64];FRACTIONS=[.1,1.];POLICIES=len(RANKS)*len(FRACTIONS)
DELTA=1e-3;tail=DELTA/(2*N*POLICIES);FEEDBACK_BITS=128
# New one-block feedback from head, symmetric normalized channel, ideal coding.
feedback_symbols=int(np.ceil(FEEDBACK_BITS/np.log2(1+P*float(np.min(H))**2)))
feedback_energy=feedback_symbols/W*P
proto=dict(scope='One stored training update, 160 blocks, fixed head7/+10dB. Six fixed whole-vector analog-probe protocols. CPU only; no winner chosen using validation, test or reference.',
 source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),backend_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
 blocks=N,block_dimension=D,head=head,channel_gain_db=10,block_total_protocol_symbols=[800,2048],bits=[6,8],ranks=RANKS,pilot_power_fractions=FRACTIONS,
 full_vector_confidence_failure_bound=DELTA,individual_tail_probability=tail,projection_seed=848,noise_seed=849,
 extra_feedback_bits_per_block=FEEDBACK_BITS,extra_feedback_symbols_per_block=feedback_symbols,extra_feedback_energy_j_per_block=feedback_energy,
 common_global_RMS_budget=global_eps,allocated_global_RMS=float(np.linalg.norm(allocated)),power_limit_w=P,
 derivation=[
 'R has independent N(0,1/p) entries. A fresh matrix is independent of the current local updates. The shared seed and existing update-norm reports are assumed available.',
 'Set gamma=sqrt(p P f)*min_i(H_i/rho_i)/||R||_2. Then every user has gamma^2 ||R v_i||^2/(p H_i^2)<=P f deterministically. This uses a public matrix norm, not individually decoded directions.',
 'Real received probe y=gamma R sum(v_i)+eta, eta~N(0,I/2). Put z=y/gamma and r_noise=sqrt(chi2_isf(tail,p)/(2 gamma^2)).',
 'With probability >=1-2tail, ||sum(v_i)||^2 <= p (||z||+r_noise)^2/chi2_ppf(tail,p). Clip this by the always valid (sum rho_i)^2. Union bound covers 160 blocks and all six reported policies.',
 'For exact-aligned main FL use kappa=sqrt(D/(2 Lmain))/(epsilon-FUquant), analog RMS powers a_i=kappa rho_i/H_i and receive scale1/kappa. Main mismatch is zero; analog noise RMS equals epsilon-FUquant.',
 'Use digital SIC powers p_i=co_i/H_i^2*(1+kappa^2 S_upper). Robust main TX power=(a_i+sqrt(p_i))^2. Check against0.32W. Probe, feedback and main transmissions are sequential and counted in the same protocol deadline.',
 'Each policy uses just its prescribed rank and power. No free online search across alternative probes. Comparative six-policy reporting is exploratory and multiplicity is covered by the confidence union bound.'
 ],limitations=[
 'Chance guarantee for digital SINR and conditional aggregation RMS differs from the original deterministic robust formulation. Failed decoding does not have a bounded unconditional MSE here.',
 'Exact CSI, synchronization and ideal SIC/coding. Finite-codeword reliability and pilot channel-estimation error are not included.',
 '128 extra feedback bits are a declared design accounting assumption, not a packet standard. Receiver energy/FEC/preamble costs unknown and omitted for every method; acquisition estimates remain optimistic.',
 'Feedback TX is reported separately from uplink TX. Pure uplink delay excludes feedback and local/SIC computation; total protocol radio delay includes added feedback.',
 'The baseline may use the entire same protocol deadline for uplink. Raw-sum AirComp supplies aggregate only; digital FU supplies individual requests. Neither fact establishes a VLM quality difference.',
 'Main exact alignment is a restricted feasible design, not a globally optimized chance-constrained transceiver. Negative findings do not rule out other norm-estimation designs.'
 ])
write('protocol.json',proto);write('started.json',dict(pid=os.getpid(),unix=time.time()))
t0=time.time();rp=np.random.default_rng(848);rn=np.random.default_rng(849);rows=[];probe_stats=[];valid=0;coverage_failures=0
enc={bits:quantize(bu,bits,32) for bits in [6,8]}
for block in range(N):
 v=bv[ids,block];rho=np.linalg.norm(v,axis=1);x=v.sum(0);true_S=float(x@x);worst_S=float(rho.sum()**2)
 for rank in RANKS:
  R=rp.normal(size=(rank,D))/np.sqrt(rank);opnorm=float(np.linalg.norm(R,ord=2));projections=v@R.T
  for fraction in FRACTIONS:
   gamma=np.sqrt(rank*P*fraction)*float(np.min(H[rho>0]/rho[rho>0]))/opnorm
   pilot_powers=gamma**2*np.sum(projections**2,axis=1)/(rank*H**2)
   assert float(pilot_powers.max())<=P*fraction*(1+1e-12)
   noise=rn.normal(size=rank)/np.sqrt(2);zz=R@x+noise/gamma
   radius=np.sqrt(chi2.isf(tail,rank)/2)/gamma
   upper=min(worst_S,rank*(float(np.linalg.norm(zz))+radius)**2/chi2.ppf(tail,rank))
   covered=upper>=true_S;coverage_failures+=int(not covered)
   pilot_energy=rank/W*float(pilot_powers.sum())
   probe_stats.append(dict(block=block,rank=rank,power_fraction=fraction,true_norm_squared=true_S,worst_norm_squared=worst_S,upper=upper,
     bound_over_worst=upper/worst_S,bound_over_true=upper/true_S,covered=bool(covered),pilot_energy_j=pilot_energy,pilot_max_power_w=float(pilot_powers.max())))
   for bits in [6,8]:
    quant,du,payload=enc[bits];qb=float(du[ids,block].sum());e=float(allocated[block]-qb);assert e>0
    for total in [800,2048]:
     L=total-rank-feedback_symbols
     row=dict(block=block,bits=bits,total_symbols=total,rank=rank,power_fraction=fraction,main_symbols=L,
      probe_uplink_energy_j=pilot_energy,feedback_tx_energy_j=feedback_energy,extra_feedback_s=feedback_symbols/W,
      pure_uplink_s=(total-feedback_symbols)/W,protocol_radio_s=total/W,confidence_covered=bool(covered),complete=False)
     if L>=D:
      kappa=np.sqrt(D/(2*L))/e;a=kappa*rho/H;co=ns['coefficients'](payload[ids,block],H,L)
      if co is not None:
       po=co/H**2*(1+kappa*kappa*upper);power=(a+np.sqrt(po))**2
       actual_required=co/H**2*(1+kappa*kappa*true_S)
       if covered:assert np.all(po>=actual_required*(1-1e-12))
       actual_rmse=float(np.sqrt(np.linalg.norm((quant[ids,block]-bu[ids,block]).sum(0))**2+e**2))
       assert actual_rmse<=allocated[block]*(1+1e-12)
       row.update(main_max_power_w=float(power.max()),main_tx_energy_j=L/W*float(power.sum()),conditional_RMS_bound=float(e+qb),actual_conditional_rmse=actual_rmse)
       if power.max()<=P*(1+1e-9) and covered:
        row.update(complete=True,uplink_tx_energy_j=pilot_energy+row['main_tx_energy_j'],total_tx_energy_j=pilot_energy+feedback_energy+row['main_tx_energy_j']);valid+=1
     rows.append(row)
 if (block+1)%40==0:print(json.dumps(dict(blocks=block+1,elapsed_s=time.time()-t0)),flush=True)
write('block_results.json',rows);write('probe_statistics.json',probe_stats)
reference=json.loads((ROOT/'results/overnight_energy_v2/block_time_allocation_v2/report.json').read_text())
summary=[]
for bits in [6,8]:
 for total in [800,2048]:
  baseline=next(r for r in reference['summary'] if r['bits']==bits and r['total_symbol_budget']==N*total)
  for rank in RANKS:
   for fraction in FRACTIONS:
    g=[r for r in rows if r['bits']==bits and r['total_symbols']==total and r['rank']==rank and r['power_fraction']==fraction]
    complete=all(r['complete'] for r in g);assert len(g)==N
    stat=[r for r in probe_stats if r['rank']==rank and r['power_fraction']==fraction]
    summary.append(dict(bits=bits,total_protocol_deadline_s=N*total/W,rank=rank,power_fraction=fraction,complete=complete,
     complete_blocks=sum(r['complete'] for r in g),uplink_tx_energy_j=sum(r['uplink_tx_energy_j'] for r in g) if complete else None,
     total_tx_energy_j=sum(r['total_tx_energy_j'] for r in g) if complete else None,
     feedback_tx_energy_j=N*feedback_energy,feedback_delay_s=N*feedback_symbols/W,pure_uplink_s=N*(total-feedback_symbols)/W,
     reduced_bound_blocks=sum(r['bound_over_worst']<1-1e-12 for r in stat),median_bound_over_worst=float(np.median([r['bound_over_worst'] for r in stat])),
     comparators={k:dict(complete=x['complete'],energy_j=x.get('energy_j')) for k,x in baseline['methods'].items()}))
# Independent synthetic simulation is a sanity check, not empirical certification of the rare-event bound.
rng=np.random.default_rng(850);mc_fail=0;mc_trials=2000
for j in range(mc_trials):
 rank=RANKS[j%3];R=rng.normal(size=(rank,D))/np.sqrt(rank);x=rng.normal(size=D);gamma=float(np.exp(rng.uniform(-2,4)))
 zz=R@x+rng.normal(size=rank)/np.sqrt(2)/gamma;rad=np.sqrt(chi2.isf(tail,rank)/2)/gamma
 upper=rank*(np.linalg.norm(zz)+rad)**2/chi2.ppf(tail,rank);mc_fail+=int(x@x>upper)
checks=dict(block_cases=len(rows),probe_cases=len(probe_stats),feasible_main_cases=valid,observed_bound_failures=coverage_failures,
 synthetic_trials=mc_trials,synthetic_coverage_failures=mc_fail,confidence_basis='Analytical Gaussian chi-square and union bounds, not finite Monte Carlo certification.',
 original_backend_unchanged=hashlib.sha256(source.read_bytes()).hexdigest()==proto['backend_sha256'])
assert checks['original_backend_unchanged'] and len(rows)==N*POLICIES*4
report=dict(protocol=proto,checks=checks,summary=summary)
write('report.json',report);write('finished.json',dict(runtime_s=time.time()-t0,checks=checks));print('COMPLETE',time.time()-t0,flush=True)
