"""CPU delay frontier under a common network TX energy cap; frozen training untouched."""
from pathlib import Path
import os,sys,json,time,hashlib,concurrent.futures,multiprocessing
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tools/overnight_energy_v2/adaptive_ofdma_v2'))
import adaptive_radio as r
import numpy as np
OUT=ROOT/'results/overnight_energy_v2/delay_budget_v1'
COUNTS=list(range(1,17))+[20,24,32,40,48,56,64]
ETAS=(0.,.025,.05,.1,.25,.5,1.)
CAPS=(.035,.05,.1,.2,None)
S=None;A=None

def init(analogs=None):
 global S,A
 from threadpoolctl import threadpool_limits
 threadpool_limits(limits=1)
 z=np.load(ROOT/'results/overnight_energy_v2/retrieval_fourway_v2/fixed_updates.npz')
 c=json.loads((ROOT/'results/overnight_energy_v2/retrieval_long200_v3/plan.json').read_text())['config']
 c=dict(c,quantization_bits=6,ofdma_subcarriers=128)
 l,u,q,delta,payload,rho,dims,total=r.make_inputs(z['learning'],z['unlearning'],c)
 head=3;users=np.delete(np.arange(len(z['channels'])),head);h=z['channels'][users,head]
 eps=max(c['absolute_tolerance_floor'],c['relative_tolerance']*float(np.linalg.norm(rho.sum(0))))
 qb=float(np.linalg.norm(delta[users].sum(0)))
 S=dict(c=c,l=l,u=u,q=q,delta=delta,payload=payload,rho=rho,dims=dims,head=head,users=users,h=h,channels=z['channels'],eps=eps,qb=qb,W=c['symbol_rate_hz'])
 A=analogs

def phases(n):
 # Both methods receive the same union of the actual OFDMA phase policy and power-of-two lengths.
 if n==1:return [1]
 return sorted(set([x for x in [1,2,4,8,16,32,n] if x<=n]+r.ofdma.phase_grid(np.array([128*n]),np.array([128]))))

def analog_job(key):
 n,eta=key;s=S;c=s['c'];la=n*s['dims']
 rho=np.linalg.norm(s['l']+s['u'],axis=-1) if eta=='raw' else s['rho']
 bo=np.zeros_like(s['payload']) if eta=='raw' else np.ceil(eta*s['payload'])
 eps=s['eps'] if eta=='raw' else s['eps']-s['qb']
 if eta!='raw':
  coeff=r.ov.received_coefficients(bo[s['users']],la,s['h'])
  if coeff is None or np.max(coeff/s['h'][:,None]**2)>c['power_limit_w']*(1+1e-10):return key,None
 a=r.analog_design(s['head'],s['channels'],rho,bo,s['dims'],la,eps,c)
 return key,a

def record(n,kind,energy,expected,err,la,eta,extra):
 return dict(frame_count=n,uplink_s=float(n*S['dims'].sum()/S['W']),kind=kind,energy_bound_j=float(energy),expected_energy_j=float(expected),raw_error_bound=float(err),analog_count=int(la),overlap_payload_fraction=eta,selected_boundary=('separate' if eta==0 else ('full' if la==n else 'partial')),**extra)

def frame_job(n):
 s=S;c=s['c'];users=s['users'];h=s['h'];H=h[:,None];dims=s['dims'];W=s['W'];total=n*dims;rows=[]
 # All feasible grid candidates are retained; energy-cap selection occurs only after enumeration.
 for la_count in phases(n):
  la=la_count*dims;lt=total-la
  for eta in ETAS:
   if la_count==n and eta!=1.:continue
   a=A.get((la_count,eta))
   if a is None:continue
   bo=np.ceil(eta*s['payload']);bt=s['payload']-bo
   coeff=r.ov.received_coefficients(bt[users],lt,h)
   if coeff is None:continue
   pt=coeff/(H*H)
   if np.max(pt)>c['power_limit_w']*(1+1e-10):continue
   fraction=np.divide(bo[users],s['payload'][users],out=np.zeros_like(bo[users]),where=s['payload'][users]>0)
   au=r.ov.audit(a.analog_rms,a.digital_sqrt_power**2,pt,a.receiver,h,s['rho'][users],s['payload'][users],la,total,fraction,dims,s['eps']-s['qb'],c)
   if not au['passed']:continue
   kind='separate_noma' if eta==0 else ('full_noma' if la_count==n else 'positive_partial_noma')
   rows.append(record(n,kind,au['tx_energy_bound_j'],au['tx_energy_expected_j'],a.error_bound+s['qb'],la_count,eta,dict(audit=au)))
  if la_count==n:continue
  a=A.get((la_count,0.))
  if a is None:continue
  dg=r.ofdma.digital(s['payload'][users],h,lt,c)
  if dg is None:continue
  eu=(la/W*a.analog_rms**2).sum(1)+dg['energy_by_user']
  if np.max(eu)>c['available_energy_j']*(1+1e-9):continue
  rows.append(record(n,'aircomp_ofdma',eu.sum(),eu.sum(),a.error_bound+s['qb'],la_count,0.,dict(access='integer_subcarrier_OFDMA',tones=128,rate_margin_bits=dg['minimum_rate_margin'],max_power_w=dg['max_power_w'],max_user_energy_j=float(eu.max()))))
 a=A.get((n,'raw'))
 if a is not None:rows.append(record(n,'aircomp_aircomp',a.energy_j,a.energy_j,a.error_bound,n,None,dict(access='raw_analog_sum',individual_FU_recovery=False)))
 for bits in [4,6,8,10,12,16]:
  ql,dl,pl=r.quantize(s['l'],bits,c['scale_metadata_bits']);qb=float(np.linalg.norm((dl[users]+s['delta'][users]).sum(0)))
  if qb>s['eps']:continue
  dg=r.ofdma.digital(pl[users]+s['payload'][users],h,total,c)
  if dg is None:continue
  rows.append(record(n,'ofdma_only',dg['energy_by_user'].sum(),dg['energy_by_user'].sum(),qb,0,0.,dict(access='integer_subcarrier_OFDMA',tones=128,learning_bits=bits,rate_margin_bits=dg['minimum_rate_margin'],max_power_w=dg['max_power_w'],max_user_energy_j=float(dg['energy_by_user'].max()))));break
 for z in rows:assert z['raw_error_bound']<=s['eps']*(1+3e-5) and np.isfinite(z['energy_bound_j'])
 return n,rows

def main():
 OUT.mkdir(parents=True,exist_ok=True);started=time.time();init()
 counts=sorted(set(x for n in COUNTS for x in phases(n)))
 jobs=[(n,e) for n in counts for e in ETAS]+[(n,'raw') for n in COUNTS]
 files=['tools/overnight_energy_v2/adaptive_ofdma_v2/adaptive_radio.py','tools/overnight_energy_v2/ofdma_v1/ofdma.py','tools/overnight_energy_v2/retrieval_fourway_v2/fast_solver.py','tools/overnight_energy_v2/partial_overlap_tx_v1/overlap_solver.py','results/overnight_energy_v2/retrieval_fourway_v2/fixed_updates.npz']
 hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files}
 protocol=dict(started_unix=started,objective='Minimize uplink duration subject to common network RF TX energy upper bound, per-user power/battery and raw RMS quality',caps_j=CAPS,total_counts=COUNTS,analog_counts=counts,overlap_payload_fractions=ETAS,bits=6,head=3,coordinates=20480,users=10,epsilon=S['eps'],quantization_bound=S['qb'],per_user_power_w=.32,per_user_battery_j=.5,bandwidth_hz=1e6,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),frozen_hashes=hashes,scope='One fixed actual update and head. Restricted integer-duration/phase/payload grid and local AO. Fastest validated grid design, NOT a global delay optimum or learning trajectory. No test/reference tuning.',fairness='Same actual 128-tone OFDMA, raw RMS including Q6 quantization for all digital FU. AirAir unquantized raw sum. Same phase grid for positive/separate NOMA and AirOFDMA. SIC/CSI/coding ideal. No local/SIC decoding time counted. None energy cap means peak and individual battery limits still apply.')
 (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2));analogs={};ctx=multiprocessing.get_context('spawn')
 with concurrent.futures.ProcessPoolExecutor(max_workers=6,mp_context=ctx,initializer=init) as pool:
  fs=[pool.submit(analog_job,j) for j in jobs]
  for f in concurrent.futures.as_completed(fs):
   k,a=f.result();analogs[k]=a
   (OUT/'status.json').write_text(json.dumps(dict(stage='analog_candidates',done=len(analogs),total=len(jobs),elapsed_s=time.time()-started)))
 rows=[]
 with concurrent.futures.ProcessPoolExecutor(max_workers=6,mp_context=ctx,initializer=init,initargs=(analogs,)) as pool:
  fs=[pool.submit(frame_job,n) for n in COUNTS]
  for f in concurrent.futures.as_completed(fs):
   n,rr=f.result();rows+=rr
   (OUT/'status.json').write_text(json.dumps(dict(stage='frame_candidates',latest=n,validated=len(rows),elapsed_s=time.time()-started)))
 methods=['full_noma','positive_partial_noma','separate_noma','adaptive_noma','aircomp_ofdma','aircomp_aircomp','ofdma_only'];frontier=[]
 for cap in CAPS:
  outcomes={}
  for method in methods:
   valid=[x for x in rows if (x['kind']==method or method=='adaptive_noma' and x['kind'] in methods[:3]) and (cap is None or x['energy_bound_j']<=cap*(1+1e-9))]
   outcomes[method]=min(valid,key=lambda x:(x['uplink_s'],x['energy_bound_j'])) if valid else None
  frontier.append(dict(cap_j=cap,methods=outcomes))
 assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
 report=dict(protocol=protocol,runtime_s=time.time()-started,analog_solved=len(analogs),analog_feasible=sum(a is not None for a in analogs.values()),candidate_count=len(rows),frontier=frontier,candidates=rows,frozen_hashes_unchanged=True)
 (OUT/'report.json').write_text(json.dumps(report,indent=2));(OUT/'finished.json').write_text(json.dumps(dict(unix=time.time(),runtime_s=report['runtime_s'],status='complete')))
 print(json.dumps(dict(done=True,runtime_s=report['runtime_s'],frontier=[dict(cap_j=f['cap_j'],methods={k:None if v is None else {x:v[x] for x in ['uplink_s','energy_bound_j','selected_boundary','overlap_payload_fraction','analog_count']} for k,v in f['methods'].items()}) for f in frontier])),flush=True)

if __name__=='__main__':main()
