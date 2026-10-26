"""Equal-input/time CPU isolation requested by user; no training/source changes."""
from pathlib import Path
import sys,os,time,json,hashlib,concurrent.futures,multiprocessing
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tools/overnight_energy_v2/adaptive_ofdma_v2'))
import adaptive_radio as radio
import numpy as np
OUT=ROOT/'results/overnight_energy_v2/noma_ofdma_priority_v1'

def worker(arg):
 from threadpoolctl import threadpool_limits
 with threadpool_limits(limits=1):return case(*arg)

def case(bits,count,mode):
 started=time.time();z=np.load(ROOT/'results/overnight_energy_v2/retrieval_fourway_v2/fixed_updates.npz')
 c=json.loads((ROOT/'results/overnight_energy_v2/retrieval_long200_v3/plan.json').read_text())['config']
 c=dict(c,quantization_bits=bits,symbols_per_coordinate=count,ofdma_subcarriers=128)
 l,u,q,delta,payload,rho,dims,total=radio.make_inputs(z['learning'],z['unlearning'],c)
 h=z['channels'];head=3;users=np.delete(np.arange(len(h)),head);H=h[users,head,None];W=c['symbol_rate_hz']
 eps=max(c['absolute_tolerance_floor'],c['relative_tolerance']*float(np.linalg.norm(rho.sum(0))))
 qb=float(np.linalg.norm(delta[users].sum(0)));re=eps-qb
 out=dict(bits=bits,frame_count=count,mode=mode,head=head,coordinates=z['learning'].shape[1],uplink_s=float(total.sum()/W),epsilon=eps,quantization_error_bound=qb,status='no_feasible_design_found')
 if mode in ['adaptive_noma','aircomp_ofdma']:
  radio.SPLITS=(0.,.025,.05,.1,1/7,.25,1.)
  d=radio.design(head,h,l,u,q,delta,payload,rho,dims,total,eps,c,mode)
  if d is None:return dict(out,wall_s=time.time()-started)
  a=d['analog'];la=d['analog_symbols'];meta=d['metadata'];bound=d['energy_j'];expected=d['expected_j']
  eta=meta.get('requested_overlap_payload_fraction',0.);au=meta.get('audit',{})
 else:
  eta=0. if mode=='separate_fixed_analog' else (1. if mode=='full_noma' else .025)
  la=total if mode=='full_noma' else 2*dims;lt=total-la;bo=np.ceil(eta*payload);bt=payload-bo
  ct=radio.ov.received_coefficients(bt[users],lt,h[users,head])
  if ct is None:return dict(out,wall_s=time.time()-started)
  pt=ct/(H*H)
  if np.max(pt)>c['power_limit_w']:return dict(out,wall_s=time.time()-started)
  a=radio.analog_design(head,h,rho,bo,dims,la,re,c)
  if a is None:return dict(out,wall_s=time.time()-started)
  fr=np.divide(bo[users],payload[users],out=np.zeros_like(bo[users]),where=payload[users]>0)
  au=radio.ov.audit(a.analog_rms,a.digital_sqrt_power**2,pt,a.receiver,h[users,head],rho[users],payload[users],la,total,fr,dims,re,c)
  if not au['passed']:return dict(out,wall_s=time.time()-started,audit=au)
  bound=au['tx_energy_bound_j'];expected=au['tx_energy_expected_j'];meta={}
 out.update(status='feasible',energy_bound_j=bound,expected_energy_j=expected,raw_error_bound=float(a.error_bound+qb),analog_s=float(la.sum()/W),overlap_fraction=eta,selected_boundary=('separate' if eta==0 else 'full' if mode=='full_noma' else 'partial'),audit=au)
 aa=a.analog_rms;ia=(H*aa).sum(0)**2
 if mode!='aircomp_ofdma':
  po=a.digital_sqrt_power**2
  out['overlap_digital_without_analog_interference_j']=float((la/W*po/(1+ia)).sum())
  out['extra_digital_for_analog_interference_j']=float((la/W*po*ia/(1+ia)).sum())
  out['cross_term_bound_j']=float((la/W*2*aa*np.sqrt(po)).sum())
 out['analog_interference_to_noise_bound']={k:float(v) for k,v in zip(['min','median','max'],[ia.min(),np.median(ia),ia.max()])}
 unit=np.divide(l[users],rho[users,:,None],out=np.zeros_like(l[users]),where=rho[users,:,None]>0)
 exact_i=np.sum(((H*aa)[:,:,None]*unit).sum(0)**2,axis=1)
 out['conditional_actual_analog_interference_to_noise']={k:float(v) for k,v in zip(['min','median','max'],[exact_i.min(),np.median(exact_i),exact_i.max()])}
 assert out['raw_error_bound']<=eps*(1+3e-5)
 out['wall_s']=time.time()-started
 return out

def ledger():
 result={}
 for label,folder in [('partial_noma','proxy_objective_v2/proposed'),('aircomp_ofdma','ofdma_v1/learning/aircomp_oma')]:
  rows=[json.loads(p.read_text()) for p in sorted((ROOT/'results/overnight_energy_v2'/folder/'rounds').glob('*.json')) if int(p.stem)<=155]
  active=[r for r in rows if r['radio'].get('requests_in_update',0)>0]
  result[label]=dict(active_rounds=[r['round'] for r in active],active_tx_bound_j=sum(r['radio']['energy_bound_j'] for r in active),active_expected_tx_j=sum(r['radio']['expected_energy_j'] for r in active),active_uplink_s=sum(r['radio']['uplink_s'] for r in active),inactive_tx_bound_j=sum(r['radio']['energy_bound_j'] for r in rows if r not in active),target_tx_bound_j=rows[-1]['cumulative_tx_energy_bound_j'],target_uplink_s=rows[-1]['cumulative_uplink_s'],cross_term_bound_j=sum(r['radio'].get('phase_energy',{}).get('waveform_cross_term_bound_j',0.) for r in active))
 return result

def main():
 OUT.mkdir(parents=True,exist_ok=True);started=time.time()
 jobs=[(bits,cnt,mode) for bits in [8,6] for cnt in [62,64] for mode in ['fixed_positive_partial','separate_fixed_analog','adaptive_noma','aircomp_ofdma']]+[(b,64,'full_noma') for b in [8,6]]
 protocol=dict(unix=started,cases=jobs,fixed_head=3,reason='Head3 is common to both completed round21 designs and saved preflight; fixes head-selection confounding.',common='Same full20480 training snapshot, channels, raw RMS, per-user power and battery; all methods include FU quantization. No final reference/test selection.',scope='Physical fixed-update diagnosis, not a VLM trajectory. Adaptive CPU grid additionally includes legacy positive overlap fractions; queued source untouched.',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),snapshot_sha256=hashlib.sha256((ROOT/'results/overnight_energy_v2/retrieval_fourway_v2/fixed_updates.npz').read_bytes()).hexdigest())
 (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2));rows=[]
 with concurrent.futures.ProcessPoolExecutor(max_workers=6,mp_context=multiprocessing.get_context('spawn')) as pool:
  fs={pool.submit(worker,j):j for j in jobs}
  for f in concurrent.futures.as_completed(fs):
   r=f.result();rows.append(r);(OUT/'cpu_results.json').write_text(json.dumps(rows,indent=2));print(json.dumps({k:r.get(k) for k in ['bits','frame_count','mode','status','energy_bound_j','overlap_fraction']}),flush=True)
 report=dict(unix=time.time(),runtime_s=time.time()-started,protocol=protocol,completed_ledger=ledger(),cases=rows)
 (OUT/'report.json').write_text(json.dumps(report,indent=2));print('PRIORITY_DIAGNOSIS_COMPLETE',flush=True)

if __name__=='__main__':main()
