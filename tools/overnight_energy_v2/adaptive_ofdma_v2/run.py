"""User-authorized 2026-10-23 adaptive/Q6 study; actual OFDMA baselines.
Frozen predecessors are read-only. All four arms are fixed before new outcomes.
"""
from pathlib import Path
import sys,os,json,time,fcntl,hashlib,subprocess,importlib.util,traceback
import adaptive_radio as radio
import numpy as np
ROOT=radio.ROOT;HERE=Path(__file__).resolve().parent;OUT=ROOT/'results/overnight_energy_v2/adaptive_ofdma_v2'
OLD=ROOT/'results/overnight_energy_v2/proxy_objective_v2';OF=ROOT/'results/overnight_energy_v2/ofdma_v1/learning';CODEC=ROOT/'results/overnight_energy_v2/codec_quality_v1'
spec=importlib.util.spec_from_file_location('adaptive_proxy_core',ROOT/'tools/overnight_energy_v2/proxy_objective_v1/run.py');core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
core.OUT=OUT
core.METHODS=['proposed','aircomp_only','aircomp_oma','oma_only']
MAP={'hybrid':'adaptive_noma','aircomp_only':'aircomp_aircomp','aircomp_oma':'aircomp_ofdma','oma_only':'ofdma_only'}
core.LABELS={'proposed':'Adaptive NOMA+AirComp (Q6 FU)','aircomp_only':'AirComp+AirComp (raw sum)','aircomp_oma':'AirComp+OFDMA (Q6 FU)','oma_only':'OFDMA-only (Q6 FU)'}
def identity(pid):
 try:
  p=Path('/proc')/str(pid);s=(p/'stat').read_text().split(') ',1)[1].split()
  return dict(pid=pid,start=s[19],state=s[0],cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode())
 except FileNotFoundError:return None
def status(**x):core.write(OUT/'status.json',dict(unix=time.time(),pid=os.getpid(),**x))
def prefix(method):
 src=OF/'ofdma_initial20' if method=='oma_only' else core.SOURCE/'proposed'
 rows=json.loads((src/'prefix_rounds.json').read_text());rows=[r for r in rows if r['round']<=20]
 assert [r['round'] for r in rows]==list(range(1,21));return src,rows
def transport(l,u,h,c,method,seed):return radio.communicate(l,u,h,c,MAP[method],seed)
def report(plan):
 threshold=.29304036883016427;methods=[]
 for m in core.METHODS:
  dest=OUT/m;rows=[json.loads(p.read_text()) for p in sorted((dest/'rounds').glob('*.json'))]
  evs=[r for r in rows if r['round']>=25 and 'evaluation' in r];hit=[];target=None
  for r in evs:
   e=r['evaluation'];worst=max(v['KL_retrain_to_candidate'] for v in e['per_request'].values() if v['images'])
   passed=(e['retained']['mR']>=threshold and e['forget']['KL_retrain_to_candidate']<=.01 and worst<=.01 and r['request_algorithm_completed'])
   hit=hit+[r] if passed else []
   if len(hit)==3 and target is None:target=dict(entry_round=hit[0]['round'],confirmation_round=r['round'],energy_j=r['cumulative_tx_energy_bound_j'],uplink_s=r['cumulative_uplink_s'])
  manifest=json.loads((dest/'manifest.json').read_text()) if (dest/'manifest.json').exists() else dict(status='pending_or_running')
  methods.append(dict(method=m,label=core.LABELS[m],status=manifest['status'],completed_rounds=rows[-1]['round'] if rows else 20,
   strict_joint_target=target,last_evaluation=evs[-1]['evaluation'] if evs else None,
   costs={k:rows[-1][k] for k in ['cumulative_tx_energy_bound_j','cumulative_uplink_s']} if rows else None))
 core.write(OUT/'summary.json',dict(methods=methods,target=dict(retained_mR=threshold,mean_and_worst_KL=.01,consecutive_checks=3,check_spacing=5),
  limits='Single seed, pure uplink, TX waveform bound, no receiver/local compute cost. Incomplete methods not ranked. MIA/KL not a deletion guarantee.'))
def verify(protocol):
 for p,d in protocol['source_hashes'].items():assert core.sha(ROOT/p)==d,p
 plan=json.loads((core.SOURCE/'plan.json').read_text())
 for p,d in plan['sources'].items():assert core.sha(ROOT/p)==d,p
 for p,d in plan['frozen_engine_hashes'].items():assert core.sha(core.FROZEN/p)==d,p
 for p,d in plan['runner_hashes'].items():assert core.sha(core.LONG/p)==d,p
 return plan
def prepare():
 OUT.mkdir(parents=True,exist_ok=True);pre=json.loads((OUT/'preflight.json').read_text());assert pre['passed'] and not pre['small'] and pre['coordinates']==20480
 for n,d in pre['sources'].items():assert core.sha(HERE/n)==d,n
 paths=list(HERE.glob('*.py'))+[ROOT/'tools/overnight_energy_v2/ofdma_v1/ofdma.py',ROOT/'tools/overnight_energy_v2/retrieval_fourway_v2/fast_solver.py',ROOT/'tools/overnight_energy_v2/partial_overlap_tx_v1/overlap_solver.py',ROOT/'tools/overnight_energy_v2/proxy_objective_v1/run.py',core.SOURCE/'plan.json',core.SOURCE/'records.json',OF/'protocol.json',OLD/'proxies/analog/round_020.pt',OF/'proxies/oma/round_020.pt']
 pred=identity(4127621)
 if pred:assert pred['start']=='876222186' and 'codec_quality_v1/run.py' in pred['cmd']
 p=OUT/'protocol.json'
 if p.exists():protocol=json.loads(p.read_text())
 else:
  protocol=dict(version='adaptive_ofdma_v2',registered_unix=time.time(),authorization='User on 2026-10-23 explicitly chose adaptive partial overlap plus reduced FU payload and required actual OFDMA.',
   seed=0,max_rounds=200,first_stage=60,methods=core.LABELS,physical_methods=MAP,
   fu_quantization_bits=6,analog_counts=list(radio.ANALOG_COUNTS)+['full_frame'],overlap_payload_splits=list(radio.SPLITS),
   selection='Communication-only minimum TX energy per round over all ten heads and the predeclared time/payload grid. Integer overlap bits. Zero-overlap boundary allowed and explicitly reported; no validation/reference-based radio selection.',
   common_quality='RMS bound relative to the RAW FL+FU sum, including per-head digital FU quantization. OFDMA-only includes FL+FU quantization. AirComp+AirComp transmits the strong raw sum with no forced quantization.',
   ofdma='Actual 128 orthogonal subcarriers, simultaneous users, bandwidth-scaled noise, total power cap per user, optimized integer allocation and FL/FU time split. No TDMA helper is called.',
   grid_scope='Adaptive global frame split per round, not unrestricted continuous/per-block joint optimum. OFDMA uses its existing larger phase grid.',
   resource='Same channels, W, power, battery, coherence and 64d frame ceiling. Energy-minimizing frame comparison; no claim of minimum possible completion delay. Local compute/SIC time excluded for every arm.',
   objective='Same frozen retained CE + retained relation KL + forget-proxy KL for all four arms. No test/final-reference tuning. Raw quality accounting is a versioned change from older canonical-Q8 experiments.',
   prefix='Reuse exactly matching no-FU analog prefix/proxy for analog arms and verified OFDMA-only prefix/proxy for digital arm; charge all original costs.',
   progression='Run every feasible arm to60 then200 with fixed settings. Earlier codec screen is diagnostic, never a final-reference-based gate or tuning signal. Stop a physically infeasible/numerically failed arm and retain all failure records.',
   queue='Wait for existing codec predecessor after existing MemPrune reference; shared GPU lock, one GPU, no SIGSTOP or frozen edits. Registered before noon and explicitly authorized in the current user turn.',
   predecessor=pred,source_hashes={str(p.relative_to(ROOT)):core.sha(p) for p in paths})
  core.write(p,protocol)
 verify(protocol);return protocol
def main():
 protocol=prepare()
 if '--prepare' in sys.argv:print('PREPARED');return
 claim=(OUT/'queue.lock').open('a');fcntl.flock(claim,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (OUT/'started.json').exists(),'Duplicate GPU campaign'
 core.write(OUT/'queue_registered.json',dict(unix=time.time(),pid=os.getpid(),identity=identity(os.getpid()),predecessor=protocol['predecessor']))
 status(stage='waiting_for_codec',predecessor=protocol['predecessor'])
 pred=protocol['predecessor']
 while pred:
  now=identity(pred['pid'])
  if now is None or now['state']=='Z' or now['start']!=pred['start']:break
  time.sleep(30)
 end=CODEC/'finished.json'
 if not end.exists() or json.loads(end.read_text()).get('status')!='completed':raise RuntimeError('Predecessor not completed; no duplicate/replacement launch')
 lock=(ROOT/'results/overnight_energy_v2/association_objective_gpu.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
 for _ in range(30):
  if not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():break
  time.sleep(2)
 else:raise RuntimeError('Unexpected GPU user after lock')
 plan=verify(protocol);plan=dict(plan,config=dict(plan['config'],quantization_bits=6,ofdma_subcarriers=128))
 core.prefix=prefix;core.report=report
 core.write(OUT/'started.json',dict(unix=time.time(),pid=os.getpid()));status(stage='loading_model')
 from train import Engine
 from dataset import RetrievalData
 engine=Engine(RetrievalData(),dict(plan));records=json.loads((core.SOURCE/'records.json').read_text());forget=records[20]['forget'];channels=np.load(core.SOURCE/'channels.npy')
 proxies={}
 for name,path in [('analog',OLD/'proxies/analog'),('oma',OF/'proxies/oma')]:
  meta=json.loads((path/'complete.json').read_text());assert meta['status']=='completed'
  proxies[name]=(core.load_theta(path/'round_020.pt',engine.device),meta)
 for stop in [60,200]:
  for method in core.METHODS:core.physical(engine,plan,records,forget,proxies,transport,channels,method,stop)
  core.write(OUT/f'stage_{stop}_finished.json',dict(unix=time.time(),settings_unchanged=True))
 verify(protocol);report(plan);core.write(OUT/'finished.json',dict(unix=time.time(),status='completed'));status(stage='completed')
if __name__=='__main__':
 try:main()
 except Exception as e:status(stage='failed',error=repr(e));traceback.print_exc();raise
 finally:radio.close()
