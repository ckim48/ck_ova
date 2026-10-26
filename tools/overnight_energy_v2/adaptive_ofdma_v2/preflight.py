from pathlib import Path
import json,hashlib,time,sys
import adaptive_radio as radio
import numpy as np
ROOT=radio.ROOT;OUT=ROOT/'results/overnight_energy_v2/adaptive_ofdma_v2'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 OUT.mkdir(parents=True,exist_ok=True);started=time.time()
 c=json.loads((ROOT/'results/overnight_energy_v2/retrieval_long200_v3/plan.json').read_text())['config']
 z=np.load(ROOT/'results/overnight_energy_v2/retrieval_fourway_v2/fixed_updates.npz');l=z['learning'];u=z['unlearning'];h=z['channels']
 small='--small' in sys.argv
 if small:l=l[:,:256];u=u[:,:256]
 checks=radio.ofdma.checks();results={}
 for mode in ['aircomp_ofdma','adaptive_noma','aircomp_aircomp','ofdma_only']:
  v,r=radio.communicate(l,u,h,c,mode,48123);assert v is not None,mode
  assert r['error_bound']<=r['epsilon']*(1+3e-5)
  if mode in ['aircomp_ofdma','ofdma_only']:
   assert r['access']=='integer_subcarrier_OFDMA' and r['tones']==128
   counts=np.array(r['counts']);assert np.all(counts.sum(0)<=128)
   assert np.allclose(np.array(r['bandwidth_fractions']),counts/128)
   assert r['minimum_rate_margin_bits']>=-1e-6 and r['max_power_w']<=.32*(1+1e-10)
  if mode=='adaptive_noma':assert r['audit']['passed']
  if mode=='aircomp_aircomp':assert r['fu_payload_bits']==0 and r['quantization_error_bound']==0
  results[mode]=r
  print(json.dumps(dict(mode=mode,energy_j=r['energy_bound_j'],wall_s=r['optimizer_wall_s'],selected=r.get('selected_boundary'))),flush=True)
 # Verify exact no-FU prefix compatibility before reusing historical costs/model.
 import radio as frozen_radio
 zl=l[:,:256];zu=np.zeros_like(zl)
 av,ar=radio.communicate(zl,zu,h,c,'adaptive_noma',48123)
 bv,br=frozen_radio.communicate(zl,zu,h,c,'hybrid',48123)
 assert np.allclose(av,bv,rtol=1e-9,atol=1e-12)
 assert abs(ar['energy_bound_j']-br['energy_bound_j'])<1e-10
 assert abs(ar['uplink_s']-br['uplink_s'])<1e-12
 checks['analog_prefix_reuse_vector_and_cost_verified']=True
 # Independent noiseless power allocation tests above include a two-user case
 # OFDMA can complete under the same peak bound whereas TDMA cannot.
 out=dict(passed=True,small=small,coordinates=l.shape[1],runtime_s=time.time()-started,checks=checks,results=results,
  sources={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},
  training_snapshot_sha256=sha(ROOT/'results/overnight_energy_v2/retrieval_fourway_v2/fixed_updates.npz'),
  comparison='CPU fixed training updates only; not VLM learning, target time or general ranking.')
 (OUT/('small_checks.json' if small else 'preflight.json')).write_text(json.dumps(out,indent=2)+'\n')
 print('PREFLIGHT_PASSED',flush=True)
if __name__=='__main__':
 try:main()
 finally:radio.close()
