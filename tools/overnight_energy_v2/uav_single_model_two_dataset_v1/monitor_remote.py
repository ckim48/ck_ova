"""Read-only JSON monitor. Never starts, stops or repairs a campaign."""
from pathlib import Path
import json,os,time
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'results/overnight_energy_v2/uav_single_model_two_dataset_v1'
def read(p):return json.loads(p.read_text()) if p.exists() else None
def identity(pid):
 p=Path('/proc')/str(pid)
 if not p.exists():return None
 try:return dict(pid=pid,start_ticks=p.joinpath('stat').read_text().split(') ',1)[1].split()[19],argv=p.joinpath('cmdline').read_bytes().replace(b'\0',b' ').decode())
 except FileNotFoundError:return None
def monitor():
 started=read(OUT/'started.json');expected=started.get('identity') if started else None
 actual=identity(expected['pid']) if expected else None
 result=dict(checked_unix=time.time(),expected=expected,actual=actual,identity_matches=actual==expected if expected else None,
  files={n:read(OUT/(n+'.json')) for n in ['status','started','launched','finished','failure','model_download_status']},datasets={})
 methods=['proposed_fu_guard_cliperase','separate_aircomp_periodic','aircomp_ofdma_periodic','sequential_digital_periodic']
 for dataset in ['rsitmd','rsicd']:
  dest=OUT/(dataset+'_seed0');data={n:read(dest/(n+'.json')) for n in ['status','started','finished','failure','actual_model_preflight','actual_FL_FU_preflight']}
  data['cache']={k:v for k,v in (read(dest/'feature_cache/manifest.json') or {}).items() if k in ['status','cache_images','cache_captions','image_server_seconds','text_server_seconds','cache_bytes','cache_sha256','frozen_encoder_no_grad']}
  data['prefix']={k:dict(progress=read(dest/k/'progress.json'),finished=read(dest/k/'finished.json')) for k in ['initial20','proxy20','reference200']}
  data['arms']={}
  for m in methods:
   arm=dest/m;latest=read(arm/'latest.json')
   data['arms'][m]=dict(started=read(arm/'started.json'),latest={k:v for k,v in (latest or {}).items() if k in ['sequence','completed_FL_rounds','kind','requests','cumulative_tx_energy_bound_j','cumulative_uplink_occupied_s']},manifest=read(arm/'manifest.json'),failure=read(arm/'failure.json'),verification=read(arm/'transmission_verification.json'))
  result['datasets'][dataset]=data
 return result
if __name__=='__main__':print('MONITOR_START'+json.dumps(monitor(),allow_nan=False)+'MONITOR_END',flush=True)
