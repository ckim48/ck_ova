from pathlib import Path
import subprocess,sys,os,json,datetime
r=Path(sys.argv[1]);s=Path(__file__).parent;o=r/'results/overnight_energy_v2/receiver_control_v1';o.mkdir(exist_ok=True)
def run(stage):
 return subprocess.run([sys.executable,'-u',str(s/'receiver_pilot.py'),'--project',str(r),'--stage',stage],cwd=r,check=True)
run('prepare')
with (o/'radio.log').open('x') as log:
 radio=subprocess.Popen([sys.executable,'-u',str(s/'receiver_radio.py'),'--project',str(r)],stdout=log,stderr=subprocess.STDOUT,cwd=r)
 (o/'radio_pid.json').write_text(json.dumps(dict(pid=radio.pid,started_at=datetime.datetime.now().isoformat())))
run('ideal')
radio.wait()
if radio.returncode:raise RuntimeError('Radio diagnostic failed; inspect log.')
# Evaluate primary Hybrid/OMA and the lowest-energy budget-eligible feedback candidate only.
# Selection uses modeled ENERGY before new model inference, never test outputs.
import numpy as np
rows=json.loads((o/'radio_results.json').read_text())
feedback=[x for x in rows if x.get('role')=='physical_model' and x['name'].startswith('feedback_') and x.get('status')=='feasible' and x.get('eligible_same_application_budget')]
names=['hybrid','oma_grid']
if feedback:names.append(min(feedback,key=lambda x:x['energy_bound_j'])['name'])
(o/'physical_selection.json').write_text(json.dumps(dict(names=names,criterion='minimum modeled energy among same-budget feedback candidates; fixed before physical inference'),indent=2))
with np.load(o/'physical_updates.npz') as z:
 selected={k:z[k].copy() for k in z.files if any(k.startswith(n+'_noise') for n in names)}
np.savez_compressed(o/'physical_updates_all.npz',**{k:v.copy() for k,v in np.load(o/'physical_updates.npz').items()})
np.savez_compressed(o/'physical_updates.npz',**selected)
run('physical')
(o/'completed.json').write_text(json.dumps(dict(status='completed',at_kst=datetime.datetime.now().isoformat())))
