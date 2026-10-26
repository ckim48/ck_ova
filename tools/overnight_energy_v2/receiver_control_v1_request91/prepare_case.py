from pathlib import Path
import hashlib,json,sys
import numpy as np
r=Path(sys.argv[1]);s=Path(__file__).parent;p=r/'results/overnight_energy_v2/receiver_control_v1';o=r/'results/overnight_energy_v2/receiver_control_v1_request91';o.mkdir(exist_ok=True)
sys.path[:0]=[str(s),str(r),str(r/'tools/overnight_energy_v2')]
from receiver_control import sensitivity_bound
from phase_split_grid import make_inputs
from receiver_pilot import sha,write
plan=json.loads((p/'plan.json').read_text());gpu=json.loads((p/'gpu_confirmation/plan.json').read_text())
plan.update(version='receiver_control_v1_request91',actual_round=91,checkpoint_sha256=gpu['cases'][1]['checkpoint_sha256'],source_vectors_sha256=gpu['cases'][1]['vector_sha256'],groups=gpu['cases'][1]['groups'],code_sha256={n:sha(s/n) for n in ['receiver_control.py','receiver_pilot.py','receiver_radio.py']},case_source_note='Legacy actual request91 from a different observed history, no independent-seed claim. Same fixed controller and radio settings as request61; radio copy differs only in output path.')
assert not (o/'plan.json').exists();write(o/'plan.json',plan)
with np.load(p/'gpu_confirmation/request91_prepared.npz') as z:arrays={k:z[k].copy() for k in z.files}
with np.load(r/'results/failed_round_diagnostic/updates.npz') as z:arrays.update(unlearning=z['unlearning'].copy(),channels=z['channels'].copy())
arrays['active']=np.flatnonzero(np.linalg.norm(arrays['unlearning'],axis=1)>1e-15)
np.savez_compressed(o/'prepared.npz',**arrays)
geometry=json.loads((p/'gpu_confirmation/request91_geometry.json').read_text())
inputs=make_inputs(arrays['learning'],arrays['unlearning'],plan['config'])
geometry.update(application_rmse_budget=float(.2*np.linalg.norm(inputs[5].sum(axis=0))),controller_lipschitz_bound=sensitivity_bound(arrays['quantized'][arrays['active']],arrays['gradient']))
write(o/'geometry.json',geometry);print(json.dumps(geometry),flush=True)
