"""Regression check against the failed first-site transport and a valid site."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/iot_vlm_v3_global_iot'))
import experiment as prior
import numpy as np
import overrun_radio
base=prior.original.baseline_module()
z=np.load(ROOT/'results/iot_uav_recovery_diagnostics/preflight_vectors.npz')
c=prior.read(prior.OUT/'rsitmd_seed0/plan.json')['config']
wrapped=overrun_radio.build(base)
l=z['FL'];zeros=np.zeros_like(l)
v,m=wrapped.analog(l,zeros,z['channels'][0],c,500000)
assert v is not None and m['overrun_fallback_used'] and not m['original_network_cap_satisfied']
assert np.isclose(m['energy_bound_j'],.11991409622145784)
assert np.isclose(m['network_cap_excess_j'],m['energy_bound_j']-.1)
assert np.isfinite(v).all()
v2,m2=wrapped.analog(l,zeros,z['channels'][1],c,500000)
v3,m3=base.base.SequentialRadio.analog(l,zeros,z['channels'][1],c,500000)
assert np.array_equal(v2,v3)
for key in ['head','frame_count','energy_bound_j','conditional_rmse','uplink_s']:assert m2[key]==m3[key],key
assert m2['original_network_cap_satisfied'] and not m2['overrun_fallback_used']
out=ROOT/'results/iot_uav_remoteclip_v3_energy_overrun';out.mkdir(exist_ok=True)
prior.write(out/'radio_checks.json',dict(passed=True,overrun_first_site=m,unchanged_feasible_site=m2))
print(json.dumps(dict(passed=True,first_site_energy_j=m['energy_bound_j'],excess_j=m['network_cap_excess_j'],feasible_site_unchanged=True)))
base.base.parent.common.base.close()
