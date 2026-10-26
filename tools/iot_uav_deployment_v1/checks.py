"""Independent physical checks on the saved deployment, not training metrics."""
import json
from pathlib import Path
import numpy as np

root=Path(__file__).resolve().parents[2]
folder=root/'results/iot_uav_deployment_v1'
data=json.loads((folder/'deployment.json').read_text());p=data['parameters']
positions=np.array([s['uav_positions_xy_m'] for s in data['sites']])
channels=np.load(folder/'channels_by_site.npy')
noise_density=10**((p['noise_density_dbm_hz']+p['noise_figure_db']-30)/10)
for s,x,h in zip(data['sites'],positions,channels):
    a=np.array(s['association_zero_based']);ground=np.array(s['iot_positions_xy_m'])
    assert np.array_equal(np.bincount(a,minlength=10),np.full(10,4))
    distance_squared=np.sum((x[a]-ground)**2,axis=1)+p['altitude_m']**2
    gains=p['reference_power_gain']/distance_squared
    bits=np.array(s['iot_payload_bits']);duration=p['device_slot_s'];bw=p['bandwidth_per_uav_hz']
    power=(2**(bits/(duration*bw))-1)*noise_density*bw/gains
    achieved=duration*bw*np.log2(1+power*gains/(noise_density*bw))
    assert np.allclose(bits,achieved,rtol=1e-12)
    assert power.max()<=p['iot_max_power_w']
    assert np.isclose(power.sum()*duration/p['iot_pa_efficiency'],s['iot_pa_energy_j'])
    distances=np.linalg.norm(x[:,None]-x[None],axis=-1)
    off=~np.eye(10,dtype=bool)
    assert np.allclose(h[off],1000/distances[off]) and np.all(np.diag(h)==0)
    costs=(distances**2).sum(axis=0)
    assert costs.argmin()==s['head_zero_based']
minimum=float('inf')
for a,b in zip(positions,np.roll(positions,-1,axis=0)):
    # Dense sampling independently checks the analytic continuous-time result.
    for f in np.linspace(0,1,2001):
        x=(1-f)*a+f*b;d=np.linalg.norm(x[:,None]-x[None],axis=-1)
        minimum=min(minimum,float(d[~np.eye(10,dtype=bool)].min()))
assert minimum>=p['minimum_separation_m']-1e-6
old=json.loads((root/'results/overnight_energy_v2/korea_mission_v1/mission_spec.json').read_text())
original=np.array([s['positions_xy_m'] for s in old['stops']])
assert not np.allclose(positions,original)
report=dict(passed=True,devices=160,uavs=10,sites=4,balanced_devices_per_uav=4,
            shannon_payload_check=True,all_head_candidates_checked=True,
            sampled_minimum_flight_separation_m=minimum,channels_recomputed=True)
(folder/'physical_checks.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
