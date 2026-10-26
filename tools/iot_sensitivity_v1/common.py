"""Registered, paired one-factor sensitivity study; no result-based selection."""
from pathlib import Path
import copy, hashlib, json, os, sys, time
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'results/iot_sensitivity_v1_20261027'
SOURCE = ROOT / 'results/iot_uav_remoteclip_v4_event_mission'
GEOMETRY = ROOT / 'results/iot_uav_deployment_v3_global_iot'
METHODS = {'proposed_fu_guard_cliperase':'Proposed', 'heuristic_ova':'H-OVA',
           'separate_aircomp_periodic':'S-AC', 'aircomp_ofdma_periodic':'AC-OF',
           'sequential_digital_periodic':'S-OF+'}
EXTRA_DATASETS=['GeoText-1652','UAVReason']
AXES = {
    'bandwidth': ('Uplink bandwidth (MHz)', [.25,.5,1,2,4], 1),
    'downlink': ('Downlink rate (Mbit/s)', [.25,.5,1,2,4], 1),
    'power': ('UAV peak power (W)', [.08,.16,.32,.64,1.28], .32),
    'network_cap': ('Network TX budget (J)', [.01,.025,.05,.1,.2], .1),
    'channel_gain': ('Channel gain offset (dB)', [-12,-6,0,6,12], 0),
    'request_bursts': ('FU release groups', [3,5,9,15,45], 9),
    'fu_steps': ('FU steps per request', [1,3,5,7,9], 5),
    'speed': ('Cruise speed (m/s)', [2.5,3.75,5,7.5,10], 5),
    'frequency': ('CPU frequency (GHz)', [.75,1,1.5,2,3], 1.5),
    'collection_slot': ('IoT slot duration (s)', [.625,1.25,2.5,5,10], 2.5),
}
def read(p): return json.loads(Path(p).read_text())
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');temp.replace(p)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for c in iter(lambda:f.read(1<<20),b''):h.update(c)
    return h.hexdigest()
def state(dest=OUT,**kw):write(Path(dest)/'status.json',dict(unix=time.time(),pid=os.getpid(),**kw))
def cases():
    rows=[dict(id='baseline',axis=None,value=None)]
    for axis,(_,values,base) in AXES.items():
        rows += [dict(id=axis+'_'+str(v).replace('.','p').replace('-','m'),axis=axis,value=v) for v in values if v!=base]
    assert len(rows)==41
    return rows
def case_id(axis,value):
    return 'baseline' if value==AXES[axis][2] else axis+'_'+str(value).replace('.','p').replace('-','m')
def load_base():
    import importlib.util
    sys.path.insert(0,str(ROOT/'tools/iot_event_mission_v1'))
    spec=importlib.util.spec_from_file_location('event_sensitivity_source',ROOT/'tools/iot_event_mission_v1/run.py')
    event=importlib.util.module_from_spec(spec);spec.loader.exec_module(event)
    return event,event.recover.original.baseline_module()
def configure(case,plan,requests,geometry,deployment,parameters,channels):
    import numpy as np
    axis,v=case['axis'],case['value'];cap=.1;c=plan['config']
    if axis=='bandwidth':
        c['symbol_rate_hz']=v*1e6;c['noise_dbm']+=10*np.log10(v);channels=channels/np.sqrt(v)
    if axis=='downlink':parameters['communication']['downlink_rate_bps']=v*1e6
    if axis=='power':c['power_limit_w']=v
    if axis=='network_cap':cap=v
    if axis=='channel_gain':channels=channels*10**(v/20)
    if axis=='frequency':parameters['compute']['frequency_hz']=v*1e9
    if axis=='fu_steps':
        for r in requests:r['required_steps']=v
    if axis=='request_bursts':
        # Preserve IDs, cohort, original burst (FU random seed), and all 45 requests.
        slots=np.array([s*36+j for s in range(5) for j in range(1,28)])
        releases=slots[np.linspace(0,len(slots)-1,v).round().astype(int)]
        for group,(indices,release) in enumerate(zip(np.array_split(np.arange(len(requests)),v),releases)):
            for i in indices:
                requests[i].update(release_group=group,release_after_adaptation_FL=int(release),
                    release_before_total_FL_round=int(release)+21,mission_site=int(release)//36)
    if axis=='speed':
        legs=[(geometry['stops'][i+1]['start_s'] if i<4 else geometry['mission_end_s'])-s['end_s'] for i,s in enumerate(geometry['stops'])]
        t=0.
        for s,leg in zip(geometry['stops'],legs):
            s['start_s']=t;s['end_s']=t+60;t+=60+leg*5/v
        geometry.update(mission_end_s=t,cruise_speed_m_s=v)
        deployment['parameters']['cruise_speed_m_s']=v
    feasible=True
    if axis=='collection_slot':
        p=deployment['parameters'];p.update(device_slot_s=v,collection_s=4*v)
        for s in deployment['sites']:
            a=np.asarray(s['association_zero_based']);u=np.asarray(s['uav_positions_xy_m'])[a]
            d2=((np.asarray(s['iot_positions_xy_m'])-u)**2).sum(1)+p['altitude_m']**2
            b=p['bandwidth_per_uav_hz'];n=10**((p['noise_density_dbm_hz']+p['noise_figure_db']-30)/10)
            power=n*b*d2/p['reference_power_gain']*np.expm1(np.log(2)*np.asarray(s['iot_payload_bits'])/(b*v))
            feasible &= bool(power.max()<=p['iot_max_power_w']*(1+1e-9))
            s.update(collection_s=4*v,iot_pa_energy_j=float(power.sum()*v/p['iot_pa_efficiency']),
                iot_tx_circuit_energy_j=40*v*p['iot_tx_circuit_w'],uav_reception_energy_j=10*4*v*p['uav_rx_circuit_w'],
                iot_max_required_power_w=float(power.max()))
            s['communication_energy_j']=sum(s[k] for k in ['iot_pa_energy_j','iot_tx_circuit_energy_j','uav_reception_energy_j'])
    return cap,channels,feasible
