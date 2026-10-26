"""Charge actual completion-driven hover once, including all service phases."""
import copy,json,math
from pathlib import Path
import numpy as np
from iot_energy import base as energy
ROOT=Path(__file__).resolve().parents[2]
PARAMETERS=copy.deepcopy(energy.PARAMETERS)
PARAMETERS['assumptions']=[x for x in PARAMETERS['assumptions'] if not x.startswith('Site hover is')]
PARAMETERS['assumptions'].append('Hover is recorded collection + compute + uplink + broadcast + scheduler wait. Depart immediately on required-work completion; no prescribed 60s minimum and no duplicated service costs.')
DEPLOYMENT=ROOT/'results/iot_uav_deployment_v3_global_iot/deployment.json'
PARAMETERS['iot_collection']=json.loads(DEPLOYMENT.read_text())['parameters']
read=energy.read

def prefix_cost(dataset):
    dataset=Path(dataset);plan=read(dataset/'plan.json')
    owner={int(i):u for u,part in enumerate(plan['partitions']) for i in part}
    lineage=read(dataset/'feature_cache/manifest.json')['lineage']
    image=energy.encoder_macs(50,768,12)+49*768*(32*32*3)+768*512
    text=energy.encoder_macs(77,512,12)+512*512
    macs=np.zeros(10)
    for record in lineage:
        if record['training_sample']:macs[owner[record['sample_id']]]+=len(record['frames'])*image+5*text
    e,t=energy.compute_cost(macs);computing=float(e.sum());delay=float(t.max());communication=0.
    for stage in ['initial20','proxy20']:
        for row in read(dataset/stage/'rounds.json'):
            e,t=energy.compute_cost(row['compute']['device_macs']);comm,down=energy.communication(row['radio'])
            computing+=float(e.sum())+PARAMETERS['communication']['head_processing_w']*row['radio']['uplink_s']
            communication+=comm;delay+=float(t.max())+row['radio']['uplink_s']+down
    return dict(energy_computing_j=computing,energy_communication_j=communication,prefix_delay_s=delay)

def method_energy(dataset,method,geometry):
    dataset=Path(dataset);directory=dataset/method
    manifest=read(directory/'manifest.json');assert manifest['status']=='completed_200FL_and_requests'
    timeline=read(directory/'mission_timeline.json');deployment=read(DEPLOYMENT)
    prefix=prefix_cost(dataset);computing=prefix['energy_computing_j'];communication=prefix['energy_communication_j']
    for path in sorted((directory/'work_attempts').glob('*.json')):
        row=read(path);assert row['status']=='transmitted'
        e,_=energy.compute_cost(row['compute']['device_macs']);computing+=float(e.sum())
    tx=[read(p) for p in sorted((directory/'transactions').glob('*.json'))]
    for row in tx:
        comm,_=energy.communication(row['radio']);communication+=comm
        computing+=PARAMETERS['communication']['head_processing_w']*row['radio']['uplink_s']
    collection_energy=sum(s['communication_energy_j'] for s in deployment['sites'])
    communication+=collection_energy
    travel_energy=0.
    for leg in timeline['legs']:
        a=np.asarray(geometry['stops'][leg['from_site']]['positions_xy_m'])
        b=np.asarray(geometry['stops'][leg['to_site']]['positions_xy_m'])
        speed=np.linalg.norm(b-a,axis=1)/leg['duration_s']
        travel_energy+=float(energy.propulsion_power(speed).sum())*leg['duration_s']
    hover=[s['hover_duration_s'] for s in timeline['sites']]
    assert math.isclose(timeline['mission_end_s'],sum(hover)+sum(l['duration_s'] for l in timeline['legs']))
    hover_energy=10*float(energy.propulsion_power(0))*sum(hover)
    flight=travel_energy+hover_energy
    requests=read(directory/'request_records.json');lat=[r['completed_s']-r['arrival_s'] for r in requests]
    energies=[r['radio']['energy_bound_j'] for r in tx]
    result=dict(method=method,energy_communication_j=communication,energy_computing_j=computing,
                energy_flight_j=flight,energy_travel_j=travel_energy,energy_hover_j=hover_energy,
                total_energy_j=communication+computing+flight,mission_delay_s=timeline['mission_end_s'],
                total_delay_s=prefix['prefix_delay_s']+timeline['mission_end_s'],prefix_delay_s=prefix['prefix_delay_s'],
                hover_by_site_s=hover,loss=manifest['evaluation']['retained']['loss'],mR=manifest['evaluation']['retained']['mR'],
                status='modeled_scenario_not_measured',iot_collection_communication_energy_j=collection_energy,
                iot_collection_seconds=sum(s['collection_duration_s'] for s in timeline['sites']),
                total_energy_scope='10 UAVs plus ground IoT transmission; recorded completion-driven mission',
                FU_mean_completion_latency_s=float(np.mean(lat)),FU_p95_completion_latency_s=float(np.percentile(lat,95)),
                FU_max_completion_latency_s=max(lat),FU_completed_requests=len(requests),
                original_network_tx_cap_j=.1,network_cap_overrun_transactions=sum(e>.1*(1+2e-6) for e in energies),
                network_cap_total_excess_j=sum(max(0.,e-.1) for e in energies),max_transaction_tx_energy_j=max(energies),
                network_cap_policy='soft, original-feasible preferred' if method=='sequential_digital_periodic' else 'original hard cap')
    assert all(math.isfinite(result[k]) and result[k]>=0 for k in ['total_energy_j','total_delay_s'])
    return result
