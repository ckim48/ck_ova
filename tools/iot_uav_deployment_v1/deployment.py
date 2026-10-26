"""Constrained IoT-driven placement, followed by nominal packet-head selection.

The physical geometry is a declared scenario, not measured sensor telemetry.
Four-method FL/FU subsequently selects its own actual heads on these channels.
"""
from pathlib import Path
import json
import hashlib
import csv
import time
import numpy as np
from scipy.optimize import linear_sum_assignment, minimize

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/iot_uav_deployment_v1'
OLD = ROOT/'results/overnight_energy_v2/korea_mission_v1'
PARAMETERS = dict(seed=48, sites=4, uavs=10, devices_per_site=40,
                  altitude_m=50., minimum_separation_m=22., allowed_radius_m=220.,
                  cruise_speed_m_s=5., hover_s=60., collection_s=10.,
                  device_slot_s=2.5, bandwidth_per_uav_hz=1e5,
                  total_collection_bandwidth_hz=1e6, iot_max_power_w=.1,
                  noise_density_dbm_hz=-174., noise_figure_db=7.,
                  reference_power_gain=1e-6, pathloss_exponent=2.,
                  nominal_update_bits=20480*32, relay_bandwidth_hz=1e6,
                  relay_slot_s=.5, uav_max_power_w=.32,
                  iot_pa_efficiency=.35, uav_pa_efficiency=.35,
                  iot_tx_circuit_w=.01, uav_rx_circuit_w=.1,
                  maximum_outer_iterations=12,
                  objective='IoT uplink PA energy plus nominal UAV packet-relay PA energy; flight is constrained, not minimized by this placement objective.',
                  head_policy='Exact enumeration of ten nominal packet heads for placement; actual FL/FU heads are reselected by existing transport solvers.',
                  reference='https://arxiv.org/abs/1703.05401',
                  scope='Fixed-altitude local alternating optimization inspired by IoT deployment literature; not a reproduction or a global optimum claim.')


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def noise(bandwidth):
    return 10**((PARAMETERS['noise_density_dbm_hz']+PARAMETERS['noise_figure_db']-30)/10)*bandwidth


def minimum_transition_separation(a, b):
    i,j=np.triu_indices(10,1)
    d=a[i]-a[j];v=(b[i]-b[j])-d
    f=np.clip(-np.sum(d*v,axis=1)/np.maximum(np.sum(v*v,axis=1),1e-30),0,1)
    return np.linalg.norm(d+f[:,None]*v,axis=1)


def create_devices(centers):
    rng=np.random.default_rng(PARAMETERS['seed']);devices=[];bits=[]
    for center in centers:
        # Synthetic spatial demand hotspots, fixed before optimizer execution.
        angles=np.array([.2,1.8,3.4,5.0])+rng.uniform(-.2,.2,4)
        hotspots=center+np.column_stack([np.cos(angles),np.sin(angles)])*rng.uniform(65,125,(4,1))
        site=[]
        for i in range(40):
            while True:
                point=hotspots[i%4]+rng.normal(0,35,2)
                if np.linalg.norm(point-center)<=175 and (not site or np.linalg.norm(np.array(site)-point,axis=1).min()>7):
                    site.append(point);break
        devices.append(site)
        bits.append(rng.choice([250000.,500000.,750000.,1000000.],40).tolist())
    return np.array(devices),np.array(bits)


def coefficients(bits):
    p=PARAMETERS
    rate=bits/p['device_slot_s']
    iot=(np.expm1(np.log(2)*rate/p['bandwidth_per_uav_hz'])*noise(p['bandwidth_per_uav_hz'])
         /p['reference_power_gain']*p['device_slot_s']/p['iot_pa_efficiency'])
    relay=(np.expm1(np.log(2)*p['nominal_update_bits']/p['relay_slot_s']/p['relay_bandwidth_hz'])
           *noise(p['relay_bandwidth_hz'])/p['reference_power_gain']*p['relay_slot_s']/p['uav_pa_efficiency'])
    return iot,relay


def association(x, devices, weights):
    result=[]
    slots=np.repeat(np.arange(10),4)
    for points,ground,w in zip(x,devices,weights):
        cost=w[:,None]*np.sum((ground[:,None]-points[None])**2,axis=-1)
        r,c=linear_sum_assignment(cost[:,slots]);a=np.empty(40,dtype=int);a[r]=slots[c]
        assert np.array_equal(np.bincount(a,minlength=10),np.full(10,4))
        result.append(a)
    return np.array(result)


def choose_heads(x, relay):
    costs=relay*np.sum((x[:,:,None]-x[:,None,:])**2,axis=-1).sum(axis=1)
    return costs.argmin(axis=1),costs


def objective(x, devices, assign, weights, relay, heads, grad=False):
    residual=x[np.arange(4)[:,None],assign]-devices
    value=float(np.sum(weights*(np.sum(residual**2,axis=-1)+PARAMETERS['altitude_m']**2)))
    gradient=np.zeros_like(x)
    for s in range(4):
        np.add.at(gradient[s],assign[s],2*weights[s,:,None]*residual[s])
        delta=x[s]-x[s,heads[s]]
        value+=float(relay*np.sum(delta**2))
        gradient[s]+=2*relay*delta
        gradient[s,heads[s]]-=2*relay*delta.sum(axis=0)
    return (value,gradient) if grad else value


def constraints(x, centers, devices, assign, weights, relay, heads):
    p=PARAMETERS;sep=p['minimum_separation_m']
    values=[(p['allowed_radius_m']**2-np.sum((x-centers[:,None])**2,axis=-1)).ravel()/1e4]
    for a,b in zip(x,np.roll(x,-1,axis=0)):
        values.append((minimum_transition_separation(a,b)**2-sep**2)/1e4)
    distances=np.sum((x[np.arange(4)[:,None],assign]-devices)**2,axis=-1)+p['altitude_m']**2
    power=weights*distances*p['iot_pa_efficiency']/p['device_slot_s']
    values.append(((p['iot_max_power_w']-power)/p['iot_max_power_w']).ravel())
    for site,head in zip(x,heads):
        power=relay*np.sum((site-site[head])**2,axis=-1)*p['uav_pa_efficiency']/p['relay_slot_s']
        values.append((p['uav_max_power_w']-power)/p['uav_max_power_w'])
    return np.concatenate(values)


def optimize(initial, devices, bits):
    centers=initial.mean(axis=1);weights,relay=coefficients(bits)
    x=initial.copy();a=association(x,devices,weights);heads,_=choose_heads(x,relay)
    start=objective(x,devices,a,weights,relay,heads);trace=[]
    for iteration in range(PARAMETERS['maximum_outer_iterations']):
        # Fixed associations and head: constrained differentiable position step.
        def f(z):
            value,g=objective(z.reshape(4,10,2)*100,devices,a,weights,relay,heads,True)
            return value/start,(g*100/start).ravel()
        result=minimize(f,(x/100).ravel(),jac=True,method='SLSQP',
                        constraints=[dict(type='ineq',fun=lambda z:constraints(z.reshape(4,10,2)*100,centers,devices,a,weights,relay,heads))],
                        options=dict(maxiter=300,ftol=1e-10,disp=False))
        candidate=result.x.reshape(4,10,2)*100
        feasible=constraints(candidate,centers,devices,a,weights,relay,heads).min()>=-1e-7
        before=objective(x,devices,a,weights,relay,heads)
        after=objective(candidate,devices,a,weights,relay,heads)
        accepted=bool(result.success and feasible and after<=before+1e-10)
        if accepted:x=candidate
        a_new=association(x,devices,weights)
        # Equal-size nominal updates make relay cost independent of association.
        assert objective(x,devices,a_new,weights,relay,heads)<=objective(x,devices,a,weights,relay,heads)+1e-10
        a=a_new;heads,head_costs=choose_heads(x,relay)
        total=objective(x,devices,a,weights,relay,heads)
        trace.append(dict(iteration=iteration,objective_j=total,position_step_accepted=accepted,
                          solver_success=bool(result.success),solver_message=str(result.message),
                          solver_iterations=int(result.nit),heads_zero_based=heads.tolist()))
        if not accepted or (len(trace)>1 and abs(trace[-2]['objective_j']-total)<1e-9):break
    assert constraints(x,centers,devices,a,weights,relay,heads).min()>=-1e-7
    assert objective(x,devices,a,weights,relay,heads)<=start+1e-9
    assert all(trace[i]['objective_j']<=trace[i-1]['objective_j']+1e-9 for i in range(1,len(trace)))
    return x,a,heads,dict(initial_objective_j=start,final_objective_j=objective(x,devices,a,weights,relay,heads),
                         trace=trace,head_candidate_energy_j=head_costs.tolist(),
                         minimum_flight_separation_m=min(float(minimum_transition_separation(s,t).min()) for s,t in zip(x,np.roll(x,-1,axis=0))))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'completed.json').exists():
        print('Existing registered deployment retained:',OUT);return
    old=json.loads((OLD/'mission_spec.json').read_text())
    initial=np.array([s['positions_xy_m'] for s in old['stops']])
    devices,bits=create_devices(initial.mean(axis=1))
    write(OUT/'registered_inputs.json',dict(parameters=PARAMETERS,devices_xy_m=devices.tolist(),bits=bits.tolist(),
                                           source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),unix=time.time()))
    x,a,heads,report=optimize(initial,devices,bits)
    weights,relay=coefficients(bits)
    spec=dict(old);clock=0.;stops=[]
    for site,stop in enumerate(old['stops']):
        if site:clock+=float(np.linalg.norm(x[site]-x[site-1],axis=1).max())/PARAMETERS['cruise_speed_m_s']
        stops.append(dict(stop,positions_xy_m=x[site].tolist(),start_s=clock,end_s=clock+60.))
        clock+=60.
    clock+=float(np.linalg.norm(x[-1]-x[0],axis=1).max())/PARAMETERS['cruise_speed_m_s']
    spec.update(version='iot_uav_deployment_v1',stops=stops,mission_end_s=clock,
                semantics='OSM-anchored synthetic IoT demand with optimized UAV positions; neither measured sensor locations nor geotagged VLM samples.',
                FL_release_s=[s['start_s']+1.2*j for s in stops for j in range(45)],
                timing='Ten seconds of IoT collection precedes each 60s FL/FU service window in deployment accounting; radio learning clock remains service-only. Safe straight-line transitions <=5m/s.',
                energy='IoT PA/circuit energy and UAV reception added to communication; communication, compute and rotary-wing flight totals use declared scenario constants.')
    for b in spec['request_bursts']:
        b['arrival_s']=stops[b['site']]['start_s']+(b['arrival_s']-old['stops'][b['site']]['start_s'])
    write(OUT/'mission_spec.json',spec)
    channels=[];records=[]
    for s in range(4):
        dist=np.linalg.norm(x[s,:,None]-x[s,None,:],axis=-1)
        h=1000/np.maximum(dist,1);np.fill_diagonal(h,0);channels.append(h)
        e=weights[s]*(np.sum((x[s,a[s]]-devices[s])**2,axis=-1)+PARAMETERS['altitude_m']**2)
        circuit=40*PARAMETERS['iot_tx_circuit_w']*PARAMETERS['device_slot_s']
        rx=10*PARAMETERS['uav_rx_circuit_w']*PARAMETERS['collection_s']
        records.append(dict(site=s,head_zero_based=int(heads[s]),head_kind='nominal_packet_planning_head_not_FL_FU_result',
                            time_s=stops[s]['start_s'],uav_positions_xy_m=x[s].tolist(),iot_positions_xy_m=devices[s].tolist(),
                            iot_payload_bits=bits[s].tolist(),association_zero_based=a[s].tolist(),
                            iot_pa_energy_j=float(e.sum()),iot_tx_circuit_energy_j=circuit,uav_reception_energy_j=rx,
                            communication_energy_j=float(e.sum())+circuit+rx,
                            collection_s=PARAMETERS['collection_s']))
    np.save(OUT/'channels_by_site.npy',np.array(channels))
    write(OUT/'deployment.json',dict(parameters=PARAMETERS,sites=records,optimization=report,
                                   baseline_formation='Original two-row placement; same IoT demand and balanced association solver',
                                   channel='Same existing noise-normalized free-space inter-UAV model, recomputed at optimized positions'))
    projection=spec['projection']
    with (OUT/'trajectory.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['mission_time_s','uav','latitude','longitude','altitude_m','east_m','north_m','phase','site'])
        for s,stop in enumerate(stops):
            for t in [stop['start_s'],stop['end_s']]:
                for u,point in enumerate(x[s]):
                    lat=projection['lat0']+np.degrees(point[1]/projection['R_m'])
                    lon=projection['lon0']+np.degrees(point[0]/(projection['R_m']*np.cos(np.radians(projection['lat0']))))
                    writer.writerow([t,u,lat,lon,PARAMETERS['altitude_m'],*point,'inspection_hover',s])
    write(OUT/'completed.json',dict(passed=True,unix=time.time(),optimization=report,
                                  scope='Geometry planning complete; new FL/FU model evaluation is a separate run.'))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
