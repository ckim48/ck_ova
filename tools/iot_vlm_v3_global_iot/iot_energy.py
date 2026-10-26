"""Extend the existing declared energy scenario with ground data collection."""
import copy
import importlib.util
import json
import inspect
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('iot_existing_energy',ROOT/'tools/uav_real_vlm_v1/energy.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
# Generalize only ledger dimensions; retain the existing physical equations.
source=inspect.getsource(base.method_energy)
assert source.count('np.zeros(4)')==3 and source.count('if index<3')==1
source=source.replace('np.zeros(4)',"np.zeros(len(geometry['stops']))")
source=source.replace('if index<3','if index<len(stops)-1')
namespace=dict(base.__dict__)
exec(compile(source,str(ROOT/'tools/uav_real_vlm_v1/energy.py'),'exec'),namespace)
base.method_energy=namespace['method_energy']
PARAMETERS=copy.deepcopy(base.PARAMETERS)
PARAMETERS['iot_collection']=json.loads((ROOT/'results/iot_uav_deployment_v3_global_iot/deployment.json').read_text())['parameters']
PARAMETERS['assumptions'].append('Before each service window: all 40 ground devices upload over ten orthogonal 100kHz bands and four 2.5s slots per UAV. Ground PA/circuit and UAV reception energy plus 10s of ten-UAV hovering are charged once per site per independent arm. Geometry and demand are common to all four methods.')


def method_energy(dataset,method,geometry):
    result=base.method_energy(dataset,method,geometry)
    deployment=json.loads((ROOT/'results/iot_uav_deployment_v3_global_iot/deployment.json').read_text())
    collection_energy=sum(s['communication_energy_j'] for s in deployment['sites'])
    collection_time=sum(s['collection_s'] for s in deployment['sites'])
    extra_flight=10*float(base.propulsion_power(0))*collection_time
    result['energy_communication_j']+=collection_energy
    result['energy_flight_j']+=extra_flight
    result['total_energy_j']+=collection_energy+extra_flight
    result['mission_delay_s']+=collection_time
    result['total_delay_s']+=collection_time
    result['hover_by_site_s']=[t+s['collection_s'] for t,s in zip(result['hover_by_site_s'],deployment['sites'])]
    result.update(iot_collection_communication_energy_j=collection_energy,iot_collection_seconds=collection_time,
                  total_energy_scope='10 UAVs plus ground IoT transmission; modeled, not measured')
    return result


def check_five_site_accounting():
    """Exercise the fifth-site ledger and closed flight tour with a toy workload."""
    import math
    import tempfile
    geometry=json.loads((ROOT/'results/iot_uav_deployment_v3_global_iot/mission_spec.json').read_text())
    assert len(geometry['stops'])==5
    with tempfile.TemporaryDirectory(prefix='iot_energy_check_') as temporary:
        folder=Path(temporary)
        records={
            'plan.json':dict(partitions=[[] for _ in range(10)]),
            'feature_cache/manifest.json':dict(lineage=[]),
            'initial20/rounds.json':[], 'proxy20/rounds.json':[],
            'fixture/manifest.json':dict(status='completed_200FL_and_requests',evaluation=dict(retained=dict(loss=0.,mR=0.))),
            'fixture/work_attempts/000.json':dict(site=4,compute=dict(device_macs=[12e9]+[0.]*9)),
        }
        for name,record in records.items():
            path=folder/name;path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps(record))
        result=method_energy(folder,'fixture',geometry)
    assert math.isclose(result['energy_computing_j'],5.375)
    assert result['iot_collection_seconds']==50.
    assert len(result['hover_by_site_s'])==5
    assert math.isclose(result['mission_delay_s'],geometry['mission_end_s']+50.)
    assert math.isclose(result['total_energy_j'],sum(result['energy_'+part+'_j'] for part in ['communication','computing','flight']))
    expected_travel=0.
    stops=geometry['stops']
    for index in range(5):
        a,b=stops[index],stops[(index+1)%5]
        seconds=(b['start_s'] if index<4 else geometry['mission_end_s'])-a['end_s']
        for start,end in zip(a['positions_xy_m'],b['positions_xy_m']):
            speed=math.dist(start,end)/seconds
            expected_travel+=float(base.propulsion_power(speed))*seconds
    expected_hover=10*float(base.propulsion_power(0))*sum(result['hover_by_site_s'])
    assert math.isclose(result['energy_flight_j'],expected_travel+expected_hover)
    return dict(passed=True,sites=5,collection_seconds=50.,fifth_site_compute_j=5.375,closed_tour_checked=True)
