"""Extend the existing declared energy scenario with ground data collection."""
import copy
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('iot_existing_energy',ROOT/'tools/uav_real_vlm_v1/energy.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
PARAMETERS=copy.deepcopy(base.PARAMETERS)
PARAMETERS['iot_collection']=json.loads((ROOT/'results/iot_uav_deployment_v1/deployment.json').read_text())['parameters']
PARAMETERS['assumptions'].append('Before each service window: all 40 ground devices upload over ten orthogonal 100kHz bands and four 2.5s slots per UAV. Ground PA/circuit and UAV reception energy plus 10s of ten-UAV hovering are charged once per site per independent arm. Geometry and demand are common to all four methods.')


def method_energy(dataset,method,geometry):
    result=base.method_energy(dataset,method,geometry)
    deployment=json.loads((ROOT/'results/iot_uav_deployment_v1/deployment.json').read_text())
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
