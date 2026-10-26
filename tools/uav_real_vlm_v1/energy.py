"""Explicit simulation model: communication + onboard computing + propulsion.

Hardware constants are declared scenario assumptions, not measured calibration.
Server runtime is never converted into UAV energy. Training is not modified by
this postprocessing model; site hover time expands if modeled work needs it.
"""
import json
import math
from pathlib import Path
import numpy as np

PARAMETERS = {
    'status': 'modeled_scenario_not_hardware_measurement',
    'propulsion': {'P0_w': 79.86, 'Pi_w': 88.63, 'tip_speed_m_s': 120.,
                   'induced_speed_m_s': 4.03, 'drag_ratio': .6, 'air_density_kg_m3': 1.225,
                   'rotor_solidity': .05, 'rotor_area_m2': .503},
    'compute': {'frequency_hz': 1.5e9, 'capacitance': 1e-27, 'macs_per_cycle': 8.,
                'active_board_overhead_w': 2., 'gradient_MAC_multiplier': 3.},
    'communication': {'pa_efficiency': .35, 'tx_circuit_w': .1, 'rx_circuit_w': .1,
                      'downlink_power_w': .32, 'downlink_rate_bps': 1e6,
                      'update_coordinates': 20480, 'broadcast_bits_per_coordinate': 32,
                      'control_bits_per_update': 512, 'head_processing_w': .5},
    'references': {
        'rotor_model': 'https://doi.org/10.1109/TWC.2019.2902559',
        'rotor_parameters': 'https://dl.ifip.org/db/conf/wons/wons2025/1571073764.pdf',
    },
    'assumptions': [
        'RF energy is the recorded conservative waveform bound divided by PA efficiency.',
        'Nine scheduled transmit circuits and one receive circuit are active during uplink; no coefficient-aware circuit sleep.',
        'Each nonzero transaction broadcasts the 32-bit aggregate and one 512-bit control message at 1 Mbps to nine peers.',
        'Compute uses a CMOS cycle model with fixed explicit scenario constants, not an RTX3080 timing-to-UAV conversion.',
        'The two adapter matrix products cost 10240 MACs per modality sample; gradient calls have factor3, accounting for forward and two backward products.',
        'Frozen-encoder MAC model includes transformer attention/projections/MLPs and patch/projection layers; activation, normalization, memory, and system overhead are represented only by the active-board overhead term.',
        'Cold feature construction and initial/proxy20 occur before takeoff and are charged once per independently deployed arm.',
        'At each site, UAV computations occur in parallel; sum of the per-update maximum device time forms the synchronous compute critical path.',
        'Site hover is max(prescribed60s, compute+uplink+broadcast+logged_wait). Travel retains actual formation displacements and original segment durations.',
        'All vehicles complete the same full return route; extra workload extends hovering. Vertical takeoff/landing and acceleration transients are outside the fixed-altitude scenario.',
    ],
}


def read(path):
    return json.loads(Path(path).read_text())


def operation_macs(counts):
    total = 0.
    for key, count in counts.items():
        if key.endswith('_no_grad_samples'):
            total += 10240 * count
        elif key.endswith('_grad_samples'):
            total += 10240 * count * PARAMETERS['compute']['gradient_MAC_multiplier']
    return total


def compute_cost(macs):
    p = PARAMETERS['compute']
    cycles = np.asarray(macs, dtype=float) / p['macs_per_cycle']
    seconds = cycles / p['frequency_hz']
    energy = p['capacitance'] * cycles * p['frequency_hz'] ** 2 + p['active_board_overhead_w'] * seconds
    return energy, seconds


def propulsion_power(speed):
    p = PARAMETERS['propulsion']; speed = np.asarray(speed, dtype=float)
    assert np.all(speed >= 0)
    a = speed ** 2 / (2 * p['induced_speed_m_s'] ** 2)
    induced = 1 / np.sqrt(np.sqrt(1 + a*a) + a)  # stable equivalent
    return (p['P0_w'] * (1 + 3 * speed**2 / p['tip_speed_m_s']**2)
            + p['Pi_w'] * induced
            + .5 * p['drag_ratio'] * p['air_density_kg_m3'] * p['rotor_solidity'] * p['rotor_area_m2'] * speed**3)


def encoder_macs(tokens, width, layers):
    return layers * (12 * tokens * width**2 + 2 * tokens**2 * width)


def communication(radio):
    p = PARAMETERS['communication']
    duration = radio['uplink_s']
    rf = radio['energy_bound_j']
    if duration == 0:
        assert abs(rf) < 1e-12
        return 0., 0.
    bits = p['update_coordinates'] * p['broadcast_bits_per_coordinate'] + p['control_bits_per_update']
    broadcast = bits / p['downlink_rate_bps']
    energy = (rf / p['pa_efficiency'] + duration * (9*p['tx_circuit_w'] + p['rx_circuit_w'])
              + broadcast * (p['downlink_power_w']/p['pa_efficiency'] + p['tx_circuit_w'] + 9*p['rx_circuit_w']))
    return energy, broadcast


def method_energy(dataset, method, geometry):
    dataset = Path(dataset)
    plan = read(dataset/'plan.json')
    owner = {int(i): u for u, part in enumerate(plan['partitions']) for i in part}
    lineage = read(dataset/'feature_cache/manifest.json')['lineage']
    image_MACs = encoder_macs(50, 768, 12) + 49*768*(32*32*3) + 768*512
    text_MACs = encoder_macs(77, 512, 12) + 512*512
    cold_macs = np.zeros(10)
    for record in lineage:
        if record['training_sample']:
            cold_macs[owner[record['sample_id']]] += len(record['frames'])*image_MACs + 5*text_MACs
    cold_energy, cold_seconds = compute_cost(cold_macs)
    computing = float(cold_energy.sum())
    communication_energy = prefix_delay = 0.
    prefix_delay += float(cold_seconds.max())
    for stage in ['initial20', 'proxy20']:
        for row in read(dataset/stage/'rounds.json'):
            energy, duration = compute_cost(row['compute']['device_macs'])
            computing += float(energy.sum())
            comm, broadcast = communication(row['radio'])
            communication_energy += comm
            computing += PARAMETERS['communication']['head_processing_w']*row['radio']['uplink_s']
            prefix_delay += float(duration.max()) + row['radio']['uplink_s'] + broadcast
    directory = dataset/method
    manifest = read(directory/'manifest.json')
    assert manifest['status'] == 'completed_200FL_and_requests'
    site_work = np.zeros(4)
    site_recorded_span = np.zeros(4)
    uplink = np.zeros(4)
    for path in sorted((directory/'work_attempts').glob('*.json')):
        record = read(path)
        energy, duration = compute_cost(record['compute']['device_macs'])
        computing += float(energy.sum())
        site_work[record['site']] += float(duration.max())
    for path in sorted((directory/'transactions').glob('*.json')):
        record = read(path); site = record['site']
        comm, broadcast = communication(record['radio'])
        communication_energy += comm
        computing += PARAMETERS['communication']['head_processing_w']*record['radio']['uplink_s']
        uplink[site] += record['radio']['uplink_s']
        site_work[site] += record['radio']['uplink_s'] + broadcast
        site_recorded_span[site] = max(site_recorded_span[site], record['end_s']-geometry['stops'][site]['start_s'])
    site_work += np.maximum(0, site_recorded_span-uplink)
    flight_energy = travel_duration = 0.
    stops = geometry['stops']
    for index, (a, b) in enumerate(zip(stops, stops[1:]+stops[:1])):
        duration = ((b['start_s']-a['end_s']) if index<3 else geometry['mission_end_s']-a['end_s'])
        distance = np.linalg.norm(np.asarray(b['positions_xy_m'])-np.asarray(a['positions_xy_m']), axis=1)
        assert duration > 0
        flight_energy += float(propulsion_power(distance/duration).sum())*duration
        travel_duration += duration
    hover = [max(stop['end_s']-stop['start_s'], workload) for stop,workload in zip(stops, site_work)]
    flight_energy += 10*float(propulsion_power(0))*sum(hover)
    result = dict(method=method, energy_communication_j=communication_energy,
                  energy_computing_j=computing, energy_flight_j=flight_energy,
                  total_energy_j=communication_energy+computing+flight_energy,
                  mission_delay_s=travel_duration+sum(hover),
                  total_delay_s=prefix_delay+travel_duration+sum(hover),
                  prefix_delay_s=prefix_delay, hover_by_site_s=hover,
                  loss=manifest['evaluation']['retained']['loss'],
                  mR=manifest['evaluation']['retained']['mR'], status='modeled_scenario_not_measured')
    assert all(math.isfinite(result[k]) and result[k]>=0 for k in ['total_energy_j','total_delay_s'])
    return result


def self_check():
    assert np.isclose(propulsion_power(0), 168.49)
    assert np.all(np.isfinite(propulsion_power(np.array([0., 5., 50., 1000.]))))
    assert operation_macs({'image_grad_samples': 2, 'text_no_grad_samples': 3}) == 9*10240
    energy, duration = compute_cost([0., 12e9])
    assert energy[0] == duration[0] == 0
    assert np.isclose(duration[1], 1)
    assert np.isclose(energy[1], 5.375)
    assert communication(dict(uplink_s=0.,energy_bound_j=0.)) == (0.,0.)
    return dict(passed=True, hover_power_w=168.49, compute_energy_for_one_active_device_second_j=5.375)
