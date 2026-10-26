import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
out=ROOT/'results/overnight_energy_v2/convergence_energy_delay_v1'
out.mkdir(parents=True,exist_ok=True)
cfg=json.loads((ROOT/'results/overnight_energy_v2/burst_joint_v3/config.json').read_text())
cfg.pop('deletion_relative_tolerance',None)
cfg.update(requesting_users=5,symbols_per_coordinate=16,round_uplink_deadline_s=.34,
           relative_tolerance=.2,max_ao_iterations=30,receive_initializations=4,evaluate_every=1)
sources=['run_clip_federated.py','clip_model.py','real_data.py','data_schedule.py','wireless.py','transport.py',
         'tools/overnight_energy_v2/phase_split_grid.py','tools/overnight_energy_v2/initialization_v11/wireless_init_v11.py',
         'tools/overnight_energy_v2/receiver_control_v1/receiver_control.py']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
plan=dict(version='convergence_energy_delay_v1',config=cfg,
    methods=['proposed','aircomp_oma','aircomp_only','oma_only','hybrid_no_coupling'],
    source_sha256={n:sha(ROOT/n) for n in sources},runner_sha256={p.name:sha(p) for p in sorted(HERE.glob('*.py'))},
    target_sweep=dict(accuracy=[round(.65+i*.01,2) for i in range(16)],loss=[round(.80+i*.02,2) for i in range(26)]),
    overhead_seconds_sensitivity=[0.,.01,.1,.5],
    timing_scope='Per-client CUDA-synchronized RTX3080 compute; max client time for concurrent UAV proxy; sequential head calibration; measured CPU design/controller time; modeled uplink; control/downlink overhead reported parametrically. Server wall and offline evaluation time separate.',
    changed_from_previous=dict(requesting_users='2 -> 5 simultaneous requesters',symbols_per_coordinate='8 -> 16: 335.872ms frame, common to all; original 167.936ms pilot preserved',
        crossmodal='only proposed uses coupling; an extra hybrid_no_coupling run isolates it',
        oma='strict per-user TDMA, joint time/energy/power/battery allocation; AirComp+OMA additionally optimizes analog amplitudes for each prescribed phase fraction',
        request_controller='same alpha=1 and Q8 canonical request vectors for all individually recovering methods; AirComp-only is additive reference',
        evaluations='1000 disjoint held-out training-validation examples every round; 64 separate calibration examples; no test-set tuning'),
    limitations=['single seed 0 pilot, not a final multi-seed study','same relative update-error policy along different model trajectories; matched task quality is assessed by common validation targets',
        '0.1J is a per-round communication allowance, not a depleted full-mission battery model',
        '8.192ms maximum block time remains below original 10ms coherence limit',
        'communication-energy bound includes original fixed overhead; compute/flight energy unmeasured',
        'pure AirComp cannot deliver individual request vectors and is not ranked as meeting that requirement',
        'no-feasible-design-found from alternating optimization is not proof of global infeasibility'])
p=out/'plan.json'
if p.exists():assert json.loads(p.read_text())==plan,'Existing plan differs; use a new version'
else:p.write_text(json.dumps(plan,indent=2)+'\n')
print(p)
