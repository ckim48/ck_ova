"""Frozen-update CPU baseline; no training or application-quality selection."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from solver import allocate_symbols,optimize,exact_statistics


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,x):
    temporary=p.with_suffix(p.suffix+'.tmp')
    temporary.write_text(json.dumps(x,indent=2)+'\n');temporary.replace(p)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,required=True)
    p.add_argument('--plan-only',action='store_true')
    args=p.parse_args();root=args.project.resolve()
    out=root/'results/overnight_energy_v2/dual_analog_v9';out.mkdir(exist_ok=True)
    source=root/'results/failed_round_diagnostic'
    meta=read(source/'metadata.json');config=meta['config'];code=Path(__file__).parent
    expected=read(root/'results/comparisons_seed0_solver_v2/suite_metadata.json')['source_sha256']
    assert all(sha(root/n)==v for n,v in expected.items())
    with np.load(source/'updates.npz') as a:
        learning,unlearning,channels=[a[k].copy() for k in ['learning','unlearning','channels']]
    block=config['block_dimension'];pad=(-learning.shape[1])%block
    blocked=np.stack([np.pad(v,((0,0),(0,pad))).reshape(len(v),-1,block) for v in [learning,unlearning]])
    rho=np.linalg.norm(blocked,axis=-1);dims=np.full(rho.shape[-1],block)
    total=dims*config['symbols_per_coordinate']
    scale=float(np.linalg.norm(rho[0].sum(axis=0)))
    solver_config=dict(config,ao_iterations=30)
    plan={'version':'dual_analog_v9_original_additive_sum_cpu_baseline',
          'round':meta['round'],'updates_sha256':sha(source/'updates.npz'),
          'core_sha256':expected,'code_sha256':{n:sha(code/n) for n in ['solver.py','run.py','test_solver.py']},
          'config':config,'solver_config':solver_config,'relative_tolerances':[.2,.6],
          'learning_time_fractions':[.125,.25,.375,.5,.625,.75,.875],
          'total_uplink_s':float(total.sum()/config['symbol_rate_hz']),
          'bound':'For the original sum: norm over blocks of [sum(stream,user) norm_bound*abs(effective_gain-1) + sqrt(sum_stream d*s_stream^2/(2*N_stream))]. Independent phase Gaussian noise; no SIC residual/quantization. No raw update geometry in optimizer.',
          'initialization':'Convex minimum-error start plus reference receive magnitudes times0.5/1/2/4; joint amplitudes across both streams at fixed receivers, then joint convex receiver update. Up to30 AO iterations, best feasible start retained.',
          'conditions':'Same original updates/channels/head set/total time/coherence/power/battery/base-head overheads and same original-sum RMS budget as H/Mixed/phase diagnostics. Inactive remote streams skip their phase, returning time to active stream.',
          'limits':['This is a stronger separated two-analog baseline, not digital OMA/TDMA and not a proposed energy win.',
                    'Finite phase grid and alternating optimization; feasible incumbent energy upper bounds, not global minima or physical-infeasibility proofs.',
                    'Independent phase noise combines in quadrature. Deterministic vector mismatch uses triangle bounds, with no free cross-client gradients.',
                    'Separate aggregates are recovered but optimized for original additive sum quality, NOT nonlinear v8 fusion quality or certified deletion.',
                    'No VLM training/inference, no validation/test parameter selection, no automatic GPU promotion.',
                    'Original fixed overheads reused as in other radio diagnostics. Extra circuit/control/calibration costs are unmeasured; no whole-UAV energy claim.']}
    manifest=out/'manifest.json'
    if manifest.exists():assert read(manifest)==plan
    else:write(manifest,plan)
    if args.plan_only:print(json.dumps(plan));return
    rows=read(out/'heads.json') if (out/'heads.json').exists() else []
    done={(x['relative_tolerance'],x['learning_time_fraction'],x['head']) for x in rows}
    def status(state,**x):write(out/'status.json',dict(status=state,at_kst=datetime.now().isoformat(timespec='seconds'),**x))
    status('running_cpu_radio_only',completed_head_designs=len(rows))
    for relative in plan['relative_tolerances']:
        epsilon=max(config['absolute_tolerance_floor'],relative*scale)
        for fraction in plan['learning_time_fractions']:
            for head in range(config['users']):
                key=(relative,fraction,head)
                if key in done:continue
                status('solving_cpu_head',relative_tolerance=relative,learning_time_fraction=fraction,head=head,completed_head_designs=len(rows))
                n=allocate_symbols(total,dims,rho,head,fraction);begin=time.monotonic()
                design=optimize(head,channels,rho,dims,n,epsilon,solver_config)
                row=dict(relative_tolerance=relative,learning_time_fraction=fraction,head=head,
                         epsilon=float(epsilon),status='no_feasible_design_found',solve_seconds=time.monotonic()-begin)
                if design is not None:
                    row.update(status='feasible_radio_only',energy_bound_j=design.energy_j,
                               error_bound=design.error_bound,history=design.history,
                               hit_ao_limit=len(design.history)>=31,
                               total_uplink_s=float(n.sum()/config['symbol_rate_hz']),
                               maximum_phase_power_w=float(np.max(design.analog_rms**2)),
                               maximum_uav_tx_energy_j=float(np.max((design.analog_rms**2*(n/config['symbol_rate_hz'])[:,None,:]).sum(axis=(0,2)))),
                               **exact_statistics(design,blocked,learning.shape[1]))
                    assert row['analytic_sum_rmse']<=design.error_bound*(1+1e-5)+1e-12
                    assert row['error_bound']<=epsilon*(1+1e-5)
                    assert row['total_uplink_s']<=plan['total_uplink_s']*(1+1e-10)
                    path=out/f'r{relative:g}_f{fraction:g}_h{head}.npz'
                    np.savez_compressed(path,**{k:getattr(design,k) for k in design.__dataclass_fields__ if k!='history'})
                rows.append(row);write(out/'heads.json',rows)
                print(json.dumps(row),flush=True)
            cases=[]
            for rel in plan['relative_tolerances']:
                for f in plan['learning_time_fractions']:
                    subset=[x for x in rows if x['relative_tolerance']==rel and x['learning_time_fraction']==f]
                    if not subset:continue
                    good=[x for x in subset if x['status']=='feasible_radio_only']
                    winner=min(good,key=lambda x:x['energy_bound_j']) if good else None
                    cases.append(dict(relative_tolerance=rel,learning_time_fraction=f,
                                      completed_heads=len(subset),status='completed' if len(subset)==config['users'] else 'partial',
                                      best_feasible_head=winner))
            write(out/'results.json',cases)
    status('completed_cpu_radio_only_not_promoted',head_designs=len(rows),cases=len(plan['relative_tolerances'])*len(plan['learning_time_fractions']))


if __name__=='__main__':
    try:main()
    except Exception as e:
        print('FAILED_CPU_RADIO',repr(e),flush=True)
        raise
