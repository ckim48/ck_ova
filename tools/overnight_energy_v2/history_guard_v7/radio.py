"""CPU radio check of actual guarded FU vectors, never a GPU promotion."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

STATUS_PATH=None


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def identity(pid):
    try:
        s=Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split();return None if s[0]=='Z' else s[19]
    except FileNotFoundError:return None


def inputs(learning,unlearning,config,epsilon):
    from wireless import quantize
    b=config['block_dimension'];pad=(-learning.shape[1])%b
    l=np.pad(learning,((0,0),(0,pad))).reshape(len(learning),-1,b)
    u=np.pad(unlearning,((0,0),(0,pad))).reshape(l.shape)
    active=np.flatnonzero(np.any(u!=0,axis=(1,2)));decoded=np.zeros_like(u)
    delta=np.zeros(u.shape[:2]);payload=np.zeros(u.shape[:2]);depths={}
    for client in active:
        for depth in [6,8,10,12,14,16]:
            reconstructed,d,bits=quantize(u[client:client+1],depth,config['scale_metadata_bits'])
            if np.linalg.norm(d)<=.25*epsilon/max(len(active),1):
                decoded[client]=reconstructed[0];delta[client]=d[0]
                payload[client]=bits[0]+3*(bits[0]>0);depths[int(client)]=depth;break
        else:raise ValueError('No allowed bit depth meets the local quantization budget.')
    assert np.linalg.norm(delta.sum(axis=0))<=.25*epsilon*(1+1e-8)
    assert np.all(np.linalg.norm(decoded-u,axis=-1)<=delta+1e-12)
    dims=np.full(l.shape[1],b);symbols=dims*config['symbols_per_coordinate']
    return l,u,decoded,delta,payload,np.linalg.norm(l,axis=-1),dims,symbols,depths


def main():
    global STATUS_PATH
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,required=True);p.add_argument('--wait-pid',type=int,required=True)
    args=p.parse_args();root=args.project.resolve();source=root/'results/overnight_energy_v2/history_guard_v7';out=source/'radio';out.mkdir(exist_ok=True);STATUS_PATH=out/'status.json'
    def save(n,x):(out/n).write_text(json.dumps(x,indent=2)+'\n')
    initial=identity(args.wait_pid);save('status.json',{'status':'waiting_for_cpu_pilot','wait_pid':args.wait_pid})
    while initial is not None and identity(args.wait_pid)==initial:time.sleep(15)
    pilot=json.loads((source/'status.json').read_text())
    if pilot['status']!='completed_cpu_screen_not_promoted' or pilot.get('nonzero_request_updates')!=2:
        save('status.json',{'status':'skipped_incomplete_or_zero_update_requests','pilot_status':pilot});return
    sys.path[:0]=[str(root),str(root/'tools/overnight_energy_v2'),str(root/'tools/overnight_energy_v2/joint_power_v3')]
    from phase_split_grid import split_symbols
    from adaptive_precision import exact_statistics
    from transport import energy_accounting
    from wireless_joint_v3 import optimize_head
    config=json.loads((root/'results/failed_round_diagnostic/metadata.json').read_text())['config']
    expected=json.loads((root/'results/comparisons_seed0_solver_v2/suite_metadata.json').read_text())['source_sha256'];assert all(sha(root/n)==h for n,h in expected.items())
    assert sha(source/'generated_updates.npz')==pilot['generated_updates_sha256']
    with np.load(source/'generated_updates.npz') as data:learning,unlearning,channels=[data[k].copy() for k in ['learning','unlearning','channels']]
    b=config['block_dimension'];blocked=np.pad(learning,((0,0),(0,(-learning.shape[1])%b))).reshape(len(learning),-1,b)
    scale=np.linalg.norm(np.linalg.norm(blocked,axis=-1).sum(axis=0))
    plan={'at_kst':datetime.now().isoformat(timespec='seconds'),'updated_method':'history_guard_v7',
          'relative_tolerances':[.2,.6],'analog_fractions':[.125,.25,.375,.5,.625,.75,.875],
          'codec':'Per-client smallest6/8/10/12/14/16 bit depth with local quantization RMS<=0.25*epsilon/announced_requester_count; 3-bit depth index per active block, float32 scale already counted.',
          'updates_sha256':sha(source/'generated_updates.npz'),'pilot_results_sha256':sha(source/'results.json'),
          'source_sha256':expected,'script_sha256':sha(__file__),
          'aux_source_sha256':{n:sha(root/n) for n in ['tools/overnight_energy_v2/phase_split_grid.py','tools/overnight_energy_v2/adaptive_precision.py','tools/overnight_energy_v2/history_guard_v7/test_codec_inputs.py']},
          'solver_sha256':sha(root/'tools/overnight_energy_v2/joint_power_v3/wireless_joint_v3.py'),
          'limits':['Changed local unlearning algorithm. These are actual one-step pilot updates, not final training results.',
                    'Shared time, power/battery, error bound and codec for Hybrid/phase; Mixed pays no digital metadata.',
                    'All heads and all seven phase fractions retained. No global-optimum or equal task-quality claim.',
                    'Wireless errors can invalidate ideal-link local loss guards. Full radio-perturbed task evaluation remains required.',
                    'Extra local guard inference/ledger energy is uncalibrated. Communication-only modeling, no automatic GPU promotion.']}
    save('manifest.json',plan);rows=[]
    for relative in [.2,.6]:
        epsilon=max(config['absolute_tolerance_floor'],relative*scale)
        try:base=inputs(learning,unlearning,config,epsilon);codec_error=None
        except ValueError as e:base=None;codec_error=str(e)
        for mode,fraction in [('hybrid',None),('mixed_aircomp',None)]+[('orthogonal',f) for f in plan['analog_fractions']]:
            save('status.json',{'status':'solving','relative_tolerance':relative,'mode':mode,'analog_fraction':fraction})
            row={'relative_tolerance':relative,'mode':mode,'analog_fraction':fraction,'epsilon':float(epsilon),'status':'no_feasible_design_found'}
            if mode=='mixed_aircomp':
                shape=blocked.shape;l=np.pad(learning+unlearning,((0,0),(0,(-learning.shape[1])%b))).reshape(shape)
                u=np.zeros_like(l);decoded=u.copy();delta=np.zeros(shape[:2]);payload=np.zeros_like(delta);rho=np.linalg.norm(l,axis=-1)
                dims=np.full(l.shape[1],b);symbols=dims*config['symbols_per_coordinate'];depths={}
            elif base is None:
                row.update(status='quantization_budget_failed',error=codec_error);rows.append(row);save('results.json',rows);continue
            else:l,u,decoded,delta,payload,rho,dims,symbols,depths=base
            row['bits_by_client']=depths;best=None;heads=[]
            for head in range(config['users']):
                a,d=symbols.copy(),symbols.copy()
                if mode=='orthogonal':a,d,_=split_symbols(symbols,payload,head,fraction)
                design=optimize_head(head,channels,rho,delta,payload,dims,a,epsilon,config,config['sic_residual_fraction'],mode,digital_symbols=d)
                heads.append({'head':head,'status':'feasible' if design is not None else 'no_feasible_design_found','energy_bound_j':design.energy_j if design is not None else None})
                if design is not None and (best is None or design.energy_j<best.energy_j):best=design
            row['heads']=heads
            if best is not None:
                row.update(energy_accounting(best,config));row.update(exact_statistics(best,l,u,decoded,learning.shape[1]));row['status']='feasible_radio_only'
                if mode=='mixed_aircomp':row['unlearning_component_error']=None
                assert np.isclose(row['uplink_s'],symbols.sum()/config['symbol_rate_hz'])
                assert row['error_bound']<=epsilon*(1+1e-5)
                assert row['analytic_total_rmse']<=row['error_bound']*(1+1e-4)+1e-12
                np.savez_compressed(out/f'r{relative:g}_{mode}_f{fraction}_design.npz',**{k:getattr(best,k) for k in best.__dataclass_fields__ if k!='history'})
            rows.append(row);save('results.json',rows);print(json.dumps({k:v for k,v in row.items() if k!='heads'}),flush=True)
    save('status.json',{'status':'completed_radio_only_not_promoted','cases':len(rows),'at_kst':datetime.now().isoformat(timespec='seconds')})


if __name__=='__main__':
    try:main()
    except Exception as error:
        if STATUS_PATH is not None:STATUS_PATH.write_text(json.dumps({'status':'failed_cpu_radio','error':str(error),'at_kst':datetime.now().isoformat(timespec='seconds')},indent=2)+'\n')
        raise
