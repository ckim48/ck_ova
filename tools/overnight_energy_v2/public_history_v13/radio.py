"""Pay coefficient delivery before comparing public-history AirComp residuals.

Learning-only fixed round100: all original transport architectures reduce to
the same single analog aggregate, so no Hybrid-over-Mixed comparison is made.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def reserve_symbols(original,dimensions,rank,remote_users):
    control_each=64*rank  #32 float bits/dimension, ideal coding at0.5bit/symbol
    total_control=control_each*remote_users
    removed=np.full(len(original),total_control//len(original));removed[:total_control%len(original)]+=1
    data=original-removed
    if np.any(data<dimensions):raise ValueError('Insufficient time for the isometric analog mappings.')
    assert int(data.sum())+total_control==int(original.sum())
    return data,control_each,total_control


def control_resources(channel,payload_bits,control_symbols,rate):
    if payload_bits==0:return np.zeros_like(channel),np.zeros_like(channel)
    required=np.expm1(np.log(2)*payload_bits/control_symbols)
    power=required/(channel*channel)
    energy=control_symbols/rate*power
    assert np.all(channel*channel*power>=required*(1-1e-12))
    return power,energy


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,required=True)
    p.add_argument('--plan-only',action='store_true');args=p.parse_args();root=args.project.resolve()
    sys.path[:0]=[str(root),str(root/'tools/overnight_energy_v2/initialization_v11')]
    from wireless_init_v11 import optimize_head
    source=root/'results/overnight_energy_v2/public_history_v13/recent_screen'
    report=json.loads((source/'results.json').read_text());assert report['round']==100 and report['status'].startswith('completed')
    c=json.loads((root/'results/failed_round_diagnostic/metadata.json').read_text())['config'];c=dict(c,max_ao_iterations=30)
    core=json.loads((root/'results/comparisons_seed0_solver_v2/suite_metadata.json').read_text())['source_sha256'];assert all(sha(root/n)==v for n,v in core.items())
    with np.load(source/'vectors.npz') as a:
        learning=a['learning'].copy();channels=a['channels'].copy()
        variants=[('unprojected_baseline',0,learning)]
        for row in report['rows']:variants.append((row['basis'],row['rank'],a[row['basis']+'_residual'].copy()))
    b=c['block_dimension'];dimension=learning.shape[1]
    blocked=lambda v:np.pad(v,((0,0),(0,(-dimension)%b))).reshape(len(v),-1,b)
    original=blocked(learning);dims=np.full(original.shape[1],b);symbols=dims*c['symbols_per_coordinate']
    epsilon=max(c['absolute_tolerance_floor'],.2*np.linalg.norm(np.linalg.norm(original,axis=-1).sum(axis=0)))
    out=root/'results/overnight_energy_v2/public_history_v13/radio';out.mkdir(exist_ok=True)
    def write(n,x):(out/n).write_text(json.dumps(x,indent=2)+'\n')
    plan=dict(version='public_history_v13_paid_coefficient_learning_only_radio',script_sha256=sha(__file__),
              test_sha256=sha(Path(__file__).with_name('test_metadata.py')),core_sha256=core,
              replay_report_sha256=sha(source/'results.json'),vectors_sha256=sha(source/'vectors.npz'),
              solver_sha256=sha(root/'tools/overnight_energy_v2/initialization_v11/wireless_init_v11.py'),
              config=c,epsilon=float(epsilon),relative_tolerance=.2,total_time_s=float(symbols.sum()/c['symbol_rate_hz']),
              variants=[dict(name=n,rank=r) for n,r,_ in variants],
              metadata='Each remote client sends32bits per public-basis coefficient in a separate TDMA64symbols/rank slot at ideal0.5bit/symbol. Required SNR sqrt(2)-1. Debit ALL coefficient symbols from data symbols; add exact per-client RF power/energy and check peak/battery. No free coefficient broadcast or time.',
              limits=['One ACTUAL learning-only round100 from a PUF public trajectory. No concurrent FU or full training result.',
                      'All architectures have zero U here and the same analog problem. This is NOT Hybrid-over-Mixed evidence.',
                      'Same original absolute update-RMS budget, total airtime, channels, power, battery, overhead and AO30/5 starts for baseline and projections.',
                      'Ideal coding is the same Shannon-style abstraction as the original. Finite-block error/headers and extra coefficient decoding/projection compute energy are uncalibrated.',
                      'RF/control savings only; any extra compute cost must be below reported break-even savings before claiming net communication/whole-system benefit.',
                      'Every comparator may use the same public history and must pay the same metadata. No test/validation use, no automatic GPU promotion.'])
    manifest=out/'manifest.json'
    if manifest.exists():assert json.loads(manifest.read_text())==plan
    else:write('manifest.json',plan)
    if args.plan_only:print(json.dumps(plan));return
    results=[]
    for name,rank,vectors in variants:
        data=blocked(vectors);rho=np.linalg.norm(data,axis=-1);zero=np.zeros_like(rho)
        ndata,ncontrol,all_control=reserve_symbols(symbols,dims,rank,c['users']-1)
        candidates=[]
        for head in range(c['users']):
            write('status.json',dict(status='solving_cpu_head',variant=name,head=head,at_kst=datetime.now().isoformat(timespec='seconds')))
            users=np.delete(np.arange(c['users']),head);h=channels[users,head]
            pc,ec=control_resources(h,32*rank,ncontrol,c['symbol_rate_hz'])
            row=dict(variant=name,rank=rank,head=head,status='no_feasible_design_found')
            if np.any(pc>c['power_limit_w']*(1+1e-6)):
                row['status']='control_peak_power_failed';candidates.append(row);write(name+'_heads.json',candidates);continue
            solver_config=dict(c,available_energy_j=c['available_energy_j']-float(ec.max(initial=0)))
            begin=time.monotonic()
            design=optimize_head(head,channels,rho,zero,zero,dims,ndata,epsilon,solver_config,0.,'mixed_aircomp')
            row['solve_seconds']=time.monotonic()-begin
            if design is not None:
                a,s=design.analog_rms,design.receiver;channel=channels[users,head,None]
                energy_data=(ndata/c['symbol_rate_hz']*a*a).sum(axis=1)
                overhead=np.full(c['users'],c['base_overhead_j']);overhead[head]+=c['head_overhead_j']
                assert np.max(a*a)<=c['power_limit_w']*(1+1e-6)
                assert np.all(energy_data+ec+overhead[users]<=c['available_energy_j']*(1+1e-6))
                quality=np.linalg.norm((np.abs(s*channel*a-rho[users]).sum(axis=0)+s*np.sqrt(dims/(2*ndata))))
                assert quality<=epsilon*(1+1e-5)
                gain=np.divide(s*channel*a,rho[users],out=np.zeros_like(a),where=rho[users]>0)
                bias=((gain-1)[...,None]*data[users]).sum(axis=0).reshape(-1)[:dimension]
                valid=np.minimum(b,np.maximum(dimension-np.arange(len(dims))*b,0))
                rmse=float(np.sqrt(bias@bias+np.sum(valid*s*s/(2*ndata))))
                assert rmse<=quality*(1+1e-5)
                data_energy=float(energy_data.sum()+overhead.sum());control_energy=float(ec.sum())
                assert np.isclose(data_energy,design.energy_j,rtol=1e-8,atol=1e-12)
                row.update(status='feasible_radio_only',total_modeled_energy_j=data_energy+control_energy,
                           analog_data_plus_original_overhead_j=data_energy,coefficient_tx_energy_j=control_energy,
                           error_bound=float(quality),analytic_rmse=rmse,total_symbols=int(ndata.sum()+all_control),
                           coefficient_payload_bits=32*rank*len(users),coefficient_control_symbols=all_control,
                           history=design.history)
                np.savez_compressed(out/f'{name}_h{head}.npz',analog_rms=a,receiver=s,users=users,rho=rho[users],
                                    data_symbols=ndata,control_power=pc,control_energy=ec,channels=channel)
            candidates.append(row)
            write(name+'_heads.json',candidates)
        good=[x for x in candidates if x['status']=='feasible_radio_only']
        best=min(good,key=lambda x:x['total_modeled_energy_j']) if good else None
        results.append(dict(variant=name,rank=rank,completed_heads=len(candidates),best=best))
        write('results.json',results);print('RESULT',json.dumps(results[-1]),flush=True)
    baseline=results[0]['best'];summary=[]
    for item in results[1:]:
        if item['best'] is not None and baseline is not None:
            saving=baseline['total_modeled_energy_j']-item['best']['total_modeled_energy_j']
            summary.append(dict(variant=item['variant'],rf_plus_original_overhead_saving_j=saving,
                                saving_fraction=saving/baseline['total_modeled_energy_j'],
                                maximum_unmodeled_extra_compute_energy_for_net_gain_j=max(saving,0.)))
    write('summary.json',summary);write('status.json',dict(status='completed_learning_only_radio_no_gpu_promotion',at_kst=datetime.now().isoformat(timespec='seconds')))


if __name__=='__main__':main()
