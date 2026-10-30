"""Predeclared asymmetric-tolerance audit; retain all attempted conditions."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import combined_iq_experiment_a as previous

ROOT, read, write, sha = previous.ROOT, previous.read, previous.write, previous.sha
OUT = ROOT/'results/combined_iq_diagnostic_v1_20261030'
CASES = {
    'balanced_replay': dict(ratio=1., channel_gain_db=0., modes=['iq']),
    'mapped_noise_control': dict(ratio=1., channel_gain_db=0., modes=['combined']),
    'fu_strict': dict(ratio=.1, channel_gain_db=0., modes=['combined', 'iq']),
    'fu_strict_weak': dict(ratio=.1, channel_gain_db=-12., modes=['combined', 'iq']),
}
ORIGINAL_MODULE = previous.radio_module


def register():
    p = OUT/'protocol.json'
    if p.exists():
        result = read(p)
        assert result['source_sha256'] == sha(Path(__file__))
        return result
    result = dict(registered_unix=time.time(), source_sha256=sha(Path(__file__)),
        parent_protocol=read(previous.OUT/'protocol.json'), cases=CASES,
        seed=0, all_results_retained=True,
        motivation='Original I/Q C8 component bounds were never >=99% active during joint rounds; C9 was active in all joint rounds. Test whether asymmetric requirements reveal a benefit, without withholding available controls from Combined.',
        asymmetry='Hold epsilon_L^2+epsilon_U^2 and the two original C9 budgets fixed at each update. Multiply epsilon_U by ratio and renormalize both epsilons to the same total budget. Thus ratio specifies FU/FL RELATIVE tolerance ratio. Inactive tasks unchanged. Tight FU is ratio0.1 for both datasets.',
        channel='Multiply complex channel amplitudes by10^(gain_db/20), identically for both arms. Keep noise, max power, RF caps, deadlines, count grid and all learning rules unchanged.',
        noise='Original paired wireless seed; no new training-seed claim. All comparisons are development diagnostics, not statistically confirmed superiority.',
        replay='Balanced I/Q must reproduce experiment-A final parameter tensors exactly. Save every joint-round scalar audit and five predetermined joint-update snapshots per dataset (joint indices0,9,18,27,36). No outcome-selected snapshots.',
        mapped_control='Construct a single combined stream for EACH I/Q solution with the SAME head, duration and final-noise covariance. x_C=1/(1/x_L+1/x_U) over active branches. Weighted Cauchy-Schwarz gives ||l+u||^2*x_C <= ||l||^2*x_L+||u||^2*x_U per user/block. Couple its single Gaussian to the normalized sum of the two original Gaussians; no sample acceptance/rejection. It must reproduce the I/Q trajectory exactly while using no greater RF power. This is a constructive diagnostic, not an independently optimized deployment algorithm.',
        snapshot_grid=dict(tolerance_ratios=[.1,.25,1.,4.,10.],channel_gain_db=[0.,-6.,-12.]),
        snapshot_scope='Same real VLM gradients, curvature and geometry for both arms at every point. Frozen-update resource sensitivity only; no inference of end-to-end mR from snapshot data. Solver failure is no feasible design found, not proof of mathematical infeasibility.',
        primary_outputs='All final retained mR, common/matched forget KL, active energy/delay; component/C9 constraint activity, all failures; constructive noise-equivalence checks. No requirement that I/Q win.',
        learning_unchanged=True, source_dependencies={str(Path(p).relative_to(ROOT)):sha(Path(p)) for p in [previous.__file__,previous.iq.__file__]})
    write(p,result)
    return result


def install_inputs(module, ratio):
    if ratio == 1.:
        return
    original = module.inputs
    def inputs(l, u, c, ell):
        rho, eps, weights, targets = original(l,u,c,ell)
        if np.all(eps > 0):
            budget=float(np.sum(eps**2))
            eps[1] *= ratio
            eps *= np.sqrt(budget/float(np.sum(eps**2)))
        return rho,eps,weights,targets
    module.inputs=inputs


def transport(mode, ratio):
    # Apply the asymmetric component budgets BEFORE collapsing the Combined stream.
    module=ORIGINAL_MODULE('iq')
    install_inputs(module,ratio)
    if mode=='combined':
        original=module.inputs
        def inputs(l,u,c,ell):
            _,eps,weights,targets=original(l,u,c,ell)
            rho=np.zeros((2,len(l),l.shape[1]//c['block_dimension']))
            rho[0]=np.linalg.norm((l+u).reshape(len(l),-1,c['block_dimension']),axis=-1)
            return rho,np.array([np.linalg.norm(eps),0.]),weights,targets
        module.inputs=inputs
    return module


def map_iq_to_combined(l,u,channels,c,meta):
    """Feasible single-stream construction for a fixed I/Q sum covariance."""
    h=np.abs(channels[:,meta['head']]).copy();h[meta['head']]=1.
    v=np.asarray(meta['block_variance']);total=v.sum(0)
    nu=c['block_dimension']/(2*c['symbol_rate_hz'])
    x=np.divide(nu,total,out=np.ones_like(total),where=total>0)
    rho=np.linalg.norm((l+u).reshape(len(l),-1,c['block_dimension']),axis=-1)
    E=rho**2/h[:,None]**2*x[None,:];E[meta['head']]=0
    original=np.asarray(meta['user_block_power_w'])*meta['frame_count']*c['block_dimension']/c['symbol_rate_hz']
    assert np.all(E <= original*(1+2e-8)+1e-18), 'Weighted Cauchy bound violated'
    p=E*c['symbol_rate_hz']/(meta['frame_count']*c['block_dimension'])
    out=dict(meta)
    out.update(physical_mode='constructed_combined_same_final_noise',
        construction='x_C=nu/(v_L+v_U); same final-noise covariance; pathwise Gaussian coupling',
        original_IQ_RF_energy_j=meta['energy_bound_j'],original_IQ_max_power_w=meta['max_power_w'],
        energy_bound_j=float(E.sum()),expected_energy_j=float(E.sum()),
        max_power_w=float(p.max()),max_user_energy_j=float(E.sum(1).max()),
        component_mse=[float(total.sum()),0.],block_variance=np.stack([total,np.zeros_like(total)]).tolist(),
        component_epsilon=[float(np.linalg.norm(meta['component_epsilon'])),0.],
        component_realized_error=[meta['realized_error_to_raw'],0.],
        user_block_power_w=p.tolist(),user_RF_energy_j=E.sum(1).tolist(),
        transceiver_x=np.stack([x,np.ones_like(x)]).tolist())
    return out


def binding_rows(root):
    rows=[]
    for mode in ['combined','iq']:
        for ds in ['CapERA','AirScape']:
            for f in sorted((root/mode/'cases/baseline'/(ds.lower()+'_seed0')/'adaptive_retrieval_fu/transactions').glob('*.json')):
                t=read(f);r=t['radio']
                if t['kind']!='joint':continue
                mse=np.array(r['component_mse']);eps=np.array(r['component_epsilon'])**2
                ratio=np.divide(mse,eps,out=np.zeros_like(mse),where=eps>0)
                target=np.array(r['normalized_distortion_limits'])
                dr=np.divide(r['distortion'],target,out=np.zeros_like(target),where=target>0)
                rows.append(dict(dataset=ds,mode=mode,sequence=t['sequence'],
                    FL_usage=ratio[0],FU_usage=ratio[1],C9_usage=float(dr.max(initial=0)),
                    n=r['frame_count'],RF_energy_j=r['energy_bound_j']))
    return rows


def run(case,mode):
    protocol=register();cfg=CASES[case];assert mode in cfg['modes']
    previous.OUT=OUT/case
    previous.register=lambda:protocol
    joint_count=0;call_count=0
    def module_factory(selected):
        mapped=case=='mapped_noise_control'
        module=transport('iq' if mapped else selected,cfg['ratio'])
        original=module.analog
        def analog(l,u,ch,c,seed,ell):
            nonlocal joint_count,call_count
            channels=ch if cfg['channel_gain_db']==0 else ch*10**(cfg['channel_gain_db']/20)
            received,meta=original(l,u,channels,c,seed,ell)
            index=call_count//180;ds=['CapERA','AirScape'][index] if index<2 else 'additional'
            if np.any(l) and np.any(u):
                j=joint_count%45
                if case=='balanced_replay' and j in [0,9,18,27,36]:
                    dest=OUT/'snapshots'/ds.lower();dest.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(dest/f'joint{j:02d}.npz',l=l,u=u,channels=channels,ell=ell)
                    write(dest/f'joint{j:02d}.json',dict(dataset=ds,joint_index=j,call=call_count+1,config=c,seed=seed))
                joint_count+=1
            call_count+=1
            if received is not None and mapped:
                meta=map_iq_to_combined(l,u,channels,c,meta)
            meta['diagnostic_case']=case;meta['FU_relative_tolerance_ratio']=cfg['ratio']
            meta['channel_gain_db']=cfg['channel_gain_db']
            return received,meta
        module.analog=analog
        return module
    previous.radio_module=module_factory
    previous.run(mode)


def queue():
    register()
    logs=OUT/'logs';logs.mkdir(exist_ok=True)
    rows=[]
    for case,cfg in CASES.items():
        for mode in cfg['modes']:
            completed=OUT/case/mode/'finished.json'
            if completed.exists():
                rows.append(dict(case=case,mode=mode,status='already_completed'));continue
            with (logs/(case+'_'+mode+'.log')).open('w') as f:
                result=subprocess.run([sys.executable,__file__,'run','--case',case,'--method',mode],
                    stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,MPLCONFIGDIR='/tmp/ova-mpl'))
            rows.append(dict(case=case,mode=mode,returncode=result.returncode,status='completed' if result.returncode==0 else 'failed'))
            write(OUT/'queue_status.json',rows);print(rows[-1],flush=True)
    write(OUT/'finished.json',dict(unix=time.time(),runs=rows))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['register','run','queue']);p.add_argument('--case',choices=list(CASES));p.add_argument('--method',choices=['combined','iq']);a=p.parse_args()
    if a.action=='register':print(register())
    elif a.action=='queue':queue()
    else:run(a.case,a.method)
