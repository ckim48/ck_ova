"""Independent physical-unit certificate for saved dual analog incumbents."""
import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,required=True)
    args=p.parse_args();root=args.project.resolve();out=root/'results/overnight_energy_v2/dual_analog_v9'
    plan=json.loads((out/'manifest.json').read_text());cfg=plan['config']
    for name,expected in plan['code_sha256'].items():
        assert hashlib.sha256((root/'tools/overnight_energy_v2/dual_analog_v9'/name).read_bytes()).hexdigest()==expected
    rows=json.loads((out/'heads.json').read_text()) if (out/'heads.json').exists() else []
    with np.load(root/'results/failed_round_diagnostic/updates.npz') as data:
        vectors=np.stack([data[k] for k in ['learning','unlearning']]);all_h=data['channels'].copy()
    dimension=vectors.shape[-1];block=cfg['block_dimension']
    blocked=np.pad(vectors,((0,0),(0,0),(0,(-dimension)%block))).reshape(2,cfg['users'],-1,block)
    sigma=np.sqrt(10**((cfg['noise_dbm']-30)/10))
    certificates=[]
    for row in rows:
        if row['status']!='feasible_radio_only':continue
        head=row['head'];fraction=row['learning_time_fraction'];relative=row['relative_tolerance']
        path=out/f'r{relative:g}_f{fraction:g}_h{head}.npz'
        with np.load(path) as data:
            a,s,n,users,rho=[data[k].copy() for k in ['analog_rms','receiver','symbols','users','rho']]
        true_rho=np.linalg.norm(blocked[:,users],axis=-1)
        np.testing.assert_allclose(rho,true_rho,rtol=1e-12,atol=1e-15)
        # Recover original |b|, |c| and |h|, then recompute equations in watts.
        b=np.divide(a*np.sqrt(n[:,None,:]),rho,out=np.zeros_like(a),where=rho>0)
        c=np.divide(s,sigma*np.sqrt(n),out=np.zeros_like(s),where=n>0)
        h=all_h[users,head,None]*sigma
        gain=c[:,None,:]*h[None,:,:]*b
        tx_block=b*b*true_rho*true_rho/cfg['symbol_rate_hz']
        phase_power=np.divide(tx_block*cfg['symbol_rate_hz'],n[:,None,:],out=np.zeros_like(tx_block),where=n[:,None,:]>0)
        assert np.max(phase_power)<=cfg['power_limit_w']*(1+1e-6)
        overhead=np.full(cfg['users'],cfg['base_overhead_j']);overhead[head]+=cfg['head_overhead_j']
        peruser=tx_block.sum(axis=(0,2))+overhead[users]
        assert np.max(peruser)<=cfg['available_energy_j']*(1+1e-6)
        joules=float(tx_block.sum()+overhead.sum())
        assert np.isclose(joules,row['energy_bound_j'],rtol=1e-8,atol=1e-12)
        assert n.sum()/cfg['symbol_rate_hz']<=plan['total_uplink_s']*(1+1e-10)
        assert np.all((n==0)|(n>=block))
        assert np.max(n/cfg['symbol_rate_hz'])<=cfg['coherence_time_s']
        deterministic=((gain-1)[...,None]*blocked[:,users]).sum(axis=(0,1)).reshape(-1)[:dimension]
        valid=np.minimum(block,np.maximum(dimension-np.arange(n.shape[1])*block,0))
        mse=float(deterministic@deterministic+np.sum(valid*sigma*sigma/2*np.sum(c*c,axis=0)))
        conservative=(rho*np.abs(gain-1)).sum(axis=(0,1))+np.sqrt(block*sigma*sigma/2*np.sum(c*c,axis=0))
        rms_bound=float(np.linalg.norm(conservative))
        assert np.isclose(np.sqrt(mse),row['analytic_sum_rmse'],rtol=1e-7,atol=1e-12)
        assert np.isclose(rms_bound,row['error_bound'],rtol=1e-7,atol=1e-12)
        assert np.sqrt(mse)<=rms_bound*(1+1e-5)+1e-12
        assert rms_bound<=row['epsilon']*(1+1e-5)
        assert all(y<=x+1e-9 for x,y in zip(row['history'],row['history'][1:]))
        certificates.append(dict(relative_tolerance=relative,learning_time_fraction=fraction,head=head,
                                 status='passed_physical_units',energy_j=joules,analytic_sum_rmse=float(np.sqrt(mse)),rms_bound=rms_bound))
    result=dict(status='all_available_incumbents_passed',at_kst=datetime.now().isoformat(timespec='seconds'),
                observed_head_designs=len(rows),checked_feasible_designs=len(certificates),certificates=certificates,
                verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                limits='Algebraic physical-unit certificate of recorded feasible incumbents; no real hardware measurement or VLM/deletion guarantee.')
    target=out/'physical_certificates.json';temporary=target.with_suffix('.tmp')
    temporary.write_text(json.dumps(result,indent=2)+'\n');temporary.replace(target)
    print(json.dumps({k:v for k,v in result.items() if k!='certificates'}))


if __name__=='__main__':main()
