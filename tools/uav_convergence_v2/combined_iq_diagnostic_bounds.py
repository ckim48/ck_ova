"""Solver-independent necessary RF-energy and peak-power feasibility bounds."""
from pathlib import Path
import numpy as np
import combined_iq_diagnostic_v1 as study


def bounds(l,u,channels,c,ell,mode,ratio):
    module=study.transport(mode,ratio)
    rho,eps,_,_=module.inputs(l,u,c,ell)
    ns=[n for n in module.COUNTS if n*l.shape[1]/c['symbol_rate_hz']<=c['round_uplink_deadline_s']+1e-12
        and n*c['block_dimension']/c['symbol_rate_hz']<=c['coherence_time_s']+1e-12]
    nmax=max(ns);nu=c['block_dimension']/(2*c['symbol_rate_hz']);rows=[]
    for head in range(len(channels)):
        h=np.abs(channels[:,head]).copy();h[head]=1
        A=rho**2/np.maximum(h[None,:,None]**2,1e-300);A[:,head]=0
        rf=0.;needed_n=0.
        for a in range(2):
            if eps[a]==0:continue
            # Min sum_m A_m*x_m s.t. sum_m nu/x_m <= eps_a^2.
            rf+=(np.sqrt(A[a].sum(0)*nu).sum())**2/eps[a]**2
            # Even assigning each entire block's power to task a gives this lower bound.
            needed_n=max(needed_n,A[a].max(0).sum()/(2*c['power_limit_w']*eps[a]**2))
        rows.append(dict(head=head,RF_energy_lower_bound_j=float(rf),
            expansion_lower_bound=float(needed_n),max_expansion=nmax,
            necessarily_infeasible=bool(rf>.1*(1+1e-6) or needed_n>nmax*(1+1e-6))))
    return rows


def main():
    source=study.OUT/'snapshot_sweep/results.json';results=study.read(source);rows=[]
    for r in results:
        z=np.load(study.ROOT/r['source']);info=study.read((study.ROOT/r['source']).with_suffix('.json'))
        head=bounds(z['l'],z['u'],z['channels']*10**(r['gain_db']/20),info['config'],z['ell'],r['mode'],r['tolerance_ratio'])
        certified=all(h['necessarily_infeasible'] for h in head)
        if r['status']=='feasible':assert not certified
        rows.append(dict(dataset=r['dataset'],joint_index=r['joint_index'],mode=r['mode'],gain_db=r['gain_db'],
            tolerance_ratio=r['tolerance_ratio'],solver_status=r['status'],
            ruled_out_by_necessary_bound=certified,
            minimum_RF_lower_bound_j=min(h['RF_energy_lower_bound_j'] for h in head),
            minimum_expansion_lower_bound=min(h['expansion_lower_bound'] for h in head),heads=head))
    failed=[r for r in rows if r['solver_status']!='feasible']
    summary=dict(passed=True,points=len(rows),failed_solver_points=len(failed),
        failures_proven_infeasible_by_necessary_bounds=sum(r['ruled_out_by_necessary_bound'] for r in failed),
        failures_not_resolved_by_these_bounds=sum(not r['ruled_out_by_necessary_bound'] for r in failed),
        source_sha256=study.sha(Path(__file__)))
    study.write(study.OUT/'snapshot_sweep/feasibility_bounds.json',rows)
    study.write(study.OUT/'snapshot_sweep/bounds_summary.json',summary)
    print(summary)


if __name__=='__main__':main()
