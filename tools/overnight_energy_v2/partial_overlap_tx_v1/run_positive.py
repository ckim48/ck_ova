"""Require actual digital traffic during the AirComp overlap, not a zero-overlap fallback."""
from pathlib import Path
import json, time
import numpy as np
import run_diagnostic as base

def main():
    out=base.OUT/'positive_overlap';out.mkdir(exist_ok=False)
    plan=json.loads((base.OUT/'plan.json').read_text());c=plan['config']
    source=base.ROOT/'results/overnight_energy_v2/crossmodal_ablation_v1/prepared/crossmodal/k5/prepared.npz'
    with np.load(source) as z:
        learning,unlearning,channels,gradient=[z[k].copy() for k in ['learning','unlearning','channels','gradient']]
    _,_,_,_,bits,rho,dims,total=base.make_inputs(learning,unlearning,c)
    la=total*.125;ld=total-la;eps=plan['application_error_budget']/plan['controller_lipschitz']
    plan['payload_split_grid']=[.025,.05,.1,1/7,.25]
    plan['zero_overlap_policy']='A zero-overlap digital solution is reported only as a separate AirComp then NOMA ablation, never as simultaneous transmission.'
    base.write(out/'plan.json',plan);rows=[];best=None
    for head in [5]+[i for i in range(len(channels)) if i!=5]:
        base.write(out/'progress.json',dict(status='running',head=head,completed_heads=len(rows)))
        start=time.perf_counter()
        d=base.optimize_partial(head,channels,rho,bits,dims,la,ld,eps,c,split_grid=plan['payload_split_grid'])
        row=dict(head=head,status='feasible' if d else 'no_feasible_design_found',solver_wall_s_excluded=time.perf_counter()-start)
        if d:
            assert d['payload_overlap_fraction']>0 and d['digital_overlap_tx_energy_j']>0
            row.update({k:v for k,v in d.items() if not isinstance(v,np.ndarray)})
            row['application_error_bound']=d['error_bound']*plan['controller_lipschitz']
            if best is None or row['tx_energy_bound_j']<best['tx_energy_bound_j']:
                best=row.copy();np.savez_compressed(out/'best_design.npz',**{k:v for k,v in d.items() if isinstance(v,np.ndarray)})
        rows.append(row);base.write(out/'head_results.json',rows)
        base.write(out/'result.json',dict(best=best,total_delay_s=plan['overlap_time_s'],completed_heads=len(rows)))
        print(json.dumps(row),flush=True)
    comparison=json.loads((base.OUT/'comparison.json').read_text())
    for r in comparison:
        if r['method']=='partial_overlap_tx':
            r['method']='AirComp_then_digital_NOMA_zero_overlap_ablation'
    if best:comparison.append(dict(method='partial_overlap_NOMA_AirComp',total_delay_s=plan['overlap_time_s'],**best))
    base.write(out/'comparison.json',comparison)
    base.write(out/'progress.json',dict(status='completed',completed_heads=len(rows),feasible=best is not None))

if __name__=='__main__':main()
