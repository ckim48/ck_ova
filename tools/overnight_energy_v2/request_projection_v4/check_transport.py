"""CPU integration check on frozen real-update snapshot, not new VLM evidence."""
from pathlib import Path
import sys,json,time
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import run as experiment
import numpy as np
core=experiment.core
ROOT=experiment.ROOT
sys.path.insert(0,str(ROOT/'tools/overnight_energy_v2/proxy_objective_v2'))
import parallel_heads


def main():
    plan=json.loads((experiment.SOURCE/'plan.json').read_text());c=plan['config']
    for name,h in plan['frozen_engine_hashes'].items():assert core.sha(core.FROZEN/name)==h
    z=np.load(ROOT/'results/overnight_energy_v2/retrieval_fourway_v2/fixed_updates.npz')
    l,u=z['learning'],z['unlearning'];q,eps=experiment.canonical(l,u,c)
    fixed_c=dict(c,absolute_tolerance_floor=eps,relative_tolerance=0.)
    m=core.load_module('projection_check_transport',core.LONG/'radio_partial.py')
    import radio
    radio.optimize_head=parallel_heads.parallel;m.optimize_head=parallel_heads.parallel
    rows=[]
    for method in ['hybrid','aircomp_only','aircomp_oma','oma_only','aircomp_local_project']:
        t0=time.time()
        if method=='aircomp_local_project':
            loc,_=experiment.controller.local_project(l,q)
            vec,meta=m.communicate(loc,np.zeros_like(u),z['channels'],fixed_c,'aircomp_only',100021)
        else:vec,meta=m.communicate(l,u,z['channels'],fixed_c,method,100021)
        assert vec is not None,method
        assert meta['error_bound']<=eps*(1+3e-5)
        assert meta['energy_bound_j']>0 and 0<meta['uplink_s']<=c['round_uplink_deadline_s']
        row=dict(method=method,feasible=True,energy_bound_j=meta['energy_bound_j'],uplink_s=meta['uplink_s'],
                 error_bound=meta['error_bound'],epsilon=eps)
        if method in ['hybrid','aircomp_oma','oma_only']:
            assert meta['individual_request_vectors_recovered']
            a,di=experiment.controlled(l,u,q,'request_project',received=vec)
            # Receiver must not use untransmitted true local vectors, even indirectly.
            b,_=experiment.controlled(np.zeros_like(l),np.zeros_like(u),q,'request_project',received=vec)
            assert np.array_equal(a,b)
            assert np.isfinite(a).all()
            row.update(no_privileged_receiver_input=True,correction_ratio=di['correction_ratio'],
                       true_harmed=di['true_objectives_harmed_first_order'])
        row['wall_s']=time.time()-t0;rows.append(row);print(json.dumps(row),flush=True)
    for name,h in plan['frozen_engine_hashes'].items():assert core.sha(core.FROZEN/name)==h
    core.write(experiment.OUT/'transport_checks.json',dict(passed=True,checks=rows,
      input_sha256=core.sha(ROOT/'results/overnight_energy_v2/retrieval_fourway_v2/fixed_updates.npz'),
      scope='Frozen previous real updates; CPU wiring/observability/constraint test only, not new-objective or learning performance.'))

if __name__=='__main__':
    try:main()
    finally:parallel_heads.close()
