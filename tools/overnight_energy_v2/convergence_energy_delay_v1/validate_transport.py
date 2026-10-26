"""Independent resource and recovery checks on small analytic test cases."""
import json
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(Path(__file__).parent),str(ROOT),str(ROOT/'tools/overnight_energy_v2'),
             str(ROOT/'tools/overnight_energy_v2/initialization_v11'),
             str(ROOT/'tools/overnight_energy_v2/receiver_control_v1')]
from trajectory_transport import oma_solve, communicate


def main():
    c=dict(users=3,block_dimension=8,symbols_per_coordinate=16,quantization_bits=8,
           scale_metadata_bits=32,symbol_rate_hz=1e6,power_limit_w=.32,available_energy_j=.1,
           base_overhead_j=1e-5,head_overhead_j=5e-5,relative_tolerance=.2,absolute_tolerance_floor=1e-8,
           sic_residual_fraction=0.,round_uplink_deadline_s=.001,coherence_time_s=.01,
           receive_initializations=2,max_ao_iterations=2)
    x=oma_solve(np.full((2,1),64.),np.full(2,10.),np.array([128]),c,np.zeros(2))
    assert x is not None
    exact=2*64*.01/1e6
    assert abs(x['energy_by_user'].sum()/exact-1)<.002
    assert np.allclose(x['fractions'],.5,atol=.002)
    assert oma_solve(np.full((2,1),64.),np.full(2,10.),np.array([1]),c,np.zeros(2)) is None
    rng=np.random.default_rng(76021)
    l=rng.normal(size=(3,16))*1e-3;u=np.zeros_like(l);u[0]=rng.normal(size=16)*2e-4
    h=np.full((3,3),20.);np.fill_diagonal(h,0)
    g=rng.normal(size=16)
    results=[]
    for mode in ['hybrid','aircomp_only','aircomp_oma','oma_only']:
        update,stats=communicate(l,u,h,c,mode,27,g)
        assert update is not None, (mode,stats)
        assert update.shape==(16,) and np.isfinite(update).all()
        assert 0 < stats['energy_bound_j'] <= .1*3
        assert stats['application_error_bound'] <= stats['epsilon']*(1+1e-5)
        assert stats['individual_requests_delivered']==(0 if mode=='aircomp_only' else 1)
        assert abs(stats['uplink_s']-.000256)<1e-12
        results.append(dict(method=mode,energy_j=stats['energy_bound_j'],bound=stats['application_error_bound']))
    print(json.dumps(dict(status='passed',analytic_oma_energy_j=exact,schemes=results)),flush=True)


if __name__=='__main__':main()
