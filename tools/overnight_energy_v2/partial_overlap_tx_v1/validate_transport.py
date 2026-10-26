"""Check actual transport and metric integration before launching GPU training."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path[:0]=[str(HERE),str(ROOT),str(ROOT/'tools/overnight_energy_v2'),
             str(ROOT/'tools/overnight_energy_v2/initialization_v11'),
             str(ROOT/'tools/overnight_energy_v2/receiver_control_v1')]
from trajectory_transport import communicate
from report_tx_only import reaccount

def main():
    out=ROOT/'results/overnight_energy_v2/partial_overlap_tx_v1'
    c=json.loads((out/'plan.json').read_text())['config']
    with np.load(ROOT/'results/overnight_energy_v2/crossmodal_ablation_v1/prepared/crossmodal/k5/prepared.npz') as z:
        learning,unlearning,channels,gradient=[z[k].copy() for k in ['learning','unlearning','channels','gradient']]
    checks=[]
    for method,delete in [('hybrid',True),('aircomp_oma',True),('aircomp_only',True),('hybrid',False)]:
        u=unlearning if delete else np.zeros_like(unlearning)
        update,t=communicate(learning,u,channels,c,method,61,gradient)
        assert update is not None and update.shape==(learning.shape[1],) and np.all(np.isfinite(update))
        assert t['application_error_bound']<=t['epsilon']*(1+2e-5)
        target_time=.293888 if method=='hybrid' and delete else .335872 if method=='aircomp_oma' else .041984
        assert abs(t['uplink_s']-target_time)<1e-10
        if method=='hybrid' and delete:
            expected=json.loads((out/'positive_overlap/result.json').read_text())['best']
            assert abs(t['energy_bound_j']-expected['tx_energy_bound_j'])<1e-6
            assert t['payload_overlap_fraction']>0 and t['individual_requests_delivered']==5
        checks.append(dict(method=method,request_round=delete,energy_bound_j=t['energy_bound_j'],uplink_s=t['uplink_s'],status='passed'))
    dummy=dict(round=61,validation=dict(loss=1.,accuracy=.7),request_users_total=5,individual_request_deliveries_total=5,
               transmission=dict(status='feasible',energy_bound_j=.1,expected_energy_j=.09,uplink_s=.294),optimizer_compute_s=1e6)
    a=reaccount([dummy],.001)[0]
    assert abs(a['total_delay_s']-.294)<1e-12 and abs(a['tx_energy_bound_j']-.099)<1e-12
    dummy['optimizer_compute_s']=0
    assert a==reaccount([dummy],.001)[0]
    result=dict(status='passed',checks=checks,metric_test='Reconstruction/solver wall time cannot change total_delay or transmit energy.',
                code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')})
    (out/'transport_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
