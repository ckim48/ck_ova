"""CPU feasibility screen on real paired round-61 updates; no model tuning."""
import json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(Path(__file__).parent),str(ROOT),str(ROOT/'tools/overnight_energy_v2'),
             str(ROOT/'tools/overnight_energy_v2/initialization_v11'),
             str(ROOT/'tools/overnight_energy_v2/receiver_control_v1')]
from trajectory_transport import communicate


def main():
    out=ROOT/'results/overnight_energy_v2/convergence_energy_delay_v1'
    c=json.loads((out/'plan.json').read_text())['config']
    rows=[]
    for method in ['proposed','aircomp_only','aircomp_oma','oma_only']:
        variant='crossmodal' if method=='proposed' else 'vanilla'
        src=ROOT/'results/overnight_energy_v2/crossmodal_ablation_v1/prepared'/variant/'k5/prepared.npz'
        with np.load(src) as z: l,u,h,g=[z[x].copy() for x in ['learning','unlearning','channels','gradient']]
        start=time.perf_counter()
        _,stats=communicate(l,u,h,c,'hybrid' if method=='proposed' else method,61021,g)
        rows.append(dict(method=method,elapsed_s=time.perf_counter()-start,**stats))
        (out/'preflight.json').write_text(json.dumps(rows,indent=2)+'\n')
        print(json.dumps({k:rows[-1].get(k) for k in ['method','status','energy_bound_j','uplink_s','elapsed_s']}),flush=True)


if __name__=='__main__':main()
