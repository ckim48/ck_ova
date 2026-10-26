"""CPU audit of overlap scheduling on unchanged RSITMD first-request updates."""
import sys, json, time, hashlib, os
from pathlib import Path
import numpy as np

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT/'tools/overnight_energy_v2/retrieval_fourway_v2'))
import common
from phase_split_grid import make_inputs
sys.path.insert(0, str(ROOT/'tools/overnight_energy_v2/partial_overlap_tx_v1'))
from overlap_solver import optimize_partial

OUT = ROOT/'results/overnight_energy_v2/retrieval_overlap_audit_v1'
SRC = ROOT/'results/overnight_energy_v2/retrieval_fourway_v2'

def write(name, obj):
    tmp = OUT/(name+'.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')
    tmp.replace(OUT/name)

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/'started.json').open('x') as f:
        json.dump(dict(pid=os.getpid(), unix=time.time()), f)
    config = json.loads((SRC/'plan.json').read_text())['config']
    config = dict(config, symbols_per_coordinate=32)
    with np.load(SRC/'fixed_updates.npz') as z:
        learning, unlearning, channels = [z[k].copy() for k in ('learning','unlearning','channels')]
    l,u,decoded,delta,payload,rho,dims,total = make_inputs(learning,unlearning,config)
    eps = max(config['absolute_tolerance_floor'],config['relative_tolerance']*float(np.linalg.norm(rho.sum(0))))
    la = 2*dims
    cases = [
        ('partial_positive_shorter',total-la,(.025,.05,.1,1/7,.25)),
        ('partial_positive_full_allowance',total,(.025,.05,.1,1/7,.25)),
        ('separated_aircomp_then_noma_ablation',total-la,(0.,)),
    ]
    write('plan.json',dict(created_unix=time.time(),config=config,epsilon=eps,
        updates_sha256=sha(SRC/'fixed_updates.npz'),script_sha256=sha(__file__),
        solver_sha256=sha(ROOT/'tools/overnight_energy_v2/partial_overlap_tx_v1/overlap_solver.py'),
        common_maximum_uplink_s=float(total.sum()/config['symbol_rate_hz']),
        cases=[dict(name=n,occupied_uplink_s=float(ld.sum()/config['symbol_rate_hz']),payload_overlap_grid=list(g)) for n,ld,g in cases],
        scope='CPU fixed identical updates, canonical Q8 target, same channel, RMS, power and deadline. No training, test evaluation, receiver controller or crossmodal objective change.',
        qualifications=['Reuses the previous partial-overlap solver unchanged; finite starts and AO do not prove global optimality.',
        'Current full-block Hybrid and optimized AirComp+TDMA baselines are retained unchanged.',
        'Zero overlap uses digital NOMA in its separate phase, so is an ablation rather than the OMA+AirComp baseline.',
        'Expected energy reaccounts the bound-optimized design; neither quantity is measured hardware energy.',
        'Smaller occupied time is not itself proof of smaller transmit energy or target-quality cumulative cost.']))
    baseline = [r for r in json.loads((SRC/'physical_preflight.json').read_text()) if r['multiplier']==32]
    write('baseline.json',baseline)
    rows=[]
    for name,ld,grid in cases:
        best=None; heads=[]; began=time.perf_counter()
        for head in range(len(channels)):
            write('status.json',dict(stage='running',case=name,head=head,completed_cases=len(rows),pid=os.getpid(),unix=time.time()))
            d=optimize_partial(head,channels,rho,payload,dims,la,ld,eps,config,split_grid=grid)
            row=dict(head=head,status='no_feasible_design_found')
            if d is not None:
                assert d['passed'] and d['error_bound']<=eps*(1+2e-5)
                row.update(status='feasible',energy_bound_j=d['tx_energy_bound_j'],overlap_fraction=d['payload_overlap_fraction'])
                if best is None or d['tx_energy_bound_j']<best['tx_energy_bound_j']:best=d
            heads.append(row)
            write(name+'_heads.json',heads)
        result=dict(method=name,status='no_feasible_design_found',heads=heads,
                    occupied_uplink_s=float(ld.sum()/config['symbol_rate_hz']),optimizer_wall_s=time.perf_counter()-began)
        if best is not None:
            result.update({k:v for k,v in best.items() if not isinstance(v,np.ndarray)},status='feasible')
            np.savez_compressed(OUT/(name+'_design.npz'),**{k:v for k,v in best.items() if isinstance(v,np.ndarray)})
        rows.append(result);write('results.json',rows)
        print(json.dumps({k:v for k,v in result.items() if k!='heads'}),flush=True)
    write('status.json',dict(stage='completed',pid=os.getpid(),unix=time.time(),completed_cases=len(rows)))

if __name__=='__main__':main()
