"""Training-update-only numerical/physical gate; no reference/test tuning."""
from pathlib import Path
import sys,json,time,importlib.util,hashlib
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
OUT=ROOT/'results/overnight_energy_v2/fu_protected_aircomp_v1'
sys.path.insert(0,str(HERE))
import radio_guard as guard
spec=importlib.util.spec_from_file_location('guard_clip_parent',ROOT/'tools/overnight_energy_v2/cliperase_fu_v1/run.py')
clip=importlib.util.module_from_spec(spec);spec.loader.exec_module(clip)

def write(name,data):clip.core.write(OUT/name,data)
def prepare():
    plan,requests=clip.prepare();OUT.mkdir(parents=True,exist_ok=True)
    parent=clip.OUT/'protocol.json';hashes=dict(json.loads(parent.read_text())['source_hashes'])
    hashes[str(parent.relative_to(ROOT))]=clip.core.sha(parent)
    for p in HERE.glob('*.py'):hashes[str(p.relative_to(ROOT))]=clip.core.sha(p)
    fixed=clip.OUT/'fixed_training_updates.npz';hashes[str(fixed.relative_to(ROOT))]=clip.core.sha(fixed)
    protocol=dict(version='fu_protected_aircomp_v1',registered_unix=time.time(),source_hashes=hashes,
      parent='cliperase_fu_v1/parallel_cliperase_paper',
      sole_change='Use FU-relative cap on total analog reconstruction RMS whenever FU is active. Reuse coefficient0.2, all FL/FU vectors, loss weights, optimizer, seeds, physical caps and frame grid.',
      formula='eps_new=min(eps_old,max(absolute_floor,relative_tolerance*FU_block_norm_bound))',
      information='Same transmitted/update block-norm statistics already assumed by the original optimization; no exact aggregate FU direction or oracle reference is used.',
      energy_delay='Same0.05J per-transmission network cap, per-UAV power and energy caps. Delay minimized on same frame grid; changed update norms/precision may change service time and hence arrival-relative training schedule.',
      scope='Single proposed-only 200FL/45request trajectory if fixed-update physical gate passes. All comparison trajectories and original sources remain frozen.',
      gate='New CLIPErase full20480-coordinate first training update must be feasible under unchanged limits; zero-FU must be exactly unchanged; reconstruction expected sum unchanged; all checks finite. Old FU snapshot is diagnostic only and cannot select a different coefficient.',
      metrics='Keep retained validation mR/loss, mean/worst request retrain KL, prediction difference, MIA, procedure/deadline count and original joint threshold. Do not rank incomplete/quality-failed outcomes by energy-to-success.',
      limits='FU block norm upper bound can exceed the actual aggregate norm; this controls total noise, not independently identifiable FU error, individual requests, or a guarantee of forgetting. Numerical floor can dominate tiny FU. A stricter cap may cost more energy or time.',
      literature='Gradient-importance aware AirComp has precedent in Sun et al., IEEE TWC2024, doi10.1109/TWC.2023.3336277. This FU-relative total-noise rule is a new experimental adaptation, not a reproduction of that scheduling algorithm.',
      tuning='Coefficient fixed from the prior radio relative tolerance before reading candidate results; only saved training updates used in preflight, no final/reference/test selection.')
    if (OUT/'protocol.json').exists():protocol=json.loads((OUT/'protocol.json').read_text())
    else:write('protocol.json',protocol);write('requests.json',requests)
    for name,digest in protocol['source_hashes'].items():assert clip.core.sha(ROOT/name)==digest,name
    return plan,requests

def trial(l,u,ch,c,seed,protected):
    vec,meta=(guard.analog if protected else guard.original.analog)(l,u,ch,c,seed)
    if vec is not None:
        assert meta['conditional_rmse']<=meta['epsilon']*(1+3e-5)+1e-10
        assert meta['max_power_w']<=c['power_limit_w']*(1+3e-5)
        assert meta['max_user_energy_j']<=c['available_energy_j']*(1+3e-5)
        assert meta['energy_bound_j']<=guard.base.CAP*(1+3e-5)
        assert np.isfinite(vec).all()
        meta['realized_reconstruction_norm']=float(np.linalg.norm(vec-(l+u).sum(0)))
    return vec,meta

def main():
    plan,_=prepare();c=plan['config'];z=np.load(clip.OUT/'fixed_training_updates.npz')
    l,u,old,ch=z['FL'],z['FU'],z['old_FU'],z['channels']
    records={}
    for name,fu in [('cliperase_FU',u),('previous_FU',old)]:
        records[name]={}
        for protected in [False,True]:
            v,m=trial(l,fu,ch,c,720000,protected)
            records[name]['protected' if protected else 'original']=m
            print(name,'protected' if protected else 'original',json.dumps(m),flush=True)
    v0,m0=trial(l,np.zeros_like(u),ch,c,720000,False)
    v1,m1=trial(l,np.zeros_like(u),ch,c,720000,True)
    assert np.array_equal(v0,v1)
    for k in ['energy_bound_j','uplink_s','epsilon','conditional_rmse','head']:assert m0[k]==m1[k]
    # Replicated stochastic recovery checks: optimizer/noise draws use only the same training snapshot.
    errors=[];bias=np.zeros(l.shape[1]);raw=(l+u).sum(0)
    # The base uses exact channel inversion. Obtain exact expectation with zero sampled noise.
    saved=guard.base.np.random.default_rng
    class ZeroNoise:
        def normal(self,loc,scale,size):return np.zeros(size)
    try:
        guard.base.np.random.default_rng=lambda seed:ZeroNoise()
        mean,mm=guard.analog(l,u,ch,c,720000)
    finally:guard.base.np.random.default_rng=saved
    assert mean is not None and np.allclose(mean,raw,rtol=1e-11,atol=1e-12)
    for scale in [.1,1.,10.]:
        nl,nu,eps,target=guard.budget(scale*l,scale*u,c)
        assert np.isclose(target,scale*guard.budget(l,u,c)[3])
    gate=records['cliperase_FU']['protected']['status']=='feasible'
    write('preflight.json',dict(passed=gate,unix=time.time(),config=c,dimension=int(l.shape[1]),
      records=records,no_FU_bitwise_identical=True,noiseless_sum_preserved=True,budget_scale_equivariance=True,
      old_fu_snapshot_is_diagnostic_only=True,policy='Use original0.2 coefficient; no sweep and no reference/test tuning.'))
    print('PREFLIGHT_PASSED',gate,flush=True)

if __name__=='__main__':
    try:main()
    finally:guard.base.close()
