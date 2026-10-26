from pathlib import Path
import json,time,hashlib
import radio_parallel as r
import numpy as np
ROOT=r.ROOT;HERE=Path(__file__).resolve().parent;OUT=ROOT/'results/overnight_energy_v2/parallel_aircomp_v1'

def main():
    OUT.mkdir(parents=True,exist_ok=True);started=time.time()
    z=np.load(ROOT/'results/overnight_energy_v2/retrieval_fourway_v2/fixed_updates.npz')
    c=json.loads((ROOT/'results/overnight_energy_v2/retrieval_long200_v3/plan.json').read_text())['config']
    results={};vectors={}
    assert r.old.ofdma.checks()['passed']
    # With inactive caps, Cauchy--Schwarz gives E*= (sum sqrt(c_m*v_m))^2/eps^2.
    toy_c=dict(c,power_limit_w=100.,available_energy_j=100.)
    toy_h=np.ones((3,3))*20.;toy=np.array([[.01,.02],[.02,.01],[.015,.01]])
    dims=np.array([128,128]);eps=.02
    a=r.exact_aligned(0,toy_h,[toy],dims,[1],eps,toy_c)[0]
    co=(toy[1:]**2/400).sum(0);v=dims/(2*c['symbol_rate_hz'])
    expected=float(np.sqrt(co*v).sum()**2/eps**2)
    assert np.isclose(a.energy_j,expected,rtol=2e-4,atol=1e-10)
    aa=r.exact_aligned(0,toy_h,[toy,toy],dims,[1,2],eps,toy_c)
    expected2=float(np.sqrt(np.tile(co*v,2)).sum()**2/eps**2)
    assert np.isclose(sum(x.energy_j for x in aa),expected2,rtol=2e-4,atol=1e-10)
    for mode in ['joint','conventional_joint','separate','ofdma']:
        (OUT/'preflight_status.json').write_text(json.dumps(dict(stage='fixed_update_radio',mode=mode,elapsed_s=time.time()-started)))
        vector,meta=r.communicate(z['learning'],z['unlearning'],z['channels'],c,mode,81001)
        assert vector is not None,mode
        vectors[mode]=vector;results[mode]=meta
        print(json.dumps(dict(mode=mode,energy_j=meta['energy_bound_j'],uplink_s=meta['uplink_s'],head=meta['head'])),flush=True)
    assert np.array_equal(vectors['joint'],vectors['conventional_joint'])
    assert results['joint']['energy_bound_j']==results['conventional_joint']['energy_bound_j']
    assert results['joint']['uplink_s']==results['conventional_joint']['uplink_s']
    # No-FU phases must collapse to the same joint aggregation, not a padded extra slot.
    nofu={}
    for mode in ['joint','separate','ofdma']:
        v,m=r.communicate(z['learning'],np.zeros_like(z['unlearning']),z['channels'],c,mode,81002)
        assert v is not None;nofu[mode]=(v,m)
    for mode in ['separate','ofdma']:
        assert np.array_equal(nofu['joint'][0],nofu[mode][0])
        for key in ['energy_bound_j','uplink_s']:assert nofu['joint'][1][key]==nofu[mode][1][key]
    # Sum before/after ideal transport is algebraically equal; no manufactured collision.
    assert np.allclose((z['learning']+z['unlearning']).sum(0),z['learning'].sum(0)+z['unlearning'].sum(0),atol=1e-15)
    result=dict(passed=True,runtime_s=time.time()-started,coordinates=int(z['learning'].shape[1]),results=results,
        identity='Proposed and conventional joint exactly identical in vector, energy and uplink duration; represented as one trajectory with an explicit alias.',
        no_fu_equivalence=True,conditional_RMS_power_energy_rate_audits=True,ofdma_integer_checks=True,convex_energy_matches_closed_form=True,
        scope='Full-dimensional fixed-update communication preflight, not convergence, forgetting success or learned-quality ranking.',
        new_source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')})
    (OUT/'preflight.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(passed=True,runtime_s=result['runtime_s'])),flush=True)

if __name__=='__main__':
    try:main()
    finally:r.close()
