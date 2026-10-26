"""Offline full-gradient oracle floor for a separate linear U AirComp stream.

This grants oracle update geometry and maximum U duration, relaxes learning
quality/battery, and permits signed real effective coefficients. The oracle is
not an implementable free-information comparator. Its convex quadratic dual
provides a numerical lower certificate for this specified receiver class.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import unittest
import cvxpy as cp
import numpy as np


def oracle(vectors,channel,peak,symbols):
    vectors=np.asarray(vectors);rho=np.linalg.norm(vectors,axis=1);active=rho>0
    if not np.any(active):return dict(primal_mse=0.,dual_lower_mse=0.,relative_dual_gap=0.)
    v=vectors[active];rho=rho[active];h=np.asarray(channel)[active]
    scale=float(rho.sum());normalized=v/rho[:,None];target=v.sum(axis=0)/scale
    gain=h*np.sqrt(peak);nu=np.sqrt(v.shape[1]/(2*symbols))
    t=cp.Variable(len(v));receiver=cp.Variable(nonneg=True)
    upper=t<=gain*receiver;lower=-t<=gain*receiver
    objective=cp.sum_squares(normalized.T@t-target)+nu*nu*cp.square(receiver)
    problem=cp.Problem(cp.Minimize(objective),[upper,lower])
    problem.solve(solver='CLARABEL',max_iter=200,tol_gap_abs=1e-10,tol_feas=1e-10,tol_gap_rel=1e-10)
    assert problem.status=='optimal' and t.value is not None
    tv=np.asarray(t.value);r=max(float(receiver.value),0)
    # Enforce feasibility explicitly before treating the primal as an upper bound.
    r=max(r,float(np.max(np.divide(np.abs(tv),gain,out=np.zeros_like(gain),where=gain>0))))
    assert np.all(np.abs(tv)<=gain*r+1e-8)
    primal=float(np.linalg.norm(normalized.T@tv-target)**2+(nu*r)**2)
    gram=normalized@normalized.T;linear=normalized@target
    lp=np.maximum(np.asarray(upper.dual_value),0);lm=np.maximum(np.asarray(lower.dual_value),0)
    eigenvalues=np.linalg.eigvalsh(gram)
    if eigenvalues.min()>1e-9:
        c=lp-lm-2*linear
        term1=float(c@np.linalg.solve(gram,c));term2=float(gain@(lp+lm))**2/(nu*nu)
        raw=float(target@target-.25*(term1+term2))
        # Numerical margin, not interval arithmetic or an all-analog impossibility theorem.
        dual=max(0.,raw-1e-8*(1+abs(term1)+abs(term2)))
        certificate='quadratic_lagrange_dual_with_nonnegative_multipliers_and_numerical_margin'
    else:
        dual=0.;certificate='zero_lower_bound_for_near_singular_geometry'
    assert dual<=primal+1e-7
    return dict(primal_mse=primal*scale*scale,dual_lower_mse=dual*scale*scale,
                relative_dual_gap=float((primal-dual)/max(primal,1e-30)),
                minimum_gram_eigenvalue=float(eigenvalues.min()),certificate=certificate)


class OracleTests(unittest.TestCase):
    def test_single_link_closed_form_and_dual(self):
        rho=.1;h=2.;peak=.32;n=8;noise_variance=1/(2*n)
        expected=rho*rho*noise_variance/(h*h*peak+noise_variance)
        result=oracle([[rho]],[h],peak,n)
        self.assertAlmostEqual(result['primal_mse'],expected,delta=expected*1e-7)
        self.assertLessEqual(result['dual_lower_mse'],expected*(1+1e-7))
        self.assertGreater(result['dual_lower_mse'],expected*(1-1e-5))
    def test_two_stream_and_zero_cases(self):
        result=oracle([[.1,0],[0,.2]],[2,3],.32,8)
        self.assertGreater(result['dual_lower_mse'],0)
        self.assertLess(result['relative_dual_gap'],1e-4)
        self.assertEqual(oracle([[0,0]],[2],.32,8)['primal_mse'],0)
    def test_longer_time_improves_floor(self):
        a=oracle([[.1,.03],[.02,.2]],[2,3],.32,8)
        b=oracle([[.1,.03],[.02,.2]],[2,3],.32,32)
        self.assertLess(b['primal_mse'],a['primal_mse'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path)
    p.add_argument('--self-test',action='store_true');args=p.parse_args()
    if args.self_test:
        result=unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(OracleTests))
        if not result.wasSuccessful():raise SystemExit(1)
        return
    assert args.project is not None
    root=args.project.resolve();source=root/'results/failed_round_diagnostic'
    c=json.loads((source/'metadata.json').read_text())['config']
    out=root/'results/overnight_energy_v2/component_fidelity_v10'
    conservative=json.loads((out/'results.json').read_text())
    with np.load(source/'updates.npz') as a:l,u,h=[a[k].copy() for k in ['learning','unlearning','channels']]
    d=u.shape[-1];b=c['block_dimension'];padding=((0,0),(0,(-d)%b))
    learning=np.pad(l,padding).reshape(len(l),-1,b);unlearning=np.pad(u,padding).reshape(len(u),-1,b)
    rows=[]
    for head in range(c['users']):
        users=np.delete(np.arange(c['users']),head);parts=[]
        for m in range(unlearning.shape[1]):
            valid=min(b,d-m*b);n=b*c['symbols_per_coordinate']-b*int(np.any(learning[users,m]!=0))
            item=oracle(unlearning[users,m,:valid],h[users,head],c['power_limit_w'],n)
            parts.append(item)
        primal=float(np.sqrt(sum(x['primal_mse'] for x in parts)))
        dual=float(np.sqrt(sum(x['dual_lower_mse'] for x in parts)))
        rows.append(dict(head=head,oracle_primal_u_rmse=primal,numerical_dual_lower_u_rmse=dual,
                         maximum_block_relative_dual_gap=max(x['relative_dual_gap'] for x in parts),
                         reference_codecs={bits:{'tau':spec['global_all_client_quantization_rms_bound'],
                           'lower_over_tau':dual/spec['global_all_client_quantization_rms_bound'],
                           'numerical_lower_exceeds_tau':dual>spec['global_all_client_quantization_rms_bound']} for bits,spec in conservative['reference_codecs'].items()},
                         blocks=parts))
    report=dict(version='component_fidelity_v10_full_geometry_oracle_relaxation',at_kst=datetime.now().isoformat(timespec='seconds'),
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                updates_sha256=hashlib.sha256((source/'updates.npz').read_bytes()).hexdigest(),
                reference_codec_report_sha256=hashlib.sha256((out/'results.json').read_bytes()).hexdigest(),heads=rows,
                limits=['Specified single-linear-receiver, separately recovered U AirComp class only; not all analog communication or a VLM deletion guarantee.',
                        'Oracle gets full true update geometry without paying signaling, maximum U time, and no learning-quality/battery constraint. This deliberately optimistic relaxation is not an implementable zero-cost method.',
                        'Signed real effective coefficients and optimal receiver allowed. Independent Gaussian noise uses valid parameter dimensions, ignoring padded coordinates.',
                        'Numerical convex primal and explicit nonnegative-multiplier quadratic dual with roundoff margin; not interval-arithmetic proof.',
                        'Separate-U target is absent from the original sum-MSE problem. Existing codec thresholds do not themselves establish an application need for that precision.',
                        'Original results stay visible; optimized digital-phase comparator can meet the codec precision too and must remain in energy comparisons. No NOMA-over-phase energy conclusion.'])
    path=out/'oracle_results.json';assert not path.exists();path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'heads':[{k:v for k,v in x.items() if k!='blocks'} for x in rows]}))


if __name__=='__main__':main()
