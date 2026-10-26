import unittest

import numpy as np

from adaptive_precision import budget_quantize, design_prequantized, exact_statistics
from phase_split_grid import make_inputs
import test_phase_split
from run_wireless import scenario
from transport import design_transport, recover


class AdaptivePrecisionTests(unittest.TestCase):
    def test_quantization_budget_and_headers(self):
        rng=np.random.default_rng(13)
        updates=rng.normal(0,0.01,(3,4,8))
        updates[0]=0
        decoded,delta,payload,depth=budget_quantize(updates,0.02,0.25)
        observed=np.linalg.norm(decoded-updates,axis=-1)
        self.assertTrue(np.all(observed<=delta+1e-14))
        self.assertLessEqual(np.linalg.norm(delta.sum(axis=0)),0.02*0.25*(1+1e-12))
        self.assertTrue(np.all(payload[0]==0))
        self.assertTrue(np.all(decoded[0]==0))
        self.assertTrue(np.all(payload[1:]==8*depth[1:]+35))

    def test_no_unlearning_reproduces_original_transport(self):
        c=test_phase_split.PhaseSplitTests().config()
        h,l,u,*_=scenario(c,0);u[:]=0
        l=l.reshape(4,-1);u=u.reshape(4,-1)
        blocked_l,blocked_u,*_=make_inputs(l,u,c)
        epsilon=0.6*np.linalg.norm(np.linalg.norm(blocked_l,axis=-1).sum(axis=0))
        decoded,delta,payload,_=budget_quantize(blocked_u,epsilon,0.25)
        expected,el,eu,ed,_=design_transport(l,u,h,c,'hybrid',epsilon)
        observed,_=design_prequantized(blocked_l,h,delta,payload,c,epsilon,'hybrid')
        self.assertIsNotNone(expected);self.assertIsNotNone(observed)
        self.assertAlmostEqual(expected.energy_j,observed.energy_j,places=10)
        np.testing.assert_allclose(recover(expected,el,eu,ed,np.random.default_rng(1)),
                                   recover(observed,blocked_l,blocked_u,decoded,np.random.default_rng(1)))

    def test_analytic_recovery_error_bounded(self):
        c=test_phase_split.PhaseSplitTests().config();h,l,u,*_=scenario(c,0)
        l=l.reshape(4,-1);u=u.reshape(4,-1)
        bl,bu,*_=make_inputs(l,u,c)
        epsilon=0.6*np.linalg.norm(np.linalg.norm(bl,axis=-1).sum(axis=0))
        decoded,delta,payload,_=budget_quantize(bu,epsilon,0.25)
        d,_=design_prequantized(bl,h,delta,payload,c,epsilon,'orthogonal',0.25)
        self.assertIsNotNone(d)
        stats=exact_statistics(d,bl,bu,decoded,l.shape[1])
        self.assertLessEqual(stats['analytic_total_rmse'],d.error_bound*(1+1e-4))
        self.assertLessEqual(stats['unlearning_component_error'],np.linalg.norm(delta[d.users].sum(axis=0)))


if __name__=='__main__':
    unittest.main()
