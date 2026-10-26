import unittest
import numpy as np
from solver import refine


class InverseBudgetChecks(unittest.TestCase):
    def test_one_link_matches_analytic_energy_minimum(self):
        seed=dict(chi=0.,normalized_channels=np.array([[20.]]),rho=np.array([[1.]]),quantization_bound=np.zeros((1,1)),
            symbols=np.array([512]),digital_symbols=np.array([512]),dimensions=np.array([512]),
            analog_rms=np.array([[.25]]),digital_sqrt_power=np.zeros((1,1)),receiver=np.array([.15]),mode=np.array('mixed_aircomp'))
        cfg=dict(symbol_rate_hz=1e6,power_limit_w=1.,base_overhead_j=0.,available_energy_j=1.)
        result,log=refine(seed,np.zeros((1,1)),cfg)
        noise=np.sqrt(.5);budget=abs(.15*20*.25-1)+.15*noise;exact=noise/(20*budget)
        self.assertTrue(log['problem_is_dcp']);self.assertAlmostEqual(result['analog_rms'][0,0],exact,places=6)
        self.assertAlmostEqual(float(result['error_bound']),budget,places=7)

    def test_zero_receiver_and_nonzero_sic_are_handled_explicitly(self):
        seed=dict(chi=0.,normalized_channels=np.ones((1,1)),rho=np.ones((1,1)),quantization_bound=np.zeros((1,1)),
            symbols=np.array([512]),digital_symbols=np.array([512]),dimensions=np.array([512]),analog_rms=np.array([[.1]]),
            digital_sqrt_power=np.zeros((1,1)),receiver=np.zeros(1),mode=np.array('mixed_aircomp'))
        cfg=dict(symbol_rate_hz=1e6,power_limit_w=1.,base_overhead_j=0.,available_energy_j=1.)
        result,_=refine(seed,np.zeros((1,1)),cfg)
        np.testing.assert_array_equal(result['receiver'],0.);np.testing.assert_array_equal(result['analog_rms'],0.)
        seed['chi']=.01
        with self.assertRaises(ValueError):refine(seed,np.zeros((1,1)),cfg)


if __name__=='__main__':unittest.main()
