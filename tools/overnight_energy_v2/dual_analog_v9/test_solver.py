import unittest
import numpy as np
from solver import allocate_symbols,bounds,optimize,exact_statistics


class DualAnalogTests(unittest.TestCase):
    def test_time_mapping_and_inactive_stream(self):
        rho=np.ones((2,3,2));rho[1,:,1]=0
        n=allocate_symbols(np.array([16,16]),np.array([2,2]),rho,0,.25)
        np.testing.assert_array_equal(n,[[4,16],[12,0]])
        with self.assertRaises(ValueError):allocate_symbols(np.array([8,8]),np.array([4,4]),rho,0,.125)

    def test_independent_noise_bound_and_mismatch(self):
        rho=np.array([[[.1,.2]],[[.3,.4]]]);h=np.array([[2.,3.]])
        s=np.array([[.02,.03],[.04,.05]]);n=np.array([[8,12],[8,4]])
        a=rho/(s[:,None,:]*h[None,:,:]);dims=np.array([2,2])
        expected=np.sqrt(np.sum(s*s*dims/(2*n),axis=0))
        np.testing.assert_allclose(bounds(a,s,h,rho,dims,n),expected,atol=1e-14)
        a[0,0,0]=0
        self.assertAlmostEqual(bounds(a,s,h,rho,dims,n)[0],expected[0]+.1)

    def test_scalar_matches_analytic_energy_and_constraints(self):
        # One active learning vector: min RMS amplitude = rho*sqrt(d/2N)/(H*epsilon).
        rho=np.zeros((2,2,1));rho[0,1,0]=.1
        channel=np.array([[0.,2.],[2.,0.]])
        config=dict(symbol_rate_hz=1000.,base_overhead_j=1e-5,head_overhead_j=1e-5,
                    available_energy_j=.1,power_limit_w=1.,round_uplink_deadline_s=.008,
                    coherence_time_s=.01,ao_iterations=100)
        n=allocate_symbols(np.array([8]),np.array([1]),rho,0,.5)
        d=optimize(0,channel,rho,np.array([1]),n,.02,config)
        self.assertIsNotNone(d)
        expected=3e-5+.008*(.1*.25/(2*.02))**2
        self.assertAlmostEqual(d.energy_j,expected,delta=expected*1e-4)
        self.assertTrue(all(b<=a+1e-9 for a,b in zip(d.history,d.history[1:])))
        blocked=np.zeros((2,2,1,1));blocked[0,1,0,0]=.1
        stats=exact_statistics(d,blocked,1)
        self.assertLessEqual(stats['analytic_sum_rmse'],d.error_bound*(1+1e-5))
        self.assertEqual(stats['stream_rms_errors'][1],0)

    def test_two_stream_receiver_and_analytic_noise(self):
        rho=np.zeros((2,2,1));rho[:,1,0]=[.1,.05]
        channel=np.array([[0.,2.],[2.,0.]])
        config=dict(symbol_rate_hz=1000.,base_overhead_j=1e-5,head_overhead_j=1e-5,
                    available_energy_j=.1,power_limit_w=1.,round_uplink_deadline_s=.008,
                    coherence_time_s=.01,ao_iterations=10)
        n=allocate_symbols(np.array([8]),np.array([1]),rho,0,.5)
        d=optimize(0,channel,rho,np.array([1]),n,.05,config)
        self.assertIsNotNone(d)
        self.assertLessEqual(d.error_bound,.05*(1+1e-5))
        blocked=rho[...,None].copy()
        stats=exact_statistics(d,blocked,1)
        coefficient=d.receiver[:,None,:]*d.channels[None,:,:]*d.analog_rms/d.rho
        bias=float(((coefficient-1)*d.rho).sum())
        variance=float(np.sum(d.receiver**2/(2*n)))
        self.assertAlmostEqual(stats['analytic_sum_rmse']**2,bias*bias+variance,places=12)
        draw=np.random.default_rng(0).normal(bias,np.sqrt(variance),100000)
        self.assertAlmostEqual(float(np.mean(draw*draw)),bias*bias+variance,delta=.015*(bias*bias+variance))


if __name__=='__main__':unittest.main()
