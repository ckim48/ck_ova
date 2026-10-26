import unittest
import numpy as np
from recovery import stream_errors,analytic_sum_rmse


class RecoveryTests(unittest.TestCase):
    def fixture(self):
        blocked=np.zeros((2,2,2,2));blocked[0,1]=[[.1,0],[.2,0]];blocked[1,1]=[[0,.3],[0,.4]]
        rho=np.linalg.norm(blocked[:,1:2],axis=-1)
        s=np.array([[.2,.4],[.3,.6]]);h=np.array([[2.,3.]])
        a=rho/(s[:,None,:]*h[None,:,:])
        return blocked,dict(analog_rms=a,receiver=s,symbols=np.array([[4,8],[4,8]]),rho=rho,channels=h,users=np.array([1]))
    def test_perfect_mapping_and_noise_zero(self):
        v,d=self.fixture();e=stream_errors(v,d,4,1,np.zeros((2,2,2)))
        np.testing.assert_allclose(e,0,atol=1e-15)
    def test_missing_signal_has_negative_remote_bias(self):
        v,d=self.fixture();d['analog_rms']*=0;d['receiver']*=0
        np.testing.assert_allclose(stream_errors(v,d,4,1),-v[:,1].reshape(2,4))
    def test_phase_noise_variance_padding_and_pairing(self):
        v,d=self.fixture();e=stream_errors(v,d,3,1,np.ones((2,2,2)))
        expected=np.sqrt(d['receiver']**2/(2*d['symbols']))
        np.testing.assert_allclose(e,np.repeat(expected,2,axis=1)[:,:3],atol=1e-15)
        expected_mse=np.sum(np.array([2,1])*np.sum(d['receiver']**2/(2*d['symbols']),axis=0))
        self.assertAlmostEqual(analytic_sum_rmse(v,d,3)**2,expected_mse)
        np.testing.assert_array_equal(stream_errors(v,d,3,91),stream_errors(v,d,3,91))


if __name__=='__main__':unittest.main()
