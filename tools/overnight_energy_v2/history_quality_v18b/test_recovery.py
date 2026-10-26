import unittest
import numpy as np
from recovery import moments, perturbation


class PhysicalRecoveryChecks(unittest.TestCase):
    def test_exact_gain_and_local_head_is_not_perturbed(self):
        blocked = np.array([[[999., 999.]], [[3., 4.]]])
        d = dict(users=np.array([1]), rho=np.array([[5.]]), channels=np.array([[2.]]),
                 analog_rms=np.array([[1.25]]), receiver=np.array([2.]), data_symbols=np.array([10]))
        bias, var, rms = moments(blocked, d, 2)
        np.testing.assert_allclose(bias, 0.)
        self.assertAlmostEqual(rms, np.sqrt(.4))
        np.testing.assert_allclose(perturbation(blocked, d, 2, 30, np.zeros((1, 2))), 0.)

    def test_mismatch_sign_and_padding_excludes_dummy_noise(self):
        blocked = np.array([[[2., 0.], [1., 0.]]])
        d = dict(users=np.array([0]), rho=np.array([[2., 1.]]), channels=np.ones((1, 1)),
                 analog_rms=np.array([[1., .5]]), receiver=np.ones(2), data_symbols=np.array([2, 2]))
        actual = perturbation(blocked, d, 3, 30, np.array([[0., 0.], [2., 10000.]]))
        np.testing.assert_allclose(actual, [-1., 0., .5])
        self.assertAlmostEqual(moments(blocked, d, 3)[2], np.sqrt(2.))

    def test_common_noise_is_rescaled_for_each_actual_receiver(self):
        blocked = np.zeros((1, 1, 2)); noise = np.array([[1., -2.]])
        d = dict(users=np.array([0]), rho=np.zeros((1, 1)), channels=np.ones((1, 1)),
                 analog_rms=np.zeros((1, 1)), receiver=np.array([3.]), data_symbols=np.array([18]))
        np.testing.assert_allclose(perturbation(blocked, d, 2, 30, noise), [.5, -1.])


if __name__ == '__main__':
    unittest.main()
