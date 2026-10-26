import unittest
import numpy as np
from recovery import moments, perturbation


class FURecoveryChecks(unittest.TestCase):
    def test_remote_mismatch_and_quantization_add_with_correct_sign(self):
        a = np.array([[[500., 500.]], [[2., 0.]]])
        u = np.array([[[900., 900.]], [[4., 0.]]]); q = u.copy(); q[1, 0, 0] = 3.
        d = dict(chi=0., users=np.array([1]), rho=np.array([[2.]]), receiver=np.array([1.]),
                 normalized_channels=np.ones((1, 1)), analog_rms=np.array([[1.]]), symbols=np.array([2]))
        np.testing.assert_allclose(perturbation(a, u, q, d, 2, 61, np.zeros((1, 2))), [-2., 0.])
        self.assertAlmostEqual(moments(a, u, q, d, 2)[2], np.sqrt(4.5))

    def test_perfect_remote_gain_does_not_perturb_local_head(self):
        a = np.array([[[999., 999.]], [[3., 4.]]]); u = np.ones_like(a)
        d = dict(chi=0., users=np.array([1]), rho=np.array([[5.]]), receiver=np.array([2.]),
                 normalized_channels=np.array([[2.]]), analog_rms=np.array([[1.25]]), symbols=np.array([10]))
        np.testing.assert_allclose(perturbation(a, u, u, d, 2, 61, np.zeros((1, 2))), 0.)
        self.assertAlmostEqual(moments(a, u, u, d, 2)[2], np.sqrt(.4))

    def test_padding_removed_and_nonzero_sic_not_silently_ignored(self):
        a = np.zeros((1, 2, 2)); d = dict(chi=0., users=np.array([0]), rho=np.zeros((1, 2)),
            receiver=np.ones(2), normalized_channels=np.ones((1, 2)), analog_rms=np.zeros((1, 2)), symbols=np.array([2, 2]))
        np.testing.assert_allclose(perturbation(a, a, a, d, 3, 61, np.array([[2., 4.], [6., 10000.]])), [1., 2., 3.])
        d['chi'] = .01
        with self.assertRaises(AssertionError):
            moments(a, a, a, d, 3)


if __name__ == '__main__':
    unittest.main()
