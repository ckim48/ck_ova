import unittest
import numpy as np
from codec import encode_residual, reserve_symbols, control_resources


class CodecChecks(unittest.TestCase):
    def test_nonzero_half_rounding_cancels_in_reconstruction(self):
        rng = np.random.default_rng(7)
        q, _ = np.linalg.qr(rng.normal(size=(97, 3)))
        v = rng.normal(size=(10, 97))*1e-7
        residual, bits, decoded = encode_residual(v, q, 2.3e-6)
        self.assertGreater(np.linalg.norm(decoded-v@q), 0.)
        self.assertEqual(bits.dtype, np.uint16)
        self.assertEqual(bits.nbytes, 10*3*2)
        np.testing.assert_allclose(residual+decoded@q.T, v, rtol=1e-12, atol=1e-21)
        np.testing.assert_allclose(bits.view(np.float16).astype(float)*2.3e-6, decoded)

    def test_scale_invalid_or_overflow_is_detected(self):
        for scale in [0., float('nan')]:
            with self.assertRaises(ValueError):
                encode_residual(np.ones((2, 2)), np.eye(2), scale)
        with self.assertRaises(ValueError):
            encode_residual(np.full((2, 2), 1e9), np.eye(2), 1e-6)

    def test_same_frame_and_actual_payload_rate(self):
        n, each, total = reserve_symbols(np.full(41, 4096), np.full(41, 512), 3, 9)
        self.assertEqual(total, 864)
        self.assertEqual(each, 96)
        self.assertEqual(int(n.sum())+total, 167936)
        h = np.array([2., 4.])
        p, e = control_resources(h, 48, each, 1e6)
        np.testing.assert_allclose(each*np.log2(1+h*h*p), 48.)
        np.testing.assert_allclose(e, p*.000096)


if __name__ == '__main__':
    unittest.main()
