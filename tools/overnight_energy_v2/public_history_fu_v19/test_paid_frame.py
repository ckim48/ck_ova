"""Check paid control and phase accounting with/without a remoteFU payload."""
from pathlib import Path
import sys
import unittest
import numpy as np
from codec import encode_residual, reserve_symbols

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT/'tools/overnight_energy_v2')]
from phase_split_grid import make_inputs, split_symbols


class PaidFrameChecks(unittest.TestCase):
    def setUp(self):
        self.c = dict(block_dimension=4, symbols_per_coordinate=64, quantization_bits=8, scale_metadata_bits=32)
        self.l = np.arange(24).reshape(3, 8)/23
        self.u = np.zeros_like(self.l); self.u[1, 0] = .2

    def test_coefficient_time_is_not_counted_twice_or_made_free(self):
        inputs = make_inputs(self.l, self.u, self.c)
        dims, total = inputs[-2:]
        nmain, each, control = reserve_symbols(total, dims, 3, 2)
        self.assertEqual(each, 96); self.assertEqual(control, 192)
        for head in [0, 1]:
            a, d, active = split_symbols(nmain, inputs[4], head, .25)
            self.assertEqual(int(a.sum()+d[active].sum()+control), int(total.sum()))
            self.assertTrue(np.all(a >= dims))
            if head == 1:
                self.assertFalse(active.any())
                np.testing.assert_array_equal(a, nmain)

    def test_projection_cannot_silently_relax_total_error_or_u_codec(self):
        baseline = make_inputs(self.l, self.u, self.c)
        basis = np.eye(8)[:, :3]
        residual, bits, decoded = encode_residual(self.l, basis, .1)
        projected = make_inputs(residual, self.u, self.c)
        np.testing.assert_allclose(residual+decoded@basis.T, self.l, rtol=1e-12, atol=1e-15)
        np.testing.assert_array_equal(baseline[3], projected[3])
        np.testing.assert_array_equal(baseline[4], projected[4])
        self.assertGreater(np.linalg.norm(baseline[5].sum(axis=0)), np.linalg.norm(projected[5].sum(axis=0)))
        # The main diagnostic deliberately uses the former, original tolerance
        # for every rank and mode, irrespective of this residual norm change.


if __name__ == '__main__':
    unittest.main()
