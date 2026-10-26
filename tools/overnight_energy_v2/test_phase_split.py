import json
import unittest
from pathlib import Path

import numpy as np

from phase_split_grid import make_inputs, optimize_fraction, split_symbols
from run_wireless import scenario
from transport import design_transport, energy_accounting


class PhaseSplitTests(unittest.TestCase):
    def config(self):
        import run_wireless
        root = Path(run_wireless.__file__).parent
        c = json.loads((root/'configs/pilot.json').read_text())
        c.update(users=4, dimension=32, relative_tolerance=0.6, sic_residual_fraction=0,
                 receive_initializations=3, max_ao_iterations=3, area_m=50)
        return c

    def test_half_split_matches_existing_comparator(self):
        c = self.config()
        h,l,u,*_ = scenario(c, 0); l=l.reshape(4,-1);u=u.reshape(4,-1)
        inputs = make_inputs(l,u,c); epsilon=0.6*np.linalg.norm(inputs[5].sum(axis=0))
        expected,*_ = design_transport(l,u,h,c,'orthogonal_equal_time',epsilon)
        observed,*_ = optimize_fraction(l,u,h,c,epsilon,0.5)
        self.assertIsNotNone(expected);self.assertIsNotNone(observed)
        self.assertAlmostEqual(expected.energy_j,observed.energy_j,places=10)

    def test_fraction_preserves_total_time_and_dimension(self):
        c=self.config();h,l,u,*_=scenario(c,0)
        inputs=make_inputs(l.reshape(4,-1),u.reshape(4,-1),c)
        for fraction in (0.125,0.25,0.5,0.75,0.875):
            for head in range(4):
                a,d,active=split_symbols(inputs[-1],inputs[4],head,fraction)
                self.assertGreaterEqual(a.min(),c['block_dimension'])
                self.assertEqual(int(a.sum()+d[active].sum()),int(inputs[-1].sum()))


if __name__ == '__main__':
    unittest.main()
