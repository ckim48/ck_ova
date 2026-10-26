"""Check common policy, common pre-request path and equal radio budgets."""
import unittest

import numpy as np

import burst_transport as bt


class BurstTransportTests(unittest.TestCase):
    def inputs(self):
        rng = np.random.default_rng(18)
        l = rng.normal(0, .0004, (3, 8)); u = np.zeros_like(l)
        u[1] = rng.normal(0, .0002, 8)
        u[2] = rng.normal(0, .0002, 8)
        h = np.array([[0., 20., 30.], [20., 0., 25.], [30., 25., 0.]])
        c = {'users': 3, 'block_dimension': 4, 'symbols_per_coordinate': 8,
             'quantization_bits': 8, 'scale_metadata_bits': 32,
             'relative_tolerance': .2, 'deletion_relative_tolerance': .6,
             'absolute_tolerance_floor': 1e-8, 'sic_residual_fraction': 0.,
             'power_limit_w': .32, 'available_energy_j': .1,
             'base_overhead_j': 1e-5, 'head_overhead_j': 5e-5,
             'symbol_rate_hz': 1e6, 'round_uplink_deadline_s': .2,
             'coherence_time_s': .01, 'max_ao_iterations': 4,
             'receive_initializations': 3}
        return l, u, h, c

    def test_no_request_is_identical_across_modes(self):
        l, u, h, c = self.inputs(); u[:] = 0
        outputs = [bt.communicate(l, u, h, c, mode, 91) for mode in bt.MODES]
        for update, stats in outputs:
            np.testing.assert_array_equal(update, outputs[0][0])
            self.assertEqual(stats['energy_bound_j'], outputs[0][1]['energy_bound_j'])
            self.assertEqual(stats['selected_mode'], 'hybrid')
            self.assertEqual(stats['relative_tolerance_used'], .2)

    def test_requests_use_common_error_and_time_budget(self):
        l, u, h, c = self.inputs()
        results = [bt.communicate(l, u, h, c, mode, 91) for mode in bt.MODES]
        epsilons = []
        for update, stats in results:
            self.assertEqual(len(update), l.shape[1])
            self.assertEqual(stats['relative_tolerance_used'], .6)
            self.assertAlmostEqual(stats['uplink_s'], 64/1e6)
            self.assertLessEqual(stats['error_bound'], stats['epsilon']*(1+1e-5))
            self.assertLessEqual(stats['expected_energy_random_codewords_j'], stats['energy_bound_j']*(1+1e-6))
            epsilons.append(stats['epsilon'])
        self.assertEqual(len(set(epsilons)), 1)
        phase = results[-1][1]
        self.assertEqual(len(phase['fraction_search']), len(bt.FRACTIONS))
        eligible = [r['energy_bound_j'] for r in phase['fraction_search'] if r['energy_bound_j'] is not None]
        self.assertAlmostEqual(phase['energy_bound_j'], min(eligible), places=12)


if __name__ == '__main__':
    unittest.main()
