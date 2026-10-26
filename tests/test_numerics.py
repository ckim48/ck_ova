import unittest

import numpy as np
import torch

from coupling import cross_modal_update
from data_schedule import dirichlet_partition, make_schedule
from run_wireless import ROOT, scenario
from wireless import empirical_error, optimize_head, quantize, receive_optimum, sic_system


class NumericalTests(unittest.TestCase):
    def test_convex_initialization_without_grid_starts_preserves_constraints(self):
        import json

        config = json.loads((ROOT / "configs/pilot.json").read_text())
        config.update(users=4, dimension=32, area_m=50, max_ao_iterations=3,
                      receive_initializations=0)
        channel, learning, _, _, delta, payload, dims, symbols, _ = scenario(config, 0)
        rho = np.linalg.norm(learning, axis=-1)
        epsilon = 0.6 * np.linalg.norm(rho.sum(axis=0))
        for mode in ("hybrid", "orthogonal"):
            with self.subTest(mode=mode):
                design = optimize_head(0, channel, rho, delta, payload, dims, symbols,
                                       epsilon, config, 0.001, mode)
                self.assertIsNotNone(design)
                self.assertLessEqual(design.error_bound, epsilon * 1.00001)
                peak = np.maximum(design.analog_rms ** 2, design.digital_sqrt_power ** 2)
                if mode == "hybrid":
                    peak = (design.analog_rms + design.digital_sqrt_power) ** 2
                self.assertLessEqual(float(peak.max()), config["power_limit_w"] * 1.000001)
                self.assertTrue(np.all(np.diff(design.history) <= 1e-10))
                self.assertIsNone(optimize_head(
                    0, channel, rho, delta + epsilon, payload, dims, symbols,
                    epsilon, config, 0.001, mode,
                ))

    def test_schedule_deletes_only_used_samples_without_reintroduction(self):
        labels = np.arange(1000) % 5
        partitions = dirichlet_partition(labels, 10, 0.1, 7)
        all_ids = np.concatenate(partitions)
        self.assertEqual(len(np.unique(all_ids)), len(labels))
        schedule = make_schedule(partitions, seed=7)
        used = [set() for _ in partitions]
        deleted = set()
        for record in schedule:
            for user, requested in enumerate(record["forget"]):
                self.assertTrue(set(requested) <= used[user] - deleted)
            new_deleted = set().union(*map(set, record["forget"]))
            self.assertFalse(deleted & new_deleted)
            deleted.update(new_deleted)
            for user, batch in enumerate(record["batches"]):
                self.assertFalse(set(batch) & deleted)
                used[user].update(batch)
        self.assertTrue(deleted)

    def test_quantizer_bound_and_empty_payload(self):
        rng = np.random.default_rng(13)
        values = rng.normal(size=(4, 3, 32))
        values[0] = 0
        for bits in (2, 4, 8, 16):
            decoded, bound, payload = quantize(values, bits)
            self.assertTrue(np.all(np.linalg.norm(decoded - values, axis=-1) <= bound + 1e-12))
            self.assertTrue(np.all(payload[0] == 0))
            self.assertTrue(np.all(decoded[0] == 0))

    def test_sic_minimum_and_instability(self):
        gamma = np.array([1.0, 2.0, 0.5])
        solution = sic_system(gamma, 0)
        np.testing.assert_allclose(solution, [4.5, 3.0, 0.5])
        with self.assertRaises(ValueError):
            sic_system(np.array([10.0, 10.0]), 0.1)
        self.assertEqual(len(sic_system(np.array([]), 0)), 0)

    def test_receiver_against_all_breakpoints(self):
        channel = np.array([2.0, 1.0, 3.0])
        analog = np.array([0.4, 0.2, 0.0])
        rho = np.array([0.02, 0.03, 0.01])
        noise = 0.4
        optimum = receive_optimum(channel, analog, rho, noise)
        candidates = np.r_[0, rho[:2] / (channel[:2] * analog[:2])]
        costs = [np.abs(value * channel * analog - rho).sum() + noise * value
                 for value in candidates]
        self.assertAlmostEqual(
            np.abs(optimum * channel * analog - rho).sum() + noise * optimum, min(costs)
        )

    def test_preconditioner_positive_definite_and_shared_unchanged(self):
        torch.manual_seed(2)
        cross = torch.randn(4, 3, dtype=torch.float64) * 10
        gradient = torch.randn(9, dtype=torch.float64)
        output, kappa, norm = cross_modal_update(gradient, cross, 4, 3, 2, 0.2)
        matrix = torch.eye(9, dtype=torch.float64)
        matrix[:4, 4:7] = kappa * cross
        matrix[4:7, :4] = kappa * cross.T
        self.assertLessEqual(kappa * norm, 0.8 + 1e-12)
        self.assertGreaterEqual(float(torch.linalg.eigvalsh(matrix).min()), 0.2 - 1e-12)
        torch.testing.assert_close(output, matrix @ gradient)
        torch.testing.assert_close(output[7:], gradient[7:])

    def test_feasible_design_monte_carlo_and_ao(self):
        import json

        config = json.loads((ROOT / "configs/pilot.json").read_text())
        config.update(users=4, dimension=32, max_ao_iterations=3, receive_initializations=4)
        channel, learning, unlearning, decoded, delta, payload, dims, symbols, _ = scenario(config, 0)
        rho = np.linalg.norm(learning, axis=-1)
        epsilon = 0.6 * np.linalg.norm(rho.sum(axis=0))
        design = optimize_head(0, channel, rho, delta, payload, dims, symbols, epsilon, config, 0)
        self.assertIsNotNone(design)
        self.assertLessEqual(design.error_bound, epsilon * 1.00001)
        self.assertTrue(np.all(np.diff(design.history) <= 1e-10))
        measured, analytic, standard_error = empirical_error(
            design, learning, unlearning, decoded, np.random.default_rng(4), 10000
        )
        self.assertLessEqual(analytic, design.error_bound ** 2)
        self.assertLess(abs(measured - analytic), 6 * standard_error)
        config["available_energy_j"] = 0
        self.assertIsNone(optimize_head(0, channel, rho, delta, payload, dims, symbols,
                                       epsilon, config, 0))


if __name__ == "__main__":
    unittest.main()
