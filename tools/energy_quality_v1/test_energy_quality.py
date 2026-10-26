import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from energy_quality import auc, classification, digest, losses, make_cohorts, nondominated, predict, reference_gap, summarize, validate_completed


class EnergyQualityTests(unittest.TestCase):
    def test_auc_direction_and_ties(self):
        self.assertEqual(auc([2, 3], [0, 1]), 1)
        self.assertEqual(auc([0, 1], [2, 3]), 0)
        self.assertEqual(auc([1, 1], [1, 1]), 0.5)
        self.assertAlmostEqual(auc([0, 1, 2], [1, 1]), 0.5)

    def test_reference_agreement_and_direction(self):
        p = np.array([[0.9, 0.1], [0.4, 0.6]])
        labels = np.array([0, 1])
        gap = reference_gap(p, p.copy(), labels)
        self.assertTrue(all(abs(x) < 1e-12 for x in gap.values()))
        q = np.array([[0.1, 0.9], [0.7, 0.3]])
        self.assertGreater(reference_gap(q, p, labels)["kl_reference_to_candidate"], 0)
        self.assertEqual(classification(p, labels)["accuracy"], 1)
        self.assertLess(losses(p, labels).mean(), losses(q, labels).mean())

    def test_observed_frontier_excludes_incomplete_quality(self):
        rows = [{"e": 1, "r": 2, "f": 3}, {"e": 2, "r": 3, "f": 4},
                {"e": 3, "r": 1, "f": 1}, {"e": 0, "r": 0}]
        self.assertEqual(nondominated(rows, ("e", "r", "f")), [0, 2])

    def test_cohorts_disjoint_class_matched_and_reproducible(self):
        class FakeData:
            train_labels = np.repeat(np.arange(100), 20)
            test_labels = np.repeat(np.arange(100), 5)

            def validation_split(self, seed, per_class):
                val = np.array([20*c+i for c in range(100) for i in range(12, 20)])
                train = np.array([20*c+i for c in range(100) for i in range(12)])
                return train, val
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            deleted = [20*c+i for c in range(100) for i in range(3)]
            retained = [20*c+i for c in range(100) for i in range(3, 12)]
            (path / "schedule.json").write_text(json.dumps([{"deleted": deleted, "retained": [retained]}]))
            config = {"seed": 0, "validation_per_class": 8}
            a = make_cohorts(FakeData(), path, config, 137)
            self.assertEqual(a, make_cohorts(FakeData(), path, config, 137))
            g = a["groups"]
            self.assertFalse(set(g["retained"]["ids"]) & set(g["forget"]["ids"]))
            self.assertFalse(set(g["validation"]["ids"]) & set(g["nonmember_validation"]["ids"]))
            expected = np.bincount(FakeData.train_labels[g["forget"]["ids"]], minlength=100)
            np.testing.assert_array_equal(expected, np.bincount(FakeData.train_labels[g["nonmember_validation"]["ids"]], minlength=100))
            np.testing.assert_array_equal(expected, np.bincount(FakeData.test_labels[g["nonmember_test"]["ids"]], minlength=100))

    def test_cached_text_forward_matches_original_model(self):
        import torch
        from transformers import CLIPConfig, CLIPModel, CLIPTextConfig, CLIPVisionConfig
        from clip_model import DeepPromptCLIP
        torch.set_num_threads(1); torch.manual_seed(5)
        t = CLIPTextConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=4, max_position_embeddings=16, eos_token_id=31)
        v = CLIPVisionConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=2,
            num_attention_heads=4, image_size=8, patch_size=4)
        model = DeepPromptCLIP(CLIPModel(CLIPConfig.from_text_vision_configs(t, v, projection_dim=8)),
                               prompt_layers=2, prompt_tokens=2).eval()
        pixels = torch.randn(3, 3, 8, 8)
        text = {"input_ids": torch.tensor([[1, 3, 31], [1, 4, 31]]),
                "attention_mask": torch.ones(2, 3, dtype=torch.long)}
        class FakeData:
            classes = [0, 1]
            def batch(self, ids, device, split):
                return pixels[ids].to(device), torch.zeros(len(ids), dtype=torch.long)
        with torch.no_grad():
            expected = model(pixels, **text, theta=model.initial_vector).double().softmax(dim=-1).numpy()
        observed = predict(model, model.initial_vector, FakeData(), text, [0, 1, 2], "train", torch.device("cpu"), 2)
        np.testing.assert_allclose(observed, expected, atol=2e-6, rtol=2e-6)

    def test_final_selection_excludes_zero_energy_ideal_and_stale_reference(self):
        import energy_quality
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); suite = root / "suite"; output = root / "audit"
            suite.mkdir(); output.mkdir()
            config = {"rounds": 1, "quantization_bits": 8}
            (suite / "config.json").write_text(json.dumps(config))
            (suite / "suite_metadata.json").write_text(json.dumps({"source_sha256": {"x.py": "frozen"}}))
            (output / "cohorts.json").write_text("{}")
            p = np.array([[0.8, 0.2], [0.2, 0.8]])
            labels = np.array([0, 1]); arrays = {}
            for group in ("validation", "retained", "forget", "test", "nonmember_validation", "nonmember_test"):
                arrays[group] = p; arrays[group + "_labels"] = labels
            for name, transport, energy in (("coupled_hybrid", "hybrid", 2),
                    ("coupled_ideal", "ideal", 0), ("retrain_reference_ideal", "ideal", 0),
                    ("coupled_hybrid_q6", "hybrid", 1)):
                d = suite / name; d.mkdir()
                (d / "schedule.json").write_text("[]")
                (d / "latest.pt").write_bytes(b"final")
                method_config = dict(config, quantization_bits=6) if name.endswith("_q6") else config
                (d / "manifest.json").write_text(json.dumps({"status": "completed", "config": method_config,
                    "algorithm": "coupled" if name.endswith("_q6") else name.removesuffix("_" + transport), "transport": transport,
                    "source_sha256": {"x.py": "frozen"}}))
                (d / "rounds.jsonl").write_text(json.dumps({"round": 1, "cumulative_energy_bound_j": energy,
                    "cumulative_expected_energy_j": energy, "transmission": {"energy_bound_j": energy, "uplink_s": 0.1}}) + "\n")
                np.savez(output / (name + ".npz"), **arrays)
                (output / (name + ".cache.json")).write_text(json.dumps({
                    "checkpoint_sha256": digest(d / "latest.pt"), "cohorts_sha256": digest(output / "cohorts.json"),
                    "evaluator_sha256": digest(energy_quality.__file__), "training_source_sha256": {"x.py": "frozen"}}))
            args = SimpleNamespace(suite=suite, output=output, validation_loss_max=1,
                                   forget_kl_max=0.01, attack_gap_max=0.01)
            summarize(args)
            report = json.loads((output / "energy_quality_report.json").read_text())
            self.assertEqual(report["lowest_observed_energy_qualifying_method"], "coupled_hybrid_q6")
            self.assertEqual(report["observed_frontier"], ["coupled_hybrid_q6"])
            variant = suite / "coupled_hybrid_q6"
            m = json.loads((variant / "manifest.json").read_text())
            m["config"]["learning_rate"] = 9
            (variant / "manifest.json").write_text(json.dumps(m))
            self.assertEqual(validate_completed(variant, suite, config, suite / "coupled_hybrid")[1], "incompatible_config")
            summarize(args)
            self.assertEqual(json.loads((output / "energy_quality_report.json").read_text())["lowest_observed_energy_qualifying_method"], "coupled_hybrid")
            (suite / "retrain_reference_ideal/latest.pt").write_bytes(b"changed")
            summarize(args)
            report = json.loads((output / "energy_quality_report.json").read_text())
            self.assertFalse(report["reference_available"])
            self.assertIsNone(report["lowest_observed_energy_qualifying_method"])


if __name__ == "__main__":
    unittest.main()
