import json
import unittest
from pathlib import Path

import numpy as np
import torch
from transformers import CLIPConfig, CLIPModel, CLIPTextConfig, CLIPVisionConfig

from clip_model import DeepPromptCLIP, projected_coupling
from coupling import cross_modal_update
from run_wireless import scenario
from transport import communicate, design_transport, energy_accounting, recover


ROOT = Path(__file__).resolve().parents[1]


def tiny_backbone():
    text = CLIPTextConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                          num_hidden_layers=2, num_attention_heads=4,
                          max_position_embeddings=16, eos_token_id=31, bos_token_id=1)
    vision = CLIPVisionConfig(hidden_size=16, intermediate_size=32,
                              num_hidden_layers=2, num_attention_heads=4,
                              image_size=8, patch_size=4)
    return CLIPModel(CLIPConfig.from_text_vision_configs(text, vision, projection_dim=8))


class RealPipelineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(5)

    def test_no_prompt_matches_original_clip(self):
        backbone = tiny_backbone().eval()
        wrapper = DeepPromptCLIP(backbone, prompt_layers=0, prompt_tokens=0)
        pixels = torch.randn(2, 3, 8, 8)
        ids = torch.tensor([[1, 3, 31, 0], [1, 4, 5, 31]])
        mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])
        expected = backbone(pixel_values=pixels, input_ids=ids, attention_mask=mask).logits_per_image
        observed = wrapper(pixels, ids, mask, wrapper.initial_vector)
        torch.testing.assert_close(observed, expected)

    def test_deep_prompts_receive_gradients_backbone_remains_frozen(self):
        model = DeepPromptCLIP(tiny_backbone(), prompt_layers=2, prompt_tokens=2,
                               checkpoint_layers=True)
        theta = model.initial_vector.clone().requires_grad_(True)
        predictions = model(torch.randn(2, 3, 8, 8),
                            torch.tensor([[1, 4, 31], [1, 6, 31]]),
                            torch.ones(2, 3, dtype=torch.long), theta)
        gradient = torch.autograd.grad(predictions.square().mean(), theta)[0]
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient[:model.visual_dimension].norm()), 0)
        self.assertGreater(float(gradient[model.visual_dimension:-1].norm()), 0)
        self.assertTrue(all(not parameter.requires_grad for parameter in model.backbone.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in model.backbone.parameters()))

    def test_projected_full_rank_matches_exact_quadratic_hessian(self):
        cross = torch.randn(3, 3, dtype=torch.float64)
        theta = torch.randn(7, dtype=torch.float64)
        gradient = torch.randn_like(theta)
        loss_function = lambda value: value[:3] @ cross @ value[3:6] + value.square().sum()
        result, statistics = projected_coupling(
            loss_function, theta, gradient, 3, 3, rank=3, difference_step=0.001
        )
        expected, _, _ = cross_modal_update(gradient, cross, 3, 3)
        torch.testing.assert_close(result, expected, atol=1e-9, rtol=1e-9)
        self.assertGreaterEqual(statistics["preconditioner_min_eigenvalue"], 0.1 - 1e-12)

    def test_equal_time_and_energy_accounting(self):
        config = json.loads((ROOT / "configs/pilot.json").read_text())
        config.update(users=4, dimension=32, relative_tolerance=0.6,
                      sic_residual_fraction=0.0, receive_initializations=3, max_ao_iterations=3,
                      area_m=50)
        channels, learning, unlearning, _, _, _, _, symbols, _ = scenario(config, 0)
        learning = learning.reshape(4, -1)
        unlearning = unlearning.reshape(4, -1)
        epsilon = 0.6 * np.linalg.norm(np.linalg.norm(learning, axis=-1).sum())
        for method in ("hybrid", "orthogonal_equal_time", "mixed_aircomp"):
            design, blocked_learning, blocked_unlearning, decoded, _ = design_transport(
                learning, unlearning, channels, config, method, epsilon
            )
            self.assertIsNotNone(design)
            statistics = energy_accounting(design, config)
            self.assertAlmostEqual(statistics["uplink_s"], symbols.sum() / config["symbol_rate_hz"])
            self.assertLessEqual(statistics["expected_energy_random_codewords_j"], design.energy_j + 1e-12)
            observed = recover(design, blocked_learning, blocked_unlearning, decoded,
                               np.random.default_rng(3))
            self.assertEqual(observed.shape, (32,))
            self.assertTrue(np.isfinite(observed).all())
        exact, statistics = communicate(learning, unlearning, channels, config, "ideal", 0)
        np.testing.assert_allclose(exact, (learning + unlearning).sum(axis=0))
        self.assertEqual(statistics["realized_error"], 0)


if __name__ == "__main__":
    unittest.main()
