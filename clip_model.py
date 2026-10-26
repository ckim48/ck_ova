from pathlib import Path

import torch
import torch.nn.functional as functional
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers import CLIPConfig, CLIPModel


class DeepPromptCLIP(nn.Module):
    def __init__(self, backbone, prompt_layers=8, prompt_tokens=2, checkpoint_layers=False):
        super().__init__()
        self.backbone = backbone.eval()
        self.backbone.requires_grad_(False)
        self.prompt_layers = prompt_layers
        self.prompt_tokens = prompt_tokens
        self.checkpoint_layers = checkpoint_layers
        visual_config = backbone.config.vision_config
        text_config = backbone.config.text_config
        if prompt_layers > min(visual_config.num_hidden_layers, text_config.num_hidden_layers):
            raise ValueError("Too many prompt layers.")
        self.visual_shape = (prompt_layers, prompt_tokens, visual_config.hidden_size)
        self.text_shape = (prompt_layers, prompt_tokens, text_config.hidden_size)
        self.visual_dimension = prompt_layers * prompt_tokens * visual_config.hidden_size
        self.text_dimension = prompt_layers * prompt_tokens * text_config.hidden_size
        visual = torch.randn(self.visual_shape) * 0.02
        text = torch.randn(self.text_shape) * 0.02
        initial = torch.cat((visual.flatten(), text.flatten(), backbone.logit_scale.detach().reshape(1)))
        self.register_buffer("initial_vector", initial)

    @classmethod
    def from_local_checkpoint(cls, path, **kwargs):
        path = Path(path)
        config = CLIPConfig.from_pretrained(path, local_files_only=True)
        backbone = CLIPModel(config)
        state = torch.load(path / "pytorch_model.bin", map_location="cpu", weights_only=True)
        expected = backbone.state_dict()
        for modality in ("text_model", "vision_model"):
            key = f"{modality}.embeddings.position_ids"
            if key in state and key not in expected:
                position_ids = getattr(backbone, modality).embeddings.position_ids
                if not torch.equal(state[key], position_ids):
                    raise ValueError("Checkpoint position IDs differ from the configured model.")
                del state[key]
        backbone.load_state_dict(state, strict=True)
        return cls(backbone, **kwargs)

    def parts(self, theta):
        text_end = self.visual_dimension + self.text_dimension
        return (
            theta[:self.visual_dimension].reshape(self.visual_shape),
            theta[self.visual_dimension:text_end].reshape(self.text_shape),
            theta[-1],
        )

    def run_layer(self, layer, hidden, padding_mask=None, causal_mask=None):
        if self.checkpoint_layers and torch.is_grad_enabled():
            return checkpoint(layer, hidden, padding_mask, causal_mask, use_reentrant=False)[0]
        return layer(hidden, padding_mask, causal_mask)[0]

    def inject(self, hidden, prompts, layer_index):
        if self.prompt_tokens == 0 or layer_index >= self.prompt_layers:
            return hidden
        after_prompt = 1 if layer_index == 0 else 1 + self.prompt_tokens
        batch_prompts = prompts[layer_index].unsqueeze(0).expand(hidden.shape[0], -1, -1)
        return torch.cat((hidden[:, :1], batch_prompts, hidden[:, after_prompt:]), dim=1)

    def image_features(self, pixels, visual):
        model = self.backbone.vision_model
        hidden = model.pre_layrnorm(model.embeddings(pixels))
        for index, layer in enumerate(model.encoder.layers):
            hidden = self.inject(hidden, visual, index)
            hidden = self.run_layer(layer, hidden)
        return self.backbone.visual_projection(model.post_layernorm(hidden[:, 0]))

    def text_features(self, input_ids, attention_mask, text):
        model = self.backbone.text_model
        hidden = model.embeddings(input_ids=input_ids)
        inserted = self.prompt_tokens if self.prompt_layers else 0
        if model.eos_token_id == 2:
            eos_positions = input_ids.argmax(dim=-1)
        else:
            eos_positions = (input_ids == model.eos_token_id).int().argmax(dim=-1)
        eos_positions = eos_positions + inserted
        if inserted:
            attention_mask = torch.cat((
                attention_mask[:, :1],
                attention_mask.new_ones((attention_mask.shape[0], inserted)),
                attention_mask[:, 1:],
            ), dim=1)
        length = hidden.shape[1] + inserted
        causal = torch.full((length, length), torch.finfo(hidden.dtype).min,
                            device=hidden.device, dtype=hidden.dtype).triu(1)
        causal = causal[None, None].expand(hidden.shape[0], 1, -1, -1)
        padding = (1 - attention_mask[:, None, None, :].to(hidden.dtype))
        padding = padding.expand(-1, 1, length, -1) * torch.finfo(hidden.dtype).min
        for index, layer in enumerate(model.encoder.layers):
            hidden = self.inject(hidden, text, index)
            hidden = self.run_layer(layer, hidden, padding, causal)
        hidden = model.final_layer_norm(hidden)
        pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), eos_positions]
        return self.backbone.text_projection(pooled)

    def forward(self, pixels, input_ids, attention_mask, theta):
        visual, text, logit_scale = self.parts(theta)
        images = functional.normalize(self.image_features(pixels, visual), dim=-1)
        texts = functional.normalize(self.text_features(input_ids, attention_mask, text), dim=-1)
        return logit_scale.exp() * images @ texts.T


def projected_coupling(loss_function, theta, forget_gradient, visual_dimension,
                       text_dimension, rank=4, kappa_requested=1.0, xi=0.1,
                       difference_step=0.001, seed=0):
    if rank < 1 or not 0 < xi <= 1 or kappa_requested < 0 or difference_step <= 0:
        raise ValueError("Invalid projected coupling configuration.")
    generator = torch.Generator(device=theta.device).manual_seed(seed)
    rank = min(rank, visual_dimension, text_dimension)
    visual_basis = torch.linalg.qr(torch.randn(
        visual_dimension, rank, generator=generator, device=theta.device, dtype=theta.dtype
    ), mode="reduced").Q
    text_basis = torch.linalg.qr(torch.randn(
        text_dimension, rank, generator=generator, device=theta.device, dtype=theta.dtype
    ), mode="reduced").Q
    text_end = visual_dimension + text_dimension
    columns = []
    for column in range(rank):
        direction = torch.zeros_like(theta)
        direction[visual_dimension:text_end] = text_basis[:, column]
        positive = (theta + difference_step * direction).detach().requires_grad_(True)
        negative = (theta - difference_step * direction).detach().requires_grad_(True)
        gradient_positive = torch.autograd.grad(loss_function(positive), positive)[0]
        gradient_negative = torch.autograd.grad(loss_function(negative), negative)[0]
        product = (gradient_positive[:visual_dimension] - gradient_negative[:visual_dimension])
        columns.append(visual_basis.T @ (product / (2 * difference_step)))
    core = torch.stack(columns, dim=1)
    norm = float(torch.linalg.matrix_norm(core, ord=2))
    kappa = min(kappa_requested, (1 - xi) / max(norm, 1e-30))
    result = forget_gradient.clone()
    result[:visual_dimension] += kappa * visual_basis @ core @ (
        text_basis.T @ forget_gradient[visual_dimension:text_end]
    )
    result[visual_dimension:text_end] += kappa * text_basis @ core.T @ (
        visual_basis.T @ forget_gradient[:visual_dimension]
    )
    return result, {"kappa": kappa, "approximate_cross_norm": norm,
                    "preconditioner_min_eigenvalue": 1 - kappa * norm, "rank": rank}
