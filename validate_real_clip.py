import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

from clip_model import DeepPromptCLIP
from real_data import CIFAR100
from run_clip_federated import require_device
from transport import communicate, link_matrix


ROOT = Path(__file__).resolve().parent


def main(device_name):
    torch.set_num_threads(4)
    torch.manual_seed(0)
    device = require_device(device_name)
    checkpoint = ROOT / "checkpoints/clip-vit-base-patch16"
    data = CIFAR100(ROOT / "data/cifar-100-python", checkpoint)
    model = DeepPromptCLIP.from_local_checkpoint(checkpoint).to(device)
    text = data.text_inputs(device)
    pixels, targets = data.batch([0, 1], device)
    theta = model.initial_vector.detach().clone().requires_grad_(True)
    started = time.monotonic()
    predictions = model(pixels, **text, theta=theta)
    value = functional.cross_entropy(predictions, targets)
    gradient = torch.autograd.grad(value, theta)[0]
    if not torch.isfinite(gradient).all():
        raise RuntimeError("Non-finite prompt gradient.")
    updated = (theta - 0.0035 * gradient).detach()
    with torch.no_grad():
        updated_loss = functional.cross_entropy(model(pixels, **text, theta=updated), targets)
    settings = json.loads((ROOT / "configs/clip_cifar100.json").read_text())
    settings.update(users=2, relative_tolerance=0.1, receive_initializations=3,
                    max_ao_iterations=3, block_dimension=2048, coherence_time_s=0.05,
                    area_m=100)
    local_update = (updated - theta.detach()).cpu().numpy()
    updates = np.stack([local_update / 2, local_update / 2])
    channels = link_matrix(2, 123, settings)
    recovered, communication = communicate(updates, np.zeros_like(updates), channels,
                                          settings, "hybrid", 0)
    transmitted_theta = theta.detach() + torch.as_tensor(recovered, device=device, dtype=theta.dtype)
    with torch.no_grad():
        wireless_loss = functional.cross_entropy(model(pixels, **text, theta=transmitted_theta), targets)
    result = {
        "status": "real_data_smoke_completed", "device": str(device),
        "model": "pretrained CLIP ViT-B/16 with 8-layer, 2-token visual/text prompts",
        "dataset": "CIFAR-100", "sample_ids": [0, 1],
        "classes": predictions.shape[1], "trainable_parameters": theta.numel(),
        "initial_batch_loss": float(value.detach()),
        "after_ideal_sgd_batch_loss": float(updated_loss),
        "after_wireless_sgd_batch_loss": float(wireless_loss),
        "visual_gradient_norm": float(gradient[:model.visual_dimension].norm()),
        "text_gradient_norm": float(gradient[model.visual_dimension:-1].norm()),
        "shared_gradient": float(gradient[-1]),
        "backbone_frozen": all(not parameter.requires_grad for parameter in model.backbone.parameters()),
        "communication": communication, "elapsed_s": time.monotonic() - started,
        "smoke_configuration": settings,
        "note": "Two-client duplicate-gradient integration smoke, not a federated benchmark or unlearning result.",
    }
    destination = ROOT / "results/real_clip_smoke"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / f"{device_name}.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(destination / f"{device_name}_updates.npz", learning=updates,
                        channels=channels, received=recovered)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    main(parser.parse_args().device)
