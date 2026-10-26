import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

from clip_model import DeepPromptCLIP, projected_coupling
from real_data import CIFAR100
from run_clip_federated import local_gradient
from transport import communicate, link_matrix
from wireless import quantize


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "results/clip/coupled_hybrid_seed0"
OUTPUT = ROOT / "results/failed_round_diagnostic"


def reconstruct():
    manifest = json.loads((SOURCE / "manifest.json").read_text())
    for name, digest in manifest["source_sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Source changed: {name}")
    config = manifest["config"]
    checkpoint = torch.load(SOURCE / "latest.pt", map_location="cpu", weights_only=True)
    schedule = json.loads((SOURCE / "schedule.json").read_text())
    record = schedule[checkpoint["round"]]
    theta = checkpoint["theta"]
    torch.set_num_threads(4)
    torch.manual_seed(config["seed"])
    model_path = ROOT / "checkpoints/clip-vit-base-patch16"
    model = DeepPromptCLIP.from_local_checkpoint(
        model_path, prompt_layers=config["prompt_layers"], prompt_tokens=config["prompt_tokens"],
        checkpoint_layers=config["checkpoint_layers"],
    )
    dataset = CIFAR100(ROOT / "data/cifar-100-python", model_path)
    device = torch.device("cpu")
    text = dataset.text_inputs(device, config["prompt_tokens"])
    rate = config["learning_rate"] * 0.5 * (
        1 + math.cos(math.pi * (record["round"] - 1) / config["rounds"])
    )
    learning = np.zeros((config["users"], len(theta)))
    unlearning = np.zeros_like(learning)
    retained_count = sum(map(len, record["retained"]))
    forget_count = sum(map(len, record["forget"]))
    for user, batch in enumerate(record["batches"]):
        print(f"Reconstruct round {record['round']} CPU gradient {user + 1}/{config['users']}",
              flush=True)
        if batch:
            gradient = local_gradient(model, theta, dataset, text, batch, device,
                                      config["microbatch_size"])
            learning[user] = (-rate * len(record["retained"][user])
                              / retained_count * gradient).numpy()
        forgotten = record["forget"][user]
        if not forgotten:
            continue
        forgotten_batch = np.random.default_rng(
            config["seed"] * 100000 + record["round"] * 100 + user
        ).choice(forgotten, min(config["batch_size"], len(forgotten)), replace=False).tolist()
        gradient = local_gradient(model, theta, dataset, text, forgotten_batch, device,
                                  config["microbatch_size"])
        if batch:
            pixels, targets = dataset.batch(batch[:config["microbatch_size"]], device)
            loss_function = lambda value: functional.cross_entropy(
                model(pixels, **text, theta=value), targets
            )
            gradient, _ = projected_coupling(
                loss_function, theta, gradient, model.visual_dimension, model.text_dimension,
                rank=config["coupling_rank"], kappa_requested=config["coupling_kappa"],
                xi=config["coupling_xi"], difference_step=config["coupling_difference_step"],
                seed=config["seed"] + record["round"] * 100 + user,
            )
        unlearning[user] = (rate * len(forgotten) / forget_count * gradient).numpy()
    channels = link_matrix(config["users"], config["seed"] + 2000, config)
    np.savez_compressed(OUTPUT / "updates.npz", learning=learning, unlearning=unlearning,
                        channels=channels)
    metadata = {
        "round": record["round"], "checkpoint_round": checkpoint["round"],
        "checkpoint_sha256": hashlib.sha256((SOURCE / "latest.pt").read_bytes()).hexdigest(),
        "device": "cpu", "config": config,
        "note": "CPU reconstruction from actual GPU checkpoint; not bitwise GPU replay or full baseline training.",
    }
    (OUTPUT / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if not (OUTPUT / "metadata.json").exists():
        reconstruct()
    metadata = json.loads((OUTPUT / "metadata.json").read_text())
    config = metadata["config"]
    arrays = np.load(OUTPUT / "updates.npz")
    learning, unlearning, channels = (arrays[key] for key in ("learning", "unlearning", "channels"))
    padding = (-learning.shape[1]) % config["block_dimension"]
    shape = (len(learning), -1, config["block_dimension"])
    learning_blocks = np.pad(learning, ((0, 0), (0, padding))).reshape(shape)
    unlearning_blocks = np.pad(unlearning, ((0, 0), (0, padding))).reshape(shape)
    epsilon = max(config["absolute_tolerance_floor"], config["relative_tolerance"]
                  * np.linalg.norm(np.linalg.norm(learning_blocks, axis=-1).sum(axis=0)))
    _, delta, _ = quantize(unlearning_blocks, config["quantization_bits"],
                          config["scale_metadata_bits"])
    report = {
        "round": metadata["round"], "epsilon": float(epsilon),
        "quantization_bound_by_head": [
            float(np.linalg.norm(np.delete(delta, head, axis=0).sum(axis=0)))
            for head in range(config["users"])
        ],
        "methods": {},
        "note": metadata["note"],
        "solver_sha256": hashlib.sha256((ROOT / "wireless.py").read_bytes()).hexdigest(),
    }
    for method in ("hybrid", "orthogonal_equal_time", "mixed_aircomp", "fixed_head"):
        print(f"Solving fixed-update comparison: {method}", flush=True)
        started = time.monotonic()
        try:
            _, metrics = communicate(learning, unlearning, channels, config, method,
                                     config["seed"] * 100000 + metadata["round"])
        except RuntimeError as error:
            metrics = {"status": "no_feasible_design_found", "error": str(error)}
        metrics["solve_elapsed_s"] = time.monotonic() - started
        report["methods"][method] = metrics
        (OUTPUT / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({method: metrics}), flush=True)
    if report["methods"]["hybrid"]["status"] == "no_feasible_design_found":
        if min(report["quantization_bound_by_head"]) > epsilon:
            report["diagnosis"] = "Every head exceeds epsilon from quantization bound alone."
        else:
            print("Testing solver initialization sensitivity only; physical constraints unchanged.",
                  flush=True)
            expanded = dict(config, receive_initializations=20)
            try:
                _, sensitivity = communicate(
                    learning, unlearning, channels, expanded, "hybrid",
                    config["seed"] * 100000 + metadata["round"],
                )
            except RuntimeError as error:
                sensitivity = {"status": "no_feasible_design_found", "error": str(error)}
            report["hybrid_solver_sensitivity_20_initializations"] = sensitivity
            print(json.dumps({"solver_sensitivity": sensitivity}), flush=True)
        (OUTPUT / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
