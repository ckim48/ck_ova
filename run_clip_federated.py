import argparse
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

from clip_model import DeepPromptCLIP, projected_coupling
from data_schedule import dirichlet_partition, make_schedule
from real_data import CIFAR100
from transport import communicate, link_matrix


ROOT = Path(__file__).resolve().parent


def require_device(name):
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable. This runner will not silently substitute a CPU experiment.")
        probe = torch.ones(16, device="cuda")
        torch.cuda.synchronize()
        if float(probe.sum()) != 16:
            raise RuntimeError("CUDA arithmetic check failed.")
    return torch.device(name)


def local_gradient(model, theta, data, text, indices, device, microbatch):
    result = torch.zeros_like(theta)
    for start in range(0, len(indices), microbatch):
        selected = indices[start:start + microbatch]
        pixels, targets = data.batch(selected, device)
        variable = theta.detach().requires_grad_(True)
        predictions = model(pixels, **text, theta=variable)
        value = functional.cross_entropy(predictions, targets)
        result += torch.autograd.grad(value, variable)[0].detach() * len(selected) / len(indices)
    return result


def evaluate(model, theta, data, text, indices, device, microbatch, split="train"):
    if len(indices) == 0:
        return {"loss": None, "accuracy": None, "samples": 0}
    total_loss = 0.0
    correct = 0
    with torch.no_grad():
        for start in range(0, len(indices), microbatch):
            selected = indices[start:start + microbatch]
            pixels, targets = data.batch(selected, device, split)
            predictions = model(pixels, **text, theta=theta)
            total_loss += float(functional.cross_entropy(predictions, targets, reduction="sum"))
            correct += int((predictions.argmax(dim=-1) == targets).sum())
    return {"loss": total_loss / len(indices), "accuracy": correct / len(indices),
            "samples": len(indices)}


def execute(args):
    config = json.loads(args.config.read_text())
    if args.seed is not None:
        config["seed"] = args.seed
    if args.algorithm == "retrain_reference" and args.transport != "ideal":
        raise ValueError("The retraining reference uses ideal communication.")
    device = require_device(args.device)
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config["seed"])
        torch.backends.cudnn.benchmark = False
    checkpoint = ROOT / "checkpoints/clip-vit-base-patch16"
    dataset = CIFAR100(ROOT / "data/cifar-100-python", checkpoint)
    training_ids, validation_ids = dataset.validation_split(
        config["seed"], config["validation_per_class"]
    )
    partitions = dirichlet_partition(
        dataset.train_labels[training_ids], config["users"],
        config["dirichlet_alpha"], config["seed"],
    )
    partitions = [training_ids[partition] for partition in partitions]
    schedule = make_schedule(
        partitions, rounds=config["rounds"], initial_rounds=config["initial_rounds"],
        requests=config["request_rounds"], initial_fraction=config["initial_fraction"],
        deletion_fraction=config["deletion_fraction"], requesting_users=config["requesting_users"],
        batch_size=config["batch_size"], seed=config["seed"],
    )
    final_deleted = set(schedule[-1]["deleted"])
    model = DeepPromptCLIP.from_local_checkpoint(
        checkpoint, prompt_layers=config["prompt_layers"], prompt_tokens=config["prompt_tokens"],
        checkpoint_layers=config["checkpoint_layers"],
    ).to(device)
    text = dataset.text_inputs(device, config["prompt_tokens"])
    theta = model.initial_vector.detach().clone()
    channels = link_matrix(config["users"], config["seed"] + 2000, config)
    destination = args.output or ROOT / "results/clip" / (
        f"{args.algorithm}_{args.transport}_seed{config['seed']}"
    )
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "rounds.jsonl").exists():
        raise FileExistsError("Use a new output directory; existing experiment will not be overwritten.")
    validation_ids = np.random.default_rng(config["seed"]).permutation(validation_ids)
    validation_ids = validation_ids[:config["evaluation_samples"]]
    manifest = {
        "config": config, "algorithm": args.algorithm, "transport": args.transport,
        "device": str(device), "device_name": torch.cuda.get_device_name() if device.type == "cuda" else "CPU",
        "torch": torch.__version__, "host": platform.node(),
        "trainable_parameters": len(theta),
        "coupling": "rank-projected central finite-difference retained-loss Hessian",
        "data": "real CIFAR-100; training validation split; pretrained CLIP ViT-B/16",
        "status": "running",
        "deletion_schedule": "shared canonical one-step schedule, sampled before algorithm comparisons",
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                          for name in ("run_clip_federated.py", "clip_model.py", "transport.py", "wireless.py")},
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (destination / "schedule.json").write_text(json.dumps(schedule) + "\n")
    np.savez_compressed(destination / "partition.npz", training_ids=training_ids,
                        validation_ids=validation_ids, channels=channels,
                        **{f"user_{index}": ids for index, ids in enumerate(partitions)})
    torch.save({"theta": theta.cpu(), "round": 0}, destination / "initial.pt")
    cumulative_bound = 0.0
    cumulative_expected = 0.0
    cohort_ids = {}
    initial_evaluation = evaluate(model, theta, dataset, text, validation_ids, device,
                                  config["microbatch_size"])
    (destination / "initial_evaluation.json").write_text(json.dumps(initial_evaluation, indent=2) + "\n")
    started = time.monotonic()
    try:
        for record in schedule:
            round_number = record["round"]
            if args.algorithm == "retrain_reference":
                record = dict(record)
                record["retained"] = [sorted(set(ids) - final_deleted) for ids in record["retained"]]
                record["batches"] = [
                    np.random.default_rng(batch_seed).choice(
                        ids, min(config["batch_size"], len(ids)), replace=False
                    ).tolist()
                    for ids, batch_seed in zip(record["retained"], record["batch_seeds"])
                ]
            rate = config["learning_rate"] * 0.5 * (
                1 + math.cos(math.pi * (round_number - 1) / config["rounds"])
            )
            learning = np.zeros((config["users"], len(theta)))
            unlearning = np.zeros_like(learning)
            retained_count = sum(map(len, record["retained"]))
            forget_count = sum(map(len, record["forget"]))
            denominator = retained_count
            if args.algorithm == "puf_regular_adapted" and forget_count:
                denominator = sum(len(forgotten) if forgotten else len(retained)
                                  for forgotten, retained in zip(record["forget"], record["retained"]))
            coupling_statistics = []
            for user, batch in enumerate(record["batches"]):
                forgotten = record["forget"][user]
                if batch and not (args.algorithm == "puf_regular_adapted" and forgotten):
                    gradient = local_gradient(
                        model, theta, dataset, text, batch, device, config["microbatch_size"]
                    )
                    learning[user] = (
                        -rate * len(record["retained"][user]) / denominator * gradient
                    ).cpu().numpy()
                if not forgotten or args.algorithm in ("retain_only", "retrain_reference"):
                    continue
                forgotten_batch = np.random.default_rng(
                    config["seed"] * 100000 + round_number * 100 + user
                ).choice(forgotten, min(config["batch_size"], len(forgotten)), replace=False).tolist()
                forget_gradient = local_gradient(
                    model, theta, dataset, text, forgotten_batch, device, config["microbatch_size"]
                )
                if args.algorithm == "coupled" and batch:
                    curvature_batch = batch[:config["microbatch_size"]]
                    pixels, targets = dataset.batch(curvature_batch, device)
                    loss_function = lambda value: functional.cross_entropy(
                        model(pixels, **text, theta=value), targets
                    )
                    forget_gradient, statistics = projected_coupling(
                        loss_function, theta, forget_gradient,
                        model.visual_dimension, model.text_dimension,
                        rank=config["coupling_rank"], kappa_requested=config["coupling_kappa"],
                        xi=config["coupling_xi"], difference_step=config["coupling_difference_step"],
                        seed=config["seed"] + round_number * 100 + user,
                    )
                    coupling_statistics.append(statistics)
                coefficient = len(forgotten) / forget_count
                if args.algorithm == "puf_regular_adapted":
                    coefficient = config["puf_unlearning_multiplier"] * len(forgotten) / denominator
                unlearning[user] = (rate * coefficient * forget_gradient).cpu().numpy()
            recovered, transmission = communicate(
                learning, unlearning, channels, config, args.transport,
                seed=config["seed"] * 100000 + round_number,
            )
            candidate = theta + torch.as_tensor(recovered, device=device, dtype=theta.dtype)
            if not torch.isfinite(candidate).all() or candidate[-1] > 20:
                raise RuntimeError("Model diverged; no clipping or silent recovery applied.")
            theta = candidate.detach()
            cumulative_bound += transmission["energy_bound_j"]
            cumulative_expected += transmission["expected_energy_random_codewords_j"]
            new_cohort = sorted(set().union(*map(set, record["forget"])))
            if new_cohort:
                cohort_ids[round_number] = np.random.default_rng(
                    config["seed"] + round_number
                ).permutation(new_cohort)[:config["evaluation_samples"]].tolist()
            metrics = {
                "round": round_number, "learning_rate": rate,
                "newly_deleted": len(new_cohort), "cumulative_energy_bound_j": cumulative_bound,
                "cumulative_expected_energy_j": cumulative_expected,
                "transmission": transmission, "coupling": coupling_statistics,
                "elapsed_s": time.monotonic() - started,
            }
            if new_cohort or round_number % config["evaluate_every"] == 0 or round_number == config["rounds"]:
                metrics["validation"] = evaluate(
                    model, theta, dataset, text, validation_ids, device, config["microbatch_size"]
                )
                metrics["fixed_forget_cohorts"] = {
                    str(request): evaluate(model, theta, dataset, text, ids, device,
                                           config["microbatch_size"])
                    for request, ids in cohort_ids.items()
                }
            with (destination / "rounds.jsonl").open("a") as handle:
                handle.write(json.dumps(metrics) + "\n")
            temporary = destination / "latest.partial.pt"
            torch.save({"theta": theta.cpu(), "round": round_number}, temporary)
            temporary.replace(destination / "latest.pt")
            print(json.dumps({"round": round_number, "energy_j": cumulative_bound,
                              "validation": metrics.get("validation"), "elapsed_s": metrics["elapsed_s"]}),
                  flush=True)
        manifest["status"] = "completed"
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = str(error)
        raise
    finally:
        (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/clip_cifar100.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--algorithm", choices=("coupled", "uncoupled", "retain_only",
                                               "puf_regular_adapted", "retrain_reference"),
                        default="coupled")
    parser.add_argument("--transport", choices=("ideal", "hybrid", "fixed_head",
                                               "orthogonal_equal_time", "mixed_aircomp"), default="ideal")
    execute(parser.parse_args())
