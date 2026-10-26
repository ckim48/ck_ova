import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

from coupling import coupled_direction
from data_schedule import dirichlet_partition, make_schedule
from run_wireless import ROOT, write_csv


def make_data(seed, samples=2000, dimension=4, classes=4):
    rng = np.random.default_rng(seed)
    prototypes = rng.normal(size=(classes, dimension))
    labels = rng.integers(classes, size=samples)
    features = prototypes[labels] + rng.normal(scale=0.9, size=(samples, dimension))
    test_labels = rng.integers(classes, size=800)
    test_features = prototypes[test_labels] + rng.normal(scale=0.9, size=(800, dimension))
    texts = prototypes + rng.normal(scale=0.35, size=prototypes.shape)
    return (
        torch.tensor(features, dtype=torch.float64), torch.tensor(labels),
        torch.tensor(test_features, dtype=torch.float64), torch.tensor(test_labels),
        torch.tensor(texts, dtype=torch.float64),
    )


def logits(theta, features, texts):
    dimension = features.shape[1]
    size = dimension ** 2
    identity = torch.eye(dimension, dtype=theta.dtype)
    visual = functional.normalize(features @ (identity + theta[:size].reshape(dimension, dimension)),
                                  dim=-1)
    textual = functional.normalize(texts @ (identity + theta[size:2 * size].reshape(dimension, dimension)),
                                   dim=-1)
    return theta[-1].exp() * visual @ textual.T


def loss(theta, features, labels, texts):
    return functional.cross_entropy(logits(theta, features, texts), labels)


def gradient(theta, features, labels, texts):
    value = theta.detach().requires_grad_(True)
    return torch.autograd.grad(loss(value, features, labels, texts), value)[0]


def evaluation(theta, features, labels, texts, indices=None):
    if indices is not None:
        if not len(indices):
            return None, None
        features, labels = features[indices], labels[indices]
    with torch.no_grad():
        predictions = logits(theta, features, texts)
        accuracy = (predictions.argmax(dim=1) == labels).double().mean()
        cross_entropy = functional.cross_entropy(predictions, labels)
    return float(accuracy), float(cross_entropy)


def run_method(method, seed, data, schedule, requested_kappa=1.0):
    features, labels, test_features, test_labels, texts = data
    size = features.shape[1] ** 2
    theta = torch.zeros(2 * size + 1, dtype=torch.float64)
    theta[-1] = math.log(2)
    final_deleted = set(schedule[-1]["deleted"])
    rows = []
    started = time.monotonic()
    for record in schedule:
        round_number = record["round"]
        rate = 0.0035 * 0.5 * (1 + math.cos(math.pi * (round_number - 1) / len(schedule)))
        retained = record["retained"]
        batches = record["batches"]
        forget = record["forget"]
        if method == "retrain_reference":
            retained = [sorted(set(indices) - final_deleted) for indices in retained]
            batches = [
                np.random.default_rng(batch_seed).choice(
                    indices, min(32, len(indices)), replace=False
                ).tolist()
                for indices, batch_seed in zip(retained, record["batch_seeds"])
            ]
        retain_count = sum(map(len, retained))
        forget_count = sum(map(len, forget))
        update = torch.zeros_like(theta)
        kappas = []
        minimum_eigenvalues = []
        if method == "puf_regular_adapted" and forget_count:
            denominator = sum(len(forget[user]) if forget[user] else len(retained[user])
                              for user in range(len(retained)))
        else:
            denominator = retain_count
        for user, batch in enumerate(batches):
            if batch and not (method == "puf_regular_adapted" and forget[user]):
                grad = gradient(theta, features[batch], labels[batch], texts)
                update -= rate * len(retained[user]) / denominator * grad
            if not forget[user] or method in ("retain_only", "retrain_reference"):
                continue
            forget_batch = np.random.default_rng(seed * 100000 + round_number * 100 + user).choice(
                forget[user], min(32, len(forget[user])), replace=False
            )
            grad = gradient(theta, features[forget_batch], labels[forget_batch], texts)
            if method == "coupled" and batch:
                retain_loss = lambda value: loss(value, features[batch], labels[batch], texts)
                grad, kappa, norm = coupled_direction(
                    retain_loss, theta, grad, size, size, requested_kappa, 0.1
                )
                kappas.append(kappa)
                minimum_eigenvalues.append(1 - kappa * norm)
            if method == "puf_regular_adapted":
                update += 20 * rate * len(forget[user]) / denominator * grad
            else:
                update += rate * len(forget[user]) / forget_count * grad
        theta += update
        if not torch.isfinite(theta).all():
            raise RuntimeError(f"Non-finite model: seed={seed} method={method} round={round_number}")
        if round_number % 10 == 0 or forget_count:
            test_accuracy, test_loss = evaluation(theta, test_features, test_labels, texts)
            forget_accuracy, forget_loss = evaluation(
                theta, features, labels, texts, record["deleted"]
            )
            row = {
                "seed": seed, "method": method, "round": round_number,
                "test_accuracy": test_accuracy, "test_loss": test_loss,
                "forget_accuracy": forget_accuracy, "forget_loss": forget_loss,
                "total_deleted": len(record["deleted"]), "newly_deleted": forget_count,
                "mean_kappa": float(np.mean(kappas)) if kappas else None,
                "min_preconditioner_eigenvalue": min(minimum_eigenvalues) if kappas else None,
                "elapsed_s": time.monotonic() - started,
            }
            rows.append(row)
    return theta, rows


def main(output):
    torch.set_num_threads(1)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    summary = []
    for seed in (0, 1, 2):
        data = make_data(seed)
        partitions = dirichlet_partition(data[1].numpy(), 10, 0.1, seed)
        schedule = make_schedule(partitions, seed=seed)
        (output / f"schedule_seed{seed}.json").write_text(json.dumps(schedule) + "\n")
        reference, reference_rows = run_method("retrain_reference", seed, data, schedule)
        rows.extend(reference_rows)
        reference_final = reference_rows[-1]
        np.save(output / f"model_seed{seed}_retrain_reference.npy", reference.numpy())
        for method in ("coupled", "uncoupled", "retain_only", "puf_regular_adapted"):
            theta, method_rows = run_method(method, seed, data, schedule)
            rows.extend(method_rows)
            final = method_rows[-1]
            summary.append({
                "seed": seed, "method": method, "test_accuracy": final["test_accuracy"],
                "forget_accuracy": final["forget_accuracy"],
                "retrain_test_accuracy": reference_final["test_accuracy"],
                "retrain_forget_accuracy": reference_final["forget_accuracy"],
                "absolute_forget_accuracy_gap": abs(
                    final["forget_accuracy"] - reference_final["forget_accuracy"]
                ),
                "parameter_distance_to_retrain": float(torch.linalg.vector_norm(theta - reference)),
            })
            np.save(output / f"model_seed{seed}_{method}.npy", theta.numpy())
            print(f"seed={seed} method={method} test={final['test_accuracy']:.4f} "
                  f"forget_gap={summary[-1]['absolute_forget_accuracy_gap']:.4f}", flush=True)
    write_csv(rows, output / "round_metrics.csv")
    write_csv(summary, output / "final_metrics.csv")
    (output / "provenance.json").write_text(json.dumps({
        "experiment": "synthetic_two_tower_learning_pilot",
        "model": "4D visual/text linear adapters; 33 trainable parameters",
        "datasets": "synthetic Gaussian class prototypes",
        "communication": "ideal",
        "rounds": 100, "users": 10, "seeds": [0, 1, 2],
        "puf": "one-step sample-unlearning adaptation, eta_u=20, not original reproduction",
        "retraining": "from same initialization; exclude all final deleted IDs from every past round",
        "not_a_clip_result": True,
        "membership_inference_evaluated": False,
        "torch": torch.__version__,
    }, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results/learning")
    main(parser.parse_args().output)
