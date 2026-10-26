"""Counterfactual multi-request single-round replay from frozen round-60 state."""
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/overnight_energy_v2/crossmodal_ablation_v1/prepared"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def main():
    global OUT
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["crossmodal", "vanilla"], required=True)
    parser.add_argument("--reference-budget-dir", type=Path,
                        help="Reuse application RMSE budgets from a reference variant")
    args = parser.parse_args()
    OUT = OUT / args.variant
    sys.path[:0] = [str(ROOT), str(ROOT / "tools/energy_quality_v1"),
                   str(ROOT / "tools/overnight_energy_v2"),
                   str(ROOT / "tools/overnight_energy_v2/receiver_control_v1")]
    import torch
    import torch.nn.functional as F
    from clip_model import DeepPromptCLIP, projected_coupling
    from real_data import CIFAR100
    from run_clip_federated import local_gradient
    from phase_split_grid import make_inputs
    from receiver_control import control, sensitivity_bound

    torch.set_num_threads(4)
    torch.manual_seed(0)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no silent CPU substitution")
    device = torch.device("cuda")
    source = ROOT / "results/overnight_energy_v2/public_history_fu_v19"
    observer = source / "observer"
    family = ROOT / "results/overnight_energy_v2/burst_joint_v3/hybrid"
    base_plan = json.loads((ROOT / "results/overnight_energy_v2/public_history_fu_v19/screen/plan.json").read_text())
    assert all(sha(ROOT / name) == value for name, value in base_plan["core_sha256"].items())
    cfg = json.loads((ROOT / "results/overnight_energy_v2/burst_joint_v3/config.json").read_text())
    schedule = json.loads((family / "schedule.json").read_text())
    record = schedule[60]
    assert record["round"] == 61
    prior = schedule[:60]
    actual = [u for u, values in enumerate(record["forget"]) if values]
    assert actual == [5, 6], actual
    additions = np.random.default_rng(6108).permutation([u for u in range(10) if u not in actual]).tolist()
    forget_by_user = {u: list(record["forget"][u]) for u in range(10)}
    for user in additions:
        used = set().union(*(set(r["batches"][user]) for r in prior))
        previously_deleted = set().union(*(set(r["forget"][user]) for r in prior))
        candidates = sorted(used - previously_deleted)
        count = int(len(candidates) * cfg["deletion_fraction"])
        forget_by_user[user] = np.random.default_rng(610800 + user).choice(
            candidates, count, replace=False).astype(int).tolist()

    state = torch.load(observer / "round_060.pt", map_location="cpu", weights_only=True)
    assert state["round"] == 60
    theta = state["theta"].to(device)
    checkpoint = ROOT / "checkpoints/clip-vit-base-patch16"
    model = DeepPromptCLIP.from_local_checkpoint(
        checkpoint, prompt_layers=cfg["prompt_layers"],
        prompt_tokens=cfg["prompt_tokens"], checkpoint_layers=True).to(device).eval()
    dataset = CIFAR100(ROOT / "data/cifar-100-python", checkpoint)
    text = dataset.text_inputs(device, cfg["prompt_tokens"])
    with np.load(family / "partition.npz") as data:
        channels = data["channels"].copy()
    # Reuse the predeclared held-out calibration gradient at the same frozen state.
    with np.load(ROOT / "results/overnight_energy_v2/receiver_control_v1/prepared.npz") as data:
        g = data["gradient"].copy()
    rate = cfg["learning_rate"] * 0.5 * (1 + math.cos(math.pi * 60 / cfg["rounds"]))
    added_batches = {}
    added_forget_grads = {}
    for user in additions:
        retained = set(record["retained"][user]) - set(forget_by_user[user])
        added_batches[user] = np.random.default_rng(record["batch_seeds"][user]).choice(
            sorted(retained), min(cfg["batch_size"], len(retained)), replace=False).astype(int).tolist()
        chosen = np.random.default_rng(cfg["seed"] * 100000 + 61 * 100 + user + 610800).choice(
            forget_by_user[user], min(cfg["batch_size"], len(forget_by_user[user])), replace=False).astype(int).tolist()
        added_forget_grads[user] = local_gradient(
            model, theta, dataset, text, chosen, device, cfg["microbatch_size"])

    for count_users in [5, 8]:
        active = sorted(actual + additions[:count_users - 2])
        retained_sets, forget_sets, batches = [], [], []
        for user in range(10):
            forgotten = set(forget_by_user[user]) if user in active else set()
            retained = set(record["retained"][user])
            if user in additions and user in active:
                retained -= forgotten
            retained_sets.append(sorted(retained))
            forget_sets.append(sorted(forgotten))
            if user in actual:
                batches.append(list(record["batches"][user]))
            elif user in additions and user in active:
                batches.append(added_batches[user])
            else:
                batches.append(list(record["batches"][user]))

        retained_total = sum(map(len, retained_sets))
        forget_total = sum(map(len, forget_sets))
        learning = np.zeros((10, len(theta)), dtype=np.float64)
        unlearning = np.zeros_like(learning)
        coupling = []
        for user, batch in enumerate(batches):
            if not batch:
                continue
            grad = local_gradient(model, theta, dataset, text, batch, device, cfg["microbatch_size"])
            learning[user] = (-rate * len(retained_sets[user]) / retained_total * grad).detach().cpu().numpy()
            if user not in active:
                continue
            if user in actual:
                chosen = np.random.default_rng(cfg["seed"] * 100000 + 61 * 100 + user).choice(
                    forget_sets[user], min(cfg["batch_size"], len(forget_sets[user])), replace=False).astype(int).tolist()
                forget_grad = local_gradient(model, theta, dataset, text, chosen, device, cfg["microbatch_size"])
            else:
                forget_grad = added_forget_grads[user]
            if args.variant == "crossmodal":
                pixels, labels = dataset.batch(batch[:cfg["microbatch_size"]], device)
                loss = lambda value: F.cross_entropy(model(pixels, **text, theta=value), labels)
                forget_grad, stats = projected_coupling(
                    loss, theta, forget_grad, model.visual_dimension, model.text_dimension,
                    rank=cfg["coupling_rank"], kappa_requested=cfg["coupling_kappa"],
                    xi=cfg["coupling_xi"], difference_step=cfg["coupling_difference_step"],
                    seed=cfg["seed"] + 61 * 100 + user)
            else:
                stats = dict(kappa=0.0, approximate_cross_norm=None,
                             preconditioner_min_eigenvalue=1.0, rank=0)
            coupling.append(dict(user=user, **stats))
            unlearning[user] = (rate * len(forget_sets[user]) / forget_total * forget_grad).detach().cpu().numpy()

        inputs = make_inputs(learning, unlearning, dict(cfg, quantization_bits=8))
        l, _, quantized, delta, payload, rho, dims, total = inputs
        flattened = quantized.reshape(10, -1)[:, :len(theta)]
        anchors = flattened[active]
        aggregate = learning.sum(axis=0)
        target, control_info = control(aggregate, anchors, g, 1.0)
        epsilon = 0.2 * float(np.linalg.norm(rho.sum(axis=0)))
        if args.reference_budget_dir is not None:
            reference = json.loads((args.reference_budget_dir / f"k{count_users}" / "geometry.json").read_text())
            epsilon = float(reference["application_rmse_budget"])
        lipschitz = float(sensitivity_bound(anchors, g))
        if not np.isfinite(lipschitz) or lipschitz <= 0 or not np.isfinite(epsilon) or epsilon <= 0:
            raise RuntimeError("Invalid application error budget")
        output = OUT / f"k{count_users}"
        output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output / "prepared.npz", learning=learning, unlearning=unlearning,
            quantized=flattened, channels=channels, gradient=g, active=np.asarray(active),
            target=target, canonical_sum=aggregate + flattened.sum(axis=0),
            delta=delta, payload=payload, rho=rho, dims=dims, total_symbols=total)
        details = dict(status="prepared", variant=args.variant, crossmodal_enabled=(args.variant == "crossmodal"), K=count_users, actual_users=actual,
            added_users=additions[:count_users - 2], active_users=active,
            forget_counts={str(u): len(forget_sets[u]) for u in active},
            retained_counts={str(u): len(retained_sets[u]) for u in range(10)},
            forget_total=forget_total, retained_total=retained_total,
            learning_norm=float(np.linalg.norm(aggregate)),
            quantized_unlearning_norm=float(np.linalg.norm(flattened)),
            target_norm=float(np.linalg.norm(target)), application_rmse_budget=epsilon,
            controller_lipschitz=lipschitz, controller=control_info, coupling=coupling,
            error_budget_source=(str(args.reference_budget_dir.resolve()) if args.reference_budget_dir else "variant_specific_20pct_update_norm"),
            learning_rate=rate, device=torch.cuda.get_device_name(0),
            frozen_checkpoint_sha256=sha(observer / "round_060.pt"),
            source_hashes={name: sha(ROOT / name) for name in
                ["run_clip_federated.py", "clip_model.py", "transport.py", "wireless.py"]})
        write(output / "geometry.json", details)
        print("PREPARED", count_users, json.dumps({k: details[k] for k in
            ["active_users", "forget_total", "controller", "application_rmse_budget", "controller_lipschitz"]}), flush=True)
    write(OUT / "prepare_status.json", dict(status="completed", variant=args.variant, crossmodal_enabled=(args.variant == "crossmodal"), cases=[5, 8]))


if __name__ == "__main__":
    main()


