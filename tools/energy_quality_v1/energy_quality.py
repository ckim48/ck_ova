"""Final-checkpoint energy/utility/forgetting audit; never changes training files."""
import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


METHODS = (
    "coupled_hybrid", "coupled_orthogonal_equal_time", "coupled_mixed_aircomp",
    "coupled_fixed_head", "uncoupled_hybrid", "puf_regular_adapted_hybrid",
    "coupled_ideal", "retrain_reference_ideal",
    "coupled_hybrid_q6",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_rounds(directory):
    path = Path(directory) / "rounds.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines(keepends=True)
            if line.endswith("\n")]


def auc(member_scores, nonmember_scores):
    """P(member score > nonmember score) + half the probability of a tie."""
    a, b = np.asarray(member_scores), np.sort(np.asarray(nonmember_scores))
    if not len(a) or not len(b):
        return None
    left = np.searchsorted(b, a, side="left")
    right = np.searchsorted(b, a, side="right")
    return float(np.mean((left + right) / (2 * len(b))))


def losses(probabilities, labels):
    return -np.log(np.maximum(probabilities[np.arange(len(labels)), labels], 1e-300))


def classification(probabilities, labels):
    if not len(labels):
        return {"loss": None, "accuracy": None, "samples": 0}
    return {"loss": float(losses(probabilities, labels).mean()),
            "accuracy": float((probabilities.argmax(axis=1) == labels).mean()),
            "samples": int(len(labels))}


def reference_gap(candidate, reference, labels):
    p, q = np.maximum(candidate, 1e-300), np.maximum(reference, 1e-300)
    return {
        "kl_reference_to_candidate": max(0.0, float(np.mean(np.sum(q * (np.log(q) - np.log(p)), axis=1)))),
        "total_variation": float(np.mean(np.abs(p - q).sum(axis=1) / 2)),
        "prediction_disagreement": float(np.mean(p.argmax(axis=1) != q.argmax(axis=1))),
        "signed_loss_gap": float(np.mean(losses(p, labels) - losses(q, labels))),
    }


def nondominated(rows, keys):
    """Indices on the observed frontier; every key is minimized."""
    eligible = [i for i, row in enumerate(rows)
                if all(row.get(k) is not None and np.isfinite(row[k]) for k in keys)]
    return [i for i in eligible if not any(
        all(rows[j][k] <= rows[i][k] for k in keys)
        and any(rows[j][k] < rows[i][k] for k in keys)
        for j in eligible if j != i)]


def stratified_sample(ids, labels, size, rng):
    ids = np.asarray(sorted(set(map(int, ids))), dtype=np.int64)
    if size >= len(ids):
        return ids.tolist()
    classes, counts = np.unique(labels[ids], return_counts=True)
    quotas = size * counts / counts.sum()
    take = np.floor(quotas).astype(int)
    order = np.argsort(-(quotas - take), kind="stable")
    take[order[:size - take.sum()]] += 1
    selected = []
    for label, count in zip(classes, take):
        selected.extend(rng.choice(ids[labels[ids] == label], count, replace=False).tolist())
    return sorted(selected)


def make_cohorts(dataset, source, config, samples):
    schedule = json.loads((source / "schedule.json").read_text())
    deleted = set(schedule[-1]["deleted"])
    retained = set().union(*(set(x) for x in schedule[-1]["retained"]))
    if deleted & retained:
        raise ValueError("Retained and deleted cohorts overlap.")
    _, validation = dataset.validation_split(config["seed"], config["validation_per_class"])
    rng = np.random.default_rng(config["seed"] + 91021)
    utility_val, attack_val = [], []
    for label in range(100):
        ids = rng.permutation(validation[dataset.train_labels[validation] == label])
        half = len(ids) // 2
        utility_val.extend(ids[:half].tolist())
        attack_val.extend(ids[half:].tolist())
    # Exact class matching prevents class imbalance from driving the loss attack.
    forget_pool, nonmember_pool = [], []
    deleted_array = np.asarray(sorted(deleted), dtype=np.int64)
    attack_array = np.asarray(attack_val, dtype=np.int64)
    for label in range(100):
        f = deleted_array[dataset.train_labels[deleted_array] == label]
        n = attack_array[dataset.train_labels[attack_array] == label]
        t = np.flatnonzero(dataset.test_labels == label)
        count = min(len(f), len(n), len(t))
        forget_pool.extend(rng.permutation(f)[:count].tolist())
        nonmember_pool.extend(rng.permutation(n)[:count].tolist())
    forget = stratified_sample(forget_pool, dataset.train_labels, samples, rng)
    val_nonmember, test_nonmember = [], []
    for label in range(100):
        count = int(np.sum(dataset.train_labels[forget] == label))
        n = np.asarray(nonmember_pool, dtype=np.int64)
        n = n[dataset.train_labels[n] == label]
        t = np.flatnonzero(dataset.test_labels == label)
        val_nonmember.extend(rng.choice(n, count, replace=False).tolist())
        test_nonmember.extend(rng.choice(t, count, replace=False).tolist())
    groups = {
        "validation": {"split": "train", "ids": stratified_sample(utility_val, dataset.train_labels, samples, rng)},
        "retained": {"split": "train", "ids": stratified_sample(retained, dataset.train_labels, samples, rng)},
        "forget": {"split": "train", "ids": forget},
        "nonmember_validation": {"split": "train", "ids": sorted(val_nonmember)},
        "test": {"split": "test", "ids": stratified_sample(range(len(dataset.test_labels)), dataset.test_labels, samples, rng)},
        "nonmember_test": {"split": "test", "ids": sorted(test_nonmember)},
    }
    if not forget:
        raise ValueError("No class-matched forget/nonmember samples available.")
    return {"seed": config["seed"], "samples_per_group_limit": samples,
            "source_schedule_sha256": digest(source / "schedule.json"), "groups": groups,
            "interpretation": "Final-time reference only. Loss-based MIA is a diagnostic, not certified unlearning. Test metrics are not used to choose configurations."}


def predict(model, theta, dataset, text, ids, split, device, batch_size):
    import torch
    import torch.nn.functional as functional
    with torch.no_grad():
        visual, textual, scale = model.parts(theta)
        text_features = functional.normalize(model.text_features(**text, text=textual), dim=-1)
        result = []
        for start in range(0, len(ids), batch_size):
            pixels, _ = dataset.batch(ids[start:start + batch_size], device, split)
            image_features = functional.normalize(model.image_features(pixels, visual), dim=-1)
            logits = scale.exp() * image_features @ text_features.T
            result.append(logits.double().softmax(dim=-1).cpu().numpy())
    return np.concatenate(result) if result else np.empty((0, len(dataset.classes)))


def validate_completed(directory, suite, protocol, source):
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        return None, "not_started"
    manifest = json.loads(manifest_path.read_text())
    records = read_rounds(directory)
    if manifest["status"] != "completed":
        return None, manifest["status"]
    if not records or records[-1]["round"] != protocol["rounds"]:
        return None, "invalid_final_round"
    expected_config = dict(protocol)
    if directory.name == "coupled_hybrid_q6":
        expected_config["quantization_bits"] = 6
        if (manifest["algorithm"], manifest["transport"]) != ("coupled", "hybrid"):
            return None, "incompatible_variant_algorithm"
    if manifest["config"] != expected_config:
        return None, "incompatible_config"
    expected = json.loads((suite / "suite_metadata.json").read_text())["source_sha256"]
    if manifest["source_sha256"] != expected:
        return None, "incompatible_source_hashes"
    if digest(directory / "schedule.json") != digest(source / "schedule.json"):
        return None, "incompatible_schedule"
    return (manifest, records), "completed"


def collect(args):
    import torch
    sys.path.insert(0, str(args.project))
    from clip_model import DeepPromptCLIP
    from real_data import CIFAR100
    from run_clip_federated import require_device
    config = json.loads((args.suite / "config.json").read_text())
    source = args.suite / "coupled_hybrid"
    args.output.mkdir(parents=True, exist_ok=True)
    for name, sha in json.loads((args.suite / "suite_metadata.json").read_text())["source_sha256"].items():
        if digest(args.project / name) != sha:
            raise ValueError(f"Training source changed: {name}")
    device = require_device(args.device)
    torch.set_num_threads(args.cpu_threads)
    checkpoint = args.project / "checkpoints/clip-vit-base-patch16"
    dataset = CIFAR100(args.project / "data/cifar-100-python", checkpoint)
    cohorts = make_cohorts(dataset, source, config, args.samples)
    cohort_path = args.output / "cohorts.json"
    if cohort_path.exists() and json.loads(cohort_path.read_text()) != cohorts:
        raise ValueError("Existing cohorts differ. Choose a new output directory.")
    write_json(cohort_path, cohorts)
    model = None
    statuses = {}
    for name in METHODS:
        directory = args.suite / name
        validated, status = validate_completed(directory, args.suite, config, source)
        statuses[name] = status
        if validated is None:
            continue
        manifest, records = validated
        checkpoint_path = directory / "latest.pt"
        provenance = {"checkpoint_sha256": digest(checkpoint_path),
                      "cohorts_sha256": digest(cohort_path), "evaluator_sha256": digest(__file__),
                      "algorithm": manifest["algorithm"], "transport": manifest["transport"],
                      "training_source_sha256": manifest["source_sha256"],
                      "initial_checkpoint_sha256": digest(directory / "initial.pt")}
        if provenance["initial_checkpoint_sha256"] != digest(source / "initial.pt"):
            # Torch archives can differ in serialization metadata; compare tensors.
            a = torch.load(directory / "initial.pt", map_location="cpu", weights_only=True)
            b = torch.load(source / "initial.pt", map_location="cpu", weights_only=True)
            if not torch.equal(a["theta"], b["theta"]):
                raise ValueError(f"Initial model differs: {name}")
        prediction_path = args.output / (name + ".npz")
        cache_path = args.output / (name + ".cache.json")
        if prediction_path.exists() and cache_path.exists() and json.loads(cache_path.read_text()) == provenance:
            print("CACHED", name, flush=True)
            continue
        checkpoint_data = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint_data["round"] != config["rounds"]:
            raise ValueError(f"Checkpoint is not final: {name}")
        if model is None:
            model = DeepPromptCLIP.from_local_checkpoint(checkpoint,
                prompt_layers=config["prompt_layers"], prompt_tokens=config["prompt_tokens"],
                checkpoint_layers=False).to(device).eval()
        theta = checkpoint_data["theta"].to(device)
        text = dataset.text_inputs(device, config["prompt_tokens"])
        predictions = {}
        for group, spec in cohorts["groups"].items():
            print("EVALUATE", name, group, len(spec["ids"]), flush=True)
            predictions[group] = predict(model, theta, dataset, text, spec["ids"], spec["split"], device, args.batch_size)
            labels = dataset.train_labels if spec["split"] == "train" else dataset.test_labels
            predictions[group + "_labels"] = labels[spec["ids"]]
        temporary = prediction_path.with_suffix(".partial.npz")
        np.savez_compressed(temporary, **predictions)
        temporary.replace(prediction_path)
        write_json(cache_path, provenance)
    write_json(args.output / "evaluation_status.json", statuses)
    summarize(args)


def summarize(args):
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((args.suite / "config.json").read_text())
    source = args.suite / "coupled_hybrid"

    def cached_predictions(name):
        directory = args.suite / name
        validated, _ = validate_completed(directory, args.suite, protocol, source)
        p = args.output / (name + ".npz")
        c = args.output / (name + ".cache.json")
        if validated is None or not p.exists() or not c.exists():
            return None
        cache = json.loads(c.read_text())
        expected = {"checkpoint_sha256": digest(directory / "latest.pt"),
                    "cohorts_sha256": digest(args.output / "cohorts.json"),
                    "evaluator_sha256": digest(__file__),
                    "training_source_sha256": validated[0]["source_sha256"]}
        if not all(cache.get(k) == v for k, v in expected.items()):
            return None
        with np.load(p) as arrays:
            return {k: arrays[k].copy() for k in arrays.files}

    reference = cached_predictions("retrain_reference_ideal")
    rows = []
    for name in METHODS:
        d = args.suite / name
        f = d / "manifest.json"
        if not f.exists():
            rows.append({"method": name, "status": "not_started"})
            continue
        m = json.loads(f.read_text()); rounds = read_rounds(d)
        row = {"method": name, "status": m["status"], "algorithm": m["algorithm"],
               "transport": m["transport"], "energy_role": "reference_only" if m["transport"] == "ideal" else "physical_model",
               "round": rounds[-1]["round"] if rounds else 0,
               "quantization_bits": m["config"].get("quantization_bits"),
               "relative_tolerance": m["config"].get("relative_tolerance")}
        if rounds:
            last = rounds[-1]
            row.update(energy_bound_j=last["cumulative_energy_bound_j"],
                       expected_energy_j=last["cumulative_expected_energy_j"],
                       cumulative_uplink_s=sum(x["transmission"]["uplink_s"] for x in rounds),
                       deletion_round_energy_j=sum(x["transmission"]["energy_bound_j"] for x in rounds if x.get("newly_deleted", 0)))
        predictions = cached_predictions(name)
        row["quality_status"] = "evaluated" if predictions is not None else "pending"
        if predictions is not None:
            for group in ("validation", "retained", "forget", "test"):
                metric = classification(predictions[group], predictions[group + "_labels"])
                for key, value in metric.items():
                    row[group + "_" + key] = value
            for nonmember, tag in (("nonmember_validation", "validation"), ("nonmember_test", "test")):
                row["forget_loss_attack_auc_" + tag] = auc(
                    -losses(predictions["forget"], predictions["forget_labels"]),
                    -losses(predictions[nonmember], predictions[nonmember + "_labels"]))
                if reference is not None:
                    ref_auc = auc(-losses(reference["forget"], reference["forget_labels"]),
                                  -losses(reference[nonmember], reference[nonmember + "_labels"]))
                    row["forget_attack_auc_gap_to_reference_" + tag] = abs(row["forget_loss_attack_auc_" + tag] - ref_auc)
            if reference is not None:
                for group in ("validation", "retained", "forget", "test"):
                    if not np.array_equal(predictions[group + "_labels"], reference[group + "_labels"]):
                        raise ValueError("Mismatched prediction cohorts.")
                    for key, value in reference_gap(predictions[group], reference[group], predictions[group + "_labels"]).items():
                        row[group + "_" + key] = value
        rows.append(row)
    physical = [x for x in rows if x.get("status") == "completed"
                and x.get("energy_role") == "physical_model" and x.get("quality_status") == "evaluated"]
    keys = ("energy_bound_j", "validation_loss", "forget_kl_reference_to_candidate")
    frontier = [physical[i]["method"] for i in nondominated(physical, keys)]
    constraints = {k: v for k, v in {
        "validation_loss": args.validation_loss_max,
        "forget_kl_reference_to_candidate": args.forget_kl_max,
        "forget_attack_auc_gap_to_reference_validation": args.attack_gap_max}.items() if v is not None}
    winner = None
    if len(constraints) == 3 and reference is not None:
        eligible = [x for x in physical if all(x.get(k) is not None and x[k] <= v for k, v in constraints.items())]
        if eligible:
            winner = min(eligible, key=lambda x: x["energy_bound_j"])["method"]
    report = {"reference_available": reference is not None,
              "observed_frontier_objectives": list(keys), "observed_frontier": frontier,
              "quality_constraints": constraints, "lowest_observed_energy_qualifying_method": winner,
              "selection_status": "thresholds_not_prespecified" if len(constraints) != 3 else ("reference_pending" if reference is None else "evaluated"),
              "limits": ["Final-time empirical reference; no certified forgetting guarantee.",
                         "No interpolation or extrapolation to untested configurations.",
                         "Only completed physical-link runs enter the energy frontier.",
                         "Test results never enter frontier objectives or eligibility decisions.",
                         "Energy is modeled communication energy, excluding GPU/flight energy.",
                         "Current orthogonal comparator separates analog and digital phases; digital users still use NOMA; split is fixed 50/50.",
                         "One seed does not establish statistical superiority."], "methods": rows}
    write_json(args.output / "energy_quality_report.json", report)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    temporary = args.output / "energy_quality.partial.csv"
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(args.output / "energy_quality.csv")
    print(json.dumps({k: report[k] for k in ("reference_available", "observed_frontier", "selection_status")}), flush=True)


def process_running(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--suite", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--wait-pid", type=int)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--validation-loss-max", type=float)
    parser.add_argument("--forget-kl-max", type=float)
    parser.add_argument("--attack-gap-max", type=float)
    args = parser.parse_args()
    args.project = args.project.resolve()
    args.suite = args.suite or args.project / "results/comparisons_seed0_solver_v2"
    args.output = args.output or args.project / "results/energy_quality_v1"
    if args.samples < 1 or args.batch_size < 1:
        parser.error("Sample count and batch size must be positive.")
    if args.summary_only:
        summarize(args); return
    if args.wait_pid:
        print("WAITING_FOR_TRAINING_QUEUE", args.wait_pid, flush=True)
        while process_running(args.wait_pid):
            time.sleep(30)
    collect(args)


if __name__ == "__main__":
    main()
