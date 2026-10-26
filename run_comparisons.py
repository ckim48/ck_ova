import argparse
import csv
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LEGACY_BASE = ROOT / "results/clip/coupled_hybrid_seed0"
SUITE = ROOT / "results/comparisons_seed0_solver_v2"
BASE = SUITE / "coupled_hybrid"
JOBS = (
    ("coupled", "orthogonal_equal_time"),
    ("coupled", "mixed_aircomp"),
    ("coupled", "fixed_head"),
    ("uncoupled", "hybrid"),
    ("puf_regular_adapted", "hybrid"),
    ("coupled", "ideal"),
    ("retrain_reference", "ideal"),
)


def read_json(path):
    for attempt in range(3):
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            if attempt == 2:
                raise
            time.sleep(0.2)


def read_rounds(directory):
    path = directory / "rounds.jsonl"
    if not path.exists():
        return []
    lines = path.read_text().splitlines(keepends=True)
    return [json.loads(line) for line in lines if line.endswith("\n")]


def validate_reference(manifest):
    if (manifest["algorithm"], manifest["transport"], manifest["device"]) != (
        "coupled", "hybrid", "cuda"
    ):
        raise ValueError("Expected the existing coupled/hybrid CUDA reference.")
    if manifest["config"]["seed"] != 0:
        raise ValueError("This comparison suite is for seed 0.")
    for name, expected in manifest["source_sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Reference source changed: {name}. Do not mix implementations.")


def same_protocol(reference, candidate):
    return (
        reference["config"] == candidate["config"]
        and reference["source_sha256"] == candidate["source_sha256"]
        and candidate["device"] == "cuda"
    )


def summarize(base=BASE, suite=SUITE):
    suite.mkdir(parents=True, exist_ok=True)
    rows = []
    directories = [base] + [suite / f"{algorithm}_{transport}" for algorithm, transport in JOBS]
    fields = (
        "method", "status", "round", "validation_loss", "validation_accuracy",
        "validation_samples", "energy_bound_j", "expected_energy_j", "uplink_s",
        "cumulative_energy_bound_j", "cumulative_expected_energy_j",
        "newly_deleted", "fixed_forget_cohorts",
    )
    for directory in directories:
        manifest_path = directory / "manifest.json"
        if not manifest_path.exists():
            rows.append({"method": directory.name, "status": "not_started"})
            continue
        manifest = read_json(manifest_path)
        records = read_rounds(directory)
        if not records:
            rows.append({"method": directory.name, "status": manifest["status"]})
        for record in records:
            validation = record.get("validation", {})
            transmission = record.get("transmission", {})
            rows.append({
                "method": directory.name, "status": manifest["status"],
                "round": record["round"],
                "validation_loss": validation.get("loss"),
                "validation_accuracy": validation.get("accuracy"),
                "validation_samples": validation.get("samples"),
                "energy_bound_j": transmission.get("energy_bound_j"),
                "expected_energy_j": transmission.get("expected_energy_random_codewords_j"),
                "uplink_s": transmission.get("uplink_s"),
                "cumulative_energy_bound_j": record.get("cumulative_energy_bound_j"),
                "cumulative_expected_energy_j": record.get("cumulative_expected_energy_j"),
                "newly_deleted": record.get("newly_deleted"),
                "fixed_forget_cohorts": json.dumps(record.get("fixed_forget_cohorts", {})),
            })
    temporary = suite / "comparison.partial.csv"
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(suite / "comparison.csv")


def wait_for_reference(source=None, check_sources=True):
    source = BASE if source is None else source
    while True:
        manifest = read_json(source / "manifest.json")
        if check_sources:
            validate_reference(manifest)
        records = read_rounds(source)
        if manifest["status"] == "completed":
            if not records or records[-1]["round"] != manifest["config"]["rounds"]:
                raise RuntimeError("Reference completion is inconsistent with its round log.")
            return manifest
        if manifest["status"] == "failed" and records:
            print(f"Reference failed after round {records[-1]['round']}: "
                  f"{manifest.get('error')}. Keeping failure; running comparators.", flush=True)
            return manifest
        if manifest["status"] != "running":
            raise RuntimeError(f"Reference stopped: {manifest.get('error', manifest['status'])}")
        progress = source / "rounds.jsonl"
        if not progress.exists():
            progress = source / "manifest.json"
        if time.time() - progress.stat().st_mtime > 1800:
            raise RuntimeError("Reference log is stale for 30 minutes. Check the original terminal.")
        print(f"Waiting for current experiment: {records[-1]['round'] if records else 0}/"
              f"{manifest['config']['rounds']}", flush=True)
        time.sleep(30)


def run_suite():
    SUITE.mkdir(parents=True, exist_ok=True)
    with (SUITE / "queue.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        subprocess.run(
            [sys.executable, "-c", "import torch; assert torch.cuda.is_available(), "
             "'CUDA is not exposed in this execution environment; no CPU fallback was started.'; "
             "print(torch.ones(1, device='cuda')); torch.cuda.synchronize()"],
            check=True,
        )
        legacy_reference = wait_for_reference(LEGACY_BASE, check_sources=False)
        reference = dict(legacy_reference, source_sha256={
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in legacy_reference["source_sha256"]
        })
        validate_reference(reference)
        provenance = {
            "reason": "Convex feasibility initialization added after real round-91 false failure.",
            "legacy_reference": str(LEGACY_BASE),
            "legacy_status": legacy_reference["status"],
            "source_sha256": reference["source_sha256"],
            "config": reference["config"],
            "note": "All methods including proposed restart from round 1; no mixing solver versions.",
        }
        provenance_path = SUITE / "suite_metadata.json"
        if provenance_path.exists() and read_json(provenance_path) != provenance:
            raise ValueError("Suite provenance changed; use a separate suite directory.")
        provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
        config_path = SUITE / "config.json"
        if config_path.exists() and read_json(config_path) != reference["config"]:
            raise ValueError("Existing comparison configuration differs from reference.")
        config_path.write_text(json.dumps(reference["config"], indent=2) + "\n")
        environment = dict(os.environ, OPENBLAS_NUM_THREADS="4", OMP_NUM_THREADS="4",
                           MPLCONFIGDIR=str(ROOT / ".cache/matplotlib"), PYTHONUNBUFFERED="1")
        outcomes = {}
        for algorithm, transport in (("coupled", "hybrid"),) + JOBS:
            validate_reference(reference)
            name = f"{algorithm}_{transport}"
            destination = SUITE / name
            manifest_path = destination / "manifest.json"
            if destination.exists():
                if manifest_path.exists():
                    candidate = read_json(manifest_path)
                    records = read_rounds(destination)
                    if (same_protocol(reference, candidate)
                            and candidate["algorithm"] == algorithm
                            and candidate["transport"] == transport
                            and candidate["status"] == "completed"
                            and records and records[-1]["round"] == reference["config"]["rounds"]
                            and (destination / "schedule.json").read_bytes()
                            == (BASE / "schedule.json").read_bytes()):
                        outcomes[name] = "completed_existing"
                        continue
                outcomes[name] = "existing_incomplete_or_incompatible_not_overwritten"
                print(f"SKIP {name}: inspect existing output; no automatic resume.", flush=True)
                continue
            destination.mkdir()
            command = [
                sys.executable, str(ROOT / "run_clip_federated.py"), "--device", "cuda",
                "--config", str(config_path), "--algorithm", algorithm, "--transport", transport,
                "--output", str(destination),
            ]
            print(f"START {name}; log: {destination / 'console.log'}", flush=True)
            with (destination / "console.log").open("w") as handle:
                result = subprocess.run(command, cwd=ROOT, env=environment,
                                        stdout=handle, stderr=subprocess.STDOUT)
            valid = False
            if result.returncode == 0 and manifest_path.exists():
                candidate = read_json(manifest_path)
                records = read_rounds(destination)
                valid = (
                    same_protocol(reference, candidate) and candidate["status"] == "completed"
                    and bool(records) and records[-1]["round"] == reference["config"]["rounds"]
                    and (destination / "schedule.json").read_bytes()
                    == (BASE / "schedule.json").read_bytes()
                )
            outcomes[name] = "completed" if valid else f"failed_or_invalid_exit_{result.returncode}"
            (SUITE / "queue_status.json").write_text(json.dumps(outcomes, indent=2) + "\n")
            summarize()
            print(f"END {name}: {outcomes[name]}", flush=True)
        (SUITE / "queue_status.json").write_text(json.dumps(outcomes, indent=2) + "\n")
        summarize()
        return 0 if all(value.startswith("completed") for value in outcomes.values()) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-only", action="store_true")
    arguments = parser.parse_args()
    if arguments.summary_only:
        summarize()
    else:
        raise SystemExit(run_suite())
