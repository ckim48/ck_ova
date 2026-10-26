"""One-shot serialized audit runner. No recurring monitor or training edits."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("radio", "quality"))
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--wait-pid", type=int)
    parser.add_argument("--with-q6-variant", action="store_true")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    if args.mode == "radio":
        command = [sys.executable, str(here / "paired_radio_audit.py"), "--project", str(args.project)]
        directory = args.project / "results/energy_quality_radio_v1"
    else:
        command = [sys.executable, str(here / "energy_quality.py"), "--project", str(args.project), "--device", "cuda"]
        directory = args.project / "results/energy_quality_v1"
    directory.mkdir(parents=True, exist_ok=True)
    status = {"status": "running", "mode": args.mode, "pid": os.getpid(), "command": command}
    (directory / "job_status.json").write_text(json.dumps(status, indent=2) + "\n")
    if args.mode == "quality" and args.wait_pid:
        from energy_quality import process_running
        status["status"] = "waiting_for_training_queue"
        status["wait_pid"] = args.wait_pid
        (directory / "job_status.json").write_text(json.dumps(status, indent=2) + "\n")
        print("WAITING_FOR_TRAINING_QUEUE", args.wait_pid, flush=True)
        while process_running(args.wait_pid):
            time.sleep(30)
    if args.mode == "quality" and args.with_q6_variant:
        from energy_quality import digest, validate_completed
        suite = args.project / "results/comparisons_seed0_solver_v2"
        baseline = json.loads((suite / "config.json").read_text())
        variant_config = dict(baseline, quantization_bits=6)
        config_path = directory / "config_q6.json"
        if config_path.exists() and json.loads(config_path.read_text()) != variant_config:
            raise ValueError("Precision-variant config differs from existing config.")
        config_path.write_text(json.dumps(variant_config, indent=2) + "\n")
        expected = json.loads((suite / "suite_metadata.json").read_text())["source_sha256"]
        if any(digest(args.project / name) != sha for name, sha in expected.items()):
            raise ValueError("Training sources changed before the queued precision experiment.")
        destination = suite / "coupled_hybrid_q6"
        if destination.exists():
            valid, reason = validate_completed(destination, suite, baseline, suite / "coupled_hybrid")
            if valid is None:
                print("PRECISION_VARIANT_EXISTING_NOT_OVERWRITTEN", reason, flush=True)
        else:
            status["status"] = "training_precision_variant"
            (directory / "job_status.json").write_text(json.dumps(status, indent=2) + "\n")
            print("START_PRECISION_VARIANT", str(destination), flush=True)
            destination.mkdir()
            with (destination / "console.log").open("x") as log:
                variant = subprocess.run([sys.executable, str(args.project / "run_clip_federated.py"),
                    "--device", "cuda", "--config", str(config_path), "--algorithm", "coupled",
                    "--transport", "hybrid", "--output", str(destination)], cwd=args.project,
                    stdout=log, stderr=subprocess.STDOUT)
            status["precision_variant_exit_code"] = variant.returncode
            print("END_PRECISION_VARIANT", variant.returncode, flush=True)
    status["status"] = "evaluating" if args.mode == "quality" else "running"
    (directory / "job_status.json").write_text(json.dumps(status, indent=2) + "\n")
    result = subprocess.run(command)
    status.update(status="completed" if result.returncode == 0 else "failed", exit_code=result.returncode)
    (directory / "job_status.json").write_text(json.dumps(status, indent=2) + "\n")
    if result.returncode == 0:
        plotted = subprocess.run([sys.executable, str(here / "plot_audits.py"), "--" + args.mode, str(directory)])
        status["plot_exit_code"] = plotted.returncode
        (directory / "job_status.json").write_text(json.dumps(status, indent=2) + "\n")
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
