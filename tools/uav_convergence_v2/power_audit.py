"""Versioned correction of inconsistent solver/receiver power tolerances.

Reject candidates above the existing receiver tolerance before head selection.
Do not relax physical limits, change seeds, or overwrite original failed runs.
"""
import argparse
import collections
import math
import os
import subprocess
import sys
import time
from pathlib import Path

from airscape import ROOT, VideoData, extra, NAME, read, write, sha
from common import cases, METHODS

ORIGINAL = ROOT / "results/airscape_sensitivity_v2_20261027"
OUT = ROOT / "results/airscape_power_audit_v1_20261027"
FIGURES = ROOT / "results/paper_pair_capera_airscape_checked_20261027"
CASE = "channel_gain_m6"
ORIGINAL_EXACT = None
REJECTIONS = 0


def strict_exact(*args, **kwargs):
    global REJECTIONS
    parts = ORIGINAL_EXACT(*args, **kwargs)
    if parts is None:
        return None
    config = args[6] if len(args) > 6 else kwargs["c"]
    peak = max(float((part.analog_rms ** 2).max()) for part in parts)
    if peak > config["power_limit_w"] * (1 + 1e-6):
        REJECTIONS += 1
        write(OUT / "candidate_rejections" / f"{os.getpid()}_{REJECTIONS:05d}.json",
              dict(head=int(args[0]), peak_w=peak, power_limit_w=config["power_limit_w"],
                   relative_excess=peak / config["power_limit_w"] - 1,
                   reason="Passed old2ppm candidate tolerance but exceeds unchanged1ppm receiver tolerance."))
        return None
    return parts


def pool_init(cap):
    global ORIGINAL_EXACT
    import radio_parallel_cap as radio
    radio.CAP = cap
    ORIGINAL_EXACT = radio.exact_aligned
    radio.exact_aligned = strict_exact


def state(dest=OUT, **kw):
    write(Path(dest) / "status.json", dict(pid=os.getpid(), unix=time.time(), **kw))


def link(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.symlink_to(os.path.relpath(source, target.parent), target_is_directory=source.is_dir())


def run():
    import worker
    import heuristic
    assert not (OUT / "protocol.json").exists(), "Preserve all attempts"
    failed = ORIGINAL / "cases" / CASE / "airscape_seed0/proposed_fu_guard_cliperase"
    assert read(failed / "outcome.json")["status"] == "failed"
    write(OUT / "original_failure.json", read(failed / "failure.json"))
    write(OUT / "protocol.json", dict(registered_unix=time.time(), case=CASE, seed=0,
        correction="Candidate power tolerance tightened from2ppm to the existing receiver1ppm. "
                   "The final receiver assertion, energy/error budgets and physical power limit are unchanged.",
        scope="Rerun all five methods at the single failed scenario from the identical shared checkpoints. "
              "Reuse all other scenarios. Original failed run and figures remain immutable.",
        original_failure_sha256=sha(failed / "failure.json"),
        original_outcomes={m: sha(failed.parent / m / "outcome.json") for m in METHODS},
        source_sha256=sha(Path(__file__)),
        radio_source_sha256=sha(ROOT / "tools/overnight_energy_v2/fourway_cap100_v1/radio_parallel_cap.py")))
    link(ORIGINAL / "prepared", OUT / "prepared")
    for case in cases():
        if case["id"] != CASE:
            link(ORIGINAL / "cases" / case["id"], OUT / "cases" / case["id"])
    link(ORIGINAL / "cases" / CASE / "capera_seed0", OUT / "cases" / CASE / "capera_seed0")
    extra.Data = VideoData
    heuristic.pool_init = pool_init
    worker.OUT, worker.EXTRA_DATASETS, worker.state = OUT, [NAME], state
    worker.run(next(case for case in cases() if case["id"] == CASE), [NAME])
    validation = verify()
    write(OUT / "verification.json", validation)
    report = render()
    write(OUT / "finished.json", dict(status="verified", unix=time.time(), verification=validation, report=report))
    state(stage="terminal", report=report)
    print({"validation": validation, "report": report}, flush=True)


def verify():
    old = ORIGINAL / "cases" / CASE / "airscape_seed0"
    new = OUT / "cases" / CASE / "airscape_seed0"
    stable, outcomes = [], {}
    for method in METHODS:
        prior, current = read(old / method / "outcome.json"), read(new / method / "outcome.json")
        assert current["status"] != "failed", current
        outcomes[method] = current["status"]
        if prior["status"] != "failed":
            assert prior["status"] == current["status"], method
            if prior["status"] == "completed":
                for key, value in prior["metrics"].items():
                    actual = current["metrics"][key]
                    if isinstance(value, (int, float)):
                        assert math.isclose(value, actual, rel_tol=1e-9, abs_tol=1e-8), (method, key, value, actual)
                    elif isinstance(value, list):
                        assert len(value) == len(actual)
                        assert all(math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-8) for a, b in zip(value, actual))
                    else:
                        assert value == actual, (method, key)
            stable.append(method)
        limit = read(new / "plan.json")["config"]["power_limit_w"]
        for path in (new / method / "transactions").glob("*.json"):
            assert read(path)["radio"]["max_power_w"] <= limit * (1 + 1e-6)
    # Existing successful trajectories must also pass the unchanged strict bound.
    checked = 0
    for directory in (ORIGINAL / "cases").glob("*/airscape_seed0"):
        limit = read(directory / "plan.json")["config"]["power_limit_w"]
        for path in directory.glob("*/transactions/*.json"):
            assert read(path)["radio"]["max_power_w"] <= limit * (1 + 1e-6)
            checked += 1
    rejections = [read(p) for p in (OUT / "candidate_rejections").glob("*.json")]
    assert rejections and all(1e-6 < r["relative_excess"] <= 2.000001e-6 for r in rejections)
    return dict(passed=True, outcomes=outcomes, unchanged_other_methods=stable,
                existing_transactions_checked=checked, rejected_numerical_candidates=len(rejections),
                rejection_excess_ppm=[min(r["relative_excess"] for r in rejections)*1e6,
                                      max(r["relative_excess"] for r in rejections)*1e6])


def render():
    import paired_report
    paired_report.OUT, paired_report.DEST = OUT, FIGURES
    paired_report.style.DEST = FIGURES
    report = paired_report.generate()
    assert report["terminal_arms"] == 410 and report["figure_count"] == 122
    assert report["status_counts"].get("failed", 0) == 0
    note = ("\n## Numerical power-check correction\n\n"
            "This checked version replaces the five AirScape arms at the -6dB channel-gain "
            "scenario with a deterministic rerun. Candidate screening now uses the same1ppm "
            "power tolerance as the unchanged final receiver check, instead of2ppm. "
            "Physical limits were not relaxed. All four previously nonfailed arms retain "
            "their original status and metrics. The original failure and all original "
            "results are preserved in `airscape_sensitivity_v2_20261027`. "
            "See `../airscape_power_audit_v1_20261027/verification.json`.\n")
    with (FIGURES / "README.md").open("a") as handle:
        handle.write(note)
    return report


def publish():
    git = ["git", "--git-dir=.git-push-ready/code-charts.git", "--work-tree=."]
    def call(args):
        return subprocess.check_output(git + args, cwd=ROOT, text=True).strip()
    assert call(["remote", "get-url", "origin"]) == "https://github.com/ckim48/ck_ova.git"
    assert not call(["diff", "--cached", "--name-only"]), "Preserve existing staged changes"
    paths = [Path(__file__)] + list(OUT.glob("*.json"))
    paths += list((OUT / "candidate_rejections").glob("*.json"))
    paths += list((OUT / "cases" / CASE / "airscape_seed0").glob("*/outcome.json"))
    paths += list(FIGURES.glob("*.*")) + list((FIGURES / "figures").glob("*.*"))
    paths = [str(p.relative_to(ROOT)) for p in paths if p.is_file()]
    subprocess.run(git + ["add", "--"] + paths, cwd=ROOT, check=True, capture_output=True)
    subprocess.run(git + ["commit", "-m", "Align radio candidate power tolerance and verify AirScape results"], cwd=ROOT, check=True, capture_output=True)
    subprocess.run(git + ["push", "origin", "main"], cwd=ROOT, check=True, capture_output=True, timeout=180)
    commit = call(["rev-parse", "HEAD"])
    assert call(["ls-remote", "origin", "refs/heads/main"]).split()[0] == commit
    write(OUT / "publication.json", dict(status="pushed", commit=commit))
    print("Pushed", commit, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["run", "publish"])
    args = parser.parse_args()
    run() if args.action == "run" else publish()
