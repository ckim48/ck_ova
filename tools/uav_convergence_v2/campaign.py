"""Isolated AirScape five-method campaign; preserves all existing experiments."""
import argparse
import collections
import fcntl
import os
import shutil
import subprocess
import sys
import time
import traceback

from airscape import ROOT, OUT as PILOTS, DATA, NAME, VideoData, extra, read, write, sha
from common import cases, METHODS, AXES, load_base, GEOMETRY

HERE = ROOT / "tools/uav_convergence_v2"
OUT = ROOT / "results/airscape_sensitivity_v2_20261027"
PRIOR = ROOT / "results/iot_sensitivity_v1_20261027"


def state(dest=OUT, **kw):
    write(dest / "status.json", dict(pid=os.getpid(), unix=time.time(), **kw))


def prepare():
    import torch
    import numpy as np
    dest = OUT / "prepared/airscape_seed0"
    if (dest / "ready.json").exists():
        return read(dest / "ready.json")
    src = PILOTS / "airscape_pilot"
    assert read(src / "finished.json")["status"] == "completed"
    assert not (dest / "started.json").exists(), "Preserve incomplete preparation"
    event, base = load_base()
    from accounting import instrumented_engine
    gpu = (ROOT / "results/overnight_energy_v2/association_objective_gpu.lock").open("a")
    fcntl.flock(gpu, fcntl.LOCK_EX)
    assert torch.cuda.is_available()
    torch.set_num_threads(1)
    for name in ["plan", "requests", "initial_records", "proxy_records", "reference_records"]:
        write(dest / (name + ".json"), read(src / (name + ".json")))
    cache = dest / "feature_cache"
    if not cache.exists():
        cache.symlink_to(os.path.relpath(src / "feature_cache", dest), target_is_directory=True)
    plan = read(dest / "plan.json")
    manifest = read(cache / "manifest.json")
    assert manifest["cache_sha256"] == sha(cache / "features.pt")
    assert manifest["subset_manifest_sha256"] == sha(DATA / "selection.json")
    write(dest / "started.json", dict(unix=time.time(), pid=os.getpid()))
    try:
        data = VideoData(NAME)
        payload = torch.load(cache / "features.pt", map_location="cpu", weights_only=False)
        engine = instrumented_engine(extra.native_engine(base, 1))(data, plan, payload)
        ids = read(dest / "initial_records.json")[0]["batches"][0]
        engine.start_work()
        gradient, loss = engine.gradient(engine.theta0, ids, 1)
        work = engine.finish_work()
        _, next_loss = engine.gradient(engine.theta0 - 1e-5 * gradient, ids, 1)
        assert torch.isfinite(gradient).all() and gradient[:-1].numel() == 20480
        assert float(gradient[:-1].norm()) > 0 and next_loss <= loss + 1e-6
        write(dest / "preflight.json", dict(passed=True, native_captions=1,
            gradient_norm=float(gradient[:-1].norm()), initial_loss=loss,
            small_step_loss=next_loss, compute=work))
        pure = base.repeat.clone_function(base.pure_trajectory, [], {"state": state})
        channel = np.load(GEOMETRY / "channels_by_site.npy")[0]
        warm, warm_cost = pure(engine, plan, read(dest / "initial_records.json"), channel, "initial20", dest)
        proxy, proxy_cost = pure(engine, plan, read(dest / "proxy_records.json"), channel, "proxy20", dest)
        write(dest / "common_prefix_cost.json", {k: warm_cost[k] + proxy_cost[k] for k in warm_cost})
        engine.set_teacher(warm)
        reference, _ = pure(engine, plan, read(dest / "reference_records.json"), channel, "reference200", dest)
        metrics, _ = engine.evaluate(reference, None)
        mia, _ = engine.mia_scores(reference)
        write(dest / "reference_metrics.json", dict(metrics=metrics,
            retained_mR_threshold=.95 * metrics["retained"]["mR"], reporting_only=True))
        write(dest / "reference_mia.json", mia)
        result = dict(status="ready", dataset=NAME, unix=time.time(), native_captions=1,
            checkpoint_sha256={stage: sha(dest / stage / "final.pt")
                               for stage in ["initial20", "proxy20", "reference200"]},
            feature_reuse="Only frozen source embeddings from pilot; all adapters initialized fresh.")
        write(dest / "ready.json", result)
        return result
    finally:
        base.base.parent.common.base.close()


def run_case(case):
    import worker
    extra.Data = VideoData
    worker.OUT = OUT
    worker.EXTRA_DATASETS = [NAME]
    worker.state = state
    worker.run(case, [NAME])


def register():
    result = read(PILOTS / "airscape_pilot/finished.json")
    control = read(PILOTS / "capera_pilot/finished.json")
    assert result["status"] == control["status"] == "completed"
    # This is a prospective exploration rule, recorded before AirScape learning.
    gate = read(PILOTS / "airscape_full_comparison_gate.json")
    gain = result["gain_percentage_points"]
    late_gain = 100 * (result["last_five_mean_mR"] - result["initial_mR"])
    gate_passed = gain >= gate["minimum_final_gain_pp"] and late_gain >= gate["minimum_late_gain_pp"]
    # Keep the failed strong-gain screen visible. The user's broader request is
    # to examine a real UAV candidate with rising/settling mR. A follow-up may
    # study a modest gain, but cannot claim that the prospective gate passed.
    decision = read(PILOTS / "airscape_continuation_decision.json")
    assert decision["strong_gain_gate_passed"] == gate_passed
    assert gate_passed or decision["scope"] == "exploratory_followup_after_failed_strong_gain_gate"
    path = OUT / "protocol.json"
    if path.exists():
        return read(path)
    protocol = dict(seed=0, dataset=NAME, methods=METHODS, cases=cases(), axes=AXES,
        registered_unix=time.time(), total_new_arms=205,
        paired_dataset="CapERA; reuse all205 already completed identical-condition outcomes",
        candidate_selection="Exploratory dataset screening on validation, not independent confirmation. "
            "All pilot candidates and failures retained; official test excluded.",
        scope=read(DATA / "selection.json")["scope"],
        pilot=result, matched_control=control, continuation_gate=gate,
        strong_gain_gate_passed=gate_passed, continuation_decision=decision,
        frozen_sources={str(p.relative_to(ROOT)): sha(p) for p in sorted(
            (ROOT / "tools/iot_sensitivity_v1").glob("*.py"))},
        sources={str(p.relative_to(ROOT)): sha(p) for p in sorted(HERE.glob("*.py"))},
        selection_sha256=sha(DATA / "selection.json"),
        input_hashes={str((PILOTS / "airscape_pilot" / name).relative_to(ROOT)):
                      sha(PILOTS / "airscape_pilot" / name)
                      for name in ["plan.json", "requests.json", "feature_cache/features.pt"]},
        comparison="Same10UAVs/5sites,200FL,45FUrequests, paired seed0, radio constraints, "
            "energy accounting and separate completion clocks as the completed campaign. "
            "Native1caption,8frame pooling. All5methods receive same dataset and checkpoints. "
            "No successful retries, no missing-value interpolation; preserve infeasible points.")
    write(path, protocol)
    return protocol


def run_all():
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / "campaign.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    protocol = register()
    for name, expected in {**protocol["frozen_sources"], **protocol["sources"], **protocol["input_hashes"]}.items():
        assert sha(ROOT / name) == expected, name
    write(OUT / "launch.json", dict(pid=os.getpid(), unix=time.time()))
    state(stage="preparation")
    prepare()
    from paired_report import generate
    runs = []
    for position, case in enumerate(cases()):
        state(stage="scenario", case=case["id"])
        parent = OUT / "cases" / case["id"]
        parent.mkdir(parents=True, exist_ok=True)
        target = parent / "capera_seed0"
        if not target.exists():
            target.symlink_to(os.path.relpath(PRIOR / "cases" / case["id"] / "capera_seed0", parent),
                             target_is_directory=True)
        log = OUT / "logs" / (case["id"] + ".log")
        log.parent.mkdir(exist_ok=True)
        with log.open("a") as handle:
            completed = subprocess.run([sys.executable, str(HERE / "campaign.py"), "case", "--case", case["id"]],
                cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
        runs.append(dict(case=case["id"], returncode=completed.returncode))
        write(OUT / "runs.json", runs)
        if completed.returncode:
            raise RuntimeError("Scenario worker failed: " + case["id"])
        if position % 4 == 0:
            report = generate()
            publication = publish()
            write(OUT / "publication.json", publication)
        state(stage="scenario_terminal", case=case["id"], report=report)
    report = generate()
    assert report["terminal_arms"] == 410
    failures = report["status_counts"].get("failed", 0)
    write(OUT / "finished.json", dict(status="finished_with_failed_arms" if failures else "finished",
                                      unix=time.time(), report=report))
    state(stage="terminal", report=report)
    write(OUT / "publication.json", publish())
    notify("UAV 비교 실험 종료" if failures else "UAV 비교 실험 완료",
           f"CapERA/AirScape: {report['terminal_arms']}/410 조건 종료, "
           f"실행 오류 {failures}건. 1×2 그래프와 결과를 저장했습니다.")


def notify(title, message):
    if shutil.which("notify-send"):
        try:
            result = subprocess.run(["notify-send", title, message], capture_output=True, timeout=10)
            write(OUT / "desktop_notification.json", dict(returncode=result.returncode, unix=time.time()))
        except Exception as error:
            write(OUT / "desktop_notification.json", dict(error=repr(error), unix=time.time()))


def publish():
    """Only task-owned code, pilot evidence, flattened results and figures."""
    from paired_report import DEST
    git = ["git", "--git-dir=.git-push-ready/code-charts.git", "--work-tree=."]
    def call(args):
        return subprocess.check_output(git + args, cwd=ROOT, text=True).strip()
    try:
        assert call(["remote", "get-url", "origin"]) == "https://github.com/ckim48/ck_ova.git"
        if call(["diff", "--cached", "--name-only"]):
            return dict(status="deferred_existing_staged_changes")
        paths = list(HERE.glob("*.py")) + list(PILOTS.glob("*.json")) + list(PILOTS.glob("*.md"))
        paths += list(PILOTS.glob("*.csv"))
        for folder in PILOTS.glob("*_pilot"):
            paths += [folder / name for name in ["protocol.json", "rounds.json", "finished.json"]]
        paths += list((PILOTS / "figures").glob("*"))
        paths += [OUT / name for name in ["protocol.json", "runs.json", "finished.json"]]
        paths += list(DEST.glob("*.json")) + list(DEST.glob("*.csv")) + list(DEST.glob("*.md"))
        paths += list((DEST / "figures").glob("*"))
        paths = [str(p.relative_to(ROOT)) for p in paths if p.is_file()]
        subprocess.run(git + ["add", "--"] + paths, cwd=ROOT, check=True, capture_output=True)
        if not call(["diff", "--cached", "--name-only"]):
            return dict(status="unchanged")
    except Exception as error:
        return dict(status="failed", error=repr(error))
    try:
        subprocess.run(git + ["commit", "-m", "Add UAV convergence pilots and AirScape comparison results"],
            cwd=ROOT, check=True, capture_output=True)
        subprocess.run(git + ["push", "origin", "main"], cwd=ROOT, check=True, capture_output=True, timeout=180)
        commit = call(["rev-parse", "HEAD"])
        assert call(["ls-remote", "origin", "refs/heads/main"]).split()[0] == commit
        return dict(status="pushed", commit=commit)
    except Exception as error:
        return dict(status="failed", error=repr(error))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "case", "all"])
    parser.add_argument("--case", default="baseline")
    args = parser.parse_args()
    try:
        if args.action == "prepare":
            prepare()
        elif args.action == "case":
            run_case(next(c for c in cases() if c["id"] == args.case))
        else:
            run_all()
    except Exception as error:
        write(OUT / ("failure_" + args.action + ".json"), dict(error=repr(error), traceback=traceback.format_exc()))
        if args.action == "all":
            notify("UAV 비교 실험 중단", "실험 실행 오류가 발생했습니다. failure_all.json에 원인을 저장했습니다.")
        raise
