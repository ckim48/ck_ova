"""Add a plain serial OFDMA control; preserve all enhanced ablations.

The new arm changes only S-OF+'s FU objective to the existing safe-proxy
objective used by S-AC and AC-OF. No new hyperparameters or result filtering.
"""
import argparse
import collections
import os
from pathlib import Path
import subprocess
import sys
import time

from airscape import ROOT, VideoData, extra, read, write, sha
from common import cases, METHODS, load_base

OUT = ROOT / "results/baseline_fairness_v1_20261027"
CHECKED = ROOT / "results/airscape_power_audit_v1_20261027"
NEW = "sequential_digital_proxy"
OLD = "sequential_digital_periodic"
DATASETS = ["CapERA", "AirScape"]
CASES = [c for c in cases() if c["id"] == "baseline" or c["axis"] == "bandwidth"]
MAIN = {m: METHODS[m] for m in ["proposed_fu_guard_cliperase", "separate_aircomp_periodic", "aircomp_ofdma_periodic"]}
MAIN[NEW] = "S-OF"


def state(dest=OUT, **kw):
    write(Path(dest) / "status.json", dict(unix=time.time(), pid=os.getpid(), **kw))


def link(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        assert target.resolve() == source.resolve(), target
    else:
        target.symlink_to(os.path.relpath(source, target.parent), target_is_directory=source.is_dir())


def register():
    path = OUT / "protocol.json"
    if path.exists():
        protocol = read(path)
        for p, digest in protocol["source_sha256"].items():
            assert sha(ROOT / p) == digest, p
        for p, digest in protocol["input_sha256"].items():
            assert sha(ROOT / p) == digest, p
        return protocol
    assert not list((OUT / "cases").glob("*/*/" + NEW + "/outcome.json"))
    sources = [Path(__file__)]
    for directory in ["iot_sensitivity_v1", "iot_event_mission_v1", "iot_vlm_energy_overrun_v1",
                      "uav_real_vlm_v1", "overnight_energy_v2"]:
        sources.extend((ROOT / "tools" / directory).rglob("*.py"))
    sources.append(ROOT / "tools/uav_convergence_v2/airscape.py")
    inputs = []
    for dataset in DATASETS:
        directory = CHECKED / "cases/baseline" / (dataset.lower() + "_seed0")
        for p in ["plan.json", "requests.json", "initial20/final.pt", "proxy20/final.pt",
                  "reference200/final.pt", "feature_cache/features.pt", "feature_cache/manifest.json"]:
            inputs.append(directory / p)
    for case in CASES:
        inputs.append(CHECKED / "cases" / case["id"] / "settings.json")
        for dataset in DATASETS:
            for method in METHODS:
                inputs.append(CHECKED / "cases" / case["id"] / (dataset.lower() + "_seed0") / method / "outcome.json")
    protocol = dict(registered_unix=time.time(), seed=0, datasets=DATASETS, cases=CASES,
        new_method=NEW, new_label="S-OF", new_arms=10, main_methods=MAIN,
        change="S-OF+ serial digital OFDMA transport with the existing original_safe_proxy FU objective of S-AC/AC-OF.",
        fixed="Model, features, native captions, split, warm20/proxy20/reference200 checkpoints, FL/FU learning rates, requests, 200 FL rounds, 45 requests x 5 FU steps, seeds, IoT geometry, optimized digital head/tones, frame candidates, physical limits and energy model.",
        digital_cap="Inherit S-OF+ policy: prefer original 0.1J cap-feasible solution; permit and report network-cap overruns under unchanged per-user power/energy, error and deadline constraints. Analog controls retain hard cap.",
        ablations={"heuristic_ova": "Same CLIPErase-inspired FU objective; centroid head and uniform allocation ablate head/resource optimization.",
                   OLD: "Same CLIPErase-inspired FU objective; serial digital OFDMA isolates transport/scheduling from the loss change."},
        interpretation="Main comparison changes the full system, not only communication. Matched-objective ablations remain required for a causal transport/head claim. Retained utility alone does not certify forgetting.",
        objective_audit="CLIPErase-inspired FM/RM/CM is adapted from published work, not an original loss. Its crossmodal argument is unused; flipping that flag would not remove CM. This version selects the already implemented safe-proxy vector function instead.",
        outcome_policy="Report all ten predeclared outcomes, including infeasibility/failure. No ranking-based tuning, sample filtering or renaming of old results as new measurements.",
        source_sha256={str(p.relative_to(ROOT)): sha(p) for p in sorted(set(sources))},
        input_sha256={str(p.relative_to(ROOT)): sha(p) for p in inputs})
    write(path, protocol)
    link(ROOT / "results/airscape_sensitivity_v2_20261027/prepared/airscape_seed0", OUT / "prepared/airscape_seed0")
    return protocol


def make_arm(base, deployment, radios, cap):
    import event_arm
    edits = [("radio=base.parent.guard if name=='proposed_fu_guard_cliperase' else base.SequentialRadio if name=='sequential_digital_periodic' else base.parent.common", "radio=Radios[name]", 1),
             ("if name!='sequential_digital_periodic':assert rm['energy_bound_j']<=.1*(1+2e-6)",
              "if name not in ['sequential_digital_periodic','sequential_digital_proxy']:assert rm['energy_bound_j']<=NETWORK_CAP*(1+2e-6)", 1)]
    maker = base.repeat.clone_function(event_arm.make_arm, [
        ("return base.repeat.clone_function", "edits.extend(EXTRA_EDITS)\n    return base.repeat.clone_function", 1),
        ("deployment=deployment,state=state))", "deployment=deployment,state=state,Radios=RADIOS,NETWORK_CAP=CAP))", 1)],
        dict(EXTRA_EDITS=edits, RADIOS=radios, CAP=cap))
    return maker(base, deployment, state)


def options(base):
    old = dict(base.base.ARMS[OLD])
    new = dict(old, objective="original_safe_proxy")
    assert {k for k in old if old[k] != new[k]} == {"objective"}
    assert new == base.base.ARMS["separate_aircomp_periodic"]
    assert not new["parallel"] and not new["digital"] and new["crossmodal"]
    return new


def run_case(case_id):
    register()
    import worker
    event, base = load_base()
    import event_energy
    original_energy = event_energy.method_energy
    def energy(*args, **kwargs):
        result = original_energy(*args, **kwargs)
        if result["method"] == NEW:
            result["network_cap_policy"] = "soft, original-feasible preferred"
        return result
    event_energy.method_energy = energy
    extra.Data = VideoData
    validate = base.repeat.clone_function(worker.validate,
        [("if ad.name!='sequential_digital_periodic':", "if ad.name not in ['sequential_digital_periodic','sequential_digital_proxy']:", 1)])
    run = base.repeat.clone_function(worker.run, [
        ("arm=make_arm(base,deployment,radios,cap);results=[]", "radios[NEW]=overrun_radio.build(base)\n    arm=make_arm(base,deployment,radios,cap);results=[]", 1),
        ("opt=base.base.ARMS['proposed_fu_guard_cliperase' if method=='heuristic_ova' else method]", "assert method==NEW\n                    opt=options(base)", 1)],
        dict(OUT=OUT, EXTRA_DATASETS=["AirScape"], state=state, make_arm=make_arm,
             validate=validate, NEW=NEW, options=options))
    assert options(base)["objective"] == "original_safe_proxy"
    run(next(c for c in CASES if c["id"] == case_id), DATASETS, [NEW])
    directory = OUT / "cases" / case_id
    assert read(directory / "settings.json") == read(CHECKED / "cases" / case_id / "settings.json")
    for dataset in DATASETS:
        source = CHECKED / "cases" / case_id / (dataset.lower() + "_seed0")
        target = directory / (dataset.lower() + "_seed0")
        for p in ["plan.json", "requests.json"]:
            assert read(source / p) == read(target / p), (dataset, p)
        for method in METHODS:
            link(source / method, target / method)
    print({"case": case_id, "outcomes": {d: read(directory / (d.lower()+"_seed0") / NEW / "outcome.json")["status"] for d in DATASETS}}, flush=True)


def verify():
    rows = []
    for case in CASES:
        for dataset in DATASETS:
            directory = OUT / "cases" / case["id"] / (dataset.lower() + "_seed0") / NEW
            outcome = read(directory / "outcome.json")
            assert outcome["status"] != "failed", outcome
            policy = read(directory / "started.json")["policy"]
            assert policy["objective"] == "original_safe_proxy" and not policy["parallel"]
            row = dict(case=case["id"], dataset=dataset, status=outcome["status"], wall_seconds=outcome["wall_seconds"])
            if outcome["status"] == "completed":
                checked = read(directory / "verification.json")
                assert checked["FL_rounds"] == 200 and checked["FU_requests"] == 45 and checked["FU_steps"] == 225
                transactions = [read(p) for p in (directory / "transactions").glob("*.json")]
                assert {r["kind"] for r in transactions} == {"fl", "fu"}
                for r in transactions:
                    assert r["radio"]["physical_mode"] == "digital_sequential_ofdma"
                    assert not r["radio"]["FL_FU_temporally_overlapped"]
                    for local in r["local"]:
                        assert "forget_loss" in local and "preservation_loss" in local
                assert outcome["metrics"]["network_cap_policy"] == "soft, original-feasible preferred"
                row.update(transactions=len(transactions), cap_overruns=outcome["metrics"]["network_cap_overrun_transactions"])
            rows.append(row)
    report = dict(passed=True, rows=rows, outcomes=dict(collections.Counter(r["status"] for r in rows)))
    write(OUT / "verification.json", report)
    return report


def run_all():
    register()
    for case in CASES:
        subprocess.run([sys.executable, str(Path(__file__)), "case", "--case", case["id"]], cwd=ROOT, check=True)
    report = verify()
    write(OUT / "finished.json", dict(unix=time.time(), **report))
    state(stage="completed", outcomes=report["outcomes"])
    print(report, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["register", "run", "case", "verify"])
    parser.add_argument("--case")
    args = parser.parse_args()
    if args.action == "register":
        print({k: v for k, v in register().items() if k not in ["source_sha256", "input_sha256"]})
    elif args.action == "case":
        run_case(args.case)
    elif args.action == "run":
        run_all()
    else:
        print(verify())
