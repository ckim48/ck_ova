"""A matched-objective ablation of the proposed FU-relative noise guard.

Use the existing joint component-noise budget instead. Do not tune a noise
coefficient, learning objective or dataset to force an intermediate curve.
"""
import argparse
import collections
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import baseline_fairness as previous
from baseline_fairness import ROOT, DATASETS, CASES, read, write, sha, load_base, link
from airscape import VideoData, extra

OUT = ROOT / "results/guard_ablation_v1_20261027"
SOURCE = previous.OUT
NEW = "unguarded_ova"
LABEL = "NG-OVA"
PROPOSED = "proposed_fu_guard_cliperase"
ORIGINAL_EXACT = None
REJECTIONS = 0


def state(dest=OUT, **kw):
    write(Path(dest) / "status.json", dict(unix=time.time(), pid=os.getpid(), **kw))


def strict_exact(*args, **kwargs):
    global REJECTIONS
    result = ORIGINAL_EXACT(*args, **kwargs)
    if result is None:
        return None
    config = args[6] if len(args) > 6 else kwargs["c"]
    peak = max(float((part.analog_rms ** 2).max()) for part in result)
    if peak > config["power_limit_w"] * (1 + 1e-6):
        REJECTIONS += 1
        write(OUT / "candidate_rejections" / f"{os.getpid()}_{REJECTIONS:05d}.json",
              dict(peak_w=peak, limit_w=config["power_limit_w"], reason="Existing receiver 1ppm bound"))
        return None
    return result


def pool_init(cap):
    global ORIGINAL_EXACT
    import radio_parallel_cap
    radio_parallel_cap.CAP = cap
    ORIGINAL_EXACT = radio_parallel_cap.exact_aligned
    radio_parallel_cap.exact_aligned = strict_exact


def build_radio(base):
    original = base.base.parent.common
    guard = base.base.parent.guard
    class UnguardedRadio:
        @staticmethod
        def analog(l, u, channels, config, seed):
            nl, nu, unprotected, protected = guard.budget(l, u, config)
            vector, meta = original.analog(l, u, channels, config, seed)
            meta.update(physical_mode="joint_unguarded", fu_noise_guard_active=False,
                FL_norm_bound=nl, FU_norm_bound=nu, original_epsilon=unprotected,
                reference_protected_epsilon=protected,
                ablation="Only FU-relative noise guard removed; existing joint component budget retained.")
            if len(np.flatnonzero(np.any(l+u, axis=1))) > 1:
                assert math.isclose(meta["epsilon"], unprotected, rel_tol=1e-10, abs_tol=1e-12)
            return vector, meta
    return UnguardedRadio


def register():
    path = OUT / "protocol.json"
    if path.exists():
        protocol = read(path)
        for group in ["source_sha256", "input_sha256"]:
            for name, digest in protocol[group].items():
                assert sha(ROOT / name) == digest, name
        return protocol
    parent = previous.register()
    assert not list((OUT / "cases").glob("*/*/"+NEW+"/outcome.json"))
    sources = dict(parent["source_sha256"])
    sources[str(Path(__file__).relative_to(ROOT))] = sha(Path(__file__))
    inputs = dict(parent["input_sha256"])
    for case in CASES:
        for dataset in DATASETS:
            path = SOURCE / "cases" / case["id"] / (dataset.lower()+"_seed0") / previous.NEW / "outcome.json"
            inputs[str(path.relative_to(ROOT))] = sha(path)
    protocol = dict(registered_unix=time.time(), datasets=DATASETS, cases=CASES, seed=0, new_arms=10,
        new_method=NEW, label=LABEL,
        definition="Same objective, joint FL/FU scheduler and optimized head/resource solver as Proposed; replace the FU-relative guard by the existing common component-noise budget.",
        formula="epsilon_NG=max(floor,relative_tolerance*hypot(FL_norm_bound,FU_norm_bound)); Proposed additionally caps this at max(floor,relative_tolerance*FU_norm_bound) when FU is active.",
        fixed="Same learned objective/weights, 200 FL rounds, 45x5 FU steps, data, warm/proxy/reference checkpoints, LR schedule, paired seeds, IoT geometry, channels, frame candidates, power limits and hard 0.1J network TX cap. No new coefficient or loss interpolation.",
        solver="Screen candidates at the existing final receiver 1ppm peak-power tolerance, as in the prior checked audit. Physical limits and the final assertion are unchanged; rejected numerical candidates are logged.",
        interpretation="A component ablation, not a separately published baseline. Removing a tighter error constraint may improve energy/delay and may leave mR unchanged. An intermediate rank is neither assumed nor an acceptance criterion.",
        outcome_policy="Report every registered outcome even if tied, better than Proposed or infeasible. Do not retune noise, select a dataset/seed, interpolate curves or replace a measured point to produce an intermediate rank.",
        source_sha256=sources, input_sha256=inputs)
    write(OUT / "protocol.json", protocol)
    link(SOURCE / "prepared/airscape_seed0", OUT / "prepared/airscape_seed0")
    return protocol


def run_case(case_id):
    register()
    import worker
    import heuristic
    _, base = load_base()
    extra.Data = VideoData
    heuristic.pool_init = pool_init
    worker.state = state
    run = base.repeat.clone_function(worker.run, [
        ("arm=make_arm(base,deployment,radios,cap);results=[]", "radios[NEW]=build_radio(base)\n    arm=make_arm(base,deployment,radios,cap);results=[]", 1),
        ("opt=base.base.ARMS['proposed_fu_guard_cliperase' if method=='heuristic_ova' else method]", "assert method==NEW\n                    opt=dict(base.base.ARMS[PROPOSED])", 1)],
        dict(OUT=OUT, EXTRA_DATASETS=["AirScape"], NEW=NEW, PROPOSED=PROPOSED,
             build_radio=build_radio, state=state))
    run(next(c for c in CASES if c["id"] == case_id), DATASETS, [NEW])
    directory = OUT / "cases" / case_id
    assert read(directory / "settings.json") == read(SOURCE / "cases" / case_id / "settings.json")
    for dataset in DATASETS:
        source = SOURCE / "cases" / case_id / (dataset.lower()+"_seed0")
        dest = directory / (dataset.lower()+"_seed0")
        for name in ["plan.json", "requests.json"]:
            assert read(dest / name) == read(source / name)
        for method in dict(previous.METHODS, **{previous.NEW: "S-OF"}):
            link(source / method, dest / method)
    print(dict(case=case_id, outcomes={d: read(directory / (d.lower()+"_seed0") / NEW / "outcome.json")["status"] for d in DATASETS}), flush=True)


def verify():
    rows = []
    for case in CASES:
        for dataset in DATASETS:
            parent = OUT / "cases" / case["id"] / (dataset.lower()+"_seed0")
            arm = parent / NEW
            result = read(arm / "outcome.json")
            assert result["status"] != "failed", result
            assert read(arm / "started.json")["policy"] == read(parent / PROPOSED / "started.json")["policy"]
            row = dict(case=case["id"], dataset=dataset, status=result["status"], wall_seconds=result["wall_seconds"])
            if result["status"] == "completed":
                verified = read(arm / "verification.json")
                assert verified["FL_rounds"] == 200 and verified["FU_steps"] == 225
                tx = [read(p) for p in sorted((arm / "transactions").glob("*.json"))]
                assert {r["kind"] for r in tx} == {"fl", "joint"}
                ratios = []
                for r in tx:
                    radio = r["radio"]
                    assert radio["physical_mode"] == "joint_unguarded" and not radio["fu_noise_guard_active"]
                    assert radio["energy_bound_j"] <= .1*(1+2e-6)
                    assert radio["original_epsilon"] >= radio["reference_protected_epsilon"]
                    if r["kind"] == "joint":
                        ratios.append(radio["epsilon"] / radio["reference_protected_epsilon"])
                        assert all("FM" in local and "RM" in local and "CM" in local for local in r["local"])
                row.update(transactions=len(tx), joint_transactions=len(ratios),
                    epsilon_ratio_min=min(ratios), epsilon_ratio_max=max(ratios))
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
    parser.add_argument("action", choices=["run", "case", "verify"])
    parser.add_argument("--case")
    args = parser.parse_args()
    if args.action == "case":
        run_case(args.case)
    elif args.action == "verify":
        print(verify())
    else:
        run_all()
