"""Complete the existing release-group sweep for the two newer learning arms."""
import argparse
import collections
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import baseline_fairness as serial
import learning_ablation as partial
from learning_ablation import ROOT, DATASETS, read, write, sha, load_base, link
from common import cases
from airscape import extra, VideoData

OUT = ROOT / "results/final_request_sweep_v1_20261027"
CHECKED = serial.CHECKED
BASELINE = partial.OUT
NEW = [partial.NEW, serial.NEW]
CASES = [c for c in cases() if c["axis"] == "request_bursts"]
ORIGINAL_EXACT = None
REJECTIONS = 0


def state(dest=OUT, **kw):
    write(Path(dest)/"status.json", dict(unix=time.time(), pid=os.getpid(), **kw))


def strict_exact(*args, **kwargs):
    global REJECTIONS
    parts = ORIGINAL_EXACT(*args, **kwargs)
    if parts is None:
        return None
    config = args[6] if len(args)>6 else kwargs["c"]
    peak = max(float((part.analog_rms**2).max()) for part in parts)
    if peak > config["power_limit_w"]*(1+1e-6):
        REJECTIONS += 1
        write(OUT/"candidate_rejections"/f"{os.getpid()}_{REJECTIONS:05d}.json",
              dict(peak_w=peak,limit_w=config["power_limit_w"],reason="Existing receiver 1ppm bound"))
        return None
    return parts


def pool_init(cap):
    global ORIGINAL_EXACT
    import radio_parallel_cap
    radio_parallel_cap.CAP = cap
    ORIGINAL_EXACT = radio_parallel_cap.exact_aligned
    radio_parallel_cap.exact_aligned = strict_exact


def make_arm(base, deployment, radios, cap):
    control = SimpleNamespace(**base.control.__dict__)
    control.synchronized_vectors = lambda objective: partial.partial_vectors(base, objective)
    # The partial FU helper is used only by the CLIPErase branch; proxy is unchanged.
    adapted = SimpleNamespace(**dict(base.__dict__, control=control))
    maker = base.repeat.clone_function(serial.make_arm, [], dict(state=state))
    return maker(adapted, deployment, radios, cap)


def options(base, method):
    assert method in NEW
    return partial.options(base) if method==partial.NEW else serial.options(base)


def register():
    path = OUT/"protocol.json"
    if path.exists():
        result = read(path)
        for group in ["source_sha256", "input_sha256"]:
            for name,digest in result[group].items():assert sha(ROOT/name)==digest,name
        return result
    parent = partial.register()
    assert not list((OUT/"cases").glob("*/*/*/outcome.json"))
    sources = dict(parent["source_sha256"])
    sources[str(Path(__file__).relative_to(ROOT))] = sha(Path(__file__))
    inputs = dict(parent["input_sha256"])
    for case in CASES:
        directory = CHECKED/"cases"/case["id"]
        paths = [directory/"settings.json"]
        for dataset in DATASETS:
            dd = directory/(dataset.lower()+"_seed0")
            paths += [dd/"plan.json", dd/"requests.json"]
            paths += [dd/m/"outcome.json" for m in serial.METHODS]
        for p in paths:inputs[str(p.relative_to(ROOT))]=sha(p)
    result = dict(registered_unix=time.time(),datasets=DATASETS,cases=CASES,new_methods=NEW,new_arms=16,seed=0,
        definition="Complete the already defined release-group sweep for FM-CM and S-OF; preserve all original outcomes.",
        workload="45 fixed requests, five FU steps each, 200 total FL rounds. Existing release groups 3/5/9/15/45 correspond to 15/9/5/3/1 requests per group. Only the release schedule changes; request IDs, data, original FU sampling seeds and workloads stay fixed. Release groups are indexed by FL progress, not exogenous requests/second.",
        fixed="Reuse the exact prior FM-CM and S-OF objectives/policies, frozen features, checkpoints, LR, geometry, radio limits and energy model. Baseline nine-group runs reused unchanged.",
        cap_policy="S-OF retains its previously authorized soft network TX cap with overrun reporting. FM-CM retains Proposed's hard cap and FU guard. Other physical limits remain hard.",
        outcome_policy="Retain every new outcome, including infeasible conditions; do not tune for rank or fill missing metrics.",
        final_quality_reporting="Reuse the existing criterion: all 45 requests completed, retained mR >= 0.95 times retained-set retrain, mean forget KL <= 0.01 and worst-request KL <= 0.01; confirmation at three distinct consecutive evaluated FL rounds. Do not tune thresholds to these outcomes.",
        source_sha256=sources,input_sha256=inputs)
    write(path,result)
    link(BASELINE/"prepared/airscape_seed0",OUT/"prepared/airscape_seed0")
    link(BASELINE/"cases/baseline",OUT/"cases/baseline")
    return result


def run_case(case_id):
    register()
    import worker, heuristic
    _,base = load_base()
    import event_energy
    extra.Data=VideoData;heuristic.pool_init=pool_init
    original_energy=event_energy.method_energy
    def energy(*args,**kwargs):
        result=original_energy(*args,**kwargs)
        if result["method"]==serial.NEW:result["network_cap_policy"]="soft, original-feasible preferred"
        return result
    event_energy.method_energy=energy
    validate=base.repeat.clone_function(worker.validate,
        [("if ad.name!='sequential_digital_periodic':","if ad.name not in ['sequential_digital_periodic','sequential_digital_proxy']:",1)])
    run=base.repeat.clone_function(worker.run,[
        ("arm=make_arm(base,deployment,radios,cap);results=[]",
         "radios[PARTIAL]=base.base.parent.guard\n    radios[SERIAL]=overrun_radio.build(base)\n    arm=make_arm(base,deployment,radios,cap);results=[]",1),
        ("opt=base.base.ARMS['proposed_fu_guard_cliperase' if method=='heuristic_ova' else method]",
         "opt=options(base,method)",1)],
        dict(OUT=OUT,EXTRA_DATASETS=["AirScape"],PARTIAL=partial.NEW,SERIAL=serial.NEW,
             state=state,make_arm=make_arm,options=options,validate=validate))
    run(next(c for c in CASES if c["id"]==case_id),DATASETS,NEW)
    directory=OUT/"cases"/case_id
    assert read(directory/"settings.json")==read(CHECKED/"cases"/case_id/"settings.json")
    for dataset in DATASETS:
        target=directory/(dataset.lower()+"_seed0");source=CHECKED/"cases"/case_id/target.name
        for name in ["plan.json","requests.json"]:assert read(target/name)==read(source/name)
        for method in serial.METHODS:link(source/method,target/method)
    print(dict(case=case_id,outcomes=[read(directory/(d.lower()+"_seed0")/m/"outcome.json")["status"] for d in DATASETS for m in NEW]),flush=True)


def verify():
    register();rows=[]
    for case in CASES:
        for dataset in DATASETS:
            directory=OUT/"cases"/case["id"]/(dataset.lower()+"_seed0")
            requests=read(directory/"requests.json")
            groups=collections.Counter(r["release_group"] for r in requests)
            assert len(requests)==45 and len(groups)==case["value"] and set(groups.values())=={45//case["value"]}
            for method in NEW:
                arm=directory/method;outcome=read(arm/"outcome.json")
                assert outcome["status"]!="failed",outcome
                assert read(arm/"started.json")["policy"]==read(BASELINE/"cases/baseline"/directory.name/method/"started.json")["policy"]
                row=dict(case=case["id"],dataset=dataset,method=method,status=outcome["status"],wall_seconds=outcome["wall_seconds"])
                if outcome["status"]=="completed":
                    check=read(arm/"verification.json")
                    assert check["FL_rounds"]==200 and check["FU_steps"]==225
                    tx=[read(p) for p in (arm/"transactions").glob("*.json")]
                    assert {r["kind"] for r in tx}==({"fl","joint"} if method==partial.NEW else {"fl","fu"})
                    assert sum(len(r["local"]) for r in tx)==225
                    for t in tx:
                        if method==partial.NEW:
                            if t["kind"]=="joint":assert t["radio"]["fu_noise_guard_active"]
                            for local in t["local"]:
                                assert local["FU_objective_weights"]==partial.WEIGHTS and local["RM_diagnostic_only"]
                                assert math.isclose(local["objective"],3*local["FM"]+3*local["CM"],rel_tol=1e-5,abs_tol=1e-7)
                        else:assert t["radio"]["physical_mode"]=="digital_sequential_ofdma"
                    row.update(transactions=len(tx),verified_FU_steps=225)
                rows.append(row)
    report=dict(passed=True,rows=rows,outcomes=dict(collections.Counter(r["status"] for r in rows)))
    write(OUT/"verification.json",report);return report


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("action",choices=["run","case","verify"]);p.add_argument("--case")
    a=p.parse_args()
    if a.action=="case":run_case(a.case)
    elif a.action=="verify":print(verify())
    else:
        register()
        for c in CASES:subprocess.run([sys.executable,str(Path(__file__)),"case","--case",c["id"]],cwd=ROOT,check=True)
        result=verify();write(OUT/"finished.json",dict(unix=time.time(),**result));state(stage="completed",outcomes=result["outcomes"])
        print(result,flush=True)
