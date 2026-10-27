"""Rerun affected single-request cases with broadcasts for nonzero head-local updates."""
import collections
import os
from pathlib import Path
import time
import numpy as np

import final_request_sweep as parent
import baseline_fairness as serial
from final_request_sweep import ROOT, BASELINE, DATASETS, read, write, sha, link, load_base
from airscape import extra, VideoData

OUT=ROOT/"results/final_broadcast_fix_v1_20261027"
SOURCE=parent.OUT
CASE=dict(id="request_bursts_45",axis="request_bursts",value=45)
METHODS=["separate_aircomp_periodic","aircomp_ofdma_periodic",serial.NEW]
ORIGINAL_COMMUNICATION=None
ENERGY=None
ORIGINAL_EXACT=None
REJECTIONS=0


def state(dest=OUT,**kw):write(Path(dest)/"status.json",dict(unix=time.time(),pid=os.getpid(),**kw))


def communication(radio):
    if radio["uplink_s"]>0 or not radio.get("broadcast_required",False):
        return ORIGINAL_COMMUNICATION(radio)
    assert abs(radio["energy_bound_j"])<1e-12
    p=ENERGY.PARAMETERS["communication"]
    bits=p["update_coordinates"]*p["broadcast_bits_per_coordinate"]+p["control_bits_per_update"]
    duration=bits/p["downlink_rate_bps"]
    energy=duration*(p["downlink_power_w"]/p["pa_efficiency"]+p["tx_circuit_w"]+9*p["rx_circuit_w"])
    return energy,duration


def strict_exact(*args,**kwargs):
    global REJECTIONS
    parts=ORIGINAL_EXACT(*args,**kwargs)
    if parts is None:return None
    c=args[6] if len(args)>6 else kwargs["c"]
    peak=max(float((p.analog_rms**2).max()) for p in parts)
    if peak>c["power_limit_w"]*(1+1e-6):
        REJECTIONS+=1;write(OUT/"candidate_rejections"/f"{os.getpid()}_{REJECTIONS:05d}.json",
            dict(peak_w=peak,limit_w=c["power_limit_w"],reason="Existing receiver 1ppm bound"));return None
    return parts


def pool_init(cap):
    global ORIGINAL_EXACT
    import radio_parallel_cap
    radio_parallel_cap.CAP=cap;ORIGINAL_EXACT=radio_parallel_cap.exact_aligned;radio_parallel_cap.exact_aligned=strict_exact


def make_arm(base,deployment,radios,cap):
    import event_arm
    edits=[
        ("radio=base.parent.guard if name=='proposed_fu_guard_cliperase' else base.SequentialRadio if name=='sequential_digital_periodic' else base.parent.common","radio=Radios[name]",1),
        ("if name!='sequential_digital_periodic':assert rm['energy_bound_j']<=.1*(1+2e-6)","if name not in ['sequential_digital_periodic','sequential_digital_proxy']:assert rm['energy_bound_j']<=NETWORK_CAP*(1+2e-6)",1),
        ("start=t;timing=clock.transaction_timing(compute,rm);end=t+timing['duration_s']",
         "rm['broadcast_required']=bool(np.any(vector != 0))\n  start=t;timing=clock.transaction_timing(compute,rm);end=t+timing['duration_s']",1)]
    maker=base.repeat.clone_function(event_arm.make_arm,[
        ("return base.repeat.clone_function","edits.extend(EXTRA_EDITS)\n    return base.repeat.clone_function",1),
        ("deployment=deployment,state=state))","deployment=deployment,state=state,Radios=RADIOS,NETWORK_CAP=CAP))",1)],
        dict(EXTRA_EDITS=edits,RADIOS=radios,CAP=cap))
    return maker(base,deployment,state)


def options(base,method):return serial.options(base) if method==serial.NEW else base.base.ARMS[method]


def register():
    path=OUT/"protocol.json"
    if path.exists():
        p=read(path)
        for group in ["source_sha256","input_sha256"]:
            for name,digest in p[group].items():assert sha(ROOT/name)==digest,name
        return p
    parent.register();assert read(SOURCE/"finished.json")["passed"]
    source=dict(read(SOURCE/"protocol.json")["source_sha256"])
    source[str(Path(__file__).relative_to(ROOT))]=sha(Path(__file__))
    inputs={};audit=[]
    # Audit every main-method release case: only these six arms contain omitted broadcasts.
    all_methods=[parent.partial.PROPOSED,parent.partial.NEW,*METHODS]
    for case in ["baseline",*[c["id"] for c in parent.CASES]]:
        for dataset in DATASETS:
            dd=SOURCE/"cases"/case/(dataset.lower()+"_seed0")
            for method in all_methods:
                arm=dd/method;missing=[]
                for f in sorted((arm/"work_attempts").glob("*.json")):
                    r=read(f)
                    if r["status"]!="transmitted":continue
                    active=r["FL_vector_norm"]>0 or r["FU_vector_norm"]>0
                    if active and r["radio"]["uplink_s"]==0 and r["timing"]["downlink_s"]==0:
                        missing.append(r["transaction"]);inputs[str(f.relative_to(ROOT))]=sha(f)
                audit.append(dict(case=case,dataset=dataset,method=method,omitted_broadcasts=len(missing),transactions=missing))
                if missing:assert case==CASE["id"] and method in METHODS,(case,dataset,method)
    affected=[r for r in audit if r["omitted_broadcasts"]]
    assert len(affected)==6
    for dataset in DATASETS:
        dd=SOURCE/"cases"/CASE["id"]/(dataset.lower()+"_seed0")
        for name in ["plan.json","requests.json"]:inputs[str((dd/name).relative_to(ROOT))]=sha(dd/name)
        for method in METHODS:
            f=dd/method/"outcome.json";inputs[str(f.relative_to(ROOT))]=sha(f)
    p=dict(registered_unix=time.time(),case=CASE,datasets=DATASETS,methods=METHODS,new_arms=6,
        correction="A nonzero decoded update requires a broadcast even when the aggregation head is the only active sender and uplink duration is zero. True zero updates still have no broadcast. Use the existing payload, downlink rate and PA/circuit coefficients.",
        experiment_policy="Rerun all six affected arms because service-clock changes can affect scheduling. Keep all previous raw outcomes; do not repair a curve by shifting values. No objectives, radio limits, requests, seeds or optimizer parameters change.",
        unaffected_scope="All other main-method burst conditions and baseline/bandwidth figures remain unchanged. The audit checks nonzero local work with absent uplink and broadcast.",
        affected_audit=audit,source_sha256=source,input_sha256=inputs)
    write(path,p);link(BASELINE/"prepared/airscape_seed0",OUT/"prepared/airscape_seed0");return p


def run():
    global ORIGINAL_COMMUNICATION,ENERGY
    register();_,base=load_base()
    import event_energy,mission_clock,worker,heuristic
    ENERGY=event_energy.energy
    assert mission_clock.energy is ENERGY
    ORIGINAL_COMMUNICATION=ENERGY.communication;ENERGY.communication=communication
    # Meaningful checks: nonzero head-local work broadcasts; zero updates don't.
    zero=dict(uplink_s=0.,energy_bound_j=0.)
    assert communication(zero)==(0.,0.)
    assert communication(dict(zero,broadcast_required=True))[1]==.655872
    heuristic.pool_init=pool_init;extra.Data=VideoData
    original_energy=event_energy.method_energy
    def energy(*args,**kwargs):
        r=original_energy(*args,**kwargs)
        if r["method"]==serial.NEW:r["network_cap_policy"]="soft, original-feasible preferred"
        return r
    event_energy.method_energy=energy
    validate=base.repeat.clone_function(worker.validate,[("if ad.name!='sequential_digital_periodic':","if ad.name not in ['sequential_digital_periodic','sequential_digital_proxy']:",1)])
    fn=base.repeat.clone_function(worker.run,[
        ("arm=make_arm(base,deployment,radios,cap);results=[]","radios[SERIAL]=overrun_radio.build(base)\n    arm=make_arm(base,deployment,radios,cap);results=[]",1),
        ("opt=base.base.ARMS['proposed_fu_guard_cliperase' if method=='heuristic_ova' else method]","opt=options(base,method)",1)],
        dict(OUT=OUT,EXTRA_DATASETS=["AirScape"],SERIAL=serial.NEW,state=state,make_arm=make_arm,options=options,validate=validate))
    fn(CASE,DATASETS,METHODS)
    rows=[]
    for dataset in DATASETS:
        dd=OUT/"cases"/CASE["id"]/(dataset.lower()+"_seed0");old=SOURCE/"cases"/CASE["id"]/dd.name
        for name in ["plan.json","requests.json"]:assert read(dd/name)==read(old/name)
        for method in [parent.partial.PROPOSED,parent.partial.NEW]:link(old/method,dd/method)
        for method in METHODS:
            arm=dd/method;o=read(arm/"outcome.json");assert o["status"]=="completed",o
            assert read(arm/"started.json")["policy"]==read(old/method/"started.json")["policy"]
            work={r["transaction"]:r for r in [read(p) for p in (arm/"work_attempts").glob("*.json")]}
            added=0
            for p in (arm/"transactions").glob("*.json"):
                t=read(p);r=work[t["sequence"]];radio=t["radio"]
                if radio["broadcast_required"]:assert t["timing"]["downlink_s"]==.655872
                else:assert t["timing"]["downlink_s"]==0
                if radio["uplink_s"]==0 and radio["broadcast_required"]:
                    added+=1;assert r["FU_vector_norm"]>0
            assert added>0
            checked=read(arm/"verification.json");assert checked["FU_steps"]==225 and checked["FL_rounds"]==200
            rows.append(dict(dataset=dataset,method=method,status=o["status"],correctly_charged_local_broadcasts=added,
                old_mean_latency_s=read(old/method/"outcome.json")["metrics"]["FU_mean_completion_latency_s"],
                new_mean_latency_s=o["metrics"]["FU_mean_completion_latency_s"]))
    result=dict(passed=True,rows=rows,outcomes=dict(collections.Counter(r["status"] for r in rows)))
    write(OUT/"verification.json",result);write(OUT/"finished.json",result);state(stage="completed",outcomes=result["outcomes"])
    print(result,flush=True)


if __name__=="__main__":run()
