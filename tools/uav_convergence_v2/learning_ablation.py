"""Learning-component ablation: keep FM/CM and remove RM from FU only.

All FL learning and Proposed communication controls remain unchanged.
The coefficient is exactly zero, not tuned to create an intermediate rank.
"""
import argparse
import collections
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from types import FunctionType, SimpleNamespace

import numpy as np
import baseline_fairness as previous
from baseline_fairness import ROOT, DATASETS, CASES, read, write, sha, load_base, link
from airscape import VideoData, extra

OUT = ROOT / "results/learning_ablation_v1_20261027"
SOURCE = ROOT / "results/guard_ablation_v1_20261027"
NEW = "joint_fm_cm_no_rm"
LABEL = "FM-CM"
PROPOSED = "proposed_fu_guard_cliperase"
WEIGHTS = dict(retention=0., forgetting=3., consistency=3.)
ORIGINAL_EXACT = None
REJECTIONS = 0


def state(dest=OUT, **kw):
    write(Path(dest)/"status.json",dict(unix=time.time(),pid=os.getpid(),**kw))


def strict_exact(*args, **kwargs):
    global REJECTIONS
    parts = ORIGINAL_EXACT(*args,**kwargs)
    if parts is None:
        return None
    config=args[6] if len(args)>6 else kwargs["c"]
    peak=max(float((part.analog_rms**2).max()) for part in parts)
    if peak>config["power_limit_w"]*(1+1e-6):
        REJECTIONS+=1
        write(OUT/"candidate_rejections"/f"{os.getpid()}_{REJECTIONS:05d}.json",
            dict(peak_w=peak,limit_w=config["power_limit_w"],reason="Existing receiver 1ppm bound"))
        return None
    return parts


def pool_init(cap):
    global ORIGINAL_EXACT
    import radio_parallel_cap
    radio_parallel_cap.CAP=cap
    ORIGINAL_EXACT=radio_parallel_cap.exact_aligned
    radio_parallel_cap.exact_aligned=strict_exact


def partial_gradient(base):
    assert base.clip.WEIGHTS==dict(retention=1.,forgetting=3.,consistency=3.)
    return base.repeat.clone_function(base.clip.request_gradient,
        [("return g,dict(","return g,dict(FU_objective_weights=dict(WEIGHTS), RM_diagnostic_only=True, ",1)],
        dict(WEIGHTS=dict(WEIGHTS)))


def partial_vectors(base, objective):
    original=base.control.synchronized_vectors(objective)
    if objective!="cliperase":
        return original
    # Identical vector/sampling/LR code; only the per-request FU gradient differs.
    return FunctionType(original.__code__,dict(original.__globals__,request_gradient=partial_gradient(base)),
        original.__name__,original.__defaults__,original.__closure__)


def make_arm(base,deployment,radios,cap):
    import worker
    control=SimpleNamespace(**base.control.__dict__)
    control.synchronized_vectors=lambda objective:partial_vectors(base,objective)
    return worker.make_arm(SimpleNamespace(**dict(base.__dict__,control=control)),deployment,radios,cap)


def options(base):
    return dict(base.base.ARMS[PROPOSED],FU_objective_weights=dict(WEIGHTS),
        learning_ablation="FU RM coefficient 1 -> 0; FL/FM/CM unchanged")


def gradient_check(base,engine,plan,requests,warm,directory):
    """Check actual training-only gradients; no validation/reference-based choice."""
    import torch
    import torch.nn.functional as F
    request=requests[0];fids=request["images"];user=request["user"]
    pool=sorted(set(plan["partitions"][user])-set(plan["forgotten_images"]))
    rids=np.random.default_rng(910000+request["burst"]*1000+user*10).choice(pool,min(16,len(pool)),replace=False).astype(int).tolist()
    assert set(fids+rids)<=set(plan["partitions"][user])
    teacher=base.core.Teacher(engine,warm)
    full,full_meta=base.clip.request_gradient(engine,warm,fids,rids,0,teacher)
    partial,partial_meta=partial_gradient(base)(engine,warm,fids,rids,0,teacher)
    variable=warm.detach().requires_grad_(True)
    captions=engine.data.training_caption_ids(rids,21)
    im,tx,scale=engine.features(variable,rids,captions)
    logits=scale.exp()*(im@tx.T);target=torch.arange(len(rids),device=logits.device)
    rm=(F.cross_entropy(logits,target)+F.cross_entropy(logits.T,target))/2
    removed=torch.autograd.grad(rm,variable)[0].detach();removed[-1]=0
    difference=full-partial
    assert torch.allclose(difference,removed,rtol=3e-5,atol=3e-7)
    assert float(removed.norm())>1e-8
    full_vectors=base.control.synchronized_vectors("cliperase")
    no_rm_vectors=partial_vectors(base,"cliperase")
    assert full_vectors.__code__.co_code==no_rm_vectors.__code__.co_code
    queue=base.RoundQueue(requests)
    fl_full,_,_=full_vectors(engine,warm,plan,queue,0.,0,"fl",[],None,{},True)
    fl_partial,_,_=no_rm_vectors(engine,warm,plan,queue,0.,0,"fl",[],None,{},True)
    assert np.array_equal(fl_full,fl_partial)
    write(directory/"learning_component_check.json",dict(passed=True,request=request["id"],
        training_only=True,removed_component="retained contrastive RM",
        full_minus_partial_equals_RM_gradient=True,removed_gradient_norm=float(removed.norm()),
        residual_max_absolute=float((difference-removed).abs().max()),FL_vectors_bit_identical=True,
        partial_weights=partial_meta["FU_objective_weights"],full_objective=full_meta["objective"],
        partial_objective=partial_meta["objective"]))


def register():
    path=OUT/"protocol.json"
    if path.exists():
        protocol=read(path)
        for group in ["source_sha256","input_sha256"]:
            for name,digest in protocol[group].items():
                assert sha(ROOT/name)==digest,name
        return protocol
    parent=previous.register()
    assert not list((OUT/"cases").glob("*/*/"+NEW+"/outcome.json"))
    sources=dict(parent["source_sha256"]);sources[str(Path(__file__).relative_to(ROOT))]=sha(Path(__file__))
    inputs=dict(parent["input_sha256"])
    for case in CASES:
        for dataset in DATASETS:
            for method in [previous.NEW,"unguarded_ova"]:
                file=SOURCE/"cases"/case["id"]/(dataset.lower()+"_seed0")/method/"outcome.json"
                inputs[str(file.relative_to(ROOT))]=sha(file)
    protocol=dict(registered_unix=time.time(),datasets=DATASETS,cases=CASES,seed=0,new_arms=10,
        new_method=NEW,label=LABEL,FU_weights=WEIGHTS,
        definition="Proposed minus the supervised retained-data contrastive term RM in FU: 3*FM + 0*RM + 3*CM. Standard FL objective and all Proposed communication controls unchanged.",
        fixed="Same FL loss/gradient, frozen backbone/adapters, data and captions, warm20/proxy20/reference200, sampled batches, LR schedule, teachers, 200 FL rounds, 45 requests x 5 FU steps, paired seeds, joint scheduler, FU-relative guard, optimized head/resources, physical limits, hard 0.1J TX cap and energy model.",
        accounting="Retained features are still required by CM. RM is computed and logged diagnostically but has zero objective/gradient contribution. The existing adapter MAC model does not separately count contrastive matrix/softmax arithmetic, for either arm; it is not a hardware runtime measurement.",
        preflight="On a fixed real training-only batch, full gradient minus partial gradient must equal independently calculated RM gradient; FL update vectors must be bit-identical.",
        objective_source="FM/RM/CM adapt CLIPErase (ACL2025); this is an ablation of the adopted FU objective, not a claim of a newly invented FL loss or an independent published baseline.",
        outcome_policy="All ten outcomes retained regardless of rank, feasibility or utility/forgetting tradeoff. RM coefficient exactly zero; no coefficient search, result interpolation, seed/dataset selection or replacement by an intermediate curve.",
        source_sha256=sources,input_sha256=inputs)
    write(path,protocol);link(SOURCE/"prepared/airscape_seed0",OUT/"prepared/airscape_seed0")
    return protocol


def run_case(case_id):
    register()
    import worker,heuristic
    _,base=load_base();extra.Data=VideoData;heuristic.pool_init=pool_init;worker.state=state
    run=base.repeat.clone_function(worker.run,[
        ("arm=make_arm(base,deployment,radios,cap);results=[]","radios[NEW]=base.base.parent.guard\n    arm=make_arm(base,deployment,radios,cap);results=[]",1),
        ("opt=base.base.ARMS['proposed_fu_guard_cliperase' if method=='heuristic_ova' else method]",
         "assert method==NEW\n                    opt=options(base)\n                    if case['id']=='baseline':gradient_check(base,engine,plan,requests,warm,dd)",1)],
        dict(OUT=OUT,EXTRA_DATASETS=["AirScape"],NEW=NEW,state=state,make_arm=make_arm,
             options=options,gradient_check=gradient_check))
    run(next(c for c in CASES if c["id"]==case_id),DATASETS,[NEW])
    directory=OUT/"cases"/case_id
    assert read(directory/"settings.json")==read(SOURCE/"cases"/case_id/"settings.json")
    for dataset in DATASETS:
        source=SOURCE/"cases"/case_id/(dataset.lower()+"_seed0");dest=directory/(dataset.lower()+"_seed0")
        for name in ["plan.json","requests.json"]:
            assert read(dest/name)==read(source/name)
        for method in dict(previous.METHODS,**{previous.NEW:"S-OF","unguarded_ova":"NG-OVA"}):
            link(source/method,dest/method)
    print(dict(case=case_id,outcomes={d:read(directory/(d.lower()+"_seed0")/NEW/"outcome.json")["status"] for d in DATASETS}),flush=True)


def verify():
    rows=[]
    for case in CASES:
        for dataset in DATASETS:
            directory=OUT/"cases"/case["id"]/(dataset.lower()+"_seed0");arm=directory/NEW
            outcome=read(arm/"outcome.json");assert outcome["status"]!="failed",outcome
            policy=read(arm/"started.json")["policy"];parent=read(directory/PROPOSED/"started.json")["policy"]
            assert all(policy[k]==value for k,value in parent.items())
            assert policy["FU_objective_weights"]==WEIGHTS
            row=dict(case=case["id"],dataset=dataset,status=outcome["status"],wall_seconds=outcome["wall_seconds"])
            if case["id"]=="baseline":assert read(directory/"learning_component_check.json")["passed"]
            if outcome["status"]=="completed":
                checked=read(arm/"verification.json");assert checked["FL_rounds"]==200 and checked["FU_steps"]==225
                tx=[read(p) for p in (arm/"transactions").glob("*.json")]
                assert {r["kind"] for r in tx}=={"fl","joint"}
                local_steps=0
                for r in tx:
                    if r["kind"]=="joint":assert r["radio"]["fu_noise_guard_active"]
                    for local in r["local"]:
                        assert local["FU_objective_weights"]==WEIGHTS and local["RM_diagnostic_only"]
                        assert math.isclose(local["objective"],3*local["FM"]+3*local["CM"],rel_tol=1e-5,abs_tol=1e-7)
                        local_steps+=1
                assert local_steps==225
                row.update(transactions=len(tx),verified_FU_steps=local_steps)
            rows.append(row)
    result=dict(passed=True,rows=rows,outcomes=dict(collections.Counter(r["status"] for r in rows)))
    write(OUT/"verification.json",result);return result


def run_all():
    register()
    for case in CASES:
        subprocess.run([sys.executable,str(Path(__file__)),"case","--case",case["id"]],cwd=ROOT,check=True)
    result=verify();write(OUT/"finished.json",dict(unix=time.time(),**result));state(stage="completed",outcomes=result["outcomes"])
    print(result,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("action",choices=["run","case","verify"]);parser.add_argument("--case")
    args=parser.parse_args()
    if args.action=="run":run_all()
    elif args.action=="case":run_case(args.case)
    else:print(verify())
