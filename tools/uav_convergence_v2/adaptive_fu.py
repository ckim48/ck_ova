"""Training-loss feedback for retrieval FU, with explicit radio feasibility repair.

Same controller for both datasets. Validation mR and reference200 are evaluation
only. This is an empirical surrogate controller, not a retrain/deletion guarantee.
"""
import argparse
import collections
import hashlib
import inspect
import linecache
import math
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import time
from types import SimpleNamespace

import numpy as np
import fu_joint_optimization as infra
from fu_retrieval_alignment import retrieval_kl
from fu_joint_optimization import ROOT, SOURCE, PROPOSED, DATASETS, read, write, sha, link

OUT = ROOT/"results/adaptive_fu_v1_20261028"
NEW = "adaptive_retrieval_fu"
CONTROLLER = dict(initial=.5, lower=.25, upper=1., gain=.5, ema=.5, denominator_floor=1e-8)
CASES = [dict(id="baseline", axis=None, value=None),
         *[dict(id=f"bandwidth_{v}", axis="bandwidth", value=v) for v in [2,4]],
         *[dict(id=f"request_bursts_{v}", axis="request_bursts", value=v) for v in [3,5,15,45]]]
ACTIVE_ENGINE = None
ACTIVE_METHOD = NEW
SAMPLING_SEED = 0
TASKROOT = OUT


def feedback(weight, ema, retain_relative, forget_relative):
    signal = float(np.clip(retain_relative-forget_relative, -1., 1.))
    smooth = CONTROLLER["ema"]*ema+(1-CONTROLLER["ema"])*signal
    desired = float(np.clip(weight*math.exp(CONTROLLER["gain"]*smooth), CONTROLLER["lower"], CONTROLLER["upper"]))
    return desired, smooth, signal


def request_gradient(engine, theta, fids, rids, step, teacher):
    import torch
    import torch.nn.functional as F
    rid = engine.adaptive_request_id
    cids = [c for i in fids for c in engine.data.caption_ids[i]]
    rcids = engine.data.training_caption_ids(rids,21+step)
    variable = theta.detach().requires_grad_(True)
    im,tx,scale = engine.features(variable,fids,cids)
    old_forget = engine.adaptive_proxy.features(fids,cids)
    fk = retrieval_kl(im,tx,*old_forget,scale)
    old_retain = teacher.features(rids,rcids)
    ri,rt,scale = engine.features(variable,rids,rcids)
    logits = scale.exp()*(ri@rt.T)
    target = torch.arange(len(rids),device=theta.device)
    rm = .5*(F.cross_entropy(logits,target)+F.cross_entropy(logits.T,target))
    ck = retrieval_kl(ri,rt,*old_retain,scale)
    with torch.no_grad():
        old_logits=scale.detach().exp()*(old_retain[0]@old_retain[1].T)
        old_rm=.5*(F.cross_entropy(old_logits,target)+F.cross_entropy(old_logits.T,target))
    current_forget=max(float(fk.detach()),0.)
    start_forget = getattr(teacher,"adaptive_initial_forget",current_forget)
    old_weight = getattr(teacher,"adaptive_weight",CONTROLLER["initial"])
    old_ema = getattr(teacher,"adaptive_ema",0.)
    dr=(float(rm.detach())-float(old_rm))/max(float(old_rm),CONTROLLER["denominator_floor"])
    df=(current_forget-start_forget)/max(start_forget,CONTROLLER["denominator_floor"])
    weight, smooth, signal = feedback(old_weight,old_ema,dr,df)
    # A separate retention basis permits feasibility repair without another backward.
    g0=torch.autograd.grad(3*fk+3*ck,variable,retain_graph=True)[0].detach();g0[-1]=0
    gr=torch.autograd.grad(rm,variable)[0].detach();gr[-1]=0
    g=g0+weight*gr
    assert torch.isfinite(g).all()
    dim=im.shape[1];n=len(theta)-1
    extra=(len(fids)*len(cids)+len(rids)*len(rcids))*(4*dim+32)
    extra+=len(rids)*len(rcids)*(dim+16)
    extra+=2*10240*(len(rids)+len(rcids))  # second adapter backward on retain graph
    extra+=16*n+64  # feedback, vector combination, worst-case two repair combinations
    infra.charge(engine,fids,int(extra))
    engine.adaptive_pending[rid]=dict(teacher=teacher,initial_forget=start_forget,ema=smooth,
        desired=weight,retention_gradient=gr[:-1].cpu().numpy(),user=engine.sample_owner[int(fids[0])])
    return g,dict(FU_variant=NEW,FM=None,RM=float(rm.detach()),CM=float(ck.detach()),
        forget_loss=float(fk.detach()),preservation_loss=float((weight*rm+3*ck).detach()),
        objective=float((3*fk+weight*rm+3*ck).detach()),FU_gradient_norm=float(g.norm()),
        RM_weight=weight,desired_RM_weight=weight,previous_RM_weight=old_weight,feedback_ema=smooth,
        feedback_signal=signal,retain_relative_change=dr,forget_relative_change=df,
        retained_teacher_CE=float(old_rm),request_initial_proxy_KL=start_forget,
        extra_control_MAC_equivalents=int(extra),images=len(fids),forget_captions=len(cids),
        retained_images=rids,retained_captions=rcids,
        controller_inputs="training-only retained CE versus request teacher and forget proxy-KL versus request start")


def start_vectors(engine,proxy):
    global ACTIVE_ENGINE
    ACTIVE_ENGINE=engine
    engine.adaptive_proxy=proxy;engine.adaptive_pending={}


def finish_vectors(engine,l,u,stats):
    for row in stats:
        engine.adaptive_pending[row["request"]]["row"]=row
    engine.adaptive_initial_u=u.copy()
    return l,u,stats


def vectors(base,objective):
    original=base.control.synchronized_vectors(objective)
    if objective!="cliperase" or ACTIVE_METHOD!=NEW:return original
    source=textwrap.dedent(inspect.getsource(original))
    edits=[("stats=[]","stats=[];start_vectors(engine,proxy)",1),
           ("g,meta=request_gradient(engine,theta,fids,rids,step,teachers[rid])",
            "engine.adaptive_request_id=rid\n        g,meta=request_gradient(engine,theta,fids,rids,step,teachers[rid])",1),
           ("u[user]+=(-core.lr_at(21+fl)*len(fids)/total*g[:-1]).cpu().numpy()",
            "u[user]+=(-core.lr_at(21+fl)*len(fids)/total*g[:-1]).cpu().numpy()\n        engine.adaptive_pending[rid]['basis']=(-core.lr_at(21+fl)*len(fids)/total)*engine.adaptive_pending[rid]['retention_gradient']",1),
           ("return l,u,stats","return finish_vectors(engine,l,u,stats)",1)]
    for old,new,count in edits:
        assert source.count(old)==count,old
        source=source.replace(old,new)
    filename="/tmp/adaptive_fu_vectors_"+hashlib.sha256(source.encode()).hexdigest()+".py"
    linecache.cache[filename]=(len(source),None,source.splitlines(True),filename)
    namespace=dict(original.__globals__,request_gradient=request_gradient,start_vectors=start_vectors,finish_vectors=finish_vectors)
    exec(compile(source,filename,"exec"),namespace)
    return namespace[original.__name__]


def radio(base):
    original=base.base.parent.guard
    class AdaptiveRadio:
        @staticmethod
        def analog(l,u,channels,config,seed):
            pending=ACTIVE_ENGINE.adaptive_pending
            if not pending:return original.analog(l,u,channels,config,seed)
            desired={rid:p["desired"] for rid,p in pending.items()}
            candidates=[desired]
            for floor in [.5,1.]:
                weights={rid:max(floor,w) for rid,w in desired.items()}
                if weights not in candidates:candidates.append(weights)
            attempts=[]
            for weights in candidates:
                u[:]=ACTIVE_ENGINE.adaptive_initial_u
                for rid,p in pending.items():u[p["user"]]+=(weights[rid]-p["desired"])*p["basis"]
                received,meta=original.analog(l,u,channels,config,seed)
                attempts.append(dict(weights={str(k):v for k,v in weights.items()},status=meta["status"],feasible=received is not None))
                if received is not None:break
            meta.update(adaptive_weight_trials=attempts,adaptive_radio_repaired=len(attempts)>1,
                        controller_scope="Same desired loss feedback; first feasible bounded retain-weight lift, with unchanged radio limits. No evaluation metrics used.")
            for rid,p in pending.items():
                row=p["row"];w=weights[rid]
                row.update(RM_weight=w,radio_weight_repaired=w!=p["desired"],radio_trials=len(attempts),
                           preservation_loss=w*row["RM"]+3*row["CM"],objective=3*row["forget_loss"]+w*row["RM"]+3*row["CM"])
                if received is not None:
                    teacher=p["teacher"];teacher.adaptive_weight=w;teacher.adaptive_ema=p["ema"]
                    teacher.adaptive_initial_forget=p["initial_forget"]
            return received,meta
    return AdaptiveRadio


def make_arm(base,deployment,radios,cap):
    import worker,event_arm
    control=SimpleNamespace(**base.control.__dict__)
    control.synchronized_vectors=lambda objective:vectors(base,objective)
    adapted=SimpleNamespace(**dict(base.__dict__,control=control))
    # Apply the previously audited broadcast rule to every new run.
    edit=("start=t;timing=clock.transaction_timing(compute,rm);end=t+timing['duration_s']",
          "rm['broadcast_required']=bool(np.any(vector != 0))\n  start=t;timing=clock.transaction_timing(compute,rm);end=t+timing['duration_s']",1)
    maker=base.repeat.clone_function(worker.make_arm,[("extra=[","extra=BROADCAST_EDITS+[",1)],dict(BROADCAST_EDITS=[edit],state=state))
    return maker(adapted,deployment,radios,cap)


def state(dest=None,**kwargs):write((Path(dest) if dest is not None else TASKROOT)/"status.json",dict(unix=time.time(),pid=os.getpid(),**kwargs))


def unit_checks():
    global ACTIVE_ENGINE
    up,_,_=feedback(.5,0.,.2,0.);down,_,_=feedback(.5,0.,0.,.2)
    assert up>.5>down
    assert feedback(.5,0.,0.,0.)[0]==.5
    for weight in [.25,.5,1.]:
        for r,f in [(1e8,0.),(0.,1e8),(-1e8,1e8)]:
            value,_,_=feedback(weight,0.,r,f);assert .25<=value<=1
    # Feasibility repair changes exactly the retention term in the desired sum.
    g0=np.array([1.,2.]);gr=np.array([-3.,4.]);eta=.01
    low=-eta*(g0+.25*gr);repaired=low+(.5-.25)*(-eta*gr)
    assert np.allclose(repaired,-eta*(g0+.5*gr))
    previous=ACTIVE_ENGINE;calls=[]
    def fake_radio(l,u,h,c,s):
        calls.append(u.copy())
        return (None,dict(status="infeasible")) if np.linalg.norm(u)<.4 else (u.sum(0),dict(status="feasible"))
    record=dict(RM=2.,CM=.1,forget_loss=.2);teacher=SimpleNamespace()
    pending=dict(desired=.25,basis=np.array([-1.,0.]),user=0,row=record,teacher=teacher,initial_forget=.2,ema=.1)
    ACTIVE_ENGINE=SimpleNamespace(adaptive_pending={0:pending},adaptive_initial_u=np.array([[-.25,0.]]))
    fake=SimpleNamespace(base=SimpleNamespace(parent=SimpleNamespace(guard=SimpleNamespace(analog=fake_radio))))
    try:
        received,meta=radio(fake).analog(np.zeros((1,2)),np.array([[-.25,0.]]),None,None,7)
        assert len(calls)==2 and np.allclose(received,[-.5,0.]) and record["RM_weight"]==.5
        assert teacher.adaptive_weight==.5 and meta["adaptive_radio_repaired"]
    finally:ACTIVE_ENGINE=previous
    return dict(passed=True,retention_regression_increases_weight=True,forget_regression_decreases_weight=True,
                bounds_and_zero_feedback=True,repair_changes_only_retention_term=True,first_feasible_radio_and_state_commit=True)


def register():
    path=OUT/"protocol.json"
    if path.exists():
        result=read(path)
        for group in ["source_sha256","input_sha256"]:
            for name,digest in result[group].items():assert sha(ROOT/name)==digest,name
        return result
    previous=read(ROOT/"results/fu_retrieval_refinement_v2_20261027/protocol.json")
    sources=dict(previous["source_sha256"]);sources[str(Path(__file__).relative_to(ROOT))]=sha(Path(__file__))
    inputs=dict(previous["input_sha256"])
    for name in ["fu_retrieval_pilot/data/comparison.json","fu_retrieval_refinement_pilot/data/comparison.json"]:inputs[name]=sha(ROOT/name)
    chart_snapshot={str(p.relative_to(ROOT/"final_chart")):sha(p) for p in (ROOT/"final_chart").rglob("*") if p.is_file()}
    result=dict(registered_unix=time.time(),method=NEW,datasets=DATASETS,cases=CASES,controller=CONTROLLER,
        runs="14 primary runs (7 conditions x2 datasets, sampling seed0), plus8 paired post-warm runs (adaptive and original Proposed, seeds1/2 x2 datasets). Fixed warm20/data/requests reused; these seeds vary minibatch/radio randomness, not data splits or initial training.",
        objective="3 forget retrieval KL to existing clean proxy20 + lambda_R retained CE +3 retained retrieval KL to request-start teacher. Frozen temperature; all native captions.",
        feedback="At every request step, normalize retained CE change by teacher CE and proxy-forget KL change by its first-step value (denominator floor1e-8). Smooth clipped retain_change-minus-forget_change with EMA0.5; multiply previous weight by exp(0.5*EMA), clip to[0.25,1]. Each request starts at0.5. Same constants and code for both datasets; no dataset-name branch.",
        feasibility="Try desired weights; if radio infeasible, raise active weights to at least0.5 then1, stopping at first feasible design. Precompute the retention gradient basis and apply its exact linear correction; retain every failed trial. Keep hard0.1J cap and all original physical constraints. Do not select a radio realization by quality.",
        objective_limits="Training CE is a surrogate for retained mR; proxy KL is a surrogate for final retrain KL. No guarantee of simultaneous metric improvement or deletion. The existing common clean proxy assumes the planned forget cohort was known before mission, as in prior baselines.",
        leakage_policy="Evaluation reference200, validation mR/KL, future evaluation results and dataset labels never enter feedback. This is a development-conditioned design after previous experiments; no independent-split claim. No outcome-based coefficient changes within this campaign.",
        accounting="Charge executed extra teacher forwards, retrieval matrices and extra retain backward, plus conservative worst-case vector repair arithmetic. Same service communication+compute+hover scope. As in existing controls, optimizer wall time is logged but not a calibrated UAV processor energy measurement; report trial counts. No extra payload coordinates.",
        final_chart_policy="Archive all previous final_chart files byte-for-byte; publish the actual adaptive trajectory consistently across round/time, bandwidth, request-load, quality-cost and quality-table charts. Keep unfavorable/infeasible outcomes visible and original-Proposed comparisons in supplement. Do not claim the new method wins all metrics.",
        source_sha256=sources,input_sha256=inputs,previous_final_chart_sha256=chart_snapshot)
    write(path,result);write(OUT/"unit_checks.json",unit_checks());return result


def run_case(case_id,seed,method):
    global ACTIVE_METHOD,SAMPLING_SEED,TASKROOT
    register();ACTIVE_METHOD=method;SAMPLING_SEED=seed
    TASKROOT=OUT if seed==0 else OUT/f"sampling_seed_{seed}"
    link(SOURCE/"prepared/airscape_seed0",TASKROOT/"prepared/airscape_seed0")
    _,base=infra.prior.load_base()
    import worker,heuristic,accounting,final_broadcast_fix as fix,event_energy,mission_clock
    infra.extra.Data=infra.VideoData;infra.OUT=TASKROOT;heuristic.pool_init=infra.pool_init
    infra.ORIGINAL_INSTRUMENTED=accounting.instrumented_engine;accounting.instrumented_engine=infra.controlled_engine
    fix.ENERGY=event_energy.energy;fix.ORIGINAL_COMMUNICATION=fix.ENERGY.communication;fix.ENERGY.communication=fix.communication
    assert mission_clock.energy is fix.ENERGY
    run=base.repeat.clone_function(worker.run,[
        ("arm=make_arm(base,deployment,radios,cap);results=[]","radios[NEW]=adaptive_radio(base)\n    arm=make_arm(base,deployment,radios,cap);results=[]",1),
        ("opt=base.base.ARMS['proposed_fu_guard_cliperase' if method=='heuristic_ova' else method]",
         "opt=dict(base.base.ARMS[PROPOSED],adaptive_controller=CONTROLLER if method==NEW else None,sampling_seed=SEED)",1),
        ("],dd,0)","],dd,SEED)",1)],
        dict(OUT=TASKROOT,EXTRA_DATASETS=["AirScape"],NEW=NEW,PROPOSED=PROPOSED,CONTROLLER=CONTROLLER,
             SEED=seed,adaptive_radio=radio,make_arm=make_arm,state=state))
    run(next(c for c in CASES if c["id"]==case_id),DATASETS,[method])
    print(dict(case=case_id,seed=seed,method=method,outcomes={ds:read(TASKROOT/"cases"/case_id/(ds.lower()+"_seed0")/method/"outcome.json")["status"] for ds in DATASETS}),flush=True)


def tasks():
    return [(c["id"],0,NEW) for c in CASES]+[("baseline",seed,m) for seed in [1,2] for m in [NEW,PROPOSED]]


def verify():
    register();rows=[]
    for case,seed,method in tasks():
        root=OUT if seed==0 else OUT/f"sampling_seed_{seed}"
        for dataset in DATASETS:
            arm=root/"cases"/case/(dataset.lower()+"_seed0")/method;o=read(arm/"outcome.json")
            row=dict(case=case,seed=seed,dataset=dataset,method=method,status=o["status"])
            assert o["status"]!="failed",o
            if o["status"]=="completed":
                v=read(arm/"verification.json");assert v["FL_rounds"]==200 and v["FU_steps"]==225
                work=[read(p) for p in sorted((arm/"work_attempts").glob("*.json"))]
                local=[r for w in work for r in w["local"]];assert len(local)==225
                if method==NEW:
                    for r in local:
                        expected,ema,signal=feedback(r["previous_RM_weight"],(r["feedback_ema"]-(1-CONTROLLER["ema"])*r["feedback_signal"])/CONTROLLER["ema"],r["retain_relative_change"],r["forget_relative_change"])
                        assert math.isclose(expected,r["desired_RM_weight"],abs_tol=1e-12)
                        assert .25<=r["desired_RM_weight"]<=r["RM_weight"]<=1.
                        assert math.isclose(r["objective"],3*r["forget_loss"]+r["RM_weight"]*r["RM"]+3*r["CM"],rel_tol=1e-5,abs_tol=1e-6)
                    row.update(weight_min=min(r["RM_weight"] for r in local),weight_max=max(r["RM_weight"] for r in local),
                               weight_mean=float(np.mean([r["RM_weight"] for r in local])),radio_repaired_steps=sum(r["radio_weight_repaired"] for r in local))
                for w in work:
                    for a,b,total in zip(w["compute"]["base_device_macs"],w["compute"]["FU_control_scalar_ops"],w["compute"]["device_macs"]):assert total==a+b
            rows.append(row)
    result=dict(passed=True,rows=rows,outcomes=dict(collections.Counter(r["status"] for r in rows)))
    write(OUT/"verification.json",result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument("action",choices=["unit","case","run","verify"])
    p.add_argument("--case",default="baseline");p.add_argument("--seed",type=int,default=0);p.add_argument("--method",default=NEW,choices=[NEW,PROPOSED]);a=p.parse_args()
    if a.action=="unit":print(unit_checks());return
    if a.action=="case":run_case(a.case,a.seed,a.method);return
    if a.action=="verify":print(verify());return
    register()
    for case,seed,method in tasks():
        subprocess.run([sys.executable,str(Path(__file__)),"case","--case",case,"--seed",str(seed),"--method",method],cwd=ROOT,check=True)
    result=verify();write(OUT/"finished.json",dict(unix=time.time(),**result));state(OUT,stage="completed",outcomes=result["outcomes"]);print(result,flush=True)


if __name__=="__main__":main()
