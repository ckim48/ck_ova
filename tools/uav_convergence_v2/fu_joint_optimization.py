"""Preregistered FU weighting/rate factorial and an exact-aggregate diagnostic.

These are exploratory adaptations, not a reproduction of EUPMU or LUR.
No retrain outputs, validation scores, or future requests enter an update.
"""
import argparse
import collections
import math
import os
import hashlib
import inspect
import linecache
import textwrap
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import learning_ablation as prior
from learning_ablation import ROOT, DATASETS, PROPOSED, read, write, sha, link, load_base
from airscape import extra, VideoData

OUT = ROOT / "results/fu_joint_optimization_v1_20261027"
SOURCE = prior.OUT
CASE = dict(id="baseline", axis=None, value=None)
VARIANTS = {
    "fu_balanced": dict(label="AW", adaptive=True, request_lr=False, ideal=False, fusion=False),
    "fu_request_lr": dict(label="RL", adaptive=False, request_lr=True, ideal=False, fusion=False),
    "fu_balanced_request_lr": dict(label="AW+RL", adaptive=True, request_lr=True, ideal=False, fusion=False),
    "fu_exact_aggregate": dict(label="Exact aggregate", adaptive=False, request_lr=False, ideal=True, fusion=False),
    "fu_direction_fusion": dict(label="DF", adaptive=False, request_lr=False, ideal=False, fusion=True),
    "fu_balanced_request_lr_fusion": dict(label="AW+RL+DF", adaptive=True, request_lr=True, ideal=False, fusion=True),
}
ACTIVE = None
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
    peak = max(float((p.analog_rms**2).max()) for p in parts)
    if peak>config["power_limit_w"]*(1+1e-6):
        REJECTIONS += 1
        write(OUT/"candidate_rejections"/f"{os.getpid()}_{REJECTIONS:05d}.json",
              dict(peak_w=peak, limit_w=config["power_limit_w"], reason="Existing receiver 1ppm tolerance"))
        return None
    return parts


def pool_init(cap):
    global ORIGINAL_EXACT
    import radio_parallel_cap
    radio_parallel_cap.CAP = cap
    ORIGINAL_EXACT = radio_parallel_cap.exact_aligned
    radio_parallel_cap.exact_aligned = strict_exact


def adaptive_mix(gf, gp):
    """Equalize gradient norms without differentiating through the weight.

    For nonzero F=3gf and P=gp, d=F+||F||/||P||*P has nonnegative
    dot products with both F and P. This is only a first-order local property;
    it is not a guarantee on finite-step losses, mR, or retrain KL.
    """
    import torch
    f = 3.*gf
    nf, np_ = float(f.norm()), float(gp.norm())
    weight = nf/np_ if nf>1e-12 and np_>1e-12 else (1. if nf<=1e-12 else 0.)
    direction = f+weight*gp
    fd, pd = float(torch.dot(f, direction)), float(torch.dot(gp, direction))
    assert fd>=-1e-6*max(nf*float(direction.norm()), 1e-12)
    assert pd>=-1e-6*max(np_*float(direction.norm()), 1e-12)
    return direction, dict(adaptive_preservation_weight=weight,
        effective_RM_weight=weight, effective_CM_weight=3*weight,
        weighted_FM_dot_direction=fd, preservation_dot_direction=pd,
        control_scalar_ops=32*gf.numel()+16,
        gradient_rule="F + norm(F)/norm(P) * P; F=3FM gradient, P=RM+3CM gradient")


def charge(engine, fids, count):
    if getattr(engine, "work_started", None) is not None:
        owners={engine.sample_owner[int(i)] for i in fids}
        assert len(owners)==1
        engine.fu_control_ops[next(iter(owners))] += count


def controlled_engine(parent):
    import accounting
    wrapped = ORIGINAL_INSTRUMENTED(parent)
    class Engine(wrapped):
        def start_work(self):
            super().start_work()
            self.fu_control_ops = [0]*10
        def finish_work(self):
            result = super().finish_work()
            result["base_device_macs"] = list(result["device_macs"])
            result["FU_control_scalar_ops"] = list(self.fu_control_ops)
            result["device_macs"] = [a+b for a,b in zip(result["device_macs"], self.fu_control_ops)]
            result["FU_control_cost_assumption"] = "Conservative 1 scalar operation = 1 MAC-equivalent at the existing modeled throughput; not a hardware measurement."
            return result
    return Engine


def gradient(base, variant):
    opt=VARIANTS[variant]
    edits=[]
    ns={}
    if opt["adaptive"]:
        edits += [("g=WEIGHTS['forgetting']*gf+gr",
                   "g,control_meta=adaptive_mix(gf,gr)\n    charge(engine,fids,control_meta['control_scalar_ops'])", 1),
                  ("return g,dict(", "return g,dict(**control_meta, ", 1),
                  ("WEIGHTS['forgetting']*fm+preservation",
                   "WEIGHTS['forgetting']*fm+control_meta['adaptive_preservation_weight']*preservation", 1)]
        ns.update(adaptive_mix=adaptive_mix, charge=charge)
    edits.append(("return g,dict(", "return g,dict(FU_variant=VARIANT, ", 1))
    ns["VARIANT"]=variant
    return base.repeat.clone_function(base.clip.request_gradient, edits, ns)


def vectors(base, objective, variant):
    original=base.control.synchronized_vectors(objective)
    if objective!="cliperase":
        return original
    edits=[]
    if VARIANTS[variant]["request_lr"]:
        edits += [("(-core.lr_at(21+fl)*len(fids)/total*g[:-1])", "(-core.lr_at(21)*len(fids)/total*g[:-1])", 1),
                  ("FU_learning_rate=core.lr_at(21+fl)", "FU_learning_rate=core.lr_at(21)", 1),
                  ("u[user]+=", "charge(engine,fids,8)\n        u[user]+=", 1)]
    if VARIANTS[variant]["fusion"]:
        edits += [("stats.append(dict(request=rid", "stats.append(dict(user=user,request=rid",1),
                  ("return l,u,stats", "return direction_fusion(l,u,stats,engine)",1)]
    # The mission runner clones the sampler once more to inject paired RNGs.
    # Register generated source so that its inspect-based clone remains valid.
    source=textwrap.dedent(inspect.getsource(original))
    for old,new,count in edits:
        assert source.count(old)==count,(variant,old)
        source=source.replace(old,new)
    filename="/tmp/fu_pilot_vectors_"+hashlib.sha256(source.encode()).hexdigest()+".py"
    linecache.cache[filename]=(len(source),None,source.splitlines(True),filename)
    ns=dict(original.__globals__,request_gradient=gradient(base,variant),charge=charge,direction_fusion=direction_fusion)
    exec(compile(source,filename,"exec"),ns)
    return ns[original.__name__]


def direction_fusion(l,u,stats,engine):
    """Local one-sided projection: preserve each active UAV's FU direction.

    l/u are already signed, sample-weighted, LR-scaled parameter updates.
    This does not require another client's vectors or an additional transmission.
    It does not imply that global sums have no cross-client interference.
    """
    result=l.copy();records={}
    for user in range(len(l)):
        uu=float(np.dot(u[user],u[user]))
        if uu<=0:
            continue
        before=float(np.dot(l[user],u[user]));nl=float(np.linalg.norm(l[user]))
        if before<0:
            result[user]-=(before/uu)*u[user]
        after=float(np.dot(result[user],u[user]))
        assert after>=-1e-10*max(nl*math.sqrt(uu),1e-30)
        assert np.linalg.norm(result[user])<=nl*(1+1e-10)+1e-14
        records[user]=dict(fusion_projected=before<0,FL_FU_dot_before=before,FL_FU_dot_after=after,
            FL_FU_cosine_before=before/(nl*math.sqrt(uu)+1e-30),
            removed_FL_component_norm=float(np.linalg.norm(l[user]-result[user])),
            fusion_rule="L' = L - min(0,L dot U)/norm(U)^2 * U; transmit L'+U",
            fusion_scalar_ops=16*l.shape[1]+16)
        if getattr(engine,"work_started",None) is not None:
            engine.fu_control_ops[user]+=records[user]["fusion_scalar_ops"]
    # Global quantities are reporting-only; the distributed rule above never uses them.
    us=u.sum(0);ls=l.sum(0);rs=result.sum(0)
    global_stats=dict(global_FL_FU_dot_before=float(np.dot(ls,us)),global_FL_FU_dot_after=float(np.dot(rs,us)))
    for row in stats:
        row.update(records[row["user"]],**global_stats)
    return result,u,stats


def radio(base, variant):
    original=base.base.parent.guard
    if not VARIANTS[variant]["ideal"]:
        return original
    class ExactAggregate:
        @staticmethod
        def analog(l,u,channels,c,seed):
            received,meta=original.analog(l,u,channels,c,seed)
            if received is None:
                return received,meta
            target=(l+u).sum(axis=0)
            meta.update(exact_aggregate_diagnostic=True,
                discarded_noisy_receiver_error_L2=float(np.linalg.norm(received-target)),
                original_realized_error_to_raw=meta.get("realized_error_to_raw"),
                realized_error_to_raw=0., applied_error_to_raw_L2=0.,
                diagnostic_scope="Replace only the applied received vector by the exact FL+FU sum. Keep feasible noisy-radio design, constraints and modeled service costs. Not a realizable noise-free system.")
            return target,meta
    return ExactAggregate


def make_arm(base,deployment,radios,cap):
    import worker
    control=SimpleNamespace(**base.control.__dict__)
    control.synchronized_vectors=lambda objective:vectors(base,objective,ACTIVE)
    adapted=SimpleNamespace(**dict(base.__dict__,control=control))
    # Correct the work-attempt rate label as well as actual FU vectors/local logs.
    import event_arm
    edits=[("FU_learning_rate=core.lr_at(21+fl) if ids else None",
            "FU_learning_rate=core.lr_at(21) if ids else None",1)] if VARIANTS[ACTIVE]["request_lr"] else []
    maker=base.repeat.clone_function(worker.make_arm,
        [("extra=[", "extra=RATE_EDITS+[",1)],dict(RATE_EDITS=edits,state=state))
    return maker(adapted,deployment,radios,cap)


def options(base):
    return dict(base.base.ARMS[PROPOSED], pilot_variant=ACTIVE, pilot_changes=VARIANTS[ACTIVE],
        FU_learning_rate_rule="constant original initial rate 0.0035 for each request's five steps" if VARIANTS[ACTIVE]["request_lr"] else "unchanged FL cosine schedule")


def component_checks(base,engine,plan,requests,warm,dd):
    import torch
    r=requests[0];pool=sorted(set(plan["partitions"][r["user"]])-set(plan["forgotten_images"]))
    ids=np.random.default_rng(910000+r["burst"]*1000+r["user"]*10).choice(pool,min(16,len(pool)),replace=False).astype(int).tolist()
    teacher=base.core.Teacher(engine,warm)
    old,old_meta=base.clip.request_gradient(engine,warm,r["images"],ids,0,teacher)
    new,meta=gradient(base,ACTIVE)(engine,warm,r["images"],ids,0,teacher)
    if not VARIANTS[ACTIVE]["adaptive"]:
        assert torch.equal(old,new)
    else:
        assert meta["weighted_FM_dot_direction"]>=0 and meta["preservation_dot_direction"]>=0
        assert abs(meta["adaptive_preservation_weight"]-1)>1e-4
    queue=base.RoundQueue(requests)
    a=base.control.synchronized_vectors("cliperase")(engine,warm,plan,queue,0.,0,"fl",[],None,{},True)[0]
    b=vectors(base,"cliperase",ACTIVE)(engine,warm,plan,queue,0.,0,"fl",[],None,{},True)[0]
    assert np.array_equal(a,b)
    write(dd/ACTIVE/"component_checks.json",dict(passed=True,training_only=True,
        FL_vectors_bit_identical=True,variant=ACTIVE,original_gradient_norm=float(old.norm()),
        candidate_gradient_norm=float(new.norm()),adaptive_weight=meta.get("adaptive_preservation_weight"),
        no_reference_or_validation_input=True))


def unit_checks():
    import torch
    # Opposed, aligned, oblique, and zero gradients: both first-order objectives.
    for gf,gp in [([1.,0.],[-2.,1.]),([1.,0.],[3.,0.]),([1.,2.],[2.,-1.]),([0.,0.],[1.,2.]),([1.,2.],[0.,0.])]:
        f,p=torch.tensor(gf,dtype=torch.double),torch.tensor(gp,dtype=torch.double)
        d,meta=adaptive_mix(f,p)
        assert torch.isfinite(d).all() and torch.dot(f,d)>=-1e-12 and torch.dot(p,d)>=-1e-12
    # The diagnostic must change only the applied received vector.
    fake=SimpleNamespace(base=SimpleNamespace(parent=SimpleNamespace(guard=SimpleNamespace(
        analog=lambda l,u,h,c,s:((l+u).sum(0)+.1,dict(realized_error_to_raw=.1,uplink_s=2.,energy_bound_j=3.))))))
    l=np.array([[1.,2.],[3.,4.]]);u=l*.1
    v,m=radio(fake,"fu_exact_aggregate").analog(l,u,None,None,None)
    assert np.array_equal(v,(l+u).sum(0)) and m["uplink_s"]==2. and m["energy_bound_j"]==3.
    l=np.array([[-2.,1.],[2.,1.],[1.,2.]])
    u=np.array([[1.,0.],[1.,0.],[0.,0.]])
    new,unchanged,rows=direction_fusion(l,u,[dict(user=0),dict(user=1)],SimpleNamespace(work_started=None))
    assert np.array_equal(unchanged,u) and np.array_equal(new[0],[0.,1.])
    assert np.array_equal(new[1:],l[1:])
    return dict(passed=True,tests=["two-objective first-order directions including opposed/zero cases","exact-sum diagnostic preserves radio costs","local fusion removes only opposed FL component and preserves FU/nonconflicting updates"])


def register():
    p=OUT/"protocol.json"
    if p.exists():
        protocol=read(p)
        for group in ["source_sha256","input_sha256"]:
            for name,digest in protocol[group].items():
                assert sha(ROOT/name)==digest,name
        return protocol
    parent=prior.register()
    assert not list((OUT/"cases").glob("*/*/*/outcome.json"))
    sources=dict(parent["source_sha256"])
    sources[str(Path(__file__).relative_to(ROOT))]=sha(Path(__file__))
    inputs={}
    for dataset in DATASETS:
        d=SOURCE/"cases/baseline"/(dataset.lower()+"_seed0")
        for name in ["plan.json","requests.json","initial20/final.pt","reference200/final.pt","proxy20/final.pt","feature_cache/features.pt","reference_metrics.json","reference_mia.json",PROPOSED+"/manifest.json",PROPOSED+"/outcome.json"]:
            f=d/name;inputs[str(f.relative_to(ROOT))]=sha(f)
    protocol=dict(registered_unix=time.time(),datasets=DATASETS,case=CASE,variants=VARIANTS,new_arms=12,seed=0,
        adaptive_rule="F=3*grad(FM), P=grad(RM+3CM); use F+lambda*P, lambda=norm(F)/norm(P), detached per local FU step. Degenerate zero norms handled explicitly. Equal gradient influence gives nonnegative first-order dot products with both objectives, not a guarantee on mR/retrain-KL/finite steps.",
        rate_rule="Request-relative constant rate: all five FU steps use original initial rate core.lr_at(21)=0.0035; original FL schedule unchanged. No hyperparameter sweep.",
        diagnostic="Exact aggregate replaces only the applied noisy receiver vector; existing radio feasibility, hard cap, scheduler, head selection and accounting remain. Resource choices can respond to the different model trajectory. A hypothetical diagnostic, not a deployable baseline.",
        local_fusion="Added before any experiment in response to the user's explicit request to replace plain L+U. At each UAV, project only the component of its LR-scaled/sample-weighted L update opposed to its U update: L'=L-min(0,L dot U)/norm(U)^2 U. Transmit L'+U with the same dimension and radio. U is the composite FU-objective update, not a pure FM gradient. A one-sided PCGrad-inspired adaptation; local first-order directional protection only, no global/deletion/mR guarantee. Compare DF alone and AW+RL+DF with their matched no-DF controls.",
        fixed="Same shared warm20 checkpoint, data, train/validation split, seed0, minibatch RNG, request teachers, 200 FL rounds, 45 requests x 5 FU steps, routes, channels, 1MHz, optimized head/resources, guard, hard 0.1J TX cap and energy coefficients.",
        evaluation="Exploratory single-seed paired pilot on existing validation split. Evaluate retained mR/loss, mean/worst forget KL, MIA gap, service energy/delay. Reference200 is reporting-only, never a teacher or gradient input. Report strict simultaneous improvement and utility tradeoffs separately; no outcome-based selection or changing thresholds.",
        accounting="Charge adaptive-control scalar arithmetic conservatively as 32*d+16 extra MAC-equivalent operations per local step; request-rate control adds8; local fusion adds16*d+16 per active device. Existing backbone/adapters/teacher feature costs unchanged. Use existing active FL/FU compute+communication+hover scope, excluding travel/collection/prefix. Global conflict diagnostics are simulation-only reporting, not used by the distributed algorithm. This is modeled, not measured hardware energy.",
        references=[dict(title="CLIPErase, ACL2025",url="https://aclanthology.org/2025.acl-long.1469/"),dict(title="EUPMU, NeurIPS2025",url="https://proceedings.neurips.cc/paper_files/paper/2025/hash/06fc38f5c21ae66ef955e28b7a78ece5-Abstract-Conference.html"),dict(title="PCGrad, NeurIPS2020",url="https://proceedings.neurips.cc/paper_files/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html")],
        attribution="The norm-balancing adaptation is this pilot's explicit rule, not claimed as an exact implementation or theoretical guarantee from the cited papers.",
        source_sha256=sources,input_sha256=inputs)
    write(p,protocol);link(SOURCE/"prepared/airscape_seed0",OUT/"prepared/airscape_seed0")
    write(OUT/"unit_checks.json",unit_checks())
    return protocol


def run_variant(variant):
    global ACTIVE, ORIGINAL_INSTRUMENTED
    ACTIVE=variant
    register()
    _,base=load_base()
    import worker,heuristic,accounting
    extra.Data=VideoData;heuristic.pool_init=pool_init
    ORIGINAL_INSTRUMENTED=accounting.instrumented_engine
    accounting.instrumented_engine=controlled_engine
    fn=base.repeat.clone_function(worker.run,[
        ("arm=make_arm(base,deployment,radios,cap);results=[]", "radios[VARIANT]=radio(base,VARIANT)\n    arm=make_arm(base,deployment,radios,cap);results=[]",1),
        ("opt=base.base.ARMS['proposed_fu_guard_cliperase' if method=='heuristic_ova' else method]", "opt=options(base)\n                    component_checks(base,engine,plan,requests,warm,dd)",1)],
        dict(OUT=OUT,EXTRA_DATASETS=["AirScape"],VARIANT=variant,radio=radio,make_arm=make_arm,options=options,component_checks=component_checks,state=state))
    fn(CASE,DATASETS,[variant])
    for dataset in DATASETS:
        d=OUT/"cases/baseline"/(dataset.lower()+"_seed0")
        for name in ["plan.json","requests.json"]:
            assert read(d/name)==read(SOURCE/"cases/baseline"/d.name/name)
    print(dict(variant=variant,outcomes={d:read(OUT/"cases/baseline"/(d.lower()+"_seed0")/variant/"outcome.json")["status"] for d in DATASETS}),flush=True)


def verify():
    register();rows=[]
    for dataset in DATASETS:
        dd=OUT/"cases/baseline"/(dataset.lower()+"_seed0")
        for method,variant in VARIANTS.items():
            arm=dd/method;o=read(arm/"outcome.json")
            row=dict(dataset=dataset,method=method,status=o["status"],wall_seconds=o["wall_seconds"])
            assert read(arm/"component_checks.json")["passed"]
            if o["status"]=="completed":
                check=read(arm/"verification.json");assert check["FL_rounds"]==200 and check["FU_steps"]==225
                tx=[read(p) for p in sorted((arm/"transactions").glob("*.json"))]
                local=[x for t in tx for x in t["local"]]
                assert len(local)==225 and {t["kind"] for t in tx}=={"fl","joint"}
                for x in local:
                    assert x["FU_variant"]==method
                    if variant["request_lr"]:assert x["FU_learning_rate"]==.0035
                    else:assert x["FU_learning_rate"]==x["FL_schedule_learning_rate"]
                    if variant["adaptive"]:
                        assert x["adaptive_preservation_weight"]>=0
                        assert x["weighted_FM_dot_direction"]>=-1e-7 and x["preservation_dot_direction"]>=-1e-7
                        w=x["adaptive_preservation_weight"]
                        assert math.isclose(x["objective"],3*x["FM"]+w*(x["RM"]+3*x["CM"]),rel_tol=1e-5,abs_tol=1e-6)
                    if variant["fusion"]:
                        assert x["FL_FU_dot_after"]>=-1e-12
                for t in tx:
                    if variant["ideal"]:assert t["radio"]["applied_error_to_raw_L2"]==0
                row.update(FL_rounds=200,FU_steps=225,transactions=len(tx),
                    median_adaptive_weight=float(np.median([x["adaptive_preservation_weight"] for x in local])) if variant["adaptive"] else None)
            rows.append(row)
    result=dict(passed=all(r["status"]!="failed" for r in rows),rows=rows,outcomes=dict(collections.Counter(r["status"] for r in rows)))
    write(OUT/"verification.json",result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument("action",choices=["run","variant","verify","unit"]);p.add_argument("--variant",choices=list(VARIANTS));args=p.parse_args()
    if args.action=="unit":print(unit_checks());return
    if args.action=="variant":run_variant(args.variant);return
    if args.action=="verify":print(verify());return
    register()
    for variant in VARIANTS:
        subprocess.run([sys.executable,str(Path(__file__)),"variant","--variant",variant],cwd=ROOT,check=True)
    result=verify();write(OUT/"finished.json",dict(unix=time.time(),**result));state(stage="completed",outcomes=result["outcomes"]);print(result,flush=True)


if __name__=="__main__":main()
