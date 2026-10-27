"""Stage 2: protect the pure forgetting component, not the composite FU vector.

This follow-up was chosen after the first pilot exposed that composite FU also
contains retention. Stage-1 outcomes remain unchanged and fully reported.
"""
import argparse
import hashlib
import inspect
import linecache
import math
from pathlib import Path
import subprocess
import sys
import textwrap
import time
import os

import numpy as np
import fu_joint_optimization as parent
from fu_joint_optimization import ROOT, SOURCE, DATASETS, PROPOSED, read, write, sha, link

FIRST=parent.OUT
OUT=ROOT/"results/fu_component_fusion_v2_20261027"
VARIANTS={
    "fu_component_project":dict(label="CP",adaptive=False,request_lr=False,ideal=False,fusion=False,component_fusion=True),
    "fu_component_project_request_lr":dict(label="CP+RL",adaptive=False,request_lr=True,ideal=False,fusion=False,component_fusion=True),
}


def project_components(l,u,f,stats,engine):
    retention=l+u-f
    projected=retention.copy();records={}
    for user in range(len(l)):
        ff=float(np.dot(f[user],f[user]))
        if ff<=0:continue
        before=float(np.dot(retention[user],f[user]))
        norm=float(np.linalg.norm(retention[user]))
        if before<0:projected[user]-=(before/ff)*f[user]
        after=float(np.dot(projected[user],f[user]))
        assert after>=-1e-10*max(norm*math.sqrt(ff),1e-30)
        assert np.linalg.norm(projected[user])<=norm*(1+1e-10)+1e-14
        records[user]=dict(component_projected=before<0,retention_forget_dot_before=before,
            retention_forget_dot_after=after,retention_forget_cosine_before=before/(norm*math.sqrt(ff)+1e-30),
            pure_forget_update_norm=math.sqrt(ff),old_composite_FU_norm=float(np.linalg.norm(u[user])),
            removed_retention_norm=float(np.linalg.norm(retention[user]-projected[user])),
            component_scalar_ops=20*l.shape[1]+16,
            component_definition="F=-eta_U*weight*3grad(FM); R=FL_update + FU_update - F; R'=R-min(0,R dot F)/norm(F)^2*F; radio aggregates R'+F and protects the F norm")
        if getattr(engine,"work_started",None) is not None:
            engine.fu_control_ops[user]+=records[user]["component_scalar_ops"]
    fs=f.sum(0);rs=retention.sum(0);ps=projected.sum(0)
    global_stats=dict(global_retention_forget_dot_before=float(np.dot(rs,fs)),
        global_retention_forget_dot_after=float(np.dot(ps,fs)))
    for row in stats:row.update(records[row["user"]],**global_stats)
    return projected,f,stats


def gradient(base,variant):
    return base.repeat.clone_function(base.clip.request_gradient,[
        ("g=WEIGHTS['forgetting']*gf+gr","g=WEIGHTS['forgetting']*gf+gr\n    engine.pure_fu_gradient=(WEIGHTS['forgetting']*gf).detach()",1),
        ("return g,dict(","return g,dict(FU_variant=VARIANT, ",1)],dict(VARIANT=variant))


def vectors(base,objective,variant):
    old=base.control.synchronized_vectors(objective)
    if objective!="cliperase":return old
    source=textwrap.dedent(inspect.getsource(old))
    rate="core.lr_at(21)" if VARIANTS[variant]["request_lr"] else "core.lr_at(21+fl)"
    changes=[("stats=[]","stats=[];pure=np.zeros_like(l)",1),
        ("u[user]+=(-core.lr_at(21+fl)*len(fids)/total*g[:-1]).cpu().numpy()",
         "u[user]+=(-RATE*len(fids)/total*g[:-1]).cpu().numpy()\n        pure[user]+=(-RATE*len(fids)/total*engine.pure_fu_gradient[:-1]).cpu().numpy()".replace("RATE",rate),1),
        ("stats.append(dict(request=rid","stats.append(dict(user=user,request=rid",1),
        ("FU_learning_rate=core.lr_at(21+fl)","FU_learning_rate="+rate,1),
        ("return l,u,stats","return project_components(l,u,pure,stats,engine)",1)]
    for a,b,count in changes:
        assert source.count(a)==count,(variant,a);source=source.replace(a,b)
    filename="/tmp/fu_component_vectors_"+hashlib.sha256(source.encode()).hexdigest()+".py"
    linecache.cache[filename]=(len(source),None,source.splitlines(True),filename)
    ns=dict(old.__globals__,request_gradient=gradient(base,variant),project_components=project_components)
    exec(compile(source,filename,"exec"),ns);return ns[old.__name__]


def unit_checks():
    from types import SimpleNamespace
    l=np.array([[-3.,1.],[2.,1.],[1.,2.]])
    f=np.array([[1.,0.],[1.,0.],[0.,0.]])
    # FU preservation contributes [+1,0] to the first device; R=[-2,1].
    u=f+np.array([[1.,0.],[0.,0.],[0.,0.]])
    r,kept,stats=project_components(l,u,f,[dict(user=0),dict(user=1)],SimpleNamespace(work_started=None))
    assert np.array_equal(r[0],[0.,1.]) and np.array_equal(kept,f)
    assert np.array_equal((r+kept)[1:],(l+u)[1:])
    assert np.dot(r[0]+kept[0],f[0])==np.dot(f[0],f[0])
    return dict(passed=True,pure_forget_unchanged=True,opposed_retention_removed=True,nonconflicting_sum_unchanged=True)


def register():
    path=OUT/"protocol.json"
    if path.exists():
        p=read(path)
        for group in ["source_sha256","input_sha256"]:
            for name,digest in p[group].items():assert sha(ROOT/name)==digest,name
        return p
    first=read(FIRST/"protocol.json");assert read(FIRST/"finished.json")["passed"]
    sources=dict(first["source_sha256"]);sources[str(Path(__file__).relative_to(ROOT))]=sha(Path(__file__))
    inputs=dict(first["input_sha256"])
    for dataset in DATASETS:
        for variant in first["variants"]:
            p=FIRST/"cases/baseline"/(dataset.lower()+"_seed0")/variant/"outcome.json";inputs[str(p.relative_to(ROOT))]=sha(p)
    p=dict(registered_unix=time.time(),stage=2,new_arms=4,variants=VARIANTS,datasets=DATASETS,seed=0,
        reason="User requested a specific fusion rule beyond plain L+U. Stage1 composite-direction projection did not address the distinction between pure forgetting and FU preservation; all first-stage outcomes are preserved. This is a subsequent exploratory hypothesis, not a preplanned independent confirmation.",
        rule="Each request's FU gradient is split into F=3grad(FM) and P=grad(RM+3CM). Apply exactly the existing sample weights and either original or initial0.0035 FU LR. At each device form signed retention update R=FL+FU_preservation and signed forgetting update F. If R dot F<0, replace R by R-(R dot F)/norm(F)^2 F. Send R'+F with original dimensionality. The FU-relative radio guard now protects the pure F component. No extra communication or encoder backpropagation.",
        scope="A one-sided PCGrad-inspired local operation. It preserves each local pure-forgetting direction to first order, but not necessarily global aggregate descent, finite-step retention, retrain matching or deletion. Different clients' gradients can conflict. Global dot products are reporting-only.",
        fixed="Same data, shared warm20, seed0, minibatch RNG, 200 FL rounds, 45 requests x5 FU steps, original FM/RM/CM terms, same request teacher definitions, 1MHz, geometry, head/resource optimizer, physical constraints and hard0.1J TX cap. Only fusion and (second arm) FU LR change.",
        accounting="Existing service compute+communication+hover model; charge extra local component decomposition/projection conservatively as20*d+16 scalar MAC-equivalents per active device. No hardware measurement. Global diagnostic calculations are not part of deployed control. The old FL_vector_norm field now denotes the projected retention component during joint updates; FU_vector_norm denotes pure forgetting.",
        reference_policy="Reference200 and validation are evaluation-only; no retrain teacher, oracle gradient, tuning on outcomes, seed selection, threshold adjustment or old-result replacement.",
        source_sha256=sources,input_sha256=inputs)
    write(path,p);link(SOURCE/"prepared/airscape_seed0",OUT/"prepared/airscape_seed0");write(OUT/"unit_checks.json",unit_checks());return p


def verify():
    r=ORIGINAL_VERIFY()
    for dataset in DATASETS:
        for method in VARIANTS:
            arm=OUT/"cases/baseline"/(dataset.lower()+"_seed0")/method
            if read(arm/"outcome.json")["status"]!="completed":continue
            for p in (arm/"work_attempts").glob("*.json"):
                w=read(p)
                for row in w["local"]:
                    assert row["retention_forget_dot_after"]>=-1e-12
                    assert row["pure_forget_update_norm"]>0
                if w["kind"]=="joint":assert w["radio"]["fu_noise_guard_active"]
    write(OUT/"verification.json",r);return r


ORIGINAL_VERIFY=parent.verify
parent.OUT=OUT;parent.VARIANTS=VARIANTS
parent.gradient=gradient;parent.vectors=vectors;parent.register=register;parent.verify=verify


def state(dest=OUT,**kw):write(Path(dest)/"status.json",dict(unix=time.time(),pid=os.getpid(),**kw))
parent.state=state


def main():
    parser=argparse.ArgumentParser();parser.add_argument("action",choices=["unit","run","variant","verify"]);parser.add_argument("--variant",choices=list(VARIANTS));args=parser.parse_args()
    if args.action=="unit":print(unit_checks());return
    if args.action=="variant":parent.run_variant(args.variant);return
    if args.action=="verify":print(verify());return
    register()
    for variant in VARIANTS:
        subprocess.run([sys.executable,str(Path(__file__)),"variant","--variant",variant],cwd=ROOT,check=True)
    result=verify();write(OUT/"finished.json",dict(unix=time.time(),**result));state(stage="completed",outcomes=result["outcomes"]);print(result,flush=True)


if __name__=="__main__":main()
