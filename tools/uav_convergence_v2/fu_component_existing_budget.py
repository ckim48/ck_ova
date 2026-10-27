"""Isolate component fusion while retaining the existing composite-FU guard.

The preceding stricter pure-F guard produced infeasible arms. Preserve them;
this registered follow-up tests the update rule under the original guard scope.
"""
from pathlib import Path
import time
import os
from types import SimpleNamespace
import numpy as np
import fu_component_fusion as component
from fu_component_fusion import ROOT,SOURCE,DATASETS,PROPOSED,read,write,sha,link

STRICT=component.OUT
OUT=ROOT/"results/fu_component_existing_budget_v3_20261027"
VARIANTS={
    "fu_component_existing_guard":dict(label="CP-E",adaptive=False,request_lr=False,ideal=False,fusion=False,component_fusion=True),
    "fu_component_existing_guard_request_lr":dict(label="CP-E+RL",adaptive=False,request_lr=True,ideal=False,fusion=False,component_fusion=True),
}
ORIGINAL_PROJECT=component.project_components


def project_components(l,u,f,stats,engine):
    r,forget,stats=ORIGINAL_PROJECT(l,u,f,stats,engine)
    preservation=u-f
    # Same desired R'+F, expressed as (R'-P)+(F+P) for the original guard.
    adjusted_learning=r-preservation
    assert np.allclose(adjusted_learning+u,r+forget,rtol=1e-12,atol=1e-15)
    for row in stats:
        row["radio_component_definition"]="Transmit L_eff=R'-P and U=F+P; exact sum is still R'+F. Existing guard protects composite U, not pure F."
        row["component_scalar_ops"]+=2*l.shape[1]
    for user in range(len(l)):
        if np.any(f[user]) and getattr(engine,"work_started",None) is not None:
            engine.fu_control_ops[user]+=2*l.shape[1]
    return adjusted_learning,u,stats


def unit_checks():
    l=np.array([[-3.,1.],[2.,1.]])
    f=np.array([[1.,0.],[1.,0.]])
    u=f+np.array([[1.,0.],[0.,0.]])
    r,kept,rows=project_components(l,u,f,[dict(user=0),dict(user=1)],SimpleNamespace(work_started=None))
    assert np.array_equal(kept,u)
    assert np.allclose(r+kept,[[1.,1.],[3.,1.]])
    assert rows[0]["retention_forget_dot_after"]==0
    return dict(passed=True,projected_sum_preserved=True,original_composite_FU_radio_input_preserved=True)


def register():
    path=OUT/"protocol.json"
    if path.exists():
        p=read(path)
        for group in ["source_sha256","input_sha256"]:
            for name,digest in p[group].items():assert sha(ROOT/name)==digest,name
        return p
    strict=read(STRICT/"protocol.json");assert read(STRICT/"finished.json")["passed"]
    sources=dict(strict["source_sha256"]);sources[str(Path(__file__).relative_to(ROOT))]=sha(Path(__file__))
    inputs=dict(strict["input_sha256"])
    for ds in DATASETS:
        for method in strict["variants"]:
            p=STRICT/"cases/baseline"/(ds.lower()+"_seed0")/method/"outcome.json"
            inputs[str(p.relative_to(ROOT))]=sha(p)
    p=dict(registered_unix=time.time(),stage=3,new_arms=4,variants=VARIANTS,datasets=DATASETS,seed=0,
        reason="All four strict pure-forget guard arms stopped with no feasible design under the existing hard TX cap. Preserve their outcomes. Isolate the same local pure-forget projection under the existing composite-FU guard; no power/budget/noise coefficient is relaxed.",
        rule="Compute F=-eta_U*3grad(FM), P=-eta_U*grad(RM+3CM), R=L+P, R'=R-min(0,R dot F)/norm(F)^2 F. Present L_eff=R'-P and U=F+P to the existing guard. Their sum is exactly R'+F. The existing FU-relative error budget refers to composite U, and does not certify protection of pure F.",
        fixed="Same original model/data/seed, 200FL,45x5FU, original loss components, paired sampling, request teachers, original1MHz, hard0.1J cap, resource solver, geometry and energy parameters. CP-E uses the original FU LR; CP-E+RL uses original initial0.0035. No outcomes chosen or discarded.",
        accounting="Same feature/backprop counts; extra local projection/re-expression arithmetic is conservatively22*d+16 scalar MAC-equivalents per active device. Active-service communication+compute+hover scope. No measured-hardware claim.",
        limits="Exploratory follow-up on the same validation split, not an independent confirmatory test. Local pure-F directional property only; cross-client interference, retention and retrain matching need empirical evaluation. Reference200 never enters gradients or teacher construction.",
        source_sha256=sources,input_sha256=inputs)
    write(path,p);link(SOURCE/"prepared/airscape_seed0",OUT/"prepared/airscape_seed0");write(OUT/"unit_checks.json",unit_checks());return p


component.OUT=OUT;component.VARIANTS=VARIANTS
component.project_components=project_components;component.register=register;component.unit_checks=unit_checks
component.parent.OUT=OUT;component.parent.VARIANTS=VARIANTS;component.parent.register=register


def state(dest=OUT,**kw):write(Path(dest)/"status.json",dict(unix=time.time(),pid=os.getpid(),**kw))
component.state=state;component.parent.state=state


def main():
    import argparse,subprocess,sys
    p=argparse.ArgumentParser();p.add_argument("action",choices=["unit","run","variant","verify"]);p.add_argument("--variant",choices=list(VARIANTS));args=p.parse_args()
    if args.action=="unit":print(unit_checks());return
    if args.action=="variant":component.parent.run_variant(args.variant);return
    if args.action=="verify":print(component.verify());return
    register()
    for variant in VARIANTS:
        subprocess.run([sys.executable,str(Path(__file__)),"variant","--variant",variant],cwd=ROOT,check=True)
    result=component.verify();write(OUT/"finished.json",dict(unix=time.time(),**result));component.state(OUT,stage="completed",outcomes=result["outcomes"]);print(result,flush=True)


if __name__=="__main__":main()
