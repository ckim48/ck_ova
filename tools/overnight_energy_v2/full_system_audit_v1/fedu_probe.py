"""Numerical audit only. Does not manufacture a missing FedU IAF loss."""
from pathlib import Path
import sys,json,time,fcntl,traceback
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'tools/overnight_energy_v2/full_system_baselines_v1'),str(ROOT)]
import run_memprune as run
import numpy as np


def main():
    import torch
    from train import Engine
    from dataset import RetrievalData
    from objectives import contrastive
    dest=run.OUT/'fedu_source_probe';dest.mkdir(exist_ok=True)
    lock=(ROOT/'results/overnight_energy_v2/association_objective_gpu.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=json.loads((run.core.SOURCE/'plan.json').read_text())
    records=json.loads((run.core.SOURCE/'records.json').read_text());rec=records[20]
    engine=Engine(RetrievalData(),plan)
    theta=run.core.load_theta(run.PREFIX/'round_020.pt',engine.device)
    user=next(u for u,f in enumerate(rec['forget']) if len(f)>=4)
    ids=rec['retained'][user][:4];cids=engine.data.training_caption_ids(ids,21)
    def loss(x):
        im,tx,s=engine.features(x,ids,cids);ix=torch.tensor(ids,device=engine.device)
        return contrastive(im,tx,s,ix,ix)
    def hvp(vec):
        x=theta.detach().requires_grad_(True);g=torch.autograd.grad(loss(x),x,create_graph=True)[0]
        result=torch.autograd.grad(g@vec,x)[0].detach();result[-1]=0;return result
    torch.manual_seed(821);direction=torch.randn_like(theta);direction[-1]=0;direction/=direction.norm()
    exact=hvp(direction);errors=[]
    for eps in [.01,.003,.001]:
        gs=[]
        for sign in [-1,1]:
            x=(theta+sign*eps*direction).detach().requires_grad_(True)
            g=torch.autograd.grad(loss(x),x)[0].detach();g[-1]=0;gs.append(g)
        fd=(gs[1]-gs[0])/(2*eps)
        errors.append(dict(epsilon=eps,relative_error=float((exact-fd).norm()/exact.norm().clamp_min(1e-15))))
    gf,_=engine.gradient(theta,rec['forget'][user][:4],21)
    v=gf.clone();history=[]
    for k in range(9):
        history.append(dict(iteration=k,norm=float(v.norm()),relative_linear_residual=float((hvp(v)-gf).norm()/gf.norm().clamp_min(1e-15))))
        v=gf+v-hvp(v)
        if not torch.isfinite(v).all():break
    result=dict(status='numerical_diagnostic_completed',user=user,training_only_batches=True,
        hvp_finite_difference=errors,unmodified_neumann_recurrence=history,
        full_fedu_training_launched=False,ranking_permitted=False,
        source_gap='Paper Eq.(7) defines L_IAF using argmin rather than a differentiable scalar; Eq.(10) needs its gradient. No official code found to resolve how Eq.(6) enters it. No arbitrary surrogate substituted.',
        caveat='One local contrastive-loss Hessian probe; convergence or divergence here is not a theorem about all FedU settings.')
    run.core.write(dest/'diagnostics.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    try:main()
    except Exception as e:
        run.core.write(run.OUT/'fedu_source_probe/failure.json',dict(error=repr(e),ranking_permitted=False));traceback.print_exc();raise
