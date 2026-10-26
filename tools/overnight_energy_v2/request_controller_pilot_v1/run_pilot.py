"""Preregistered ideal-link pilot of receiver-side request constraints.

Preserves frozen v2/v3 sources and results. No test-set access or radio ranking.
"""
from pathlib import Path
import sys, json, hashlib, time, os, fcntl, traceback
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FROZEN = ROOT / 'tools/overnight_energy_v2/retrieval_fourway_v2'
SOURCE = ROOT / 'results/overnight_energy_v2/retrieval_long200_v3'
OUT = ROOT / 'results/overnight_energy_v2/request_controller_pilot_v1'
sys.path[:0] = [str(FROZEN), str(ROOT), str(ROOT/'tools/overnight_energy_v2'),
               str(ROOT/'tools/overnight_energy_v2/initialization_v11'),
               str(ROOT/'tools/overnight_energy_v2/receiver_control_v1')]
ARMS = ['baseline_sum', 'aggregate_qp', 'request_qp', 'retain_only_control']
MARGIN = .1
TRUST = 2.


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def status(**x):
    write(OUT/'status.json', dict(unix=time.time(), pid=os.getpid(), **x))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def project(d0, r, hs, beta, lr, mode, margin=MARGIN, trust=TRUST):
    import cvxpy as cp
    hs = np.asarray(hs, dtype=np.float64)
    weighted = beta @ hs
    if mode == 'aggregate_qp':
        hs_used = weighted[None, :]
        delta = margin*lr*np.sum(hs_used**2, axis=1)
    else:
        hs_used = hs
        delta = margin*lr*beta*np.sum(hs**2, axis=1)
    raw_a = np.vstack([r, hs_used]); raw_b = np.r_[0., -delta]
    norms = np.linalg.norm(raw_a, axis=1)
    active = norms > 1e-14
    if np.any((~active) & (raw_b < 0)):
        return None, dict(status='zero_gradient_positive_margin')
    a = raw_a[active]/norms[active, None]
    scale = max(float(np.linalg.norm(d0)), 1e-12)
    b = raw_b[active]/norms[active]/scale
    _, s, vt = np.linalg.svd(a, full_matrices=False)
    rank = int(np.sum(s > max(s[0]*1e-10, 1e-14)))
    basis = vt[:rank]
    y0 = basis @ (d0/scale)
    perp = d0/scale-basis.T @ y0
    assert np.linalg.norm(perp) < 1e-7, 'Nominal direction outside constraint span'
    y = cp.Variable(rank)
    constraints = [a @ basis.T @ y <= b, cp.norm(y, 2) <= trust]
    problem = cp.Problem(cp.Minimize(.5*cp.sum_squares(y-y0)), constraints)
    problem.solve(solver='CLARABEL', tol_gap_abs=1e-10, tol_feas=1e-10,
                  tol_gap_rel=1e-10, max_iter=300)
    diag = dict(status=problem.status, nominal_norm=scale, rank=rank,
                max_nominal_normalized_violation=float(np.max(a @ (d0/scale)-b)),
                margin_fraction=margin, trust_multiplier=trust)
    if problem.status not in ['optimal', 'optimal_inaccurate']:
        return None, diag
    direction = scale*(basis.T @ y.value)
    violation = float(np.max(a @ (direction/scale)-b))
    diag.update(max_normalized_violation=violation,
                norm_ratio=float(np.linalg.norm(direction)/scale),
                correction_ratio=float(np.linalg.norm(direction-d0)/scale),
                retained_first_order_change=float(r @ direction),
                request_first_order_changes=(hs @ direction).tolist(),
                request_target_changes=(-margin*lr*beta*np.sum(hs**2,axis=1)).tolist())
    if violation > 2e-6 or diag['norm_ratio'] > trust+2e-6:
        diag['status'] = 'rejected_numerical_violation'; return None, diag
    return direction, diag


def cpu_checks():
    # Same aggregate, different individual feasible optimum; includes norm trust region.
    d0=np.array([-1.,-1.]); r=np.ones(2)
    first,_=project(d0,r,np.array([[1.,0.],[0.,1.]]),np.array([.5,.5]),1.,'request_qp',margin=1.)
    assert np.allclose(first,d0,atol=1e-6)
    # Explicit normalized margin convention differs from the illustrative constant-margin note.
    second,diag=project(d0,r,np.array([[2.,0.],[-1.,1.]]),np.array([.5,.5]),1.,'request_qp',margin=.1)
    assert second is not None and not np.allclose(second,d0,atol=1e-5)
    assert diag['max_normalized_violation'] < 2e-6
    failed,_=project(d0,r,np.array([[1.,0.],[-1.,0.]]),np.array([.5,.5]),1.,'request_qp',margin=.1)
    assert failed is None
    return dict(identity_feasible=True, per_request_information_matters=True,
                opposite_request_infeasibility_detected=True)


def collect(engine, theta, rec, k, forget):
    count=sum(map(len,rec['retained'])); nf=sum(map(len,forget))
    lr=engine.plan['learning_rate']; learning=[]; unlearning=[]; hs=[]; beta=[]; stats=[]
    for u,ids in enumerate(rec['batches']):
        g,loss=engine.gradient(theta,ids,k)
        learning.append((-lr*len(rec['retained'][u])/count*g[:-1]).cpu().numpy().astype(float))
        uu=np.zeros(len(theta)-1); row=dict(user=u, retained_loss=loss)
        if forget[u]:
            fids=np.random.default_rng(9000+k*10+u).choice(forget[u],min(16,len(forget[u])),replace=False).astype(int).tolist()
            fg,floss=engine.gradient(theta,fids,k)
            rg,kl=engine.gradient(theta,ids,k,'relation')
            # Exactly the existing proposed local objective, with minimization sign.
            h=-engine.plan['forget_weight']*fg+engine.plan['crossmodal_weight']*rg
            b=len(forget[u])/nf
            uu=(-lr*b*h[:-1]).cpu().numpy().astype(float)
            hs.append(h[:-1].cpu().numpy().astype(float)); beta.append(b)
            row.update(forget_CE=floss, retained_relation_KL=kl,
                       request_objective=-floss+kl, forget_batch=fids,
                       retained_batch=ids, beta=b)
        unlearning.append(uu);stats.append(row)
    learning=np.asarray(learning);unlearning=np.asarray(unlearning)
    r=-learning.sum(0)/lr
    return learning, unlearning, r, np.asarray(hs), np.asarray(beta), stats


def evaluate(engine, theta, final_reference, horizon_reference, requests):
    import torch
    evaluation,prob=engine.evaluate(theta,final_reference)
    plan=engine.plan; gallery=plan['gallery_images']
    def extra(ids,reference):
        ids=[i for i in ids if i in gallery]
        if not ids:return dict(images=0)
        ii=torch.tensor([gallery.index(i) for i in ids]); ci=(ii[:,None]*5+torch.arange(5)).flatten()
        oldi,oldt=reference; li,lt=prob
        ki=(oldi[ii].exp()*(oldi[ii]-li[ii])).sum(1).mean()
        kt=(oldt[ci].exp()*(oldt[ci]-lt[ci])).sum(1).mean()
        return dict(images=len(ids),KL_retrain_to_candidate=max(0.,float((ki+kt)/2)),
                    i2t_top1_disagreement=float((oldi[ii].argmax(1)!=li[ii].argmax(1)).float().mean()),
                    t2i_top1_disagreement=float((oldt[ci].argmax(1)!=lt[ci].argmax(1)).float().mean()))
    evaluation['matched_horizon_retrain']={name:extra(ids,horizon_reference) for name,ids in
                    [('retained',plan['validation_images']),('forget',plan['forget_probe'])]}
    evaluation['per_request_final_retrain']={str(u):extra(ids,final_reference) for u,ids in enumerate(requests) if ids}
    return evaluation,prob


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    write(OUT/'cpu_checks.json',cpu_checks())
    if '--cpu-only' in sys.argv:
        print('CPU checks passed',flush=True);return
    lock=(ROOT/'results/overnight_energy_v2/association_objective_gpu.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'started.json').exists(),'Refuse duplicate run'
    import torch
    from train import Engine, update_theta
    from dataset import RetrievalData
    plan=json.loads((SOURCE/'plan.json').read_text())
    for name,digest in plan['sources'].items():assert sha(ROOT/name)==digest,name
    for name,digest in plan['frozen_engine_hashes'].items():assert sha(FROZEN/name)==digest,name
    records=json.loads((SOURCE/'records.json').read_text()); forget=records[20]['forget']
    protocol=dict(version='request_controller_pilot_v1',arms=ARMS,seed=0,
        common_checkpoint=str(SOURCE/'proposed/round_020.pt'),checkpoint_sha256=sha(SOURCE/'proposed/round_020.pt'),
        rounds=[21,22,23,24,25],evaluations=[20,21,25],learning_rate=.0035,
        margin_fraction=MARGIN,retained_first_order_budget=0.,trust_multiplier=TRUST,
        objective='Same frozen v3 proposed objective: J_u=-forget contrastive CE + retained relation KL to pre-request teacher. No Hessian P.',
        interpretation='Algorithm-only ideal-link pilot; no claim about radio energy or NOMA advantage. Same local objectives and batches across algorithm arms.',
        control='retain_only_control deliberately omits unlearning and is not an eligible successful deletion method.',
        thresholds='Fixed before running, no parameter sweep. Infeasible branches stop and remain incomplete.',
        reference='Final200 retrain and matched-horizon retrain used only for reporting, never as training targets.',
        test_policy='Official test images never evaluated or used for this pilot.',
        source_sha256=sha(__file__),frozen_engine_hashes=plan['frozen_engine_hashes'])
    write(OUT/'protocol.json',protocol);write(OUT/'started.json',dict(pid=os.getpid(),unix=time.time()))
    status(stage='loading_model')
    engine=Engine(RetrievalData(),dict(plan,learning_rate=.0035))
    theta=torch.load(SOURCE/'proposed/round_020.pt',map_location='cpu',weights_only=True)['theta'].to(engine.device)
    engine.set_teacher(theta)
    def reference(k):
        z=np.load(SOURCE/f'retrain/probabilities_{k:03d}.npz')
        return tuple(torch.from_numpy(z[key].copy()) for key in ['log_image','log_text'])
    final_reference=reference(200)
    initial,_=evaluate(engine,theta,final_reference,reference(20),forget)
    write(OUT/'initial_evaluation.json',initial)
    status(stage='collecting_common_round21_gradients')
    shared=collect(engine,theta,records[20],21,forget)
    l,u,r,hs,beta,stats=shared
    original_l,original_u,_=engine.updates(theta,records[20],21,'proposed',forget)
    assert np.allclose(l,original_l,rtol=1e-6,atol=1e-10)
    assert np.allclose(u,original_u,rtol=1e-6,atol=1e-10)
    write(OUT/'frozen_objective_equivalence.json',dict(passed=True,
        max_learning_difference=float(np.max(np.abs(l-original_l))),
        max_unlearning_difference=float(np.max(np.abs(u-original_u)))))
    np.savez_compressed(OUT/'common_gradients_round21.npz',learning=l,unlearning=u,r=r,hs=hs,beta=beta)
    states={arm:theta.clone() for arm in ARMS}; alive={arm:True for arm in ARMS}
    results={}
    # First compare every arm on exactly the same actual checkpoint and gradients.
    # Then continue each branch with its own recomputed gradients through round25.
    schedule=[(arm,21) for arm in ARMS]+[(arm,k) for arm in ARMS for k in range(22,26)]
    for arm,k in schedule:
        if not alive[arm]:continue
        t0=time.time();status(stage='running',arm=arm,round=k)
        parts=shared if k==21 else collect(engine,states[arm],records[k-1],k,forget)
        l,u,r,hs,beta,stats=parts;d0=(l+u).sum(0)
        if arm=='baseline_sum':d=d0;diag=dict(status='unmodified_sum')
        elif arm=='retain_only_control':d=l.sum(0);diag=dict(status='no_unlearning_control')
        else:d,diag=project(d0,r,hs,beta,.0035,arm)
        if d is None:
            alive[arm]=False;results[arm]=dict(status='incomplete_controller_infeasible',failed_round=k,diagnostic=diag)
            write(OUT/arm/'manifest.json',results[arm]);continue
        diag.update(retained_first_order_change=float(r @ d),
                    request_first_order_changes=(hs @ d).tolist(),
                    request_objectives_worsen_first_order=int(np.sum(hs @ d>1e-10)))
        states[arm]=update_theta(states[arm],d)
        row=dict(arm=arm,round=k,controller=diag,local=stats)
        dest=OUT/arm;dest.mkdir(exist_ok=True)
        if k in [21,25]:
            evaluation,prob=evaluate(engine,states[arm],final_reference,reference(k),forget)
            row['evaluation']=evaluation
            np.savez_compressed(dest/f'probabilities_{k:03d}.npz',log_image=prob[0].numpy(),log_text=prob[1].numpy())
        torch.save(dict(theta=states[arm].cpu(),round=k),dest/f'round_{k:03d}.pt')
        row['wall_s']=time.time()-t0;write(dest/f'round_{k:03d}.json',row)
        print(json.dumps(dict(arm=arm,round=k,wall_s=row['wall_s'],controller=diag,evaluation=row.get('evaluation'))),flush=True)
        if k==25:
            mia,_=engine.mia_scores(states[arm]);write(dest/'mia.json',mia)
            results[arm]=dict(status='completed_pilot',evaluation=row['evaluation'],mia=mia)
            write(dest/'manifest.json',dict(status='completed_pilot',rounds=5))
            write(OUT/'summary.json',dict(protocol=protocol,initial=initial,results=results))
    # Verify no source mutation occurred during the pilot.
    for name,digest in plan['frozen_engine_hashes'].items():assert sha(FROZEN/name)==digest,name
    write(OUT/'summary.json',dict(protocol=protocol,initial=initial,results=results))
    status(stage='completed',arms=list(results));print('PILOT_COMPLETED',flush=True)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        status(stage='failed',error=repr(exc));traceback.print_exc();raise
