"""Optimizer/aggregation-matched retain-only reference for the MemPrune pilot.

No oracle parameters feed the candidate. Ideal-link reference energy is not
included in competitive energy rankings. A 20-round matching prior reference
is reused; all subsequent rounds use the candidate's local Adam procedure.
"""
from pathlib import Path
import sys,json,time,fcntl,traceback
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'tools/overnight_energy_v2/full_system_baselines_v1'),str(ROOT)]
import run_memprune as run
import numpy as np


def update_report(plan,ref):
    from scipy.special import logsumexp
    gallery=plan['gallery_images'];forget=plan['forget_probe']
    positions={i:j for j,i in enumerate(gallery)}
    records=json.loads((run.core.SOURCE/'records.json').read_text())
    groups={'all':forget,**{str(u):sorted(set(f)&set(forget)) for u,f in enumerate(records[20]['forget']) if f}}
    rows=[];hits=[];target=None
    for p in sorted((run.DEST/'rounds').glob('*.json')):
        row=json.loads(p.read_text());k=row['round']
        path=run.DEST/f'probabilities_{k:03d}.npz'
        if not path.exists():continue
        values=np.load(path);li=values['log_image'];lt=values['log_text'];out={}
        for name,ids in groups.items():
            if not ids:continue
            ix=np.array([positions[i] for i in ids]);ci=(5*ix[:,None]+np.arange(5)).reshape(-1)
            ki=(np.exp(ref[0][ix])*(ref[0][ix]-li[ix])).sum(1).mean()
            kt=(np.exp(ref[1][ci])*(ref[1][ci]-lt[ci])).sum(1).mean()
            out[name]=dict(images=len(ids),KL_retrain_to_candidate=max(0.,float((ki+kt)/2)),
                i2t_top1_disagreement=float(np.mean(ref[0][ix].argmax(1)!=li[ix].argmax(1))),
                t2i_top1_disagreement=float(np.mean(ref[1][ci].argmax(1)!=lt[ci].argmax(1))))
        worst=max(v['KL_retrain_to_candidate'] for key,v in out.items() if key!='all')
        ok=k>=25 and row['evaluation']['retained']['mR']>=.29304036883016427 and worst<=.01
        hits=hits+[row] if ok else []
        if target is None and len(hits)>=3:
            target=dict(entry_round=hits[0]['round'],confirmation_round=k,energy_j=row['cumulative_tx_energy_bound_j'],uplink_s=row['cumulative_uplink_s'])
        rows.append(dict(round=k,metrics=out,retained=row['evaluation']['retained'],energy_j=row['cumulative_tx_energy_bound_j'],uplink_s=row['cumulative_uplink_s']))
    run.core.write(run.DEST/'matched_reference_evaluation.json',dict(method='FedMemPrune VLM/OFDMA adaptation',
        reference='Same local optimizer/aggregation without pruning, retain-only from initialization; ideal radio',
        target=target,rows=rows,scope='Single preregistered rho=.4 and seed0; full external-method ranking not available.'))


def main():
    import torch
    from train import Engine,update_theta
    from dataset import RetrievalData
    dest=run.OUT/'fedmemprune_matched_reference';dest.mkdir(exist_ok=True)
    own=(dest/'launch.lock').open('a');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
    lock=(ROOT/'results/overnight_energy_v2/association_objective_gpu.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (dest/'started.json').exists(),'Duplicate reference launch'
    run.core.write(dest/'started.json',dict(unix=time.time(),source_hash=run.core.sha(Path(__file__))))
    plan=json.loads((run.core.SOURCE/'plan.json').read_text());records=json.loads((run.core.SOURCE/'records.json').read_text())
    engine=Engine(RetrievalData(),plan);theta=run.core.load_theta(run.core.SOURCE/'retrain/round_020.pt',engine.device)
    forbidden=set(plan['forgotten_images']);forget=records[20]['forget']
    run.core.write(dest/'protocol.json',dict(start='retained-only original matched 20-SGD-step prefix',
        prefix_sha256=run.core.sha(run.core.SOURCE/'retrain/round_020.pt'),round21='No parameter change; matches candidate gradient-selection transaction',
        local_optimizer='Adam 1e-4; 3 full local epochs; batch16',aggregation='uniform mean of 10 client model deltas',
        physical_link='ideal evaluation reference, no competitive energy claim',forbidden_images=len(forbidden)))
    for k in range(22,201):
        t0=time.time();updates=[]
        for u,ids in enumerate(records[k-1]['retained']):
            remaining=[i for i in ids if i not in forbidden];assert not (set(remaining)&forbidden)
            run.status(stage='memprune_matched_reference',round=k,user=u)
            delta,_=run.local_adam(engine,theta,remaining,k,u);updates.append(delta)
        theta=(theta+torch.stack(updates).mean(0)).detach()
        if k%5==0 or k==200:
            metrics,probs=engine.evaluate(theta)
            run.core.save_theta(dest/f'round_{k:03d}.pt',theta,k)
            np.savez_compressed(dest/f'probabilities_{k:03d}.npz',log_image=probs[0].numpy(),log_text=probs[1].numpy())
            run.core.write(dest/'rounds'/f'{k:03d}.json',dict(round=k,evaluation=metrics,wall_s=time.time()-t0))
        print(json.dumps(dict(stage='reference',round=k,wall_s=time.time()-t0)),flush=True)
    mia,scores=engine.mia_scores(theta);run.core.write(dest/'mia.json',mia)
    np.save(dest/'mia_scores.npy',scores)
    update_report(plan,[x.numpy() for x in probs]);run.core.write(dest/'finished.json',dict(unix=time.time(),rounds=200))
    run.status(stage='memprune_and_matched_reference_completed',round=200)


if __name__=='__main__':
    try:main()
    except Exception as e:run.status(stage='matched_reference_failed',error=repr(e));traceback.print_exc();raise
