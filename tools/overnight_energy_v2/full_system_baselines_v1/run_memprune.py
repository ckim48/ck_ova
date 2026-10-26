"""FedMemPrune VLM/OFDMA adaptation, not the original CNN experiment.

Algorithm 1: retained gradient upload -> lowest-|mean gradient| reset ->
retained-data FL. Native Adam 1e-4 / three full local epochs are preserved.
The established digital 20-round initialization is reused and fully charged.
Source/pilot gates must pass before continuing to the 200 communication limit.
"""
from pathlib import Path
import sys,os,json,time,hashlib,fcntl,traceback,argparse,importlib.util
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'tools/overnight_energy_v2/ofdma_v1'))
import radio_ofdma as radio
sys.path.insert(0,str(HERE))
import numpy as np
from kernels import reset_low_gradient,self_check
spec=importlib.util.spec_from_file_location('baseline_core',ROOT/'tools/overnight_energy_v2/proxy_objective_v1/run.py')
core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
OUT=ROOT/'results/overnight_energy_v2/full_system_baselines_v1'
DEST=OUT/'fedmemprune_rho040'
PREFIX=ROOT/'results/overnight_energy_v2/ofdma_v1/learning/ofdma_initial20'


def status(**x):core.write(OUT/'status.json',dict(unix=time.time(),pid=os.getpid(),**x))


def chunks(ids,seed,batch=16):
    ordered=np.random.default_rng(seed).permutation(ids).astype(int).tolist()
    parts=[ordered[i:i+batch] for i in range(0,len(ordered),batch)]
    if len(parts)>1 and len(parts[-1])==1:parts[-2]+=parts.pop()
    return parts


def local_adam(engine,theta,ids,k,user):
    import torch
    variable=theta.detach().clone().requires_grad_(True)
    optimizer=torch.optim.Adam([variable],lr=1e-4)
    steps=0;loss_sum=0.;start=time.time()
    for epoch in range(3):
        for batch in chunks(ids,510000+k*1000+user*10+epoch):
            grad,loss=engine.gradient(variable.detach(),batch,k+epoch)
            variable.grad=grad;optimizer.step();optimizer.zero_grad(set_to_none=True)
            steps+=1;loss_sum+=loss
    assert torch.equal(variable[-1],theta[-1])
    return (variable.detach()-theta),dict(user=user,steps=steps,epochs=3,local_loss=loss_sum/max(1,steps),wall_s=time.time()-start)


def gradient_upload(engine,theta,rec,k):
    import torch
    result=np.zeros((10,len(theta)-1));stats=[]
    for user,ids in enumerate(rec['retained']):
        status(stage='fedmemprune_gradient_collection',round=k,user=user)
        accum=torch.zeros_like(theta);loss=0.;seen=0
        for batch in chunks(ids,490000+user):
            g,l=engine.gradient(theta,batch,k)
            accum+=len(batch)*g;loss+=len(batch)*l;seen+=len(batch)
        result[user]=(accum[:-1]/seen/10).cpu().numpy()
        stats.append(dict(user=user,examples=seen,retained_loss=loss/seen))
    return result,stats


def evaluate(engine,theta,reference,plan,forget):
    import torch
    values,probs=engine.evaluate(theta,reference)
    li,lt=probs;oldi,oldt=reference;gallery=plan['gallery_images'];per={}
    for user,pool in enumerate(forget):
        ids=sorted(set(pool)&set(plan['forget_probe']))
        if not ids:continue
        indices=torch.tensor([gallery.index(i) for i in ids]);ci=(indices[:,None]*5+torch.arange(5)).flatten()
        ki=(oldi[indices].exp()*(oldi[indices]-li[indices])).sum(1).mean()
        kt=(oldt[ci].exp()*(oldt[ci]-lt[ci])).sum(1).mean()
        per[str(user)]=dict(images=len(ids),KL_retrain_to_candidate=max(0.,float((ki+kt)/2)))
    values['per_request']=per
    return values,probs


def summary(plan):
    rows=[json.loads(p.read_text()) for p in sorted((DEST/'rounds').glob('*.json'))]
    evals=[r for r in rows if 'evaluation' in r]
    threshold=.29304036883016427;hits=[];target=None
    for r in evals:
        worst=max(x['KL_retrain_to_candidate'] for x in r['evaluation']['per_request'].values())
        ok=r['round']>=25 and r['evaluation']['retained']['mR']>=threshold and worst<=.01
        hits=hits+[r] if ok else []
        if len(hits)>=3 and target is None:
            target=dict(entry_round=hits[0]['round'],confirmation_round=r['round'],
                energy_j=r['cumulative_tx_energy_bound_j'],uplink_s=r['cumulative_uplink_s'])
    core.write(DEST/'summary.json',dict(status='running',completed_communication_rounds=rows[-1]['round'] if rows else 20,
        latest=evals[-1] if evals else None,joint_target_against_common_reference=target,
        ranked=False,ranking_limitation='VLM adaptation pilot; method-matched retraining reference pending. Common-reference KL is descriptive.',
        retained_mR_threshold=threshold,per_request_KL_threshold=.01,consecutive_checks=3))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stop',type=int,default=200);parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    assert 21<=args.stop<=200
    OUT.mkdir(parents=True,exist_ok=True);DEST.mkdir(exist_ok=True)
    own=(OUT/'launch.lock').open('a');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
    lock=(ROOT/'results/overnight_energy_v2/association_objective_gpu.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert args.resume or not (DEST/'started.json').exists(),'Duplicate launch'
    import subprocess
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    import torch
    from train import Engine,update_theta
    from dataset import RetrievalData
    checks=self_check();core.write(OUT/'kernel_checks.json',checks)
    plan=json.loads((core.SOURCE/'plan.json').read_text());records=json.loads((core.SOURCE/'records.json').read_text())
    for name,digest in plan['sources'].items():assert core.sha(ROOT/name)==digest,name
    for name,digest in plan['frozen_engine_hashes'].items():assert core.sha(core.FROZEN/name)==digest,name
    assert core.sha(ROOT/'tools/overnight_energy_v2/proxy_objective_v1/run.py')=='6a5e2af576547e550bcdad8a2867b12ae31ebb79d08e3da41c9bb7685ab08bca'
    source_hashes={p.name:core.sha(p) for p in HERE.glob('*.py')}
    if args.resume:assert source_hashes==json.loads((DEST/'started.json').read_text())['source_hashes'],'Source changed during campaign'
    else:core.write(DEST/'started.json',dict(pid=os.getpid(),unix=time.time(),source_hashes=source_hashes))
    status(stage='loading_model');engine=Engine(RetrievalData(),dict(plan))
    forget=records[20]['forget'];channels=np.load(core.SOURCE/'channels.npy')
    raw=np.load(core.SOURCE/'retrain/probabilities_200.npz')
    reference=(torch.tensor(raw['log_image']),torch.tensor(raw['log_text']))
    prefix=json.loads((PREFIX/'prefix_rounds.json').read_text())
    total_e=prefix[-1]['cumulative_tx_energy_bound_j'];total_t=prefix[-1]['cumulative_uplink_s']
    theta=core.load_theta(PREFIX/'round_020.pt',engine.device);start_round=21
    existing=sorted((DEST/'rounds').glob('*.json')) if (DEST/'rounds').exists() else []
    if existing:
        previous=json.loads(existing[-1].read_text());start_round=previous['round']+1
        theta=core.load_theta(DEST/f'round_{previous["round"]:03d}.pt',engine.device)
        total_e=previous['cumulative_tx_energy_bound_j'];total_t=previous['cumulative_uplink_s']
    core.write(DEST/'protocol.json',dict(method='FedMemPrune VLM/OFDMA adaptation',paper='10.1145/3770855.3817785',
        preserved=['mean retained gradients','lowest absolute mean gradient parameters','fresh original-distribution initialization','remaining-data federated fine-tuning'],
        ratio=.4,ratio_selection='Preregistered non-IID value from paper, not chosen using this benchmark outputs',
        optimizer='Adam',lr=1e-4,full_local_epochs=3,local_batch_size=16,
        adaptations=['CNN -> common frozen CLIP + visual/text prompts','task loss -> bidirectional image/text contrastive loss',
            'digital OFDMA deployment and common quantization-error budget','existing common digital SGD initial 20 rounds',
            '200 communication-round cap, including one gradient-selection upload','local Adam state resets each round'],
        no_proposed_proxy=True,no_relation_KL=True,original_paper_reproduction=False,
        parameter_count=len(theta)-1,initial_prefix_sha256=core.sha(PREFIX/'round_020.pt'),
        prefix_energy_j=total_e,prefix_uplink_s=total_t,reference='existing common retain-only output; method-matched reference pending',
        ranking_permitted=False,request_round=21,forgotten_images=len(plan['forgotten_images']),
        energy_scope='TX-only: all initial learning, gradient selection and recovery uploads; no receiver computation',
        timing_scope='uplink occupied time; local training measured separately',stop=args.stop))
    core.write(DEST/'prefix_rounds.json',prefix)
    for k in range(start_round,args.stop+1):
        rec=records[k-1];t0=time.time();status(stage='fedmemprune',round=k,stop=args.stop)
        if k==21:
            uploads,local=gradient_upload(engine,theta,rec,k);stage='gradient_selection_and_reset'
        else:
            uploads=np.zeros((10,len(theta)-1));local=[];stage='retained_finetune'
            for user,ids in enumerate(rec['retained']):
                status(stage='fedmemprune_local_training',round=k,user=user,stop=args.stop)
                delta,details=local_adam(engine,theta,ids,k,user)
                uploads[user]=(delta[:-1]/10).cpu().numpy();local.append(details)
        vector,r=radio.communicate(uploads,np.zeros_like(uploads),channels,dict(plan['config'],ofdma_subcarriers=128),'ofdma_only',300000+k)
        if vector is None:
            core.write(DEST/'failure.json',dict(status='no_feasible_radio_design_found',round=k,radio=r,ranking_permitted=False))
            status(stage='stopped_no_feasible_radio_design',round=k);return
        if k==21:
            fresh=torch.randn(len(theta)-1,generator=torch.Generator().manual_seed(71531)).numpy()*.02
            reset,selected=reset_low_gradient(theta[:-1].cpu().numpy(),vector,fresh,.4)
            theta=theta.clone();theta[:-1]=torch.tensor(reset,device=engine.device)
            core.write(DEST/'pruning.json',dict(selected_count=len(selected),selection_sha256=hashlib.sha256(selected.tobytes()).hexdigest(),
                fresh_seed=71531,initialization='N(0,0.02^2), same as common prompt initialization',
                wire_gradient_error=float(np.linalg.norm(vector-uploads.sum(0)))))
        else:theta=update_theta(theta,vector)
        total_e+=r['energy_bound_j'];total_t+=r['uplink_s']
        row=dict(round=k,stage=stage,radio=r,local=local,cumulative_tx_energy_bound_j=total_e,
            post_request_tx_energy_j=total_e-prefix[-1]['cumulative_tx_energy_bound_j'],
            cumulative_uplink_s=total_t,local_wall_s=time.time()-t0)
        if k%5==0 or k==21 or k==args.stop:
            row['evaluation'],probs=evaluate(engine,theta,reference,plan,forget)
            np.savez_compressed(DEST/f'probabilities_{k:03d}.npz',log_image=probs[0].numpy(),log_text=probs[1].numpy())
        row['wall_s']=time.time()-t0
        core.save_theta(DEST/f'round_{k:03d}.pt',theta,k)
        core.write(DEST/'rounds'/f'{k:03d}.json',row);summary(plan)
        print(json.dumps(dict(round=k,stage=stage,wall_s=row['wall_s'],evaluation=row.get('evaluation'))),flush=True)
    mia,scores=engine.mia_scores(theta);core.write(DEST/'mia.json',mia);np.save(DEST/'mia_scores.npy',scores)
    core.write(DEST/'finished.json',dict(status='completed',rounds=args.stop,unix=time.time(),ranking_permitted=False))
    status(stage='fedmemprune_completed',round=args.stop)


if __name__=='__main__':
    try:main()
    except Exception as e:status(stage='failed',error=repr(e));traceback.print_exc();raise
    finally:radio.close()
