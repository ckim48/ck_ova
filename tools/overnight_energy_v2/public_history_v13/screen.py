"""Client-local public-history projection screen, no radio or GPU promotion."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def identity(pid):
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()
        return None if fields[0]=='Z' else fields[19]
    except FileNotFoundError:return None
def normsum(v,width):
    return float(np.linalg.norm(np.linalg.norm(np.pad(v,((0,0),(0,(-v.shape[1])%width))).reshape(len(v),-1,width),axis=-1).sum(axis=0)))
def projection(v,basis):
    # The client forms its residual using the decoded float32 coefficients, so
    # the receiver uses exactly the same coefficients and reconstruction basis.
    coefficients=(v@basis).astype(np.float32).astype(np.float64)
    residual=v-coefficients@basis.T
    error=np.linalg.norm(residual.sum(axis=0)+coefficients.sum(axis=0)@basis.T-v.sum(axis=0))
    assert error<=1e-12*max(np.linalg.norm(v),1e-12)
    assert np.all(np.linalg.norm(residual,axis=1)<=np.linalg.norm(v,axis=1)*(1+1e-7)+1e-20)
    return residual,coefficients,float(error)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,required=True)
    p.add_argument('--legacy-only',action='store_true');args=p.parse_args();root=args.project.resolve()
    sys.path[:0]=[str(root),str(root/'tools/energy_quality_v1')]
    import torch
    core=read(root/'results/comparisons_seed0_solver_v2/suite_metadata.json')['source_sha256']
    assert all(sha(root/n)==v for n,v in core.items())
    c=read(root/'results/failed_round_diagnostic/metadata.json')['config']
    family=root/'results/overnight_energy_v2/public_history_v13'
    if args.legacy_only:
        out=family/'legacy_screen';out.mkdir(exist_ok=True);assert not (out/'results.json').exists()
        source=root/'results/clip/coupled_hybrid_seed0'
        assert sha(source/'latest.pt')==read(root/'results/failed_round_diagnostic/metadata.json')['checkpoint_sha256']
        theta=torch.load(source/'latest.pt',map_location='cpu',weights_only=True)['theta'].numpy().astype(float)
        initial=torch.load(source/'initial.pt',map_location='cpu',weights_only=True)['theta'].numpy().astype(float)
        with np.load(root/'results/failed_round_diagnostic/updates.npz') as a:l,u=[a[k].copy() for k in ['learning','unlearning']]
        record=read(source/'schedule.json')[90];alpha=np.asarray([len(x) for x in record['retained']],float);alpha/=alpha.sum()
        eta=lambda k:c['learning_rate']*.5*(1+np.cos(np.pi*(k-1)/c['rounds']))
        predictor=alpha[:,None]*eta(91)/sum(eta(k) for k in range(1,91))*(theta-initial)[None,:]
        q=(theta-initial)/np.linalg.norm(theta-initial);residual,coefficient,error=projection(l,q[:,None])
        mixed,mixed_coefficient,mixed_error=projection(l+u,q[:,None])
        result=dict(status='completed_norm_screen_not_radio_or_gpu_promoted',at_kst=datetime.now().isoformat(timespec='seconds'),
                    source_sha256=core,script_sha256=sha(__file__),checkpoint_sha256=sha(source/'latest.pt'),initial_sha256=sha(source/'initial.pt'),
                    updates_sha256=sha(root/'results/failed_round_diagnostic/updates.npz'),
                    historical_rate_mean_normsum_ratio=normsum(l-predictor,c['block_dimension'])/normsum(l,c['block_dimension']),
                    public_displacement_projection_learning_ratio=normsum(residual,c['block_dimension'])/normsum(l,c['block_dimension']),
                    public_displacement_projection_mixed_ratio=normsum(mixed,c['block_dimension'])/normsum(l+u,c['block_dimension']),
                    learning_reconstruction_error=error,mixed_reconstruction_error=mixed_error,
                    coefficient_metadata_bits_per_head=32*(c['users']-1),
                    limits=['Public model displacement includes learning, unlearning and radio effects; no free exact learning-gradient history.',
                            'Fixed original legacy state; no validation/test outputs. No radio/energy guarantee.',
                            'Mean predictor worsens normsum. Projection gains are tiny and extra coefficient delivery is unpaid here, so no immediate radio/GPU promotion.'])
        write(out/'results.json',result)
        np.savez_compressed(out/'vectors.npz',learning=l,unlearning=u,public_direction=q,historical_rate_prediction=predictor,
                            projected_learning_residual=residual,learning_coefficients=coefficient,
                            projected_mixed_residual=mixed,mixed_coefficients=mixed_coefficient)
        print(json.dumps(result));return
    out=family/'recent_screen';out.mkdir(exist_ok=True);assert not (out/'plan.json').exists(),'Do not duplicate replay'
    state=read(root/'tools/overnight_energy_v2/campaign_state.json')
    waiting={name:(state[name],identity(state[name])) for name in ['public_history_v13_observer_pid','u_precision_v12_cpu_pid']}
    plan=dict(version='public_history_v13_fixed_actual_round100_learning_only',script_sha256=sha(__file__),core_sha256=core,
              waiting_for={name:pid for name,(pid,_) in waiting.items()},cpu_threads=4,
              fixed_target='Actual PUF round100 retained-learning minibatches at observed PUBLIC theta99. No new/deferred unlearning requests invented.',
              bases=['latest available public global increment','span of at most4 most recent public global increments ending at99'],
              limits=['Pure learning-only communication screen from a PUF trajectory, not a coupled trajectory or final task evaluation.',
                      'Only public states at round<=99 can form predictors for round100. No private historical gradients or other-method final checkpoints.',
                      '32-bit coefficient per client per basis dimension requires reliable delivery; time/power/circuit energy not yet optimized or free.',
                      'Apply identical public information and metadata cost to every comparator before any energy claim. No validation/test selection or automatic GPU promotion.'])
    write(out/'plan.json',plan)
    def status(s,**extra):write(out/'status.json',dict(status=s,at_kst=datetime.now().isoformat(timespec='seconds'),**extra))
    status('waiting_for_observer_and_prior_cpu_model');deadline=time.monotonic()+1800
    while any(start is not None and identity(pid)==start for pid,start in waiting.values()):
        if time.monotonic()>deadline:status('skipped_wait_timeout');return
        time.sleep(10)
    observer=family/'observer';states=read(observer/'states.json');eligible=sorted((x for x in states if x['round']<=99),key=lambda x:x['round'])
    if len(eligible)<2 or eligible[-1]['round']!=99:status('skipped_insufficient_public_history',captured_rounds=[x['round'] for x in eligible]);return
    eligible=eligible[-5:];snapshots=[]
    for item in eligible:
        p=observer/f"round_{item['round']:03}.pt";assert sha(p)==item['file_sha256']
        checkpoint=torch.load(p,map_location='cpu',weights_only=True);assert checkpoint['round']==item['round']
        snapshots.append(checkpoint['theta'].numpy().astype(float))
    assert all(y['round']==x['round']+1 for x,y in zip(eligible,eligible[1:])), 'Public model increments must be consecutive'
    differences=np.stack([v-u for u,v in zip(snapshots,snapshots[1:])],axis=1)
    norms=np.linalg.norm(differences,axis=0);assert np.all(norms>0)
    normalized=differences/norms
    left,singular,_=np.linalg.svd(normalized,full_matrices=False)
    rank=min(4,int(np.sum(singular>singular[0]*1e-10)))
    bases={'latest_global_increment':normalized[:,-1:],f'recent_public_span_rank{rank}':left[:,:rank]}
    theta=torch.as_tensor(snapshots[-1],dtype=torch.float32);device=torch.device('cpu');torch.set_num_threads(4);torch.manual_seed(c['seed'])
    from clip_model import DeepPromptCLIP
    from real_data import CIFAR100
    from run_clip_federated import local_gradient
    model_path=root/'checkpoints/clip-vit-base-patch16';dataset=CIFAR100(root/'data/cifar-100-python',model_path)
    model=DeepPromptCLIP.from_local_checkpoint(model_path,prompt_layers=c['prompt_layers'],prompt_tokens=c['prompt_tokens'],checkpoint_layers=False).eval()
    text=dataset.text_inputs(device,c['prompt_tokens']);source=root/'results/comparisons_seed0_solver_v2/puf_regular_adapted_hybrid'
    schedule=read(source/'schedule.json');record=schedule[99];assert record['round']==100 and not any(record['forget'])
    alpha=np.asarray([len(x) for x in record['retained']],float);alpha/=alpha.sum()
    eta=c['learning_rate']*.5*(1+np.cos(np.pi*99/c['rounds']));learning=[]
    for user,batch in enumerate(record['batches']):
        assert set(batch).issubset(set(record['retained'][user]))
        status('replaying_actual_local_learning_on_cpu',user=user)
        gradient=local_gradient(model,theta,dataset,text,batch,device,c['microbatch_size'])
        learning.append((-eta*alpha[user]*gradient).numpy().astype(float))
    learning=np.stack(learning)
    with np.load(source/'partition.npz') as a:channels=a['channels'].copy()
    baseline=normsum(learning,c['block_dimension']);rows=[];saved=dict(learning=learning,channels=channels,alpha=alpha)
    for name,basis in bases.items():
        residual,coefficient,error=projection(learning,basis)
        rows.append(dict(basis=name,rank=basis.shape[1],normsum_ratio=normsum(residual,c['block_dimension'])/baseline,
                         reconstruction_error=error,per_client_norm_ratio=(np.linalg.norm(residual,axis=1)/np.linalg.norm(learning,axis=1)).tolist(),
                         coefficient_bits_per_head=32*basis.shape[1]*(c['users']-1)))
        saved[name+'_basis']=basis;saved[name+'_residual']=residual;saved[name+'_coefficients']=coefficient
    np.savez_compressed(out/'vectors.npz',**saved)
    report=dict(status='completed_cpu_norm_screen_no_radio_or_gpu_promotion',at_kst=datetime.now().isoformat(timespec='seconds'),
                round=100,public_rounds=[x['round'] for x in eligible],public_state_metadata_sha256=sha(observer/'states.json'),
                schedule_sha256=sha(source/'schedule.json'),source_manifest_sha256=sha(source/'manifest.json'),
                script_sha256=sha(__file__),baseline_learning_normsum=baseline,rows=rows,limits=plan['limits'])
    write(out/'results.json',report);status(report['status']);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
