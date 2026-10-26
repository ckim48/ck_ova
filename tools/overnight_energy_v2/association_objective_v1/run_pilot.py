"""Preregistered ideal-link mechanism screen. No test data or transport claims."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path[:0]=[str(HERE),str(ROOT)]
from objectives import retained_terms, forget_term, self_check
from clip_model import DeepPromptCLIP
from real_data import CIFAR100


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    temp=p.with_suffix('.tmp');temp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');temp.replace(p)


def main():
    out=ROOT/'results/overnight_energy_v2/association_objective_v1'
    out.mkdir(parents=True,exist_ok=True)
    lock=(ROOT/'results/overnight_energy_v2/association_objective_gpu.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (out/'plan.json').exists(), 'Refuse duplicate/overwrite; version separately.'
    assert torch.cuda.is_available()
    torch.set_num_threads(4);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
    device=torch.device('cuda');started=time.time()
    olddir=ROOT/'results/overnight_energy_v2/partial_overlap_tx_v1/learning_comparison'
    oldplan=json.loads((olddir/'plan.json').read_text());c=oldplan['config']
    assert all(sha(ROOT/n)==h for n,h in oldplan['source_sha256'].items())
    assert sha(ROOT/oldplan['warm_checkpoint'])==oldplan['warm_checkpoint_sha256']
    rec=json.loads((olddir/'records.json').read_text())[0]
    ckpt=ROOT/'checkpoints/clip-vit-base-patch16'
    dataset=CIFAR100(ROOT/'data/cifar-100-python',ckpt)
    train,heldout=dataset.validation_split(0,50)
    rng=np.random.default_rng(20261021)
    forgotten_train=[];forgotten_eval=[]
    for pool in rec['forget']:
        ids=rng.permutation(pool).astype(int).tolist()
        forgotten_train.extend(ids[:64]);forgotten_eval.extend(ids[64:104])
    retained_pool=sorted(set().union(*map(set,rec['retained'])))
    retained_train=rng.choice(retained_pool,512,replace=False).astype(int).tolist()
    # Reuse heldout TRAINING-validation, never CIFAR test examples.
    retained_eval=rng.permutation(heldout)[:512].astype(int).tolist()
    assert not set(forgotten_train)&set(forgotten_eval)
    assert not set(retained_train)&(set(forgotten_train)|set(forgotten_eval)|set(retained_eval))
    assert not set(retained_eval)&(set(forgotten_train)|set(forgotten_eval))
    groups=dict(retain_train=retained_train,forget_train=forgotten_train,retain_validation=retained_eval,forget_monitor=forgotten_eval)
    kinds=['retain_only','gradient_ascent','cliperase_paper','cliperase_balanced','neutral_balanced']
    plan=dict(version='association_objective_v1',created_unix=time.time(),pid=os.getpid(),
      methods=kinds,steps=24,batch_size=32,microbatch=4,learning_rate=.0035,optimizer='SGD, no momentum',
      consistency_weight=3.,paper_forget_weight=3.,evaluation_steps=[0,1,4,12,24],seed=0,
      warm_checkpoint=oldplan['warm_checkpoint'],warm_checkpoint_sha256=oldplan['warm_checkpoint_sha256'],
      source_sha256={n:sha(ROOT/n) for n in oldplan['source_sha256']},
      experiment_sha256={p.name:sha(p) for p in HERE.glob('*.py')},groups=groups,
      references=['https://aclanthology.org/2025.acl-long.1469/','https://arxiv.org/html/2311.12047v2'],
      objective_definition={
        'retained':'class-prototype CE, plus 3*consistency for new methods',
        'cliperase_paper':'mean normalized image-text matched cosine; KL(teacher||student) of normalized image/text feature softmax on retained data; lambda_f=3',
        'cliperase_balanced':'same objective; frozen lambda_f=norm(retain CE gradient)/norm(forget cosine gradient), from first training batch only',
        'neutral_balanced':'squared distance of matched cosine to mean teacher UNRELATED retained-pair cosine; retain prediction KL and squared normalized image/text feature distances; same training-gradient balance',
        'gradient_ascent':'retained CE minus forget CE, lambda_f=1',
        'temperature':'pre-request teacher and student logit-scale held fixed in ALL arms to isolate alignment changes'},
      scope='Centralized pooled-gradient mechanism pilot with ideal communication, before full federated transport trajectories. Same original K5 request IDs; sample-pair association suppression, not whole-class deletion.',
      changes=['P removed','multi-step request handling instead of a single request-round step','constant prompt-SGD step size rather than old round61 cosine value','fixed old model retention teacher','CLIP prototype CE instead of paper pair-batch contrastive loss','normalized similarity readout for MultiDelete-inspired surrogate, not original fusion-output loss'],
      interpretation='Association weakening is not equivalent to erasing training influence. No matched retrain reference exists in this pilot; KL-to-teacher is retention drift, not forgetting. No certified deletion or MIA conclusion. Test arrays may be loaded by the existing dataset class but are never sampled/evaluated.',
      promotion_gate=dict(forget_alignment_gap_reduction_fraction=.10,max_retain_validation_loss_increase=.02,max_retain_validation_accuracy_drop=.01,criterion='All three at final step24; check forget training and held-out request-monitor separately. Gate only supports follow-up, not deletion claim.'),
      transport_followup='Only after meaningful mechanism result. Apply identical objective to Hybrid, same-resource single-sum AirComp, strong AirComp+OMA, and OMA-only. No extra receiver controller; same canonical quantized target and budgets. Costs uplink transmit waveform energy/time only.')
    write(out/'plan.json',plan);write(out/'self_checks.json',self_check())
    write(out/'status.json',dict(status='initializing',pid=os.getpid(),started_unix=started))
    model=DeepPromptCLIP.from_local_checkpoint(ckpt,prompt_layers=8,prompt_tokens=2,checkpoint_layers=True).to(device).eval()
    theta0=torch.load(ROOT/plan['warm_checkpoint'],map_location='cpu',weights_only=True)['theta'].to(device)
    text=dataset.text_inputs(device,2)
    with torch.no_grad():
        vp,tp,scale=model.parts(theta0)
        teacher_text=F.normalize(model.text_features(**text,text=tp),dim=-1).detach()
    teacher_images={}
    @torch.no_grad()
    def cache_teacher():
        ids=sorted(set().union(*map(set,groups.values())))
        for start in range(0,len(ids),16):
            batch=ids[start:start+16];pix,_=dataset.batch(batch,device)
            feats=F.normalize(model.image_features(pix,vp),dim=-1)
            for i,f in zip(batch,feats):teacher_images[i]=f.detach().cpu()
    cache_teacher()
    with torch.no_grad():
        im=torch.stack([teacher_images[i] for i in retained_train]).to(device)
        score=im@teacher_text.T;lab=torch.tensor(dataset.train_labels[retained_train],device=device)
        unrelated_target=float(((score.sum(-1)-score.gather(1,lab[:,None]).squeeze(1))/99).mean())
    write(out/'teacher_metadata.json',dict(unrelated_retained_pair_mean_cosine=unrelated_target,teacher_role='frozen pre-request model; not a retrained reference'))

    def gradient(theta,ids,kind,forget=False):
        result=torch.zeros_like(theta);values={}
        for j in range(0,len(ids),4):
            selected=ids[j:j+4];pix,labels=dataset.batch(selected,device)
            variable=theta.detach().requires_grad_(True);v,t,s=model.parts(variable)
            images=F.normalize(model.image_features(pix,v),dim=-1)
            texts=F.normalize(model.text_features(**text,text=t),dim=-1)
            if forget:
                value=forget_term(images,texts,s,labels,kind,unrelated_target);terms={'forget':value}
            else:
                oldim=torch.stack([teacher_images[i] for i in selected]).to(device)
                ce,cons=retained_terms(images,texts,s,labels,oldim,teacher_text,kind)
                value=ce+3*cons;terms={'retain_ce':ce,'consistency':cons}
            g=torch.autograd.grad(value,variable)[0].detach();g[-1]=0
            result+=g*len(selected)/len(ids)
            for key,val in terms.items():values[key]=values.get(key,0.)+float(val.detach())*len(selected)/len(ids)
        assert torch.isfinite(result).all()
        return result,values

    @torch.no_grad()
    def metrics(theta,ids):
        v,t,s=model.parts(theta);texts=F.normalize(model.text_features(**text,text=t),dim=-1)
        rows=[];predictions=[]
        for start in range(0,len(ids),16):
            selected=ids[start:start+16];pix,labels=dataset.batch(selected,device)
            images=F.normalize(model.image_features(pix,v),dim=-1);sim=images@texts.T;logits=s.exp()*sim
            oldim=torch.stack([teacher_images[i] for i in selected]).to(device)
            oldlog=scale.exp()*oldim@teacher_text.T
            matched=sim.gather(1,labels[:,None]).squeeze(1);other=(sim.sum(-1)-matched)/99
            pred=logits.argmax(-1);probs=F.softmax(logits,dim=-1)
            losses=F.cross_entropy(logits,labels,reduction='none')
            kl=(F.softmax(oldlog,dim=-1)*(F.log_softmax(oldlog,dim=-1)-F.log_softmax(logits,dim=-1))).sum(-1)
            image_drift=1-(images*oldim).sum(-1)
            text_drift=1-(texts[labels]*teacher_text[labels]).sum(-1)
            rows.extend(torch.stack([losses,(pred==labels).float(),matched,matched-other,probs.gather(1,labels[:,None]).squeeze(1),kl,image_drift,text_drift],dim=-1).cpu().tolist())
            predictions.extend(probs.cpu().tolist())
        mean=np.mean(rows,axis=0)
        return dict(zip(['loss','accuracy','matched_cosine','alignment_gap_vs_unrelated','true_class_probability','kl_pre_request_teacher_to_candidate','image_cosine_drift','text_cosine_drift'],map(float,mean)))|{'samples':len(ids)},np.asarray(predictions,dtype=np.float32)

    base={}
    for group in ['retain_validation','forget_monitor','forget_train']:
        base[group],_=metrics(theta0,groups[group])
    write(out/'initial_metrics.json',base)
    # Training batches and balance factors fixed before examining any candidate validation result.
    brng=np.random.default_rng(70101)
    batches=[(brng.choice(retained_train,32,replace=False).tolist(),brng.choice(forgotten_train,32,replace=False).tolist()) for _ in range(24)]
    write(out/'training_batches.json',batches)
    train_r,train_f=batches[0]
    grad_r,_=gradient(theta0,train_r,'retain_only');norm_r=float(grad_r.norm())
    balance={}
    for kind in kinds:
        if kind=='retain_only':balance[kind]=dict(weight=0.);continue
        g,_=gradient(theta0,train_f,kind,True);norm=float(g.norm())
        weight=(norm_r/max(norm,1e-12)) if kind.endswith('balanced') else (3. if kind=='cliperase_paper' else 1.)
        balance[kind]=dict(weight=weight,retain_gradient_norm=norm_r,forget_gradient_norm=norm,visual_gradient_norm=float(g[:model.visual_dimension].norm()),text_gradient_norm=float(g[model.visual_dimension:-1].norm()))
    write(out/'training_only_balance.json',balance)
    summaries=[]
    for kind in kinds:
        dest=out/kind;dest.mkdir(exist_ok=False);theta=theta0.clone()
        weight=balance[kind]['weight'];rows=[]
        for step,(retain_ids,forget_ids) in enumerate(batches,1):
            t0=time.time();gr,rt=gradient(theta,retain_ids,kind)
            gf,ft=(gradient(theta,forget_ids,kind,True) if weight else (torch.zeros_like(theta),{}))
            delta=-.0035*(gr+weight*gf)
            theta=(theta+delta).detach();assert torch.isfinite(theta).all()
            row=dict(method=kind,step=step,objective_terms=rt|ft,forget_weight=weight,
              retained_update_norm=float((.0035*gr).norm()),forget_update_norm=float((.0035*weight*gf).norm()),
              total_update_norm=float(delta.norm()),step_wall_s=time.time()-t0)
            if step in [1,4,12,24]:
                row['evaluation']={}
                for group in ['retain_validation','forget_monitor','forget_train']:
                    row['evaluation'][group],preds=metrics(theta,groups[group])
                    if step==24:np.savez_compressed(dest/(group+'_predictions.npz'),probabilities=preds,ids=groups[group])
                torch.save(dict(theta=theta.cpu(),pilot_step=step),dest/f'step_{step:03d}.pt')
            rows.append(row)
            with (dest/'steps.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            write(out/'status.json',dict(status='running',method=kind,step=step,total_steps_per_method=24,method_index=kinds.index(kind)+1,method_count=len(kinds),elapsed_s=time.time()-started,pid=os.getpid()))
            print(json.dumps(dict(method=kind,step=step,elapsed_s=time.time()-started,evaluation=row.get('evaluation'))),flush=True)
        final=rows[-1]['evaluation'];gate=plan['promotion_gate']
        gap0=base['forget_monitor']['alignment_gap_vs_unrelated'];gap=final['forget_monitor']['alignment_gap_vs_unrelated']
        reduction=1-gap/gap0 if gap0>0 else None
        loss_change=final['retain_validation']['loss']-base['retain_validation']['loss']
        acc_change=final['retain_validation']['accuracy']-base['retain_validation']['accuracy']
        passed=bool(reduction is not None and reduction>=.10 and loss_change<=.02 and acc_change>=-.01)
        summary=dict(method=kind,metrics=final,forget_alignment_gap_reduction_fraction=reduction,retain_loss_change=loss_change,retain_accuracy_change=acc_change,promotion_gate_passed=passed,forget_weight=weight)
        summaries.append(summary);write(out/'summary.json',summaries)
    write(out/'status.json',dict(status='completed',elapsed_s=time.time()-started,pid=os.getpid(),promotion_candidates=[x['method'] for x in summaries if x['promotion_gate_passed'] and x['method'] not in ['retain_only','gradient_ascent']],next_action='Review mechanism and disjoint monitoring results before federated transport; no automatic long training.'))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        out=ROOT/'results/overnight_energy_v2/association_objective_v1'
        if out.exists():write(out/'error.json',dict(error=repr(exc),pid=os.getpid()))
        raise
