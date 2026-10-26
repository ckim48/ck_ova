"""Actual history-weighted FU gradients and a CPU one-step validation screen."""
import argparse
from collections import defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

STATUS_PATH=None


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def weighted_gradient(model,theta,dataset,text,ids,weights,device,microbatch):
    import torch
    import torch.nn.functional as F
    result=torch.zeros_like(theta)
    for start in range(0,len(ids),microbatch):
        value=theta.detach().requires_grad_(True)
        pixels,targets=dataset.batch(ids[start:start+microbatch],device)
        losses=F.cross_entropy(model(pixels,**text,theta=value),targets,reduction='none')
        w=torch.as_tensor(weights[start:start+microbatch],dtype=value.dtype,device=device)
        loss=(losses*w).sum()/len(ids)
        result+=torch.autograd.grad(loss,value)[0].detach()
    return result


def main():
    global STATUS_PATH
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--project',type=Path,required=True)
    parser.add_argument('--plan-only',action='store_true');args=parser.parse_args();root=args.project.resolve()
    sys.path[:0]=[str(root),str(root/'tools/energy_quality_v1')]
    import torch
    import torch.nn.functional as F
    from clip_model import DeepPromptCLIP,projected_coupling
    from real_data import CIFAR100
    from run_clip_federated import local_gradient
    import energy_quality as eq
    source=root/'results/failed_round_diagnostic';meta=json.loads((source/'metadata.json').read_text());c=meta['config']
    checkpoint=root/'results/clip/coupled_hybrid_seed0/latest.pt';schedule_path=checkpoint.parent/'schedule.json'
    assert sha(checkpoint)==meta['checkpoint_sha256']
    expected=json.loads((root/'results/comparisons_seed0_solver_v2/suite_metadata.json').read_text())['source_sha256']
    assert all(sha(root/n)==h for n,h in expected.items())
    assert sha(schedule_path)==sha(root/'results/comparisons_seed0_solver_v2/coupled_hybrid/schedule.json')
    old=root/'results/overnight_energy_v2/single_step_quality_pilot';old_manifest=json.loads((old/'manifest.json').read_text())
    assert old_manifest['checkpoint_sha256']==sha(checkpoint) and old_manifest['updates_sha256']==sha(source/'updates.npz')
    groups=json.loads((old/'cohorts.json').read_text());assert set(groups)=={'validation','forget'}
    quality=root/'results/energy_quality_v1';ref_cache=json.loads((quality/'retrain_reference_ideal.cache.json').read_text())
    assert ref_cache['checkpoint_sha256']==sha(root/'results/comparisons_seed0_solver_v2/retrain_reference_ideal/latest.pt')
    assert ref_cache['cohorts_sha256']==sha(quality/'cohorts.json') and ref_cache['evaluator_sha256']==sha(eq.__file__)
    out=root/'results/overnight_energy_v2/history_weighted_v6';out.mkdir(parents=True,exist_ok=True)
    STATUS_PATH=out/'status.json'
    plan={'version':'history_weighted_v6_cpu_pilot','checkpoint_sha256':sha(checkpoint),
          'updates_sha256':sha(source/'updates.npz'),'schedule_sha256':sha(schedule_path),
          'source_sha256':expected,'script_sha256':sha(__file__),
          'validation_test_sha256':sha(Path(__file__).with_name('test_history_estimator.py')),
          'groups_sha256':sha(old/'cohorts.json'),'reference_cache_sha256':sha(quality/'retrain_reference_ideal.npz'),
          'reference_metadata_sha256':sha(quality/'retrain_reference_ideal.cache.json'),
          'baseline_prediction_sha256':{n:sha(old/(n+'.npz')) for n in ['before_step','ideal_transport']},
          'rule':'c_z=sum prior eta_L*alpha/batch; unbiased uniform minibatch update |D_f|/b * sum c_z*grad f_z, then same P coupling. No current eta_U or beta applied a second time.',
          'groups':'Same previously frozen 200 validation + 200 forget examples; no test samples.',
          'promotion':'None automatic. A one-step screen cannot justify final forgetting or equal long-run quality. Radio and all comparator checks required before GPU promotion.',
          'device':'cpu','cpu_threads':4,'limits':['Reference cached inference was GPU; pilot inference is CPU. Full GPU trajectory audit is required before final comparison.',
              'Legacy round90 state and one-step round91 update, not a replay of current final trajectories.',
              'First-order current-gradient inverse approximation; no claim to undo nonlinear historical SGD or indirect influence.',
              'Computation/storage energy for maintaining coefficient ledger and weighted gradients is not calibrated.']}
    path=out/'plan.json'
    if path.exists():assert json.loads(path.read_text())==plan,'Plan changed; use another version.'
    eq.write_json(path,plan)
    if args.plan_only:print(json.dumps(plan));return
    def status(value,**extra):eq.write_json(out/'status.json',dict(status=value,at_kst=datetime.now().isoformat(timespec='seconds'),**extra))
    status('loading_cpu_model');torch.set_num_threads(4);torch.manual_seed(c['seed']);device=torch.device('cpu')
    state=torch.load(checkpoint,map_location='cpu',weights_only=True);assert state['round']==90;theta=state['theta']
    model_path=root/'checkpoints/clip-vit-base-patch16';dataset=CIFAR100(root/'data/cifar-100-python',model_path)
    model=DeepPromptCLIP.from_local_checkpoint(model_path,prompt_layers=c['prompt_layers'],prompt_tokens=c['prompt_tokens'],checkpoint_layers=True).eval()
    text=dataset.text_inputs(device,c['prompt_tokens'])
    with np.load(source/'updates.npz') as data:l,old_u,channels=[data[k].copy() for k in ['learning','unlearning','channels']]
    schedule=json.loads(schedule_path.read_text());record=schedule[90];masses=defaultdict(float)
    for rec in schedule[:90]:
        rate=c['learning_rate']*.5*(1+np.cos(np.pi*(rec['round']-1)/c['rounds']));total=sum(map(len,rec['retained']))
        for retained,batch in zip(rec['retained'],rec['batches']):
            coef=rate*len(retained)/total/len(batch)
            for z in batch:masses[int(z)]+=coef
    rate=c['learning_rate']*.5*(1+np.cos(np.pi*90/c['rounds']));forget_total=sum(map(len,record['forget']))
    gradients_path=out/'generated_updates.npz'
    if gradients_path.exists():
        with np.load(gradients_path) as data:new_u=data['unlearning'].copy()
    else:
        new_u=np.zeros_like(old_u);checks=[]
        for user,forgotten in enumerate(record['forget']):
            if not forgotten:continue
            status('computing_actual_weighted_gradient',user=user)
            batch=np.random.default_rng(c['seed']*100000+91*100+user).choice(forgotten,min(c['batch_size'],len(forgotten)),replace=False).tolist()
            weights=[len(forgotten)*masses[int(z)] for z in batch];assert min(weights)>0
            normal=local_gradient(model,theta,dataset,text,batch,device,c['microbatch_size'])
            weighted=weighted_gradient(model,theta,dataset,text,batch,weights,device,c['microbatch_size'])
            retained=record['batches'][user][:c['microbatch_size']];pixels,targets=dataset.batch(retained,device)
            loss_function=lambda value:F.cross_entropy(model(pixels,**text,theta=value),targets)
            kwargs=dict(rank=c['coupling_rank'],kappa_requested=c['coupling_kappa'],xi=c['coupling_xi'],difference_step=c['coupling_difference_step'],seed=c['seed']+91*100+user)
            normal,_=projected_coupling(loss_function,theta,normal,model.visual_dimension,model.text_dimension,**kwargs)
            weighted,_=projected_coupling(loss_function,theta,weighted,model.visual_dimension,model.text_dimension,**kwargs)
            reconstructed=(rate*len(forgotten)/forget_total*normal).numpy()
            relative=float(np.linalg.norm(reconstructed-old_u[user])/max(np.linalg.norm(old_u[user]),1e-30));assert relative<1e-4,relative
            new_u[user]=weighted.numpy();checks.append({'user':user,'old_update_reconstruction_relative_error':relative,
                'new_update_norm':float(np.linalg.norm(new_u[user])),'old_update_norm':float(np.linalg.norm(old_u[user])),
                'sampled_weight_min':min(weights),'sampled_weight_max':max(weights)})
            print('GRADIENT',json.dumps(checks[-1]),flush=True)
        np.savez_compressed(gradients_path,learning=l,unlearning=new_u,channels=channels)
        eq.write_json(out/'gradient_checks.json',checks)
    status('evaluating_cpu_one_step');model.checkpoint_layers=False
    with np.load(quality/'retrain_reference_ideal.npz') as f:full_ref={k:f[k].copy() for k in f.files}
    full_groups=json.loads((quality/'cohorts.json').read_text())['groups'];reference={}
    for group,spec in groups.items():
        index={z:i for i,z in enumerate(full_groups[group]['ids'])};reference[group]=full_ref[group][[index[z] for z in spec['ids']]]
    rows=[]
    for name,update in [('before_step',None),('old_ideal_step',None),('retained_step_only',l.sum(axis=0)),('history_weighted_ideal_step',(l+new_u).sum(axis=0))]:
        cache=(old/('ideal_transport.npz' if name=='old_ideal_step' else 'before_step.npz')) if update is None else out/(name+'.npz')
        if cache.exists():
            with np.load(cache) as data:prob={k:data[k].copy() for k in data.files}
        else:
            value=theta+torch.as_tensor(update,dtype=theta.dtype);prob={}
            for group,spec in groups.items():
                print('PREDICT',name,group,len(spec['ids']),flush=True)
                prob[group]=eq.predict(model,value,dataset,text,spec['ids'],spec['split'],device,32)
            np.savez_compressed(cache,**prob)
        row={'method':name}
        for group,spec in groups.items():
            labels=dataset.train_labels[spec['ids']];row[group]=eq.classification(prob[group],labels)
            row[group]['gap_to_final_retraining']=eq.reference_gap(prob[group],reference[group],labels)
        rows.append(row);eq.write_json(out/'results.json',rows);print('RESULT',json.dumps(row),flush=True)
    status('completed_cpu_screen_not_promoted',generated_updates_sha256=sha(gradients_path))


if __name__=='__main__':
    try:main()
    except Exception as error:
        if STATUS_PATH is not None:
            temp=STATUS_PATH.with_suffix('.error.partial.json')
            temp.write_text(json.dumps({'status':'failed_cpu_screen','error':str(error),'at_kst':datetime.now().isoformat(timespec='seconds')},indent=2)+'\n');temp.replace(STATUS_PATH)
        raise
