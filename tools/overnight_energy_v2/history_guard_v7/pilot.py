"""Local retained-gradient projection and training-batch backtracking screen."""
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


def project_direction(direction,retained_gradient):
    direction=np.asarray(direction,dtype=float);g=np.asarray(retained_gradient,dtype=float)
    norm=float(g@g)
    if norm<=1e-30:return direction.copy()
    return direction-max(0.,float(g@direction))/norm*g


def main():
    global STATUS_PATH
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,required=True);p.add_argument('--plan-only',action='store_true')
    args=p.parse_args();root=args.project.resolve();sys.path[:0]=[str(root),str(root/'tools/energy_quality_v1')]
    import torch
    from clip_model import DeepPromptCLIP
    from real_data import CIFAR100
    import energy_quality as eq
    source=root/'results/overnight_energy_v2/history_weighted_v6';prior=json.loads((source/'plan.json').read_text())
    assert json.loads((source/'status.json').read_text())['status']=='completed_cpu_screen_not_promoted'
    assert sha(source/'generated_updates.npz')=='1bb97606d82b1ed5410a5ec80231b97bcc1e68b7e8a5093302788b93a2db2df8'
    legacy=root/'results/clip/coupled_hybrid_seed0';checkpoint=legacy/'latest.pt';assert sha(checkpoint)==prior['checkpoint_sha256']
    schedule_path=legacy/'schedule.json';assert sha(schedule_path)==prior['schedule_sha256']
    c=json.loads((root/'results/failed_round_diagnostic/metadata.json').read_text())['config']
    expected=json.loads((root/'results/comparisons_seed0_solver_v2/suite_metadata.json').read_text())['source_sha256'];assert all(sha(root/n)==h for n,h in expected.items())
    out=root/'results/overnight_energy_v2/history_guard_v7';out.mkdir(parents=True,exist_ok=True);STATUS_PATH=out/'status.json'
    old=root/'results/overnight_energy_v2/single_step_quality_pilot';groups=json.loads((old/'cohorts.json').read_text())
    steps=[2.**(-j) for j in range(9)]+[0.]
    plan={'version':'history_guard_v7_local_cpu_screen','source_sha256':expected,'script_sha256':sha(__file__),
          'test_sha256':sha(Path(__file__).with_name('test_projection.py')),'prior_plan_sha256':sha(source/'plan.json'),
          'prior_generated_updates_sha256':sha(source/'generated_updates.npz'),'checkpoint_sha256':sha(checkpoint),
          'groups_sha256':sha(old/'cohorts.json'),'step_grid':steps,'cpu_threads':4,
          'rule':'Project each local history-weighted U onto g_retained^T U<=0. Backtrack on the SAME local training batches: retained loss at theta+vL+tU <= retained loss(theta), and weighted-forget loss >= its theta+vL value, within numeric tolerance. No reference/validation/test outputs select t.',
          'limits':['Local training-batch guards do not guarantee global retention, retraining equivalence or forgetting.',
                    'One legacy round; current gradients approximate history. A zero step does not fulfill a deletion request.',
                    'Extra projection and repeated local inference cost is uncalibrated. No radio/GPU promotion.',
                    'Same frozen 200 validation/200 forget screen; reference cache GPU, pilot inference CPU.']}
    path=out/'plan.json'
    if path.exists():assert json.loads(path.read_text())==plan,'Plan changed.'
    eq.write_json(path,plan)
    if args.plan_only:print(json.dumps(plan));return
    def status(s,**extra):eq.write_json(STATUS_PATH,dict(status=s,at_kst=datetime.now().isoformat(timespec='seconds'),**extra))
    status('loading_cpu_model');torch.set_num_threads(4);torch.manual_seed(c['seed']);device=torch.device('cpu')
    theta=torch.load(checkpoint,map_location='cpu',weights_only=True)['theta'];model_path=root/'checkpoints/clip-vit-base-patch16'
    dataset=CIFAR100(root/'data/cifar-100-python',model_path)
    model=DeepPromptCLIP.from_local_checkpoint(model_path,prompt_layers=c['prompt_layers'],prompt_tokens=c['prompt_tokens'],checkpoint_layers=False).eval()
    text=dataset.text_inputs(device,c['prompt_tokens']);schedule=json.loads(schedule_path.read_text());record=schedule[90]
    with np.load(source/'generated_updates.npz',allow_pickle=False) as a:l,history_u,channels=[a[k].copy() for k in ['learning','unlearning','channels']]
    rate=c['learning_rate']*.5*(1+np.cos(np.pi*90/c['rounds']));total=sum(map(len,record['retained']))
    masses=defaultdict(float)
    for rec in schedule[:90]:
        eta=c['learning_rate']*.5*(1+np.cos(np.pi*(rec['round']-1)/c['rounds']));denom=sum(map(len,rec['retained']))
        for retained,batch in zip(rec['retained'],rec['batches']):
            for z in batch:masses[int(z)]+=eta*len(retained)/denom/len(batch)
    def losses(update,ids):
        value=theta+torch.as_tensor(update,dtype=theta.dtype)
        prob=eq.predict(model,value,dataset,text,ids,'train',device,32)
        return eq.losses(prob,dataset.train_labels[ids])
    guarded=np.zeros_like(history_u);guard_rows=[]
    for user,forgotten in enumerate(record['forget']):
        if not forgotten:continue
        status('backtracking_local_training_losses',user=user)
        batch=record['batches'][user];alpha=len(record['retained'][user])/total;g=-l[user]/(rate*alpha)
        direction=project_direction(history_u[user],g)
        assert float(g@direction)<=1e-10*max(1.,np.linalg.norm(g)*np.linalg.norm(direction))
        fbatch=np.random.default_rng(c['seed']*100000+91*100+user).choice(forgotten,min(c['batch_size'],len(forgotten)),replace=False).tolist()
        weights=np.asarray([len(forgotten)*masses[int(z)] for z in fbatch])
        retained_before=float(losses(np.zeros_like(direction),batch).mean())
        forget_post_learning=float(np.mean(losses(l[user],fbatch)*weights));trials=[];chosen=0.
        for step in steps:
            candidate=l[user]+step*direction;retained=float(losses(candidate,batch).mean())
            forget=float(np.mean(losses(candidate,fbatch)*weights))
            passes=retained<=retained_before+1e-6*(1+abs(retained_before)) and forget>=forget_post_learning-1e-8*(1+abs(forget_post_learning))
            trials.append({'step':step,'retained_training_loss':retained,'weighted_forget_training_loss':forget,'passes_local_guards':bool(passes)})
            if passes:chosen=step;break
        guarded[user]=chosen*direction
        guard_rows.append({'user':user,'selected_step':chosen,'accepted_nonzero_update':bool(np.linalg.norm(guarded[user])>0),
                           'history_direction_norm':float(np.linalg.norm(history_u[user])),'projected_direction_norm':float(np.linalg.norm(direction)),
                           'accepted_update_norm':float(np.linalg.norm(guarded[user])),'retained_training_loss_before':retained_before,
                           'weighted_forget_training_loss_post_learning':forget_post_learning,'trials':trials})
        eq.write_json(out/'guard_results.json',guard_rows);print('GUARD',json.dumps(guard_rows[-1]),flush=True)
    np.savez_compressed(out/'generated_updates.npz',learning=l,unlearning=guarded,channels=channels)
    status('evaluating_fixed_cohorts',nonzero_updates=sum(x['accepted_nonzero_update'] for x in guard_rows))
    quality=root/'results/energy_quality_v1';refmeta=json.loads((quality/'retrain_reference_ideal.cache.json').read_text())
    assert refmeta['checkpoint_sha256']==sha(root/'results/comparisons_seed0_solver_v2/retrain_reference_ideal/latest.pt') and refmeta['evaluator_sha256']==sha(eq.__file__)
    full_groups=json.loads((quality/'cohorts.json').read_text())['groups']
    with np.load(quality/'retrain_reference_ideal.npz') as a:ref={k:a[k].copy() for k in a.files}
    prior_rows=json.loads((source/'results.json').read_text());rows=[x for x in prior_rows if x['method'] in ['old_ideal_step','retained_step_only','history_weighted_ideal_step']]
    value=theta+torch.as_tensor((l+guarded).sum(axis=0),dtype=theta.dtype);prob={};row={'method':'history_guarded_ideal_step'}
    for group,spec in groups.items():
        prob[group]=eq.predict(model,value,dataset,text,spec['ids'],spec['split'],device,32)
        indices={z:i for i,z in enumerate(full_groups[group]['ids'])};target=ref[group][[indices[z] for z in spec['ids']]]
        labels=dataset.train_labels[spec['ids']];row[group]=eq.classification(prob[group],labels)
        row[group]['gap_to_final_retraining']=eq.reference_gap(prob[group],target,labels)
    np.savez_compressed(out/'history_guarded_ideal_step.npz',**prob);rows.append(row);eq.write_json(out/'results.json',rows)
    status('completed_cpu_screen_not_promoted',generated_updates_sha256=sha(out/'generated_updates.npz'),nonzero_request_updates=sum(x['accepted_nonzero_update'] for x in guard_rows))
    print('RESULT',json.dumps(row),flush=True)


if __name__=='__main__':
    try:main()
    except Exception as error:
        if STATUS_PATH is not None:
            temp=STATUS_PATH.with_suffix('.error.partial.json');temp.write_text(json.dumps({'status':'failed_cpu_screen','error':str(error),'at_kst':datetime.now().isoformat(timespec='seconds')},indent=2)+'\n');temp.replace(STATUS_PATH)
        raise
