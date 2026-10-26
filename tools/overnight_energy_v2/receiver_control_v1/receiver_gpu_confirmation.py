"""GPU confirmation after the previous campaign finishes; original pilot preserved.

Adds a second pre-existing actual request91 without parameter tuning. Exact
duplicate model vectors share predictions. No full training is started.
"""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from receiver_pilot import sha, write
from receiver_control import control


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,required=True)
    args=p.parse_args();root=args.project.resolve();scripts=Path(__file__).parent
    parent=root/'results/overnight_energy_v2/receiver_control_v1'
    out=parent/'gpu_confirmation';out.mkdir(exist_ok=True)
    lock=(out/'gpu.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    old=json.loads((root/'results/overnight_energy_v2/burst_joint_v3/status.json').read_text())
    assert old['status']=='completed','Previous training and essential audit must finish first.'
    active=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout.strip()
    assert not active,'Do not overlap another GPU computation.'
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
        try:command=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        if str(root) in command and any(n in command for n in ['run_variant.py','run_clip_federated.py','audit_variants.py']):
            raise RuntimeError('Previous project GPU process still active: '+proc.name)
    sys.path[:0]=[str(root),str(root/'tools/energy_quality_v1'),str(root/'tools/overnight_energy_v2'),str(root/'tools/overnight_energy_v2/initialization_v11')]
    import torch
    import torch.nn.functional as functional
    from clip_model import DeepPromptCLIP
    from real_data import CIFAR100
    from energy_quality import predict,classification,reference_gap,stratified_sample,auc,losses
    from phase_split_grid import make_inputs
    from transport import recover
    from wireless_init_v11 import Design
    torch.set_num_threads(4);torch.manual_seed(0);device=torch.device('cuda')
    baseplan=json.loads((parent/'plan.json').read_text());c=baseplan['config']
    assert all(sha(root/n)==v for n,v in baseplan['core_sha256'].items())
    assert all(sha(scripts/n)==v for n,v in baseplan['code_sha256'].items())
    modelpath=root/'checkpoints/clip-vit-base-patch16'
    dataset=CIFAR100(root/'data/cifar-100-python',modelpath)
    canonical=json.loads((root/'results/energy_quality_v1/cohorts.json').read_text())['groups']
    legacy=root/'results/failed_round_diagnostic';meta=json.loads((legacy/'metadata.json').read_text())
    legacy_state=root/'results/clip/coupled_hybrid_seed0/latest.pt'
    assert sha(legacy_state)==meta['checkpoint_sha256']
    record=json.loads((legacy_state.parent/'schedule.json').read_text())[90];assert record['round']==91
    assert not set(baseplan['calibration_ids']).intersection(i for key in ['retained','forget'] for ids in record[key] for i in ids)
    groups61=baseplan['groups'];groups91={'validation':groups61['validation']}
    for group,key in [('retained','retained'),('forget','forget')]:
        available=set(i for ids in record[key] for i in ids)
        pool=sorted(available.intersection(canonical[group]['ids']))
        ids=stratified_sample(pool,dataset.train_labels,min(200,len(pool)),np.random.default_rng(9161061))
        assert len(ids)>0
        groups91[group]=dict(ids=ids,split='train')
    plan=dict(version='receiver_control_v1_gpu_confirmation',base_plan_sha256=sha(parent/'plan.json'),
        code_sha256=sha(__file__),device='cuda',primary_alpha=1.,sensitivity_alphas=[.25,.5],
        cases=[dict(name='request61',checkpoint_sha256=baseplan['checkpoint_sha256'],groups=groups61),
               dict(name='request91_legacy',checkpoint_sha256=sha(legacy_state),vector_sha256=sha(legacy/'updates.npz'),groups=groups91)],
        calibration_ids=baseplan['calibration_ids'],noise_seeds=baseplan['common_noise_seeds'],
        physical_selection_rule='Hybrid, optimized phase OMA, minimum-energy half-budget-eligible paid feedback Mixed, plus ordinary sum Mixed when its deterministic ideal equals the controller target.',
        limits=baseplan['assumptions']+['Request61 and legacy91 use different observed histories; not two independent training seeds or a paired trajectory.',
            'CPU ideal inference was superseded solely for speed after previous GPU campaign completed; original sources/logs/results preserved.',
            'Candidate model parameters are deduplicated by exact float32 bytes before inference. Identical candidates receive identical predictions.',
            'No full training, no test use, and no settings selected from forget/test output.'])
    if (out/'plan.json').exists():assert json.loads((out/'plan.json').read_text())==plan
    else:write(out/'plan.json',plan)
    def status(value,**extra):write(out/'status.json',dict(status=value,at_kst=datetime.now().isoformat(timespec='seconds'),**extra))
    model=DeepPromptCLIP.from_local_checkpoint(modelpath,prompt_layers=c['prompt_layers'],prompt_tokens=c['prompt_tokens'],checkpoint_layers=True).to(device).eval()
    text=dataset.text_inputs(device,c['prompt_tokens'])
    canonical_cache=root/'results/energy_quality_v1/retrain_reference_ideal.npz'
    allrows=[];cache={}
    def evaluate(case,theta,groups,updates):
        references={}
        if canonical_cache.exists():
            with np.load(canonical_cache) as z:
                for group,spec in groups.items():
                    pos={v:i for i,v in enumerate(canonical[group]['ids'])}
                    references[group]=z[group][[pos[i] for i in spec['ids']]].copy()
        for name,update in updates.items():
            key=case+'_'+name
            value=theta+torch.as_tensor(update,dtype=theta.dtype)
            fingerprint=__import__('hashlib').sha256(value.numpy().tobytes()).hexdigest()
            cachekey=(case,fingerprint)
            status('evaluating',case=case,candidate=name,completed=len(allrows))
            if cachekey in cache:
                origin,probs=cache[cachekey]
            else:
                vector=value.to(device);probs={}
                for group,spec in groups.items():
                    print('PREDICT',key,group,len(spec['ids']),flush=True)
                    probs[group]=predict(model,vector,dataset,text,spec['ids'],spec['split'],device,32)
                origin=key;cache[cachekey]=(origin,probs)
                np.savez_compressed(out/(key+'_predictions.npz'),**probs)
            row=dict(case=case,name=name,model_sha256=fingerprint,prediction_source=origin,
                update_norm=float(np.linalg.norm(update)),status='evaluated')
            for group,spec in groups.items():
                labels=dataset.train_labels[spec['ids']]
                row[group]=classification(probs[group],labels)
                if references:
                    row[group]['stage_mismatched_retrain100_gap']=reference_gap(probs[group],references[group],labels)
            row['small_cohort_loss_attack_auc']=auc(-losses(probs['forget'],dataset.train_labels[groups['forget']['ids']]),
                -losses(probs['validation'],dataset.train_labels[groups['validation']['ids']]))
            if references:
                refauc=auc(-losses(references['forget'],dataset.train_labels[groups['forget']['ids']]),
                    -losses(references['validation'],dataset.train_labels[groups['validation']['ids']]))
                row['small_cohort_auc_absolute_gap_to_stage_mismatched_reference']=abs(row['small_cohort_loss_attack_auc']-refauc)
            allrows.append(row);write(out/'quality.json',allrows);print('RESULT',json.dumps(row),flush=True)
    statepath=root/'results/overnight_energy_v2/public_history_fu_v19/observer/round_060.pt'
    theta61=torch.load(statepath,map_location='cpu',weights_only=True)['theta']
    with np.load(parent/'prepared.npz') as z:
        updates61={k:z[k].copy() for k in z.files if k.startswith('control_') or k in ['before_step','learning_only','raw_original_sum','canonical_sum']}
    evaluate('request61',theta61,groups61,updates61)
    theta91=torch.load(legacy_state,map_location='cpu',weights_only=True)['theta']
    with np.load(legacy/'updates.npz') as z:l91,u91=[z[k].copy() for k in ['learning','unlearning']]
    status('calibrating_request91')
    grad=torch.zeros_like(theta91,device=device)
    for start in range(0,len(plan['calibration_ids']),4):
        ids=plan['calibration_ids'][start:start+4]
        value=theta91.to(device).detach().requires_grad_(True)
        pixels,labels=dataset.batch(ids,device,'train')
        loss=functional.cross_entropy(model(pixels,**text,theta=value),labels)
        grad+=torch.autograd.grad(loss,value)[0].detach()*len(ids)/len(plan['calibration_ids'])
    g91=grad.cpu().numpy().astype(float);input91=make_inputs(l91,u91,c)
    q91=input91[2].reshape(len(l91),-1)[:,:l91.shape[1]];active91=np.flatnonzero(np.linalg.norm(u91,axis=1)>1e-15)
    A91=l91.sum(axis=0);anchors91=q91[active91]
    updates91=dict(before_step=np.zeros_like(A91),learning_only=A91,raw_original_sum=(l91+u91).sum(axis=0),canonical_sum=A91+anchors91.sum(axis=0))
    geom=dict(active_users=active91.tolist(),controls={},
        cos_learning_requests=(anchors91@A91/(np.linalg.norm(anchors91,axis=1)*np.linalg.norm(A91))).tolist(),
        cos_retention_requests=(anchors91@g91/(np.linalg.norm(anchors91,axis=1)*np.linalg.norm(g91))).tolist())
    for alpha,name in [(1.,'control_alpha1'),(.25,'control_alpha025'),(.5,'control_alpha05')]:
        try:updates91[name],geom['controls'][name]=control(A91,anchors91,g91,alpha)
        except ValueError as e:geom['controls'][name]=dict(status='no_feasible_projection_found',reason=str(e))
    np.savez_compressed(out/'request91_prepared.npz',gradient=g91,learning=l91,quantized=q91,**updates91)
    write(out/'request91_geometry.json',geom);print('GEOMETRY91',json.dumps(geom),flush=True)
    evaluate('request91_legacy',theta91,groups91,updates91)
    while True:
        state=json.loads((parent/'radio_status.json').read_text())
        if state['status']=='completed':break
        status('waiting_for_radio',radio=state)
        pid=json.loads((parent/'radio_pid.json').read_text())['pid']
        try:os.kill(pid,0)
        except ProcessLookupError:raise RuntimeError('Radio process ended before completion.')
        time.sleep(20)
    rows=json.loads((parent/'radio_results.json').read_text())
    feedback=[x for x in rows if x.get('role')=='physical_model' and x['name'].startswith('feedback_') and x.get('status')=='feasible' and x.get('eligible_same_application_budget')]
    names=['hybrid','oma_grid']
    if feedback:names.append(min(feedback,key=lambda x:x['energy_bound_j'])['name'])
    physical={}
    with np.load(parent/'physical_updates.npz') as z:
        for k in z.files:
            if any(k.startswith(name+'_noise') for name in names):physical[k]=z[k].copy()
    # At this event controller is exactly inactive. Plain sum AirComp then has
    # the same deterministic target, with no oracle feedback or sketches needed.
    if np.array_equal(updates61['canonical_sum'],updates61['control_alpha1']):
        genie=next(x for x in rows if x['name']=='genie_mixed_reference')
        if genie['status']=='feasible':
            with np.load(parent/'genie_mixed_reference_design.npz') as z:fields={k:z[k].copy() for k in z.files}
            des=Design(head=genie['head'],energy_j=genie['radio_energy_bound_j'],error_bound=genie['error_bound'],history=[],chi=0.,mode='mixed_aircomp',**fields)
            with np.load(parent/'prepared.npz') as z:L,Q=[z[k].copy() for k in ['learning','quantized']]
            lb=make_inputs(L+Q,np.zeros_like(Q),c)[0];zero=np.zeros_like(lb)
            for seed in plan['noise_seeds']:
                physical['ordinary_sum_mixed_noise'+str(seed)]=recover(des,lb,zero,zero,np.random.default_rng(seed))[:len(theta61)]
            names.append('ordinary_sum_mixed')
            write(out/'ordinary_sum_mixed_identity.json',dict(event=61,exact_target_equal=True,
                explanation='All primary controller duals are zero. Ordinary sum L+Q(U) equals the new deterministic target at this event; no receiver-derived weights are needed.',
                energy_bound_j=genie['energy_bound_j'],frame_s=genie['total_frame_s'],error_bound=genie['error_bound']))
    write(out/'physical_selection.json',dict(names=names,rule=plan['physical_selection_rule']))
    np.savez_compressed(out/'physical_updates.npz',**physical)
    evaluate('request61',theta61,groups61,physical)
    write(out/'completion.json',dict(status='completed',candidate_rows=len(allrows),unique_predictions=len(cache),
        existing_training_sources_intact=all(sha(root/n)==v for n,v in baseplan['core_sha256'].items())))
    status('completed',candidate_rows=len(allrows),unique_predictions=len(cache));print('COMPLETED',flush=True)


if __name__=='__main__':main()
