"""Sequential GPU evaluation of request91 physical candidates, after request61."""
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
from receiver_pilot import sha,write


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,required=True)
    args=p.parse_args();r=args.project.resolve()
    family=r/'results/overnight_energy_v2';parent=family/'receiver_control_v1';out=family/'receiver_control_v1_request91';gpu=out/'gpu_quality';gpu.mkdir(exist_ok=True)
    def status(value,**extra):write(gpu/'status.json',dict(status=value,at_kst=datetime.now().isoformat(timespec='seconds'),**extra))
    while True:
        radio=out/'radio_status.json';prior=parent/'gpu_confirmation/completion.json'
        if radio.exists() and json.loads(radio.read_text())['status']=='completed' and prior.exists():
            active=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout.strip()
            if not active:break
        status('waiting_for_radio_and_prior_gpu_completion')
        for pidfile,complete in [(out/'radio_pid.json',radio.exists() and json.loads(radio.read_text())['status']=='completed'),
                                  (parent/'gpu_confirmation/launch_pid.json',prior.exists())]:
            if not complete and pidfile.exists():
                try:os.kill(json.loads(pidfile.read_text())['pid'],0)
                except ProcessLookupError:raise RuntimeError('Required preceding experiment ended before completion: '+str(pidfile))
        time.sleep(20)
    lock=(gpu/'gpu.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    sys.path[:0]=[str(r),str(r/'tools/energy_quality_v1')]
    import torch
    from clip_model import DeepPromptCLIP
    from real_data import CIFAR100
    from energy_quality import predict,classification,reference_gap,auc,losses
    torch.set_num_threads(4);torch.manual_seed(0);device=torch.device('cuda')
    caseplan=json.loads((out/'plan.json').read_text());c=caseplan['config'];groups=caseplan['groups']
    assert all(sha(r/n)==v for n,v in caseplan['core_sha256'].items())
    radio_rows=json.loads((out/'radio_results.json').read_text())
    feedback=[x for x in radio_rows if x.get('role')=='physical_model' and x['name'].startswith('feedback_') and x.get('status')=='feasible' and x.get('eligible_same_application_budget')]
    names=['hybrid','oma_grid']
    if feedback:names.append(min(feedback,key=lambda x:x['energy_bound_j'])['name'])
    plan=dict(version='receiver_control_v1_request91_physical_quality',code_sha256=sha(__file__),
        case_plan_sha256=sha(out/'plan.json'),physical_updates_sha256=sha(out/'physical_updates.npz'),
        groups=groups,selected_names=names,device='cuda',noise_seeds=caseplan['common_noise_seeds'],
        selection_rule='Hybrid/OMA and energy-only minimum among same-budget feasible paid-feedback Mixed; fixed before physical inference.',
        limits=caseplan['assumptions']+['Small-cohort MIA is descriptive only. No statistical superiority, deletion certificate or matched-stage retrain reference.'])
    write(gpu/'plan.json',plan)
    checkpoint=r/'results/clip/coupled_hybrid_seed0/latest.pt';assert sha(checkpoint)==caseplan['checkpoint_sha256']
    theta=torch.load(checkpoint,map_location='cpu',weights_only=True)['theta']
    modelpath=r/'checkpoints/clip-vit-base-patch16';dataset=CIFAR100(r/'data/cifar-100-python',modelpath)
    model=DeepPromptCLIP.from_local_checkpoint(modelpath,prompt_layers=c['prompt_layers'],prompt_tokens=c['prompt_tokens'],checkpoint_layers=True).to(device).eval()
    text=dataset.text_inputs(device,c['prompt_tokens'])
    canonical=json.loads((r/'results/energy_quality_v1/cohorts.json').read_text())['groups']
    reference={}
    with np.load(r/'results/energy_quality_v1/retrain_reference_ideal.npz') as z:
        for group,spec in groups.items():
            pos={v:i for i,v in enumerate(canonical[group]['ids'])}
            reference[group]=z[group][[pos[i] for i in spec['ids']]].copy()
    with np.load(out/'physical_updates.npz') as z:
        updates={k:z[k].copy() for k in z.files if any(k.startswith(name+'_noise') for name in names)}
    with np.load(out/'prepared.npz') as z:
        anchors=z['quantized'][z['active']];g=z['gradient'];learning=z['learning'].sum(axis=0)
    rows=[]
    for name,update in updates.items():
        status('evaluating',candidate=name,completed=len(rows),total=len(updates))
        value=(theta+torch.as_tensor(update,dtype=theta.dtype)).to(device);probs={}
        for group,spec in groups.items():
            print('PREDICT',name,group,flush=True)
            probs[group]=predict(model,value,dataset,text,spec['ids'],spec['split'],device,32)
        np.savez_compressed(gpu/(name+'_predictions.npz'),**probs)
        row=dict(name=name,status='evaluated',request_directional_progress=(anchors@update/np.linalg.norm(anchors,axis=1)**2).tolist(),
            retention_directional_change_relative_to_true_learning=float(g@(update-learning)),
            requests_transmitted=len(anchors),deletion_certified=False)
        for group,spec in groups.items():
            y=dataset.train_labels[spec['ids']]
            row[group]=classification(probs[group],y)
            row[group]['stage_mismatched_retrain100_gap']=reference_gap(probs[group],reference[group],y)
        row['small_cohort_loss_attack_auc']=auc(-losses(probs['forget'],dataset.train_labels[groups['forget']['ids']]),
            -losses(probs['validation'],dataset.train_labels[groups['validation']['ids']]))
        refauc=auc(-losses(reference['forget'],dataset.train_labels[groups['forget']['ids']]),
            -losses(reference['validation'],dataset.train_labels[groups['validation']['ids']]))
        row['small_cohort_auc_absolute_gap_to_stage_mismatched_reference']=abs(row['small_cohort_loss_attack_auc']-refauc)
        rows.append(row);write(gpu/'quality.json',rows);print('RESULT',json.dumps(row),flush=True)
    write(gpu/'completion.json',dict(status='completed',candidates=len(rows),existing_sources_intact=all(sha(r/n)==v for n,v in caseplan['core_sha256'].items())))
    status('completed',candidates=len(rows))


if __name__=='__main__':main()
