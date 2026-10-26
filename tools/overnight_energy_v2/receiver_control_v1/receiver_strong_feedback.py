"""Audit every feedback head and evaluate energy-optimal eligible alternatives.

Zero additional circuit power and .01W sensitivity are both retained. Selecting
by ex-post sketch bias is an optimistic offline comparator, never an online
certificate. No controller parameter is changed and no training is started.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import numpy as np
from receiver_pilot import sha,write
from receiver_control import project_rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,required=True)
    args=p.parse_args();r=args.project.resolve();f=r/'results/overnight_energy_v2';out=f/'receiver_control_v1/strong_feedback';out.mkdir(exist_ok=True)
    def status(s,**extra):write(out/'status.json',dict(status=s,at_kst=datetime.now().isoformat(timespec='seconds'),**extra))
    while True:
        required=[f/'receiver_control_v1/gpu_confirmation/completion.json',f/'receiver_control_v1_request91/gpu_quality/completion.json']
        if all(x.exists() for x in required):
            active=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout.strip()
            if not active:break
        status('waiting_for_previous_evaluations');time.sleep(20)
    sys.path[:0]=[str(r),str(r/'tools/energy_quality_v1'),str(r/'tools/overnight_energy_v2'),str(r/'tools/overnight_energy_v2/initialization_v11')]
    import torch
    from clip_model import DeepPromptCLIP
    from real_data import CIFAR100
    from energy_quality import predict,classification,reference_gap,auc,losses
    from phase_split_grid import make_inputs
    from wireless_init_v11 import Design
    from transport import recover
    torch.set_num_threads(4);torch.manual_seed(0);device=torch.device('cuda')
    selections={}
    for case in ['receiver_control_v1','receiver_control_v1_request91']:
        d=f/case;rows=json.loads((d/'radio_results.json').read_text())
        valid=[x for x in rows if x.get('role')=='head_trial_physical' and x.get('status')=='feasible' and x.get('eligible_same_application_budget')]
        choices={}
        for criterion in ['energy_bound_j','zero_extra_circuit_energy_j']:
            if valid:choices[criterion]=min(valid,key=lambda x:x[criterion])
        selections[case]=choices
    plan=dict(version='receiver_control_v1_strong_feedback',code_sha256=sha(__file__),
        selection='Minimum ENERGY among ALL head trials passing the same ex-post application bound, under both zero additional circuit power and .01W sensitivity. Controller and sketch settings unchanged.',
        choices=selections,limits=['The ex-post sketch bias uses simulation ground truth, so this deliberately optimistic comparator does not establish online QoS feasibility.',
            'The .01W added circuit model is an unmeasured design choice. The zero-additional-circuit comparison avoids claiming advantage from that assumption.',
            'Three common channel-noise draws are not independent training seeds. References are stage-mismatched; MIA/directional proxies do not certify deletion.'])
    write(out/'plan.json',plan)
    base=json.loads((f/'receiver_control_v1/plan.json').read_text());c=base['config'];modelpath=r/'checkpoints/clip-vit-base-patch16'
    assert all(sha(r/n)==v for n,v in base['core_sha256'].items())
    dataset=CIFAR100(r/'data/cifar-100-python',modelpath)
    model=DeepPromptCLIP.from_local_checkpoint(modelpath,prompt_layers=c['prompt_layers'],prompt_tokens=c['prompt_tokens'],checkpoint_layers=True).to(device).eval()
    text=dataset.text_inputs(device,c['prompt_tokens']);canonical=json.loads((r/'results/energy_quality_v1/cohorts.json').read_text())['groups']
    results=[]
    for case,choices in selections.items():
        d=f/case;caseplan=json.loads((d/'plan.json').read_text());groups=caseplan['groups']
        source=(r/'results/overnight_energy_v2/public_history_fu_v19/observer/round_060.pt') if case=='receiver_control_v1' else (r/'results/clip/coupled_hybrid_seed0/latest.pt')
        assert sha(source)==caseplan['checkpoint_sha256'];theta=torch.load(source,map_location='cpu',weights_only=True)['theta']
        with np.load(d/'prepared.npz') as z:L,Q,g,active=[z[n].copy() for n in ['learning','quantized','gradient','active']]
        reference={}
        with np.load(r/'results/energy_quality_v1/retrain_reference_ideal.npz') as z:
            for group,spec in groups.items():
                pos={v:i for i,v in enumerate(canonical[group]['ids'])};reference[group]=z[group][[pos[i] for i in spec['ids']]].copy()
        bytrial={row['name']:row for row in choices.values()}
        for trial,row in bytrial.items():
            match=re.fullmatch(r'feedback_mixed_r(\d+)_(safe_norm|paid_norm)_head(\d+)',trial);assert match
            rank=int(match[1]);head=int(match[3]);generator=np.random.default_rng(1281061)
            projection=(generator.integers(0,2,size=(rank,L.shape[1]),dtype=np.int8)*2-1).astype(float)/np.sqrt(rank)
            ls=L@projection.T;us=Q@projection.T;gs=g@projection.T
            def code(x):
                scale=(np.max(np.abs(x),axis=-1)/32767).astype(np.float32).astype(float)
                return np.rint(x/np.where(scale>0,scale,1)[:,None])*scale[:,None]
            ld=code(ls);ud=code(us);ld[head]=ls[head];ud[head]=us[head]
            norms=np.linalg.norm(Q[active],axis=1).astype(np.float32).astype(float)
            for j,user in enumerate(active):
                if user==head:norms[j]=np.linalg.norm(Q[user])
            A=ld.sum(axis=0);N=np.vstack([ud[active]/norms[:,None],-gs/np.linalg.norm(g)])
            _,dual,_=project_rows(A+ud[active].sum(axis=0),N,np.r_[norms,N[-1]@A])
            weights=(1+dual[:-1]/norms).astype(np.float32).astype(float)
            combo=L.copy();combo[active]+=weights[:,None]*Q[active];correction=-dual[-1]*g/np.linalg.norm(g)
            with np.load(d/'prepared.npz') as z:target=z['control_alpha1'].copy()
            bias=float(np.linalg.norm(combo.sum(axis=0)+correction-target))
            assert abs(bias-row['sketch_bias_norm'])<1e-11
            with np.load(d/(trial+'_design.npz')) as z:fields={n:z[n].copy() for n in z.files}
            design=Design(head=head,energy_j=row['radio_energy_bound_j'],error_bound=row['error_bound'],history=[],chi=0.,mode='mixed_aircomp',**fields)
            lb=make_inputs(combo,np.zeros_like(Q),c)[0];zero=np.zeros_like(lb)
            for seed in base['common_noise_seeds']:
                name=trial+'_noise'+str(seed);status('evaluating',case=case,candidate=name)
                update=recover(design,lb,zero,zero,np.random.default_rng(seed))[:len(theta)]+correction
                value=(theta+torch.as_tensor(update,dtype=theta.dtype)).to(device);probs={}
                for group,spec in groups.items():probs[group]=predict(model,value,dataset,text,spec['ids'],spec['split'],device,32)
                np.savez_compressed(out/(case+'_'+name+'_predictions.npz'),**probs)
                item=dict(case=case,name=name,trial=trial,noise_seed=seed,status='evaluated',
                    selected_for=[k for k,v in choices.items() if v['name']==trial],
                    energy_with_extra_circuit_j=row['energy_bound_j'],energy_zero_extra_circuit_j=row['zero_extra_circuit_energy_j'],
                    request_directional_progress=(Q[active]@update/np.linalg.norm(Q[active],axis=1)**2).tolist())
                for group,spec in groups.items():
                    y=dataset.train_labels[spec['ids']];item[group]=classification(probs[group],y)
                    item[group]['stage_mismatched_retrain100_gap']=reference_gap(probs[group],reference[group],y)
                item['small_cohort_loss_attack_auc']=auc(-losses(probs['forget'],dataset.train_labels[groups['forget']['ids']]),-losses(probs['validation'],dataset.train_labels[groups['validation']['ids']]))
                refauc=auc(-losses(reference['forget'],dataset.train_labels[groups['forget']['ids']]),-losses(reference['validation'],dataset.train_labels[groups['validation']['ids']]))
                item['small_cohort_auc_absolute_gap_to_stage_mismatched_reference']=abs(item['small_cohort_loss_attack_auc']-refauc)
                results.append(item);write(out/'quality.json',results);print('RESULT',json.dumps(item),flush=True)
    status('completed',candidates=len(results));write(out/'completion.json',dict(status='completed',candidate_rows=len(results)))


if __name__=='__main__':main()
