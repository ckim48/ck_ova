"""CPU screen of one-step transport perturbations, NOT a retraining audit."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,required=True)
    parser.add_argument('--samples',type=int,default=200)
    parser.add_argument('--cpu-threads',type=int,default=2)
    parser.add_argument('--plan-only',action='store_true')
    args=parser.parse_args();root=args.project.resolve()
    sys.path[:0]=[str(root),str(root/'tools/energy_quality_v1')]
    import torch
    from clip_model import DeepPromptCLIP
    from real_data import CIFAR100
    from transport import design_transport,recover,energy_accounting
    from phase_split_grid import optimize_fraction
    from energy_quality import predict,classification,reference_gap,stratified_sample,digest,write_json
    source=root/'results/failed_round_diagnostic'
    metadata=json.loads((source/'metadata.json').read_text());config=metadata['config']
    checkpoint=root/'results/clip/coupled_hybrid_seed0/latest.pt'
    assert digest(checkpoint)==metadata['checkpoint_sha256'],'Legacy checkpoint changed.'
    expected=json.loads((root/'results/comparisons_seed0_solver_v2/suite_metadata.json').read_text())['source_sha256']
    assert all(digest(root/n)==sha for n,sha in expected.items())
    output=root/'results/overnight_energy_v2/single_step_quality_pilot';output.mkdir(parents=True,exist_ok=True)
    settings={'samples_per_group':args.samples,'groups':['validation','forget'],
        'noise_seeds':[91,1091,2091],'methods':[
            {'name':'hybrid_default','mode':'hybrid','relative_tolerance':0.2},
            {'name':'hybrid_loose','mode':'hybrid','relative_tolerance':0.6},
            {'name':'phase_loose','mode':'orthogonal','relative_tolerance':0.6,'analog_fraction':0.125},
            {'name':'mixed_loose','mode':'mixed_aircomp','relative_tolerance':0.6}],
        'quantization_bits':8,'checkpoint_sha256':digest(checkpoint),
        'updates_sha256':digest(source/'updates.npz'),'training_source_sha256':expected,
        'evaluator_sha256':digest(root/'tools/energy_quality_v1/energy_quality.py'),
        'script_sha256':digest(__file__),
        'interpretation':'Exploratory CPU one-step screen from a legacy round-90 checkpoint and reconstructed actual round-91 updates. Ideal one-step output is a TRANSPORT reference, not an unlearned/retrained model. Three noise draws are not independent training seeds. No test images are used. Small loss/KL here does not prove preserved long-run utility or forgetting.'}
    path=output/'manifest.json'
    if path.exists() and json.loads(path.read_text())!=settings:raise ValueError('Existing pilot differs; use a new version.')
    write_json(path,settings)
    if args.plan_only:
        print(json.dumps(settings,indent=2));return
    write_json(output/'status.json',{'status':'running','started_at_kst':datetime.now().isoformat(timespec='seconds')})
    torch.set_num_threads(args.cpu_threads)
    device=torch.device('cpu')
    model_path=root/'checkpoints/clip-vit-base-patch16'
    dataset=CIFAR100(root/'data/cifar-100-python',model_path)
    frozen=json.loads((root/'results/energy_quality_v1/cohorts.json').read_text())
    rng=np.random.default_rng(20261021)
    groups={name:{'split':'train','ids':stratified_sample(frozen['groups'][name]['ids'],dataset.train_labels,args.samples,rng)} for name in settings['groups']}
    write_json(output/'cohorts.json',groups)
    with np.load(source/'updates.npz') as data:
        l,u,channels=[data[k].copy() for k in ('learning','unlearning','channels')]
    original=torch.load(checkpoint,map_location='cpu',weights_only=True)
    assert original['round']==90
    theta=original['theta'];ideal_update=(l+u).sum(axis=0)
    model=DeepPromptCLIP.from_local_checkpoint(model_path,prompt_layers=config['prompt_layers'],
        prompt_tokens=config['prompt_tokens'],checkpoint_layers=False).eval()
    text=dataset.text_inputs(device,config['prompt_tokens'])
    metrics_path=output/'results.json';rows=json.loads(metrics_path.read_text()) if metrics_path.exists() else []
    done={r['name'] for r in rows}
    reference={}
    def evaluate(name,update,extra):
        vector=theta+torch.as_tensor(update,dtype=theta.dtype)
        cache=output/(name+'.npz')
        if cache.exists():
            with np.load(cache) as a:probabilities={k:a[k].copy() for k in a.files}
        else:
            probabilities={}
            for group,spec in groups.items():
                print('PREDICT',name,group,len(spec['ids']),flush=True)
                probabilities[group]=predict(model,vector,dataset,text,spec['ids'],spec['split'],device,32)
            np.savez_compressed(cache,**probabilities)
        if name=='ideal_transport':reference.update(probabilities)
        row={'name':name,**extra}
        for group,spec in groups.items():
            labels=dataset.train_labels[spec['ids']]
            row[group]=classification(probabilities[group],labels)
            if reference:
                row[group]['gap_to_ideal_transport']=reference_gap(probabilities[group],reference[group],labels)
        if name not in done:
            rows.append(row);write_json(metrics_path,rows);done.add(name)
        print('RESULT',json.dumps(row),flush=True)
    evaluate('ideal_transport',ideal_update,{'role':'one_step_transport_reference_not_retraining'})
    evaluate('before_step',np.zeros_like(ideal_update),{'role':'pre_request_step_signal_size'})
    for setting in settings['methods']:
        c=dict(config,relative_tolerance=setting['relative_tolerance'])
        b=c['block_dimension'];bl=np.pad(l,((0,0),(0,(-l.shape[1])%b))).reshape(len(l),-1,b)
        epsilon=max(c['absolute_tolerance_floor'],c['relative_tolerance']*np.linalg.norm(np.linalg.norm(bl,axis=-1).sum(axis=0)))
        if all(setting['name']+'_'+str(seed) in done for seed in settings['noise_seeds']):continue
        if setting['mode']=='orthogonal':
            design,inputs,_=optimize_fraction(l,u,channels,c,epsilon,setting['analog_fraction'])
            blocked_l,blocked_u,decoded=inputs[:3]
        else:
            design,blocked_l,blocked_u,decoded,_=design_transport(l,u,channels,c,setting['mode'],epsilon)
        if design is None:
            row={'name':setting['name'],'status':'no_feasible_design_found'}
            rows.append(row);write_json(metrics_path,rows);print(json.dumps(row),flush=True);continue
        stats=energy_accounting(design,c)
        for seed in settings['noise_seeds']:
            name=setting['name']+'_'+str(seed)
            if name in done:continue
            update=recover(design,blocked_l,blocked_u,decoded,np.random.default_rng(seed))[:l.shape[1]]
            evaluate(name,update,{'status':'feasible','noise_seed':seed,'relative_tolerance':c['relative_tolerance'],**stats})
    summary=[]
    for setting in settings['methods']:
        values=[r for r in rows if r['name'].startswith(setting['name']+'_') and r.get('status')=='feasible']
        if not values:continue
        item={'method':setting['name'],'draws':len(values),'energy_bound_j':values[0]['energy_bound_j']}
        for group in settings['groups']:
            for key in ('loss','accuracy'):
                a=[r[group][key] for r in values];item[group+'_'+key+'_mean']=float(np.mean(a));item[group+'_'+key+'_noise_std']=float(np.std(a))
            for key in ('kl_reference_to_candidate','prediction_disagreement','signed_loss_gap'):
                a=[r[group]['gap_to_ideal_transport'][key] for r in values]
                item[group+'_'+key+'_mean']=float(np.mean(a));item[group+'_'+key+'_noise_std']=float(np.std(a))
        summary.append(item)
    write_json(output/'summary.json',summary)
    write_json(output/'status.json',{'status':'completed','completed_at_kst':datetime.now().isoformat(timespec='seconds')})
    print('SUMMARY',json.dumps(summary),flush=True)


if __name__=='__main__':main()
