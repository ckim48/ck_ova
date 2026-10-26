"""Ideal-link application screen for nonlinear fusion, not an energy result."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

STATUS_PATH=None


def fuse_direction(learning,unlearning):
    l=np.asarray(learning,dtype=float);u=np.asarray(unlearning,dtype=float)
    radius=float(np.linalg.norm(l));norm=float(np.linalg.norm(u))
    if radius<=1e-30:return np.zeros_like(u)
    clipped=u*min(1.,radius/max(norm,1e-30))
    return clipped-min(0.,float(l@clipped))/(radius*radius)*l


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    global STATUS_PATH
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,required=True);p.add_argument('--plan-only',action='store_true')
    args=p.parse_args();root=args.project.resolve();sys.path[:0]=[str(root),str(root/'tools/energy_quality_v1')]
    import torch
    from clip_model import DeepPromptCLIP
    from real_data import CIFAR100
    import energy_quality as eq
    source=root/'results/overnight_energy_v2/history_weighted_v6';guard=root/'results/overnight_energy_v2/history_guard_v7'
    assert json.loads((guard/'status.json').read_text())['status']=='completed_cpu_screen_not_promoted'
    prior=json.loads((source/'plan.json').read_text());checkpoint=root/'results/clip/coupled_hybrid_seed0/latest.pt'
    assert sha(checkpoint)==prior['checkpoint_sha256'];expected=prior['source_sha256'];assert all(sha(root/n)==h for n,h in expected.items())
    old=root/'results/overnight_energy_v2/single_step_quality_pilot';groups=json.loads((old/'cohorts.json').read_text());assert sha(old/'cohorts.json')==prior['groups_sha256']
    out=root/'results/overnight_energy_v2/global_fusion_v8';out.mkdir(parents=True,exist_ok=True);STATUS_PATH=out/'status.json'
    plan={'version':'global_fusion_v8_ideal_application_only','script_sha256':sha(__file__),
          'test_sha256':sha(Path(__file__).with_name('test_fusion.py')),'source_sha256':expected,
          'checkpoint_sha256':sha(checkpoint),'history_updates_sha256':sha(source/'generated_updates.npz'),
          'original_updates_sha256':sha(root/'results/failed_round_diagnostic/updates.npz'),
          'groups_sha256':sha(old/'cohorts.json'),
          'fusion':'Clip U to norm(L), then remove negative L-dot-U component. Update=L+adjusted_U. Global fusion uses aggregated L/U; local comparison applies identical operation to each client before summation.',
          'protocol_change':'Nonlinear post-recovery fusion replaces the original simple L+U sum. A single sum alone does not identify its components. Separately recovered dual-analog AirComp must join Hybrid and phase as a strong future comparator; no energy advantage is claimed from this screen.',
          'variants':['local_history_fusion','global_history_fusion','global_originalU_fusion'],
          'limits':['No radio/noise or energy simulation and no GPU promotion. Changed problem, not a replacement of original manuscript/results.',
                    'First-order retained descent is conditional on using an accurate aggregate gradient; it is not a global loss or forgetting guarantee.',
                    'If L=0 then clipped U=0, which does not fulfill a deletion request. Norm cap tied to current learning is an explicit limitation.',
                    'A nonzero fused aggregate does not establish successful handling or forgetting of each individual request.',
                    'One legacy state; fixed200 validation/200forget. Reference inference GPU/pilot CPU. No oracle outputs select fusion coefficients.']}
    path=out/'plan.json'
    if path.exists():assert json.loads(path.read_text())==plan
    eq.write_json(path,plan)
    if args.plan_only:print(json.dumps(plan));return
    def status(s,**extra):eq.write_json(STATUS_PATH,dict(status=s,at_kst=datetime.now().isoformat(timespec='seconds'),**extra))
    status('loading_cpu_model');torch.set_num_threads(4);torch.manual_seed(0);device=torch.device('cpu')
    theta=torch.load(checkpoint,map_location='cpu',weights_only=True)['theta'];c=json.loads((root/'results/failed_round_diagnostic/metadata.json').read_text())['config']
    model_path=root/'checkpoints/clip-vit-base-patch16';dataset=CIFAR100(root/'data/cifar-100-python',model_path)
    model=DeepPromptCLIP.from_local_checkpoint(model_path,prompt_layers=c['prompt_layers'],prompt_tokens=c['prompt_tokens'],checkpoint_layers=False).eval();text=dataset.text_inputs(device,c['prompt_tokens'])
    with np.load(source/'generated_updates.npz') as a:l,h=[a[k].copy() for k in ['learning','unlearning']]
    with np.load(root/'results/failed_round_diagnostic/updates.npz') as a:original_u=a['unlearning'].copy()
    ls=l.sum(axis=0);local=np.stack([fuse_direction(lu,hu) for lu,hu in zip(l,h)])
    global_h=fuse_direction(ls,h.sum(axis=0));global_old=fuse_direction(ls,original_u.sum(axis=0))
    np.savez_compressed(out/'ideal_fusion_vectors.npz',learning=l,history_u=h,local_fused_u=local,global_fused_history_u=global_h,global_fused_original_u=global_old)
    quality=root/'results/energy_quality_v1';cache=json.loads((quality/'retrain_reference_ideal.cache.json').read_text())
    assert cache['checkpoint_sha256']==sha(root/'results/comparisons_seed0_solver_v2/retrain_reference_ideal/latest.pt') and cache['evaluator_sha256']==sha(eq.__file__)
    full_groups=json.loads((quality/'cohorts.json').read_text())['groups']
    with np.load(quality/'retrain_reference_ideal.npz') as a:ref={k:a[k].copy() for k in a.files}
    rows=json.loads((guard/'results.json').read_text())
    for name,adjusted in [('local_history_fusion',local.sum(axis=0)),('global_history_fusion',global_h),('global_originalU_fusion',global_old)]:
        status('predicting_ideal_variant',variant=name);value=theta+torch.as_tensor(ls+adjusted,dtype=theta.dtype);prob={}
        row={'method':name,'learning_norm':float(np.linalg.norm(ls)),'adjusted_unlearning_norm':float(np.linalg.norm(adjusted)),
             'nonzero_unlearning':bool(np.linalg.norm(adjusted)>1e-30),'learning_dot_unlearning':float(ls@adjusted)}
        for group,spec in groups.items():
            prob[group]=eq.predict(model,value,dataset,text,spec['ids'],spec['split'],device,32)
            index={z:i for i,z in enumerate(full_groups[group]['ids'])};target=ref[group][[index[z] for z in spec['ids']]];labels=dataset.train_labels[spec['ids']]
            row[group]=eq.classification(prob[group],labels);row[group]['gap_to_final_retraining']=eq.reference_gap(prob[group],target,labels)
        np.savez_compressed(out/(name+'.npz'),**prob);rows.append(row);eq.write_json(out/'results.json',rows);print('RESULT',json.dumps(row),flush=True)
    status('completed_ideal_application_screen_no_energy_or_gpu_claim')


if __name__=='__main__':
    try:main()
    except Exception as error:
        if STATUS_PATH is not None:STATUS_PATH.write_text(json.dumps({'status':'failed_cpu_screen','error':str(error)},indent=2)+'\n')
        raise
