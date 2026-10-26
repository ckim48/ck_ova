"""Sequential CPU quality check of three all-head convex-refined designs."""
import argparse
from datetime import datetime,timezone,timedelta
import hashlib
import json
from pathlib import Path
import sys
import numpy as np


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,required=True);p.add_argument('--plan-only',action='store_true')
    args=p.parse_args();root=args.project.resolve();f=root/'results/overnight_energy_v2';heads=f/'inverse_heads_v29b';prior=f/'inverse_budget_v29';previous=f/'fu_quality_v22'
    sys.path[:0]=[str(root),str(root/'tools/energy_quality_v1'),str(root/'tools/overnight_energy_v2/inverse_budget_v29'),str(root/'tools/overnight_energy_v2/fu_quality_v22')]
    import energy_quality as eq
    from certificate import check
    from recovery import moments,perturbation
    assert json.loads((heads/'status.json').read_text())['status']=='completed61_budget_vectors_all610head_checks_no_gpu_promotion'
    assert json.loads((f/'certified_error_quality_v28/status.json').read_text())['status']=='completed6_cpu_paid_report_vectors_and18_independent_cohort_checks_no_gpu_promotion'
    summary=json.loads((heads/'summary.json').read_text());assert summary['cases']==61 and summary['head_candidates']==610
    registered=json.loads((prior/'plan.json').read_text());pilot=json.loads((previous/'plan.json').read_text())
    for n,h in registered['core_sha256'].items():assert sha(root/n)==h
    for n,h in registered['code_sha256'].items():assert sha(root/'tools/overnight_energy_v2/inverse_budget_v29'/n)==h
    for n,h in pilot['code_sha256'].items():assert sha(root/'tools/overnight_energy_v2/fu_quality_v22'/n)==h
    assert sha(eq.__file__)==pilot['evaluator_sha256']
    source=f/'public_history_fu_v19/screen';c=json.loads((source/'plan.json').read_text())['config'];groups=pilot['groups']
    with np.load(source/'vectors.npz') as data:
        learning,unlearning,channels=[data[k].copy() for k in ['learning','unlearning','channels']]
        bases={k:data[f'rank{k}_basis'].copy() for k in [1,3]};public_scale=float(data['public_scale'])
    assert sha(source/'vectors.npz')==registered['input_vector_sha256']
    dimension=learning.shape[1];width=c['block_dimension'];block=lambda x:np.pad(x,((0,0),(0,(-dimension)%width))).reshape(len(x),-1,width)
    ideal=(learning+unlearning).sum(axis=0);updates={};roles={};selected=[]
    for row in summary['selected']:
        case=registered['case_inputs'][row['case']];mode=case['mode'];rank=case['rank'];bits=case['bits'];path=root/row['design_path'];assert sha(path)==row['design_sha256']
        with np.load(path) as data:d={k:data[k].copy() for k in data.files}
        cert=check(learning,unlearning,channels,c,bits,mode,registered['epsilon'],d,reports=case['reports'],rank=rank,basis=bases.get(rank),public_scale=public_scale)
        np.testing.assert_allclose(cert['energy_upper_j'],row['refined_energy_j'],rtol=1e-8,atol=1e-12)
        raw=learning+unlearning if mode=='mixed_aircomp' else learning
        if rank:
            coeff=((raw@bases[rank])/public_scale).astype(np.float16).astype(float)*public_scale;raw=raw-coeff@bases[rank].T
        analog=block(raw);digital=block(np.zeros_like(unlearning) if mode=='mixed_aircomp' else unlearning)
        levels=2**(bits-1)-1;scale=(np.max(np.abs(digital),axis=-1)/levels).astype(np.float32).astype(float)
        decoded=np.clip(np.rint(digital/np.where(scale>0,scale,1.)[...,None]),-levels,levels)*scale[...,None]
        np.testing.assert_allclose(moments(analog,digital,decoded,d,dimension)[2],cert['analytic_rmse'],rtol=1e-8)
        selected.append(dict(mode=mode,rank=rank,bits=bits,reports=case['reports'],case=row['case'],head=int(d['head']),energy_bound_j=cert['energy_upper_j'],
            source_design_path=row['design_path'],source_design_sha256=row['design_sha256'],certificate=cert))
        for seed in pilot['noise_seeds']:
            name=f'{mode}_noise{seed}';updates[name]=ideal+perturbation(analog,digital,decoded,d,dimension,seed)
            roles[name]=dict(mode=mode,rank=rank,bits=bits,reports=case['reports'],noise_seed=seed)
    assert len(updates)==9
    out=f/'inverse_quality_v30';out.mkdir(exist_ok=True)
    def write(n,v):
        target=out/n;tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(target)
    plan=dict(version='inverse_quality_v30',actual_round=61,code_sha256=sha(__file__),core_sha256=registered['core_sha256'],
        input_vector_sha256=sha(source/'vectors.npz'),public_checkpoint_sha256=pilot['public_checkpoint_sha256'],previous_pilot_plan_sha256=sha(previous/'plan.json'),
        complete61case_610head_summary_sha256=sha(heads/'summary.json'),convex_source_sha256=registered['code_sha256'],
        selected_by='Lowest independently certified communication energy per architecture from completed all61case/610head sweep, BEFORE new VLM outputs. No task/test-based selection.',
        selected=selected,groups=groups,noise_seeds=pilot['noise_seeds'],cpu_threads=4,
        update_sha256={n:hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for n,v in updates.items()},
        reused_ideal_probability_sha256=sha(previous/'ideal_one_step.npz'),
        baseline_v22_sha256=sha(previous/'results.json'),baseline_v28_sha256=sha(f/'certified_error_quality_v28/results.json'),
        limits=['Nine new one-step vectors: Hybrid/Mixed/phase times3common noises, same actualCPUFU61 updates/publictheta60/200pernon-testcohort. Prior model inferences are reused.',
            'Each selected design respects original global0.2 error/time/power/battery; perblockbudgets are fixed from feasible seeds, not optimized globally. Head enumeration is complete for those61budget vectors, not every possible budget/codec/time.',
            'All quantizer-scale/historycoefficient/error-report RF,time,power and batteries paid. Additional solver/codec/basis/header/circuit computation energy unmeasured.',
            'Report final-retrain100 gaps only descriptively because training stage differs. One actualstep and3noise draws are not fulltrainingseeds, taskqualityequivalence or certified forgetting. No smallpilotMIA, testarrays, GPU/currentqueue changes orautomaticpromotion.',
            'CPU model inference starts before noon only; already-started pilot may finish safely.'])
    if (out/'plan.json').exists():assert json.loads((out/'plan.json').read_text())==plan
    else:write('plan.json',plan)
    if args.plan_only:print(json.dumps(dict(variants=9,selected=[{k:v for k,v in x.items() if k!='certificate'} for x in selected])));return
    assert not (out/'results.json').exists()
    if datetime.now(timezone(timedelta(hours=9)))>=datetime(2026,10,21,12,tzinfo=timezone(timedelta(hours=9))):write('status.json',dict(status='not_started_noon_cutoff'));return
    np.savez_compressed(out/'update_vectors.npz',**updates)
    import torch
    from clip_model import DeepPromptCLIP
    from real_data import CIFAR100
    write('status.json',dict(status='loading_cpu_model',at_kst=datetime.now().isoformat(timespec='seconds')))
    torch.set_num_threads(4);torch.manual_seed(c['seed']);device=torch.device('cpu');checkpoint=source.parent/'observer/round_060.pt'
    assert sha(checkpoint)==pilot['public_checkpoint_sha256'];theta=torch.load(checkpoint,map_location='cpu',weights_only=True)['theta']
    modelpath=root/'checkpoints/clip-vit-base-patch16';dataset=CIFAR100(root/'data/cifar-100-python',modelpath)
    model=DeepPromptCLIP.from_local_checkpoint(modelpath,prompt_layers=c['prompt_layers'],prompt_tokens=c['prompt_tokens'],checkpoint_layers=False).eval()
    text=dataset.text_inputs(device,c['prompt_tokens']);canonical=root/'results/energy_quality_v1';full=json.loads((canonical/'cohorts.json').read_text())['groups'];reference={};labels={}
    assert sha(canonical/'cohorts.json')==pilot['canonical_cohorts_sha256']
    with np.load(canonical/'retrain_reference_ideal.npz') as data:
        for group,spec in groups.items():
            lookup={z:i for i,z in enumerate(full[group]['ids'])};indices=[lookup[z] for z in spec['ids']]
            reference[group]=data[group][indices].copy();labels[group]=data[group+'_labels'][indices].copy()
    with np.load(previous/'ideal_one_step.npz') as data:idealprob={g:data[g].copy() for g in groups}
    rows=[];checks=[]
    for name,update in updates.items():
        write('status.json',dict(status='evaluating_cpu_variant',variant=name,completed_variants=len(rows),total_variants=9,at_kst=datetime.now().isoformat(timespec='seconds')))
        vector=theta+torch.as_tensor(update,dtype=theta.dtype);probs={g:eq.predict(model,vector,dataset,text,s['ids'],s['split'],device,32) for g,s in groups.items()}
        np.savez_compressed(out/(name+'.npz'),**probs);row=dict(method=name,**roles[name])
        for group in groups:
            q=probs[group];n=len(labels[group]);assert np.all(np.isfinite(q)) and np.all(q>0);np.testing.assert_allclose(q.sum(axis=1),1.,rtol=1e-12,atol=1e-14)
            row[group]=eq.classification(q,labels[group]);manual=[float(-np.log(q[np.arange(n),labels[group]]).mean()),float(np.mean(q.argmax(axis=1)==labels[group]))]
            np.testing.assert_allclose(manual,[row[group]['loss'],row[group]['accuracy']],rtol=1e-12,atol=1e-14)
            for label,ref in [('ideal_step',idealprob[group]),('final_retrain_descriptive_only',reference[group])]:
                gap=eq.reference_gap(q,ref,labels[group]);row[group]['gap_to_'+label]=gap
                independent=[max(0.,float(np.mean(np.sum(ref*(np.log(ref)-np.log(q)),axis=1)))),float(np.mean(ref.argmax(axis=1)!=q.argmax(axis=1)))]
                np.testing.assert_allclose(independent,[gap['kl_reference_to_candidate'],gap['prediction_disagreement']],rtol=1e-12,atol=1e-14)
            checks.append(dict(method=name,cohort=group,probability_sha256=sha(out/(name+'.npz')),status='independent_metric_arithmetic_passed'))
        rows.append(row);write('results.json',rows);print('RESULT',json.dumps(row),flush=True)
    summary=[]
    for mode in ['hybrid','mixed_aircomp','phase_grid']:
        entries=[r for r in rows if r['mode']==mode];assert len(entries)==3
        for group in groups:
            metrics={k:float(np.mean([r[group][k] for r in entries])) for k in ['loss','accuracy']}
            for ref in ['ideal_step','final_retrain_descriptive_only']:
                for key in ['kl_reference_to_candidate','prediction_disagreement']:metrics[ref+'_'+key]=float(np.mean([r[group]['gap_to_'+ref][key] for r in entries]))
            summary.append(dict(mode=mode,cohort=group,noise_draws=3,**metrics))
    write('summary.json',dict(selected=selected,metrics=summary,independent_metric_checks=checks,limits=plan['limits']))
    write('status.json',dict(status='completed9_cpu_convex_refined_vectors_and27_independent_cohort_checks_no_gpu_promotion',at_kst=datetime.now().isoformat(timespec='seconds')))


if __name__=='__main__':main()
