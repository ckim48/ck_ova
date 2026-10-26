"""Prespecified CPU screen: does separate U fidelity change this one-step VLM?

Original0.2 condition only. The entire70-head radio grid is complete, while the
unrelated0.6 radio grid may continue. Choose the cheapest0.2 dual design using
energy only. Freeze all variants BEFORE inference; no output-based tuning.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from recovery import stream_errors,analytic_sum_rmse

STATUS_PATH=None
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest_array(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def main():
    global STATUS_PATH
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,required=True)
    p.add_argument('--plan-only',action='store_true');args=p.parse_args();root=args.project.resolve()
    sys.path[:0]=[str(root),str(root/'tools/energy_quality_v1')]
    import energy_quality as eq
    from wireless import quantize
    source=root/'results/failed_round_diagnostic';meta=json.loads((source/'metadata.json').read_text());c=meta['config']
    checkpoint=root/'results/clip/coupled_hybrid_seed0/latest.pt';assert sha(checkpoint)==meta['checkpoint_sha256']
    expected=json.loads((root/'results/comparisons_seed0_solver_v2/suite_metadata.json').read_text())['source_sha256']
    assert all(sha(root/n)==v for n,v in expected.items())
    radio=root/'results/overnight_energy_v2/dual_analog_v9';radio_plan=json.loads((radio/'manifest.json').read_text())
    assert all(sha(root/'tools/overnight_energy_v2/dual_analog_v9'/n)==v for n,v in radio_plan['code_sha256'].items())
    selected_rows=[x for x in json.loads((radio/'heads.json').read_text()) if x['relative_tolerance']==.2]
    assert len(selected_rows)==70 and len({(x['learning_time_fraction'],x['head']) for x in selected_rows})==70
    winner=min((x for x in selected_rows if x['status']=='feasible_radio_only'),key=lambda x:x['energy_bound_j'])
    path=radio/f"r0.2_f{winner['learning_time_fraction']:g}_h{winner['head']}.npz"
    with np.load(path) as data:design={k:data[k].copy() for k in data.files}
    with np.load(source/'updates.npz') as data:l,u=[data[k].copy() for k in ['learning','unlearning']]
    b=c['block_dimension'];padding=((0,0),(0,(-l.shape[1])%b))
    blocked=np.stack([np.pad(v,padding).reshape(len(v),-1,b) for v in [l,u]])
    assert np.isclose(analytic_sum_rmse(blocked,design,l.shape[1]),winner['analytic_sum_rmse'],rtol=1e-8)
    ideal=(l+u).sum(axis=0);updates={};roles={};noise_seeds=[91,1091,2091]
    for bits in [8,6]:
        decoded,delta,_=quantize(blocked[1],bits,c['scale_metadata_bits'])
        error=(decoded[design['users']]-blocked[1,design['users']]).sum(axis=0).reshape(-1)[:l.shape[1]]
        name=f'u_codec{bits}_only'
        updates[name]=ideal+error;roles[name]=dict(role='component_control_oracle_L_not_a_physical_transport',quantization_bits=bits,
              unlearning_error_norm=float(np.linalg.norm(error)),u_quantization_rms_bound=float(np.linalg.norm(delta[design['users']].sum(axis=0))))
    for seed in noise_seeds:
        errors=stream_errors(blocked,design,l.shape[1],seed)
        for label,perturbation,role in [
                ('u_analog_only',errors[1],'component_control_oracle_L_not_a_physical_transport'),
                ('full_dual_analog',errors.sum(axis=0),'physical_dual_transport_one_step_screen')]:
            name=f'{label}_{seed}';updates[name]=ideal+perturbation
            roles[name]=dict(role=role,noise_seed=seed,realized_sum_error_norm=float(np.linalg.norm(perturbation)),
                             realized_u_error_norm=float(np.linalg.norm(errors[1])))
            if label=='full_dual_analog':roles[name]['radio_design_energy_j']=winner['energy_bound_j']
    old=root/'results/overnight_energy_v2/single_step_quality_pilot';groups=json.loads((old/'cohorts.json').read_text())
    old_plan=json.loads((old/'manifest.json').read_text())
    assert old_plan['checkpoint_sha256']==sha(checkpoint) and old_plan['updates_sha256']==sha(source/'updates.npz')
    assert old_plan['evaluator_sha256']==sha(eq.__file__)
    quality=root/'results/energy_quality_v1';refmeta=json.loads((quality/'retrain_reference_ideal.cache.json').read_text())
    assert refmeta['checkpoint_sha256']==sha(root/'results/comparisons_seed0_solver_v2/retrain_reference_ideal/latest.pt')
    assert refmeta['cohorts_sha256']==sha(quality/'cohorts.json') and refmeta['evaluator_sha256']==sha(eq.__file__)
    out=root/'results/overnight_energy_v2/u_precision_v12';out.mkdir(exist_ok=True);STATUS_PATH=out/'status.json'
    plan=dict(version='u_precision_v12_original_fixed_state_cpu_screen',source_sha256=expected,
              checkpoint_sha256=sha(checkpoint),updates_sha256=sha(source/'updates.npz'),
              code_sha256={n:sha(Path(__file__).parent/n) for n in ['pilot.py','recovery.py','test_recovery.py']},
              dual_radio_manifest_sha256=sha(radio/'manifest.json'),selected_radio_rows_sha256=hashlib.sha256(json.dumps(selected_rows,sort_keys=True).encode()).hexdigest(),
              selected_radio_design_sha256=sha(path),selected_radio_winner=winner,
              groups_sha256=sha(old/'cohorts.json'),baseline_prediction_sha256=sha(old/'ideal_transport.npz'),
              reference_cache_sha256=sha(quality/'retrain_reference_ideal.npz'),reference_metadata_sha256=sha(quality/'retrain_reference_ideal.cache.json'),
              evaluator_sha256=sha(eq.__file__),noise_seeds=noise_seeds,cpu_threads=4,
              variants={name:dict(update_sha256=digest_array(value),**roles[name]) for name,value in updates.items()},
              question='Compare existing6/8-bit U delivery with actual analog-U perturbations; isolate U using an explicitly ideal-L control, and also evaluate full physical dual recovery. No learned steps/coefficients selected.',
              limits=['One fixed legacy round90 state/one round91 step; fixed200 validation/200forget. No test data.',
                      'Three common noise draws are not independent training seeds. No statistical equivalence or full-trajectory inference.',
                      'Ideal-L component controls have no physical end-to-end energy rank; only full_dual is a feasible transport screen.',
                      'Final retraining cache inference GPU, candidate inference CPU; numeric platform differences possible.',
                      'No new MIA estimate from the small pilot. Full100-round MIA remains in canonical report.',
                      'No small MSE/KL, payload delivery or changed codec-fidelity target is a deletion guarantee. No automatic GPU promotion.'])
    manifest=out/'plan.json'
    if manifest.exists():assert json.loads(manifest.read_text())==plan
    eq.write_json(manifest,plan)
    if args.plan_only:print(json.dumps(plan));return
    import torch
    from clip_model import DeepPromptCLIP
    from real_data import CIFAR100
    def status(value,**extra):eq.write_json(STATUS_PATH,dict(status=value,at_kst=datetime.now().isoformat(timespec='seconds'),**extra))
    status('loading_cpu_model');torch.set_num_threads(4);torch.manual_seed(c['seed']);device=torch.device('cpu')
    np.savez_compressed(out/'update_vectors.npz',**updates)
    theta=torch.load(checkpoint,map_location='cpu',weights_only=True)['theta']
    model_path=root/'checkpoints/clip-vit-base-patch16';dataset=CIFAR100(root/'data/cifar-100-python',model_path)
    model=DeepPromptCLIP.from_local_checkpoint(model_path,prompt_layers=c['prompt_layers'],prompt_tokens=c['prompt_tokens'],checkpoint_layers=False).eval()
    text=dataset.text_inputs(device,c['prompt_tokens'])
    with np.load(old/'ideal_transport.npz') as a:baseline={k:a[k].copy() for k in a.files}
    full_groups=json.loads((quality/'cohorts.json').read_text())['groups']
    with np.load(quality/'retrain_reference_ideal.npz') as a:full_ref={k:a[k].copy() for k in a.files}
    ref={}
    for group,spec in groups.items():
        lookup={z:i for i,z in enumerate(full_groups[group]['ids'])}
        ref[group]=full_ref[group][[lookup[z] for z in spec['ids']]]
    def metric_row(name,prob,extra):
        row=dict(method=name,**extra)
        for group,spec in groups.items():
            labels=dataset.train_labels[spec['ids']];row[group]=eq.classification(prob[group],labels)
            row[group]['gap_to_final_retraining']=eq.reference_gap(prob[group],ref[group],labels)
            row[group]['gap_to_ideal_transport']=eq.reference_gap(prob[group],baseline[group],labels)
        return row
    rows=json.loads((out/'results.json').read_text()) if (out/'results.json').exists() else []
    if not rows:rows.append(metric_row('ideal_transport_reused',baseline,dict(role='ideal_one_step_reference_not_retraining')))
    done={x['method'] for x in rows}
    for name,update in updates.items():
        if name in done:continue
        status('predicting_cpu_variant',variant=name,completed_variants=len(done)-1)
        vector=theta+torch.as_tensor(update,dtype=theta.dtype);prob={}
        for group,spec in groups.items():prob[group]=eq.predict(model,vector,dataset,text,spec['ids'],spec['split'],device,32)
        np.savez_compressed(out/(name+'.npz'),**prob)
        rows.append(metric_row(name,prob,roles[name]));eq.write_json(out/'results.json',rows)
        print('RESULT',json.dumps(rows[-1]),flush=True)
    summary=[]
    for family in ['u_codec8_only','u_codec6_only','u_analog_only','full_dual_analog']:
        group=[x for x in rows if x['method']==family or x['method'].startswith(family+'_')]
        record=dict(family=family,evaluated_variants=len(group),stochastic_noise_draws=sum('noise_seed' in x for x in group))
        for cohort in groups:
            for metric in ['loss','accuracy']:
                values=[x[cohort][metric] for x in group];record[f'{cohort}_{metric}_mean']=float(np.mean(values));record[f'{cohort}_{metric}_noise_std']=float(np.std(values))
            for target in ['final_retraining','ideal_transport']:
                values=[x[cohort]['gap_to_'+target]['kl_reference_to_candidate'] for x in group]
                record[f'{cohort}_kl_to_{target}_mean']=float(np.mean(values));record[f'{cohort}_kl_to_{target}_noise_std']=float(np.std(values))
        summary.append(record)
    eq.write_json(out/'summary.json',summary);status('completed_cpu_precision_sensitivity_no_gpu_promotion',new_variants=len(updates))


if __name__=='__main__':
    try:main()
    except Exception as e:
        if STATUS_PATH is not None:STATUS_PATH.write_text(json.dumps(dict(status='failed_cpu_precision_screen',error=str(e)))+'\n')
        raise
