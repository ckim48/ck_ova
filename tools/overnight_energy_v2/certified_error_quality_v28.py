"""CPU task check of paid quantization-error reporting at actualFU61."""
import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import sys
import numpy as np


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True); parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args(); root = args.project.resolve(); family = root/'results/overnight_energy_v2'
    sys.path[:0] = [str(root), str(root/'tools/energy_quality_v1'), str(root/'tools/overnight_energy_v2/certified_error_v27'), str(root/'tools/overnight_energy_v2/fu_quality_v22')]
    import energy_quality as eq
    from certificate import check
    from recovery import moments, perturbation
    low = family/'certified_error_v27'; previous = family/'fu_quality_v22'; canonical = root/'results/energy_quality_v1'
    assert json.loads((low/'status.json').read_text())['status'] == 'completed27_paid_error_certificate_radio_cases_no_gpu_promotion'
    assert json.loads((family/'fu_quality_audit_v25/status.json').read_text())['status'] == 'passed19_vectors_and57_cohort_predictions_and6_energy_only_selections'
    registered = json.loads((low/'plan.json').read_text()); pilot = json.loads((previous/'plan.json').read_text())
    for n, value in registered['core_sha256'].items(): assert sha(root/n) == value
    for n, value in registered['code_sha256'].items(): assert sha(root/'tools/overnight_energy_v2/certified_error_v27'/n) == value
    for n, value in pilot['code_sha256'].items(): assert sha(root/'tools/overnight_energy_v2/fu_quality_v22'/n) == value
    assert sha(root/'tools/energy_quality_v1/energy_quality.py') == pilot['evaluator_sha256']
    source = family/'public_history_fu_v19/screen'; c = registered['config']; groups = pilot['groups']; epsilon = registered['epsilon']
    with np.load(source/'vectors.npz') as data: learning, unlearning, channels = [data[k].copy() for k in ['learning', 'unlearning', 'channels']]
    assert sha(source/'vectors.npz') == registered['input_vector_sha256']
    rows = json.loads((low/'results.json').read_text()); assert len(rows) == 27
    certificates = json.loads((low/'certificates.json').read_text())
    assert len(certificates) == sum(r['status']=='feasible_radio_only' for r in rows)
    old = json.loads((family/'public_history_fu_v19/radio/results.json').read_text())
    selected = []; updates = {}; physical = []; base_rows = {}
    no_report = family/'fu_precision_quality_v26'
    assert json.loads((no_report/'status.json').read_text())['status'].startswith('completed6_cpu_lowbit_vectors')
    no_report_radio = json.loads((family/'fu_precision_v24/results.json').read_text())
    assert len(no_report_radio) == 24
    dimension = learning.shape[1]; width = c['block_dimension']; ideal_update = (learning+unlearning).sum(axis=0)
    block = lambda x: np.pad(x, ((0, 0), (0, (-dimension) % width))).reshape(len(x), -1, width)
    analog, digital = block(learning), block(unlearning)
    for mode in ['hybrid', 'phase_grid']:
        old_eligible = [x for x in old if x['relative_tolerance'] == .2 and x['rank'] == 0 and x['mode'] == mode and x['status'] == 'feasible_radio_only']
        baseline = min(old_eligible, key=lambda x: x['energy_bound_j']); base_rows[mode] = baseline
        eligible = [x for x in rows if x['mode'] == mode and x['status'] == 'feasible_radio_only']
        baseline6 = min([x for x in no_report_radio if x['quantization_bits']==6 and x['mode']==mode and x['status']=='feasible_radio_only'], key=lambda x:x['energy_bound_j'])
        chosen = min(eligible, key=lambda x: x['energy_bound_j'])
        assert chosen['energy_bound_j'] < min(baseline['energy_bound_j'],baseline6['energy_bound_j']), 'No radio improvement; do not silently run extra model inference.'
        bits = chosen['quantization_bits']; assert bits == 6, 'Current registered energy winner expected6bit; no silent protocol change.'
        path = low/f"q{bits}_{mode}_f{chosen['analog_fraction']}_design.npz"
        assert any(z['design_sha256']==sha(path) for z in certificates)
        with np.load(path) as data: d = {k: data[k].copy() for k in data.files}
        certificate = check(learning, unlearning, channels, c, bits, mode, epsilon, d, chosen)
        physical.append(dict(mode=mode, quantization_bits=bits, **certificate))
        levels = 2**(bits-1)-1; scale = (np.max(np.abs(digital), axis=-1)/levels).astype(np.float32).astype(float)
        decoded = np.clip(np.rint(digital/np.where(scale > 0, scale, 1.)[..., None]), -levels, levels)*scale[..., None]
        np.testing.assert_allclose(moments(analog, digital, decoded, d, dimension)[2], chosen['analytic_total_rmse'], rtol=1e-8)
        selected.append(dict(mode=mode, quantization_bits=bits, design_path=str(path.relative_to(root)), design_sha256=sha(path),
            energy_bound_j=chosen['energy_bound_j'], baseline8bit_energy_bound_j=baseline['energy_bound_j'],
            energy_reduction_fraction=1-chosen['energy_bound_j']/baseline['energy_bound_j'], baseline6bit_energy_bound_j=baseline6['energy_bound_j'],
            further_reduction_vs6bit_fraction=1-chosen['energy_bound_j']/baseline6['energy_bound_j'], report_rf_energy_j=chosen['report_rf_energy_j'],
            report_payload_bits=chosen['report_payload_bits'], report_control_symbols=chosen['report_control_symbols'], analytic_rmse=chosen['analytic_total_rmse']))
        for seed in pilot['noise_seeds']:
            name = f'{mode}_q{bits}_noise{seed}'; updates[name] = ideal_update+perturbation(analog, digital, decoded, d, dimension, seed)
    assert len(updates) == 6
    out = family/'certified_error_quality_v28'; out.mkdir(exist_ok=True)
    def write(name, value):
        target = out/name; tmp = target.with_suffix('.tmp'); tmp.write_text(json.dumps(value, indent=2)+'\n'); tmp.replace(target)
    plan = dict(version='certified_error_quality_v28', actual_round=61, code_sha256=sha(__file__), config=c,
        core_sha256=registered['core_sha256'], input_vector_sha256=sha(source/'vectors.npz'), public_checkpoint_sha256=pilot['public_checkpoint_sha256'],
        complete27case_radio_sha256=sha(low/'results.json'), lowbit_plan_sha256=sha(low/'plan.json'), independent_v25_audit_sha256=sha(family/'fu_quality_audit_v25/results.json'),
        previous_pilot_plan_sha256=sha(previous/'plan.json'), baseline_quality_rows_sha256=sha(previous/'results.json'), no_report6_quality_rows_sha256=sha(no_report/'results.json'),
        no_report6_radio_sha256=sha(family/'fu_precision_v24/results.json'),
        selected_by='Minimum modeled energy from ALL27paid-report cases per architecture; require improvement over completed no-report6/8bit incumbents before inference. Reported packet RF/time/power/battery already included. Mixed and rank3 baselines remain eligible without new inference.',
        selected=selected, physical_certificates=physical, groups=groups, noise_seeds=pilot['noise_seeds'], cpu_threads=4,
        update_sha256={n: hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for n, v in updates.items()},
        reused_ideal_probabilities_sha256=sha(previous/'ideal_one_step.npz'),
        limits=['Six new one-step vectors;19existingV22 and6V26 vectors/predictions are reused, not rerun. Same actualFU61 updates/theta60,200validation/200retained/200currentDF61forget, common61/1061/2061noise.',
            'Same original absolute error/time/power/battery. Extra1312bit/remote-requester error reports cost2624symbols and RF energy, subtracted from original frame, plus32bit quantizer scales. Strong7fractionphase includes exact minimum-dimension boundary. Complete27case grid and independent certificates precede inference.',
            'One actual step and3commonnoise draws are not trainingseeds, full-trajectory quality equivalence or deletion certification. Final retained-retrain100 differs in stage and its gap is descriptive.',
            'No test arrays/selection, no new MIA from200samples, no GPU work or automatic long-training promotion. Extra encoder/codec/header/compute energy unmeasured.',
            'Separate fixed-bit candidate protocols, not online bit adaptation using several free quantization-error profiles. Both original worst-bound protocols and Mixed remain comparators.',
            'A new CPU model pilot begins before noon only; already-started inference may safely finish. Prior source/results/failures and GPU queue remain unchanged.'])
    if (out/'plan.json').exists(): assert json.loads((out/'plan.json').read_text()) == plan
    else: write('plan.json', plan)
    if args.plan_only:
        print(json.dumps(dict(selected=selected, variants=6, groups={k:len(v['ids']) for k,v in groups.items()}))); return
    assert not (out/'results.json').exists(), 'Do not duplicate this pilot.'
    if datetime.now(timezone(timedelta(hours=9))) >= datetime(2026, 10, 21, 12, tzinfo=timezone(timedelta(hours=9))):
        write('status.json', dict(status='not_started_noon_cutoff')); return
    np.savez_compressed(out/'update_vectors.npz', **updates)
    import torch
    from clip_model import DeepPromptCLIP
    from real_data import CIFAR100
    write('status.json', dict(status='loading_cpu_model', at_kst=datetime.now().isoformat(timespec='seconds')))
    torch.set_num_threads(4); torch.manual_seed(c['seed']); device = torch.device('cpu')
    theta = torch.load(source.parent/'observer/round_060.pt', map_location='cpu', weights_only=True)['theta']
    assert sha(source.parent/'observer/round_060.pt') == pilot['public_checkpoint_sha256']
    model_path = root/'checkpoints/clip-vit-base-patch16'; dataset = CIFAR100(root/'data/cifar-100-python', model_path)
    model = DeepPromptCLIP.from_local_checkpoint(model_path, prompt_layers=c['prompt_layers'], prompt_tokens=c['prompt_tokens'], checkpoint_layers=False).eval()
    text = dataset.text_inputs(device, c['prompt_tokens']); full = json.loads((canonical/'cohorts.json').read_text())['groups']; reference = {}; labels = {}
    assert sha(canonical/'cohorts.json') == pilot['canonical_cohorts_sha256']
    with np.load(canonical/'retrain_reference_ideal.npz') as data:
        for group, spec in groups.items():
            lookup = {z:i for i,z in enumerate(full[group]['ids'])}; idx = [lookup[z] for z in spec['ids']]
            reference[group] = data[group][idx].copy(); labels[group] = data[group+'_labels'][idx].copy()
    with np.load(previous/'ideal_one_step.npz') as data: ideal_prob = {group:data[group].copy() for group in groups}
    result_rows = []; checks = []
    for name, update in updates.items():
        write('status.json', dict(status='evaluating_cpu_variant', variant=name, completed_variants=len(result_rows), total_variants=6, at_kst=datetime.now().isoformat(timespec='seconds')))
        vector = theta+torch.as_tensor(update, dtype=theta.dtype)
        probabilities = {g:eq.predict(model, vector, dataset, text, s['ids'], s['split'], device, 32) for g,s in groups.items()}
        np.savez_compressed(out/(name+'.npz'), **probabilities)
        mode = name.split('_q')[0]; row = dict(method=name, mode=mode, protocol='paid_actual_error_report', quantization_bits=6, noise_seed=int(name.split('noise')[-1]))
        for group in groups:
            p = probabilities[group]; n = len(labels[group]); np.testing.assert_allclose(p.sum(axis=1), 1., rtol=1e-12, atol=1e-14)
            assert np.all(np.isfinite(p)) and np.all(p > 0)
            row[group] = eq.classification(p, labels[group])
            manual_loss = float(-np.log(p[np.arange(n),labels[group]]).mean()); manual_acc = float(np.mean(p.argmax(axis=1)==labels[group]))
            np.testing.assert_allclose([manual_loss,manual_acc],[row[group]['loss'],row[group]['accuracy']],rtol=1e-12,atol=1e-14)
            for label, ref in [('ideal_step',ideal_prob[group]),('final_retrain_descriptive_only',reference[group])]:
                gap = eq.reference_gap(p,ref,labels[group]); row[group]['gap_to_'+label] = gap
                manual_kl = max(0.,float(np.mean(np.sum(ref*(np.log(ref)-np.log(p)),axis=1))))
                manual_disagreement = float(np.mean(ref.argmax(axis=1)!=p.argmax(axis=1)))
                np.testing.assert_allclose([manual_kl,manual_disagreement],[gap['kl_reference_to_candidate'],gap['prediction_disagreement']],rtol=1e-12,atol=1e-14)
            checks.append(dict(method=name,cohort=group,probability_sha256=sha(out/(name+'.npz')),status='independent_cached_metric_arithmetic_passed'))
        result_rows.append(row); write('results.json',result_rows); print('RESULT',json.dumps(row),flush=True)
    old_rows = json.loads((previous/'results.json').read_text()); no_report_rows=json.loads((no_report/'results.json').read_text()); summary=[]
    for mode in ['hybrid','phase_grid']:
        for group in groups:
            variants=[('worst_bound8',8,[r for r in old_rows if r.get('mode')==mode and r.get('rank')==0]),
                      ('worst_bound6',6,[r for r in no_report_rows if r['mode']==mode]),
                      ('paid_actual_error_report6',6,[r for r in result_rows if r['mode']==mode])]
            for protocol,bits,entries in variants:
                assert len(entries)==3
                metrics={k:float(np.mean([r[group][k] for r in entries])) for k in ['loss','accuracy']}
                for ref in ['ideal_step','final_retrain_descriptive_only']:
                    for key in ['kl_reference_to_candidate','prediction_disagreement']:
                        metrics[ref+'_'+key]=float(np.mean([r[group]['gap_to_'+ref][key] for r in entries]))
                summary.append(dict(mode=mode,protocol=protocol,cohort=group,quantization_bits=bits,noise_draws=3,**metrics))
    write('summary.json',dict(selected=selected,metrics=summary,independent_metric_checks=checks,limits=plan['limits']))
    write('status.json',dict(status='completed6_cpu_paid_report_vectors_and18_independent_cohort_checks_no_gpu_promotion',at_kst=datetime.now().isoformat(timespec='seconds')))


if __name__=='__main__': main()
