"""Evaluate versioned trajectories on the existing frozen final-time cohorts."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np


def validate_trial(root, output, mode, plan, eq):
    directory = output/mode; path = directory/'manifest.json'
    if not path.exists():
        return None, 'not_started'
    manifest = json.loads(path.read_text()); rows = eq.read_rounds(directory)
    if manifest['status'] != 'completed':
        return None, manifest['status']
    if (manifest.get('variant') != 'burst_joint_v3' or manifest['algorithm'] != 'coupled' or manifest['device'] != 'cuda'
            or manifest['transport'] != mode or manifest['config'] != plan['config']
            or manifest['source_sha256'] != plan['source_sha256']):
        return None, 'incompatible_variant_provenance'
    sidecar = json.loads((directory/'variant_provenance.json').read_text())
    if (sidecar['source_sha256'] != plan['source_sha256'] or sidecar['mode'] != mode
            or sidecar['config'] != plan['config']):
        return None, 'incompatible_provenance_sidecar'
    if [r['round'] for r in rows] != list(range(1, plan['config']['rounds']+1)):
        return None, 'invalid_round_sequence'
    source = root/'results/comparisons_seed0_solver_v2/coupled_hybrid'
    if eq.digest(directory/'schedule.json') != eq.digest(source/'schedule.json'):
        return None, 'incompatible_schedule'
    schedule = json.loads((directory/'schedule.json').read_text())
    for item, row in zip(schedule, rows):
        requested = len(set().union(*map(set, item['forget'])))
        if row['newly_deleted'] != requested or row['transmission']['status'] != 'feasible':
            return None, 'incomplete_request_delivery'
    if not np.isclose(sum(r['transmission']['energy_bound_j'] for r in rows), rows[-1]['cumulative_energy_bound_j']):
        return None, 'invalid_energy_total'
    return (manifest, rows), 'completed'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--summary-only', action='store_true')
    args = parser.parse_args(); root = args.project.resolve(); output = args.output.resolve()
    sys.path[:0] = [str(root), str(root/'tools/energy_quality_v1')]
    import energy_quality as eq
    plan = json.loads((output/'plan.json').read_text())
    if eq.digest(__file__) != plan['audit_sha256'] or eq.digest(eq.__file__) != plan['numeric_evaluator_sha256']:
        raise ValueError('Frozen evaluator code changed.')
    for name, expected in plan['source_sha256'].items():
        if eq.digest(root/name) != expected:
            raise ValueError('Experimental source changed: '+name)
    cohort_source = root/'results/energy_quality_v1/cohorts.json'
    cohorts = json.loads(cohort_source.read_text())
    if eq.digest(cohort_source) != plan['cohorts_sha256']:
        raise ValueError('Frozen cohort file changed.')
    evaluation = output/'quality'; evaluation.mkdir(exist_ok=True)
    target_cohorts = evaluation/'cohorts.json'
    if target_cohorts.exists() and target_cohorts.read_bytes() != cohort_source.read_bytes():
        raise ValueError('Existing variant cohorts differ.')
    target_cohorts.write_bytes(cohort_source.read_bytes())
    baseline_suite = root/'results/comparisons_seed0_solver_v2'
    protocol = json.loads((baseline_suite/'config.json').read_text())
    baseline_source = baseline_suite/'coupled_hybrid'
    reference = None
    reference_directory = baseline_suite/'retrain_reference_ideal'
    reference_cache = root/'results/energy_quality_v1/retrain_reference_ideal.cache.json'
    reference_prediction = reference_cache.with_name('retrain_reference_ideal.npz')
    valid_reference, _ = eq.validate_completed(reference_directory, baseline_suite, protocol, baseline_source)
    if valid_reference is not None and reference_cache.exists() and reference_prediction.exists():
        cached = json.loads(reference_cache.read_text())
        required = {'checkpoint_sha256': eq.digest(reference_directory/'latest.pt'),
            'cohorts_sha256': plan['cohorts_sha256'], 'evaluator_sha256': eq.digest(eq.__file__),
            'training_source_sha256': valid_reference[0]['source_sha256']}
        if all(cached.get(k) == v for k,v in required.items()):
            with np.load(reference_prediction) as data:
                reference = {k:data[k].copy() for k in data.files}
    model = dataset = device = text = None
    results = []
    for mode in plan['modes']:
        directory = output/mode
        validated, status = validate_trial(root, output, mode, plan, eq)
        rows = eq.read_rounds(directory)
        row = {'method': 'burst_joint_v3/'+mode, 'status': status, 'energy_role': 'physical_model',
            'quality_status': 'pending', 'round': rows[-1]['round'] if rows else 0}
        if rows:
            row.update(energy_bound_j=rows[-1]['cumulative_energy_bound_j'],
                expected_energy_j=rows[-1]['cumulative_expected_energy_j'],
                cumulative_uplink_s=sum(r['transmission']['uplink_s'] for r in rows),
                deletion_rounds_delivered=sum(bool(r['newly_deleted']) for r in rows))
        if validated is not None:
            cache_path = evaluation/(mode+'.cache.json'); prediction_path = evaluation/(mode+'.npz')
            provenance = {'checkpoint_sha256': eq.digest(directory/'latest.pt'),
                'initial_checkpoint_sha256': eq.digest(directory/'initial.pt'),
                'cohorts_sha256': plan['cohorts_sha256'], 'numeric_evaluator_sha256': plan['numeric_evaluator_sha256'],
                'audit_sha256': plan['audit_sha256'], 'source_sha256': plan['source_sha256'],
                'plan_sha256': eq.digest(output/'plan.json')}
            predictions = None
            if prediction_path.exists() and cache_path.exists() and json.loads(cache_path.read_text()) == provenance:
                with np.load(prediction_path) as data:
                    predictions = {k:data[k].copy() for k in data.files}
            elif not args.summary_only:
                import torch
                from clip_model import DeepPromptCLIP
                from real_data import CIFAR100
                from run_clip_federated import require_device
                checkpoint = torch.load(directory/'latest.pt', map_location='cpu', weights_only=True)
                initial = torch.load(directory/'initial.pt', map_location='cpu', weights_only=True)
                original_initial = torch.load(baseline_source/'initial.pt', map_location='cpu', weights_only=True)
                if checkpoint['round'] != plan['config']['rounds'] or not torch.equal(initial['theta'], original_initial['theta']):
                    raise ValueError('Final checkpoint or initial model differs: '+mode)
                if model is None:
                    device = require_device('cuda'); torch.set_num_threads(4)
                    checkpoint_root = root/'checkpoints/clip-vit-base-patch16'
                    dataset = CIFAR100(root/'data/cifar-100-python', checkpoint_root)
                    expected_cohorts = eq.make_cohorts(dataset, baseline_source, protocol, 1000)
                    if expected_cohorts != cohorts:
                        raise ValueError('Cohort reconstruction differs.')
                    model = DeepPromptCLIP.from_local_checkpoint(checkpoint_root,
                        prompt_layers=protocol['prompt_layers'], prompt_tokens=protocol['prompt_tokens'],
                        checkpoint_layers=False).to(device).eval()
                    text = dataset.text_inputs(device, protocol['prompt_tokens'])
                theta = checkpoint['theta'].to(device); predictions = {}
                for group, spec in cohorts['groups'].items():
                    print('EVALUATE', mode, group, len(spec['ids']), flush=True)
                    predictions[group] = eq.predict(model, theta, dataset, text, spec['ids'], spec['split'], device, 32)
                    labels = dataset.train_labels if spec['split'] == 'train' else dataset.test_labels
                    predictions[group+'_labels'] = labels[spec['ids']]
                temp = prediction_path.with_suffix('.partial.npz'); np.savez_compressed(temp, **predictions); temp.replace(prediction_path)
                eq.write_json(cache_path, provenance)
            if predictions is not None:
                row['quality_status'] = 'evaluated'
                for group in ['validation', 'retained', 'forget', 'test']:
                    row.update({group+'_'+k:v for k,v in eq.classification(predictions[group], predictions[group+'_labels']).items()})
                    if reference is not None:
                        if not np.array_equal(predictions[group+'_labels'], reference[group+'_labels']):
                            raise ValueError('Reference cohort labels differ.')
                        row.update({group+'_'+k:v for k,v in eq.reference_gap(predictions[group], reference[group], predictions[group+'_labels']).items()})
                for nonmember, tag in [('nonmember_validation', 'validation'), ('nonmember_test', 'test')]:
                    attack = eq.auc(-eq.losses(predictions['forget'], predictions['forget_labels']),
                                    -eq.losses(predictions[nonmember], predictions[nonmember+'_labels']))
                    row['forget_loss_attack_auc_'+tag] = attack
                    if reference is not None:
                        ref_attack = eq.auc(-eq.losses(reference['forget'], reference['forget_labels']),
                            -eq.losses(reference[nonmember], reference[nonmember+'_labels']))
                        row['forget_attack_auc_gap_to_reference_'+tag] = abs(attack-ref_attack)
        results.append(row)
    eligible = [r for r in results if r['status'] == 'completed' and r['quality_status'] == 'evaluated']
    keys = ['energy_bound_j', 'validation_loss', 'forget_kl_reference_to_candidate']
    report = {'reference_available': reference is not None,
        'observed_frontier': [eligible[i]['method'] for i in eq.nondominated(eligible, keys)],
        'frontier_objectives': keys, 'selection_status': 'no_task_threshold_winner_claimed', 'methods': results,
        'limits': ['One seed; final empirical retained-data retraining reference, not certified forgetting.',
            'Request-round RMS budget 0.6 is a changed protocol; original 0.2 results are preserved separately.',
            'Completed physical runs only; no ideal zero-energy reference or failed run enters the frontier.',
            'All three modes receive identical time, power and error envelopes; phase split uses a grid and retains digital NOMA.',
            'Only modeled communication energy; local computation and UAV flight excluded.',
            'Test metrics are reporting only; fixed experiment order/config never depends on them.']}
    eq.write_json(evaluation/'report.json', report)
    print(json.dumps({'reference_available': reference is not None, 'methods': [{k:r[k] for k in ['method','status','quality_status']} for r in results]}), flush=True)


if __name__ == '__main__':
    main()
