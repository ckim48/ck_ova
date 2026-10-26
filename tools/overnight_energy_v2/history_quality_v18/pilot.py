"""CPU VLM output sensitivity of the preregistered actual round30 transports."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from recovery import moments, perturbation


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', type=Path, required=True)
    p.add_argument('--plan-only', action='store_true')
    args = p.parse_args(); root = args.project.resolve()
    sys.path[:0] = [str(root), str(root/'tools/energy_quality_v1')]
    import energy_quality as eq
    family = root/'results/overnight_energy_v2/public_history_early_v17'
    radio = family/'radio'; screen = family/'screen'
    assert json.loads((radio/'status.json').read_text())['status'].startswith('completed')
    audit = json.loads((radio/'physical_certificates.json').read_text())
    assert audit['checked_saved_radio_designs'] == 30
    radio_plan = json.loads((radio/'manifest.json').read_text())
    for n, expected in radio_plan['code_sha256'].items():
        assert sha(root/'tools/overnight_energy_v2/public_history_early_v17'/n) == expected
    for n, expected in radio_plan['core_sha256'].items():
        assert sha(root/n) == expected
    checkpoint = family/'observer/round_029.pt'
    assert sha(checkpoint) == radio_plan['public_states']['29']
    with np.load(screen/'vectors.npz') as data:
        learning = data['learning'].copy()
    c = radio_plan['config']; dimension = learning.shape[1]; width = c['block_dimension']
    block = lambda v: np.pad(v, ((0, 0), (0, (-dimension) % width))).reshape(len(v), -1, width)
    ideal = learning.sum(axis=0); updates = {'ideal_one_step': ideal}; roles = {}
    radio_rows = json.loads((radio/'results.json').read_text())
    noise_seeds = [30, 1030, 2030]
    selected = {}
    for name in ['unprojected_baseline', 'recent_public_span_rank3']:
        row = next(x for x in radio_rows if x['variant'] == name)['best']
        assert row is not None
        path = radio/f"{name}_h{row['head']}.npz"
        with np.load(path) as data:
            design = {k: data[k].copy() for k in data.files}
        if name == 'unprojected_baseline':
            vectors = learning
        else:
            with np.load(radio/(name+'_codec.npz')) as data:
                vectors = data['residual'].copy()
        blocked = block(vectors)
        assert np.isclose(moments(blocked, design, dimension)[2], row['analytic_rmse'], rtol=1e-8)
        selected[name] = dict(row=row, design_sha256=sha(path))
        for seed in noise_seeds:
            error = perturbation(blocked, design, dimension, seed)
            method = name+'_'+str(seed)
            updates[method] = ideal+error
            roles[method] = dict(role='physical_learning_only_one_step_transport', noise_seed=seed,
                                 radio_energy_j=row['total_modeled_energy_j'], realized_update_error=float(np.linalg.norm(error)),
                                 analytic_rmse=row['analytic_rmse'])
    cohorts_path = root/'results/overnight_energy_v2/single_step_quality_pilot/cohorts.json'
    groups = json.loads(cohorts_path.read_text())
    quality = root/'results/energy_quality_v1'
    refmeta = json.loads((quality/'retrain_reference_ideal.cache.json').read_text())
    assert refmeta['checkpoint_sha256'] == sha(root/'results/comparisons_seed0_solver_v2/retrain_reference_ideal/latest.pt')
    assert refmeta['cohorts_sha256'] == sha(quality/'cohorts.json') and refmeta['evaluator_sha256'] == sha(eq.__file__)
    out = root/'results/overnight_energy_v2/history_quality_v18'; out.mkdir(exist_ok=True)
    plan = dict(version='history_quality_v18_actual_round30', actual_round=30,
                core_sha256=radio_plan['core_sha256'], evaluator_sha256=sha(eq.__file__),
                code_sha256={n: sha(Path(__file__).with_name(n)) for n in ['pilot.py', 'recovery.py', 'test_recovery.py']},
                public_checkpoint_sha256=sha(checkpoint), early_radio_manifest_sha256=sha(radio/'manifest.json'),
                early_vector_sha256=sha(screen/'vectors.npz'), selected_designs=selected,
                frozen_cohorts_sha256=sha(cohorts_path), reference_metadata_sha256=sha(quality/'retrain_reference_ideal.cache.json'),
                noise_seeds=noise_seeds, cpu_threads=4,
                variants={n: dict(update_sha256=hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest(), **roles.get(n, dict(role='ideal_one_step_reference_only'))) for n, v in updates.items()},
                limits=['One preselected actual round30, seven vectors, frozen200 utility-validation and200 future-forget probes. Neither codec/rank/noise nor round is chosen from these task outputs.',
                        'Round30 precedes requests: the future-forget probe is output sensitivity, NOT evidence of unlearning or request fulfillment. Final-retraining gaps are descriptive across different training stages.',
                        'Common three Gaussian noise draws are not training seeds or a confidence interval; no full-trajectory or statistical equivalence claim.',
                        'All transports physically checked at the same time/power/battery/absolute RMS. Generic FL codec, all original architectures coincide whenU=0. No NOMA superiority.',
                        'Extra computation/coding/headers energy unmeasured, RFsaving alone not whole-system benefit. No MIA extrapolation from this small pilot.',
                        'CPU inference only, no GPU/new test arrays/configuration selection. Existing full-run queue/source hashes unchanged. No automatic GPU promotion.'])
    target = out/'plan.json'
    if target.exists():
        assert json.loads(target.read_text()) == plan
    else:
        eq.write_json(target, plan)
    if args.plan_only:
        print(json.dumps(plan)); return
    assert not (out/'results.json').exists(), 'Do not duplicate completed/active CPU inference.'
    import torch
    from clip_model import DeepPromptCLIP
    from real_data import CIFAR100
    def status(value, **extra):
        eq.write_json(out/'status.json', dict(status=value, at_kst=datetime.now().isoformat(timespec='seconds'), **extra))
    status('loading_cpu_model'); torch.set_num_threads(4); torch.manual_seed(c['seed']); device = torch.device('cpu')
    theta = torch.load(checkpoint, map_location='cpu', weights_only=True)['theta']
    np.savez_compressed(out/'update_vectors.npz', **updates)
    model_path = root/'checkpoints/clip-vit-base-patch16'; dataset = CIFAR100(root/'data/cifar-100-python', model_path)
    model = DeepPromptCLIP.from_local_checkpoint(model_path, prompt_layers=c['prompt_layers'], prompt_tokens=c['prompt_tokens'], checkpoint_layers=False).eval()
    text = dataset.text_inputs(device, c['prompt_tokens'])
    full_groups = json.loads((quality/'cohorts.json').read_text())['groups']
    reference = {}
    with np.load(quality/'retrain_reference_ideal.npz') as data:
        for name, spec in groups.items():
            lookup = {value: k for k, value in enumerate(full_groups[name]['ids'])}
            reference[name] = data[name][[lookup[z] for z in spec['ids']]].copy()
    rows = []; ideal_predictions = None
    for name, update in updates.items():
        status('evaluating_cpu_variant', variant=name, completed_variants=len(rows))
        vector = theta+torch.as_tensor(update, dtype=theta.dtype)
        predictions = {group: eq.predict(model, vector, dataset, text, spec['ids'], spec['split'], device, 32) for group, spec in groups.items()}
        if name == 'ideal_one_step':
            ideal_predictions = predictions
        np.savez_compressed(out/(name+'.npz'), **predictions)
        row = dict(method=name, **roles.get(name, dict(role='ideal_one_step_reference_only')))
        for group, spec in groups.items():
            labels = dataset.train_labels[spec['ids']]
            row[group] = eq.classification(predictions[group], labels)
            row[group]['gap_to_ideal_step'] = eq.reference_gap(predictions[group], ideal_predictions[group], labels)
            row[group]['gap_to_final_retrain_descriptive_only'] = eq.reference_gap(predictions[group], reference[group], labels)
        rows.append(row); eq.write_json(out/'results.json', rows)
        print('RESULT', json.dumps(row), flush=True)
    summary = []
    for family_name in ['unprojected_baseline', 'recent_public_span_rank3']:
        group_rows = [x for x in rows if x['method'].startswith(family_name+'_')]
        record = dict(family=family_name, noise_draws=len(group_rows))
        for cohort in groups:
            for metric in ['loss', 'accuracy']:
                values = [x[cohort][metric] for x in group_rows]
                record[cohort+'_'+metric+'_mean'] = float(np.mean(values))
                record[cohort+'_'+metric+'_noise_std'] = float(np.std(values))
            values = [x[cohort]['gap_to_ideal_step']['kl_reference_to_candidate'] for x in group_rows]
            record[cohort+'_kl_to_ideal_step_mean'] = float(np.mean(values))
            record[cohort+'_kl_to_ideal_step_noise_std'] = float(np.std(values))
        summary.append(record)
    eq.write_json(out/'summary.json', summary)
    status('completed_cpu_single_step_quality_no_gpu_promotion', variants=len(rows))


if __name__ == '__main__':
    main()
