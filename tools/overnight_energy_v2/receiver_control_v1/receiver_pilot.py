"""Independent actual-request61 CPU pilot; no training-source changes."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, value):
    temporary = p.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(p)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', type=Path, required=True)
    p.add_argument('--stage', choices=['prepare', 'ideal', 'physical'], default='prepare')
    a = p.parse_args(); root = a.project.resolve(); scripts = Path(__file__).parent
    sys.path[:0] = [str(root), str(root/'tools/energy_quality_v1'), str(root/'tools/overnight_energy_v2')]
    import torch
    import torch.nn.functional as functional
    from clip_model import DeepPromptCLIP
    from real_data import CIFAR100
    from energy_quality import predict, classification, reference_gap, stratified_sample, auc, losses
    from phase_split_grid import make_inputs
    from receiver_control import control, sensitivity_bound, test
    torch.set_num_threads(4); torch.manual_seed(0)
    out = root/'results/overnight_energy_v2/receiver_control_v1'; out.mkdir(exist_ok=True)
    source = root/'results/overnight_energy_v2/public_history_fu_v19'
    previous = json.loads((source/'screen/plan.json').read_text())
    assert all(sha(root/n) == v for n, v in previous['core_sha256'].items())
    config = dict(previous['config'], quantization_bits=8, max_ao_iterations=30)
    statefile = source/'observer/round_060.pt'; vectors = source/'screen/vectors.npz'
    cohortfile = root/'results/overnight_energy_v2/fu_quality_v22/plan.json'
    groups = json.loads(cohortfile.read_text())['groups']
    canonicalfile = root/'results/energy_quality_v1/cohorts.json'
    canonical = json.loads(canonicalfile.read_text())['groups']
    model_path = root/'checkpoints/clip-vit-base-patch16'
    dataset = CIFAR100(root/'data/cifar-100-python', model_path)
    evaluation_ids = set(i for spec in groups.values() for i in spec['ids'])
    pool = sorted(set(canonical['validation']['ids'])-evaluation_ids)
    calibration = stratified_sample(pool, dataset.train_labels, 64, np.random.default_rng(641061))
    assert not set(calibration).intersection(evaluation_ids)
    plan = dict(version='receiver_control_v1', actual_round=61, config=config, core_sha256=previous['core_sha256'],
                source_vectors_sha256=sha(vectors), checkpoint_sha256=sha(statefile),
                evaluator_sha256=sha(root/'tools/energy_quality_v1/energy_quality.py'),
                cohorts_source_sha256=sha(cohortfile), canonical_cohorts_sha256=sha(canonicalfile),
                groups=groups, calibration_ids=calibration, primary_alpha=1., sensitivity_alphas=[.25, .5],
                relative_application_error=.2, common_noise_seeds=[61, 1061, 2061], device='cpu', cpu_threads=4,
                code_sha256={n:sha(scripts/n) for n in ['receiver_control.py', 'receiver_pilot.py', 'receiver_radio.py']},
                assumptions=['NEW head-accessible calibration64 from held-out training validation, disjoint from evaluation; computation energy unmeasured.',
                    'NEW canonical 8-bit request anchors for every architecture; local head uses the same codec. Raw protocol retained as control.',
                    'QP constrains directional proxies only, not actual loss or certified deletion. No request dropped silently.',
                    'Actual round61 replay is one seed/one request event, not a trajectory or independent-seed study.',
                    'Final retained-retrain100 reference is STAGE-MISMATCHED and only descriptive.',
                    'Only test-free validation and retained/current-request forget outputs are evaluated.',
                    'Ordinary Mixed without side information implements the original sum, not the new controller; paid-feedback Mixed is included.',
                    'No long-training promotion on directional proxy or small update MSE alone.'])
    target = out/'plan.json'
    if target.exists():
        assert json.loads(target.read_text()) == plan, 'Immutable plan differs; use a new version.'
    else:
        write(target, plan)
    def status(value, **extra):
        write(out/(a.stage+'_status.json'), dict(status=value, at_kst=datetime.now().isoformat(timespec='seconds'), **extra))
    status('loading_model')
    state = torch.load(statefile, map_location='cpu', weights_only=True)
    assert state['round'] == 60
    theta = state['theta']; device = torch.device('cpu')
    with np.load(vectors) as z:
        learning, unlearning, channels = [z[n].copy() for n in ['learning', 'unlearning', 'channels']]
    model = DeepPromptCLIP.from_local_checkpoint(model_path, prompt_layers=config['prompt_layers'],
        prompt_tokens=config['prompt_tokens'], checkpoint_layers=True).eval()
    text = dataset.text_inputs(device, config['prompt_tokens'])
    prepared = out/'prepared.npz'
    if a.stage == 'prepare':
        write(out/'controller_tests.json', test())
        if prepared.exists():
            status('already_prepared'); return
        status('computing_disjoint_calibration_gradient', completed=0, total=len(calibration))
        gradient = torch.zeros_like(theta); began = time.monotonic()
        for start in range(0, len(calibration), 4):
            ids = calibration[start:start+4]
            value = theta.detach().clone().requires_grad_(True)
            pixels, labels = dataset.batch(ids, device, 'train')
            loss = functional.cross_entropy(model(pixels, **text, theta=value), labels)
            gradient += torch.autograd.grad(loss, value)[0].detach()*len(ids)/len(calibration)
            status('computing_disjoint_calibration_gradient', completed=min(start+4, len(calibration)), total=len(calibration))
            print('CALIBRATION', start+len(ids), len(calibration), time.monotonic()-began, flush=True)
        inputs = make_inputs(learning, unlearning, config)
        quantized = inputs[2].reshape(len(learning), -1)[:, :learning.shape[1]].copy()
        active = np.flatnonzero(np.linalg.norm(unlearning, axis=1)>1e-15)
        aggregate = learning.sum(axis=0); anchors = quantized[active]; g = gradient.numpy().astype(float)
        budget = .2*np.linalg.norm(inputs[5].sum(axis=0))
        lip = sensitivity_bound(anchors, g)
        updates = dict(before_step=np.zeros_like(aggregate), learning_only=aggregate,
                       raw_original_sum=(learning+unlearning).sum(axis=0), canonical_sum=aggregate+anchors.sum(axis=0))
        details = dict(active_users=active.tolist(), learning_norm=float(np.linalg.norm(aggregate)),
            calibration_gradient_norm=float(np.linalg.norm(g)), calibration_seconds=time.monotonic()-began,
            cos_learning_requests=(anchors@aggregate/(np.linalg.norm(anchors,axis=1)*np.linalg.norm(aggregate))).tolist(),
            cos_retention_requests=(anchors@g/(np.linalg.norm(anchors,axis=1)*np.linalg.norm(g))).tolist(),
            codec_error_norm=float(np.linalg.norm(quantized-unlearning)), application_rmse_budget=float(budget),
            controller_lipschitz_bound=float(lip), controls={})
        for alpha, name in [(1., 'control_alpha1'), (.25, 'control_alpha025'), (.5, 'control_alpha05')]:
            try:
                updates[name], details['controls'][name] = control(aggregate, anchors, g, alpha)
                details['controls'][name]['status'] = 'feasible'
            except ValueError as e:
                details['controls'][name] = dict(status='no_feasible_projection_found', reason=str(e))
        np.savez_compressed(prepared, learning=learning, unlearning=unlearning, quantized=quantized,
            channels=channels, gradient=g, active=active, **updates)
        write(out/'geometry.json', details)
        status('completed', prepared_sha256=sha(prepared)); print('GEOMETRY', json.dumps(details), flush=True)
        return
    with np.load(prepared) as z:
        updates = {k:z[k].copy() for k in z.files if k.startswith('control_') or k in ['before_step','learning_only','raw_original_sum','canonical_sum']}
    if a.stage == 'physical':
        with np.load(out/'physical_updates.npz') as z:
            updates = {k:z[k].copy() for k in z.files}
    # Reference probabilities map the fixed canonical IDs; NEVER read its test group.
    reference = {}
    refpath = root/'results/energy_quality_v1/retrain_reference_ideal.npz'
    if refpath.exists():
        with np.load(refpath) as z:
            for group, spec in groups.items():
                positions = {v:i for i,v in enumerate(canonical[group]['ids'])}
                reference[group] = z[group][[positions[i] for i in spec['ids']]].copy()
    rowsfile = out/(a.stage+'_quality.json')
    rows = json.loads(rowsfile.read_text()) if rowsfile.exists() else []
    done = {row['name'] for row in rows}
    canonical_predictions = None
    cp = out/'canonical_sum_predictions.npz'
    if cp.exists():
        with np.load(cp) as z: canonical_predictions = {g:z[g].copy() for g in groups}
    for name, update in updates.items():
        if name in done: continue
        status('evaluating', candidate=name, completed=len(done), total=len(updates))
        vector = theta+torch.as_tensor(update, dtype=theta.dtype)
        probs = {}
        for group, spec in groups.items():
            print('PREDICT', name, group, len(spec['ids']), flush=True)
            probs[group] = predict(model, vector, dataset, text, spec['ids'], spec['split'], device, 16)
        np.savez_compressed(out/(name+'_predictions.npz'), **probs)
        if name == 'canonical_sum': canonical_predictions = probs
        row = dict(name=name, status='evaluated', update_norm=float(np.linalg.norm(update)))
        for group, spec in groups.items():
            y = dataset.train_labels[spec['ids']]
            row[group] = classification(probs[group], y)
            if reference:
                row[group]['stage_mismatched_retrain100_gap'] = reference_gap(probs[group], reference[group], y)
            if canonical_predictions is not None:
                row[group]['gap_to_canonical_sum'] = reference_gap(probs[group], canonical_predictions[group], y)
        row['small_cohort_loss_attack_auc'] = auc(-losses(probs['forget'], dataset.train_labels[groups['forget']['ids']]),
            -losses(probs['validation'], dataset.train_labels[groups['validation']['ids']]))
        if reference:
            refauc = auc(-losses(reference['forget'],dataset.train_labels[groups['forget']['ids']]),
                -losses(reference['validation'],dataset.train_labels[groups['validation']['ids']]))
            row['small_cohort_auc_absolute_gap_to_stage_mismatched_reference'] = abs(row['small_cohort_loss_attack_auc']-refauc)
        rows.append(row); write(rowsfile, rows); print('RESULT',json.dumps(row),flush=True)
    status('completed', candidates=len(rows)); print('COMPLETED',a.stage,flush=True)


if __name__ == '__main__':
    main()
