"""Queued CPU task sensitivity at real request61, after ALL fair radio cases."""
import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from recovery import moments, perturbation


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    for attempt in range(10):
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            if attempt == 9:
                raise
            time.sleep(.02)


def identity(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()
        return None if fields[0] == 'Z' else fields[19]
    except FileNotFoundError:
        return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', type=Path, required=True)
    p.add_argument('--wait-pid', type=int, required=True)
    p.add_argument('--plan-only', action='store_true')
    args = p.parse_args(); root = args.project.resolve()
    sys.path[:0] = [str(root), str(root/'tools/energy_quality_v1')]
    import energy_quality as eq
    out = root/'results/overnight_energy_v2/fu_quality_v22'; out.mkdir(exist_ok=True)
    family = root/'results/overnight_energy_v2/public_history_fu_v19'; boundary = root/'results/overnight_energy_v2/public_history_fu_v19b'
    source = family/'screen'; registered = read(source/'plan.json'); c = registered['config']
    core = registered['core_sha256']; assert all(sha(root/n) == v for n, v in core.items())
    canonical = root/'results/energy_quality_v1'; cohorts = read(canonical/'cohorts.json')['groups']
    old_pilot_cohorts = read(root/'results/overnight_energy_v2/single_step_quality_pilot/cohorts.json')
    record = read(root/'results/overnight_energy_v2/burst_joint_v3/hybrid/schedule.json')[60]
    requested = set().union(*map(set, record['forget']))
    current_ids = [z for z in cohorts['forget']['ids'] if z in requested]
    # Deterministic cohort construction before any model outputs; no new
    # reference inference: IDs remain inside the canonical reference cache.
    rng = np.random.default_rng(2026102161)
    if len(current_ids) > 200:
        current_ids = sorted(rng.choice(current_ids, 200, replace=False).tolist())
    assert current_ids
    groups = dict(validation=old_pilot_cohorts['validation'],
                  retained=dict(split='train', ids=cohorts['retained']['ids'][:200]),
                  forget=dict(split='train', ids=current_ids))
    refmeta = read(canonical/'retrain_reference_ideal.cache.json')
    assert refmeta['cohorts_sha256'] == sha(canonical/'cohorts.json') and refmeta['evaluator_sha256'] == sha(eq.__file__)
    checkpoint = family/'observer/round_060.pt'
    plan = dict(version='fu_quality_v22_actual_request61', actual_round=61, relative_tolerance=.2,
        ranks=[0, 3], modes=['hybrid', 'mixed_aircomp', 'phase_grid'], noise_seeds=[61, 1061, 2061],
        wait_pid=args.wait_pid, cpu_threads=4, core_sha256=core, code_sha256={n: sha(Path(__file__).with_name(n)) for n in ['pilot.py', 'recovery.py', 'test_recovery.py']},
        input_vector_sha256=sha(source/'vectors.npz'), public_checkpoint_sha256=sha(checkpoint),
        canonical_cohorts_sha256=sha(canonical/'cohorts.json'), groups=groups, evaluator_sha256=sha(eq.__file__),
        reference_metadata_sha256=sha(canonical/'retrain_reference_ideal.cache.json'),
        certificate_source_sha256=sha(root/'tools/overnight_energy_v2/transport_certificate_v20.py'),
        no_new_start_after_kst='2026-10-21T12:00:00+09:00',
        limits=['Prespecified original0.2 envelope, all three architectures and both original/rank3, three common noise draws. Choose each phase incumbent from ALL54v19 and4v19b cases by ENERGY ONLY before inference.',
                'Current-request forget cohort is the fixed canonical final-forget cohort intersected with actualDF61, capped200 with predetermined seed. Utility200 and retained200 fixed; no test arrays or output-based settings.',
                'Nineteen one-step vectors (ideal plus18physical), not nineteen training runs. Three noise draws are not independent training seeds or statistical equivalence.',
                'Final retained-retrain100 reference differs in training stage from this round61 pilot; its KL/prediction gaps are descriptive, not certified forgetting. Ideal-step KL measures transport fidelity only.',
                'No MIA estimate from the small pilot. Full completed100-round MIA remains in canonical/versioned reports. Delivery/smallMSE/KL is not deletion proof.',
                'All coefficient RF/time accounted; extra basis/projection/decoding/coding/headers energy unmeasured. CPU replay numerical/projected-random realization differs from CUDA; all candidates share identical frozen vectors.',
                'No GPU work, source/queue edits or automatic long-training promotion. Begin before noon only, safely finish an already-started pilot afterwards if needed.'])
    target = out/'plan.json'
    if target.exists():
        assert read(target) == plan
    else:
        eq.write_json(target, plan)
    eq.write_json(out/'cohorts.json', groups)
    if args.plan_only:
        print(json.dumps(plan)); return
    def status(value, **extra):
        eq.write_json(out/'status.json', dict(status=value, at_kst=datetime.now().isoformat(timespec='seconds'), **extra))
    status('waiting_for_complete_radio_and_phase_boundary', wait_pid=args.wait_pid)
    initial = identity(args.wait_pid); deadline = datetime(2026, 10, 21, 12, tzinfo=timezone(timedelta(hours=9)))
    while initial is not None and identity(args.wait_pid) == initial:
        if datetime.now(timezone(timedelta(hours=9))) >= deadline:
            status('not_started_noon_window_ended'); return
        time.sleep(15)
    if datetime.now(timezone(timedelta(hours=9))) >= deadline:
        status('not_started_noon_window_ended'); return
    assert read(family/'radio/status.json')['status'].startswith('completed') and read(boundary/'status.json')['status'].startswith('completed')
    status('independently_checking_all_radio_candidates')
    certificate = subprocess.run([sys.executable, str(root/'tools/overnight_energy_v2/transport_certificate_v20.py'),
        '--project', str(root), '--label', 'fu_quality_v22'], cwd=root, capture_output=True, text=True)
    (out/'radio_certificate.log').write_text(certificate.stdout+certificate.stderr)
    assert certificate.returncode == 0, 'Physical certificate failed; no VLM inference.'
    all_rows = [(row, family/'radio') for row in read(family/'radio/results.json')]+[(row, boundary) for row in read(boundary/'results.json')]
    assert len(all_rows) == 58
    with np.load(source/'vectors.npz') as data:
        learning, unlearning = [data[k].copy() for k in ['learning', 'unlearning']]
        q = data['rank3_basis'].copy(); public_scale = float(data['public_scale'])
    dimension = learning.shape[1]; width = c['block_dimension']
    block = lambda v: np.pad(v, ((0, 0), (0, (-dimension) % width))).reshape(len(v), -1, width)
    ideal = (learning+unlearning).sum(axis=0); updates = dict(ideal_one_step=ideal); roles = {}; selected = []
    for mode in plan['modes']:
        for rank in plan['ranks']:
            eligible = [(row, directory) for row, directory in all_rows if row['relative_tolerance'] == .2 and row['mode'] == mode and row['rank'] == rank and row['status'] == 'feasible_radio_only']
            assert eligible, 'No feasible complete radio design for prespecified quality family.'
            row, directory = min(eligible, key=lambda pair: pair[0]['energy_bound_j'])
            path = directory/f"r0.2_rank{rank}_{mode}_f{row['analog_fraction']}_design.npz"
            with np.load(path) as data:
                design = {k: data[k].copy() for k in data.files}
            raw = learning+unlearning if mode == 'mixed_aircomp' else learning
            analog = raw
            if rank:
                coefficient = ((raw@q)/public_scale).astype(np.float16).astype(float)*public_scale
                analog = raw-coefficient@q.T
            digital = np.zeros_like(unlearning) if mode == 'mixed_aircomp' else unlearning
            analog, digital = block(analog), block(digital)
            levels = 2**(c['quantization_bits']-1)-1
            scale = (np.max(np.abs(digital), axis=-1)/levels).astype(np.float32).astype(float)
            decoded = np.clip(np.rint(digital/np.where(scale > 0, scale, 1.)[..., None]), -levels, levels)*scale[..., None]
            assert np.isclose(moments(analog, digital, decoded, design, dimension)[2], row['analytic_total_rmse'], rtol=1e-8)
            selected.append(dict(mode=mode, rank=rank, design_path=str(path.relative_to(root)), design_sha256=sha(path), row=row))
            for seed in plan['noise_seeds']:
                error = perturbation(analog, digital, decoded, design, dimension, seed)
                name = f'{mode}_rank{rank}_noise{seed}'
                updates[name] = ideal+error; roles[name] = dict(role='physical_request61_one_step_transport', mode=mode, rank=rank,
                    noise_seed=seed, communication_energy_upper_j=row['energy_bound_j'], analytic_rmse=row['analytic_total_rmse'])
    assert len(updates) == 19
    eq.write_json(out/'selected_plan.json', dict(selected=selected, update_sha256={n: hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for n, v in updates.items()},
        selected_by='Energy only from complete fair58case radio pool before any inference', registered_plan_sha256=sha(target)))
    np.savez_compressed(out/'update_vectors.npz', **updates)
    import torch
    from clip_model import DeepPromptCLIP
    from real_data import CIFAR100
    status('loading_cpu_model'); torch.set_num_threads(4); torch.manual_seed(c['seed']); device = torch.device('cpu')
    theta = torch.load(checkpoint, map_location='cpu', weights_only=True)['theta']
    model_path = root/'checkpoints/clip-vit-base-patch16'; dataset = CIFAR100(root/'data/cifar-100-python', model_path)
    model = DeepPromptCLIP.from_local_checkpoint(model_path, prompt_layers=c['prompt_layers'], prompt_tokens=c['prompt_tokens'], checkpoint_layers=False).eval()
    text = dataset.text_inputs(device, c['prompt_tokens']); reference = {}
    with np.load(canonical/'retrain_reference_ideal.npz') as data:
        for group, spec in groups.items():
            lookup = {z: k for k, z in enumerate(cohorts[group]['ids'])}
            reference[group] = data[group][[lookup[z] for z in spec['ids']]].copy()
    rows = []; ideal_prob = None
    for name, update in updates.items():
        status('evaluating_cpu_variant', variant=name, completed_variants=len(rows), total_variants=19)
        vector = theta+torch.as_tensor(update, dtype=theta.dtype)
        probabilities = {group: eq.predict(model, vector, dataset, text, spec['ids'], spec['split'], device, 32) for group, spec in groups.items()}
        if name == 'ideal_one_step':
            ideal_prob = probabilities
        np.savez_compressed(out/(name+'.npz'), **probabilities)
        row = dict(method=name, **roles.get(name, dict(role='ideal_one_step_reference_only')))
        for group, spec in groups.items():
            labels = dataset.train_labels[spec['ids']]
            row[group] = eq.classification(probabilities[group], labels)
            row[group]['gap_to_ideal_step'] = eq.reference_gap(probabilities[group], ideal_prob[group], labels)
            row[group]['gap_to_final_retrain_descriptive_only'] = eq.reference_gap(probabilities[group], reference[group], labels)
        rows.append(row); eq.write_json(out/'results.json', rows)
        print('RESULT', json.dumps(row), flush=True)
    summary = []
    for mode in plan['modes']:
        for rank in plan['ranks']:
            entries = [row for row in rows if row.get('mode') == mode and row.get('rank') == rank]
            item = dict(mode=mode, rank=rank, noise_draws=len(entries))
            for group in groups:
                for metric in ['loss', 'accuracy']:
                    values = [row[group][metric] for row in entries]
                    item[group+'_'+metric+'_mean'] = float(np.mean(values)); item[group+'_'+metric+'_noise_std'] = float(np.std(values))
                for reference_name in ['ideal_step', 'final_retrain_descriptive_only']:
                    for metric in ['kl_reference_to_candidate', 'prediction_disagreement']:
                        values = [row[group]['gap_to_'+reference_name][metric] for row in entries]
                        item[group+'_'+reference_name+'_'+metric+'_mean'] = float(np.mean(values))
            summary.append(item)
    eq.write_json(out/'summary.json', summary); status('completed_cpu_request61_quality_no_gpu_promotion', variants=len(rows))


if __name__ == '__main__':
    main()
