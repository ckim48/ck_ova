"""CPU replay of the preselected actual request61 from public theta60."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', type=Path, required=True)
    p.add_argument('--plan-only', action='store_true')
    args = p.parse_args(); root = args.project.resolve()
    out = root/'results/overnight_energy_v2/public_history_fu_v19/screen'; out.mkdir(exist_ok=True)
    family = out.parent; observer = family/'observer'; scripts = Path(__file__).parent
    source = root/'results/overnight_energy_v2/burst_joint_v3/hybrid'
    observer_plan = json.loads((observer/'plan.json').read_text())
    assert observer_plan['target_local_round'] == 61 and observer_plan['basis_history_rounds'] == [57, 58, 59, 60]
    assert json.loads((observer/'status.json').read_text())['last_round'] == 60
    assert json.loads((root/'results/overnight_energy_v2/history_quality_v18b/status.json').read_text())['status'].startswith('completed')
    core = observer_plan['core_sha256']; assert all(sha(root/n) == v for n, v in core.items())
    config = json.loads((root/'results/overnight_energy_v2/burst_joint_v3/config.json').read_text())
    parent = json.loads((root/'results/overnight_energy_v2/public_history_fp16_v15/manifest.json').read_text())
    assert sha(scripts/'codec.py') == parent['code_sha256']['codec.py']
    assert sha(root/'tools/overnight_energy_v2/initialization_v11/wireless_init_v11.py') == parent['solver_sha256']
    plan = dict(version='public_history_fu_v19', actual_round=61, public_rounds=[57, 58, 59, 60], config=config,
                core_sha256=core, schedule_sha256=sha(source/'schedule.json'), observer_plan_sha256=sha(observer/'plan.json'),
                code_sha256={n: sha(scripts/n) for n in ['observe.py', 'replay.py', 'radio.py', 'codec.py', 'test_paid_frame.py']},
                parent_solver_sha256=parent['solver_sha256'], cpu_threads=4,
                variants=['unprojected', 'rank1', 'rank3'], relative_tolerances=[.2, .6],
                architectures=['hybrid', 'mixed_aircomp', 'phase_grid'], phase_fractions=[.125, .25, .375, .5, .625, .75, .875],
                quantization_bits=8, ao_iterations=30,
                limits=['Actual request61 minibatches and coupled update functions at observed public theta60; CPU numerical/projected-random-basis realization can differ from CUDA. All transport comparisons use exactly the same frozen replay vectors.',
                        'Only past public57..60 states form the basis/scale. No private gradient history, future states, validation/test choice, or new GPU work.',
                        'Hybrid/phase compress L, Mixed compresses L+U; ALL get identical public basis, IEEEhalf codec, paid metadata RF/time, same original absolute error/power/battery and strong solver.',
                        'Original uncompressed/rank1 alternatives retained. No automatic GPU promotion or claim of deletion/full-trajectory energy benefit.'])
    target = out/'plan.json'
    if target.exists():
        assert json.loads(target.read_text()) == plan
    else:
        target.write_text(json.dumps(plan, indent=2)+'\n')
    if args.plan_only:
        print(json.dumps(plan)); return
    assert not (out/'vectors.npz').exists(), 'Do not duplicate an active/completed replay.'
    import torch
    import torch.nn.functional as functional
    torch.set_num_threads(4); torch.manual_seed(config['seed']); sys.path.insert(0, str(root))
    from clip_model import DeepPromptCLIP, projected_coupling
    from real_data import CIFAR100
    from run_clip_federated import local_gradient
    def status(value, **extra):
        target = out/'status.json'; tmp = target.with_suffix('.tmp')
        tmp.write_text(json.dumps(dict(status=value, at_kst=datetime.now().isoformat(timespec='seconds'), **extra), indent=2)+'\n'); tmp.replace(target)
    status('loading_observed_public_states')
    states = json.loads((observer/'states.json').read_text()); public = []
    for k in [57, 58, 59, 60]:
        path = observer/f'round_{k:03}.pt'; assert sha(path) == next(x['file_sha256'] for x in states if x['round'] == k)
        state = torch.load(path, map_location='cpu', weights_only=True); assert state['round'] == k
        public.append(state['theta'].numpy().astype(float))
    increments = np.stack([b-a for a, b in zip(public, public[1:])], axis=1)
    norms = np.linalg.norm(increments, axis=0); assert np.all(norms > 0)
    normalized = increments/norms
    left, singular, _ = np.linalg.svd(normalized, full_matrices=False)
    assert np.sum(singular > singular[0]*1e-10) == 3
    theta = torch.as_tensor(public[-1], dtype=torch.float32); device = torch.device('cpu')
    model_path = root/'checkpoints/clip-vit-base-patch16'; dataset = CIFAR100(root/'data/cifar-100-python', model_path)
    model = DeepPromptCLIP.from_local_checkpoint(model_path, prompt_layers=config['prompt_layers'], prompt_tokens=config['prompt_tokens'], checkpoint_layers=True).eval()
    text = dataset.text_inputs(device, config['prompt_tokens'])
    record = json.loads((source/'schedule.json').read_text())[60]; assert record['round'] == 61
    rate = config['learning_rate']*.5*(1+np.cos(np.pi*60/config['rounds']))
    retained_count = sum(map(len, record['retained'])); forget_count = sum(map(len, record['forget']))
    assert forget_count > 0
    learning = np.zeros((config['users'], len(theta))); unlearning = np.zeros_like(learning); coupling = []
    for user, batch in enumerate(record['batches']):
        status('replaying_cpu_learning', user=user)
        assert set(batch).issubset(set(record['retained'][user]))
        if batch:
            gradient = local_gradient(model, theta, dataset, text, batch, device, config['microbatch_size'])
            learning[user] = (-rate*len(record['retained'][user])/retained_count*gradient).numpy()
        forgotten = record['forget'][user]
        if not forgotten:
            continue
        status('replaying_cpu_coupled_unlearning', user=user)
        forgotten_batch = np.random.default_rng(config['seed']*100000+61*100+user).choice(forgotten, min(config['batch_size'], len(forgotten)), replace=False).tolist()
        grad = local_gradient(model, theta, dataset, text, forgotten_batch, device, config['microbatch_size'])
        if batch:
            pixels, targets = dataset.batch(batch[:config['microbatch_size']], device)
            loss_function = lambda value: functional.cross_entropy(model(pixels, **text, theta=value), targets)
            grad, stats = projected_coupling(loss_function, theta, grad, model.visual_dimension, model.text_dimension,
                rank=config['coupling_rank'], kappa_requested=config['coupling_kappa'], xi=config['coupling_xi'],
                difference_step=config['coupling_difference_step'], seed=config['seed']+61*100+user)
            coupling.append(dict(user=user, **stats))
        unlearning[user] = (rate*len(forgotten)/forget_count*grad).numpy()
    with np.load(source/'partition.npz') as data:
        channels = data['channels'].copy()
    np.savez_compressed(out/'vectors.npz', learning=learning, unlearning=unlearning, channels=channels,
                        rank1_basis=normalized[:, -1:], rank3_basis=left[:, :3], public_scale=float(norms[-1]))
    result = dict(status='completed_actual_request_cpu_replay', at_kst=datetime.now().isoformat(timespec='seconds'),
                  actual_round=61, learning_rate=float(rate), forgotten_samples=forget_count,
                  requesting_users=[k for k, x in enumerate(record['forget']) if x], coupling=coupling,
                  plan_sha256=sha(out/'plan.json'), vectors_sha256=sha(out/'vectors.npz'),
                  source_schedule_sha256=sha(source/'schedule.json'), public_state_metadata_sha256=sha(observer/'states.json'))
    (out/'results.json').write_text(json.dumps(result, indent=2)+'\n'); status(result['status'])
    print('SCREEN', json.dumps(result), flush=True)
    del model, dataset, text, gradient, grad
    import gc
    gc.collect()
    subprocess.run([sys.executable, '-u', str(scripts/'radio.py'), '--project', str(root)], cwd=root, check=True)


if __name__ == '__main__':
    main()
