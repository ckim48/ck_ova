"""Pre-registered actual round30 replay, followed by paid16bit radio checks."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', type=Path, required=True)
    args = p.parse_args(); root = args.project.resolve()
    family = root/'results/overnight_energy_v2/public_history_early_v17'
    out = family/'screen'; out.mkdir(exist_ok=True)
    observer = family/'observer'; source = root/'results/overnight_energy_v2/burst_joint_v3/hybrid'
    observer_plan = json.loads((observer/'plan.json').read_text())
    config = json.loads((root/'results/overnight_energy_v2/burst_joint_v3/config.json').read_text())
    core = observer_plan['core_sha256']; assert all(sha(root/n) == v for n, v in core.items())
    parent = json.loads((root/'results/overnight_energy_v2/public_history_fp16_v15/manifest.json').read_text())
    scripts = Path(__file__).parent
    assert sha(scripts/'codec.py') == parent['code_sha256']['codec.py']
    assert sha(root/'tools/overnight_energy_v2/initialization_v11/wireless_init_v11.py') == parent['solver_sha256']
    plan = dict(version='public_history_early_v17', registered_at_kst=datetime.now().isoformat(timespec='seconds'),
                actual_round=30, public_history_rounds=[26, 27, 28, 29], core_sha256=core, config=config,
                observer_plan_sha256=sha(observer/'plan.json'), source_schedule_sha256=sha(source/'schedule.json'),
                code_sha256={n: sha(scripts/n) for n in ['replay.py', 'radio.py', 'codec.py', 'observe.py']},
                parent_codec_sha256=parent['code_sha256']['codec.py'], parent_solver_sha256=parent['solver_sha256'],
                variants=['unprojected_baseline', 'latest_global_increment', 'recent_public_span_rank3'],
                cpu_threads=4, purpose='Check preselected earlier actual round30, with substantially larger learning rate than the late round100 fixture. No selection of rounds/bases/codec from validation or test outputs.',
                limits=['No FU requests at this round. Generic FL transport only; all original modes reduce to one analog sum.',
                        'All variants use identical observed PUBLIC burst26..29 states, actual local round30 minibatches, channels, absolute RMS0.2, total airtime, power, battery and AO30/5starts.',
                        'Each16bit/rank coefficient and all TDMA airtime/RF energy paid, with residual formed from decoded coefficients. No private history or future-model information.',
                        'No VLM inference/test selection and no automatic GPU promotion. Projection, basis-generation, memory/coding compute energy remain unmeasured; report network break-even.'])
    assert not (out/'plan.json').exists(), 'Do not duplicate a registered early replay.'
    (out/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    def status(value, **extra):
        path = out/'status.json'; tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(dict(status=value, at_kst=datetime.now().isoformat(timespec='seconds'), **extra), indent=2)+'\n'); tmp.replace(path)
    deadline = time.monotonic()+2400
    status('waiting_for_public_theta29')
    while True:
        state = json.loads((observer/'status.json').read_text())
        if state['status'].startswith('completed'):
            break
        if time.monotonic() > deadline:
            status('skipped_observer_wait_timeout'); return
        time.sleep(10)
    rows = json.loads((observer/'states.json').read_text())
    import torch
    torch.set_num_threads(4); torch.manual_seed(config['seed'])
    public = []
    for k in [26, 27, 28, 29]:
        path = observer/f'round_{k:03}.pt'
        assert sha(path) == next(x['file_sha256'] for x in rows if x['round'] == k)
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        assert checkpoint['round'] == k < 30
        public.append(checkpoint['theta'].numpy().astype(float))
    increments = np.stack([b-a for a, b in zip(public, public[1:])], axis=1)
    norms = np.linalg.norm(increments, axis=0); assert np.all(norms > 0)
    normalized = increments/norms
    left, singular, _ = np.linalg.svd(normalized, full_matrices=False)
    assert np.sum(singular > singular[0]*1e-10) == 3
    bases = {'latest_global_increment': normalized[:, -1:], 'recent_public_span_rank3': left[:, :3]}
    sys.path.insert(0, str(root))
    from clip_model import DeepPromptCLIP
    from real_data import CIFAR100
    from run_clip_federated import local_gradient
    theta = torch.as_tensor(public[-1], dtype=torch.float32); device = torch.device('cpu')
    model_path = root/'checkpoints/clip-vit-base-patch16'
    dataset = CIFAR100(root/'data/cifar-100-python', model_path)
    model = DeepPromptCLIP.from_local_checkpoint(model_path, prompt_layers=config['prompt_layers'], prompt_tokens=config['prompt_tokens'], checkpoint_layers=False).eval()
    text = dataset.text_inputs(device, config['prompt_tokens'])
    record = json.loads((source/'schedule.json').read_text())[29]
    assert record['round'] == 30 and not any(record['forget'])
    alpha = np.asarray([len(x) for x in record['retained']], float); alpha /= alpha.sum()
    eta = config['learning_rate']*.5*(1+np.cos(np.pi*29/config['rounds']))
    learning = []
    for user, batch in enumerate(record['batches']):
        assert set(batch).issubset(set(record['retained'][user]))
        status('replaying_actual_cpu_local_update', user=user)
        gradient = local_gradient(model, theta, dataset, text, batch, device, config['microbatch_size'])
        learning.append((-eta*alpha[user]*gradient).numpy().astype(float))
    learning = np.stack(learning)
    with np.load(source/'partition.npz') as data:
        channels = data['channels'].copy()
    saved = dict(learning=learning, channels=channels, alpha=alpha, public_scale=float(norms[-1]))
    info = []
    for name, basis in bases.items():
        saved[name+'_basis'] = basis
        info.append(dict(basis=name, rank=basis.shape[1]))
    np.savez_compressed(out/'vectors.npz', **saved)
    result = dict(status='completed_actual_early_cpu_replay', round=30, public_rounds=[26, 27, 28, 29],
                  at_kst=datetime.now().isoformat(timespec='seconds'), learning_rate=float(eta), public_scale=float(norms[-1]),
                  rows=info, plan_sha256=sha(out/'plan.json'), observer_states_sha256=sha(observer/'states.json'),
                  source_schedule_sha256=sha(source/'schedule.json'), config=config)
    (out/'results.json').write_text(json.dumps(result, indent=2)+'\n')
    status('completed_actual_early_cpu_replay')
    print('SCREEN', json.dumps(result), flush=True)
    # Release model objects before the separate CPU-only radio calculation.
    del model, dataset, text, gradient
    import gc
    gc.collect()
    subprocess.run([sys.executable, str(scripts/'radio.py'), '--project', str(root)], cwd=root, check=True)


if __name__ == '__main__':
    main()
