"""Run the unchanged training loop with an explicitly versioned transport."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def source_hashes(root):
    names = ['run_clip_federated.py', 'clip_model.py', 'transport.py', 'wireless.py',
             'real_data.py', 'data_schedule.py', 'tools/overnight_energy_v2/phase_split_grid.py',
             'tools/overnight_energy_v2/joint_power_v3/wireless_joint_v3.py',
             'tools/overnight_energy_v2/burst_joint_v3/burst_transport.py',
             'tools/overnight_energy_v2/burst_joint_v3/run_variant.py']
    return {n: hashlib.sha256((root/n).read_bytes()).hexdigest() for n in names}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('hybrid', 'mixed_aircomp', 'phase_grid'), required=True)
    args = parser.parse_args(); root = args.project.resolve()
    sys.path[:0] = [str(root), str(root/'tools/overnight_energy_v2'),
                   str(root/'tools/overnight_energy_v2/joint_power_v3')]
    import run_clip_federated as training
    import burst_transport
    plan = json.loads((args.output.parent/'plan.json').read_text())
    hashes = source_hashes(root)
    if hashes != plan['source_sha256']:
        raise ValueError('Experimental sources differ from the frozen plan.')
    config = json.loads(args.config.read_text())
    if config != plan['config']:
        raise ValueError('Experimental configuration differs from the frozen plan.')
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise FileExistsError('Existing variant output will not be overwritten.')
    provenance = {'variant': 'burst_joint_v3', 'source_sha256': hashes,
        'config': config, 'mode': args.mode,
        'change': 'Runtime transport substitution only; unchanged local updates, canonical data schedule and initial model. Original energy/peak/battery/SIC constraints remain in force.'}
    (args.output/'variant_provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    training.communicate = burst_transport.communicate
    train_args = argparse.Namespace(config=args.config, output=args.output, seed=None,
        algorithm='coupled', transport=args.mode, device='cuda', cpu_threads=4)
    try:
        training.execute(train_args)
    finally:
        path = args.output/'manifest.json'
        if path.exists():
            manifest = json.loads(path.read_text())
            manifest['upstream_source_sha256'] = manifest['source_sha256']
            manifest['source_sha256'] = hashes
            manifest['variant'] = provenance['variant']
            manifest['transport_policy'] = 'RMS relative tolerance 0.2 normally, 0.6 for nonzero unlearning updates; common to all modes. Phase time split grid-optimized.'
            path.write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
