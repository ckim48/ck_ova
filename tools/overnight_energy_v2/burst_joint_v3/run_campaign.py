"""Serialize a prespecified revised-protocol comparison after existing work."""
import argparse
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run_variant import source_hashes


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pid_identity(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()
        return None if fields[0] == 'Z' else fields[19]
    except FileNotFoundError:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--wait-pid', type=int, required=True)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args(); root = args.project.resolve(); script_dir = Path(__file__).resolve().parent
    output = root/'results/overnight_energy_v2/burst_joint_v3'; output.mkdir(parents=True, exist_ok=True)
    baseline = json.loads((root/'results/comparisons_seed0_solver_v2/config.json').read_text())
    config = dict(baseline, deletion_relative_tolerance=.6)
    upstream = json.loads((root/'results/comparisons_seed0_solver_v2/suite_metadata.json').read_text())['source_sha256']
    if any(digest(root/n) != sha for n,sha in upstream.items()):
        raise ValueError('Original training sources changed.')
    plan = {'variant': 'burst_joint_v3', 'config': config, 'modes': ['hybrid', 'mixed_aircomp', 'phase_grid'],
        'source_sha256': source_hashes(root), 'audit_sha256': digest(script_dir/'audit_variants.py'),
        'numeric_evaluator_sha256': digest(root/'tools/energy_quality_v1/energy_quality.py'),
        'controller_sha256': digest(__file__), 'cohorts_sha256': digest(root/'results/energy_quality_v1/cohorts.json'),
        'adapter_validation_script_sha256': digest(script_dir/'validate_legacy_round.py'),
        'validation_updates_sha256': digest(root/'results/failed_round_diagnostic/updates.npz'),
        'wait_pid': args.wait_pid, 'no_new_training_after_kst': '2026-10-21T12:00:00+09:00',
        'screening_evidence': ['Completed fixed-update error-budget map and joint-power v3 comparison.',
            'Completed 200-validation/200-forget, three-noise-seed single-step VLM screen at 8 bits, relative error 0.6; no prediction flips, output KL below 1e-6 in all methods. This is not a retraining reference.'],
        'reason': 'Energy spikes occur when learning and unlearning share the channel. Test a common larger update-error allowance only in those rounds while preserving the original learning-only allowance. Compare joint Hybrid to the strong mixed baseline and grid-optimized phase transmission under the same changed protocol.',
        'integrity': 'Original source/results preserved. Fixed mode order and config, no test-based choice. Each GPU training/audit is serialized after existing quality and retain-only control. No overwriting or automatic resume. Baselines remain fully eligible.'}
    plan_path = output/'plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError('Existing campaign plan differs; choose another version.')
    plan_path.write_text(json.dumps(plan, indent=2)+'\n')
    (output/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    if args.plan_only:
        print(json.dumps(plan, indent=2)); return
    def status(value, **extra):
        path = output/'status.json'; temporary = path.with_suffix('.partial.json')
        temporary.write_text(json.dumps(dict(status=value, pid=os.getpid(), at_kst=datetime.now().isoformat(), **extra), indent=2)+'\n'); temporary.replace(path)
    with (output/'campaign.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status('waiting_for_existing_retain_control', wait_pid=args.wait_pid)
        identity = pid_identity(args.wait_pid)
        while identity is not None and pid_identity(args.wait_pid) == identity:
            time.sleep(30)
        validation_path = output/'legacy_adapter_validation.json'
        validation = json.loads(validation_path.read_text()) if validation_path.exists() else {}
        if (validation.get('status') != 'passed' or validation.get('source_sha256') != plan['source_sha256']
                or validation.get('updates_sha256') != plan['validation_updates_sha256']
                or validation.get('validation_script_sha256') != plan['adapter_validation_script_sha256']):
            status('not_started_adapter_validation_missing_or_failed'); return
        deadline = datetime(2026,10,21,12,tzinfo=timezone(timedelta(hours=9)))
        logs = output/'logs'; logs.mkdir(exist_ok=True)
        env = dict(os.environ, OPENBLAS_NUM_THREADS='4', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
        completed = {}
        for mode in plan['modes']:
            while True:
                active = []
                for p in Path('/proc').glob('[0-9]*/cmdline'):
                    try:
                        command = p.read_bytes().replace(b'\0', b' ').decode(errors='replace')
                    except (FileNotFoundError, ProcessLookupError, PermissionError):
                        continue
                    if str(root)+'/' in command and any(x in command for x in ['run_clip_federated.py', 'run_variant.py', 'energy_quality.py', 'audit_variants.py']):
                        active.append(p.parent.name)
                if not active:
                    break
                status('waiting_for_other_project_gpu_work', active_pids=active)
                time.sleep(30)
                if datetime.now(timezone(timedelta(hours=9))) >= deadline:
                    break
            if datetime.now(timezone(timedelta(hours=9))) >= deadline:
                status('window_ended_no_more_training', outcomes=completed, next_mode=mode); return
            if source_hashes(root) != plan['source_sha256']:
                raise ValueError('Frozen experimental source changed before training.')
            directory = output/mode
            if directory.exists():
                completed[mode] = 'existing_output_not_overwritten'
                continue
            status('training', mode=mode, outcomes=completed)
            with (logs/(mode+'.log')).open('x') as log:
                result = subprocess.run([sys.executable, '-u', str(script_dir/'run_variant.py'),
                    '--project', str(root), '--config', str(output/'config.json'), '--output', str(directory), '--mode', mode],
                    cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
            completed[mode] = 'training_finished' if result.returncode == 0 else 'training_failed_exit_'+str(result.returncode)
            status('evaluating_started_trajectory', mode=mode, outcomes=completed)
            # Evaluation completes the trajectory that began before noon; it may finish later.
            with (logs/'quality.log').open('a') as log:
                audited = subprocess.run([sys.executable, '-u', str(script_dir/'audit_variants.py'),
                    '--project', str(root), '--output', str(output)], cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
            if audited.returncode:
                status('audit_failed', mode=mode, exit_code=audited.returncode, outcomes=completed); return
        status('completed', outcomes=completed)


if __name__ == '__main__':
    main()
