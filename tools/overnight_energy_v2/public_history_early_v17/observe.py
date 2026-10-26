"""Observe existing burst broadcasts through29; do not alter the GPU job."""
import argparse
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import time
import torch


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def identity(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()
        return None if fields[0] == 'Z' else fields[19]
    except FileNotFoundError:
        return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', type=Path, required=True)
    p.add_argument('--pid', type=int, required=True)
    args = p.parse_args(); root = args.project.resolve()
    source = root/'results/overnight_energy_v2/burst_joint_v3/hybrid'
    out = root/'results/overnight_energy_v2/public_history_early_v17/observer'; out.mkdir(parents=True, exist_ok=True)
    assert not (out/'plan.json').exists(), 'Do not duplicate the observer.'
    core = json.loads((root/'results/comparisons_seed0_solver_v2/suite_metadata.json').read_text())['source_sha256']
    assert all(sha(root/n) == v for n, v in core.items())
    process_identity = identity(args.pid); assert process_identity is not None
    torch.set_num_threads(1)
    initial = torch.load(source/'latest.pt', map_location='cpu', weights_only=True)
    assert initial['round'] < 27, 'Pre-register the earlier-stage screen before required public states occur.'
    plan = dict(started_at_kst=datetime.now().isoformat(timespec='seconds'), source_gpu_pid=args.pid,
                source_process_identity=process_identity, source_directory=str(source), observer_sha256=sha(__file__),
                core_sha256=core, target_local_round=30, last_usable_public_round=29, basis_history_rounds=[26, 27, 28, 29],
                fixed_variants=['unprojected', 'latest_public_increment_rank1', 'last3_public_increment_span_rank3'],
                metadata='Same v15 shared-scale IEEEhalf coefficients, fixed16bits/rank, paid32symbols/rank at0.5bit/symbol, no new tuned parameters.',
                purpose='Independent earlier learning-stage validation of the modest late-round radio saving before any long training promotion.',
                limits='Read-only observation of actual burst trajectory, not a coupled-original or FU request. All compared transports receive the same past public information; no private gradient history or future states. Do not alter current GPU queue/source hashes.')
    (out/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    rows = []; last_round = -1; errors = 0; deadline = time.monotonic()+2400
    while True:
        try:
            blob = (source/'latest.pt').read_bytes()
            state = torch.load(io.BytesIO(blob), map_location='cpu', weights_only=True)
            k = int(state['round'])
            if k > last_round and k <= 29:
                assert state['theta'].numel() == 20481 and bool(torch.isfinite(state['theta']).all())
                target = out/f'round_{k:03}.pt'; assert not target.exists()
                target.write_bytes(blob)
                rows.append(dict(round=k, file_sha256=hashlib.sha256(blob).hexdigest(), captured_at_kst=datetime.now().isoformat(timespec='seconds')))
                (out/'states.json').write_text(json.dumps(rows, indent=2)+'\n'); last_round = k
        except (EOFError, RuntimeError, OSError, ValueError):
            errors += 1
        finished = last_round >= 29 or identity(args.pid) != process_identity or time.monotonic() > deadline
        status = dict(status='completed_read_only_observation' if finished else 'observing_public_states',
                      at_kst=datetime.now().isoformat(timespec='seconds'), last_round=last_round, states=len(rows), transient_read_errors=errors)
        (out/'status.json').write_text(json.dumps(status, indent=2)+'\n')
        if finished:
            break
        time.sleep(2)
    print(json.dumps(status), flush=True)


if __name__ == '__main__':
    main()
