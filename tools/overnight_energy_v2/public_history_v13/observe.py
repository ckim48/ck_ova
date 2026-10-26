"""Read-only capture of public broadcast states; never alters the trainer."""
import argparse
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import time
import torch


def identity(pid):
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()
        return None if fields[0]=='Z' else fields[19]
    except FileNotFoundError:return None


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,required=True)
    p.add_argument('--pid',type=int,required=True);args=p.parse_args();root=args.project.resolve()
    source=root/'results/comparisons_seed0_solver_v2/puf_regular_adapted_hybrid'
    out=root/'results/overnight_energy_v2/public_history_v13/observer';out.mkdir(parents=True,exist_ok=True)
    assert not (out/'plan.json').exists(),'Do not duplicate the observer'
    expected=json.loads((root/'results/comparisons_seed0_solver_v2/suite_metadata.json').read_text())['source_sha256']
    assert all(hashlib.sha256((root/n).read_bytes()).hexdigest()==v for n,v in expected.items())
    process_identity=identity(args.pid);assert process_identity is not None
    plan=dict(source_method='puf_regular_adapted_hybrid',source_pid=args.pid,source_process_identity=process_identity,
              started_at_kst=datetime.now().isoformat(timespec='seconds'),core_sha256=expected,
              observer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              purpose='Collect already-public global theta states near the end of this existing run, for offline predictive/subspace transport diagnostics. No training or GPU operation.',
              limits='These are PUF trajectory states, not an original-coupled trajectory or private local update history. Every compared transport must receive the same public state in a paired diagnostic; a full new run must build its own history. Extra coefficient signaling and numerical reconstruction errors must be paid/checked before an energy claim.')
    (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');torch.set_num_threads(1)
    rows=[];last_round=-1;errors=0;deadline=time.monotonic()+900
    while True:
        try:
            blob=(source/'latest.pt').read_bytes()
            state=torch.load(io.BytesIO(blob),map_location='cpu',weights_only=True)
            round_id=int(state['round']);theta=state['theta']
            assert theta.ndim==1 and theta.numel()==20481 and bool(torch.isfinite(theta).all())
            if round_id>last_round:
                target=out/f'round_{round_id:03}.pt';temporary=target.with_suffix('.tmp')
                assert not target.exists();temporary.write_bytes(blob);temporary.replace(target)
                rows.append(dict(round=round_id,at_kst=datetime.now().isoformat(timespec='seconds'),
                                 file_sha256=hashlib.sha256(blob).hexdigest(),theta_sha256=hashlib.sha256(theta.numpy().tobytes()).hexdigest()))
                (out/'states.json').write_text(json.dumps(rows,indent=2)+'\n');last_round=round_id
        except (EOFError,RuntimeError,OSError,ValueError):errors+=1
        finished=last_round>=100 or identity(args.pid)!=process_identity or time.monotonic()>deadline
        status=dict(status='completed_read_only_observation' if finished else 'observing_public_states',
                    at_kst=datetime.now().isoformat(timespec='seconds'),states=len(rows),last_round=last_round,transient_read_errors=errors)
        (out/'status.json').write_text(json.dumps(status,indent=2)+'\n')
        if finished:break
        time.sleep(2)
    print(json.dumps(status),flush=True)


if __name__=='__main__':main()
