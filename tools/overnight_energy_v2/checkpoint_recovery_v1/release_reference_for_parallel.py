"""Checkpoint-preserving release of a verified stopped GPU lock owner."""
from pathlib import Path
import os,signal,time,json,hashlib,shutil
import torch
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'results/overnight_energy_v2/parallel_aircomp_v1'
OUT.mkdir(exist_ok=True)
expected={74211:('879993836','adaptive_ofdma_v2/run.py'),4127621:('876222186','codec_quality_v1/run.py'),4065456:('874548230','full_system_audit_v1/queue.py'),4177738:('877568358','full_system_audit_v1/matched_reference.py')}
def identity(pid):
    p=Path('/proc')/str(pid)
    if not p.exists():return None
    s=(p/'stat').read_text().split(') ',1)[1].split()
    return dict(pid=pid,start=s[19],state=s[0],cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode())
before={}
for pid,(start,name) in expected.items():
    now=identity(pid);assert now and now['start']==start and name in now['cmd'] and now['state']=='T',now
    before[str(pid)]=now
dest=ROOT/'results/overnight_energy_v2/full_system_baselines_v1/fedmemprune_matched_reference'
p=dest/'round_180.pt';z=torch.load(p,map_location='cpu',weights_only=True)
assert z['round']==180 and torch.isfinite(z['theta']).all()
backup=OUT/'preserved_reference';backup.mkdir(exist_ok=True)
for name in ['round_180.pt','probabilities_180.npz','protocol.json','started.json']:
    a=dest/name;b=backup/name
    if not b.exists():shutil.copy2(a,b)
    assert hashlib.sha256(a.read_bytes()).digest()==hashlib.sha256(b.read_bytes()).digest()
receipt=dict(unix=time.time(),reason='User explicitly requested new parallel AirComp experiments while unrelated queues remain stopped. Release stopped GPU owner after preserving its round180 checkpoint.',
    before=before,checkpoint=str(p),checkpoint_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
    discarded_unsaved_work='Stopped at round184/user5; rounds181 onward must be recomputed if this reference is explicitly resumed.',
    recovery='tools/overnight_energy_v2/checkpoint_recovery_v1/reference_checkpoint_recovery.py',automatic_resume=False)
(OUT/'reference_release_intent.json').write_text(json.dumps(receipt,indent=2))
os.kill(4177738,signal.SIGTERM)
os.kill(4177738,signal.SIGCONT)
for _ in range(50):
    now=identity(4177738)
    if now is None or now['state']=='Z':break
    time.sleep(.1)
else:raise RuntimeError('Reference did not exit; do not start GPU task')
receipt['after']={str(pid):identity(pid) for pid in expected}
for pid in [74211,4127621,4065456]:assert receipt['after'][str(pid)]['state']=='T'
(OUT/'reference_released.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(dict(released=True,checkpoint_round=180,other_queues_still_stopped=True)))
