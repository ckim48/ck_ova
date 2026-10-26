"""Explicit manual recovery ONLY; never automatically invoked by the new study.

Resume the interrupted MemPrune reference from the saved global model at180.
Local Adam is reinitialized and seeded per client/round by the frozen routine.
Original code stays immutable; existing probabilities/checkpoints are preserved.
"""
from pathlib import Path
import importlib.util,inspect,hashlib
ROOT=Path(__file__).resolve().parents[3]
source=ROOT/'tools/overnight_energy_v2/full_system_audit_v1/matched_reference.py'
expected='aae4247a200347aa4167ab5d13fc244059b5a30513e6239d408399f6f621e2c2'
assert hashlib.sha256(source.read_bytes()).hexdigest()==expected
spec=importlib.util.spec_from_file_location('preserved_matched_reference',source)
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
dest=mod.run.OUT/'fedmemprune_matched_reference'
assert not (dest/'finished.json').exists()
saved=sorted(dest.glob('round_*.pt'));checkpoint=saved[-1]
last=int(checkpoint.stem.split('_')[1]);assert last>=180
text=inspect.getsource(mod.main)
old="assert not (dest/'started.json').exists(),'Duplicate reference launch'"
assert text.count(old)==1
text=text.replace(old,"assert not (dest/'finished.json').exists(),'Already complete'")
old="theta=run.core.load_theta(run.core.SOURCE/'retrain/round_020.pt',engine.device)"
assert text.count(old)==1
text=text.replace(old,f"theta=run.core.load_theta(dest/'{checkpoint.name}',engine.device)")
assert text.count('range(22,201)')==1
text=text.replace('range(22,201)',f'range({last+1},201)')
original_write=mod.run.core.write
def protected_write(path,value):
    path=Path(path)
    if path.parent==dest and path.name=='started.json':
        path=dest/'resumed_from_checkpoint.json'
        value=dict(value,checkpoint=str(checkpoint),resume_round=last+1)
    if path.parent==dest and path.name=='protocol.json' and path.exists():return
    original_write(path,value)
mod.run.core.write=protected_write
exec(compile(text,str(source)+' [checkpoint recovery]', 'exec'),mod.__dict__)
if __name__=='__main__':mod.main()
