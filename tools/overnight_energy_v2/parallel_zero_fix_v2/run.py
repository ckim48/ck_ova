"""Finish the registered experiment if its last, zero-LR step hits an empty solver.

No running/frozen source is edited. Resume saved trajectories, same configuration.
Zero FL and FU payload needs no radio transmission, identically for all arms.
"""
from pathlib import Path
import os,sys,time,json,hashlib,inspect,importlib.util,fcntl,traceback
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'results/overnight_energy_v2/parallel_aircomp_v1'
WATCH_PID=105979;WATCH_START='880806825'

def identity(pid):
    p=Path('/proc')/str(pid)
    if not p.exists():return None
    s=(p/'stat').read_text().split(') ',1)[1].split()
    return dict(pid=pid,start=s[19],state=s[0],cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode())

def save(name,value):
    p=OUT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(value,indent=2));t.replace(p)

def main():
    guard=(OUT/'zero_fix_queue.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    original=identity(WATCH_PID)
    assert original and original['start']==WATCH_START and 'parallel_aircomp_v1/run.py' in original['cmd']
    save('zero_fix_registered.json',dict(unix=time.time(),identity=identity(os.getpid()),predecessor=original,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        reason='Cosine LR at round200 is exactly zero; all-zero vector makes the original convex variable dimension zero. Handle silence for every arm; no hyperparameter or quality-based change.',
        policy='Wait for predecessor to exit. Resume only a verified empty-dimension failure at checkpoint199. All old results/logs remain. Other failures require inspection.'))
    while True:
        now=identity(WATCH_PID)
        if now is None or now['state']=='Z' or now['start']!=WATCH_START:break
        time.sleep(10)
    if (OUT/'finished.json').exists():return
    status=json.loads((OUT/'status.json').read_text());log=(OUT/'run.log').read_text()
    assert status.get('stage')=='failed' and 'Invalid dimensions (0,)' in log,(status,log[-300:])
    assert (OUT/'proposed/rounds/199.json').exists() and not (OUT/'proposed/rounds/200.json').exists()
    sys.path.insert(0,str(ROOT/'tools/overnight_energy_v2/parallel_aircomp_v1'))
    spec=importlib.util.spec_from_file_location('parallel_resume_base',ROOT/'tools/overnight_energy_v2/parallel_aircomp_v1/run.py')
    base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
    import numpy as np
    old_transport=base.radio.communicate
    def transport(l,u,channels,c,mode,seed):
        if np.any(l) or np.any(u):return old_transport(l,u,channels,c,mode,seed)
        return np.zeros(l.shape[1]),dict(status='feasible',physical_mode=mode,access='zero_update_silence',
            head=None,energy_bound_j=0.,expected_energy_j=0.,uplink_s=0.,frame_count=0,
            network_tx_cap_j=base.radio.CAP,max_user_energy_j=0.,max_power_w=0.,
            error_bound=0.,epsilon=c['absolute_tolerance_floor'],conditional_rmse=0.,
            quantization_error_bound=0.,requests_in_update=0,realized_error_to_raw=0.,
            optimizer_wall_s=0.,phases=[],zero_update_audit=True)
    base.radio.communicate=transport
    c={'absolute_tolerance_floor':1e-8};zero=np.zeros((10,20480))
    for mode in ['joint','conventional_joint','separate','ofdma']:
        v,m=transport(zero,zero,np.ones((10,10)),c,mode,0)
        assert not np.any(v) and m['uplink_s']==0 and m['energy_bound_j']==0
    original_write=base.core.write
    def write(path,value):
        path=Path(path)
        if path==OUT/'started.json':path=OUT/'zero_fix_resumed.json'
        original_write(path,value)
    base.core.write=write
    source=inspect.getsource(base.main)
    needle="assert not (OUT/'started.json').exists(),'Duplicate launch'"
    assert source.count(needle)==1
    source=source.replace(needle,"assert not (OUT/'finished.json').exists(),'Already completed'")
    save('zero_fix_applied.json',dict(unix=time.time(),predecessor_failure=status,zero_update_checks=True,
        affects='Only exactly zero FL/FU at the zero-learning-rate endpoint; all nonzero update calls go to the unchanged transport.',
        resume='Original per-round checkpoint logic; replay only unfinished rounds; reuse all complete40-round stages.',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    # Avoid replaying stage40 evaluation/manifest on a saved round199 trajectory.
    assert source.count('for stop in [40,200]:')==1
    source=source.replace('for stop in [40,200]:','for stop in [200]:')
    exec(compile(source,str(Path(__file__))+' [resume main]','exec'),base.__dict__)
    try:base.main()
    finally:base.radio.close()

if __name__=='__main__':
    try:main()
    except Exception as e:
        save('zero_fix_failed.json',dict(unix=time.time(),error=repr(e)));traceback.print_exc();raise
