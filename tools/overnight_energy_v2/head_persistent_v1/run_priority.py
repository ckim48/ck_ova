"""Temporarily suspend MemPrune, run the short pilot, then resume exact state.

The paused job retains its GPU mutex and CUDA memory, blocking other queues.
Only the priority worker computes on the GPU during this documented lease.
No experiment source or checkpoint is modified. A separate watchdog restores
the paused process after supervisor failure, worker failure, or a one-hour cap.
"""
from pathlib import Path
import os, sys, json, time, signal, subprocess, fcntl, traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT/'results/overnight_energy_v2/head_persistent_v1'
MEM = 4065463
WAITING = 4110339
LEASE = OUT/'priority_lease.json'
DONE = OUT/'priority_finished.json'
MAX_SECONDS = 3600

def write(path, data):
    tmp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)

def identity(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()
        return {'pid': pid, 'start': fields[19], 'state': fields[0]}
    except FileNotFoundError:
        return None

def alive(item):
    now = identity(item['pid'])
    return now is not None and now['start'] == item['start'] and now['state'] != 'Z'

def command(pid):
    return Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()

def gpu_pids():
    text = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
    return {int(x.strip()) for x in text.splitlines() if x.strip().isdigit()}

def terminate(item):
    if not item or not alive(item): return
    os.kill(item['pid'], signal.SIGTERM)
    for _ in range(100):
        if not alive(item): return
        time.sleep(.1)
    if alive(item): os.kill(item['pid'], signal.SIGKILL)
    for _ in range(100):
        if not alive(item): return
        time.sleep(.1)
    raise RuntimeError('Worker could not be terminated')

def recover(lease, reason):
    terminate(lease.get('worker'))
    # Wait for the worker CUDA context to disappear before restoring MemPrune.
    worker = lease.get('worker', {})
    for _ in range(120):
        if worker.get('pid') not in gpu_pids(): break
        time.sleep(.5)
    else: raise RuntimeError('CUDA teardown did not finish; do not overlap jobs')
    mem = lease['memprune']
    resumed = False
    if alive(mem):
        if identity(mem['pid'])['state'] in ('T', 't'):
            os.kill(mem['pid'], signal.SIGCONT)
        resumed = True
    write(DONE, {'unix': time.time(), 'reason': reason, 'memprune_resumed': resumed,
                 'pause_wall_s': time.time()-lease['unix'],
                 'worker_returncode': lease.get('worker_returncode')})

def watchdog():
    while True:
        if DONE.exists(): return
        lease = json.loads(LEASE.read_text())
        if not alive(lease['supervisor']) or time.time()-lease['unix'] > MAX_SECONDS+60:
            # Handle an interruption between Popen and the worker receipt.
            if not lease.get('worker'):
                for p in Path('/proc').iterdir():
                    if not p.name.isdigit(): continue
                    try:
                        cmd = command(int(p.name))
                        if str(HERE/'run_priority.py') in cmd and '--worker' in cmd:
                            lease['worker'] = identity(int(p.name)); break
                    except (FileNotFoundError, PermissionError): pass
            recover(lease, 'watchdog_recovery')
            return
        time.sleep(2)

def worker():
    lease = json.loads(LEASE.read_text())
    assert alive(lease['supervisor']), 'Supervisor is unavailable'
    assert alive(lease['memprune']) and identity(MEM)['state'] in ('T', 't')
    assert gpu_pids() <= {MEM}, 'Unexpected GPU user'
    import run_pilot as pilot
    plan, protocol = pilot.prepare()
    assert not (OUT/'started.json').exists(), 'Refuse duplicate pilot'
    for name, digest in protocol['sources'].items():
        assert pilot.core.sha(ROOT/name) == digest, name
    pilot.write(OUT/'started.json', {'unix': time.time(), 'pid': os.getpid(),
        'scheduler': 'priority pause/resume', 'paused_memprune_pid': MEM})
    try:
        pilot.status(stage='loading_model', scheduler='priority')
        pilot.run(plan)
        for name, digest in protocol['sources'].items():
            assert pilot.core.sha(ROOT/name) == digest, name
    except Exception as exc:
        pilot.status(stage='failed', error=repr(exc)); raise

def supervise():
    own = (OUT/'priority.lock').open('a')
    fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not LEASE.exists() and not DONE.exists(), 'Priority run already registered'
    assert not (OUT/'started.json').exists(), 'Pilot already started'
    assert 'full_system_baselines_v1/run_memprune.py' in command(MEM)
    assert Path(f'/proc/{MEM}/cwd').resolve() == ROOT
    assert 'head_persistent_v1/run_pilot.py' in command(WAITING)
    assert gpu_pids() == {MEM}, 'Unexpected GPU job'
    free = int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free',
                                       '--format=csv,noheader,nounits'], text=True).strip())
    assert free >= 4096, 'Insufficient spare GPU memory'
    queued = json.loads((OUT/'status.json').read_text())
    assert queued['stage'] == 'queued_waiting_for_gpu_lock' and queued['pid'] == WAITING
    # No model was loaded in this waiter; its only job is waiting on the mutex.
    terminate(identity(WAITING))
    claim = (OUT/'pilot.lock').open('a')
    fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lease = {'unix': time.time(), 'supervisor': identity(os.getpid()),
        'memprune': identity(MEM), 'replaced_waiter': WAITING,
        'method': 'SIGSTOP/SIGCONT, exact in-memory continuation, no source edits',
        'pre_pause_status': json.loads((ROOT/'results/overnight_energy_v2/full_system_baselines_v1/status.json').read_text()),
        'max_pilot_wall_s': MAX_SECONDS}
    write(LEASE, lease)
    guard_log = (OUT/'priority_watchdog.log').open('a')
    subprocess.Popen([sys.executable, str(HERE/'run_priority.py'), '--watchdog'],
        stdin=subprocess.DEVNULL, stdout=guard_log, stderr=subprocess.STDOUT,
        start_new_session=True)
    def interrupted(signum, frame): raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    reason = 'priority_worker_failed'
    try:
        os.kill(MEM, signal.SIGSTOP)
        for _ in range(100):
            if identity(MEM)['state'] in ('T', 't'): break
            time.sleep(.1)
        else: raise RuntimeError('MemPrune did not suspend')
        # Let outstanding short CUDA kernels finish; require two quiet samples.
        quiet = 0
        for _ in range(30):
            util = int(subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu',
                           '--format=csv,noheader,nounits'], text=True).strip())
            quiet = quiet+1 if util <= 5 else 0
            if quiet >= 2: break
            time.sleep(1)
        else: raise RuntimeError('GPU did not drain after suspension')
        log = (OUT/'priority_pilot.log').open('a')
        child = subprocess.Popen([str(ROOT/'.venv/bin/python'), '-u', str(HERE/'run_priority.py'), '--worker'],
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        lease['worker'] = identity(child.pid); write(LEASE, lease)
        while child.poll() is None:
            assert alive(lease['memprune']) and identity(MEM)['state'] in ('T', 't'), 'MemPrune unexpectedly resumed'
            if time.time()-lease['unix'] > MAX_SECONDS: raise TimeoutError('Priority pilot time cap')
            time.sleep(2)
        lease['worker_returncode'] = child.returncode; write(LEASE, lease)
        reason = 'pilot_completed' if child.returncode == 0 else 'priority_worker_failed'
    finally:
        recover(lease, reason)

if __name__ == '__main__':
    try:
        if '--watchdog' in sys.argv: watchdog()
        elif '--worker' in sys.argv: worker()
        else: supervise()
    except BaseException:
        traceback.print_exc(); raise
