"""Queue the approved energy-overrun sensitivity run behind active GPU work."""
from pathlib import Path
import os,json,subprocess,sys,time
root=Path(__file__).resolve().parents[2]
out=root/'results/iot_uav_remoteclip_v3_energy_overrun';out.mkdir(exist_ok=True)
assert not (out/'launch.json').exists(),'Continuation already queued'
env=dict(os.environ,PYTHONPATH=str(root/'.cache/uav_experiment_deps'),MPLCONFIGDIR='/tmp/ova_ck_mpl',GIT_TERMINAL_PROMPT='0')
with (out/'run.log').open('ab',buffering=0) as log:
    child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).with_name('run.py'))],cwd=root,env=env,
                           stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
(out/'launch.json').write_text(json.dumps(dict(pid=child.pid,unix=time.time(),detached_session=True),indent=2))
print('Queued energy-overrun continuation PID:',child.pid,flush=True)
