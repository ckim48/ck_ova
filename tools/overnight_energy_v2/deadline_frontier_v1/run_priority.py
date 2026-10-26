"""Run only after the current GPU method exits; always resume its paused queue."""
from pathlib import Path
import json,os,signal,subprocess,time
ROOT=Path(__file__).resolve().parents[3]
state=ROOT/'work/convergence_energy_delay_v1/priority_radio_guard.json'
s=json.loads(state.read_text());parent=s['paused_queue'];worker=s['waiting_worker']
try:
 while True:
  p=Path(f'/proc/{worker}/stat')
  if not p.exists() or p.read_text().split()[2]=='Z':break
  time.sleep(10)
 gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
 assert not gpu, 'Do not interfere with another GPU workload: '+gpu
 s['status']='running_radio_frontier';state.write_text(json.dumps(s,indent=2))
 env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
 r=subprocess.run([str(ROOT/'.venv/bin/python'),'-u',str(Path(__file__).with_name('run_frontier.py'))],cwd=ROOT,env=env,timeout=5400)
 s['returncode']=r.returncode
finally:
 p=Path(f'/proc/{parent}/cmdline')
 if p.exists() and b'deletion_window_screen_v1/run_queue.py' in p.read_bytes():os.kill(parent,signal.SIGCONT)
 s['status']='radio_finished_gpu_queue_resumed';s['updated_unix']=time.time();state.write_text(json.dumps(s,indent=2))
