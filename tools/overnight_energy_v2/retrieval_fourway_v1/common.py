from pathlib import Path
import hashlib,json,sys,time
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OUT=ROOT/'results/overnight_energy_v2/retrieval_fourway_v1'
sys.path[:0]=[str(HERE),str(ROOT),str(ROOT/'tools/overnight_energy_v2'),
 str(ROOT/'tools/overnight_energy_v2/initialization_v11'),
 str(ROOT/'tools/overnight_energy_v2/receiver_control_v1')]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');t.replace(p)
def status(**x):write(OUT/'status.json',dict(updated_unix=time.time(),**x))

