"""Identical objective/protocol, CPU head candidates run concurrently."""
from pathlib import Path
import os,sys,json,shutil,time
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'tools/overnight_energy_v2/proxy_objective_v1'))
import run as core
import parallel_heads
core.OUT=ROOT/'results/overnight_energy_v2/proxy_objective_v2'
original_load=core.load_module

def load_module(name,path):
    module=original_load(name,path)
    # Frozen transport functions keep all targets, grids and audits unchanged.
    import radio
    radio.optimize_head=parallel_heads.parallel
    module.optimize_head=parallel_heads.parallel
    return module

def main():
    checks=json.loads((core.OUT/'parallel_checks.json').read_text())
    assert checks['passed']
    old=ROOT/'results/overnight_energy_v2/proxy_objective_v1'
    # Only completed ideal diagnostics are reused. No partial radio trajectory is imported.
    reused=[]
    for rel in ['proxies/ideal','ideal_pilot']:
        src=old/rel;dest=core.OUT/rel
        assert (src/('complete.json' if rel=='proxies/ideal' else 'gate.json')).exists()
        if not dest.exists():shutil.copytree(src,dest)
        reused.append(dict(path=rel,source=str(src)))
    core.write(core.OUT/'execution_version.json',dict(version='proxy_objective_v2',source_version='proxy_objective_v1',
      reason='Parallel CPU evaluation of unchanged independent candidate-head solves; same learning objective, budgets, physical constraints, starts and solver.',
      workers=8,reused_completed_ideal_diagnostics=reused,wrapper_sha256=core.sha(__file__),
      parallel_solver_sha256=core.sha(HERE/'parallel_heads.py'),created_unix=time.time()))
    core.load_module=load_module
    core.main()

if __name__=='__main__':
    try:main()
    except Exception as exc:
        core.status(stage='failed',error=repr(exc));raise
    finally:parallel_heads.close()
