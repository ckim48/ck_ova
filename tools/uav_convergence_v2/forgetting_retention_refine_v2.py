"""Two intermediate FU coefficients after the documented five-point screen."""
import argparse
import fcntl
from pathlib import Path
import time
import forgetting_retention_refine_v1 as study

ROOT,read,write,sha=study.ROOT,study.read,study.write,study.sha
OUT=ROOT/'results/forgetting_retention_refine_v2_20261030'
VARIANTS={'base100':study.first.spec(late_uniform=100)}
for weight in [.97,.98]:VARIANTS[f'retain{weight:g}']=study.first.spec(late_uniform=100,retention=weight)


def register():
    path=OUT/'protocol.json'
    if path.exists():
        p=read(path);assert p['source_sha256']==sha(Path(__file__));return p
    p=dict(registered_unix=time.time(),source_sha256=sha(Path(__file__)),variants=VARIANTS,
        inherited_source_sha256=sha(Path(study.__file__)),first_source_sha256=sha(Path(study.first.__file__)),
        purpose='Five-point screen decreased KL but lost retained mR. Test two smaller coefficient changes,0.97/0.98, plus unchanged late-uniform control. Finish this finite extension rather than assuming joint leadership.',
        design='Same common setting on both datasets, seed0 development; same200 rounds/45 requests/225 cycles. All earlier failures preserved. Same frozen common and policy-matched references, full gallery and paired no-FU control; no evaluation model/probability enters training.',
        selection='Maintain original P-KS retained mR on both datasets, then compare actual physical mean/worst matched KL against all unchanged comparators. Report common KL and no-FU findings regardless. No outcome is guaranteed.')
    write(path,p);return p


def install():
    study.OUT=OUT;study.VARIANTS=VARIANTS;study.register=register;study.install()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['screen','physical','evaluate']);p.add_argument('--variant');a=p.parse_args()
    install();register()
    if a.action=='physical':
        import joint_quality_physical_v2 as physical
        assert a.variant in VARIANTS;physical.OUT=OUT/'physical';physical.run('retention',a.variant)
    else:
        import torch
        torch.set_num_threads(1);assert torch.cuda.is_available()
        if a.action=='screen':
            for ds in study.first.audit.DATASETS:study.first.screen(ds)
        with (ROOT/'results/overnight_energy_v2/association_objective_gpu.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            for ds in study.first.audit.DATASETS:study.evaluate(ds,'screen' if a.action=='screen' else 'physical')
