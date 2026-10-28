"""Correct a reporting assertion that compared reference-dependent fields.

The v1 audit stopped at its first comparison because retained KL/top1 disagreement
change with the reference. Require identical recall/loss, not identical new KL.
No candidate, reference training or numerical evaluation rule changes.
"""
from pathlib import Path
import pks_budget400_check as old

OUT=old.ROOT/'results/pks_budget400_check_v2_20261028'
SOURCE=Path(__file__)
run=old.study.q.infra.cached_clone(old.run,[
 ("sources=dict(study.register()['source_sha256']);sources[str(Path(__file__).relative_to(ROOT))]=sha(Path(__file__))",
  "sources=dict(study.register()['source_sha256']);sources[str(Path(__file__).relative_to(ROOT))]=sha(Path(__file__));sources[str(CORRECTION_SOURCE.relative_to(ROOT))]=sha(CORRECTION_SOURCE)",1),
 ("scope='Exact unchanged P-KS continuation", "assertion_correction='Require exact mR, loss and six retrieval recalls; reference-dependent KL and top1 disagreements must be allowed to change. Preserve the original v1 stopped audit.',scope='Exact unchanged P-KS continuation",1),
 ("assert e['retained']==points[m]['evaluation']['retained']",
  "assert all(e['retained'][key]==points[m]['evaluation']['retained'][key] for key in ['mR','loss','i2t_R1','i2t_R5','i2t_R10','t2i_R1','t2i_R5','t2i_R10'])",1)],
 dict(OUT=OUT,CORRECTION_SOURCE=SOURCE))


if __name__=='__main__':
 for dataset in old.study.q.infra.DATASETS:run(dataset)
